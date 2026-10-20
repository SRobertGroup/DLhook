#!/usr/bin/env python
"""Partial segmentation annotation: paint only the parts the model gets wrong.

    python -m ui.seg_annotator --folder data/crops/cropped_new_set
    python -m ui.seg_annotator --folder data/crops/cropped_new_set --queue multi/results/seg_v2_assessment/tricky_crops.csv
    python -m ui.seg_annotator --folder data/crops/cropped_berrybox_val --split val      # held-out test strokes

Each crop is shown with the four-class model's prediction as thin contours (P toggles them). Paint
strokes where it is wrong -- typically hypocotyl through the hook up to the junction, background where
the model invents tissue, cotyledon where it misses some -- and leave everything else unpainted:
unpainted pixels keep the training pseudo-label. Strokes are saved in the RootPainter annotation
format that multi/src/mask_merge.merge_dataset burns over the pseudo-labels (utils/seg_strokes.py),
under <out>/<class>/<train|val>/<crop file>; <out> defaults to data/annotations/seg_strokes/<folder>.

Junction and collar markers (M toggles them) show where the hypocotyl should start and stop: your
earlier clicks when the crop was annotated, else the landmark model's reading. J or C, then a click,
moves that marker; the correction is saved with the strokes in data/annotations/landmark_marks/
<folder>.csv (utils/landmark_marks.py), which landmark retraining can use (`landmarks.marks_dir`).

Keys: 1 cotyledon, 2 hypocotyl, 3 radicle, B background, E erase; [ / ] brush size; Ctrl+Z undo;
Enter or Right: save and next; Left: save and previous; S: discard this crop's unsaved strokes and go
on; P: prediction contours on/off. Mouse: left-drag paints, right-drag pans, wheel zooms, R fits.

Without --queue the crops are sampled like the angle annotator (--per-seedling, --seed, --min-frame,
the folder's manifest). A --queue CSV needs a crop_file column (optionally folder, reason, score);
rows of other folders are ignored. The tool resumes after the last crop you left.
"""
import argparse
import csv
import os
import sys
import tkinter as tk
from tkinter import messagebox
from tkinter.filedialog import askdirectory

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ui.zoomable_canvas import ZoomableImageCanvas  # noqa: E402
from utils.angle_annotation import default_manifest, load_queue  # noqa: E402
from utils.landmark_edit import COLLAR_COLOR, JUNCTION_COLOR  # noqa: E402
from utils.landmark_marks import FolderMarks, landmark_model  # noqa: E402
from utils.seg_strokes import BACKGROUND, CLASS_VALUES, UNLABELLED, load_strokes, save_strokes  # noqa: E402

TOOLS = [  # (key, label, stroke value, colour)
    ("1", "Cotyledon", CLASS_VALUES["cotyledon"], "#00d000"),
    ("2", "Hypocotyl", CLASS_VALUES["hypocotyl"], "#ffb000"),
    ("3", "Radicle", CLASS_VALUES["radicle"], "#ff00ff"),
    ("b", "Background", BACKGROUND, "#3a7bff"),
    ("e", "Erase", UNLABELLED, "#ffffff"),
]
STROKE_ALPHA = 0.45
PREDICTION_COLORS = {1: "#00ff00", 2: "#ffe600", 3: "#ff00ff"}   # thin contours, as in the analysis window
VISITED_FILE = "_visited.txt"
MIN_BRUSH, MAX_BRUSH, DEFAULT_BRUSH = 1, 20, 2
MARK_COLORS = {"junction": JUNCTION_COLOR, "collar": COLLAR_COLOR}
MARK_LABELS = {"junction": "J", "collar": "collar"}


def default_out_dir(folder):
    return os.path.join(_REPO_ROOT, "data", "annotations", "seg_strokes", os.path.basename(os.path.normpath(folder)))


def read_queue_csv(path, folder):
    """[(crop_file, reason)] from a queue CSV, in file order, keeping rows of `folder` (by basename)
    when the CSV has a folder column, and only crops present in the folder."""
    name = os.path.basename(os.path.normpath(folder))
    out, seen = [], set()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            crop = row.get("crop_file") or ""
            if not crop or crop in seen:
                continue
            if row.get("folder") and os.path.basename(os.path.normpath(row["folder"])) != name:
                continue
            if not os.path.exists(os.path.join(folder, crop)):
                continue
            seen.add(crop)
            out.append((crop, row.get("reason", "")))
    return out


def prediction_contours(label_map):
    """Thin contour shapes of the predicted classes (label map 0..3)."""
    from ui.analysis_window import mask_contour_shapes
    shapes = []
    for value, color in PREDICTION_COLORS.items():
        shapes += mask_contour_shapes((label_map == value).astype(np.uint8) * 255, color, alpha=0.8)
    return shapes


def multiclass_predictor(checkpoint=None):
    """crop BGR -> label map with the four-class model at the training geometry (loaded on first use)."""
    state = {}

    def predict(image):
        if "model" not in state:
            from models.multiclass_inference import MulticlassInference
            from models.segmentation_backends import default_multiclass_checkpoint
            from models.UNetInference import MARGIN
            from utils.paths import repo_path
            state["model"] = MulticlassInference(repo_path(checkpoint or default_multiclass_checkpoint()),
                                                 num_classes=4, in_size=252 + 2 * MARGIN, out_size=252,
                                                 margin=MARGIN)
        return state["model"].segment_many_argmax([image])[0]
    return predict


class SegAnnotator(tk.Toplevel):
    CANVAS_W, CANVAS_H = 560, 720

    def __init__(self, master, folder, items, out_dir, split="train", predict=None, marks=None):
        """items: [(crop_file, reason)]. predict: crop BGR -> label map, or None (no contours).
        marks: a utils.landmark_marks.FolderMarks for the junction / collar markers, or None."""
        super().__init__(master)
        self.title("DLhook - segmentation strokes")
        self.folder, self.items, self.out_dir, self.split = folder, items, out_dir, split
        self.predict = predict
        self._pred_cache = {}
        self.index = 0
        self.strokes = None
        self.stroke_split = None
        self.dirty = False
        self._last = None
        self.tool_var = tk.StringVar(value="2")
        self.brush = DEFAULT_BRUSH
        self.show_pred = tk.BooleanVar(value=predict is not None)
        self.marks = marks
        self.show_marks = tk.BooleanVar(value=marks is not None)
        self._armed = None          # "junction" / "collar": the next left click moves that marker
        self._actions = []          # this crop's edits, in order: "stroke" or ("mark", name, previous)
        self.marks_dirty = False
        self._build()
        self.index = self._resume_index()
        if self.items:
            self._load()

    # ---- layout --------------------------------------------------------------------------------
    def _build(self):
        self.canvas = ZoomableImageCanvas(self, width=self.CANVAS_W, height=self.CANVAS_H, pan=True)
        self.canvas.grid(row=0, column=0, rowspan=2, padx=6, pady=6)
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "_last", None))
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())

        side = tk.Frame(self)
        side.grid(row=0, column=1, sticky="n", padx=6, pady=6)
        self.item_label = tk.Label(side, text="", font=("Helvetica", 11, "bold"), justify=tk.LEFT)
        self.item_label.pack(anchor="w")
        self.reason_label = tk.Label(side, text="", fg="#a05000", justify=tk.LEFT, wraplength=260)
        self.reason_label.pack(anchor="w", pady=(0, 8))
        tk.Label(side, text="Paint:").pack(anchor="w")
        for key, label, _, color in TOOLS:
            tk.Radiobutton(side, text=f"{label}  ({key.upper()})", variable=self.tool_var, value=key,
                           fg="black", selectcolor=color if key != "e" else "#dddddd",
                           indicatoron=False, width=18, anchor="w").pack(anchor="w", pady=1)
        self.brush_label = tk.Label(side, text="")
        self.brush_label.pack(anchor="w", pady=(8, 0))
        tk.Checkbutton(side, text="Model prediction (P)", variable=self.show_pred,
                       command=self._render).pack(anchor="w", pady=(8, 0))
        if self.marks is not None:
            tk.Checkbutton(side, text="Junction / collar (M)", variable=self.show_marks,
                           command=self._render).pack(anchor="w")
            marks_row = tk.Frame(side)
            marks_row.pack(anchor="w")
            tk.Button(marks_row, text="Move junction (J)", command=lambda: self.arm("junction")).pack(side=tk.LEFT)
            tk.Button(marks_row, text="Move collar (C)", command=lambda: self.arm("collar")).pack(side=tk.LEFT, padx=4)
        buttons = tk.Frame(side)
        buttons.pack(anchor="w", pady=8)
        tk.Button(buttons, text="< Prev", command=self.previous).pack(side=tk.LEFT)
        tk.Button(buttons, text="Save + next >", command=self.next).pack(side=tk.LEFT, padx=4)
        tk.Button(side, text="Undo (Ctrl+Z)", command=self.undo).pack(anchor="w")
        tk.Button(side, text="Discard + next (S)", command=self.skip).pack(anchor="w", pady=2)
        self.status_label = tk.Label(side, text="", justify=tk.LEFT, wraplength=260)
        self.status_label.pack(anchor="w", pady=(8, 0))
        tk.Label(side, justify=tk.LEFT, fg="#555555", text=(
            "Left-drag: paint   Right-drag: pan\nWheel: zoom   R: fit\n"
            "Unpainted pixels keep the model's\npseudo-label: paint only the errors.")).pack(anchor="w", pady=(8, 0))

        for key, *_ in TOOLS:
            for k in {key, key.upper()}:
                self.bind(f"<KeyPress-{k}>", lambda e, key=key: self.tool_var.set(key))
        self.bind("<bracketleft>", lambda e: self.set_brush(self.brush - 1))
        self.bind("<bracketright>", lambda e: self.set_brush(self.brush + 1))
        self.bind("<Control-z>", lambda e: self.undo())
        self.bind("<Return>", lambda e: self.next())
        self.bind("<Right>", lambda e: self.next())
        self.bind("<Left>", lambda e: self.previous())
        for k in ("s", "S"):
            self.bind(f"<KeyPress-{k}>", lambda e: self.skip())
        for k in ("p", "P"):
            self.bind(f"<KeyPress-{k}>", lambda e: (self.show_pred.set(not self.show_pred.get()), self._render()))
        for k in ("r", "R"):
            self.bind(f"<KeyPress-{k}>", lambda e: self.canvas.reset_zoom())
        if self.marks is not None:
            for k in ("m", "M"):
                self.bind(f"<KeyPress-{k}>", lambda e: (self.show_marks.set(not self.show_marks.get()), self._render()))
            for k, name in (("j", "junction"), ("J", "junction"), ("c", "collar"), ("C", "collar")):
                self.bind(f"<KeyPress-{k}>", lambda e, name=name: self.arm(name))
            self.bind("<Escape>", lambda e: self.arm(None))
        self.set_brush(self.brush)

    # ---- state ---------------------------------------------------------------------------------
    def _visited_path(self):
        return os.path.join(self.out_dir, VISITED_FILE)

    def _visited(self):
        try:
            with open(self._visited_path(), encoding="utf-8") as fh:
                return {line.strip() for line in fh if line.strip()}
        except OSError:
            return set()

    def _mark_visited(self, crop):
        if crop in self._visited():
            return
        os.makedirs(self.out_dir, exist_ok=True)
        with open(self._visited_path(), "a", encoding="utf-8") as fh:
            fh.write(crop + "\n")

    def _resume_index(self):
        visited = self._visited()
        for i, (crop, _) in enumerate(self.items):
            if crop not in visited:
                return i
        return 0

    def _load(self):
        crop, reason = self.items[self.index]
        self.image = cv2.imread(os.path.join(self.folder, crop), cv2.IMREAD_COLOR)
        self.strokes, self.stroke_split = load_strokes(self.out_dir, crop, self.image.shape)
        self.dirty = False
        self._actions, self._armed = [], None
        self.item_label.configure(text=f"{self.index + 1}/{len(self.items)}  {crop}")
        self.reason_label.configure(text=reason or "")
        self.canvas.set_image(self.image, reset_view=True)
        self._render()

    def _render(self):
        if self.strokes is None:
            return
        shapes = []
        if self.show_pred.get() and self.predict is not None:
            crop = self.items[self.index][0]
            if crop not in self._pred_cache:
                self._pred_cache[crop] = prediction_contours(self.predict(self.image))
            shapes += self._pred_cache[crop]
        for key, _, value, color in TOOLS:
            if value == UNLABELLED:
                continue
            mask = self.strokes.labels == value
            if mask.any():
                shapes.append({"kind": "mask", "mask": mask, "color": color, "alpha": STROKE_ALPHA})
        marks = self.current_marks() if self.marks is not None else None
        if marks and self.show_marks.get():
            for name in ("junction", "collar"):
                if marks[name] is not None:
                    shapes.append({"kind": "point", "xy": marks[name], "color": MARK_COLORS[name],
                                   "label": f"{MARK_LABELS[name]} ({marks[name + '_source']})"})
        self.canvas.set_overlay(shapes)
        counts = self.strokes.counts()
        split = self.stroke_split or self.split
        self.status_label.configure(text=(
            ("unsaved changes" if self.dirty else "saved" if counts else "no strokes")
            + f"  [{split}]" + ("\n" + ", ".join(f"{k} {v} px" for k, v in counts.items()) if counts else "")
            + (self._marks_status(marks) if marks else "")))

    # ---- junction / collar markers -------------------------------------------------------------
    def current_marks(self):
        return self.marks.get(self.items[self.index][0]) if self.marks is not None and self.items else None

    def _marks_status(self, marks):
        text = "\n" + " | ".join(f"{n}: {marks[n + '_source'] or 'none'}" for n in ("junction", "collar"))
        if self._armed:
            text += f"\nClick to place the {self._armed} (Esc cancels)"
        return text

    def arm(self, name):
        """The next left click moves `name` ("junction" / "collar"); None cancels."""
        if self.marks is None:
            return
        self._armed = name
        if name:
            self.show_marks.set(True)
        self._render()

    def place_mark(self, name, xy):
        crop = self.items[self.index][0]
        self._actions.append(("mark", name, self.marks.get(crop)))
        self.marks.correct(crop, name, (float(xy[0]), float(xy[1])))
        self.marks_dirty = True
        self._render()

    def _undo_mark(self, action):
        _, name, previous = action
        self.marks.restore(self.items[self.index][0], name, previous)
        self.marks_dirty = True

    def set_brush(self, size):
        self.brush = int(min(MAX_BRUSH, max(MIN_BRUSH, size)))
        self.brush_label.configure(text=f"Brush radius: {self.brush} px  ([ / ])")

    def _value(self):
        return next(value for key, _, value, _ in TOOLS if key == self.tool_var.get())

    # ---- painting ------------------------------------------------------------------------------
    def _image_xy(self, event):
        return self.canvas.canvas_to_image(event.x, event.y)

    def _press(self, event):
        if self.strokes is None:
            return
        self.canvas.focus_set()
        p = self._image_xy(event)
        if self._armed:
            name, self._armed = self._armed, None
            self.place_mark(name, p)
            return
        self.strokes.begin_stroke()
        self._actions.append("stroke")
        self.strokes.paint(p, p, self.brush, self._value())
        self._last = p
        self.dirty = True
        self._render()

    def _drag(self, event):
        if self.strokes is None or self._last is None:
            return
        p = self._image_xy(event)
        self.strokes.paint(self._last, p, self.brush, self._value())
        self._last = p
        self._render()

    def undo(self):
        if self.strokes is None:
            return
        if self._actions and self._actions[-1] != "stroke":
            self._undo_mark(self._actions.pop())
            self._render()
            return
        if self._actions:
            self._actions.pop()
        if self.strokes.undo():
            self.dirty = True
            self._render()

    # ---- navigation ----------------------------------------------------------------------------
    def save(self):
        """Write this crop's strokes (keeping the split they were saved under) and the marker
        corrections. True when strokes were written."""
        if self.marks is not None and self.marks_dirty:
            self.marks.save()
            self.marks_dirty = False
        if self.strokes is None or not self.dirty:
            return False
        crop = self.items[self.index][0]
        save_strokes(self.out_dir, crop, self.strokes, self.stroke_split or self.split)
        self.stroke_split = None if self.strokes.empty else (self.stroke_split or self.split)
        self.dirty = False
        return True

    def _go(self, index):
        if not self.items:
            return
        self._mark_visited(self.items[self.index][0])
        if 0 <= index < len(self.items):
            self.index = index
            self._load()
        else:
            self._render()
            self.status_label.configure(text="End of the queue -- all crops visited.")

    def next(self):
        self.save()
        self._go(self.index + 1)

    def previous(self):
        self.save()
        if self.index > 0:
            self._go(self.index - 1)

    def skip(self):
        self.dirty = False
        for action in reversed(self._actions):     # unsaved marker moves are discarded too
            if action != "stroke":
                self._undo_mark(action)
        if self.marks is not None and self.marks_dirty:
            self.marks.save()
            self.marks_dirty = False
        self._go(self.index + 1)


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", default=None, help="Crop folder ({id}-crop-{frame}.png, optional manifest.csv)")
    p.add_argument("--queue", default=None, help="CSV with crop_file (and folder, reason) -- the crops to show, in order")
    p.add_argument("--out", default=None, help="Stroke folder (default data/annotations/seg_strokes/<folder name>)")
    p.add_argument("--split", choices=("train", "val"), default="train",
                   help="Split for NEW strokes: val for crops of held-out seedlings (a test set). Default train")
    p.add_argument("--per-seedling", type=int, default=5, help="Without --queue: random crops per seedling (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-frame", type=int, default=0, help="Without --queue: skip each seedling's first N frames")
    p.add_argument("--checkpoint", default=None, help="Four-class checkpoint for the contours (default: the GUI's)")
    p.add_argument("--no-prediction", action="store_true", help="Do not run the model (no contours)")
    p.add_argument("--no-markers", action="store_true", help="No junction / collar markers")
    p.add_argument("--no-landmark-model", action="store_true",
                   help="Markers from the annotation CSVs only (no landmark model fallback)")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    root = tk.Tk()
    root.withdraw()
    folder = args.folder or askdirectory(title="Crop folder to annotate")
    if not folder:
        root.destroy()
        return 1
    if args.queue:
        items = read_queue_csv(args.queue, folder)
    else:
        items = [(w.filename, "") for w in load_queue(folder, args.per_seedling or None, args.seed,
                                                     default_manifest(folder), min_frame=args.min_frame)]
    if not items:
        messagebox.showerror("Nothing to annotate", f"No crops found in {folder}", parent=root)
        root.destroy()
        return 1
    out_dir = args.out or default_out_dir(folder)
    predict = None if args.no_prediction else multiclass_predictor(args.checkpoint)
    marks = None if args.no_markers else FolderMarks(folder, model=None if args.no_landmark_model else landmark_model())
    app = SegAnnotator(root, folder, items, out_dir, split=args.split, predict=predict, marks=marks)

    def close():
        app.save()
        root.destroy()
    app.protocol("WM_DELETE_WINDOW", close)
    print(f"{len(items)} crops; strokes -> {out_dir}")
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
