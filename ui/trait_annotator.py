#!/usr/bin/env python
"""Measure the hypocotyl traits by hand: length, curvature and diameter.

    python -m ui.trait_annotator                                   # BerryBox v2 val (held out), measurable frames
    python -m ui.trait_annotator --folder data/crops/cropped_new_set --per-seedling 3
    python -m multi.validate_traits --truth data/annotations/traits/cropped_berrybox_val.csv   # score the pipeline

Per frame:
  1. MIDLINE: click along the middle of the hypocotyl from the JUNCTION (where the cotyledons meet
     it) to the COLLAR (where the root starts), as many points as the curve needs -- more in the hook.
  2. Tab, then DIAMETER: three edge-to-edge transects across the stem (two clicks each), spread along
     it, perpendicular to the stem.
The junction and collar you clicked earlier (angle / root annotation) are shown as hollow guides.
Length and curvature come from a smooth curve through your midline clicks, computed exactly like the
pipeline's midline (utils/trait_annotation.py); the diameter is the median transect.

The model's midline is hidden while you measure; V shows it once the frame is saved, to compare.

Keys: Enter save + next; Tab midline / diameter; Backspace undo; R reset; S skip (cannot measure);
Left / Right previous / next; V model midline (saved frames). Mouse: wheel zoom, right-drag pan.

Results go to data/annotations/traits/<folder>.csv (saved after every frame; the tool resumes).
"""
import argparse
import os
import sys
import tkinter as tk
from tkinter import messagebox

import cv2

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ui.zoomable_canvas import ZoomableImageCanvas  # noqa: E402
from utils.angle_annotation import default_manifest, load_queue  # noqa: E402
from utils.landmark_edit import COLLAR_COLOR, HYPO_COLOR, JUNCTION_COLOR  # noqa: E402
from utils.landmark_marks import FolderMarks  # noqa: E402
from utils.trait_annotation import DIAMETER, N_TRANSECTS, TraitStore, hand_midline  # noqa: E402

TRANSECT_COLOR = "#2ec4b6"
MODEL_COLOR = "#ffe600"
DEFAULT_FOLDER = os.path.join("data", "crops", "cropped_berrybox_val")
HELP_TEXT = (
    "Midline: junction -> collar, along the stem\n"
    "Tab: diameter (3 edge-to-edge transects)\n"
    "Enter: save and next    S: skip\n"
    "Backspace: undo    R: reset\n"
    "Left / Right: previous / next\n"
    "V: model midline (saved frames)\n"
    "Wheel: zoom    Right-drag: pan"
)


def default_out(folder):
    return os.path.join(_REPO_ROOT, "data", "annotations", "traits", os.path.basename(os.path.normpath(folder)) + ".csv")


def model_traits_fn(checkpoint=None):
    """(image, junction, collar) -> (model midline or None, traits dict) on the GUI's four-class
    hypocotyl mask (loaded on first use)."""
    from ui.seg_annotator import multiclass_predictor
    from utils.hypocotyl_traits import hypocotyl_midline, hypocotyl_traits
    predict = multiclass_predictor(checkpoint)

    def run(image, junction, collar):
        mask = (predict(image) == 2).astype("uint8") * 255
        return hypocotyl_midline(mask, junction, collar), hypocotyl_traits(mask, junction, collar)
    return run


def _fmt(t, hand=True):
    def f(v, spec):
        return "-" if v != v else format(v, spec)
    n = f"{t['n_transects']}/{N_TRANSECTS}" if hand else str(t["n_transects"])
    return (f"length {f(t['length_px'], '.1f')} px\n"
            f"curvature mean {f(t['curvature_mean'], '.4f')} / max {f(t['curvature_max'], '.4f')} 1/px\n"
            f"diameter {f(t['diameter_px'], '.1f')} px ({n} transects)")


class TraitAnnotator(tk.Toplevel):
    CANVAS_W, CANVAS_H = 560, 720

    def __init__(self, master, folder, queue, store, marks=None, model_traits=None):
        """queue: [WorkItem]; store: TraitStore; marks: FolderMarks for the guides (or None);
        model_traits: (image, junction, collar) -> (midline, traits) for V (or None)."""
        super().__init__(master)
        self.title("DLhook - hypocotyl traits")
        self.folder, self.queue, self.store, self.marks = folder, queue, store, marks
        self.model_traits = model_traits
        self.index = 0
        self.session = None
        self.show_model = False
        self._model_cache = {}
        self._shown_seedling = None
        self._build()
        self.index = next((i for i, item in enumerate(queue) if store.get(item) is None), 0)
        if queue:
            self._load()

    # ---- layout --------------------------------------------------------------------------------
    def _build(self):
        self.canvas = ZoomableImageCanvas(self, width=self.CANVAS_W, height=self.CANVAS_H, pan=True)
        self.canvas.grid(row=0, column=0, rowspan=2, padx=8, pady=8)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())
        panel = tk.Frame(self)
        panel.grid(row=0, column=1, sticky="n", padx=(0, 12), pady=8)
        self.info_label = tk.Label(panel, text="", justify=tk.LEFT, font=("Helvetica", 12, "bold"))
        self.info_label.pack(anchor="w")
        self.file_label = tk.Label(panel, text="", justify=tk.LEFT, fg="#666")
        self.file_label.pack(anchor="w", pady=(0, 8))
        self.progress_label = tk.Label(panel, text="", justify=tk.LEFT)
        self.progress_label.pack(anchor="w", pady=(0, 8))
        self.prompt_label = tk.Label(panel, text="", justify=tk.LEFT, wraplength=280,
                                     font=("Helvetica", 11), fg="#b4531f")
        self.prompt_label.pack(anchor="w", pady=(0, 8))
        legend = tk.Frame(panel)
        legend.pack(anchor="w", pady=(0, 8))
        for color, text in ((HYPO_COLOR, "midline"), (TRANSECT_COLOR, "transects"), (MODEL_COLOR, "model (V)")):
            tk.Label(legend, text="●", fg=color).pack(side=tk.LEFT)
            tk.Label(legend, text=text).pack(side=tk.LEFT, padx=(0, 8))
        self.result_label = tk.Label(panel, text="", justify=tk.LEFT, font=("Helvetica", 11))
        self.result_label.pack(anchor="w", pady=6)
        self.model_label = tk.Label(panel, text="", justify=tk.LEFT, fg="#7a6a00")
        self.model_label.pack(anchor="w")
        buttons = tk.Frame(panel)
        buttons.pack(anchor="w", pady=6)
        tk.Button(buttons, text="Save & next", width=12, command=self.confirm).grid(row=0, column=0, padx=(0, 6), pady=2)
        tk.Button(buttons, text="Midline / diameter", width=14, command=self.toggle_mode).grid(row=0, column=1, pady=2)
        tk.Button(buttons, text="Undo", width=12, command=self.undo).grid(row=1, column=0, padx=(0, 6), pady=2)
        tk.Button(buttons, text="Reset", width=14, command=self.reset).grid(row=1, column=1, pady=2)
        tk.Button(buttons, text="Skip frame", width=12, command=self.skip).grid(row=2, column=0, padx=(0, 6), pady=2)
        tk.Button(buttons, text="Model midline (V)", width=14, command=self.toggle_model).grid(row=2, column=1, pady=2)
        tk.Button(buttons, text="◀ Previous", width=12, command=lambda: self.move(-1)).grid(row=3, column=0, padx=(0, 6), pady=2)
        tk.Button(buttons, text="Next ▶", width=14, command=lambda: self.move(1)).grid(row=3, column=1, pady=2)
        tk.Label(panel, text=HELP_TEXT, justify=tk.LEFT, fg="#666").pack(anchor="w", pady=(12, 0))

        self.bind("<Return>", lambda e: self.confirm())
        self.bind("<Tab>", lambda e: (self.toggle_mode(), "break")[1])
        for k in ("<BackSpace>", "z", "Z"):
            self.bind(k, lambda e: self.undo())
        for k in ("r", "R"):
            self.bind(k, lambda e: self.reset())
        for k in ("s", "S"):
            self.bind(k, lambda e: self.skip())
        for k in ("v", "V"):
            self.bind(k, lambda e: self.toggle_model())
        self.bind("<Left>", lambda e: self.move(-1))
        self.bind("<Right>", lambda e: self.move(1))

    # ---- state ---------------------------------------------------------------------------------
    @property
    def item(self):
        return self.queue[self.index]

    def _load(self):
        item = self.item
        self.image = cv2.imread(os.path.join(self.folder, item.filename), cv2.IMREAD_COLOR)
        self.canvas.set_image(self.image, reset_view=item.seedling_key != self._shown_seedling)
        self._shown_seedling = item.seedling_key
        self.session = self.store.session_for(item)
        self.show_model = False
        self._refresh()

    @property
    def saved(self):
        row = self.store.get(self.item)
        return bool(row) and row["status"] == "measured"

    # ---- actions -------------------------------------------------------------------------------
    def _on_click(self, event):
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        x, y = self.canvas.canvas_to_image(event.x, event.y)
        if not (0 <= x < w and 0 <= y < h):
            return
        self.session.add(x, y)
        self._refresh()

    def toggle_mode(self):
        self.session.toggle_mode()
        self._refresh()

    def undo(self):
        self.session.undo()
        self._refresh()

    def reset(self):
        self.session.reset()
        self._refresh()

    def confirm(self):
        if not self.session.complete:
            self.bell()
            return False
        self.store.save_measured(self.item, self.session)
        self._advance()
        return True

    def skip(self):
        self.store.save_skipped(self.item)
        self._advance()

    def move(self, step):
        if step > 0:
            if self.store.get(self.item) is None:
                if not self.session.empty:
                    self.bell()
                    self.prompt_label.configure(text="Clicks not saved -- Enter to save, or R to reset, before moving on")
                    return
                self.store.save_skipped(self.item)
            self._advance()
        elif self.index > 0:
            self.index -= 1
            self._load()

    def _advance(self):
        if self.index + 1 < len(self.queue):
            self.index += 1
            self._load()
        else:
            self._refresh()
            self.prompt_label.configure(text=f"All {len(self.queue)} frames reviewed. Saved to {self.store.path}")

    def toggle_model(self):
        """Show / hide the model midline -- only on a saved frame, so measuring stays blind."""
        if self.model_traits is None or not self.saved:
            self.bell()
            self.model_label.configure(text="Save the frame first (the model stays hidden while you measure)."
                                       if self.model_traits is not None else "No model loaded.")
            return
        self.show_model = not self.show_model
        self._refresh()

    def _model(self):
        key = self.item.filename
        mid = self.session.midline
        anchor = (tuple(mid[0]), tuple(mid[-1]))
        if self._model_cache.get(key, (None,))[0] != anchor:
            self._model_cache[key] = (anchor, self.model_traits(self.image, anchor[0], anchor[1]))
        return self._model_cache[key][1]

    # ---- display -------------------------------------------------------------------------------
    def _shapes(self):
        shapes = []
        guides = self.marks.get(self.item.filename, use_model=False) if self.marks is not None else None
        for name, color in (("junction", JUNCTION_COLOR), ("collar", COLLAR_COLOR)):
            if guides and guides.get(name) is not None:
                shapes.append({"kind": "point", "xy": guides[name], "color": color, "hollow": True})
        if self.show_model and self.session.complete:
            mid, _ = self._model()
            if mid is not None:
                shapes.append({"kind": "polyline", "pts": [tuple(p) for p in mid], "color": MODEL_COLOR})
        mid = hand_midline(self.session.midline)
        if mid is not None:
            shapes.append({"kind": "polyline", "pts": [tuple(p) for p in mid], "color": HYPO_COLOR})
        for i, p in enumerate(self.session.midline):
            last = i == len(self.session.midline) - 1 and i > 0
            shapes.append({"kind": "point", "xy": p, "color": JUNCTION_COLOR if i == 0 else COLLAR_COLOR if last else HYPO_COLOR,
                           "label": "J" if i == 0 else ""})
        for a, b in self.session.transects:
            shapes.append({"kind": "line", "from": a, "to": b, "color": TRANSECT_COLOR})
        if self.session.pending is not None:
            shapes.append({"kind": "point", "xy": self.session.pending, "color": TRANSECT_COLOR})
        return shapes

    def _refresh(self):
        if not self.queue or self.session is None:
            return
        item = self.item
        done, skipped = self.store.counts(self.queue)
        row = self.store.get(item)
        status = ("  [saved]" if row["status"] == "measured" else "  [skipped]") if row else ""
        self.info_label.configure(text=f"{item.seedling_label}  —  frame {self.index + 1} of {len(self.queue)}{status}")
        self.file_label.configure(text=item.filename)
        self.progress_label.configure(text=f"Measured: {done}    Skipped: {skipped}    "
                                           f"Remaining: {len(self.queue) - done - skipped}")
        mode = "DIAMETER" if self.session.mode == DIAMETER else "MIDLINE"
        self.prompt_label.configure(text=f"[{mode}] {self.session.prompt()}")
        self.result_label.configure(text=_fmt(self.session.traits()) if not self.session.empty else "")
        if self.show_model and self.session.complete:
            self.model_label.configure(text="model: " + _fmt(self._model()[1], hand=False).replace("\n", "\nmodel: "))
        else:
            self.model_label.configure(text="")
        self.canvas.set_overlay(self._shapes())


def measurable_queue(queue, clicked, include_all=False):
    """The frames with a clicked junction and collar (measurable, visible collar) unless include_all."""
    if include_all:
        return list(queue)
    return [w for w in queue if (clicked.get(w.filename) or {}).get("collar") is not None]


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", default=os.path.join(_REPO_ROOT, DEFAULT_FOLDER),
                   help="Crop folder (default: BerryBox v2 val, held out of every training)")
    p.add_argument("--out", default=None, help="Traits CSV (default data/annotations/traits/<folder name>.csv)")
    p.add_argument("--per-seedling", type=int, default=0, help="Random frames per seedling (0 = all; default 0)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-frame", type=int, default=0, help="Leave out each seedling's first N frames")
    p.add_argument("--all", action="store_true",
                   help="Every frame, not only those with a clicked junction and collar")
    p.add_argument("--checkpoint", default=None, help="Four-class checkpoint for V (default: the GUI's)")
    p.add_argument("--no-model", action="store_true", help="Do not load the model (V disabled)")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    folder = args.folder
    marks = FolderMarks(folder, model=None)
    queue = load_queue(folder, args.per_seedling or None, args.seed, default_manifest(folder), min_frame=args.min_frame)
    queue = measurable_queue(queue, marks.clicked, args.all)
    root = tk.Tk()
    root.withdraw()
    if not queue:
        messagebox.showerror("Nothing to measure", f"No frame with a clicked junction and collar in {folder} "
                             "(use --all for every frame).", parent=root)
        root.destroy()
        return 1
    out = args.out or default_out(folder)
    app = TraitAnnotator(root, folder, queue, TraitStore(out), marks=marks,
                         model_traits=None if args.no_model else model_traits_fn(args.checkpoint))
    app.protocol("WM_DELETE_WINDOW", root.destroy)
    print(f"{len(queue)} frames; traits -> {out}")
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
