#!/usr/bin/env python
"""Ground-truth angle annotation tool.

Shows seedling crops (the `{id}-crop-{frame}.png` files the GUI writes to
data/images/) one at a time, zoomed, and lets you measure the apical-hook angle
by hand. The result is a CSV of ground truth to validate the automatic
pipeline against (docs/AUDIT.md, finding H-5).

Run:
    python -m ui.angle_annotator --folder data/images
    python -m ui.angle_annotator --folder data/images --per-seedling 15 --seed 1
    python -m ui.angle_annotator --folder cropped_training_set --split val   # multiclass training crops

Per frame, click five points (mouse wheel zooms at the cursor):
    1. the JUNCTION where the cotyledons meet the hypocotyl
    2-3. two points on the hypocotyl axis
    4-5. two points on the cotyledon axis
Each axis is oriented automatically away from the junction, so the order of the
two points on an axis does not matter. Tick "Overhook" if the hook folded back
past closed. See utils/angle_annotation.py for the maths; the angle is stored
in the app's bio convention (180 = closed, decreasing as it opens, > 180 =
overhooked), so it compares directly with the exported `bio_angle`.

Human-error study: --repeat-of angle_landmarks_train.csv,angle_ground_truth.csv --out angle_repeat.csv
re-offers 200 already-measured frames blind (see utils/angle_annotation.repeat_queue); compare the two
measurements with python -m multi.analyze_repeatability.

Frames are a reproducible random sample (--per-seedling, --seed), grouped by
seedling and shuffled within it so the annotation is blind to time order. The
CSV (default: <repo>/angle_ground_truth.csv) is saved after every frame and
the tool resumes where you stopped.

The multiclass training crops (cropped_training_set/, made by multi/recrop_plates.py)
work too: its manifest.csv is picked up automatically, which makes seedlings
unique per (series, id) -- crop_id alone repeats across series -- and records
the source frame as `img_name`. --split val keeps only crops from images held
out of training, the unbiased set for validating the four-class model.

Note: DLhook wipes data/ at startup and exit, so copy the crops out of
data/images/ to a folder of your own if you want to annotate across sessions.
The CSV is deliberately NOT written next to the crops for that reason.
"""
import argparse
import os
import sys
import tkinter as tk
from tkinter import messagebox
from tkinter.filedialog import askdirectory

import cv2

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ui.zoomable_canvas import ZoomableImageCanvas  # noqa: E402
from utils.angle_annotation import (  # noqa: E402
    AngleClickSession, AnnotationStore, default_manifest, load_queue, orient_axis, repeat_queue,
)

HYPO_COLOR = "#ff9f1c"
COTYL_COLOR = "#2ec4b6"
JUNCTION_COLOR = "#ff3b30"
HELP_TEXT = (
    "Mouse wheel: zoom at cursor\n"
    "Enter / Space: save and next\n"
    "Backspace / Z: undo last point\n"
    "R: reset points      O: overhook\n"
    "S: skip (cannot measure)\n"
    "Left: previous\n"
    "Right: next -- a frame you did not\n"
    "measure is recorded as skipped"
)


class AngleAnnotatorApp(tk.Tk):
    CANVAS_W, CANVAS_H = 560, 720

    def __init__(self, folder, out_path, per_seedling=10, seed=0, manifest_path=None, split=None,
                 patch_index_dir=None, queue=None):
        super().__init__()
        self.title("DLhook - angle ground truth")
        self.folder = folder
        self.queue = queue if queue is not None else load_queue(
            folder, per_seedling, seed, manifest_path, split, patch_index_dir)
        if not self.queue:
            messagebox.showerror("No crops found", chr(10).join((
                "No usable {id}-crop-{frame}.png files in:", folder, "",
                "Run Start Analysis in DLhook first (it writes them to data/images/),",
                "or point --folder at cropped_training_set.")))
            self.destroy()
            raise SystemExit(1)
        self.store = AnnotationStore(out_path)
        self.index = 0
        self._shown_seedling = None
        self.session = AngleClickSession()
        self.overhook = tk.BooleanVar(value=False)
        self.overhook.trace_add("write", lambda *_: self._refresh())

        self._build_widgets()
        self._bind_keys()
        # Resume at the first frame with no record yet
        self.index = next((i for i, item in enumerate(self.queue) if self.store.get(item) is None), 0)
        self._load_current()

    # --- widgets ----------------------------------------------------------

    def _build_widgets(self):
        self.canvas = ZoomableImageCanvas(self, width=self.CANVAS_W, height=self.CANVAS_H)
        self.canvas.grid(row=0, column=0, rowspan=2, padx=8, pady=8)
        self.canvas.bind("<Button-1>", self._on_click)

        panel = tk.Frame(self)
        panel.grid(row=0, column=1, sticky="n", padx=(0, 12), pady=8)
        self.info_label = tk.Label(panel, text="", justify=tk.LEFT, font=("Helvetica", 12, "bold"))
        self.info_label.pack(anchor="w")
        self.file_label = tk.Label(panel, text="", justify=tk.LEFT, fg="#666")
        self.file_label.pack(anchor="w", pady=(0, 10))
        self.progress_label = tk.Label(panel, text="", justify=tk.LEFT)
        self.progress_label.pack(anchor="w", pady=(0, 10))

        self.prompt_label = tk.Label(panel, text="", justify=tk.LEFT, wraplength=260,
                                     font=("Helvetica", 11), fg="#b4531f")
        self.prompt_label.pack(anchor="w", pady=(0, 10))

        legend = tk.Frame(panel)
        legend.pack(anchor="w", pady=(0, 10))
        for color, text in ((JUNCTION_COLOR, "junction"), (HYPO_COLOR, "hypocotyl axis"),
                            (COTYL_COLOR, "cotyledon axis")):
            tk.Label(legend, text="●", fg=color).pack(side=tk.LEFT)
            tk.Label(legend, text=text).pack(side=tk.LEFT, padx=(0, 10))

        tk.Checkbutton(panel, text="Overhook (folded back past closed)  [O]",
                       variable=self.overhook).pack(anchor="w")
        self.result_label = tk.Label(panel, text="", justify=tk.LEFT, font=("Helvetica", 12))
        self.result_label.pack(anchor="w", pady=10)

        buttons = tk.Frame(panel)
        buttons.pack(anchor="w", pady=4)
        self.save_button = tk.Button(buttons, text="Save & next", width=12, command=self._confirm)
        self.save_button.grid(row=0, column=0, padx=(0, 6), pady=2)
        tk.Button(buttons, text="Undo point", width=12, command=self._undo).grid(row=0, column=1, pady=2)
        tk.Button(buttons, text="Reset", width=12, command=self._reset).grid(row=1, column=0, padx=(0, 6), pady=2)
        tk.Button(buttons, text="Skip frame", width=12, command=self._skip).grid(row=1, column=1, pady=2)
        tk.Button(buttons, text="◀ Previous", width=12, command=lambda: self._move(-1)).grid(row=2, column=0, padx=(0, 6), pady=2)
        tk.Button(buttons, text="Next ▶", width=12, command=lambda: self._move(1)).grid(row=2, column=1, pady=2)

        tk.Label(panel, text=HELP_TEXT, justify=tk.LEFT, fg="#666").pack(anchor="w", pady=(14, 0))

    def _bind_keys(self):
        for key in ("<Return>", "<space>"):
            self.bind(key, lambda e: self._confirm())
        for key in ("<BackSpace>", "z", "Z"):
            self.bind(key, lambda e: self._undo())
        for key in ("r", "R"):
            self.bind(key, lambda e: self._reset())
        for key in ("o", "O"):
            self.bind(key, lambda e: self.overhook.set(not self.overhook.get()))
        for key in ("s", "S"):
            self.bind(key, lambda e: self._skip())
        self.bind("<Left>", lambda e: self._move(-1))
        self.bind("<Right>", lambda e: self._move(1))

    # --- navigation -------------------------------------------------------

    @property
    def item(self):
        return self.queue[self.index]

    def _load_current(self):
        item = self.item
        img = cv2.imread(os.path.join(self.folder, item.filename))
        if img is None:
            messagebox.showerror("Cannot open image", item.filename)
            return
        # A new seedling has a different crop size: refit. Frames of the same
        # seedling keep the zoom/position you set, so you stay on the cotyledon.
        self.canvas.set_image(img, reset_view=item.seedling_key != self._shown_seedling)
        self._shown_seedling = item.seedling_key

        saved_points = self.store.points_for(item)
        self.session = AngleClickSession(saved_points)
        row = self.store.get(item)
        self.overhook.set(bool(row and row["overhook"] == "1"))
        self._refresh()

    def _move(self, step):
        if step > 0:
            # Moving on from a frame you did not measure counts as skipping it:
            # otherwise "looked at and rejected" cannot be told apart from
            # "never reached" in the CSV.
            if self.store.get(self.item) is None:
                if self.session.complete:
                    self.bell()
                    self.prompt_label.configure(
                        text="Five points placed but not saved -- Enter to save, or R to reset, before moving on")
                    return
                self.store.save_skipped(self.item)
            self._advance()
            return
        new_index = self.index + step
        if 0 <= new_index < len(self.queue):
            self.index = new_index
            self._load_current()

    def _advance(self):
        if self.index + 1 < len(self.queue):
            self.index += 1
            self._load_current()
        else:
            self._refresh()
            messagebox.showinfo("Done", f"All {len(self.queue)} frames reviewed.\n\nSaved to:\n{self.store.path}")

    # --- annotation actions ----------------------------------------------

    def _on_click(self, event):
        h, w = self.canvas._image.shape[:2]
        x, y = self.canvas.canvas_to_image(event.x, event.y)
        if not (0 <= x < w and 0 <= y < h):
            return
        self.session.add(x, y)
        self._refresh()

    def _undo(self):
        self.session.undo()
        self._refresh()

    def _reset(self):
        self.session.reset()
        self._refresh()

    def _confirm(self):
        if not self.session.complete:
            self.bell()
            return
        try:
            self.store.save_measured(self.item, self.session, self.overhook.get())
        except ValueError as exc:
            messagebox.showwarning("Cannot compute angle", str(exc))
            return
        self._advance()

    def _skip(self):
        self.store.save_skipped(self.item)
        self._advance()

    # --- display ----------------------------------------------------------

    def _overlay_shapes(self):
        pts = self.session.named_points()
        shapes = []
        colors = {"junction": JUNCTION_COLOR, "hypo_1": HYPO_COLOR, "hypo_2": HYPO_COLOR,
                  "cotyl_1": COTYL_COLOR, "cotyl_2": COTYL_COLOR}
        labels = {"junction": "J", "hypo_1": "H1", "hypo_2": "H2", "cotyl_1": "C1", "cotyl_2": "C2"}
        for name, xy in pts.items():
            shapes.append({"kind": "point", "xy": xy, "color": colors[name], "label": labels[name]})

        if "junction" in pts:
            j = pts["junction"]
            for a, b, color in (("hypo_1", "hypo_2", HYPO_COLOR), ("cotyl_1", "cotyl_2", COTYL_COLOR)):
                if a in pts and b in pts:
                    shapes.append({"kind": "line", "from": pts[a], "to": pts[b], "color": color, "dash": True})
                    try:  # arrow from the junction along the auto-oriented axis
                        ux, uy = orient_axis(pts[a], pts[b], j)
                    except ValueError:
                        continue
                    length = max(40.0, 1.2 * max(abs(pts[a][0] - j[0]) + abs(pts[a][1] - j[1]),
                                                  abs(pts[b][0] - j[0]) + abs(pts[b][1] - j[1])))
                    shapes.append({"kind": "line", "from": j, "to": (j[0] + ux * length, j[1] + uy * length),
                                   "color": color, "arrow": True})
        return shapes

    def _refresh(self):
        item = self.item
        done, skipped = self.store.counts(self.queue)
        row = self.store.get(item)
        status = ""
        if row:
            status = "  [already saved]" if row["status"] == "measured" else "  [skipped]"

        self.info_label.configure(text=f"{item.seedling_label}  —  frame {self.index + 1} of {len(self.queue)}{status}")
        self.file_label.configure(text=item.filename)
        self.progress_label.configure(text=f"Measured: {done}    Skipped: {skipped}    Remaining: {len(self.queue) - done - skipped}")
        self.prompt_label.configure(text=self.session.prompt())
        self.canvas.set_overlay(self._overlay_shapes())

        if self.session.complete:
            try:
                theta, bio = self.session.result(self.overhook.get())
                self.result_label.configure(text=f"angle between axes: {theta:.1f}°\nbio angle: {bio:.1f}°", fg="black")
            except ValueError as exc:
                self.result_label.configure(text=str(exc), fg="#b42318")
        else:
            self.result_label.configure(text="", fg="black")


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", default=None,
                        help="Folder of {id}-crop-{frame}.png crops (default: ask; e.g. data/images)")
    parser.add_argument("--manifest", default=None,
                        help="recrop_plates manifest.csv (default: <folder>/manifest.csv if present). Needed for "
                             "multi-series crop sets, where crop_id repeats across series")
    parser.add_argument("--split", choices=("train", "val"), default=None,
                        help="Only crops in this multiclass-training split. Use 'val' to validate on images the "
                             "four-class model never trained on")
    parser.add_argument("--patch-index-dir", default=os.path.join(_REPO_ROOT, "multi", "processed", "patch_index"),
                        help="Where train_patches.csv / val_patches.csv are (for --split)")
    parser.add_argument("--out", default=None, help="Ground-truth CSV (default: <repo>/angle_ground_truth.csv)")
    parser.add_argument("--per-seedling", type=int, default=10,
                        help="Random frames to annotate per seedling; 0 = every frame (default: 10)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the frame sample (default: 0)")
    parser.add_argument("--repeat-of", default=None, metavar="CSV[,CSV]",
                        help="Blind re-measurement for a human-error study: instead of the usual sample, offer "
                             "--repeat-n frames already measured in these CSVs, at random, in shuffled order, with "
                             "none of the earlier clicks shown. Write to a NEW --out so the originals stay untouched")
    parser.add_argument("--repeat-n", type=int, default=200, help="Frames for --repeat-of (default 200)")
    parser.add_argument("--max-per-seedling", type=int, default=4,
                        help="At most this many --repeat-of frames per seedling (default 4; 0 = no cap)")
    parser.add_argument("--mark-unrecorded-skipped", action="store_true",
                        help="Do not open the window: record every frame of this sample that has no row yet as "
                             "skipped, then exit. For CSVs made before Next recorded skips")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    folder = args.folder
    if not folder:
        root = tk.Tk()
        root.withdraw()
        folder = askdirectory(title="Folder with the seedling crops")
        root.destroy()
        if not folder:
            return 1
    out_path = args.out or os.path.join(_REPO_ROOT, "angle_ground_truth.csv")
    manifest = args.manifest or default_manifest(folder)
    if args.mark_unrecorded_skipped:
        queue = load_queue(folder, args.per_seedling or None, args.seed, manifest, args.split, args.patch_index_dir)
        store = AnnotationStore(out_path)
        added = store.skip_unrecorded(queue)
        done, skipped = store.counts(queue)
        print(f"Marked {added} unrecorded frame(s) as skipped. {out_path}: "
              f"{done} measured, {skipped} skipped, {len(queue)} in the sample.")
        return 0
    queue = None
    if args.repeat_of:
        sources = [p.strip() for p in args.repeat_of.split(",") if p.strip()]
        if any(os.path.abspath(p) == os.path.abspath(out_path) for p in sources):
            print("--out must be a new file, not one of the --repeat-of sources")
            return 2
        queue = repeat_queue(sources, args.repeat_n, args.seed, args.max_per_seedling or None, folder)
        print(f"Blind repeat: {len(queue)} frames of {len({i.seedling_key for i in queue})} seedlings -> {out_path}")
    app = AngleAnnotatorApp(folder, out_path, per_seedling=args.per_seedling or None, seed=args.seed,
                            manifest_path=manifest, split=args.split, patch_index_dir=args.patch_index_dir,
                            queue=queue)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
