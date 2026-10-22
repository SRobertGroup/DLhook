#!/usr/bin/env python
"""Manual crop-region drawing tool.

Replaces the automatic seedling/row detector that used to feed
multi/recrop_plates.py --boxes-file (removed for being unreliable -- see
multi/recrop_plates.py's module docstring). There are only 16 series in
example_data/, so drawing crop boxes by hand once per series is cheap and
exact, and this is now the *only* way to produce a boxes-file.

For each series (an immediate subdirectory of --example-data that contains
image files -- see discover_series_last_frames below) this loads the LAST
frame -- late frames show the seedling largest/darkest against the plate,
easiest to box accurately -- and lets the user draw one crop box per
seedling. The same fixed set of boxes is then reused for every frame of that
series by recrop_plates.py, exactly like the GUI's own crop-box workflow.

Run:
    python -m multi.draw_crops --example-data example_data --out data/crop_boxes/crop_boxes.json

Drawing a validation set next to existing ones (read-only reference boxes):
    python -m multi.draw_crops --out data/crop_boxes/crop_boxes_validation.json --existing training=data/crops/cropped_training_set
        --existing new=data/crops/cropped_new_set --existing open=data/crops/cropped_open_set
`--existing [LABEL=]PATH` is repeatable; PATH is a crop folder (its manifest.csv, which records what
was actually cut), a manifest.csv, or a boxes .json. Each reference set is drawn per series in its own
colour as thin dashed boxes tagged LABEL#seedling-number, on top of the plate; they cannot be
selected, moved or drawn through by accident. A box of yours that touches a reference box turns
magenta and the status line names the worst overlap, so a validation set can be kept free of plants
that were trained on. Press E to hide / show the references.

Output (data/crop_boxes/crop_boxes.json), consumed unmodified by
multi.recrop_plates.load_boxes_file:
    {"F1_Plate_2_YS": [{"cx": 428, "cy": 2783, "half_w": 120, "half_h": 90}, ...]}

Design decision -- drag defines the SEEDLING EXTENT, not the final box:
    Dragging on empty canvas draws the seedling's bounding extent (its actual
    visible top/bottom/left/right), exactly like clicking two points in the
    legacy GUI. multi.src.recrop_geometry.compute_crop_box then pads that
    extent (1.8x width, 1.2x height, floored at a 30 px half-size) into the
    box that gets stored and that recrop_plates.py crops from -- this keeps
    hand-drawn boxes on the same geometry as the rest of the pipeline. The
    PADDED result is what's drawn on screen (live, while dragging) so what
    you see while drawing is exactly what will be cropped.

    Once a box exists, moving it (drag its body) or resizing it (drag a
    corner handle) edits that already-padded box DIRECTLY -- it does not
    re-run compute_crop_box, which would pad an already-padded box a second
    time. Resize is a plain rectangle edit (floored at the same 30 px
    minimum half-size), not a re-derivation from a fresh "extent".

Frame/series discovery (see discover_series_last_frames): series come from
multi.recrop_plates.discover_series (directories only -- the stray *.csv/
*.txt files sitting directly in example_data/ are skipped); frames within a
series come from multi.src.recrop_geometry.list_image_files, which is
non-recursive (example_data/F1_Plate_2_YS/segmented_s0..s3/ hold 800 mask
PNGs that must NOT be treated as plate frames). "Last frame" is chosen by
acquisition order: utils.preprocess_model_input.order_series (embedded EXIF/
TIFF capture time) when every frame has a timestamp and they aren't all
identical, else a natural/numeric sort of the filenames (natural_sort_key)
-- NOT a plain lexicographic sort, which is wrong for example_data/MB's
non-zero-padded "MB_1_0.jpg" .. "MB_1_64.jpg" (lexicographically "last" is
"MB_1_9.jpg"). This also intentionally does not fall through to
order_series's own internal legacy fallback (first-run-of-digits +
lexicographic tiebreak), which can degenerate the same way.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import threading
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2  # noqa: E402
import tkinter as tk  # noqa: E402

from multi.recrop_plates import discover_series  # noqa: E402
from multi.src.recrop_geometry import (  # noqa: E402,F401
    DEFAULT_MIN_BOX_HALF_SIZE,
    compute_crop_box,
    list_image_files,
)
# The canvas (and its pure helpers) moved to ui/ so the main window can use it without importing
# multi/; the names stay importable from here.
from ui.crop_box_canvas import (  # noqa: E402,F401
    REFERENCE_COLORS,
    CropBoxCanvas,
    box_overlap,
    resize_rect_from_corners,
    round_box,
    worst_reference_overlap,
)
from utils.gui_thread_safety import ProgressReporter  # noqa: E402
from utils.preprocess_model_input import order_series  # noqa: E402


# ---------------------------------------------------------------------------
# Series / frame discovery -- pure, headless, unit-tested in
# tests/test_draw_crops.py.
# ---------------------------------------------------------------------------

def natural_sort_key(name: str):
    """Human/natural sort key: alternating text/number chunks, numbers
    compared as integers. Unlike order_series's own legacy fallback (first
    run of digits only, ties broken lexicographically), this compares EVERY
    number in the name, so "MB_1_9.jpg" sorts before "MB_1_64.jpg" instead of
    after it."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]


def _capture_times_are_usable(capture_times: dict) -> bool:
    """True if order_series's second return value can be trusted to reflect
    real acquisition order: every frame has an embedded timestamp, and they
    are not all identical (mirrors order_series's own docstring -- a folder
    stamped with one shared time is exactly as untrustworthy as one with no
    times at all)."""
    times = list(capture_times.values())
    return bool(times) and all(t is not None for t in times) and len(set(times)) > 1


def pick_last_frame(dir_path: str, filenames: list) -> str:
    """The filename that comes acquisition-last: EXIF/TIFF capture-time order
    (order_series) when usable, else our own natural/numeric sort -- never
    order_series's own legacy (first-digit-run) fallback, and never a plain
    lexicographic sort."""
    if not filenames:
        raise ValueError(f"no image files found in {dir_path!r}")
    ordered, capture_times = order_series(dir_path, filenames)
    if _capture_times_are_usable(capture_times):
        return ordered[-1]
    return sorted(filenames, key=natural_sort_key)[-1]


def discover_series_last_frames(example_data_dir) -> list:
    """[(series_name, series_dir_path, last_frame_filename), ...], one entry
    per series discovered by multi.recrop_plates.discover_series (directories
    only, each must contain at least one image file directly inside it --
    stray files in example_data/ are skipped), sorted by series name. Each
    frame is the series' last frame per pick_last_frame, drawn only from
    list_image_files (non-recursive)."""
    series_dirs = discover_series(Path(example_data_dir))
    result = []
    for name in sorted(series_dirs):
        dir_path = str(series_dirs[name])
        files = list_image_files(dir_path)
        last_frame = pick_last_frame(dir_path, files)
        result.append((name, dir_path, last_frame))
    return result


# ---------------------------------------------------------------------------
# Box-store persistence -- atomic write, resumable, exact schema
# multi.recrop_plates.load_boxes_file requires.
# ---------------------------------------------------------------------------

class BoxStore:
    """Holds {series_name: [{"cx","cy","half_w","half_h"}, ...]} and persists
    it to `path` as JSON on every mutation (atomic: write to a temp file in
    the same directory, then os.replace -- so a crash mid-write never leaves
    a corrupt/partial data/crop_boxes/crop_boxes.json). Loads any existing file at
    construction time so the tool is resumable across runs."""

    def __init__(self, path: str):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def get(self, series_name: str) -> list:
        return [dict(box) for box in self.data.get(series_name, [])]

    def set(self, series_name: str, boxes: list) -> None:
        self.data[series_name] = [dict(box) for box in boxes]
        self._save_atomic()

    def _save_atomic(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".crop_boxes_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, indent=2)
            os.replace(tmp_path, self.path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise


def _rect_to_box(x1, y1, x2, y2) -> dict:
    return {"cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "half_w": (x2 - x1) / 2, "half_h": (y2 - y1) / 2}


def load_reference_boxes(spec: str):
    """(label, {series: [box, ...]}) from an `--existing` value `[LABEL=]PATH`.

    PATH is a crop folder (its manifest.csv), a manifest.csv (one box per (series, crop_id),
    from the pixels actually cut, so the list index is the crop id in the file names) or a boxes
    .json ({series: [{cx, cy, half_w, half_h}, ...]}). The label defaults to the file or folder
    name."""
    label, sep, path = spec.rpartition("=")
    if not sep:
        label, path = "", spec
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"--existing {spec!r}: {path} not found")
    label = label or (path.parent.name if path.suffix == ".csv" else path.stem)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            return label, {k: [dict(b) for b in v] for k, v in json.load(fh).items()}
    import csv
    per = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            per.setdefault(row["series"], {}).setdefault(int(row["crop_id"]), _rect_to_box(
                *(float(row[k]) for k in ("x1", "y1", "x2", "y2"))))
    return label, {series: [boxes[i] for i in sorted(boxes)] for series, boxes in per.items()}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class DrawCropsApp(tk.Tk):
    """The Tk app: series navigation + progress readout around one
    CropBoxCanvas. Frame images are loaded on a background thread (plate
    TIFFs can be multi-megapixel) and handed back to the main thread through
    a ProgressReporter queue, drained by a self-re-arming `self.after` pump
    -- the worker thread never touches any Tk widget directly."""

    POLL_MS = 50

    def __init__(self, series_list: list, box_store: BoxStore, canvas_size=(1000, 720), references=None):
        super().__init__()
        self.title("DLhook - manual crop-box drawing")
        self.series_list = series_list
        self.box_store = box_store
        self.references = list(references or [])      # [(label, {series: [box, ...]})], read-only
        self.index = 0
        self.reporter = ProgressReporter()
        self._load_token = 0

        self._build_ui(canvas_size)
        self._bind_shortcuts()
        self._goto(0)
        self.after(self.POLL_MS, self._pump_progress)

    # ---- UI construction ----

    def _build_ui(self, canvas_size):
        top = tk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        self.status_var = tk.StringVar(value="")
        tk.Label(top, textvariable=self.status_var, anchor="w", font=("Segoe UI", 11)).pack(side=tk.LEFT)
        for k, (label, _) in enumerate(self.references):
            tk.Label(top, text=f"  - - {label}", fg=REFERENCE_COLORS[k % len(REFERENCE_COLORS)],
                     font=("Segoe UI", 10, "bold")).pack(side=tk.RIGHT)

        width, height = canvas_size
        self.canvas = CropBoxCanvas(self, width=width, height=height, on_change=self._on_boxes_changed)
        self.canvas.pack(side=tk.TOP, padx=8, pady=4)

        bottom = tk.Frame(self)
        bottom.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        tk.Button(bottom, text="<< Prev (P)", command=self.prev_series).pack(side=tk.LEFT)
        tk.Button(bottom, text="Next (N) >>", command=self.next_series).pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(bottom, text="Delete box (Del)", command=self._delete_selected).pack(side=tk.LEFT, padx=(18, 0))
        tk.Button(bottom, text="Reset zoom (R)", command=self.canvas.reset_zoom).pack(side=tk.LEFT, padx=(6, 0))
        if self.references:
            tk.Button(bottom, text="Show/hide existing (E)", command=self.canvas.toggle_references).pack(
                side=tk.LEFT, padx=(18, 0))

        help_text = ("Draw: drag empty space   Move: drag a box   Resize: drag its corner handle   "
                     "Select: click a box   Delete: Del/Backspace   Deselect: Esc   "
                     "Prev/Next series: P/N or Left/Right   Zoom: mouse wheel   Reset zoom: R")
        tk.Label(self, text=help_text, anchor="w", fg="#555555", wraplength=canvas_size[0]).pack(
            side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))

    def _bind_shortcuts(self):
        for key in ("<Right>", "<n>", "<N>"):
            self.bind(key, lambda e: self.next_series())
        for key in ("<Left>", "<p>", "<P>"):
            self.bind(key, lambda e: self.prev_series())
        for key in ("<r>", "<R>"):
            self.bind(key, lambda e: self.canvas.reset_zoom())
        self.bind("<Escape>", lambda e: self.canvas.clear_selection())
        for key in ("<e>", "<E>"):
            self.bind(key, lambda e: self.canvas.toggle_references())

    # ---- series navigation / async frame loading ----

    def next_series(self):
        if self.index < len(self.series_list) - 1:
            self._goto(self.index + 1)

    def prev_series(self):
        if self.index > 0:
            self._goto(self.index - 1)

    def _delete_selected(self):
        self.canvas.delete_selected()

    def _goto(self, index: int):
        self.index = index
        name, dir_path, frame_name = self.series_list[index]
        self.status_var.set(
            f"series {index + 1}/{len(self.series_list)}, loading {frame_name} ...  ({name})")

        self._load_token += 1
        token = self._load_token
        frame_path = os.path.join(dir_path, frame_name)

        def worker(token=token, frame_path=frame_path):
            image = cv2.imread(frame_path)
            self.reporter.report(kind="frame_loaded", token=token, image=image)

        threading.Thread(target=worker, daemon=True).start()

    def _pump_progress(self):
        for event in self.reporter.drain():
            if event.get("kind") == "frame_loaded" and event.get("token") == self._load_token:
                self._apply_loaded_frame(event["image"])
        self.after(self.POLL_MS, self._pump_progress)

    def _apply_loaded_frame(self, image):
        name, dir_path, frame_name = self.series_list[self.index]
        if image is None:
            self.status_var.set(
                f"series {self.index + 1}/{len(self.series_list)}, FAILED to load {frame_name}  ({name})")
            return

        self.canvas.set_image(image)
        self.canvas.reset_zoom()  # set_image only auto-fits on the very first image ever
        self.canvas.set_references([(label, REFERENCE_COLORS[k % len(REFERENCE_COLORS)], per.get(name, []))
                                    for k, (label, per) in enumerate(self.references)])
        self.canvas.set_boxes(self.box_store.get(name))
        self._update_status()

    def _on_boxes_changed(self, boxes):
        name = self.series_list[self.index][0]
        self.box_store.set(name, boxes)
        self._update_status()

    def _update_status(self):
        name = self.series_list[self.index][0]
        n_boxes = len(self.canvas.boxes)
        text = f"series {self.index + 1}/{len(self.series_list)}, {n_boxes} boxes  ({name})"
        n_ref = sum(len(b) for _, _, b in self.canvas.references)
        if self.references:
            text += f"  -  {n_ref} existing boxes shown"
            conflicts = self.canvas.conflicts()
            if conflicts:
                worst = max(conflicts, key=lambda c: c[1])
                text += (f"  -  {len(conflicts)} of yours touch one (worst: box {worst[0] + 1} x "
                         f"{worst[2]}#{worst[3]}, {worst[1]:.0%})")
        self.status_var.set(text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--example-data", default=None,
                         help="Directory of series subfolders (default: <repo>/example_data)")
    parser.add_argument("--out", default=None,
                         help="data/crop_boxes/crop_boxes.json path -- loaded on startup if it exists (resumable), "
                              "saved atomically on every box edit (default: data/crop_boxes/crop_boxes.json)")
    parser.add_argument("--existing", action="append", default=[], metavar="[LABEL=]PATH",
                         help="Show an earlier set of boxes read-only (repeatable): a crop folder, its "
                              "manifest.csv, or a boxes .json. Your boxes that touch one turn magenta.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    example_data_dir = args.example_data or os.path.join(_REPO_ROOT, "example_data")
    out_path = args.out or os.path.join(_REPO_ROOT, "data/crop_boxes/crop_boxes.json")

    series_list = discover_series_last_frames(example_data_dir)
    if not series_list:
        print(f"No series with image files found under {example_data_dir}")
        return 1

    references = [load_reference_boxes(spec) for spec in args.existing]
    for label, per in references:
        print(f"existing '{label}': {sum(len(v) for v in per.values())} boxes in {len(per)} series")
    box_store = BoxStore(out_path)
    app = DrawCropsApp(series_list, box_store, references=references)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
