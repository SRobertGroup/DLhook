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
    python -m multi.draw_crops --example-data example_data --out crop_boxes.json

Drawing a validation set next to existing ones (read-only reference boxes):
    python -m multi.draw_crops --out crop_boxes_validation.json --existing training=cropped_training_set
        --existing new=cropped_new_set --existing open=cropped_open_set
`--existing [LABEL=]PATH` is repeatable; PATH is a crop folder (its manifest.csv, which records what
was actually cut), a manifest.csv, or a boxes .json. Each reference set is drawn per series in its own
colour as thin dashed boxes tagged LABEL#seedling-number, on top of the plate; they cannot be
selected, moved or drawn through by accident. A box of yours that touches a reference box turns
magenta and the status line names the worst overlap, so a validation set can be kept free of plants
that were trained on. Press E to hide / show the references.

Output (crop_boxes.json), consumed unmodified by
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
from multi.src.recrop_geometry import (  # noqa: E402
    DEFAULT_MIN_BOX_HALF_SIZE,
    compute_crop_box,
    list_image_files,
)
from ui.zoomable_canvas import ZoomableImageCanvas  # noqa: E402
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
    a corrupt/partial crop_boxes.json). Loads any existing file at
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


# ---------------------------------------------------------------------------
# Pure geometry helpers used by CropBoxCanvas during a resize drag -- kept
# free of Tk state so they can be unit-tested directly.
# ---------------------------------------------------------------------------

def resize_rect_from_corners(x1: float, y1: float, x2: float, y2: float,
                              min_half_size: float = DEFAULT_MIN_BOX_HALF_SIZE) -> dict:
    """A plain axis-aligned rectangle from two opposite corners (image
    space), floored at `min_half_size` -- deliberately NOT compute_crop_box:
    resizing an existing (already-padded) box must not re-apply padding."""
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)
    half_w = max((max_x - min_x) / 2, min_half_size)
    half_h = max((max_y - min_y) / 2, min_half_size)
    return {"cx": (min_x + max_x) / 2, "cy": (min_y + max_y) / 2, "half_w": half_w, "half_h": half_h}


REFERENCE_COLORS = ("#2e86de", "#27ae60", "#8e44ad", "#16a085", "#c0392b", "#7f8c8d")


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


def box_overlap(box: dict, other: dict) -> float:
    """Shared area over the smaller box (0 = disjoint, 1 = one lies inside the other); the measure
    multi.check_crop_overlap reports."""
    from multi.check_crop_overlap import overlap_fraction
    from utils.crop_layout import box_to_rect
    return overlap_fraction(box_to_rect(box), box_to_rect(other))


def worst_reference_overlap(box: dict, references: list):
    """(fraction, label, seedling_number) of the reference box `box` overlaps most, or (0.0, "", 0).
    `references` is [(label, [box, ...])] for the current series."""
    best = (0.0, "", 0)
    for label, boxes in references:
        for i, ref in enumerate(boxes):
            frac = box_overlap(box, ref)
            if frac > best[0]:
                best = (frac, label, i + 1)
    return best


def round_box(box: dict) -> dict:
    """Round a box's fields to int -- the schema load_boxes_file/recrop_plates
    expect (drag arithmetic produces floats)."""
    return {
        "cx": int(round(box["cx"])), "cy": int(round(box["cy"])),
        "half_w": int(round(box["half_w"])), "half_h": int(round(box["half_h"])),
    }


# ---------------------------------------------------------------------------
# Canvas -- extends ZoomableImageCanvas (ui/zoomable_canvas.py, off limits to
# edit) rather than reimplementing zoom/pan/image display.
# ---------------------------------------------------------------------------

class CropBoxCanvas(ZoomableImageCanvas):
    """Adds crop-box draw/move/resize/select/delete on top of
    ZoomableImageCanvas. All box geometry (self.boxes, drag state) is kept in
    IMAGE space; only _box_canvas_rect / _handle_positions ever convert to
    canvas space, and only for hit-testing/drawing.

    Mouse map (bound directly on the canvas):
      - Button-1 on empty space, drag, release: draw a new box (drag defines
        the seedling extent; the padded box is shown live and is what gets
        stored -- see module docstring).
      - Button-1 on a box's body, drag: select it and move it.
      - Button-1 on a selected box's corner handle, drag: resize it.
    Keyboard (bound on the canvas; <Enter> grabs focus so these work):
      - Delete / BackSpace: delete the selected box.
    """

    HANDLE_HIT_PX = 12   # hit-test tolerance in SCREEN/canvas pixels, zoom-independent
    HANDLE_DRAW_PX = 5
    MIN_DRAG_PX = 4       # below this, a press+release is a click, not a draw
    MIN_HALF_SIZE = DEFAULT_MIN_BOX_HALF_SIZE

    COLOR_NORMAL = "#d78a5e"
    COLOR_SELECTED = "#ff3b30"
    COLOR_PREVIEW = "#00c8ff"
    COLOR_CONFLICT = "#ff00aa"      # a box that touches a reference (--existing) box

    def __init__(self, parent, on_change=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.boxes = []            # image-space dicts: cx, cy, half_w, half_h
        self.selected_index = None
        self.on_change = on_change  # called with get_boxes() after each completed mutation

        self._drag_mode = None            # None | "draw" | "move" | "resize"
        self._drag_anchor_img = None      # draw/resize: the fixed reference point (image space)
        self._drag_start_canvas = None    # draw/move: press position (canvas space)
        self._drag_orig_box = None        # move: a copy of the box being moved
        self._drag_handle = None          # resize: which corner is being dragged
        self._preview_box = None          # draw: live padded-preview box
        self.references = []              # [(label, color, [box, ...])] of the series on screen, read-only
        self.show_references = True

        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", lambda e: self.focus_set())
        self.bind("<Delete>", self._on_delete_key)
        self.bind("<BackSpace>", self._on_delete_key)

    # ---- public API used by the app ----

    def set_boxes(self, boxes: list) -> None:
        self.boxes = [dict(box) for box in boxes]
        self.selected_index = None
        self._drag_mode = None
        self._preview_box = None
        self._redraw()

    def get_boxes(self) -> list:
        return [dict(box) for box in self.boxes]

    def set_references(self, references: list) -> None:
        """Read-only boxes drawn under the editable ones: [(label, color, [box, ...])]."""
        self.references = [(label, color, [dict(b) for b in boxes]) for label, color, boxes in references]
        self._redraw()

    def toggle_references(self) -> None:
        self.show_references = not self.show_references
        self._redraw()

    def conflicts(self) -> list:
        """[(index, fraction, label, seedling_number)] of the editable boxes that touch a reference box."""
        refs = [(label, boxes) for label, _, boxes in self.references]
        out = []
        for i, box in enumerate(self.boxes):
            frac, label, number = worst_reference_overlap(box, refs)
            if frac > 0:
                out.append((i, frac, label, number))
        return out

    def delete_selected(self) -> bool:
        if self.selected_index is None:
            return False
        del self.boxes[self.selected_index]
        self.selected_index = None
        self._notify_change()
        self._redraw()
        return True

    def clear_selection(self) -> None:
        self.selected_index = None
        self._drag_mode = None
        self._preview_box = None
        self._redraw()

    # ---- image space <-> canvas space (box geometry only; pixel drawing is
    # inherited from ZoomableImageCanvas) ----

    def _img_to_canvas(self, x, y):
        x1, y1, _, _ = self._visible_bounds()
        return (x - x1) * self.zoom, (y - y1) * self.zoom

    def _box_canvas_rect(self, box):
        cx1, cy1 = self._img_to_canvas(box["cx"] - box["half_w"], box["cy"] - box["half_h"])
        cx2, cy2 = self._img_to_canvas(box["cx"] + box["half_w"], box["cy"] + box["half_h"])
        return cx1, cy1, cx2, cy2

    def _handle_positions(self, box):
        x1, y1, x2, y2 = self._box_canvas_rect(box)
        return {"nw": (x1, y1), "ne": (x2, y1), "sw": (x1, y2), "se": (x2, y2)}

    def _hit_handle(self, canvas_x, canvas_y):
        """Handle hit-test of the currently SELECTED box only, within
        HANDLE_HIT_PX screen pixels regardless of zoom."""
        if self.selected_index is None:
            return None
        box = self.boxes[self.selected_index]
        for name, (hx, hy) in self._handle_positions(box).items():
            if abs(canvas_x - hx) <= self.HANDLE_HIT_PX and abs(canvas_y - hy) <= self.HANDLE_HIT_PX:
                return name
        return None

    def _hit_box(self, canvas_x, canvas_y):
        """Topmost (last-drawn) box whose body contains this canvas point."""
        for i in range(len(self.boxes) - 1, -1, -1):
            x1, y1, x2, y2 = self._box_canvas_rect(self.boxes[i])
            if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                return i
        return None

    # ---- mouse handlers ----

    def _on_press(self, event):
        self.focus_set()
        if self._image is None:
            return

        handle = self._hit_handle(event.x, event.y)
        if handle is not None:
            box = self.boxes[self.selected_index]
            opposite = {"nw": "se", "ne": "sw", "sw": "ne", "se": "nw"}[handle]
            corners = {
                "nw": (box["cx"] - box["half_w"], box["cy"] - box["half_h"]),
                "ne": (box["cx"] + box["half_w"], box["cy"] - box["half_h"]),
                "sw": (box["cx"] - box["half_w"], box["cy"] + box["half_h"]),
                "se": (box["cx"] + box["half_w"], box["cy"] + box["half_h"]),
            }
            self._drag_mode = "resize"
            self._drag_handle = handle
            self._drag_anchor_img = corners[opposite]
            return

        hit_idx = self._hit_box(event.x, event.y)
        if hit_idx is not None:
            self.selected_index = hit_idx
            self._drag_mode = "move"
            self._drag_start_canvas = (event.x, event.y)
            self._drag_orig_box = dict(self.boxes[hit_idx])
            self._redraw()
            return

        # Empty space: start drawing a new box; deselect whatever was selected.
        self.selected_index = None
        self._drag_mode = "draw"
        self._drag_anchor_img = self.canvas_to_image(event.x, event.y)
        self._drag_start_canvas = (event.x, event.y)
        self._preview_box = None
        self._redraw()

    def _on_drag(self, event):
        if self._drag_mode is None or self._image is None:
            return

        if self._drag_mode == "draw":
            img_x, img_y = self.canvas_to_image(event.x, event.y)
            ax, ay = self._drag_anchor_img
            min_x, max_x = min(ax, img_x), max(ax, img_x)
            min_y, max_y = min(ay, img_y), max(ay, img_y)
            # Live preview shows the PADDED box -- what you see is what gets
            # cropped, even while still dragging.
            self._preview_box = compute_crop_box(min_x, max_x, min_y, max_y)
            self._redraw()

        elif self._drag_mode == "move":
            sx, sy = self._drag_start_canvas
            dx_img = (event.x - sx) / self.zoom
            dy_img = (event.y - sy) / self.zoom
            box = self.boxes[self.selected_index]
            box["cx"] = self._drag_orig_box["cx"] + dx_img
            box["cy"] = self._drag_orig_box["cy"] + dy_img
            self._redraw()

        elif self._drag_mode == "resize":
            img_x, img_y = self.canvas_to_image(event.x, event.y)
            ax, ay = self._drag_anchor_img
            new_box = resize_rect_from_corners(ax, ay, img_x, img_y, self.MIN_HALF_SIZE)
            self.boxes[self.selected_index].update(new_box)
            self._redraw()

    def _on_release(self, event):
        if self._drag_mode == "draw":
            if self._preview_box is not None:
                sx, sy = self._drag_start_canvas
                dragged_far_enough = (abs(event.x - sx) >= self.MIN_DRAG_PX
                                       or abs(event.y - sy) >= self.MIN_DRAG_PX)
                if dragged_far_enough:
                    self.boxes.append(round_box(self._preview_box))
                    self.selected_index = len(self.boxes) - 1
                    self._notify_change()
            self._preview_box = None

        elif self._drag_mode in ("move", "resize") and self.selected_index is not None:
            self.boxes[self.selected_index] = round_box(self.boxes[self.selected_index])
            self._notify_change()

        self._drag_mode = None
        self._drag_handle = None
        self._drag_anchor_img = None
        self._drag_start_canvas = None
        self._drag_orig_box = None
        self._redraw()

    def _on_delete_key(self, event):
        self.delete_selected()
        return "break"

    def _notify_change(self):
        if self.on_change is not None:
            self.on_change(self.get_boxes())

    # ---- drawing ----

    def _redraw(self):
        # ZoomableImageCanvas._redraw does self.delete("all"); persistent
        # canvas item ids never survive, so every box is redrawn from
        # scratch on top of it each time.
        super()._redraw()
        if self._image is None:
            return
        if self.show_references:
            for label, color, boxes in self.references:
                for n, ref in enumerate(boxes, start=1):
                    x1, y1, x2, y2 = self._box_canvas_rect(ref)
                    self.create_rectangle(x1, y1, x2, y2, outline=color, width=1, dash=(5, 3), tags="ref_box")
                    self.create_text(x1 + 3, y1 + 2, text=f"{label}#{n}", anchor="nw", fill=color,
                                     font=("Segoe UI", 8, "bold"), tags="ref_box")
        conflicting = {i for i, *_ in self.conflicts()} if self.references else set()
        for i, box in enumerate(self.boxes):
            self._draw_one_box(box, selected=(i == self.selected_index), conflict=(i in conflicting))
        if self._preview_box is not None:
            self._draw_one_box(self._preview_box, selected=True, preview=True)

    def _draw_one_box(self, box, selected, preview=False, conflict=False):
        x1, y1, x2, y2 = self._box_canvas_rect(box)
        if preview:
            outline, width, dash = self.COLOR_PREVIEW, 2, (4, 2)
        elif selected:
            outline, width, dash = self.COLOR_SELECTED, 3, None
        elif conflict:
            outline, width, dash = self.COLOR_CONFLICT, 3, None
        else:
            outline, width, dash = self.COLOR_NORMAL, 2, None

        kwargs = {"outline": outline, "width": width, "tags": "crop_box"}
        if dash:
            kwargs["dash"] = dash
        self.create_rectangle(x1, y1, x2, y2, **kwargs)

        if selected and not preview:
            r = self.HANDLE_DRAW_PX
            for hx, hy in self._handle_positions(box).values():
                self.create_rectangle(hx - r, hy - r, hx + r, hy + r,
                                       fill=outline, outline="white", tags="crop_box")


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
                         help="crop_boxes.json path -- loaded on startup if it exists (resumable), "
                              "saved atomically on every box edit (default: <repo>/crop_boxes.json)")
    parser.add_argument("--existing", action="append", default=[], metavar="[LABEL=]PATH",
                         help="Show an earlier set of boxes read-only (repeatable): a crop folder, its "
                              "manifest.csv, or a boxes .json. Your boxes that touch one turn magenta.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    example_data_dir = args.example_data or os.path.join(_REPO_ROOT, "example_data")
    out_path = args.out or os.path.join(_REPO_ROOT, "crop_boxes.json")

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
