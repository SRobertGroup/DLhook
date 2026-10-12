"""Zoomable canvas for drawing, moving, resizing and deleting seedling crop boxes.

Shared by the main window (seedling_measurement.py) and the stand-alone crop drawing tool
(multi/draw_crops.py). It extends ZoomableImageCanvas rather than reimplementing zoom and image
display. The GUI must not import multi/, so everything the canvas needs lives in ui/ and utils/.

Design decision -- a drag defines the SEEDLING EXTENT, not the final box:
    Dragging on empty canvas draws the seedling's bounding extent (its visible top, bottom, left
    and right); utils.crop_layout.compute_crop_box then pads that extent (1.8x width, 1.2x
    height, floored at a 30 px half-size) into the box that is stored and later cropped. The
    PADDED result is what is drawn live while dragging, so what you see is what will be cropped.

    Once a box exists, moving it (drag its body) or resizing it (drag a corner handle of the
    selected box) edits that already-padded box DIRECTLY -- it never re-runs compute_crop_box,
    which would pad it a second time. Resize is a plain rectangle edit floored at the same
    minimum half-size.

Optional features, all off by default so draw_crops behaves as before:
    allow_collar        Shift+click inside a box sets / drags its collar override (stored as
                        box["collar"] = [x, y] in image pixels, drawn as a purple dot);
                        Shift+right-click inside a box clears it.
    highlight_overlaps  boxes that overlap another box are drawn magenta.
    show_labels         every box carries its 1-based number (its position in self.boxes).
    auto_resize         the canvas follows the size of its widget (grid/pack with expand).
    bind_view_keys()    Escape deselects, R resets the zoom (bound on the canvas).
    order_key / normalize are the application's business: see set_boxes(selected=...).
"""
from __future__ import annotations

import tkinter as tk

from ui.zoomable_canvas import ZoomableImageCanvas
from utils.crop_layout import (
    DEFAULT_MIN_BOX_HALF_SIZE,
    box_to_rect,
    compute_crop_box,
    find_overlapping_pairs,
)

REFERENCE_COLORS = ("#2e86de", "#27ae60", "#8e44ad", "#16a085", "#c0392b", "#7f8c8d")
COLLAR_COLOR = "#7b2cbf"
SHIFT_MASK = 0x0001


# ---------------------------------------------------------------------------
# Pure helpers (no Tk state; unit-tested directly).
# ---------------------------------------------------------------------------

def resize_rect_from_corners(x1: float, y1: float, x2: float, y2: float,
                              min_half_size: float = DEFAULT_MIN_BOX_HALF_SIZE) -> dict:
    """A plain axis-aligned rectangle from two opposite corners (image space), floored at
    `min_half_size` -- deliberately NOT compute_crop_box: resizing an existing (already padded)
    box must not re-apply padding."""
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)
    half_w = max((max_x - min_x) / 2, min_half_size)
    half_h = max((max_y - min_y) / 2, min_half_size)
    return {"cx": (min_x + max_x) / 2, "cy": (min_y + max_y) / 2, "half_w": half_w, "half_h": half_h}


def round_box(box: dict) -> dict:
    """Round a box's fields to int -- the schema load_boxes_file/recrop_plates expect (drag
    arithmetic produces floats). An optional "collar" [x, y] is kept (rounded too)."""
    out = {
        "cx": int(round(box["cx"])), "cy": int(round(box["cy"])),
        "half_w": int(round(box["half_w"])), "half_h": int(round(box["half_h"])),
    }
    if box.get("collar") is not None:
        out["collar"] = [int(round(box["collar"][0])), int(round(box["collar"][1]))]
    return out


def clamp_point_to_box(x: float, y: float, box: dict):
    x1, y1, x2, y2 = box_to_rect(box)
    return min(max(x, x1), x2), min(max(y, y1), y2)


def box_overlap(box: dict, other: dict) -> float:
    """Shared area over the smaller box (0 = disjoint, 1 = one lies inside the other); the same
    measure multi.check_crop_overlap.overlap_fraction reports."""
    a, b = box_to_rect(box), box_to_rect(other)
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return (w * h) / smaller if smaller > 0 else 0.0


def worst_reference_overlap(box: dict, references: list):
    """(fraction, label, seedling_number) of the reference box `box` overlaps most, or
    (0.0, "", 0). `references` is [(label, [box, ...])] for the current series."""
    best = (0.0, "", 0)
    for label, boxes in references:
        for i, ref in enumerate(boxes):
            frac = box_overlap(box, ref)
            if frac > best[0]:
                best = (frac, label, i + 1)
    return best


def _copy_box(box: dict) -> dict:
    out = dict(box)
    if out.get("collar") is not None:
        out["collar"] = list(out["collar"])
    return out


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------

class CropBoxCanvas(ZoomableImageCanvas):
    """Crop-box draw/move/resize/select/delete on top of ZoomableImageCanvas. All box geometry
    (self.boxes, drag state) is kept in IMAGE space; only the *_canvas_* helpers convert to
    canvas space, and only for hit-testing and drawing.

    Mouse map (bound directly on the canvas):
      - Button-1 on empty space, drag, release: draw a new box (see module docstring).
      - Button-1 on a box's body, drag: select it and move it.
      - Button-1 on a selected box's corner handle, drag: resize it.
      - Shift+Button-1 inside a box (allow_collar): set / drag that box's collar.
      - Shift+Button-3 inside a box (allow_collar): clear that box's collar.
      - Button-3 or Button-2 drag (no Shift): pan the zoomed view.
      - Mouse wheel: zoom around the cursor (ZoomableImageCanvas).
    Keyboard (bound on the canvas; <Enter> grabs focus so these work):
      - Delete / BackSpace: delete the selected box.
      - Escape / R (after bind_view_keys()): deselect / reset zoom.
    """

    HANDLE_HIT_PX = 12   # hit-test tolerance in SCREEN/canvas pixels, zoom-independent
    HANDLE_DRAW_PX = 5
    MIN_DRAG_PX = 4       # below this, a press+release is a click, not a draw
    MIN_HALF_SIZE = DEFAULT_MIN_BOX_HALF_SIZE
    COLLAR_DRAW_PX = 5

    COLOR_NORMAL = "#d78a5e"
    COLOR_SELECTED = "#ff3b30"
    COLOR_PREVIEW = "#00c8ff"
    COLOR_CONFLICT = "#ff00aa"      # a box that touches a reference box / another box
    COLOR_COLLAR = COLLAR_COLOR

    def __init__(self, parent, on_change=None, allow_collar=False, highlight_overlaps=False,
                 show_labels=False, auto_resize=False, **kwargs):
        super().__init__(parent, **kwargs)
        self.boxes = []            # image-space dicts: cx, cy, half_w, half_h [, collar]
        self.selected_index = None
        self.on_change = on_change  # called with get_boxes() after each completed mutation
        self.allow_collar = allow_collar
        self.highlight_overlaps = highlight_overlaps
        self.show_labels = show_labels

        self._drag_mode = None            # None | "draw" | "move" | "resize" | "collar" | "pan"
        self._drag_anchor_img = None      # draw/resize: the fixed reference point (image space)
        self._drag_start_canvas = None    # draw/move/pan: press position (canvas space)
        self._drag_orig_box = None        # move: a copy of the box being moved
        self._drag_handle = None          # resize: which corner is being dragged
        self._drag_index = None           # collar: the box whose collar is being set
        self._preview_box = None          # draw: live padded-preview box
        self.references = []              # [(label, color, [box, ...])] of the series on screen, read-only
        self.show_references = True

        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", lambda e: self.focus_set())
        self.bind("<Delete>", self._on_delete_key)
        self.bind("<BackSpace>", self._on_delete_key)
        self.bind("<Button-3>", self._on_right_press)
        self.bind("<B3-Motion>", self._on_pan_drag)
        self.bind("<ButtonRelease-3>", self._on_pan_release)
        self.bind("<Button-2>", self._on_pan_press)
        self.bind("<B2-Motion>", self._on_pan_drag)
        self.bind("<ButtonRelease-2>", self._on_pan_release)
        if auto_resize:
            self.bind("<Configure>", self._on_configure)

    # ---- public API used by the app ----

    def set_boxes(self, boxes: list, selected=None) -> None:
        self.boxes = [_copy_box(box) for box in boxes]
        self.selected_index = selected if selected is not None and 0 <= selected < len(self.boxes) else None
        self._drag_mode = None
        self._preview_box = None
        self._redraw()

    def get_boxes(self) -> list:
        return [_copy_box(box) for box in self.boxes]

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

    def overlapping_pairs(self) -> list:
        """Index pairs (i, j), i < j, of editable boxes that overlap each other."""
        return find_overlapping_pairs(self.boxes)

    def delete_selected(self) -> bool:
        if self.selected_index is None:
            return False
        del self.boxes[self.selected_index]
        self.selected_index = None
        self._notify_change()
        self._redraw()
        return True

    def set_collar(self, index: int, x: float, y: float) -> None:
        """Set the collar override of box `index` (image pixels, clamped into the box)."""
        box = self.boxes[index]
        cx, cy = clamp_point_to_box(x, y, box)
        box["collar"] = [cx, cy]

    def clear_collar(self, index=None) -> bool:
        """Remove the collar override of box `index` (default: the selected box)."""
        index = self.selected_index if index is None else index
        if index is None or index >= len(self.boxes) or self.boxes[index].get("collar") is None:
            return False
        del self.boxes[index]["collar"]
        self._notify_change()
        self._redraw()
        return True

    def clear_selection(self) -> None:
        self.selected_index = None
        self._drag_mode = None
        self._preview_box = None
        self._redraw()

    def bind_view_keys(self) -> None:
        """Escape deselects, R / r resets the zoom (bound on the canvas, which takes focus on <Enter>,
        so typing in an Entry elsewhere in the window does not trigger them)."""
        self.bind("<Escape>", lambda e: self.clear_selection())
        for key in ("<r>", "<R>"):
            self.bind(key, lambda e: self.reset_zoom())

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
        """Handle hit-test of the currently SELECTED box only, within HANDLE_HIT_PX screen
        pixels regardless of zoom."""
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

        if self.allow_collar and event.state & SHIFT_MASK:
            hit = self._hit_box(event.x, event.y)
            if hit is not None:
                self.selected_index = hit
                self._drag_mode = "collar"
                self._drag_index = hit
                self._apply_collar_at(event)
                self._redraw()
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
            self._drag_orig_box = _copy_box(self.boxes[hit_idx])
            self._redraw()
            return

        # Empty space: start drawing a new box; deselect whatever was selected.
        self.selected_index = None
        self._drag_mode = "draw"
        self._drag_anchor_img = self.canvas_to_image(event.x, event.y)
        self._drag_start_canvas = (event.x, event.y)
        self._preview_box = None
        self._redraw()

    def _apply_collar_at(self, event):
        x, y = self.canvas_to_image(event.x, event.y)
        self.set_collar(self._drag_index, x, y)

    def _on_drag(self, event):
        if self._drag_mode in (None, "pan") or self._image is None:
            return

        if self._drag_mode == "draw":
            img_x, img_y = self.canvas_to_image(event.x, event.y)
            ax, ay = self._drag_anchor_img
            min_x, max_x = min(ax, img_x), max(ax, img_x)
            min_y, max_y = min(ay, img_y), max(ay, img_y)
            # Live preview shows the PADDED box -- what you see is what gets cropped.
            self._preview_box = compute_crop_box(min_x, max_x, min_y, max_y)
            self._redraw()

        elif self._drag_mode == "move":
            sx, sy = self._drag_start_canvas
            dx_img = (event.x - sx) / self.zoom
            dy_img = (event.y - sy) / self.zoom
            box = self.boxes[self.selected_index]
            box["cx"] = self._drag_orig_box["cx"] + dx_img
            box["cy"] = self._drag_orig_box["cy"] + dy_img
            if self._drag_orig_box.get("collar") is not None:    # the collar travels with its box
                ox, oy = self._drag_orig_box["collar"]
                box["collar"] = [ox + dx_img, oy + dy_img]
            self._redraw()

        elif self._drag_mode == "resize":
            img_x, img_y = self.canvas_to_image(event.x, event.y)
            ax, ay = self._drag_anchor_img
            new_box = resize_rect_from_corners(ax, ay, img_x, img_y, self.MIN_HALF_SIZE)
            self.boxes[self.selected_index].update(new_box)
            self._redraw()

        elif self._drag_mode == "collar":
            self._apply_collar_at(event)
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
                    self._preview_box = None
                    self._finish_drag()
                    self._notify_change()
                    self._redraw()
                    return
            self._preview_box = None

        elif self._drag_mode in ("move", "resize", "collar") and self.selected_index is not None:
            idx = self.selected_index
            box = round_box(self.boxes[idx])
            if box.get("collar") is not None:        # a resize may have left the collar outside
                box["collar"] = list(clamp_point_to_box(box["collar"][0], box["collar"][1], box))
                box["collar"] = [int(round(box["collar"][0])), int(round(box["collar"][1]))]
            self.boxes[idx] = box
            self._finish_drag()
            self._notify_change()
            self._redraw()
            return

        self._finish_drag()
        self._redraw()

    def _finish_drag(self):
        self._drag_mode = None
        self._drag_handle = None
        self._drag_anchor_img = None
        self._drag_start_canvas = None
        self._drag_orig_box = None
        self._drag_index = None

    def _on_right_press(self, event):
        self.focus_set()
        if self._image is None:
            return
        if self.allow_collar and event.state & SHIFT_MASK:
            hit = self._hit_box(event.x, event.y)
            if hit is not None:
                self.clear_collar(hit)
            return
        self._on_pan_press(event)

    def _on_pan_press(self, event):
        if self._image is None or self._drag_mode is not None:
            return
        self._drag_mode = "pan"
        self._drag_start_canvas = (event.x, event.y)

    def _on_pan_drag(self, event):
        if self._drag_mode != "pan" or self._image is None:
            return
        sx, sy = self._drag_start_canvas
        self._center_x -= (event.x - sx) / self.zoom
        self._center_y -= (event.y - sy) / self.zoom
        self._drag_start_canvas = (event.x, event.y)
        self._clamp_center()
        self._redraw()

    def _on_pan_release(self, event):
        if self._drag_mode == "pan":
            self._finish_drag()

    def _on_delete_key(self, event):
        self.delete_selected()
        return "break"

    def _notify_change(self):
        if self.on_change is not None:
            self.on_change(self.get_boxes())

    # ---- window resize ----

    def _on_configure(self, event):
        if event.width < 40 or event.height < 40:
            return
        if (event.width, event.height) == (self.display_width, self.display_height):
            return
        was_fit = self._image is None or abs(self.zoom - self._base_zoom) < 1e-9
        self.display_width, self.display_height = event.width, event.height
        if self._image is None:
            return
        if was_fit:
            self.reset_zoom()
            return
        h, w = self._image.shape[:2]
        self._base_zoom = min(self.display_width / w, self.display_height / h)
        self.zoom = max(self.zoom, self._base_zoom)
        self._clamp_center()
        self._redraw()

    # ---- drawing ----

    def _redraw(self):
        # ZoomableImageCanvas._redraw does self.delete("all"); persistent canvas item ids never
        # survive, so every box is redrawn from scratch on top of it each time.
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
        if self.highlight_overlaps:
            conflicting |= {k for pair in self.overlapping_pairs() for k in pair}
        for i, box in enumerate(self.boxes):
            self._draw_one_box(box, selected=(i == self.selected_index), conflict=(i in conflicting),
                               label=str(i + 1) if self.show_labels else None)
        if self._preview_box is not None:
            self._draw_one_box(self._preview_box, selected=True, preview=True)

    def _draw_one_box(self, box, selected, preview=False, conflict=False, label=None):
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

        if label is not None and not preview:
            # Seedling number (crop_id + 1) just inside the top-left corner, clear of the handle.
            tx, ty = x1 + self.HANDLE_DRAW_PX + 3, y1 + self.HANDLE_DRAW_PX + 1
            text = self.create_text(tx, ty, text=label, anchor=tk.NW, fill="white",
                                    font=("Helvetica", 11, "bold"), tags="crop_box")
            bbox = self.bbox(text)
            if bbox:
                bg = self.create_rectangle(bbox[0] - 2, bbox[1], bbox[2] + 2, bbox[3],
                                           fill=outline, outline="", tags="crop_box")
                self.tag_lower(bg, text)

        if self.allow_collar and not preview and box.get("collar") is not None:
            px, py = self._img_to_canvas(box["collar"][0], box["collar"][1])
            r = self.COLLAR_DRAW_PX
            self.create_oval(px - r, py - r, px + r, py + r, fill=self.COLOR_COLLAR,
                             outline="white", width=1, tags="crop_box")
