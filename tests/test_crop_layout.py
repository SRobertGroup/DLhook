"""Tests for utils/crop_layout.py (non-overlapping, spatially numbered crop
boxes) and its wiring into Gui's point-placement / undo / resize handlers.

GUI tests follow tests/test_gui_segmentation_routing.py: Gui is built with
`Gui.__new__(Gui)` and given only the attributes the methods touch, with a
fake canvas standing in for Tk.
"""
from __future__ import annotations

import random

from utils.crop_layout import (
    box_to_rect, clamp_to_image, clamp_to_neighbours, content_rect,
    find_overlapping_pairs, rect_to_box, rects_overlap, resolve_overlaps,
    spatial_order,
)


def _box(x1, y1, x2, y2):
    return rect_to_box(x1, y1, x2, y2)


def _contains(outer, inner):
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


# --- pure geometry ---------------------------------------------------------

def test_rect_to_box_stays_inside_odd_rects():
    box = rect_to_box(10, 10, 21, 31)
    assert _contains((10, 10, 21, 31), box_to_rect(box))


def test_touching_rects_do_not_overlap():
    assert not rects_overlap((0, 0, 10, 10), (10, 0, 20, 10))
    assert rects_overlap((0, 0, 11, 10), (10, 0, 20, 10))


def test_side_by_side_boxes_split_at_content_midline():
    contents = [(100, 50, 110, 300), (150, 60, 160, 290)]
    boxes = [_box(60, 20, 150, 330), _box(110, 30, 200, 320)]
    out = resolve_overlaps(boxes, contents)

    assert find_overlapping_pairs(out) == []
    left, right = box_to_rect(out[0]), box_to_rect(out[1])
    assert left[2] <= 130 <= right[0]          # split at (110 + 150) // 2
    assert _contains(left, contents[0]) and _contains(right, contents[1])


def test_vertically_stacked_boxes_split_along_y():
    contents = [(100, 50, 120, 150), (100, 200, 120, 300)]
    boxes = [_box(80, 30, 140, 190), _box(80, 160, 140, 320)]
    out = resolve_overlaps(boxes, contents)
    assert find_overlapping_pairs(out) == []
    assert box_to_rect(out[0])[3] <= 175 <= box_to_rect(out[1])[1]


def test_untouched_boxes_are_returned_unchanged():
    boxes = [_box(0, 0, 50, 50), _box(100, 0, 150, 50)]
    out = resolve_overlaps(boxes, [(10, 10, 40, 40), (110, 10, 140, 40)])
    assert out[0] is boxes[0] and out[1] is boxes[1]


def test_crossing_contents_are_left_overlapping():
    contents = [(100, 50, 160, 300), (120, 60, 180, 290)]
    boxes = [_box(80, 30, 180, 320), _box(100, 40, 200, 310)]
    out = resolve_overlaps(boxes, contents)
    assert find_overlapping_pairs(out) == [(0, 1)]


def test_random_rows_end_up_non_overlapping_and_keep_content():
    rng = random.Random(0)
    for n in range(5, 16):
        contents, boxes = [], []
        for k in range(n):
            x = 60 * k + rng.randint(0, 10)
            start, end = (x, 400 + rng.randint(-5, 5)), (x + rng.randint(-8, 8), 400 - rng.randint(80, 200))
            c = content_rect(start, end)
            contents.append(c)
            boxes.append(_box(c[0] - 45, c[1] - 20, c[2] + 45, c[3] + 20))
        out = resolve_overlaps(boxes, contents)
        assert find_overlapping_pairs(out) == []
        for b, c in zip(out, contents):
            r = box_to_rect(b)
            # floor rounding in rect_to_box may give up at most 1 px at the far edge
            assert r[0] <= c[0] and r[1] <= c[1] and r[2] >= c[2] - 1 and r[3] >= c[3] - 1


def test_clamp_to_image_trims_at_borders():
    box = clamp_to_image(_box(-20, -10, 80, 90), 60, 200)
    r = box_to_rect(box)
    assert r[0] >= 0 and r[1] >= 0 and r[2] <= 60
    inside = _box(10, 10, 50, 50)
    assert clamp_to_image(inside, 60, 200) is inside


def test_resize_stops_at_neighbour_edge():
    prev = _box(0, 0, 100, 100)
    neighbour = _box(120, 0, 220, 100)
    grown = _box(0, 0, 180, 100)
    r = box_to_rect(clamp_to_neighbours(prev, grown, [neighbour]))
    assert r[2] <= 120
    assert not rects_overlap(r, box_to_rect(neighbour))


def test_resize_ignores_neighbour_already_overlapping():
    prev = _box(0, 0, 100, 100)
    neighbour = _box(90, 0, 190, 100)
    grown = _box(0, 0, 150, 100)
    assert clamp_to_neighbours(prev, grown, [neighbour]) is grown


def test_spatial_order_is_row_major_with_jitter():
    rng = random.Random(1)
    boxes, expected = [], []
    for row, cy in enumerate((200, 600)):
        for col in range(4):
            boxes.append({"cx": 100 + 150 * col, "cy": cy + rng.randint(-40, 40),
                          "half_w": 50, "half_h": 120 + rng.randint(-30, 30)})
            expected.append(row * 4 + col)
    shuffled = list(range(8))
    rng.shuffle(shuffled)
    order = spatial_order([boxes[k] for k in shuffled])
    assert [shuffled[k] for k in order] == expected


def test_spatial_order_empty():
    assert spatial_order([]) == []


# --- Gui wiring --------------------------------------------------------------

class _FakeCanvas:
    def __init__(self):
        self.items = {}
        self._next = 1

    def _new(self, kind, **kw):
        item_id = self._next
        self._next += 1
        self.items[item_id] = {"kind": kind, "coords": [0, 0, 0, 0], **kw}
        return item_id

    def create_rectangle(self, *coords, **kw):
        return self._new("rect", **kw)

    def create_text(self, *coords, **kw):
        return self._new("text", **kw)

    def create_oval(self, *coords, **kw):
        return self._new("oval", **kw)

    def coords(self, item_id, *coords):
        self.items[item_id]["coords"] = list(coords)

    def itemconfigure(self, item_id, **kw):
        self.items[item_id].update(kw)

    def bbox(self, item_id):
        x, y = self.items[item_id]["coords"][:2]
        return (x, y, x + 10, y + 14)

    def tag_raise(self, *args):
        pass

    def delete(self, *item_ids):
        for item_id in item_ids:
            self.items.pop(item_id)


class _FakeLabel:
    def __init__(self):
        self.text = ""

    def configure(self, text):
        self.text = text


class _FakeButton(dict):
    pass


def _make_gui():
    from seedling_measurment import Gui

    gui = Gui.__new__(Gui)
    gui.canvas = _FakeCanvas()
    gui.progress_bar_label = _FakeLabel()
    gui.button_crop = _FakeButton()
    gui.x_length, gui.y_length = 1000, 1000
    gui.width1, gui.height1 = 1000, 1000
    gui.min_box_half_size = 30
    gui.crop_padding_width_fraction = 0.4
    gui.crop_padding_height_fraction = 0.10
    gui.seedling_pairs, gui.selected_points_debug = [], []
    gui.crop_boxes, gui._box_canvas_items, gui._point_canvas_items = [], [], []
    gui._crop_add_seq, gui._next_crop_seq = [], 0
    gui._overlapping_crops = set()
    gui._pending_start = None
    gui.button_circle_check = True
    return gui


def _place(gui, start, end):
    gui.seedling_pairs.append({"start": start, "end": end})
    gui.selected_points_debug.append(start)
    gui._point_canvas_items.append((gui.canvas.create_oval(), gui.canvas.create_oval()))
    gui._add_crop_box(start, end)


def _labels(gui):
    return [gui.canvas.items[item["label"]]["text"] for item in gui._box_canvas_items]


def test_gui_renumbers_spatially_and_removes_overlap():
    gui = _make_gui()
    # 40 px apart: each auto-derived box is floored to 60 px wide, so
    # neighbours overlap until resolve_overlaps trims them.
    _place(gui, (500, 800), (505, 600))   # middle
    _place(gui, (460, 800), (455, 600))   # left
    _place(gui, (540, 800), (542, 600))   # right

    assert [p["start"][0] for p in gui.seedling_pairs] == [460, 500, 540]
    assert [s[0] for s in gui.selected_points_debug] == [460, 500, 540]
    assert _labels(gui) == ["1", "2", "3"]
    assert find_overlapping_pairs(gui.crop_boxes) == []
    assert gui.crop_boxes[1]["half_w"] < gui.min_box_half_size   # middle box was trimmed
    assert gui.transformed_mid_points == [(b["cx"], b["cy"]) for b in gui.crop_boxes]


def test_gui_undo_removes_last_placed_not_last_listed():
    gui = _make_gui()
    _place(gui, (500, 800), (505, 600))
    _place(gui, (300, 800), (295, 600))   # placed last, but sorted to index 0
    n_items = len(gui.canvas.items)

    gui._undo_last_point()

    assert [p["start"][0] for p in gui.seedling_pairs] == [500]
    assert _labels(gui) == ["1"]
    assert len(gui.canvas.items) == n_items - 9   # 2 ovals + rect + 4 handles + label + label bg


def test_gui_flags_crossing_seedlings():
    gui = _make_gui()
    _place(gui, (500, 800), (560, 600))
    _place(gui, (560, 800), (500, 600))
    assert gui._overlapping_crops == {0, 1}
    rect = gui._box_canvas_items[0]["rect"]
    assert gui.canvas.items[rect]["outline"] == "#ff3b30"
    assert "overlap" in gui.progress_bar_label.text
