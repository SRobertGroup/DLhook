"""Tests for utils/crop_layout.py (spatially numbered crop boxes, overlap detection, clamping).

The main window's box handling (canvas, spatial renumbering, seed point) is tested in
tests/test_main_window_boxes.py. resolve_overlaps / clamp_to_neighbours are no longer called by the
GUI (overlaps are highlighted, not trimmed) but stay here as tested utilities.
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
