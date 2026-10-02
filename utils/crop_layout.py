"""Layout rules for the per-seedling crop boxes drawn on the main window.

Pure geometry, no Tk: boxes are the GUI's own dicts
``{"cx", "cy", "half_w", "half_h"}`` in full-resolution image pixels, and the
crop actually cut is the half-open pixel range ``[cx - half_w, cx + half_w)``
(see Gui.start_analysis). Rects here are ``(x1, y1, x2, y2)`` in that same
half-open convention, so two rects that merely touch do not overlap.

Three rules are enforced:

* boxes never extend past the image (clamp_to_image) -- the cut is clamped
  anyway, this just makes the drawn box match it;
* boxes never overlap a neighbour: a newly placed seedling is split from its
  neighbours at the midline of the gap between their clicked points
  (resolve_overlaps), and a corner drag stops at a neighbour's edge
  (clamp_to_neighbours);
* crop IDs follow the plate, top-to-bottom then left-to-right
  (spatial_order), not the order the user happened to click in.
"""

def box_to_rect(box):
    return (box["cx"] - box["half_w"], box["cy"] - box["half_h"],
            box["cx"] + box["half_w"], box["cy"] + box["half_h"])


def rect_to_box(x1, y1, x2, y2):
    """Largest symmetric box contained in the rect. Flooring (rather than
    rounding) keeps the box inside the rect, so converting a non-overlapping
    rect back to a box can never re-introduce a 1-px overlap."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    return {
        "cx": (x1 + x2) // 2,
        "cy": (y1 + y2) // 2,
        "half_w": (x2 - x1) // 2,
        "half_h": (y2 - y1) // 2,
    }


def rects_overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def find_overlapping_pairs(boxes):
    """Index pairs (i, j), i < j, of boxes that still overlap."""
    rects = [box_to_rect(b) for b in boxes]
    return [(i, j)
            for i in range(len(rects))
            for j in range(i + 1, len(rects))
            if rects_overlap(rects[i], rects[j])]


def content_rect(start, end):
    """Bounding rect of a seedling's two clicked points -- the part of its box
    that overlap resolution must never trim away."""
    return (min(start[0], end[0]), min(start[1], end[1]),
            max(start[0], end[0]), max(start[1], end[1]))


def clamp_to_image(box, width, height):
    x1, y1, x2, y2 = box_to_rect(box)
    clamped = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
    if clamped == (x1, y1, x2, y2):
        return box
    return rect_to_box(*clamped)


def _gap(a_lo, a_hi, b_lo, b_hi):
    """Signed gap between two 1-D intervals (negative = they overlap), and
    whether `a` is the lower one."""
    if a_hi <= b_lo:
        return b_lo - a_hi, True
    if b_hi <= a_lo:
        return a_lo - b_hi, False
    return -(min(a_hi, b_hi) - max(a_lo, b_lo)), a_lo <= b_lo


def resolve_overlaps(boxes, contents):
    """Trim overlapping boxes apart; returns a new list of boxes.

    For each overlapping pair, the axis along which the two seedlings' clicked
    points are furthest apart is picked (x for seedlings side by side), and
    both facing edges are pulled back to the midline of that gap. Neither box
    is trimmed inside its own content rect. Trimming only ever shrinks a box,
    so one pass over all pairs is enough: a later trim cannot re-open an
    earlier pair. Pairs whose clicked points themselves overlap on both axes
    cannot be split and are left as they are -- find them afterwards with
    find_overlapping_pairs. Boxes that needed no trim are returned unchanged.
    """
    rects = [list(box_to_rect(b)) for b in boxes]
    changed = [False] * len(boxes)

    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if not rects_overlap(rects[i], rects[j]):
                continue
            ci, cj = contents[i], contents[j]
            gap_x, i_left = _gap(ci[0], ci[2], cj[0], cj[2])
            gap_y, i_above = _gap(ci[1], ci[3], cj[1], cj[3])
            if max(gap_x, gap_y) < 0:
                continue  # clicked points cross: unresolvable, flagged by caller

            axis, i_first = (0, i_left) if gap_x >= gap_y else (1, i_above)
            lo, hi = (i, j) if i_first else (j, i)
            # Midline of the gap between lo's content far edge and hi's near edge
            mid = (contents[lo][axis + 2] + contents[hi][axis]) // 2
            if rects[lo][axis + 2] > mid:
                rects[lo][axis + 2] = mid
                changed[lo] = True
            if rects[hi][axis] < mid:
                rects[hi][axis] = mid
                changed[hi] = True

    return [rect_to_box(*r) if c else b for b, r, c in zip(boxes, rects, changed)]


def clamp_to_neighbours(prev_box, new_box, others):
    """Shrink `new_box` (a resize of `prev_box`) so it overlaps none of
    `others`. For each neighbour it now hits, the edge facing that neighbour
    -- the side on which `prev_box` was clear of it -- stops at the
    neighbour's edge. Neighbours `prev_box` already overlapped (an
    unresolvable pair) are ignored. Falls back to `prev_box` if the clamp
    would collapse the box."""
    p = box_to_rect(prev_box)
    n = list(box_to_rect(new_box))
    for other in others:
        o = box_to_rect(other)
        if not rects_overlap(n, o) or rects_overlap(p, o):
            continue
        if p[2] <= o[0]:
            n[2] = min(n[2], o[0])
        elif p[0] >= o[2]:
            n[0] = max(n[0], o[2])
        elif p[3] <= o[1]:
            n[3] = min(n[3], o[1])
        else:
            n[1] = max(n[1], o[3])
    if n[2] - n[0] < 2 or n[3] - n[1] < 2:
        return prev_box
    if tuple(n) == box_to_rect(new_box):
        return new_box
    return rect_to_box(*n)


def spatial_order(boxes):
    """Permutation of box indices in reading order: rows top-to-bottom, each
    row left-to-right. A box joins the current row when its vertical extent
    overlaps the row's first box by at least half the shorter of the two --
    robust to seedlings of different lengths in one row, which share the sown
    seed line but not their box centres."""
    if not boxes:
        return []
    by_cy = sorted(range(len(boxes)), key=lambda k: (boxes[k]["cy"], k))
    rows = []
    for k in by_cy:
        b = boxes[k]
        if rows:
            a = boxes[rows[-1][0]]
            overlap = (min(a["cy"] + a["half_h"], b["cy"] + b["half_h"])
                       - max(a["cy"] - a["half_h"], b["cy"] - b["half_h"]))
            if overlap >= 0.5 * min(2 * a["half_h"], 2 * b["half_h"]):
                rows[-1].append(k)
                continue
        rows.append([k])
    return [k for row in rows for k in sorted(row, key=lambda k: (boxes[k]["cx"], k))]
