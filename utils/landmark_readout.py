"""Angle readout from predicted landmark fields -- pure numpy.

Lives under utils/ (not multi/src/) because models/multiclass_inference.py
needs it at inference time and models/ never imports multi/; multi/src/
landmarks.py re-exports it for the training side.

The landmark head predicts a junction HEATMAP and two unit-vector FIELDS
(hypocotyl, cotyledon) pointing away from the junction. The junction is the
heatmap peak; each direction is the mean field within `radius` pixels of it --
where both rays start and the field is supervised (radius must stay <= the
ray width used for the targets). theta is the angle between the two
directions; the bio angle is 180 - theta. `bio` never exceeds 180. An overhook (the cotyledon has swung past the
stem) is found by a geometric test (rays_cross): the cotyledon ray from its base and the hypocotyl ray
from the junction meet. It is reported as the `overhook` flag and as `bio_overhook`, the signed reading
(180 + theta when flagged, else 180 - theta). The excess is only a few degrees, near the repeatability of
the clicks, so `bio` stays the folded value the rest of the pipeline uses.
"""
from __future__ import annotations

import math

import numpy as np

READOUT_RADIUS = 3.0
PEAK_FOUND = 0.1          # lowest heatmap peak accepted as a junction
MIN_BASE_TIP = 5.0        # px: base and tip closer than this give no usable direction
MIN_JUNCTION_TIP = 3.0    # px: a tip this close to the junction gives no usable direction
OVERHOOK_MAX_THETA = 45.0  # deg: a crossing at a wider angle is not an overhook (every clicked one is below this)
TIP_RANGE = (3.0, 40.0)   # px: junction -> tip distances that count as plausible when choosing a junction


def theta_between(u, v) -> float:
    """Angle in degrees (0..180) between two unit vectors."""
    return float(np.degrees(np.arccos(np.clip(u[0] * v[0] + u[1] * v[1], -1.0, 1.0))))


def rays_cross(origin_a, dir_a, origin_b, dir_b) -> bool:
    """True when the ray origin_a + t*dir_a (t > 0) meets the ray origin_b + s*dir_b (s > 0).

    Overhook test: a = the cotyledon from its base, b = the hypocotyl from the junction. Closed or
    open hooks give rays that run parallel or apart; a cotyledon pointing back across the stem gives
    rays that converge and meet."""
    a = np.array([dir_a[0], dir_a[1]], dtype=float)
    b = np.array([dir_b[0], dir_b[1]], dtype=float)
    m = np.array([a, -b]).T
    if abs(np.linalg.det(m)) < 1e-9:
        return False
    t, s = np.linalg.solve(m, np.array(origin_b, dtype=float) - np.array(origin_a, dtype=float))
    return bool(t > 0 and s > 0)


def bio_from_theta(theta: float) -> float:
    """Bio convention: 180 = closed, decreasing as the hook opens."""
    return 180.0 - theta


def _collar_readout(collar_hm, root_vec, radius, peak_threshold):
    none = {"collar": None, "collar_peak": float(collar_hm.max()) if collar_hm.size else 0.0, "root_dir": None}
    if collar_hm.size == 0 or float(collar_hm.max()) < peak_threshold:
        return none
    cy, cx = np.unravel_index(int(np.argmax(collar_hm)), collar_hm.shape)
    yy, xx = np.mgrid[0:collar_hm.shape[0], 0:collar_hm.shape[1]]
    disc = np.hypot(xx - cx, yy - cy) <= radius
    ax, ay = float(root_vec[0][disc].mean()), float(root_vec[1][disc].mean())
    norm = math.hypot(ax, ay)
    return {**none, "collar": (cx + 0.5, cy + 0.5), "root_dir": (ax / norm, ay / norm) if norm > 1e-6 else None}


def _peak(hm, threshold):
    """(x, y, value) of the heatmap maximum, pixel-centre coordinates, or None below `threshold`."""
    if hm is None or hm.size == 0 or float(hm.max()) < threshold:
        return None
    y, x = np.unravel_index(int(np.argmax(hm)), hm.shape)
    return x + 0.5, y + 0.5, float(hm.max())


def _local_maxima(hm, k=4, rel=0.3, nms=6):
    """Up to k heatmap peaks (y, x, value) at least rel * max high, `nms` px apart."""
    work = hm.copy()
    top = float(hm.max())
    out = []
    for _ in range(k):
        y, x = np.unravel_index(int(np.argmax(work)), work.shape)
        if work[y, x] < rel * top or work[y, x] <= 0:
            break
        out.append((y, x, float(hm[y, x])))
        work[max(0, y - nms):y + nms + 1, max(0, x - nms):x + nms + 1] = 0
    return out


def _unit(dx, dy, min_length):
    n = math.hypot(dx, dy)
    return None if n < min_length else (dx / n, dy / n)


def readout_from_fields(hm, vec, radius=READOUT_RADIUS, peak_threshold=PEAK_FOUND,
                        collar_hm=None, root_vec=None, base_hm=None, tip_hm=None, measurable=None):
    """Angle from predicted fields: hm (H, W) probabilities, vec (4, H, W).

    `collar_hm` (H, W) and `root_vec` (2, H, W), when given, add the plant collar (its
    heatmap peak) and the root direction (mean field within `radius` of it) as
    "collar", "collar_peak" and "root_dir" -- all None when no collar peak reaches
    `peak_threshold`. They never change the angle readout.

    `base_hm` and `tip_hm` (H, W), from a head with the cotyledon base and tip, change it. The
    cotyledon direction is read, in this order, as base -> tip (peaks more than MIN_BASE_TIP px
    apart), junction -> tip (more than MIN_JUNCTION_TIP px apart), or from the direction field
    at the junction as before; the result says which in "cotyl_path". A tip also arbitrates
    between junction peaks: of the heatmap's local maxima the one whose distance to the tip is
    plausible (TIP_RANGE) wins, which stops an open hook from being read at a point that slid
    down the hypocotyl.

    `measurable`, from a head with the "measurable" logit, is the crop's probability that an
    annotator could measure the hook; it is passed through as "measurable_p" and never changes the
    readout.

    Returns None when no junction peak is found or a direction is degenerate,
    else a dict with junction (x, y), theta, bio, the two unit
    directions and the heatmap peak."""
    if hm.size == 0 or float(hm.max()) < peak_threshold:
        return None
    jy, jx = np.unravel_index(int(np.argmax(hm)), hm.shape)
    tip = _peak(tip_hm, peak_threshold)
    base = _peak(base_hm, peak_threshold) if tip is not None else None
    if tip is not None:
        best = None
        for y, x, value in _local_maxima(hm):
            d = math.hypot(tip[0] - (x + 0.5), tip[1] - (y + 0.5))
            score = value * (1.0 if TIP_RANGE[0] <= d <= TIP_RANGE[1] else 0.3)
            if best is None or score > best[0]:
                best = (score, y, x)
        jy, jx = best[1], best[2]
    yy, xx = np.mgrid[0:hm.shape[0], 0:hm.shape[1]]
    disc = np.hypot(xx - jx, yy - jy) <= radius
    mean = [float(vec[c][disc].mean()) for c in range(4)]
    dirs = []
    for ax, ay in ((mean[0], mean[1]), (mean[2], mean[3])):
        n = math.hypot(ax, ay)
        if n < 1e-6:
            return None
        dirs.append((ax / n, ay / n))
    path = "field"
    if tip is not None:
        along = None
        if base is not None:
            along = _unit(tip[0] - base[0], tip[1] - base[1], MIN_BASE_TIP)
            path = "base_tip"
        if along is None:
            along = _unit(tip[0] - (jx + 0.5), tip[1] - (jy + 0.5), MIN_JUNCTION_TIP)
            path = "junction_tip"
        if along is not None:
            dirs[1] = along
        else:
            path = "field"
    theta = theta_between(*dirs)
    overhook = bool(base is not None and path == "base_tip" and theta < OVERHOOK_MAX_THETA
                    and rays_cross(base[:2], dirs[1], (jx + 0.5, jy + 0.5), dirs[0]))
    extra = {}
    if collar_hm is not None and root_vec is not None:
        extra = _collar_readout(collar_hm, root_vec, radius, peak_threshold)
    if measurable is not None:
        extra["measurable_p"] = float(measurable)
    return {**extra, "junction": (jx + 0.5, jy + 0.5), "theta": theta,
            "bio": bio_from_theta(theta), "bio_overhook": 180.0 + theta if overhook else bio_from_theta(theta),
            "hypo_dir": dirs[0], "cotyl_dir": dirs[1],
            "peak": float(hm.max()), "cotyl_path": path, "overhook": overhook,
            "tip": None if tip is None else tip[:2], "base": None if base is None else base[:2]}
