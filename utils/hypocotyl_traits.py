"""Hypocotyl length, curvature and diameter from a binary hypocotyl mask -- pure numpy/OpenCV/skimage.

All distances are in crop pixels; `to_mm` converts with a scale. The hypocotyl is described by its
midline: the longest skeleton path of the mask's largest component, oriented from the junction
end, optionally cut at the collar, resampled every 1 px and smoothed. Length is the arc length,
curvature the signed curvature of the smoothed midline and the diameter the median transect width
perpendicular to the midline.
"""
from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

SMOOTH_WINDOW = 15            # moving-average window on the 1 px resampled midline
CURVATURE_TRIM_PX = 5         # first/last px of the midline ignored for curvature
STATION_SPACING_PX = 5.0      # diameter transects every this many px along the midline
STATION_INNER_FRACTION = 0.8  # ... over the inner 80% of it
MARCH_STEP_PX = 0.25
MAX_HALF_WIDTH_PX = 60.0      # a transect running further than this on one side is dropped
MIN_PATH_PX = 12              # shorter midlines give no traits
BORDER_PX = 2                 # slack for "touches the crop border"

_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _binary(mask):
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 127).astype(np.uint8)


def _largest_component(binary):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == k).astype(np.uint8)


def _skeleton_longest_path(component):
    """Longest geodesic path (list of (x, y), 8-connected skeleton pixels), or None."""
    from skimage.morphology import skeletonize
    skel = skeletonize(component.astype(bool))
    ys, xs = np.nonzero(skel)
    if len(xs) < 2:
        return None
    pix = set(zip(ys.tolist(), xs.tolist()))

    def bfs(start):
        parent = {start: None}
        order = deque([start])
        last = start
        while order:
            cur = order.popleft()
            last = cur
            for dy, dx in _NEIGHBOURS:
                nb = (cur[0] + dy, cur[1] + dx)
                if nb in pix and nb not in parent:
                    parent[nb] = cur
                    order.append(nb)
        return last, parent

    far1, _ = bfs(next(iter(pix)))
    far2, parent = bfs(far1)
    path = []
    cur = far2
    while cur is not None:
        path.append((float(cur[1]), float(cur[0])))
        cur = parent[cur]
    return path if len(path) >= 2 else None


def _resample_smooth(path, window=SMOOTH_WINDOW):
    """Resample a polyline every 1 px of arc length and smooth it with a moving average whose ends
    are padded by odd reflection (a straight line stays exactly straight up to its end points)."""
    pts = np.asarray(path, dtype=float)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 2:
        return None
    grid = np.arange(0.0, s[-1], 1.0)
    grid = np.append(grid, s[-1]) if s[-1] - grid[-1] > 0.5 else grid
    xy = np.stack([np.interp(grid, s, pts[:, 0]), np.interp(grid, s, pts[:, 1])], axis=1)
    half = min(window // 2, (len(xy) - 1) // 2)
    if half < 1:
        return xy
    k = 2 * half + 1
    pad_lo = 2 * xy[0] - xy[half:0:-1]
    pad_hi = 2 * xy[-1] - xy[-2:-half - 2:-1]
    padded = np.vstack([pad_lo, xy, pad_hi])
    kernel = np.ones(k) / k
    return np.stack([np.convolve(padded[:, c], kernel, mode="valid") for c in (0, 1)], axis=1)


def _oriented_path(mask, junction):
    """(raw path oriented junction-end first, component, binary) or None."""
    binary = _binary(mask)
    if binary is None or not binary.any():
        return None
    comp = _largest_component(binary)
    if comp is None:
        return None
    path = _skeleton_longest_path(comp)
    if path is None:
        return None
    if junction is not None:
        jx, jy = float(junction[0]), float(junction[1])
        d_first = math.hypot(path[0][0] - jx, path[0][1] - jy)
        d_last = math.hypot(path[-1][0] - jx, path[-1][1] - jy)
        if d_last < d_first:
            path = path[::-1]
    return path, comp


def _extend_end(pts, comp, at_start, look=8, max_ext=40):
    """Skeletons stop about half a width short of the mask's ends; push one end of `pts` outward
    along its own direction (estimated over `look` px) until it leaves the mask."""
    seg = pts[:look + 1] if at_start else pts[-look - 1:]
    d = (seg[0] - seg[-1]) if at_start else (seg[-1] - seg[0])
    n = float(np.hypot(*d))
    if n < 1e-9:
        return pts
    d = d / n
    tip = pts[0] if at_start else pts[-1]
    h, w = comp.shape
    last = tip
    for step in range(1, max_ext + 1):
        p = tip + d * step
        xi, yi = int(round(p[0])), int(round(p[1]))
        if not (0 <= xi < w and 0 <= yi < h) or not comp[yi, xi]:
            break
        last = p
    if last is tip:
        return pts
    return np.vstack([last[None], pts]) if at_start else np.vstack([pts, last[None]])


def _cut_at_collar(path, collar):
    pts = np.asarray(path, dtype=float)
    d = np.hypot(pts[:, 0] - float(collar[0]), pts[:, 1] - float(collar[1]))
    return pts[: int(np.argmin(d)) + 1]


def hypocotyl_midline(mask, junction, collar=None):
    """Smoothed hypocotyl midline as an (N, 2) float array of (x, y), junction end first, one point
    per px of arc length; None when the mask is empty, there is no junction or the path is too short."""
    got = _midline_and_flags(mask, junction, collar)
    return None if got is None else got[0]


def _midline_and_flags(mask, junction, collar=None):
    if junction is None:
        return None
    oriented = _oriented_path(mask, junction)
    if oriented is None:
        return None
    path, comp = oriented
    truncated = False
    path = _extend_end(np.asarray(path, dtype=float), comp, at_start=True)
    if collar is not None:
        path = _cut_at_collar(path, collar)
        if len(path) < 2:
            return None
    else:
        path = _extend_end(path, comp, at_start=False)
        h, w = comp.shape
        x, y = path[-1]
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
        r = float(dist[int(round(y)), int(round(x))])
        edge = min(x, y, w - 1 - x, h - 1 - y)
        truncated = edge <= r + BORDER_PX
    mid = _resample_smooth(path)
    if mid is None or len(mid) < MIN_PATH_PX:
        return None
    return mid, truncated


def _arc_length(mid):
    return float(np.hypot(*np.diff(mid, axis=0).T).sum())


def _curvature_stats(mid, trim=CURVATURE_TRIM_PX):
    if len(mid) < 2 * trim + 5:
        return float("nan"), float("nan")
    x, y = mid[:, 0], mid[:, 1]
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    k = (dx * ddy - dy * ddx) / np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-12)
    k = np.abs(k[trim:len(k) - trim])
    return float(np.mean(k)), float(np.max(k))


def _transect_widths(binary, mid):
    """Widths (px) of the transects perpendicular to the midline at stations every 5 px."""
    n = len(mid)
    lo, hi = int(round(n * (1 - STATION_INNER_FRACTION) / 2)), int(round(n * (1 + STATION_INNER_FRACTION) / 2))
    idx = [i for i in range(max(lo, 1), min(hi, n - 1), int(STATION_SPACING_PX))]
    if not idx:
        return []
    h, w = binary.shape
    t = np.arange(0.0, MAX_HALF_WIDTH_PX + MARCH_STEP_PX, MARCH_STEP_PX)
    field = binary.astype(np.float32)
    widths = []
    for i in idx:
        tan = mid[i + 1] - mid[i - 1]
        norm = math.hypot(*tan)
        if norm < 1e-9:
            continue
        nx, ny = -tan[1] / norm, tan[0] / norm
        half = []
        for sign in (1.0, -1.0):
            xs = mid[i, 0] + sign * nx * t
            ys = mid[i, 1] + sign * ny * t
            inside = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
            vals = cv2.remap(field, xs.astype(np.float32).reshape(1, -1), ys.astype(np.float32).reshape(1, -1),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[0]
            below = np.nonzero(vals < 0.5)[0]
            if len(below) == 0 or vals[0] < 0.5:
                half = None            # cap exceeded, or the midline left the mask here
                break
            first = int(below[0])
            if not inside[: first + 1].all():
                half = None            # the transect ran into the image border
                break
            half.append(float(t[first]))
        if half:
            widths.append(half[0] + half[1])
    return widths


_NAN_TRAITS = {"length_px": float("nan"), "length_truncated": False, "curvature_mean": float("nan"),
               "curvature_max": float("nan"), "diameter_px": float("nan"), "n_transects": 0}


def hypocotyl_traits(mask, junction, collar=None):
    """dict(length_px, length_truncated, curvature_mean, curvature_max, diameter_px, n_transects).

    length is the midline arc length from the junction (to the collar when given, else to the end of
    the mask; `length_truncated` then flags a mask end at the crop border). Curvature is mean and max
    |k| in 1/px. All NaN/False when the mask is empty, there is no junction or the midline is too short."""
    out = dict(_NAN_TRAITS)
    got = _midline_and_flags(mask, junction, collar)
    if got is None:
        return out
    mid, truncated = got
    out["length_px"] = _arc_length(mid)
    out["length_truncated"] = bool(truncated)
    out["curvature_mean"], out["curvature_max"] = _curvature_stats(mid)
    widths = _transect_widths(_binary(mask), mid)
    out["n_transects"] = len(widths)
    if widths:
        out["diameter_px"] = float(np.median(widths))
    return out


def scale_from_line(p1, p2, length_mm):
    """um per pixel from a line drawn between two image points over a known length in mm.
    ValueError for a zero-length line or a non-positive length."""
    px = math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1]))
    if px < 1e-6:
        raise ValueError("the two calibration points coincide")
    if not _num(length_mm) or length_mm <= 0:
        raise ValueError("the known length must be a positive number of mm")
    return float(length_mm) * 1000.0 / px


def _num(v):
    return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))


def to_mm(traits, um_per_px):
    """Physical units of a traits dict: length_mm, diameter_mm and curvature per mm. NaN for a
    missing value; {} when no scale is given."""
    if not _num(um_per_px) or um_per_px <= 0 or traits is None:
        return {}
    mm_per_px = float(um_per_px) / 1000.0
    nan = float("nan")

    def lin(key):
        v = traits.get(key)
        return v * mm_per_px if _num(v) else nan

    def inv(key):
        v = traits.get(key)
        return v / mm_per_px if _num(v) else nan

    return {"length_mm": lin("length_px"), "diameter_mm": lin("diameter_px"),
            "curvature_mean_per_mm": inv("curvature_mean"), "curvature_max_per_mm": inv("curvature_max")}
