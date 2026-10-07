"""Angle readout from predicted landmark fields -- pure numpy.

Lives under utils/ (not multi/src/) because models/multiclass_inference.py
needs it at inference time and models/ never imports multi/; multi/src/
landmarks.py re-exports it for the training side.

The landmark head predicts a junction HEATMAP and two unit-vector FIELDS
(hypocotyl, cotyledon) pointing away from the junction. The junction is the
heatmap peak; each direction is the mean field within `radius` pixels of it --
where both rays start and the field is supervised (radius must stay <= the
ray width used for the targets). theta is the angle between the two
directions; the bio angle is 180 - theta. There is no overhook (180 + theta) reading: the
flag is a few degrees past 180 deg, below the repeatability of the clicks themselves.
"""
from __future__ import annotations

import math

import numpy as np

READOUT_RADIUS = 3.0
PEAK_FOUND = 0.1          # lowest heatmap peak accepted as a junction


def theta_between(u, v) -> float:
    """Angle in degrees (0..180) between two unit vectors."""
    return float(np.degrees(np.arccos(np.clip(u[0] * v[0] + u[1] * v[1], -1.0, 1.0))))


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


def readout_from_fields(hm, vec, radius=READOUT_RADIUS, peak_threshold=PEAK_FOUND,
                        collar_hm=None, root_vec=None):
    """Angle from predicted fields: hm (H, W) probabilities, vec (4, H, W).

    `collar_hm` (H, W) and `root_vec` (2, H, W), when given, add the plant collar (its
    heatmap peak) and the root direction (mean field within `radius` of it) as
    "collar", "collar_peak" and "root_dir" -- all None when no collar peak reaches
    `peak_threshold`. They never change the angle readout.

    Returns None when no junction peak is found or a direction is degenerate,
    else a dict with junction (x, y), theta, bio, the two unit
    directions and the heatmap peak."""
    if hm.size == 0 or float(hm.max()) < peak_threshold:
        return None
    jy, jx = np.unravel_index(int(np.argmax(hm)), hm.shape)
    yy, xx = np.mgrid[0:hm.shape[0], 0:hm.shape[1]]
    disc = np.hypot(xx - jx, yy - jy) <= radius
    mean = [float(vec[c][disc].mean()) for c in range(4)]
    dirs = []
    for ax, ay in ((mean[0], mean[1]), (mean[2], mean[3])):
        n = math.hypot(ax, ay)
        if n < 1e-6:
            return None
        dirs.append((ax / n, ay / n))
    theta = theta_between(*dirs)
    extra = {}
    if collar_hm is not None and root_vec is not None:
        extra = _collar_readout(collar_hm, root_vec, radius, peak_threshold)
    return {**extra, "junction": (jx + 0.5, jy + 0.5), "theta": theta,
            "bio": bio_from_theta(theta), "hypo_dir": dirs[0], "cotyl_dir": dirs[1],
            "peak": float(hm.max())}
