"""Temporal clean-up for landmark-head angle readings -- pure numpy.

utils/angle_timeseries.py is built for ellipse fits, whose per-frame angle aliases to
180 - a / a + 180 / 360 - a in sustained runs, so it re-branches the whole series. A landmark
reading has no such ambiguity: theta (the angle between the two predicted axis directions,
0..180) is well defined and the bio angle is simply 180 - theta (there is no overhook reading: the
flag is a few degrees past 180 deg, below the repeatability of the hand clicks). Feeding those
values through the ellipse DP only hurts: it flips readings that were already right (MAE 88.9 deg
instead of 22.4 deg on the hand-measured test frames).

So this module only does what a landmark series needs: a confidence gate (frames whose junction
peak is weak count as missing), optional Hampel filtering and smoothing of theta, and linear
interpolation across short gaps. Hampel and smoothing are OFF by default: on the same frames they
made the error worse (MAE 22.4 -> 24.9 deg).
"""
from __future__ import annotations

import math

from utils.angle_timeseries import _hampel_filter, _smooth
from utils.landmark_readout import bio_from_theta

MIN_PEAK = 0.2                 # frames with a weaker junction peak are treated as missing
HAMPEL_WINDOW = 0              # off: measured to hurt
SMOOTH_WINDOW = 1              # off
MAX_GAP_FRAMES = 3             # missing runs up to this long are interpolated


def _valid(readout, min_peak):
    return readout is not None and readout.get("theta") is not None and readout.get("peak", 1.0) >= min_peak


def _interpolate_gaps(values, max_gap):
    """Linear interpolation across runs of NaN no longer than max_gap that have a
    value on both sides; everything else stays NaN."""
    out = list(values)
    n = len(out)
    i = 0
    while i < n:
        if not math.isnan(out[i]):
            i += 1
            continue
        j = i
        while j < n and math.isnan(out[j]):
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap:
            lo, hi = out[i - 1], out[j]
            for k in range(i, j):
                out[k] = lo + (hi - lo) * (k - i + 1) / (j - i + 1)
        i = j
    return out


def reconstruct_landmark_series(readouts, min_peak=MIN_PEAK, hampel_window=HAMPEL_WINDOW,
                                smooth_window=SMOOTH_WINDOW, max_gap=MAX_GAP_FRAMES):
    """readouts: ordered per-frame landmark readout dicts (utils.landmark_readout) or None.
    Returns the bio angles (180 - theta) of the series, same length, NaN where a frame stays missing."""
    n = len(readouts)
    theta = [float(r["theta"]) if _valid(r, min_peak) else None for r in readouts]
    no_protect = [False] * n
    theta = _hampel_filter(theta, no_protect, window=hampel_window)
    theta = _smooth(theta, no_protect, window=smooth_window)
    bio = [float("nan") if v is None or math.isnan(v) else bio_from_theta(v) for v in theta]
    return _interpolate_gaps(bio, max_gap)
