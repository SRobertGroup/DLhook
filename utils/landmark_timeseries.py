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

import numpy as np

from utils.angle_timeseries import _hampel_filter, _smooth
from utils.landmark_readout import bio_from_theta

MIN_PEAK = 0.2                 # frames with a weaker junction peak are treated as missing
HAMPEL_WINDOW = 0              # off: measured to hurt
SMOOTH_WINDOW = 1              # off
MAX_GAP_FRAMES = 3             # missing runs up to this long are interpolated


def _valid(readout, min_peak):
    return readout is not None and not readout.get("gated") and readout.get("theta") is not None and readout.get("peak", 1.0) >= min_peak


def _interpolate_gaps(values, max_gap, blocked=None):
    """Linear interpolation across runs of NaN no longer than max_gap that have a
    value on both sides; everything else stays NaN. A run containing a `blocked` frame (the
    readability gate said it carries no information) is never filled."""
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
        if i > 0 and j < n and (j - i) <= max_gap and not (blocked and any(blocked[i:j])):
            lo, hi = out[i - 1], out[j]
            for k in range(i, j):
                out[k] = lo + (hi - lo) * (k - i + 1) / (j - i + 1)
        i = j
    return out


def monotone_opening(bio, step=2.5, lo=0.0, hi=180.0):
    """Best non-increasing fit to a bio-angle series, in the L1 sense (robust to a lone wrong frame).

    After germination a hook is maintained and then opens; it does not close again (docs/AUDIT.md
    section 8: the errors are lone frames read far from their neighbours). The fit is the dynamic
    programme over angle levels every `step` degrees: level_t <= level_{t-1}, cost sum |obs_t - level_t|,
    NaN frames cost nothing and take whatever level the path has there. An input with no
    value at all comes back unchanged. Unlike a median filter this constrains the order of the
    states rather than averaging, so a long closed run stays closed and a long open run stays open."""
    obs = np.asarray(bio, dtype=float)
    if obs.size == 0 or np.isnan(obs).all():
        return [float(v) for v in obs]
    levels = np.arange(lo, hi + step / 2, step)
    n, k = len(obs), len(levels)
    cost = np.where(np.isnan(obs)[:, None], 0.0, np.abs(np.nan_to_num(obs)[:, None] - levels[None, :]))
    best = np.zeros((n, k))
    back = np.zeros((n, k), dtype=int)
    best[0] = cost[0]
    for t in range(1, n):
        # level l at t may follow any level l' >= l at t - 1: a running minimum from the top
        prev = best[t - 1]
        run_min = np.minimum.accumulate(prev[::-1])[::-1]
        arg = np.zeros(k, dtype=int)
        cur = k - 1
        for l in range(k - 1, -1, -1):
            if prev[l] <= prev[cur]:
                cur = l
            arg[l] = cur
        best[t] = cost[t] + run_min
        back[t] = arg
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmin(best[-1]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t][path[t]]
    return [float(levels[i]) for i in path]


def reconstruct_landmark_series(readouts, min_peak=MIN_PEAK, hampel_window=HAMPEL_WINDOW,
                                smooth_window=SMOOTH_WINDOW, max_gap=MAX_GAP_FRAMES):
    """readouts: ordered per-frame landmark readout dicts (utils.landmark_readout) or None.
    A readout dict with "gated": True (utils/landmark_gate) is missing and also blocks interpolation
    across its gap. Returns the bio angles (180 - theta) of the series, same length, NaN where a
    frame stays missing."""
    n = len(readouts)
    theta = [float(r["theta"]) if _valid(r, min_peak) else None for r in readouts]
    no_protect = [False] * n
    theta = _hampel_filter(theta, no_protect, window=hampel_window)
    theta = _smooth(theta, no_protect, window=smooth_window)
    bio = [float("nan") if v is None or math.isnan(v) else bio_from_theta(v) for v in theta]
    blocked = [bool(r and r.get("gated")) for r in readouts]
    return _interpolate_gaps(bio, max_gap, blocked)
