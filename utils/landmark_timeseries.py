"""Temporal reconstruction for landmark-head angle readings -- pure numpy.

utils/angle_timeseries.py is built for ellipse fits, whose per-frame angle
aliases to 180 - a / a + 180 / 360 - a in sustained runs, so it re-branches the
whole series. A landmark reading has no such ambiguity: theta (the angle between
the two predicted axis directions, 0..180) is well defined, and the only open
question per frame is the SIGN -- bio = 180 - theta (hooked/opening) or
180 + theta (overhooked). Feeding bio values through the ellipse DP therefore
only hurts: it flips readings that were already right.

This module cleans theta on its own (confidence gate, Hampel, smoothing), then
chooses the overhook state per frame with a two-state Viterbi:

  cost = sum_t |bio_t(s_t) - bio_{t-1}(s_{t-1})|        continuity of the angle
       + switch_penalty * [s_t != s_{t-1}]              overhook is a run, not a flicker
       + overhook_weight * -log p_t(s_t)                the (weak) per-frame head

Continuity does NOT identify the sign by itself: theta 20, 10, 0, 10, 20 is
equally continuous as 160, 170, 180, 170, 160 (a bounce) and as 160, 170, 180,
190, 200 (a crossing into overhook). It only forbids flicker (a switch at theta
costs 2*theta). The sign therefore comes from the per-frame head and the prior;
measured on the 134 hand-measured frames, with continuity alone the states flip
at random (MAE 62 deg), and the head's probability is weak (AUC 0.66, values all
below 0.5), so this layer cannot yet recover overhook.

Hampel filtering and smoothing of theta are available but OFF by default: on the
same frames they made the error worse (MAE 22.4 -> 24.9 deg).
"""
from __future__ import annotations

import math

import numpy as np

from utils.angle_timeseries import _hampel_filter, _smooth

MIN_PEAK = 0.2                 # frames with a weaker junction peak are treated as missing
HAMPEL_WINDOW = 0               # off: measured to hurt (see module docstring)
SMOOTH_WINDOW = 1               # off
SWITCH_PENALTY_DEG = 20.0
OVERHOOK_WEIGHT_DEG = 5.0      # degrees of cost per nat of the head's log-probability
OVERHOOK_PRIOR_DEG = 2.0       # extra per-frame cost of the overhook state (the minority, ~25%)
MAX_GAP_FRAMES = 3             # missing runs up to this long are interpolated
_P_CLIP = 0.02


def _valid(readout, min_peak):
    return readout is not None and readout.get("theta") is not None and readout.get("peak", 1.0) >= min_peak


def choose_overhook_states(theta, probs=None, switch_penalty=SWITCH_PENALTY_DEG,
                           overhook_weight=OVERHOOK_WEIGHT_DEG, overhook_prior=OVERHOOK_PRIOR_DEG):
    """Viterbi over theta (degrees, no gaps): a list of bools, True = overhooked."""
    n = len(theta)
    if n == 0:
        return []
    if probs is None:
        probs = [None] * n

    def unary(t, s):
        p = probs[t]
        cost = overhook_prior if s else 0.0
        if p is not None and overhook_weight:
            p = min(max(float(p), _P_CLIP), 1.0 - _P_CLIP)
            cost += overhook_weight * -math.log(p if s else 1.0 - p)
        return cost

    def bio(t, s):
        return 180.0 + theta[t] if s else 180.0 - theta[t]

    cost = [[unary(0, 0), unary(0, 1)]]
    back = [[0, 0]]
    for t in range(1, n):
        row, brow = [], []
        for s in (0, 1):
            options = [cost[-1][q] + abs(bio(t, s) - bio(t - 1, q)) + (switch_penalty if q != s else 0.0)
                       for q in (0, 1)]
            q = int(np.argmin(options))
            row.append(options[q] + unary(t, s))
            brow.append(q)
        cost.append(row)
        back.append(brow)
    s = int(np.argmin(cost[-1]))
    path = [s]
    for t in range(n - 1, 0, -1):
        s = back[t][s]
        path.append(s)
    return [bool(v) for v in reversed(path)]


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
                                smooth_window=SMOOTH_WINDOW, switch_penalty=SWITCH_PENALTY_DEG,
                                overhook_weight=OVERHOOK_WEIGHT_DEG, overhook_prior=OVERHOOK_PRIOR_DEG,
                                max_gap=MAX_GAP_FRAMES):
    """readouts: ordered per-frame landmark readout dicts (utils.landmark_readout)
    or None. Returns (bio_angles, overhook_flags), same length: NaN / False for
    frames that stay missing."""
    n = len(readouts)
    theta = [float(r["theta"]) if _valid(r, min_peak) else None for r in readouts]
    no_protect = [False] * n
    theta = _hampel_filter(theta, no_protect, window=hampel_window)
    theta = _smooth(theta, no_protect, window=smooth_window)

    idx = [i for i, v in enumerate(theta) if v is not None and not math.isnan(v)]
    states = choose_overhook_states(
        [theta[i] for i in idx],
        [readouts[i].get("overhook_prob") for i in idx],
        switch_penalty, overhook_weight, overhook_prior)

    bio = [float("nan")] * n
    flags = [False] * n
    for i, s in zip(idx, states):
        bio[i] = 180.0 + theta[i] if s else 180.0 - theta[i]
        flags[i] = s
    bio = _interpolate_gaps(bio, max_gap)
    known = set(idx)
    for i in range(n):                         # interpolated frames take the side of 180 they ended up on
        if i not in known and not math.isnan(bio[i]):
            flags[i] = bio[i] > 180.0
    return bio, flags
