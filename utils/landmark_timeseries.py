"""Temporal clean-up for landmark-head angle readings -- pure numpy.

utils/angle_timeseries.py is built for ellipse fits, whose per-frame angle aliases to
180 - a / a + 180 / 360 - a in sustained runs, so it re-branches the whole series. A landmark
reading has no such ambiguity: theta (the angle between the two predicted axis directions,
0..180) is well defined and the bio angle is simply 180 - theta (there is no overhook reading: the
flag is a few degrees past 180 deg, below the repeatability of the hand clicks). Feeding those
values through the ellipse DP only hurts: it flips readings that were already right (MAE 88.9 deg
instead of 22.4 deg on the hand-measured test frames).

So this module only does what a landmark series needs: a confidence gate (frames whose junction
peak is weak count as missing) and linear interpolation across short gaps (reconstruct_landmark_series,
method "none"), plus the robust clean-up the GUI uses (reconstruct_landmark_series_detail, method
"robust"): readings far from the median of their readable neighbours are rejected, a short running
median smooths the rest. A plain Hampel / moving average on theta made the old 134-frame sparse test
worse (MAE 22.4 -> 24.9 deg); the robust version was chosen on dense hand series instead
(multi/tune_series.py: 9 BerryBox v2 plants to choose, tested on 7 others and 57 sparse plants:
MAE 17.0 -> 13.4 and 14.3 -> 12.2 deg, frames off by > 20 deg 20% -> 15% and 17% -> 15%).
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


# ---- robust reconstruction ------------------------------------------------------------------------
# Readings are bio angles of readable frames. Three error types seen on the dense hand series
# (multi/evaluate_series.py, 16 BerryBox v2 plants): short spikes, runs of a few frames read ~150 deg
# off (cotyledon axis reversed) and short islands of readings where the annotator saw nothing.
# Defaults from multi/tune_series.py (the simplest setting within 0.3 deg of the best on the choosing
# plants; islands and measurable_p weighting did not earn their place there, so they are off).
ROBUST_WIDE = 3                # reference: median of up to this many readable neighbours per side (+ itself)
ROBUST_SPAN = 18               # ... no further than this many frames away
ROBUST_REJECT_DEG = 45.0       # a reading further than this from its reference is rejected
ROBUST_NARROW = 3              # running median over this many readable readings (1 = off)
ROBUST_ISLAND_MAX = 0          # readable runs this short (0 = off) ...
ROBUST_ISLAND_GAP = 6          # ... between missing runs this long (or the series end) are dropped
ROBUST_WEIGHTED = False        # weight the reference by the measurable probability
MANUAL_WEIGHT = 2.0             # a Manual frame counts double in a reference: an anchor, not a veto
EDIT_REJECTED, EDIT_ISLAND, EDIT_INTERPOLATED = "rejected", "island", "interpolated"


def _weighted_median(values, weights):
    order = np.argsort(values)
    v, w = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, c[-1] / 2.0)])


def _islands(finite, manual, island_max, island_gap):
    """Bool array: readable frames in runs of <= island_max with >= island_gap missing frames (or the
    series end) on both sides. Manual frames are never part of an island."""
    n = len(finite)
    out = np.zeros(n, bool)
    if island_max <= 0:
        return out
    i = 0
    while i < n:
        if not finite[i]:
            i += 1
            continue
        j = i
        while j < n and finite[j]:
            j += 1
        if j - i <= island_max and not manual[i:j].any():
            lo = i - 1
            while lo >= 0 and not finite[lo]:
                lo -= 1
            hi = j
            while hi < n and not finite[hi]:
                hi += 1
            gap_before = i if lo < 0 else i - lo - 1
            gap_after = n - j if hi >= n else hi - j
            if (lo < 0 or gap_before >= island_gap) and (hi >= n or gap_after >= island_gap):
                out[i:j] = True
        i = j
    return out


def robust_series(bio, weights=None, manual=None, blocked=None, wide=ROBUST_WIDE, span=ROBUST_SPAN,
                  reject_deg=ROBUST_REJECT_DEG, narrow=ROBUST_NARROW, island_max=ROBUST_ISLAND_MAX,
                  island_gap=ROBUST_ISLAND_GAP, max_gap=MAX_GAP_FRAMES):
    """(bio, edits): islands dropped, readings far from the weighted median of their readable
    neighbours rejected, a running median over the rest, short gaps interpolated (never across a
    `blocked` frame). `manual` frames are never changed and weigh MANUAL_WEIGHT in the references.
    edits[i] is "" / "island" / "rejected" / "interpolated"."""
    v = np.array([np.nan if x is None else float(x) for x in bio], float)
    n = len(v)
    manual = np.zeros(n, bool) if manual is None else np.asarray(manual, bool)
    w = np.ones(n) if weights is None else np.array([1.0 if (x is None or not np.isfinite(x)) else max(float(x), 1e-3)
                                                     for x in weights])
    w[manual] = MANUAL_WEIGHT
    edits = [""] * n
    finite = np.isfinite(v)
    isl = _islands(finite, manual, island_max, island_gap)
    for i in np.nonzero(isl)[0]:
        v[i], edits[i] = np.nan, EDIT_ISLAND
    if wide > 0:
        idx = np.nonzero(np.isfinite(v))[0]
        reject = []
        for j, i in enumerate(idx):
            if manual[i]:
                continue
            # the frame itself counts too: at a real step its own side wins the tie
            nb = [k for k in idx[max(0, j - wide):j + 1 + wide] if abs(k - i) <= span]
            if len(nb) >= 3 and abs(v[i] - _weighted_median(v[nb], w[nb])) > reject_deg:
                reject.append(i)
        for i in reject:
            v[i], edits[i] = np.nan, EDIT_REJECTED
    if narrow > 1:
        idx = np.nonzero(np.isfinite(v))[0]
        half = narrow // 2
        out = v.copy()
        for j, i in enumerate(idx):
            if manual[i]:
                continue
            h = min(half, j, len(idx) - 1 - j)          # symmetric: the first / last reading is kept
            nb = [k for k in idx[j - h:j + h + 1] if abs(k - i) <= span]
            out[i] = float(np.median(v[nb]))
        v = out
    before = np.isfinite(v)
    filled = _interpolate_gaps([float(x) for x in v], max_gap, blocked)
    for i in range(n):
        if not before[i] and not math.isnan(filled[i]) and not edits[i]:
            edits[i] = EDIT_INTERPOLATED
    return filled, edits


def reconstruct_landmark_series_detail(readouts, method="robust", min_peak=MIN_PEAK, **robust_kwargs):
    """(bio, edits) of a landmark series. readouts: per-frame dicts (theta, peak, optional
    measurable_p, "manual": True for a fixed Manual reading, "gated": True for a frame the readability
    gate rejected) or None. method "none" is the plain reconstruction (reconstruct_landmark_series
    defaults: gates and short-gap interpolation only)."""
    if method == "none":
        bio = reconstruct_landmark_series(readouts, min_peak=min_peak)
        return bio, [EDIT_INTERPOLATED if not _valid(r, min_peak) and not math.isnan(b) else ""
                     for r, b in zip(readouts, bio)]
    if method != "robust":
        raise ValueError(f"unknown reconstruction method {method!r}")
    weighted = robust_kwargs.pop("weighted", ROBUST_WEIGHTED)
    bio = [bio_from_theta(float(r["theta"])) if _valid(r, min_peak) else None for r in readouts]
    weights = [r.get("measurable_p") if (weighted and r) else None for r in readouts]
    manual = [bool(r and r.get("manual")) for r in readouts]
    blocked = [bool(r and r.get("gated")) for r in readouts]
    return robust_series(bio, weights, manual, blocked, **robust_kwargs)
