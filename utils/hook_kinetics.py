"""Kinetics of the apical hook: a closing / maintenance / opening curve fitted to one seedling's angle series.

Descriptive only: the fit never changes bio_angle, the states or the series filter. Angles are in the bio
convention (180 = closed); time is in hours since germination (or frames when there is no time axis).

Four candidate curves, with s(x) = 1 / (1 + exp(-x)):

    double    a + (p - a) s(k1 (t - t1)) - (p - b) s(k2 (t - t2)),  t2 = t1 + d, d >= 0
    opening   p - (p - b) s(k2 (t - t2))          first seen already closed
    closing   a + (p - a) s(k1 (t - t1))          closes and never opens
    flat      p                                   no transition

a = starting angle, p = closed plateau, b = final angle (all within LEVEL_BOUNDS), t1 / t2 = closing / opening
midpoints, k1 / k2 = steepness (1/h). Each is fitted with a robust (Huber) loss, frames weighted (Manual frames
count more), and the simplest curve the data support is kept by BIC; a model can also be forced.

Traits: each phase starts / ends at 10 % / 90 % of its change (t -+ ln 9 / k); maintenance runs from the end
of closing to the start of opening and is negative when the two overlap (no maintenance phase); the peak is
the highest point of the fitted curve over the observed time span, which lies below the plateau p when the
phases overlap. A phase end beyond the last observed frame is an extrapolation and is flagged.
"""
from __future__ import annotations

import math

import numpy as np

MODELS = ("flat", "closing", "opening", "double")
LEVEL_BOUNDS = (0.0, 270.0)          # a, p, b (deg)
K_BOUNDS = (0.01, 5.0)               # steepness, per hour (per frame without a time axis)
HUBER_SCALE = 10.0                   # deg: residuals beyond this count linearly (a mirrored run pulls little)
MIN_EXTRA_POINTS = 2                 # a model needs at least n_params + this many frames
LN9 = math.log(9.0)
N_PARAMS = {"flat": 1, "closing": 4, "opening": 4, "double": 7}


def _sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))      # = 1 / (1 + exp(-x)), overflow-free


def curve(model, params, t):
    """Fitted angle at times `t` for `model` with its parameter dict."""
    t = np.asarray(t, dtype=float)
    if model == "flat":
        return np.full_like(t, params["p"])
    if model == "closing":
        return params["a"] + (params["p"] - params["a"]) * _sig(params["k1"] * (t - params["t1"]))
    if model == "opening":
        return params["p"] - (params["p"] - params["b"]) * _sig(params["k2"] * (t - params["t2"]))
    if model == "double":
        return (params["a"] + (params["p"] - params["a"]) * _sig(params["k1"] * (t - params["t1"]))
                - (params["p"] - params["b"]) * _sig(params["k2"] * (t - params["t2"])))
    raise ValueError(f"unknown model {model!r}")


def _unpack(model, x):
    if model == "flat":
        return {"p": x[0]}
    if model == "closing":
        return {"a": x[0], "p": x[1], "t1": x[2], "k1": x[3]}
    if model == "opening":
        return {"p": x[0], "b": x[1], "t2": x[2], "k2": x[3]}
    a, p, b, t1, d, k1, k2 = x
    return {"a": a, "p": p, "b": b, "t1": t1, "t2": t1 + d, "k1": k1, "k2": k2}


def _starts_and_bounds(model, t, y):
    lo_l, hi_l = LEVEL_BOUNDS
    t0, t9 = float(t.min()), float(t.max())
    span = max(t9 - t0, 1.0)
    lo_t, hi_t = t0 - 0.5 * span, t9 + 0.5 * span
    clip = lambda v: float(np.clip(v, lo_l + 1e-6, hi_l - 1e-6))
    top, low = clip(np.percentile(y, 90)), clip(np.percentile(y, 10))
    first, last = clip(np.median(y[: max(3, len(y) // 8)])), clip(np.median(y[-max(3, len(y) // 8):]))
    times = [t0 + f * span for f in (0.15, 0.35, 0.55, 0.75)]
    ks = [4.0 / span, 12.0 / span]
    if model == "flat":
        return [[float(np.median(y))]], ([lo_l], [hi_l])
    if model == "closing":
        starts = [[first, top, tc, k] for tc in times for k in ks]
        return starts, ([lo_l, lo_l, lo_t, K_BOUNDS[0]], [hi_l, hi_l, hi_t, K_BOUNDS[1]])
    if model == "opening":
        starts = [[top, last, tc, k] for tc in times for k in ks]
        return starts, ([lo_l, lo_l, lo_t, K_BOUNDS[0]], [hi_l, hi_l, hi_t, K_BOUNDS[1]])
    starts = [[first, top, last, t1, d, k, k] for t1 in times[:3] for d in (0.15 * span, 0.4 * span) for k in ks]
    return starts, ([lo_l, lo_l, lo_l, lo_t, 0.0, K_BOUNDS[0], K_BOUNDS[0]],
                    [hi_l, hi_l, hi_l, hi_t, 2.0 * span, K_BOUNDS[1], K_BOUNDS[1]])


def _huber_cost(r, scale=HUBER_SCALE):
    a = np.abs(r)
    return float(np.sum(np.where(a <= scale, 0.5 * a * a, scale * (a - 0.5 * scale))))


def _direction_penalty(model, params):
    """The plateau is the closed end: closing rises to it (p >= a), opening falls from it (p >= b). Without this
    an "opening" curve with b > p would mimic closing (and the reverse), and BIC could not tell them apart."""
    gaps = []
    if model in ("closing", "double"):
        gaps.append(params["a"] - params["p"])
    if model in ("opening", "double"):
        gaps.append(params["b"] - params["p"])
    return DIRECTION_WEIGHT * np.maximum(np.array(gaps, dtype=float), 0.0)


DIRECTION_WEIGHT = 50.0


def _fit_one(model, t, y, w):
    from scipy.optimize import least_squares

    sw = np.sqrt(w)
    starts, bounds = _starts_and_bounds(model, t, y)
    best = None

    def residuals(x):
        params = _unpack(model, x)
        return np.concatenate([sw * (curve(model, params, t) - y), _direction_penalty(model, params)])

    for x0 in starts:
        x0 = np.clip(x0, np.array(bounds[0]) + 1e-9, np.array(bounds[1]) - 1e-9)
        res = least_squares(residuals, x0, bounds=bounds, loss="huber", f_scale=HUBER_SCALE, max_nfev=2000)
        cost = _huber_cost(res.fun)
        if best is None or cost < best[0]:
            best = (cost, res.x)
    return _unpack(model, best[1]), best[0]


def fit_kinetics(t, y, weights=None, model=None):
    """Fit one seedling. `t` (h since germination, or frames), `y` bio angles (NaNs dropped), `weights` per
    frame (default 1). `model` forces one of MODELS; None chooses by BIC among those with enough frames.

    Returns None when fewer than 2 frames remain, else {"model", "params", "bic": {model: value}, "n",
    "t_first", "t_last", "forced", "rmse", "traits"}."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(t) if weights is None else np.asarray(weights, dtype=float)
    keep = np.isfinite(t) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    t, y, w = t[keep], y[keep], w[keep]
    if len(t) < 2:
        return None
    order = np.argsort(t)
    t, y, w = t[order], y[order], w[order]
    n = len(t)
    candidates = [model] if model else [m for m in MODELS if n >= N_PARAMS[m] + MIN_EXTRA_POINTS] or ["flat"]
    fits, bic = {}, {}
    for m in candidates:
        params, cost = _fit_one(m, t, y, w)
        fits[m] = params
        # robust stand-in for the residual variance: 2 * Huber cost / n (equal to RSS / n inside the scale)
        bic[m] = n * math.log(max(2.0 * cost / n, 1e-9)) + N_PARAMS[m] * math.log(n)
    chosen = model or min(bic, key=bic.get)
    params = fits[chosen]
    resid = curve(chosen, params, t) - y
    return {"model": chosen, "params": {k: float(v) for k, v in params.items()}, "bic": bic, "n": n,
            "t_first": float(t[0]), "t_last": float(t[-1]), "forced": bool(model),
            "rmse": float(np.sqrt(np.mean(resid ** 2))),
            "traits": kinetics_traits(chosen, params, float(t[0]), float(t[-1]))}


def kinetics_traits(model, params, t_first, t_last):
    """Phase times, durations, peak, and which of them are extrapolated beyond the observed span."""
    out = {}
    if model in ("closing", "double"):
        h = LN9 / params["k1"]
        out.update(closing_mid=params["t1"], closing_start=params["t1"] - h, closing_end=params["t1"] + h,
                   closing_duration=2 * h)
    if model in ("opening", "double"):
        h = LN9 / params["k2"]
        out.update(opening_mid=params["t2"], opening_start=params["t2"] - h, opening_end=params["t2"] + h,
                   opening_duration=2 * h)
    if model == "double":
        out["maintenance_duration"] = out["opening_start"] - out["closing_end"]
    grid = np.linspace(t_first, t_last, 400)
    values = curve(model, params, grid)
    i = int(np.argmax(values))
    out["peak_angle"], out["peak_time"] = float(values[i]), float(grid[i])
    beyond = [k for k in ("closing_end", "opening_start", "opening_end") if k in out and out[k] > t_last]
    before = [k for k in ("closing_start",) if k in out and out[k] < t_first]
    out["extrapolated"] = ",".join(before + beyond)
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in out.items()}


def frame_axis(n, elapsed_minutes, time_zero):
    """(time of every frame since time_zero, unit "h" or "frame") for a series of `n` frames. Hours from the
    elapsed minutes when they are usable, else frame indices; time_zero None (or out of range) = frame 0."""
    start = 0 if time_zero is None or not 0 <= time_zero < n else int(time_zero)
    if elapsed_minutes is not None and len(elapsed_minutes) >= n and n:
        m = np.array([np.nan if v is None else float(v) for v in elapsed_minutes[:n]], dtype=float)
        if np.isfinite(m[start:]).any():
            zero = m[start] if np.isfinite(m[start]) else np.nanmin(m[start:])
            return (m - zero) / 60.0, "h"
    return np.arange(n, dtype=float) - start, "frame"


def seedling_points(frame_results, seed_id, elapsed_minutes, time_zero, manual_weight=2.0):
    """(t, y, weights, frame indices) of one seedling for fit_kinetics.

    `elapsed_minutes` per frame (None / NaN entries, or None for the whole list, give a frame-index axis);
    frames before `time_zero` (the germination frame index; None = from the first frame) are left out. t is
    in hours since time_zero (frames when there is no usable time axis); Manual frames get `manual_weight`."""
    n = len(frame_results)
    start = 0 if time_zero is None or not 0 <= time_zero < n else int(time_zero)
    axis, _ = frame_axis(n, elapsed_minutes, time_zero)
    t, y, w, idx = [], [], [], []
    for i in range(start, n):
        result = frame_results[i]
        if result is None:
            continue
        angle = result.get("angle_dict", {}).get(seed_id)
        if not isinstance(angle, (int, float)) or not np.isfinite(angle) or not np.isfinite(axis[i]):
            continue
        t.append(axis[i]); y.append(float(angle)); idx.append(i)
        w.append(manual_weight if result.get("state_dict", {}).get(seed_id) == "Manual" else 1.0)
    return np.array(t), np.array(y), np.array(w), idx


TRAIT_COLUMNS = ["seedling_id", "model", "forced", "n_frames", "time_unit", "time_zero_frame", "rmse",
                 "closing_start", "closing_mid", "closing_end", "closing_duration",
                 "maintenance_duration",
                 "opening_start", "opening_mid", "opening_end", "opening_duration",
                 "peak_time", "peak_angle", "extrapolated"]


def trait_row(seed_id, fit, time_unit, time_zero_frame):
    """One row of the per-seedling kinetics file (TRAIT_COLUMNS); empty cells for traits the model lacks."""
    row = {c: "" for c in TRAIT_COLUMNS}
    row.update(seedling_id=seed_id, time_unit=time_unit,
               time_zero_frame="" if time_zero_frame is None else time_zero_frame)
    if fit is None:
        row["model"] = "none"
        return row
    row.update(model=fit["model"], forced=int(fit["forced"]), n_frames=fit["n"], rmse=round(fit["rmse"], 2))
    for k, v in fit["traits"].items():
        row[k] = round(v, 2) if isinstance(v, float) else v
    return row
