"""Temporal consistency checks on one seedling's landmark readouts (flag only).

Two facts of a growing etiolated seedling, measured on the hand annotations (737 consecutive
annotated frame pairs of 279 seedlings, five annotation sets, crop pixels):

* The collar barely moves: median step between annotated frames 2.6 px, median absolute deviation
  from the seedling's own median 1.7 px; the 95th percentile of collar step / junction-collar
  distance is 0.19. The only real drift seen is slow, ~25 px over a whole series
  (dark_grown_arabidopsis_kinematics).
* The junction moves away from the collar as the hypocotyl elongates: the junction-collar distance
  increased in 99.7% of pairs, never dropped by more than 2.7 px, and the junction was never below
  the collar.

temporal_flags() marks the frames whose model readout breaks either rule. It never changes an
angle: the GUI shows the flag (analysis window, kinematics plot, export column) and the user
decides. Pure numpy, no Tk.
"""
import math

import numpy as np

# Rolling median of the readable collars: the seedling's "stable" collar for each frame.
COLLAR_WINDOW = 7
# A collar is flagged when it sits further from the stable collar than both bounds below.
COLLAR_JUMP_PX = 15.0
COLLAR_JUMP_FRAC = 0.25          # x junction-collar distance (measured p95 of step/distance: 0.19)
# A junction is flagged when its distance to the stable collar drops this much below the
# running maximum of earlier frames (real data: never more than 2.7 px).
JC_DROP_PX = 5.0
JC_TRAIL = 3                     # readable frames in the trailing median the reference is built from
# Fewer readable frames than this: nothing to compare against, no flags.
MIN_READABLE = 3

COLLAR_FLAG = "collar"
JUNCTION_FLAG = "junction"


def _point(p):
    if p is None:
        return None
    try:
        x, y = float(p[0]), float(p[1])
    except (TypeError, ValueError, IndexError):
        return None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def _dist(a, b):
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _empty():
    return {"flag": "", "collar_ref": None, "collar_dev": None, "jc_dist": None}


def _stable_collars(collars, manual):
    """Per frame, the rolling median of the readable collars around it (None where no collar is
    readable at all). Manual collars inside the window take precedence over model ones."""
    idx = [i for i, c in enumerate(collars) if c is not None]
    out = [None] * len(collars)
    if not idx:
        return out
    half = COLLAR_WINDOW // 2
    for i in range(len(collars)):
        k = int(np.argmin([abs(j - i) for j in idx]))            # nearest readable collar
        window = idx[max(0, k - half):k + half + 1]
        trusted = [j for j in window if manual[j]]
        use = trusted or window
        out[i] = (float(np.median([collars[j][0] for j in use])),
                  float(np.median([collars[j][1] for j in use])))
    return out


def temporal_flags(points, manual=None):
    """Flags for one seedling's frames, in time order.

    `points[i]` is None for a frame without a usable readout (no readout, readability-gated),
    else {"junction": (x, y) or None, "collar": (x, y) or None} in crop pixels. `manual[i]` marks
    frames whose landmarks the user placed: trusted, never flagged, and they reset the junction
    reference. Returns one dict per frame: "flag" ("", "collar", "junction" or "collar+junction"),
    "collar_ref" (the stable collar used, or None), "collar_dev" and "jc_dist" (px, or None)."""
    n = len(points)
    manual = list(manual) if manual is not None else [False] * n
    junctions = [_point(p.get("junction")) if p else None for p in points]
    collars = [_point(p.get("collar")) if p else None for p in points]
    out = [_empty() for _ in range(n)]
    if sum(1 for p, j in zip(points, junctions) if p and j is not None) < MIN_READABLE:
        return out

    refs = _stable_collars(collars, manual)
    dists = [None] * n
    for i in range(n):
        if points[i] is None:
            continue
        out[i]["collar_ref"] = refs[i]
        if collars[i] is not None and refs[i] is not None:
            out[i]["collar_dev"] = _dist(collars[i], refs[i])
        if junctions[i] is not None and refs[i] is not None:
            dists[i] = _dist(junctions[i], refs[i])
            out[i]["jc_dist"] = dists[i]

    # Junction reference: the running maximum of the median of the last TRAIL readable distances,
    # earlier frames only. One junction read too high cannot raise it (the median ignores it), and
    # a frame read too low cannot lower it (it is judged before it enters). Manual frames reset it.
    reference, trail = None, []
    for i in (i for i in range(n) if dists[i] is not None):
        flags = []
        if not manual[i]:
            dev = out[i]["collar_dev"]
            if dev is not None and dev > max(COLLAR_JUMP_PX, COLLAR_JUMP_FRAC * dists[i]):
                flags.append(COLLAR_FLAG)
            if reference is not None and dists[i] < reference - JC_DROP_PX:
                flags.append(JUNCTION_FLAG)
        out[i]["flag"] = "+".join(flags)
        if manual[i]:
            reference, trail = dists[i], [dists[i]]     # trusted, even when lower than before
        elif JUNCTION_FLAG not in flags:
            trail = (trail + [dists[i]])[-JC_TRAIL:]
            smoothed = float(np.median(trail))
            reference = smoothed if reference is None else max(reference, smoothed)
    # Collar flags on frames whose junction is unreadable (no distance): absolute bound only.
    for i in range(n):
        if dists[i] is None and not manual[i] and out[i]["collar_dev"] is not None \
                and out[i]["collar_dev"] > COLLAR_JUMP_PX:
            out[i]["flag"] = COLLAR_FLAG
    return out


def describe_flag(flag, collar_dev=None):
    """One-line explanation of a flag for the analysis window ("" when there is none)."""
    parts = []
    if COLLAR_FLAG in (flag or ""):
        dev = f" {collar_dev:.0f} px" if isinstance(collar_dev, (int, float)) else ""
        parts.append(f"collar jumped{dev} from the seedling's collar")
    if JUNCTION_FLAG in (flag or ""):
        parts.append("junction moved back toward the collar")
    return "; ".join(parts)
