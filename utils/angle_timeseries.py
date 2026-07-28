"""
Per-seedling temporal reconstruction of the apical hook angle.

CONVENTION. This module works entirely in the "bio" angle convention:
**180 = fully closed**, decreasing toward 0 as the hook opens, with genuine
overhooking (folded back past closed) running above 180. This is the
convention the user's thresholds are stated in (closed/overhooking > 160,
opening < 150) and the one the export's bio_angle column and the kinematics
180-degree reference line use.

Note that AngleCalculator.compute_biological_angle
(utils/apicalhook_angle.py) actually emits the OPPOSITE convention -- ~0 when
closed, growing toward ~180 as the hook opens -- so the caller
(Gui._reconstruct_series_for_crop) converts each automated reading to bio via
`180 - raw` before handing it here; manual overrides are already stored in
bio. Everything below therefore assumes bio input.

Two failure modes of the independent per-frame geometry show up once frames
are viewed as a time series instead of in isolation:

1. Branch/orientation swaps: the cotyledon/hypocotyl ellipse fits are
   direction-ambiguous, so the true angle `a` can alias to `180 - a`,
   `a + 180`, or `360 - a`. These swaps happen in sustained multi-frame runs
   (a whole stretch reported on the wrong branch), not just isolated frames.
2. Outlier spikes: a single bad frame (segmentation glitch) can report a
   wildly wrong angle even though the branch itself is resolved correctly.

reconstruct_series() takes one seedling's ordered per-frame bio readings and
produces a temporally-consistent angle plus a two-state classification
(Closed / Opening) anchored in the biological prior that a dark-grown hook
forms, holds closed, then opens monotonically -- and does not re-close.
"""

import numpy as np

# Biological state thresholds (user-specified): a smoothed angle above
# CLOSED_THRESHOLD_DEG is unambiguously closed/overhooking; below
# OPENING_THRESHOLD_DEG it is unambiguously opening. The gap between the two
# is a hysteresis deadband that holds whatever state was last confidently
# entered, so noise sitting right at the boundary doesn't flicker the label.
CLOSED_THRESHOLD_DEG = 160.0
OPENING_THRESHOLD_DEG = 150.0

# Consecutive non-gap, non-manual frames the angle must stay below
# OPENING_THRESHOLD_DEG before the state machine commits to "Opening" -- a
# real dark-grown hook opens over many frames, not one, so a short dip must
# not flip the label. Tune against real data: a longer window resists noise
# but delays genuinely fast openers; a shorter one is more responsive but
# more exposed to the ellipse's known closed-phase unreliability (see the
# module docstring, failure mode 1).
OPENING_SUSTAIN_FRAMES = 5

# Global branch alignment (see _align_branches). Each frame has up to 4
# ellipse-alias candidate values; a dynamic program picks one per frame to
# minimize a transition cost along the whole series. The cost is
# SMOOTHNESS-DOMINANT (frame-to-frame absolute change) with only a MILD
# asymmetric surcharge on moves in the re-closing direction:
#
#   cost(prev -> cur) = |cur - prev| , plus (BRANCH_CLOSE_ASYMMETRY - 1) *
#                       (the re-closing part of the move)
#
# so a step that re-closes the hook costs BRANCH_CLOSE_ASYMMETRY times a step
# of equal size that opens it. Two design notes:
#
#  - Smoothness has to stay primary. A purely monotone (decrease-only) prior
#    was tried and rejected: real openings have small frame-to-frame wobble,
#    and a strong monotone penalty makes the DP escape onto a `+/-180` alias
#    to avoid a 1-2 degree dip, wrecking already-correct seedlings.
#  - But pure symmetric smoothness was ALSO rejected (this is the "fake
#    round-trip" failure noted in earlier revisions): with no asymmetry the
#    cheapest path through a sustained wrong-branch run is often to follow it
#    down and back rather than pay the one-time jump onto the correct branch.
#    The mild asymmetry (calibrated ~1.8-2.2 against the four F1_Plate_2_YS
#    seedlings; 2.0 is the default) is just enough to break that tie toward
#    the biological "keeps opening" shape without over-penalizing real
#    wobble. Above ~4 it regresses to the monotone failure above.
BRANCH_CLOSE_ASYMMETRY = 2.0
# Weak pull toward starting the trajectory at the closed plateau (180 in bio
# convention): the first resolved frame's candidates are seeded with a small
# cost proportional to their distance from CLOSED_ANGLE_BIO, so an otherwise
# free choice of absolute branch prefers "starts closed" (a dark-grown hook
# does).
CLOSED_ANGLE_BIO = 180.0
BRANCH_START_ANCHOR_WEIGHT = 0.1

HAMPEL_WINDOW = 3
HAMPEL_N_SIGMAS = 3.0
# MAD-to-sigma scale factor for a normal distribution (0.6745 = the standard
# consistency constant so MAD approximates the standard deviation).
_MAD_TO_SIGMA = 1.4826

# Simple centered moving-average window applied after outlier rejection.
SMOOTH_WINDOW = 3

# While the state machine reports "Closed", optionally floor the emitted
# angle to the highest smoothed value seen so far this Closed run, instead
# of showing a dip that contradicts the label. Off by default: calibrating
# against real data (img_angle_data.csv) showed this creates a misleading
# *cliff* right at the Closed->Opening boundary -- the displayed angle holds
# artificially flat (e.g. 165) while the underlying smoothed signal is
# already eroding underneath it (e.g. 140, 107, 56, 34), then suddenly drops
# to match it the instant the state flips. Showing the real smoothed value
# throughout is more honest: it surfaces the ellipse's known closed-phase
# instability (failure mode 1) as a visible decline rather than hiding it.
FLOOR_CLOSED_ANGLE = False

CLOSED_STATE = "Closed"
OPENING_STATE = "Opening"
MANUAL_STATE = "Manual"


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, str):
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def _branch_candidates(value):
    """
    The values a single true angle can alias to under the two known ellipse
    ambiguities: a vector-sign flip (`180 - value`) and a Closed<->Overhooked
    miscall, which shifts the reported value by exactly 180 (`value + 180`,
    and their composition `360 - value`).

    Each of the four is offered BOTH in [0, 360) and in its `- 360`
    representation, so the alignment DP can lay a trajectory on a continuous
    line and let genuine overhook dip just past 180 (or a near-closed value
    sit just below 0) instead of wrapping the full 360 and looking like a
    huge jump. The base four-value set is closed under `x -> 180 - x`, so this
    is convention-independent (identical whether `value` is expressed as
    bio or raw).
    """
    base = {
        value % 360.0,
        (180.0 - value) % 360.0,
        (value + 180.0) % 360.0,
        (360.0 - value) % 360.0,
    }
    out = set()
    for b in base:
        out.add(b)
        out.add(b - 360.0)
    return sorted(out)


def _align_branches(angles, manual_mask):
    """
    Global branch alignment via dynamic programming (see BRANCH_CLOSE_ASYMMETRY).

    Each non-missing, non-manual frame contributes its _branch_candidates as
    the states of one DP stage; a manual frame is pinned to its single given
    value (a fixed anchor that the surrounding path must connect through);
    missing frames are skipped entirely (the transition is measured between
    the nearest present frames on either side). The path minimizing the total
    smoothness-dominant, mildly-asymmetric transition cost is chosen and its
    per-frame value returned.

    Unlike a local-window correction, this resolves SUSTAINED wrong-branch
    runs (a whole stretch reported on the `180 - a` alias): the alternative
    branch only wins if committing to it lowers the whole-path cost, which a
    sustained run does but an isolated smooth trend does not.

    Known limitation: a value and its `180 - a` mirror coincide at 90 and are
    only a few degrees apart near it, so around the 90-degree crossover the
    two branches are nearly free to swap -- the resolved value there can shift
    by a few degrees (harmless, since the branches themselves nearly agree).
    Separately, the LAST present frame has no right-hand neighbor to constrain
    it, so a coarse final opening step can flip to its nearer mirror (seen as
    a small end-of-trajectory overshoot on real data); the downstream smoother
    damps it, and it does not affect interior frames.

    Returns a list the same length as `angles`, NaN where missing. Values may
    fall slightly outside [0, 360) (the continuous representation); callers
    that need a display value can leave them as-is or re-fold.
    """
    n = len(angles)
    present = []
    cand = {}
    for i, a in enumerate(angles):
        if _is_missing(a):
            continue
        present.append(i)
        cand[i] = [float(a)] if manual_mask[i] else _branch_candidates(float(a))

    resolved = [np.nan] * n
    if not present:
        return resolved

    first = present[0]
    # cost[c] = best total cost of a path ending at candidate value c for the
    # current frame; back[c] = the previous frame's chosen value on that path.
    cost = {c: abs(c - CLOSED_ANGLE_BIO) * BRANCH_START_ANCHOR_WEIGHT for c in cand[first]}
    back_pointers = {first: {c: None for c in cand[first]}}

    for i in present[1:]:
        new_cost, new_back = {}, {}
        for c in cand[i]:
            best_total, best_prev = None, None
            for pc, pcost in cost.items():
                delta = c - pc
                if delta >= 0:
                    # bio increases -> hook re-closing -> surcharged
                    step = BRANCH_CLOSE_ASYMMETRY * delta
                else:
                    step = -delta
                total = pcost + step
                if best_total is None or total < best_total:
                    best_total, best_prev = total, pc
            new_cost[c] = best_total
            new_back[c] = best_prev
        cost, back_pointers[i] = new_cost, new_back

    # backtrack from the cheapest endpoint
    chosen = {}
    ci = min(cost, key=cost.get)
    for i in reversed(present):
        chosen[i] = ci
        ci = back_pointers[i][ci]

    for i in present:
        resolved[i] = chosen[i]
    return resolved


def _local_median_and_sigma(values, i, window, min_neighbors):
    """Median and scaled-MAD of the non-missing neighbors of index i within
    `window` frames on each side (i itself excluded). Returns (None, None)
    if there aren't at least `min_neighbors` such values."""
    n = len(values)
    lo, hi = max(0, i - window), min(n, i + window + 1)
    neighborhood = [values[j] for j in range(lo, hi) if j != i and values[j] is not None]
    if len(neighborhood) < min_neighbors:
        return None, None
    median = float(np.median(neighborhood))
    mad = float(np.median([abs(v - median) for v in neighborhood]))
    return median, mad * _MAD_TO_SIGMA


def _hampel_filter(values, protect_mask, window=HAMPEL_WINDOW, n_sigmas=HAMPEL_N_SIGMAS):
    """
    Rolling median + MAD outlier rejection, on top of already branch-
    resolved values: a point more than `n_sigmas` scaled-MADs from its local
    neighbors' median is replaced by that median -- catches a magnitude-only
    glitch that isn't explained by any of the branch reflections (so
    _align_branches left it as-is). Points where protect_mask is True
    (manual overrides) are never replaced, though they still contribute to
    their neighbors' local statistics.
    """
    values = [None if _is_missing(v) else float(v) for v in values]
    n = len(values)
    out = list(values)

    for i in range(n):
        if protect_mask[i] or values[i] is None:
            continue
        median, sigma = _local_median_and_sigma(values, i, window, min_neighbors=3)
        if median is None:
            continue
        if abs(values[i] - median) > n_sigmas * sigma:
            out[i] = median

    return out


def _smooth(values, protect_mask, window=SMOOTH_WINDOW):
    """Centered NaN-aware moving average; manual frames pass through
    untouched but still contribute to neighboring windows."""
    values = list(values)
    n = len(values)
    out = list(values)
    radius = window // 2

    for i in range(n):
        if protect_mask[i] or _is_missing(values[i]):
            continue
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        neighborhood = [values[j] for j in range(lo, hi) if not _is_missing(values[j])]
        if not neighborhood:
            continue
        out[i] = float(np.mean(neighborhood))

    return out


def _classify_states(values, manual_mask):
    """
    Hysteresis + monotone state machine: starts Closed (a dark-grown hook
    forms closed), switches to Opening only once the angle has stayed below
    OPENING_THRESHOLD_DEG for OPENING_SUSTAIN_FRAMES consecutive non-gap,
    non-manual frames, and never reverts to Closed afterward. The 150-160
    deadband is implicit: neither threshold check fires there, so the
    current state simply holds. Manual frames report state "Manual" without
    affecting the streak counter's *decision* other than contributing their
    own value if present (so a run of manual edits doesn't stall automatic
    detection of a real opening indefinitely).
    """
    states = [None] * len(values)
    current = CLOSED_STATE
    opening_streak = 0

    for i, v in enumerate(values):
        missing = _is_missing(v)

        if not missing:
            if v < OPENING_THRESHOLD_DEG:
                opening_streak += 1
                if opening_streak >= OPENING_SUSTAIN_FRAMES:
                    current = OPENING_STATE
            elif v > CLOSED_THRESHOLD_DEG:
                opening_streak = 0
            # Inside the deadband: streak neither grows nor resets.

        states[i] = MANUAL_STATE if manual_mask[i] else current

    return states


def _floor_closed_angle(values, states):
    """Within a Closed run, never report a value lower than the highest
    smoothed value seen so far in that run (see FLOOR_CLOSED_ANGLE's
    docstring at the top of this module)."""
    out = list(values)
    running_high = None

    for i, state in enumerate(states):
        if state != CLOSED_STATE or _is_missing(out[i]):
            running_high = None
            continue
        running_high = out[i] if running_high is None else max(running_high, out[i])
        out[i] = running_high

    return out


def reconstruct_series(angles, is_manual=None):
    """
    angles: ordered per-frame angle readings for one seedling in the BIO
        convention (0-360 scale, 180 = closed, decreasing as the hook opens,
        >180 = overhooked -- see the module docstring), with None/NaN/'-' for
        a missing frame. The caller is responsible for converting the
        automated per-frame geometry to this convention first.
    is_manual: parallel list of bools marking frames the user has manually
        overridden -- treated as fixed anchors, never re-branched or
        smoothed, and always reported with state "Manual".

    Returns (angles_out, states_out), same length as `angles`. A missing
    frame stays missing (NaN) in angles_out; its state carries forward from
    the last resolved frame.
    """
    n = len(angles)
    manual_mask = list(is_manual) if is_manual is not None else [False] * n
    if len(manual_mask) != n:
        raise ValueError("is_manual must be the same length as angles")

    resolved = _align_branches(angles, manual_mask)
    filtered = _hampel_filter(resolved, manual_mask)
    smoothed = _smooth(filtered, manual_mask)
    states = _classify_states(smoothed, manual_mask)

    if FLOOR_CLOSED_ANGLE:
        smoothed = _floor_closed_angle(smoothed, states)

    angles_out = [np.nan if _is_missing(v) else float(v) for v in smoothed]
    return angles_out, states


def sync_angle_and_state_lists(result):
    """
    Rebuilds a process_single_frame result dict's seed_ids-aligned
    angle_list/state_list projections from its authoritative angle_dict/
    state_dict -- shared by every caller that mutates those dicts directly
    (a manual override in SeedlingAnalysisWindow, or this module's own
    per-crop reconstruction pass in Gui._reconstruct_series_for_crop) so the
    two representations never drift apart.
    """
    seed_ids = result["seed_ids"]
    angle_list = []
    for sid in seed_ids:
        a = result["angle_dict"].get(sid, '-')
        angle_list.append(round(a) if isinstance(a, (int, float)) and not np.isnan(a) else '-')
    result["angle_list"] = angle_list
    result["state_list"] = [result.get("state_dict", {}).get(sid, '-') for sid in seed_ids]
