import numpy as np

from utils.angle_timeseries import (
    reconstruct_series,
    _align_branches,
    _classify_states,
    CLOSED_STATE,
    OPENING_STATE,
    MANUAL_STATE,
)

# All angles below are in the BIO convention the module works in: 180 = closed,
# decreasing toward 0 as the hook opens, >180 = overhooked.


def test_clean_closed_to_opening_curve_stays_stable():
    angles = [178, 176, 179, 170, 140, 100, 60, 20, 5, 2]
    angles_out, states = reconstruct_series(angles)

    assert states[0] == CLOSED_STATE
    assert states[-1] == OPENING_STATE
    # State must be monotone: once Opening, never back to Closed.
    seen_opening = False
    for s in states:
        if s == OPENING_STATE:
            seen_opening = True
        elif s == CLOSED_STATE:
            assert not seen_opening


def test_injected_180_flip_is_resolved():
    # True trajectory holds near 178 the whole time; two isolated frames are
    # reported as their 180-x alias (2 instead of 178). The global alignment
    # snaps them back onto the closed plateau because doing so lowers the
    # whole-path smoothness cost.
    angles = [178, 179, 178, 2, 178, 179, 178, 177, 2, 177, 178, 179]
    angles_out, states = reconstruct_series(angles)

    for a in angles_out:
        assert a > 150, f"expected all frames resolved near the closed plateau, got {a}"
    assert all(s == CLOSED_STATE for s in states)


def test_sustained_wrong_branch_run_is_realigned():
    # The failure the global pass exists for (and that a local-window resolver
    # cannot fix): a whole multi-frame STRETCH is reported on the 180-x alias,
    # so it looks locally smooth and self-consistent. Here the true trajectory
    # opens 178 -> ~40, but frames 3..7 are flipped to 180-x (their mirror
    # about 90). Global alignment must pull that run back onto the real
    # opening curve, not accept the mirrored (fake) shape.
    true_curve = [178, 150, 120, 95, 75, 60, 50, 42, 30, 20]
    reported = list(true_curve)
    for i in range(3, 8):
        reported[i] = 180 - reported[i]  # sustained mirror flip
    aligned = _align_branches(reported, [False] * len(reported))

    for i in range(len(true_curve)):
        assert abs(aligned[i] - true_curve[i]) < 15, (
            f"frame {i}: aligned {aligned[i]:.1f} not near true {true_curve[i]}"
        )


def test_real_small_dips_are_not_flipped_away():
    # Guard against the mirror of the above: a densely-sampled, genuinely-
    # opening curve with small frame-to-frame wobble (real openings are not
    # perfectly monotone) must NOT be "corrected" onto a far alias. The mild
    # asymmetry stays weak enough to leave honest data essentially alone --
    # the only permitted movement is a few degrees right at the 90-degree
    # crossover, where a value and its 180-x mirror are only a few degrees
    # apart anyway (so the choice barely matters).
    angles = [176, 170, 162, 150, 143, 145, 130, 122, 124, 110, 100, 92, 80, 72, 64, 52, 44]
    aligned = _align_branches(angles, [False] * len(angles))
    for i, a in enumerate(angles):
        assert abs(aligned[i] - a) < 6, (
            f"frame {i}: honest reading {a} was altered to {aligned[i]:.1f}"
        )


def test_single_frame_spike_is_rejected():
    angles = [175, 176, 174, 5, 175, 177, 174]  # index 3 is a lone bad frame
    angles_out, states = reconstruct_series(angles)

    assert angles_out[3] > 150
    assert all(s == CLOSED_STATE for s in states)


def test_multi_frame_overhook_spike_is_rejected():
    # Genuine trajectory sits near 178 throughout; a short run of frames
    # falsely reports an overhooked (>180) reading due to a segmentation
    # glitch, similar to the real 228/225/197/265/183 spike observed.
    angles = [178, 176, 179, 228, 225, 197, 178, 176]
    angles_out, states = reconstruct_series(angles)

    assert all(s == CLOSED_STATE for s in states)
    for a in angles_out:
        assert 150 < a < 210


def test_missing_frames_stay_missing_and_carry_state_forward():
    # Gradual post-gap decline (matching real sampled cadence) rather than a
    # single one-frame jump -- see test_state_never_reverts_to_closed... for
    # why an abrupt full round-trip isn't a realistic scenario for a
    # continuity-based resolver.
    angles = [178, 176, "-", np.nan, None, 174, 140, 90, 40, 15, 5, 2]
    angles_out, states = reconstruct_series(angles)

    for i in (2, 3, 4):
        assert np.isnan(angles_out[i])
        assert states[i] == CLOSED_STATE  # carried forward, not reset

    assert states[-1] == OPENING_STATE


def test_manual_frame_is_untouched_and_labeled_manual():
    angles = [178, 176, 174, 999, 176, 174]
    is_manual = [False, False, False, True, False, False]
    angles_out, states = reconstruct_series(angles, is_manual=is_manual)

    assert angles_out[3] == 999
    assert states[3] == MANUAL_STATE
    # Neighbors are unaffected by the manual value's magnitude.
    assert states[2] == CLOSED_STATE
    assert states[4] == CLOSED_STATE


def test_state_never_reverts_to_closed_after_opening():
    # A full round trip back to ~180 within a couple of frames is not
    # something reconstruct_series's continuity-based resolver can (or
    # should) read as "opened, then closed again" -- from local continuity
    # alone that's indistinguishable from staying closed with a couple of
    # noisy low readings, the same ambiguity failure mode 1 describes. So
    # this checks the state machine's own monotonicity guarantee directly:
    # even if the (already branch-resolved, already smoothed) angle signal
    # handed to it contains a late high reading -- e.g. from a glitch
    # downstream of branch resolution -- the emitted state must not revert
    # to Closed. Enough consecutive low frames are included to legitimately
    # cross OPENING_SUSTAIN_FRAMES before the late high reading appears.
    values = [178, 176, 20, 15, 10, 12, 18, 170, 165, 172]
    manual_mask = [False] * len(values)
    states = _classify_states(values, manual_mask)

    opening_index = states.index(OPENING_STATE)
    assert all(s == OPENING_STATE for s in states[opening_index:])


def test_empty_and_single_element_series():
    assert reconstruct_series([]) == ([], [])

    angles_out, states = reconstruct_series([178])
    assert states == [CLOSED_STATE]
    assert angles_out[0] == 178
