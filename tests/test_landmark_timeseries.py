import math

from utils.landmark_timeseries import choose_overhook_states, reconstruct_landmark_series


def ro(theta, peak=0.9, p=None):
    return {"theta": theta, "peak": peak, "overhook_prob": p}


def test_a_consistently_confident_head_sets_the_state():
    theta = [30.0, 20.0, 10.0, 15.0, 25.0, 35.0]
    assert choose_overhook_states(theta, probs=[0.9] * 6) == [True] * 6
    assert choose_overhook_states(theta, probs=[0.1] * 6) == [False] * 6


def test_monotone_opening_stays_not_overhooked():
    assert choose_overhook_states([5, 10, 20, 35, 50, 70, 90]) == [False] * 7


def test_overhook_run_is_not_flickered_by_one_noisy_probability():
    probs = [0.9] * 4 + [0.1] + [0.9] * 4
    assert choose_overhook_states([40.0] * 9, probs) == [True] * 9


def test_outlier_and_gaps_are_handled():
    readouts = [ro(30), ro(32), ro(120), ro(34), None, None, ro(40), ro(41, peak=0.05), ro(43), ro(44)]
    bio, flags = reconstruct_landmark_series(readouts, hampel_window=2, smooth_window=3)
    assert 140 < bio[2] < 160                       # the 120 glitch is replaced by its neighbours
    assert not math.isnan(bio[4]) and not math.isnan(bio[5])   # short gap interpolated
    assert not math.isnan(bio[7])                   # low-peak frame interpolated, not trusted
    assert not any(flags)


def test_long_gap_stays_missing():
    bio, _ = reconstruct_landmark_series([ro(30), None, None, None, None, None, ro(40)])
    assert math.isnan(bio[3]) and not math.isnan(bio[0])


def test_empty_and_all_missing():
    assert reconstruct_landmark_series([]) == ([], [])
    bio, flags = reconstruct_landmark_series([None, None])
    assert all(math.isnan(v) for v in bio) and flags == [False, False]
