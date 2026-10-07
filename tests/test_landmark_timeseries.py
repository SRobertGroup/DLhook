import math

from utils.landmark_timeseries import reconstruct_landmark_series


def ro(theta, peak=0.9):
    return {"theta": theta, "peak": peak}


def test_bio_is_180_minus_theta_and_never_overhooked():
    bio = reconstruct_landmark_series([ro(0), ro(30), ro(90), ro(180)])
    assert bio == [180.0, 150.0, 90.0, 0.0]
    assert max(bio) <= 180.0


def test_weak_peaks_and_missing_frames_are_gaps_that_short_gaps_interpolate():
    readouts = [ro(30), None, None, ro(60), ro(61, peak=0.05), ro(63)]
    bio = reconstruct_landmark_series(readouts)
    assert bio[0] == 150.0 and bio[3] == 120.0
    assert bio[1] == 140.0 and bio[2] == 130.0                      # linear across the 2-frame gap
    assert not math.isnan(bio[4]) and abs(bio[4] - 118.5) < 1e-9    # the weak frame is interpolated, not trusted


def test_long_gap_and_edges_stay_missing():
    bio = reconstruct_landmark_series([None, ro(30), None, None, None, None, None, ro(40)])
    assert math.isnan(bio[0]) and math.isnan(bio[3]) and bio[1] == 150.0


def test_hampel_and_smoothing_are_off_by_default_but_available():
    readouts = [ro(30), ro(32), ro(120), ro(34), ro(36)]
    assert reconstruct_landmark_series(readouts)[2] == 60.0           # a glitch is left alone by default
    cleaned = reconstruct_landmark_series(readouts, hampel_window=2, smooth_window=3)
    assert 140 < cleaned[2] < 160


def test_empty_and_all_missing():
    assert reconstruct_landmark_series([]) == []
    assert all(math.isnan(v) for v in reconstruct_landmark_series([None, None]))
