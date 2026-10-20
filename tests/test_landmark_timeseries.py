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


# ---- robust reconstruction (the GUI's) ------------------------------------------------------------
from utils.landmark_timeseries import reconstruct_landmark_series_detail as robust  # noqa: E402


def _bio(values):
    return [None if v is None else {"theta": 180.0 - v, "peak": 0.9} for v in values]


def test_robust_rejects_a_spike_and_a_short_dive():
    series = [170.0] * 8 + [20.0] * 3 + [168.0] * 8 + [40.0] + [166.0] * 6
    bio, edits = robust(_bio(series))
    assert all(abs(b - 168) < 4 for b in bio)                       # dive and spike gone, gaps filled
    assert edits[8:11] == ["rejected"] * 3 and edits[19] == "rejected"


def test_robust_keeps_a_real_step_and_reclosing():
    step = [170.0] * 10 + [80.0] * 10
    bio, edits = robust(_bio(step))
    assert bio[:10] == [170.0] * 10 and bio[10:] == [80.0] * 10 and not any(edits)
    reclose = [170.0] * 6 + [120.0] * 6 + [40.0] * 6 + [90.0] * 6      # opens, then closes again
    bio, _ = robust(_bio(reclose))
    assert bio[-1] == 90.0 and min(bio) == 40.0


def test_robust_never_changes_manual_frames_and_uses_them_as_anchors():
    rs = _bio([170.0] * 5 + [30.0] + [170.0] * 5)
    rs[5]["manual"] = True                                           # a Manual 30 deg stays
    bio, edits = robust(rs)
    assert bio[5] == 30.0 and edits[5] == ""


def test_robust_gated_frames_block_interpolation_and_none_method_is_the_plain_one():
    rs = _bio([150.0, 150.0, 150.0]) + [{"theta": None, "gated": True}] + _bio([150.0, 150.0, 150.0])
    bio, edits = robust(rs)
    assert math.isnan(bio[3]) and edits[3] == ""
    plain = [ro(30), None, ro(32), ro(120), ro(34)]
    assert robust(plain, method="none")[0] == reconstruct_landmark_series(plain)
    assert robust(plain, method="none")[1][1] == "interpolated"


def test_robust_islands_when_enabled():
    rs = _bio([150.0] * 5 + [None] * 8 + [100.0] + [None] * 8)
    bio, edits = robust(rs, island_max=2)
    assert edits[13] == "island" and math.isnan(bio[13])
    assert robust(rs)[1][13] != "island"                             # the rule is off by default
