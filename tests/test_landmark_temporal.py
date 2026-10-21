"""Time-series consistency flags on landmark readouts (utils/landmark_temporal.py) and their wiring:
the Gui writes them onto the readouts without touching angles, the export and kinematics show them."""
from types import SimpleNamespace

import numpy as np
import pytest

from utils.landmark_temporal import (COLLAR_FLAG, JC_DROP_PX, JUNCTION_FLAG, describe_flag,
                                     temporal_flags)


def _growing(n=10, collar=(40.0, 300.0), start=60.0, step=12.0):
    """A well-behaved seedling: fixed collar, junction rising straight up from it."""
    return [{"junction": (collar[0], collar[1] - (start + k * step)), "collar": collar} for k in range(n)]


def test_steady_growth_has_no_flags():
    out = temporal_flags(_growing())
    assert [o["flag"] for o in out] == [""] * 10
    assert out[3]["collar_ref"] == pytest.approx((40.0, 300.0))
    assert out[0]["jc_dist"] == pytest.approx(60.0) and out[-1]["jc_dist"] == pytest.approx(168.0)


def test_single_collar_jump_is_flagged_but_slow_drift_is_not():
    pts = _growing()
    pts[5] = dict(pts[5], collar=(40.0, 270.0))          # 30 px off, distance ~120: 0.25*120 = 30 -> use 35
    pts[5]["collar"] = (40.0, 265.0)
    out = temporal_flags(pts)
    assert out[5]["flag"] == COLLAR_FLAG and out[5]["collar_dev"] == pytest.approx(35.0)
    assert all(o["flag"] == "" for i, o in enumerate(out) if i != 5)

    drift = [{"junction": (40.0 + k * 0.1, 300.0 + k * 25 / 70 - (60 + 8 * k)),
              "collar": (40.0 + k * 0.1, 300.0 + k * 25 / 70)} for k in range(70)]
    assert all(o["flag"] == "" for o in temporal_flags(drift))


def test_collar_bound_is_relative_to_the_seedling_size():
    pts = _growing(start=200.0)                            # long hypocotyl: 0.25 * ~250 > 20 px
    pts[4] = dict(pts[4], collar=(40.0, 280.0))            # 20 px off: within the relative bound
    assert temporal_flags(pts)[4]["flag"] == ""


def test_junction_moving_back_toward_the_collar():
    pts = _growing()
    j = pts[6]["junction"]
    pts[6] = dict(pts[6], junction=(j[0], j[1] + 40))      # 28 px below the previous frame's distance
    out = temporal_flags(pts)
    assert out[6]["flag"] == JUNCTION_FLAG
    assert out[7]["flag"] == ""                            # the spike did not lower the reference

    small = _growing(step=0.0)
    small[5] = dict(small[5], junction=(40.0, 300.0 - 60.0 + JC_DROP_PX - 1))   # 4 px back: noise
    assert all(o["flag"] == "" for o in temporal_flags(small))


def test_one_high_spike_does_not_flag_every_later_frame():
    pts = _growing()
    j = pts[3]["junction"]
    pts[3] = dict(pts[3], junction=(j[0], j[1] - 80))      # junction read far too high once
    out = temporal_flags(pts)
    assert [o["flag"] for i, o in enumerate(out) if i > 3] == [""] * 6


def test_manual_frames_are_trusted_and_reset_the_reference():
    pts = _growing()
    pts[5] = dict(pts[5], junction=(40.0, 300.0 - 40.0))   # much shorter than before, but placed by hand
    pts[6] = dict(pts[6], junction=(40.0, 300.0 - 50.0))   # consistent with the manual frame
    manual = [i == 5 for i in range(10)]
    out = temporal_flags(pts, manual)
    assert out[5]["flag"] == ""
    assert out[6]["flag"] == ""                            # judged against the manual distance
    assert temporal_flags(pts)[6]["flag"] == JUNCTION_FLAG # without the manual frame it would be flagged


def test_manual_collar_takes_precedence_in_its_window():
    pts = _growing()
    pts[5] = dict(pts[5], collar=(40.0, 270.0))
    out = temporal_flags(pts, [i == 5 for i in range(10)])
    assert out[6]["collar_ref"] == pytest.approx((40.0, 270.0)) and out[6]["flag"] == COLLAR_FLAG


def test_unreadable_frames_and_short_series():
    pts = _growing(4)
    pts[1] = None
    out = temporal_flags(pts)
    assert out[1] == {"flag": "", "collar_ref": None, "collar_dev": None, "jc_dist": None}
    assert all(o["flag"] == "" for o in temporal_flags(_growing(2)))
    assert temporal_flags([]) == []


def test_missing_junction_keeps_an_absolute_collar_check():
    pts = _growing()
    pts[4] = {"junction": None, "collar": (40.0, 250.0)}
    out = temporal_flags(pts)
    assert out[4]["flag"] == COLLAR_FLAG and out[4]["jc_dist"] is None


def test_describe_flag():
    assert describe_flag("") == ""
    assert "22 px" in describe_flag("collar", 22.4)
    assert "junction moved back" in describe_flag("collar+junction", 30) and "collar jumped" in describe_flag("collar+junction", 30)


# ----------------------------------------------------------------- Gui wiring
def _readout(junction, collar, theta=20.0):
    return {"junction": junction, "collar": collar, "theta": theta, "bio": 180 - theta, "peak": 0.9,
            "hypo_dir": (0.0, 1.0), "cotyl_dir": (0.5, -0.5), "overhook": False}


def _result(readout):
    return {"angle_dict": {1: readout["theta"]}, "angle_list": ['-'], "state_dict": {1: "-"}, "state_list": ['-'],
            "seed_ids": [1], "angle_source": "landmarks", "landmarks": {1: readout}, "traits": {1: {}}}


def test_gui_writes_flags_without_changing_angles():
    import seedling_measurement as sm

    results = [_result(_readout(p["junction"], p["collar"])) for p in _growing(6)]
    results[3]["landmarks"][1]["collar"] = (40.0, 250.0)
    g = sm.Gui.__new__(sm.Gui)
    g.frame_results_by_crop = {0: results}
    g._reconstruct_series_for_crop(0)
    flags = [r["landmarks"][1]["temporal_flag"] for r in results]
    assert flags == ["", "", "", COLLAR_FLAG, "", ""]
    assert all(r["angle_dict"][1] == pytest.approx(160.0) for r in results)    # angles untouched
    assert results[3]["landmarks"][1]["collar_ref"] == pytest.approx((40.0, 300.0))

    # a manual edit of that frame clears its flag on the next reconstruction
    results[3]["state_dict"][1] = "Manual"
    results[3]["angle_dict"][1] = 150.0
    results[3]["landmarks"][1] = dict(_readout((40.0, 300.0 - 96.0), (40.0, 301.0)), edit="manual")
    g._reconstruct_series_for_crop(0)
    assert results[3]["landmarks"][1]["temporal_flag"] == ""


def test_export_row_carries_the_flag_and_reference_collar():
    from utils.export_rows import LANDMARK_COLUMNS, build_export_row

    readout = dict(_readout((40.0, 200.0), (40.0, 250.0)), temporal_flag="collar", collar_ref=(40.0, 300.0),
                   measurable_p=0.83)
    row = build_export_row("f.png", 1, _result(readout), None, None)
    assert row["temporal_flag"] == "collar" and (row["collar_ref_x"], row["collar_ref_y"]) == (40.0, 300.0)
    assert row["measurable_p"] == 0.83
    assert LANDMARK_COLUMNS.index("temporal_flag") == LANDMARK_COLUMNS.index("unreadable") + 2
    assert build_export_row("f.png", 1, _result(_readout((1, 2), (3, 4))), None, None)["temporal_flag"] == ""


def test_kinematics_rings_flagged_points():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    try:
        from ui.kinematics_window import FLAG_LEGEND_LABEL, KinematicsWindow

        results = [_result(_readout(p["junction"], p["collar"])) for p in _growing(4)]
        for r in results:
            r["state_dict"][1] = "Closed"
        results[2]["landmarks"][1]["temporal_flag"] = "junction"
        gui = SimpleNamespace(root=root, file_list=[f"f{i}.png" for i in range(4)], time_deltas=[],
                              germination_detector=None, frame_results_by_crop={0: results},
                              open_seedling_frame=lambda *a, **k: None)
        window = KinematicsWindow(gui)
        scatter = next(iter(window._pick_targets))
        widths = list(scatter.get_linewidths())
        assert len(widths) == 4 and widths[2] > 0 and widths[0] == widths[1] == widths[3] == 0   # only frame 2
        assert FLAG_LEGEND_LABEL in [t.get_text() for t in window.ax.get_legend().get_texts()]
        window.destroy()
    finally:
        root.destroy()
