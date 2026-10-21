"""Landmark readability gate: pure module, ensemble gate fields, GUI series handling, export, env switch."""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from seedling_measurement import Gui
from utils import landmark_gate as lg
from utils.angle_timeseries import UNREADABLE_STATE
from utils.export_rows import LANDMARK_COLUMNS, build_export_row, export_columns

FIT = os.path.join(os.path.dirname(__file__), "..", "multi", "results", "landmark_gate_fit.json")


# ------------------------------------------------------------------ pure module
def test_probability_monotonic_in_each_feature():
    base = dict(peak=0.5, tip_peak=0.5, member_theta_sd=20.0)
    p0 = lg.unreadable_probability(**base)
    assert lg.unreadable_probability(**{**base, "peak": 0.7}) < p0
    assert lg.unreadable_probability(**{**base, "tip_peak": 0.7}) < p0
    assert lg.unreadable_probability(**{**base, "member_theta_sd": 40.0}) > p0
    assert 0.0 < p0 < 1.0


def test_tip_none_uses_fill_and_extremes_do_not_overflow():
    assert lg.unreadable_probability(0.5, None, 10) == lg.unreadable_probability(0.5, lg.TIP_FILL, 10)
    assert lg.unreadable_probability(1000.0, 1000.0, 0.0) == pytest.approx(0.0)
    assert lg.unreadable_probability(-1000.0, -1000.0, 1e6) == pytest.approx(1.0)


def test_threshold_boundary():
    p = lg.unreadable_probability(0.3, 0.3, 50.0)
    assert lg.is_unreadable(0.3, 0.3, 50.0, threshold=p)           # >= is unreadable
    assert not lg.is_unreadable(0.3, 0.3, 50.0, threshold=p + 1e-9)
    assert lg.is_unreadable(0.0, 0.0, 90.0) and not lg.is_unreadable(0.95, 0.9, 0.0)


@pytest.mark.skipif(not os.path.exists(FIT), reason="fit json not present")
def test_constants_match_fit_json():
    with open(FIT) as fh:
        fit = json.load(fh)
    for k, v in fit["weights"].items():
        assert lg.GATE_WEIGHTS[k] == pytest.approx(v, abs=1e-6)
    assert lg.GATE_BIAS == pytest.approx(fit["bias"], abs=1e-6)
    assert lg.GATE_THRESHOLD == pytest.approx(fit["threshold"], abs=1e-6)
    assert lg.TIP_FILL == pytest.approx(fit["fill"]["tip_peak"], abs=1e-6)
    assert lg.MEMBER_SD_FILL == fit["fill"]["member_theta_sd"]


def test_gate_enabled_env():
    assert lg.gate_enabled({})
    assert lg.gate_enabled({"DLHOOK_LANDMARK_GATE": "1"})
    for off in ("0", "false", "OFF", "no"):
        assert not lg.gate_enabled({"DLHOOK_LANDMARK_GATE": off})


def test_annotate_readout():
    assert lg.annotate_readout(None) is None
    r = lg.annotate_readout({"peak": 0.9, "theta": 30.0}, np.full((4, 4), 0.8), [30.0, 40.0, 50.0])
    assert r["tip_peak"] == pytest.approx(0.8) and r["member_theta_sd"] == pytest.approx(np.std([30, 40, 50]))
    assert r["member_thetas"] == [30.0, 40.0, 50.0] and r["unreadable"] is False
    r = lg.annotate_readout({"peak": 0.9, "theta": 30.0}, None, [30.0, None, 50.0])
    assert r["member_theta_sd"] == 90.0 and r["tip_peak"] is None
    r = lg.annotate_readout({"peak": 0.9, "theta": 30.0})
    assert r["member_theta_sd"] == 0.0 and r["member_thetas"] is None


# ------------------------------------------------------------------ ensemble
def _fields(peak, vec_dir, tip=None, size=16):
    hm = np.zeros((size, size), np.float32)
    hm[8, 8] = peak
    vec = np.zeros((4, size, size), np.float32)
    vec[0], vec[1] = vec_dir[0]
    vec[2], vec[3] = vec_dir[1]
    extra = None
    if tip is not None:
        t = np.zeros((size, size), np.float32)
        t[2, 8] = tip
        z = np.zeros((size, size), np.float32)
        extra = {"collar_hm": z, "root_vec": np.zeros((2, size, size), np.float32), "base_hm": z, "tip_hm": t}
    return hm, vec, extra


class _Member:
    def __init__(self, fields):
        self.fields = fields

    def predict_landmark_fields(self, images, flip_tta=None):
        return [self.fields for _ in images]


def test_ensemble_gate_keys_and_sd():
    from models.multiclass_inference import LandmarkEnsemble
    a = _fields(0.9, ((0, 1), (1, 0)))
    c = _fields(0.9, ((0, 1), (0, -1)))                # a different theta
    out = LandmarkEnsemble([_Member(a), _Member(a), _Member(c)]).predict_landmarks([object()])[0]
    assert out["member_thetas"][0] == pytest.approx(out["member_thetas"][1])
    assert out["member_theta_sd"] == pytest.approx(np.std(out["member_thetas"]))
    assert out["member_theta_sd"] > 10
    assert out["tip_peak"] is None                     # no member has a tip head
    assert out["unreadable_p"] == pytest.approx(
        lg.unreadable_probability(out["peak"], None, out["member_theta_sd"]))
    assert out["unreadable"] == (out["unreadable_p"] >= lg.GATE_THRESHOLD)


def test_ensemble_failed_member_gives_sd_90_and_none_stays_none():
    from models.multiclass_inference import LandmarkEnsemble
    good = _fields(0.9, ((0, 1), (1, 0)))
    dead = _fields(0.05, ((0, 1), (1, 0)))             # below the peak threshold: its own readout is None
    out = LandmarkEnsemble([_Member(good), _Member(good), _Member(dead)]).predict_landmarks([object()])[0]
    assert out is not None and out["member_theta_sd"] == 90.0
    assert out["member_thetas"][2] is None and out["member_thetas"][0] is not None
    none = _fields(0.01, ((0, 1), (1, 0)))
    assert LandmarkEnsemble([_Member(none), _Member(none)]).predict_landmarks([object()]) == [None]


def test_ensemble_tip_peak_from_merged_tip_heatmap():
    from models.multiclass_inference import LandmarkEnsemble
    with_tip = _fields(0.9, ((0, 1), (1, 0)), tip=0.8)
    plain = _fields(0.9, ((0, 1), (1, 0)))
    out = LandmarkEnsemble([_Member(with_tip), _Member(plain)]).predict_landmarks([object()])[0]
    assert out is not None and out["tip_peak"] == pytest.approx(0.8)


# ------------------------------------------------------------------ GUI series
def _readout(theta, peak=0.9, **kw):
    r = {"junction": (30.5, 20.5), "theta": theta, "bio": 180 - theta, "hypo_dir": (0.0, 1.0),
         "cotyl_dir": (0.5, -0.5), "peak": peak, "overhook": False, "unreadable": False,
         "unreadable_p": 0.1, "edit": "model"}
    r.update(kw)
    return r


def _result(seed_id, readout, gated=False):
    if gated:
        readout = dict(readout, unreadable=True, unreadable_p=0.83, gated=True, gated_theta=readout["theta"])
    theta = float("nan") if (gated or not readout) else readout["theta"]
    return {"hook": None, "angle_dict": {seed_id: theta}, "angle_list": ['-'], "state_dict": {seed_id: "-"},
            "state_list": ['-'], "seed_ids": [seed_id], "angle_source": "landmarks",
            "landmarks": {seed_id: readout}, "traits": {seed_id: {}}}


def _gui(results):
    g = Gui.__new__(Gui)
    g.frame_results_by_crop = {0: results}
    return g


def test_gated_frames_nan_unreadable_and_not_interpolated():
    results = [_result(1, _readout(10)), _result(1, _readout(50), gated=True), _result(1, _readout(30))]
    _gui(results)._reconstruct_series_for_crop(0)
    assert math.isnan(results[1]["angle_dict"][1])                 # NOT 160 (interpolated)
    assert results[1]["state_dict"][1] == UNREADABLE_STATE
    assert math.isnan(results[1]["raw_angle_dict"][1])
    assert results[1]["landmarks"][1]["gated_theta"] == 50
    assert results[0]["state_dict"][1] == "Closed" and results[0]["angle_dict"][1] == pytest.approx(170)
    assert results[1]["state_list"] == [UNREADABLE_STATE] and results[1]["angle_list"] == ['-']


def test_gap_with_gated_frame_is_not_filled_even_for_plain_missing_neighbours():
    results = [_result(1, _readout(10)), _result(1, None), _result(1, _readout(50), gated=True),
               _result(1, _readout(30))]
    _gui(results)._reconstruct_series_for_crop(0)
    assert math.isnan(results[1]["angle_dict"][1]) and math.isnan(results[2]["angle_dict"][1])


def test_gated_series_idempotent():
    results = [_result(1, _readout(10)), _result(1, _readout(50), gated=True), _result(1, _readout(30))]
    g = _gui(results)
    g._reconstruct_series_for_crop(0)
    snap = [(r["angle_dict"][1], r["state_dict"][1], dict(r["raw_angle_dict"])) for r in results]
    g._reconstruct_series_for_crop(0)
    again = [(r["angle_dict"][1], r["state_dict"][1], dict(r["raw_angle_dict"])) for r in results]
    assert repr(snap) == repr(again)


def test_manual_anchor_on_gated_frame_is_kept():
    results = [_result(1, _readout(10)), _result(1, _readout(50), gated=True), _result(1, _readout(30))]
    results[1]["angle_dict"][1] = 123.0
    results[1]["state_dict"][1] = "Manual"
    results[1]["landmarks"][1] = {"theta": 57.0, "bio": 123.0, "edit": "manual"}
    g = _gui(results)
    g._reconstruct_series_for_crop(0)
    g._reconstruct_series_for_crop(0)
    assert results[1]["angle_dict"][1] == 123.0 and results[1]["state_dict"][1] == "Manual"


def _frame_gui():
    g = Gui.__new__(Gui)
    g.cotyl_time_series_by_crop, g.germ_time_series_by_crop = {}, {}
    g._seed_point = lambda *a, **k: None
    g._trait_mask = lambda *a, **k: None
    return g


def test_process_landmark_frame_gates_and_env_override(monkeypatch):
    readout = _readout(40.0, unreadable=True, unreadable_p=0.83)
    monkeypatch.delenv("DLHOOK_LANDMARK_GATE", raising=False)
    res = _frame_gui()._process_landmark_frame("0-crop-a.png", 0, None, readout, None, None, None)
    assert math.isnan(res["angle_dict"][1])
    lm = res["landmarks"][1]
    assert lm["gated"] is True and lm["gated_theta"] == 40.0 and lm["edit"] == "model" and lm["unreadable_p"] == 0.83
    monkeypatch.setenv("DLHOOK_LANDMARK_GATE", "0")
    res = _frame_gui()._process_landmark_frame("0-crop-a.png", 0, None, readout, None, None, None)
    assert res["angle_dict"][1] == 40.0 and not res["landmarks"][1].get("gated")
    monkeypatch.delenv("DLHOOK_LANDMARK_GATE")
    ok = _frame_gui()._process_landmark_frame("0-crop-a.png", 0, None, _readout(40.0), None, None, None)
    assert ok["angle_dict"][1] == 40.0


# ------------------------------------------------------------------ export
def test_export_columns_order_and_values():
    cols = export_columns(None)
    assert cols.index("unreadable_p") == cols.index("overhook_flag") + 1
    assert cols.index("unreadable") == cols.index("unreadable_p") + 1
    assert cols[:7] == ['img_name', 'seedling_id', 'raw_angle', 'state', 'bio_angle',
                        'time_elapsed_min', 'germination_frame']
    assert {"unreadable_p", "unreadable"} <= set(LANDMARK_COLUMNS)
    results = [_result(1, _readout(10)), _result(1, _readout(50), gated=True), _result(1, _readout(30))]
    _gui(results)._reconstruct_series_for_crop(0)
    row = build_export_row("f.png", 1, results[1], 1.0, None)
    assert row["bio_angle"] == '' and row["state"] == UNREADABLE_STATE and row["raw_angle"] == ''
    assert row["unreadable"] == 1 and row["unreadable_p"] == 0.83
    row = build_export_row("f.png", 1, results[0], 1.0, None)
    assert row["unreadable"] == 0 and row["unreadable_p"] == 0.1
