"""Landmark angle source wiring in the Gui (bare Gui.__new__, no Tk): routing, per-frame result,
series reconstruction (idempotency, manual anchors), germination-unknown, export row helper."""
from __future__ import annotations

import math
import os

import cv2
import numpy as np
import pytest

import seedling_measurement as sm
from models import segmentation_backends
from seedling_measurement import Gui
from utils.export_rows import BASE_COLUMNS, LANDMARK_COLUMNS, TRAIT_COLUMNS, MM_COLUMNS, export_columns, build_export_row
from utils.germination_detector import GerminationDetector
from utils.mask_store import MaskStore


def _readout(theta, peak=0.9, **kw):
    r = {"junction": (30.5, 20.5), "theta": theta, "bio": 180 - theta, "bio_overhook": 180 - theta,
         "hypo_dir": (0.0, 1.0), "cotyl_dir": (0.5, -0.5), "peak": peak, "cotyl_path": "base_tip",
         "overhook": False, "tip": (40.0, 5.0), "base": (31.0, 15.0)}
    r.update(kw)
    return r


def _result(seed_id, readout, source="landmarks"):
    theta = readout["theta"] if readout else float("nan")
    return {"hook": None, "angle_dict": {seed_id: theta}, "angle_list": ['-'], "state_dict": {seed_id: "-"},
            "state_list": ['-'], "seed_ids": [seed_id], "cotyl_mask": None, "hypo_mask": None,
            "germ_mask": None, "angle_source": source, "landmarks": {seed_id: readout}, "traits": {seed_id: {}}}


def _gui_with(crop_id, results):
    g = Gui.__new__(Gui)
    g.frame_results_by_crop = {crop_id: results}
    return g


# ---------------------------------------------------------------- reconstruction
def test_landmark_branch_uses_landmark_reconstruction_not_branch_dp(monkeypatch):
    calls = {"lm": 0}
    real = sm.reconstruct_landmark_series_detail

    def spy(readouts, *a, **k):
        calls["lm"] += 1
        return real(readouts, *a, **k)

    monkeypatch.setattr(sm, "reconstruct_landmark_series_detail", spy)
    monkeypatch.setattr(sm, "reconstruct_series", lambda *a, **k: pytest.fail("ellipse DP used"))
    results = [_result(1, _readout(t)) for t in (10, 12, 14)]
    g = _gui_with(0, results)
    g._reconstruct_series_for_crop(0)
    assert calls["lm"] == 1
    assert [r["angle_dict"][1] for r in results] == pytest.approx([170, 168, 166])   # bio = 180 - theta
    assert all(r["state_dict"][1] == "Closed" for r in results)
    assert [r["angle_list"] for r in results] == [[170], [168], [166]]
    assert [r["state_list"] for r in results] == [["Closed"]] * 3


def test_none_readout_is_nan_and_short_gap_interpolated():
    results = [_result(1, _readout(10)), _result(1, None), _result(1, _readout(30))]
    g = _gui_with(0, results)
    g._reconstruct_series_for_crop(0)
    assert results[1]["angle_dict"][1] == pytest.approx(160.0)       # gap of 1 between 170 and 150
    assert math.isnan(results[1]["raw_angle_dict"][1])
    # a long gap stays NaN
    results = [_result(1, _readout(10))] + [_result(1, None) for _ in range(5)]
    g = _gui_with(0, results)
    g._reconstruct_series_for_crop(0)
    assert all(math.isnan(r["angle_dict"][1]) for r in results[1:])
    assert all(r["state_dict"][1] for r in results)


def test_idempotent_and_raw_captured_once():
    results = [_result(1, _readout(t)) for t in (10, 20, 30, 40)]
    g = _gui_with(0, results)
    g._reconstruct_series_for_crop(0)
    first = [(r["angle_dict"][1], r["state_dict"][1]) for r in results]
    raw = [dict(r["raw_angle_dict"]) for r in results]
    g._reconstruct_series_for_crop(0)
    assert [(r["angle_dict"][1], r["state_dict"][1]) for r in results] == first
    assert [r["raw_angle_dict"] for r in results] == raw
    assert [r["raw_angle_dict"][1] for r in results] == [10, 20, 30, 40]


def test_manual_frames_are_kept_and_anchor_the_series():
    results = [_result(1, _readout(t)) for t in (10, 20, 30, 40)]
    results[1]["angle_dict"][1] = 123.0
    results[1]["state_dict"][1] = "Manual"
    g = _gui_with(0, results)
    g._reconstruct_series_for_crop(0)
    assert results[1]["angle_dict"][1] == 123.0
    assert results[1]["state_dict"][1] == "Manual"
    assert "raw_angle_dict" not in results[1]
    assert results[0]["angle_dict"][1] == pytest.approx(170)
    g._reconstruct_series_for_crop(0)
    assert results[1]["angle_dict"][1] == 123.0


def test_low_peak_frame_counts_as_missing():
    results = [_result(1, _readout(10)), _result(1, _readout(90, peak=0.05)), _result(1, _readout(30))]
    _gui_with(0, results)._reconstruct_series_for_crop(0)
    assert results[1]["angle_dict"][1] == pytest.approx(160.0)       # interpolated, not 90


def test_ellipse_series_still_uses_ellipse_branch(monkeypatch):
    seen = {}

    def fake(angles, is_manual=None):
        seen["angles"] = list(angles)
        return [float(a) for a in angles], ["Closed"] * len(angles)

    monkeypatch.setattr(sm, "reconstruct_series", fake)
    monkeypatch.setattr(sm, "reconstruct_landmark_series_detail", lambda *a, **k: pytest.fail("landmark path used"))
    results = [{"angle_dict": {1: 20.0}, "state_dict": {1: "-"}, "seed_ids": [1], "angle_list": [20],
                "state_list": ["-"], "angle_source": "ellipse"}]
    _gui_with(0, results)._reconstruct_series_for_crop(0)
    assert seen["angles"] == [160.0]


# ---------------------------------------------------------------- routing / store
class _FakeStore(MaskStore):
    pass


class _FakeBackend:
    def predict_into(self, store, paths):
        pass


class _FakeRunner:
    def __init__(self):
        self.calls = []

    def predict_paths(self, paths):
        self.calls.append(list(paths))
        return {os.path.basename(p): _readout(15) for p in paths}


@pytest.fixture
def fake_backend(monkeypatch):
    monkeypatch.setattr(sm, "get_backend", lambda name: _FakeBackend())
    monkeypatch.setattr(sm, "resolve_backend_name", lambda: "multiclass")


def test_segment_paths_fills_landmark_store(monkeypatch, fake_backend):
    runner = _FakeRunner()
    monkeypatch.setattr(sm, "resolve_angle_source", lambda: "landmarks")
    monkeypatch.setattr(sm, "get_landmark_runner", lambda: runner)
    g = Gui.__new__(Gui)           # no landmark_store attribute at all
    g.mask_store = MaskStore()
    g._segment_paths(["data/images/0-crop-a.png", "data/images/1-crop-a.png"])
    assert set(g.landmark_store) == {"0-crop-a.png", "1-crop-a.png"}
    assert g.landmark_store["0-crop-a.png"]["theta"] == 15


def test_segment_paths_records_no_reading_for_an_unreadable_crop(monkeypatch, fake_backend):
    class Partial(_FakeRunner):
        def predict_paths(self, paths):
            return {os.path.basename(p): _readout(15) for p in paths if "bad" not in p}

    monkeypatch.setattr(sm, "resolve_angle_source", lambda: "landmarks")
    monkeypatch.setattr(sm, "get_landmark_runner", lambda: Partial())
    g = Gui.__new__(Gui)
    g.mask_store = MaskStore()
    g.landmark_store = {"0-crop-old.png": _readout(20)}
    g._segment_paths(["data/images/0-crop-a.png", "data/images/0-crop-bad.png"])
    assert g.landmark_store["0-crop-bad.png"] is None              # a NaN frame, so frame indices stay aligned
    assert g.landmark_store["0-crop-a.png"]["theta"] == 15
    assert g.landmark_store["0-crop-old.png"]["theta"] == 20        # other crops' readings untouched


def test_segment_paths_leaves_store_alone_for_ellipse(monkeypatch, fake_backend):
    monkeypatch.setattr(sm, "resolve_angle_source", lambda: "ellipse")
    monkeypatch.setattr(sm, "get_landmark_runner", lambda: pytest.fail("runner used"))
    g = Gui.__new__(Gui)
    g.mask_store = MaskStore()
    g.landmark_store = {}
    g._segment_paths(["data/images/0-crop-a.png"])
    assert g.landmark_store == {}


def test_segment_paths_survives_runner_failure(monkeypatch, fake_backend, caplog):
    class Boom:
        def predict_paths(self, paths):
            raise RuntimeError("no gpu")

    monkeypatch.setattr(sm, "resolve_angle_source", lambda: "landmarks")
    monkeypatch.setattr(sm, "get_landmark_runner", lambda: Boom())
    g = Gui.__new__(Gui)
    g.mask_store = MaskStore()
    g.landmark_store = {"keep": 1}
    g._segment_paths(["data/images/0-crop-a.png"])
    assert g.landmark_store == {"keep": 1}
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_reset_frame_accumulators_clears_per_crop():
    g = Gui.__new__(Gui)
    g.cotyl_time_series_by_crop = {0: [1], 10: [1]}
    g.germ_time_series_by_crop = {0: [1], 10: [1]}
    g.landmark_store = {"0-crop-a.png": 1, "0-crop-b.png": 2, "1-crop-a.png": 3, "10-crop-a.png": 4}
    g.reset_frame_accumulators(1)
    assert set(g.landmark_store) == {"0-crop-a.png", "0-crop-b.png", "10-crop-a.png"}
    g.reset_frame_accumulators(0)
    assert set(g.landmark_store) == {"10-crop-a.png"}
    g.reset_frame_accumulators()
    assert g.landmark_store == {}


def test_mask_store_defaults_cover_trait_mask():
    ms = MaskStore()
    ms.put_raw("0-crop-a.png", "hypocotyl_binary", np.zeros((3, 3), np.uint8))
    ms.discard_raw(["0-crop-a.png"])
    assert ms.get("0-crop-a.png", "hypocotyl_binary") is None
    ms.put_raw("0-crop-a.png", "hypocotyl_binary", np.zeros((3, 3), np.uint8))
    ms.clear_crop(0)
    assert ms.get("0-crop-a.png", "hypocotyl_binary") is None


# ---------------------------------------------------------------- process_single_frame
def _bare_gui_for_frames(tmp_path, monkeypatch, with_masks):
    os.makedirs(tmp_path / "data" / "images")
    img = np.full((120, 60, 3), 90, np.uint8)
    cv2.imwrite(str(tmp_path / "data" / "images" / "0-crop-a.png"), img)
    monkeypatch.chdir(tmp_path)
    g = Gui.__new__(Gui)
    g.mask_store = MaskStore()
    g.cotyl_time_series_by_crop = {}
    g.germ_time_series_by_crop = {}
    g.crop_boxes = [{"half_w": 30, "half_h": 60}]
    g.landmark_store = {"0-crop-a.png": _readout(25, collar=(30.0, 60.0))}
    if with_masks:
        bar = np.zeros((120, 60), np.uint8)
        bar[20:100, 25:35] = 255
        g.mask_store.put_raw("0-crop-a.png", "1", bar.copy())
        g.mask_store.put_raw("0-crop-a.png", "2", bar.copy())
        g.mask_store.put_raw("0-crop-a.png", "hypocotyl_binary", bar.copy())
    return g


def test_process_single_frame_landmark_mode_without_masks(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "resolve_angle_source", lambda: "landmarks")
    g = _bare_gui_for_frames(tmp_path, monkeypatch, with_masks=False)
    r = g.process_single_frame("0-crop-a.png", 0)
    assert r is not None
    assert r["angle_source"] == "landmarks" and r["seed_ids"] == [1]
    assert r["angle_dict"][1] == 25.0                    # raw theta, 0 = closed
    assert r["hook"] is None and r["cotyl_mask"] is None and r["hypo_mask"] is None and r["germ_mask"] is None
    assert r["landmarks"][1]["edit"] == "model" and r["landmarks"][1]["theta"] == 25
    assert math.isnan(r["traits"][1]["length_px"])
    assert g.germ_time_series_by_crop[0] == [[]]


def test_process_single_frame_landmark_mode_traits_and_none_readout(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "resolve_angle_source", lambda: "landmarks")
    g = _bare_gui_for_frames(tmp_path, monkeypatch, with_masks=True)
    r = g.process_single_frame("0-crop-a.png", 0)
    assert r["traits"][1]["length_px"] == pytest.approx(40, abs=4)   # junction (30,20) to collar (30,60)
    assert r["traits"][1]["diameter_px"] == pytest.approx(10, abs=1.5)
    assert r["cotyl_mask"] is not None and r["hypo_mask"] is not None

    g.landmark_store["0-crop-a.png"] = None               # model found nothing: still the landmark source
    r2 = g.process_single_frame("0-crop-a.png", 0, frame_index=0)
    assert r2["angle_source"] == "landmarks" and math.isnan(r2["angle_dict"][1])
    assert r2["landmarks"][1] is None
    assert len(g.germ_time_series_by_crop[0]) == 1        # replaced in place, not appended


def test_process_single_frame_ellipse_when_frame_has_no_readout(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "resolve_angle_source", lambda: "landmarks")
    g = _bare_gui_for_frames(tmp_path, monkeypatch, with_masks=False)
    g.landmark_store = {}
    assert g.process_single_frame("0-crop-a.png", 0) is None     # masks missing -> legacy None


# ---------------------------------------------------------------- germination without "4"
def test_germination_unknown_without_radicle_mask(monkeypatch, caplog):
    monkeypatch.setattr(sm, "resolve_germination_method", lambda: "rule")
    g = Gui.__new__(Gui)
    g.germination_detector = None
    g.frame_results_by_crop = {0: [_result(1, _readout(10)), _result(1, _readout(20))]}
    g.cropped_filenames_by_crop = {}
    g.germ_time_series_by_crop = {0: [[], []]}
    g._ensure_germination_detected(0)
    g._ensure_germination_detected(0)
    assert g.germination_detector.get_time_zero(0) is None
    assert sum(r.levelname == "WARNING" for r in caplog.records) == 1


def test_germination_rule_still_runs_with_radicle_mask(monkeypatch):
    monkeypatch.setattr(sm, "resolve_germination_method", lambda: "rule")
    called = {}

    class Det(GerminationDetector):
        def detect(self, *a, **k):
            called["yes"] = True
            return None

    class Dbg:
        def get(self):
            return 0

    g = Gui.__new__(Gui)
    g.germination_detector = Det()
    res = _result(1, _readout(10))
    res["germ_mask"] = np.zeros((4, 4), np.uint8)
    g.frame_results_by_crop = {0: [res]}
    g.cropped_filenames_by_crop = {}
    g.germ_time_series_by_crop = {0: [[]]}
    g.crop_boxes = [{"half_w": 10, "half_h": 10}]
    g.debug_var = Dbg()
    g._ensure_germination_detected(0)
    assert called.get("yes")


# ---------------------------------------------------------------- export helper
def test_export_columns_old_first_and_mm_only_with_scale():
    base = export_columns(None)
    assert base[:7] == BASE_COLUMNS == ['img_name', 'seedling_id', 'raw_angle', 'state', 'bio_angle',
                                        'time_elapsed_min', 'germination_frame']
    assert base[7:] == LANDMARK_COLUMNS + TRAIT_COLUMNS
    assert base[7] == 'angle_source' and base[-1] == 'hypocotyl_diameter_px'
    assert not set(MM_COLUMNS) & set(base)
    assert export_columns(10.0) == base + MM_COLUMNS


def test_build_export_row_landmark_and_empty():
    res = _result(1, dict(_readout(25, collar=(30.0, 60.0), root_dir=(0.0, 1.0)), edit="model"))
    res["angle_dict"][1] = 150.0
    res["raw_angle_dict"] = {1: 25.0}
    res["state_dict"][1] = "Closed"
    res["traits"][1] = {"length_px": 40.0, "length_truncated": False, "curvature_mean": 0.01,
                        "curvature_max": 0.02, "diameter_px": 10.0, "n_transects": 7}
    row = build_export_row("f.png", 1, res, 12.345, 3)
    assert list(row) == export_columns(None)
    assert (row['raw_angle'], row['bio_angle'], row['state'], row['time_elapsed_min'], row['germination_frame']) == \
        (25, 150, "Closed", 12.35, 3)
    assert row['angle_source'] == "landmarks" and row['landmark_edit'] == "model"
    assert (row['junction_x'], row['junction_y']) == (30.5, 20.5)
    assert (row['collar_x'], row['collar_y']) == (30.0, 60.0)
    assert row['overhook_flag'] == 0 and row['landmark_peak'] == 0.9
    assert row['hypocotyl_length_px'] == 40.0 and row['hypocotyl_length_truncated'] is False
    mm = build_export_row("f.png", 1, res, 12.345, None, um_per_px=10.0)
    assert list(mm) == export_columns(10.0)
    assert mm['hypocotyl_length_mm'] == pytest.approx(0.4) and mm['germination_frame'] == ''
    assert mm['hypocotyl_curvature_mean_per_mm'] == pytest.approx(1.0)

    empty = build_export_row("f.png", 2, None, None, None, um_per_px=10.0)
    assert list(empty) == export_columns(10.0)
    assert all(empty[c] == '' for c in export_columns(10.0)[2:]) and empty['img_name'] == "f.png"


def test_build_export_row_ellipse_result_has_empty_landmark_cells():
    res = {"angle_dict": {1: 140.4}, "raw_angle_dict": {1: 39.6}, "state_dict": {1: "Opening"},
           "seed_ids": [1], "angle_source": "ellipse", "traits": {}}
    row = build_export_row("f.png", 1, res, 0.0, None)
    assert row['angle_source'] == "ellipse" and row['bio_angle'] == 140 and row['raw_angle'] == 40
    assert all(row[c] == '' for c in LANDMARK_COLUMNS[1:] + TRAIT_COLUMNS)


def test_series_edit_is_stored_and_rederived(monkeypatch):
    results = [_result(1, _readout(t)) for t in (10, 11, 12, 160, 13, 14, 15)]
    g = _gui_with(0, results)
    g._reconstruct_series_for_crop(0)
    edits = [r["landmarks"][1]["series_edit"] for r in results]
    assert edits[3] == "rejected" and edits.count("rejected") == 1
    assert results[3]["angle_dict"][1] == pytest.approx(167.0, abs=2)      # filled from its neighbours
    assert results[3]["raw_angle_dict"][1] == 160                           # the reading itself is kept
    g._reconstruct_series_for_crop(0)                                       # idempotent
    assert [r["landmarks"][1]["series_edit"] for r in results] == edits


def test_export_has_series_edit():
    from utils.export_rows import build_export_row, export_columns
    assert "series_edit" in export_columns()
    r = _result(1, dict(_readout(10), series_edit="rejected"))
    assert build_export_row("f.png", 1, r, None, None)["series_edit"] == "rejected"
