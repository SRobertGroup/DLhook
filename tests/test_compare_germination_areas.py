"""Tests for multi/compare_germination_areas.py, the read-only three-arm
(germ_v1 / germ_v2 / multiclass) germination-frame comparison used to gate
flipping the GUI's default segmentation backend.

House style: synthesise everything in-test, no real weights, no real
images, no GPU. Predictors are stubbed exactly like
tests/test_segmentation_backends.py stubs MulticlassInference."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_MULTI_DIR = str(Path(__file__).resolve().parent.parent / "multi")
if _MULTI_DIR not in sys.path:
    sys.path.insert(0, _MULTI_DIR)

import compare_germination_areas as cga  # noqa: E402


# --- contour recipe -----------------------------------------------------------

def test_germ_contour_recipe_extracts_known_blob_area():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 20, 255, -1)

    contours = cga.germ_contours_from_mask(mask)

    assert len(contours) == 1
    # A filled circle of radius 20 has area pi*20^2 ~= 1257; cv2's
    # rasterized contour area is close but not exact -- allow slack.
    assert abs(cv2.contourArea(contours[0]) - np.pi * 20 ** 2) < 150


def test_germ_contour_recipe_keeps_small_blob_with_fewer_than_5_points():
    """The 5-point floor (a cv2.fitEllipse prerequisite) belongs to the
    cotyledon/hypocotyl angle path only -- seedling_measurment.py:1067's
    comment is explicit that germ contours are never floored. A 2x2 pixel
    square rasterizes to a 4-point contour; it must survive here."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:12, 10:12] = 255

    contours = cga.germ_contours_from_mask(mask)

    assert len(contours) == 1
    assert len(contours[0]) < 5
    assert cv2.contourArea(contours[0]) >= 0  # a 2x2 square has near-zero cv2 contour area, but exists


def test_germ_contour_recipe_none_mask_is_no_contours():
    assert cga.germ_contours_from_mask(None) == []


def test_germ_contour_recipe_applies_threshold_at_127():
    """Values above 127 count as foreground even if not already exactly
    255 -- cv2.threshold(img_germ, 127, 255, THRESH_BINARY), matching
    seedling_measurment.py:1033 exactly."""
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:25, 5:25] = 200  # above 127, below 255

    contours = cga.germ_contours_from_mask(mask)

    assert len(contours) == 1
    assert cv2.contourArea(contours[0]) > 0


# --- multiclass class extraction ----------------------------------------------

def test_class_index_mask_extracts_exactly_class_3_and_nothing_else():
    label_map = np.array([
        [0, 1, 2, 3],
        [3, 2, 1, 0],
        [1, 0, 3, 2],
    ], dtype=np.uint8)

    mask = cga.class_index_mask(label_map, 3)

    assert mask.dtype == np.uint8
    np.testing.assert_array_equal(mask, (label_map == 3).astype(np.uint8) * 255)
    # Every other class must be entirely absent from the extracted mask.
    for other_class in (0, 1, 2):
        assert not np.any(mask[label_map == other_class])
    assert np.count_nonzero(mask) == np.count_nonzero(label_map == 3)


class _FakeMulticlassInference:
    """Stands in for models.multiclass_inference.MulticlassInference:
    records the geometry it was constructed with and returns a
    test-supplied label map per file instead of running real inference."""

    label_maps_to_return = {}
    construct_calls = []

    def __init__(self, checkpoint_path, num_classes=None, in_size=None, out_size=None, margin=None):
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes
        self.in_size = in_size
        self.out_size = out_size
        self.margin = margin
        type(self).construct_calls.append(
            dict(checkpoint_path=checkpoint_path, num_classes=num_classes,
                 in_size=in_size, out_size=out_size, margin=margin)
        )

    def predict_files_labelmaps(self, image_paths):
        return type(self).label_maps_to_return


def test_run_multiclass_arm_extracts_exactly_class_3(monkeypatch):
    label_map = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    _FakeMulticlassInference.label_maps_to_return = {"frame.png": label_map}
    _FakeMulticlassInference.construct_calls = []
    monkeypatch.setattr(cga, "MulticlassInference", _FakeMulticlassInference)

    result = cga.run_multiclass_arm("fake_checkpoint.pt", ["frame.png"])

    np.testing.assert_array_equal(result["frame.png"], (label_map == 3).astype(np.uint8) * 255)
    assert np.count_nonzero(result["frame.png"]) == 1  # exactly one class-3 pixel

    # Pinned training geometry (264/252/MARGIN), not UNetInference's
    # 572/560/6 live-GUI default -- same reasoning as
    # test_segmentation_backends.py's MulticlassBackend geometry pin.
    call = _FakeMulticlassInference.construct_calls[0]
    assert call["in_size"] == cga.MULTICLASS_IN_SIZE == 264
    assert call["out_size"] == cga.MULTICLASS_OUT_SIZE == 252
    assert call["margin"] == cga.MARGIN
    assert call["num_classes"] == 4


class _FakeBinaryPredictor:
    def __init__(self, weight_path, masks_to_return):
        self.weight_path = weight_path
        self._masks_to_return = masks_to_return
        self.predict_files_calls = []

    def predict_files(self, image_paths, label="0"):
        self.predict_files_calls.append((tuple(image_paths), label))
        return self._masks_to_return


def test_run_binary_arm_requests_label_4(monkeypatch):
    fake_mask = {"a.png": np.zeros((10, 10), dtype=np.uint8)}
    predictor = _FakeBinaryPredictor("weights/RootPainter_weights/germ_v1.pkl", fake_mask)
    monkeypatch.setattr(cga, "get_predictor", lambda path: predictor)

    result = cga.run_binary_arm("weights/RootPainter_weights/germ_v1.pkl", ["a.png"])

    assert result is fake_mask
    assert predictor.predict_files_calls == [(("a.png",), "4")]


# --- seed point proxy -----------------------------------------------------------

def test_seed_point_bottom_center():
    box = {"half_w": 30, "half_h": 90}
    assert cga.seed_point_for_box(box, "bottom-center") == (30.0, 179.0)


def test_seed_point_center():
    box = {"half_w": 30, "half_h": 90}
    assert cga.seed_point_for_box(box, "center") == (30.0, 90.0)


def test_seed_point_unknown_mode_raises():
    box = {"half_w": 30, "half_h": 90}
    try:
        cga.seed_point_for_box(box, "top-left")
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- three-arm comparison summary -----------------------------------------------

def test_compare_arms_classifies_exact_within_1_differ_and_both_none():
    detected = {
        "germ_v1": {1: 8, 2: 5, 3: None, 4: 20},
        "germ_v2": {1: 8, 2: 6, 3: None, 4: 10},
        "multiclass": {1: 8, 2: 6, 3: None, 4: 20},
    }

    summary = cga.compare_arms(detected)

    v1_v2 = summary[("germ_v1", "germ_v2")]
    assert v1_v2["agree_exact"] == 1     # seedling 1: 8 == 8
    assert v1_v2["agree_within_1"] == 1  # seedling 2: |5-6| == 1
    assert v1_v2["differ"] == 1          # seedling 4: |20-10| == 10
    assert v1_v2["both_none"] == 1       # seedling 3: both None
    assert v1_v2["one_none_seedlings"] == []

    v2_mc = summary[("germ_v2", "multiclass")]
    assert v2_mc["agree_exact"] == 2     # seedling 1 (8==8) and seedling 2 (6==6)
    assert v2_mc["agree_within_1"] == 0
    assert v2_mc["differ"] == 1          # seedling 4: |10-20| == 10
    assert v2_mc["both_none"] == 1       # seedling 3


def test_compare_arms_flags_one_arm_detects_other_does_not():
    detected = {
        "germ_v1": {1: None},
        "germ_v2": {1: 5},
        "multiclass": {1: 5},
    }

    summary = cga.compare_arms(detected)

    assert summary[("germ_v1", "germ_v2")]["one_none_seedlings"] == [1]
    assert summary[("germ_v1", "multiclass")]["one_none_seedlings"] == [1]
    assert summary[("germ_v2", "multiclass")]["one_none_seedlings"] == []
    assert summary[("germ_v2", "multiclass")]["agree_exact"] == 1


def test_classify_pair_all_categories():
    assert cga.classify_pair(None, None) == "both_none"
    assert cga.classify_pair(None, 5) == "one_none"
    assert cga.classify_pair(5, None) == "one_none"
    assert cga.classify_pair(5, 5) == "agree_exact"
    assert cga.classify_pair(5, 6) == "agree_within_1"
    assert cga.classify_pair(6, 5) == "agree_within_1"
    assert cga.classify_pair(5, 8) == "differ"


def test_overall_agreement_within_1_true_when_every_seedling_agrees():
    detected = {
        "germ_v1": {1: 8, 2: 20},
        "germ_v2": {1: 9, 2: 21},
        "multiclass": {1: 8, 2: 20},
    }

    all_agree, checked, disagreeing, incomplete = cga.overall_agreement_within_1(detected)

    assert all_agree is True
    assert checked == [1, 2]
    assert disagreeing == []
    assert incomplete == []


def test_overall_agreement_within_1_false_and_reports_disagreeing_seedling():
    detected = {
        "germ_v1": {1: 8, 2: 5},
        "germ_v2": {1: 9, 2: 40},
        "multiclass": {1: 8, 2: 6},
    }

    all_agree, checked, disagreeing, incomplete = cga.overall_agreement_within_1(detected)

    assert all_agree is False
    assert disagreeing == [2]
    assert checked == [1, 2]
    assert incomplete == []


def test_overall_agreement_within_1_excludes_incomplete_seedlings():
    detected = {
        "germ_v1": {1: 8, 2: None},
        "germ_v2": {1: 9, 2: 5},
        "multiclass": {1: 8, 2: 5},
    }

    all_agree, checked, disagreeing, incomplete = cga.overall_agreement_within_1(detected)

    assert incomplete == [2]
    assert checked == [1]
    assert all_agree is True  # seedling 1 alone still agrees within 1
