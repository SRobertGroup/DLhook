import cv2
import numpy as np

from utils.germination_detector import (
    AREA_THRESHOLD_FRACTION,
    CONFIRM_HITS,
    CONFIRM_WINDOW,
    GerminationDetector,
    PROXIMITY_RADIUS_FRACTION,
)

# A crop roughly the size of a real one (example_data/F1_Plate_2_YS crops are
# 60-100 wide by 302-376 tall), with the seed coat near the bottom centre.
CROP_W, CROP_H = 100, 340
CROP_SIZE = (CROP_W + CROP_H) / 2
SEED_POINT = (50, 330)

RADIUS = CROP_SIZE * PROXIMITY_RADIUS_FRACTION
THRESHOLD = (RADIUS ** 2) * np.pi * AREA_THRESHOLD_FRACTION


def _contours(blobs):
    """blobs: [(cx, cy, radius)] -> the contour list one frame would produce."""
    mask = np.zeros((CROP_H, CROP_W), dtype=np.uint8)
    for cx, cy, r in blobs:
        cv2.circle(mask, (cx, cy), r, 255, -1)
    return list(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])


def _germ_blob():
    """A blob at the seed coat, comfortably over the area threshold."""
    r = int(np.ceil(np.sqrt(THRESHOLD / np.pi))) + 3
    return _contours([(SEED_POINT[0], SEED_POINT[1], r)])


def _empty():
    return _contours([])


def _detect(frames, detector=None):
    detector = detector or GerminationDetector()
    return detector, detector.detect(0, frames, SEED_POINT, CROP_SIZE)


def test_sustained_germination_is_detected_at_its_first_frame():
    frames = [_empty()] * 5 + [_germ_blob()] * 10
    _, got = _detect(frames)

    assert got == 5


def test_single_frame_blip_is_not_germination():
    frames = [_empty()] * 5 + [_germ_blob()] + [_empty()] * 9
    _, got = _detect(frames)

    assert got is None


def test_intermittent_signal_still_detects():
    """The germ mask is present in only ~40-55% of frames on real data, so a
    strict two-consecutive-frames rule missed germination that was plainly
    visible. Hit / miss / hit must confirm."""
    frames = [_empty()] * 4 + [_germ_blob(), _empty(), _germ_blob()] + [_empty()] * 5
    _, got = _detect(frames)

    assert got == 4
    assert CONFIRM_HITS == 2 and CONFIRM_WINDOW == 3


def test_blob_far_from_the_seed_point_is_ignored():
    """Germination is defined at the seed coat; a cotyledon-height blob at the
    top of the crop is not it."""
    far = _contours([(SEED_POINT[0], 20, 20)])
    frames = [far] * 10
    _, got = _detect(frames)

    assert got is None


def test_no_germ_mask_at_all_is_reported_as_a_segmentation_failure():
    """example_data/F1_Plate_2_YS seedling 0's real behaviour: germ_v1 produced
    17 px of mask across 50 frames. No threshold can rescue that, and the
    diagnostic has to say so rather than look like a threshold miss -- it's the
    difference between 'tune the threshold' and 'set the frame by hand'."""
    detector, got = _detect([_empty()] * 20)

    assert got is None
    assert "NO GERM MASK" in detector.describe(0)


def test_germ_mask_present_but_off_target_is_reported_differently():
    big_but_far = _contours([(SEED_POINT[0], 20, 30)])
    detector, got = _detect([big_but_far] * 20)

    assert got is None
    description = detector.describe(0)
    assert "NO GERM MASK" not in description
    assert "not near the seed" in description


def test_manual_override_wins_over_detection_and_can_be_cleared():
    detector, got = _detect([_empty()] * 5 + [_germ_blob()] * 5)
    assert got == 5
    assert detector.has_override(0) is False

    detector.set_override(0, 2)
    assert detector.get_time_zero(0) == 2
    assert detector.has_override(0) is True

    detector.clear_override(0)
    assert detector.get_time_zero(0) == 5
    assert detector.has_override(0) is False


def test_override_works_for_a_seedling_that_never_detected():
    detector = GerminationDetector()

    detector.set_override(3, 7)

    assert detector.get_time_zero(3) == 7
    assert detector.get_time_zero(4) is None  # untouched seedling


def test_series_too_short_to_confirm_does_not_crash():
    _, got = _detect([_germ_blob()])

    assert got is None
