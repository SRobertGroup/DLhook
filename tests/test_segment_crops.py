import sys
from pathlib import Path

import cv2
import numpy as np

# multi/ is not a package (no __init__.py) -- put it on sys.path so
# `import segment_crops` resolves the same way the CLI script itself
# resolves its own `from src...` imports (matches tests/test_recrop_plates.py).
_MULTI_DIR = str(Path(__file__).resolve().parent.parent / "multi")
if _MULTI_DIR not in sys.path:
    sys.path.insert(0, _MULTI_DIR)

import segment_crops  # noqa: E402


# ---------------------------------------------------------------------------
# uint8 probability round-trip -- the highest-risk part of this module (see
# its CRITICAL CONVENTION docstring): 255 = high probability, 0 = low, and
# it must NOT be inverted the way MaskStore.dump() inverts masks.
# ---------------------------------------------------------------------------

def test_prob_to_uint8_round_trip_preserves_values_within_one_255th():
    prob = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=np.float32)
    encoded = segment_crops.prob_to_uint8(prob)

    assert encoded.dtype == np.uint8
    decoded = encoded.astype(np.float32) / 255.0
    assert np.allclose(decoded, prob, atol=1.0 / 255.0)


def test_prob_to_uint8_polarity_high_probability_is_255_not_0():
    # The single easiest mistake this module could make: inverting the
    # convention the way the legacy on-disk MASK format does.
    prob = np.array([[0.0, 1.0]], dtype=np.float32)
    encoded = segment_crops.prob_to_uint8(prob)
    assert encoded[0, 0] == 0    # low probability -> low byte value
    assert encoded[0, 1] == 255  # high probability -> high byte value


def test_prob_to_uint8_clamps_out_of_range_inputs():
    # _segment_many returns a softmax output so this shouldn't happen, but
    # writing a valid PNG must never raise regardless.
    prob = np.array([[-0.1, 1.1]], dtype=np.float32)
    encoded = segment_crops.prob_to_uint8(prob)
    assert encoded[0, 0] == 0
    assert encoded[0, 1] == 255


def test_prob_to_uint8_round_trips_through_an_actual_png(tmp_path):
    rng = np.random.default_rng(0)
    prob = rng.random((16, 20), dtype=np.float32)
    encoded = segment_crops.prob_to_uint8(prob)

    path = tmp_path / "prob.png"
    cv2.imwrite(str(path), encoded)
    reread = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    assert reread.dtype == np.uint8
    assert reread.shape == prob.shape
    assert np.allclose(reread.astype(np.float32) / 255.0, prob, atol=1.0 / 255.0)


# ---------------------------------------------------------------------------
# Resumability: a crop whose three class outputs already all exist is
# skipped unless --force.
# ---------------------------------------------------------------------------

def test_needs_processing_true_when_no_outputs_exist(tmp_path):
    assert segment_crops.needs_processing(tmp_path, "cropA", force=False) is True


def test_needs_processing_false_when_all_three_outputs_exist(tmp_path):
    for cname in segment_crops.CLASS_WEIGHTS:
        segment_crops.output_path(tmp_path, "cropA", cname).write_bytes(b"\x00")
    assert segment_crops.needs_processing(tmp_path, "cropA", force=False) is False


def test_needs_processing_true_when_only_some_outputs_exist(tmp_path):
    class_names = list(segment_crops.CLASS_WEIGHTS)
    segment_crops.output_path(tmp_path, "cropA", class_names[0]).write_bytes(b"\x00")
    assert segment_crops.needs_processing(tmp_path, "cropA", force=False) is True


def test_needs_processing_true_when_force_even_if_all_outputs_exist(tmp_path):
    for cname in segment_crops.CLASS_WEIGHTS:
        segment_crops.output_path(tmp_path, "cropA", cname).write_bytes(b"\x00")
    assert segment_crops.needs_processing(tmp_path, "cropA", force=True) is True


# ---------------------------------------------------------------------------
# segment_crops() end-to-end against a stubbed predictor -- no real model,
# no GPU, no network. Verifies: correct output filenames, correct
# probability values written (polarity), and that already-done crops are
# skipped while incomplete/missing ones are (re)processed.
# ---------------------------------------------------------------------------

class _FakePredictor:
    """Stands in for UNetInference: _segment_many returns a foreground-
    probability map per image, here just the image's mean intensity broadcast
    over its shape (deterministic, cheap, and enough to prove real per-image
    values -- not a constant -- flow through to the written files)."""

    def __init__(self, value):
        self.value = value

    def _segment_many(self, images):
        return [np.full(img.shape[:2], self.value, dtype=np.float32) for img in images]


def _make_get_predictor(values_by_path):
    def get_predictor(path):
        return _FakePredictor(values_by_path[path])
    return get_predictor


def test_segment_crops_writes_expected_files_with_correct_probabilities(tmp_path):
    crops_dir = tmp_path / "crops"
    out_dir = tmp_path / "out"
    crops_dir.mkdir()

    img = np.zeros((8, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(crops_dir / "0-crop-SeriesA_frame1.png"), img)

    # Distinct, recognizable probabilities per class so polarity/identity
    # mix-ups between classes would fail this test.
    values = {
        str(segment_crops.CLASS_WEIGHTS["cotyledon"]): 0.1,
        str(segment_crops.CLASS_WEIGHTS["hypocotyl"]): 0.5,
        str(segment_crops.CLASS_WEIGHTS["radicle"]): 0.9,
    }
    get_predictor = _make_get_predictor(values)

    n_processed, n_skipped, n_failed = segment_crops.segment_crops(
        crops_dir, out_dir, force=False, chunk_size=64,
        get_predictor_fn=get_predictor, image_chunk_default=64,
    )

    assert (n_processed, n_skipped, n_failed) == (1, 0, 0)

    for cname, prob in [("cotyledon", 0.1), ("hypocotyl", 0.5), ("radicle", 0.9)]:
        out_path = segment_crops.output_path(out_dir, "0-crop-SeriesA_frame1", cname)
        assert out_path.exists()
        written = cv2.imread(str(out_path), cv2.IMREAD_UNCHANGED)
        assert written.dtype == np.uint8
        assert abs(int(written[0, 0]) - round(prob * 255)) <= 1


def test_segment_crops_skips_fully_done_crop_without_calling_predictor(tmp_path):
    crops_dir = tmp_path / "crops"
    out_dir = tmp_path / "out"
    crops_dir.mkdir()
    out_dir.mkdir()

    img = np.zeros((4, 4, 3), dtype=np.uint8)
    cv2.imwrite(str(crops_dir / "0-crop-Series_frame.png"), img)
    for cname in segment_crops.CLASS_WEIGHTS:
        segment_crops.output_path(out_dir, "0-crop-Series_frame", cname).write_bytes(b"\x00")

    def get_predictor(path):
        raise AssertionError("predictor should never be constructed when nothing is pending")

    n_processed, n_skipped, n_failed = segment_crops.segment_crops(
        crops_dir, out_dir, force=False, chunk_size=64,
        get_predictor_fn=get_predictor, image_chunk_default=64,
    )
    assert (n_processed, n_skipped, n_failed) == (0, 1, 0)


def test_segment_crops_force_reprocesses_a_fully_done_crop(tmp_path):
    crops_dir = tmp_path / "crops"
    out_dir = tmp_path / "out"
    crops_dir.mkdir()
    out_dir.mkdir()

    img = np.zeros((4, 4, 3), dtype=np.uint8)
    cv2.imwrite(str(crops_dir / "0-crop-Series_frame.png"), img)
    for cname in segment_crops.CLASS_WEIGHTS:
        segment_crops.output_path(out_dir, "0-crop-Series_frame", cname).write_bytes(b"\x00")

    values = {str(p): 0.7 for p in segment_crops.CLASS_WEIGHTS.values()}
    get_predictor = _make_get_predictor(values)

    n_processed, n_skipped, n_failed = segment_crops.segment_crops(
        crops_dir, out_dir, force=True, chunk_size=64,
        get_predictor_fn=get_predictor, image_chunk_default=64,
    )
    assert (n_processed, n_skipped, n_failed) == (1, 0, 0)

    written = cv2.imread(
        str(segment_crops.output_path(out_dir, "0-crop-Series_frame", "cotyledon")),
        cv2.IMREAD_UNCHANGED,
    )
    assert abs(int(written[0, 0]) - round(0.7 * 255)) <= 1


def test_segment_crops_reports_unreadable_files(tmp_path):
    crops_dir = tmp_path / "crops"
    out_dir = tmp_path / "out"
    crops_dir.mkdir()

    # Valid extension, garbage bytes -- cv2.imread returns None for this.
    (crops_dir / "0-crop-Bad_frame.png").write_bytes(b"not a real png")

    values = {str(p): 0.5 for p in segment_crops.CLASS_WEIGHTS.values()}
    get_predictor = _make_get_predictor(values)

    n_processed, n_skipped, n_failed = segment_crops.segment_crops(
        crops_dir, out_dir, force=False, chunk_size=64,
        get_predictor_fn=get_predictor, image_chunk_default=64,
    )
    assert (n_processed, n_skipped, n_failed) == (0, 0, 1)


def test_segment_crops_respects_chunk_size_across_multiple_chunks(tmp_path):
    crops_dir = tmp_path / "crops"
    out_dir = tmp_path / "out"
    crops_dir.mkdir()

    for i in range(5):
        img = np.full((4, 4, 3), i, dtype=np.uint8)
        cv2.imwrite(str(crops_dir / f"0-crop-Series_frame{i}.png"), img)

    values = {str(p): 0.42 for p in segment_crops.CLASS_WEIGHTS.values()}
    get_predictor = _make_get_predictor(values)

    n_processed, n_skipped, n_failed = segment_crops.segment_crops(
        crops_dir, out_dir, force=False, chunk_size=2,  # forces 3 chunks for 5 crops
        get_predictor_fn=get_predictor, image_chunk_default=64,
    )
    assert (n_processed, n_skipped, n_failed) == (5, 0, 0)
    for i in range(5):
        for cname in segment_crops.CLASS_WEIGHTS:
            assert segment_crops.output_path(out_dir, f"0-crop-Series_frame{i}", cname).exists()
