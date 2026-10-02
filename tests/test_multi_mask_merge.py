import warnings
from pathlib import Path

import cv2
import numpy as np

from multi.src.mask_merge import (
    CLASS_INDEX,
    IGNORE_VALUE,
    apply_human_overrides,
    decode_rootpainter_annotation,
    discover_annotation_files,
    merge_dataset,
    merge_dataset_from_probs,
    _merge_probs_to_labels,
)

THRESHOLDS = {"cotyledon": (0.05, 0.5), "hypocotyl": (0.05, 0.5), "radicle": (0.02, 0.3)}


def _rgba(r, g, a=None):
    """Build an HxWx3 (no alpha) or HxWx4 RGBA array from boolean R/G/A masks."""
    h, w = r.shape
    channels = [
        np.where(r, 255, 0).astype(np.uint8),
        np.where(g, 255, 0).astype(np.uint8),
        np.zeros((h, w), dtype=np.uint8),  # blue is always 0
    ]
    if a is not None:
        channels.append(np.where(a, 255, 0).astype(np.uint8))
    return np.stack(channels, axis=-1)


def test_decode_reads_foreground_background_and_undefined():
    r = np.array([[True, False], [False, False]])
    g = np.array([[False, True], [False, False]])
    a = np.array([[True, True], [False, False]])  # bottom row undefined (A=0)
    rgba = _rgba(r, g, a)

    fg, bg, defined = decode_rootpainter_annotation(rgba)

    assert fg.tolist() == [[True, False], [False, False]]
    assert bg.tolist() == [[False, True], [False, False]]
    assert defined.tolist() == [[True, True], [False, False]]


def test_decode_no_alpha_channel_falls_back_to_r_or_g():
    """The dlhook_cotyledon/annotations/train/3-crop-IMG_1104.png defect:
    RGB only, no alpha at all. "defined" must come from R | G, not crash."""
    r = np.array([[True, False, False]])
    g = np.array([[False, True, False]])
    rgb = _rgba(r, g, a=None)
    assert rgb.shape[-1] == 3

    fg, bg, defined = decode_rootpainter_annotation(rgb)

    assert fg.tolist() == [[True, False, False]]
    assert bg.tolist() == [[False, True, False]]
    assert defined.tolist() == [[True, True, False]]


def test_single_class_clearing_t_hi_wins_outright():
    cot = np.array([[0.9]], dtype=np.float32)
    hyp = np.array([[0.01]], dtype=np.float32)
    rad = np.array([[0.01]], dtype=np.float32)

    label, order = _merge_probs_to_labels(
        {"cotyledon": cot, "hypocotyl": hyp, "radicle": rad}, THRESHOLDS, conflict_margin=0.15
    )

    assert label[0, 0] == CLASS_INDEX["cotyledon"]


def test_all_classes_below_t_lo_is_background():
    cot = np.array([[0.01]], dtype=np.float32)
    hyp = np.array([[0.01]], dtype=np.float32)
    rad = np.array([[0.005]], dtype=np.float32)

    label, _ = _merge_probs_to_labels(
        {"cotyledon": cot, "hypocotyl": hyp, "radicle": rad}, THRESHOLDS, conflict_margin=0.15
    )

    assert label[0, 0] == 0


def test_ambiguous_middle_band_with_no_winner_is_ignored():
    """No class clears t_hi, but cotyledon sits in the (t_lo, t_hi) band --
    there is no consensus, so the pixel must be IGNORE, not background."""
    cot = np.array([[0.2]], dtype=np.float32)  # between 0.05 and 0.5
    hyp = np.array([[0.01]], dtype=np.float32)
    rad = np.array([[0.005]], dtype=np.float32)

    label, _ = _merge_probs_to_labels(
        {"cotyledon": cot, "hypocotyl": hyp, "radicle": rad}, THRESHOLDS, conflict_margin=0.15
    )

    assert label[0, 0] == IGNORE_VALUE


def test_a_winner_outright_is_not_overridden_by_another_classs_ignore_band():
    """cotyledon clears t_hi and wins outright even though hypocotyl is
    sitting in its own ambiguous middle band at the same pixel."""
    cot = np.array([[0.9]], dtype=np.float32)
    hyp = np.array([[0.3]], dtype=np.float32)  # ambiguous band, but irrelevant: cotyledon won
    rad = np.array([[0.005]], dtype=np.float32)

    label, _ = _merge_probs_to_labels(
        {"cotyledon": cot, "hypocotyl": hyp, "radicle": rad}, THRESHOLDS, conflict_margin=0.15
    )

    assert label[0, 0] == CLASS_INDEX["cotyledon"]


def test_two_classes_clearing_t_hi_with_a_clear_margin_the_higher_wins():
    cot = np.array([[0.55]], dtype=np.float32)
    hyp = np.array([[0.9]], dtype=np.float32)  # margin 0.35 > 0.15
    rad = np.array([[0.01]], dtype=np.float32)

    label, _ = _merge_probs_to_labels(
        {"cotyledon": cot, "hypocotyl": hyp, "radicle": rad}, THRESHOLDS, conflict_margin=0.15
    )

    assert label[0, 0] == CLASS_INDEX["hypocotyl"]


def test_two_classes_within_conflict_margin_is_ignored():
    cot = np.array([[0.6]], dtype=np.float32)
    hyp = np.array([[0.55]], dtype=np.float32)  # margin 0.05 <= 0.15
    rad = np.array([[0.01]], dtype=np.float32)

    label, _ = _merge_probs_to_labels(
        {"cotyledon": cot, "hypocotyl": hyp, "radicle": rad}, THRESHOLDS, conflict_margin=0.15
    )

    assert label[0, 0] == IGNORE_VALUE


def test_conflict_margin_boundary_is_inclusive_of_ignore():
    """margin exactly equal to conflict_margin still counts as a conflict
    (the rule is "within", i.e. <=)."""
    cot = np.array([[0.65]], dtype=np.float32)
    hyp = np.array([[0.50001]], dtype=np.float32)  # margin ~= 0.15
    rad = np.array([[0.01]], dtype=np.float32)

    label, _ = _merge_probs_to_labels(
        {"cotyledon": cot, "hypocotyl": hyp, "radicle": rad}, THRESHOLDS, conflict_margin=0.15
    )

    assert label[0, 0] == IGNORE_VALUE

    # Nudge the margin just over the line: now there must be a clear winner.
    cot2 = np.array([[0.6501]], dtype=np.float32)
    hyp2 = np.array([[0.5]], dtype=np.float32)
    label2, _ = _merge_probs_to_labels(
        {"cotyledon": cot2, "hypocotyl": hyp2, "radicle": rad}, THRESHOLDS, conflict_margin=0.15
    )
    assert label2[0, 0] == CLASS_INDEX["cotyledon"]


def test_human_foreground_stroke_overrides_pseudo_label():
    label = np.array([[0, 2]], dtype=np.uint8)  # pseudo-label said background, hypocotyl
    foreground = np.array([[True, False]])
    background = np.array([[False, False]])

    touched = apply_human_overrides(label, "cotyledon", foreground, background)

    assert touched is True
    assert label.tolist() == [[CLASS_INDEX["cotyledon"], 2]]


def test_human_background_stroke_overrides_pseudo_label_to_zero():
    label = np.array([[1, 2]], dtype=np.uint8)  # pseudo-label said cotyledon, hypocotyl
    foreground = np.array([[False, False]])
    background = np.array([[True, False]])

    apply_human_overrides(label, "cotyledon", foreground, background)

    assert label.tolist() == [[0, 2]]


def test_undefined_pixels_leave_the_pseudo_label_untouched():
    label = np.array([[1, 255]], dtype=np.uint8)
    foreground = np.array([[False, False]])
    background = np.array([[False, False]])

    touched = apply_human_overrides(label, "cotyledon", foreground, background)

    assert touched is False
    assert label.tolist() == [[1, 255]]


def test_train_val_duplicate_annotation_keeps_train_and_warns(tmp_path):
    """The known dlhook_hypocotyl defect: one filename in both train/ and
    val/. discover_annotation_files must keep train and drop val, not just
    for that one hardcoded name."""
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()
    (train_dir / "dup.png").write_bytes(b"train-bytes")
    (val_dir / "dup.png").write_bytes(b"val-bytes")
    (val_dir / "only_in_val.png").write_bytes(b"val-bytes-2")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        index = discover_annotation_files(tmp_path)

    assert index["dup.png"] == train_dir / "dup.png"
    assert index["only_in_val.png"] == val_dir / "only_in_val.png"
    assert any("both train" in str(w.message) for w in caught)


def test_disk_fed_merge_matches_inline_merge_byte_identical(tmp_path, monkeypatch):
    """The key correctness test for the disk-fed merge path: given the same
    effective per-class probabilities, `merge_dataset` (inline inference)
    and `merge_dataset_from_probs` (reading uint8 PNGs from disk) must
    produce byte-identical label PNGs -- both call `_merge_probs_to_labels`
    verbatim, so this proves the disk-fed wiring doesn't drift from it.

    The probabilities are quantized to uint8 ONCE, up front, and that
    already-quantized value is what both paths see (the inline path via a
    fake predictor, the disk-fed path via the PNG written from the same
    array) -- so this isolates the merge algorithm/wiring from the separate
    quantization-error question covered by
    test_uint8_probability_quantization_round_trips_within_one_255th.
    """
    h, w = 6, 7
    rng = np.random.default_rng(0)
    raw_probs = {
        "cotyledon": rng.random((h, w)).astype(np.float32),
        "hypocotyl": rng.random((h, w)).astype(np.float32),
        "radicle": rng.random((h, w)).astype(np.float32),
    }
    quantized_u8 = {cname: np.round(p * 255).astype(np.uint8) for cname, p in raw_probs.items()}
    quantized_f32 = {cname: u8.astype(np.float32) / 255.0 for cname, u8 in quantized_u8.items()}

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    image_name = "0-crop-test.png"
    raw_image = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    cv2.imwrite(str(raw_dir / image_name), raw_image)

    class _FakePredictor:
        def __init__(self, cname):
            self.cname = cname

        def _segment_many(self, images):
            return [quantized_f32[self.cname] for _ in images]

    def _fake_get_predictor(model_path, **kwargs):
        return _FakePredictor(Path(model_path).stem)

    monkeypatch.setattr("models.UNetInference.get_predictor", _fake_get_predictor)

    inline_out = tmp_path / "inline_out"
    inline_report = tmp_path / "inline_report.csv"
    weights = {"cotyledon": "cotyledon", "hypocotyl": "hypocotyl", "radicle": "radicle"}
    merge_dataset(raw_dir, inline_out, inline_report, weights=weights)

    prob_dir = tmp_path / "probs"
    prob_dir.mkdir()
    for cname, u8 in quantized_u8.items():
        cv2.imwrite(str(prob_dir / f"0-crop-test-{cname}.png"), u8)

    disk_out = tmp_path / "disk_out"
    disk_report = tmp_path / "disk_report.csv"
    merge_dataset_from_probs(raw_dir, prob_dir, disk_out, disk_report)

    inline_label = cv2.imread(str(inline_out / image_name), cv2.IMREAD_GRAYSCALE)
    disk_label = cv2.imread(str(disk_out / image_name), cv2.IMREAD_GRAYSCALE)

    assert inline_label is not None and disk_label is not None
    assert np.array_equal(inline_label, disk_label)
    assert (inline_out / image_name).read_bytes() == (disk_out / image_name).read_bytes()


def test_uint8_probability_quantization_round_trips_within_one_255th():
    """merge_dataset_from_probs converts a loaded uint8 probability PNG via
    `prob_u8.astype(np.float32) / 255.0`. Verify that conversion never
    drifts a true probability by more than 1/255 -- the maximum error uint8
    storage can introduce."""
    rng = np.random.default_rng(1)
    true_probs = rng.random((200,)).astype(np.float32)

    quantized_u8 = np.round(true_probs * 255).astype(np.uint8)
    recovered = quantized_u8.astype(np.float32) / 255.0

    assert np.all(np.abs(recovered - true_probs) <= 1.0 / 255.0 + 1e-6)


def test_missing_probability_map_warns_and_skips_the_crop(tmp_path):
    """A crop missing one of its three probability maps must be skipped
    with a clear warning, not silently treated as all-background for the
    missing class -- a missing file and a confident-background file must
    not look the same."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    image_name = "0-crop-test.png"
    cv2.imwrite(str(raw_dir / image_name), np.zeros((4, 4, 3), dtype=np.uint8))

    prob_dir = tmp_path / "probs"
    prob_dir.mkdir()
    # cotyledon and hypocotyl probability maps exist; radicle is missing.
    cv2.imwrite(str(prob_dir / "0-crop-test-cotyledon.png"), np.full((4, 4), 200, dtype=np.uint8))
    cv2.imwrite(str(prob_dir / "0-crop-test-hypocotyl.png"), np.zeros((4, 4), dtype=np.uint8))

    out_dir = tmp_path / "out"
    report_path = tmp_path / "report.csv"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = merge_dataset_from_probs(raw_dir, prob_dir, out_dir, report_path)

    assert result.processed == []
    assert result.skipped == [image_name]
    assert not (out_dir / image_name).exists()
    assert any("missing probability map" in str(w.message) for w in caught)
