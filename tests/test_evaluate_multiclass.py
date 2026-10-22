import numpy as np
import pytest

from multi.evaluate_multiclass import ClassScorer, _per_image_stats, _precision_recall


def test_precision_recall_basic():
    precision, recall = _precision_recall(tp=8, fp=2, fn=2)
    assert precision == pytest.approx(0.8)
    assert recall == pytest.approx(0.8)


def test_precision_recall_nan_when_no_positive_predictions_at_all():
    precision, recall = _precision_recall(tp=0, fp=0, fn=5)
    assert np.isnan(precision)
    assert recall == 0.0


def test_defined_pixel_masking_excludes_undefined_pixels():
    """A pixel that looks like a foreground stroke but is not `defined` must
    never enter tp/fn -- _per_image_stats masks foreground/background by
    `defined` itself rather than trusting the caller to have already done
    so."""
    pred_fg = np.array([[True, True, False]])
    foreground = np.array([[True, True, False]])
    background = np.array([[False, False, False]])
    defined = np.array([[True, False, False]])  # pixel 1 looks foreground but is undefined

    tp, fp, fn, n_fg = _per_image_stats(pred_fg, foreground, background, defined)

    assert tp == 1
    assert fp == 0
    assert fn == 0
    assert n_fg == 1


def test_defined_pixel_masking_excludes_undefined_background_too():
    pred_fg = np.array([[False, True]])  # pixel 1: false positive, but undefined
    foreground = np.array([[False, False]])
    background = np.array([[True, True]])
    defined = np.array([[True, False]])

    tp, fp, fn, n_fg = _per_image_stats(pred_fg, foreground, background, defined)

    assert tp == 0
    assert fp == 0  # the only background pixel that's a false positive is undefined
    assert fn == 0
    assert n_fg == 0


def test_class_scorer_micro_and_macro_recall_diverge_on_one_missed_image():
    """The exact failure mode the module docstring calls out: many small
    images that are perfectly recalled, plus one image that is entirely
    missed but contributes few foreground pixels relative to the pooled
    total. Micro recall (pixel-pooled) barely moves; macro recall (mean of
    per-image recall) tanks, because it counts the miss as one whole image
    out of twelve regardless of pixel count."""
    scorer = ClassScorer()

    # 11 images, 1000 foreground pixels each, all perfectly recalled.
    for _ in range(11):
        pred_fg = np.ones((1, 1000), dtype=bool)
        foreground = np.ones((1, 1000), dtype=bool)
        background = np.zeros((1, 1000), dtype=bool)
        defined = np.ones((1, 1000), dtype=bool)
        scorer.add(pred_fg, foreground, background, defined)

    # 1 image, only 10 foreground pixels, entirely missed (0 predicted).
    pred_fg = np.zeros((1, 10), dtype=bool)
    foreground = np.ones((1, 10), dtype=bool)
    background = np.zeros((1, 10), dtype=bool)
    defined = np.ones((1, 10), dtype=bool)
    scorer.add(pred_fg, foreground, background, defined)

    summary = scorer.summary()

    assert summary["n_images"] == 12
    assert summary["n_images_scored"] == 12
    assert summary["n_images_below_0.05_recall"] == 1

    expected_micro_recall = 11000 / 11010
    expected_macro_recall = 11 / 12
    assert summary["micro_recall"] == pytest.approx(expected_micro_recall)
    assert summary["macro_recall"] == pytest.approx(expected_macro_recall)
    # The whole point: pooling (micro) hides what averaging (macro) reveals.
    assert summary["micro_recall"] > summary["macro_recall"]


def test_class_scorer_skips_images_with_no_foreground_pixels_from_macro_average():
    """An image with zero foreground-stroke pixels for this class has an
    undefined per-image recall and must not be folded into the macro
    average (nor counted as a below-threshold image)."""
    scorer = ClassScorer()

    pred_fg = np.zeros((1, 5), dtype=bool)
    foreground = np.zeros((1, 5), dtype=bool)  # no foreground stroke pixels at all
    background = np.ones((1, 5), dtype=bool)
    defined = np.ones((1, 5), dtype=bool)
    scorer.add(pred_fg, foreground, background, defined)

    summary = scorer.summary()

    assert summary["n_images"] == 1
    assert summary["n_images_scored"] == 0
    assert summary["n_images_below_0.05_recall"] == 0
    assert np.isnan(summary["macro_recall"])


def test_class_scorer_micro_precision_pools_false_positives_correctly():
    scorer = ClassScorer()

    # Image 1: 5 tp, 5 fp
    scorer.add(
        pred_fg=np.array([[True] * 10]),
        foreground=np.array([[True] * 5 + [False] * 5]),
        background=np.array([[False] * 5 + [True] * 5]),
        defined=np.ones((1, 10), dtype=bool),
    )
    # Image 2: 5 tp, 0 fp
    scorer.add(
        pred_fg=np.array([[True] * 5 + [False] * 5]),
        foreground=np.array([[True] * 5 + [False] * 5]),
        background=np.array([[False] * 10]),
        defined=np.ones((1, 10), dtype=bool),
    )

    summary = scorer.summary()
    # Pooled: tp=10, fp=5 -> precision = 10/15
    assert summary["micro_precision"] == pytest.approx(10 / 15)
    assert summary["micro_recall"] == pytest.approx(1.0)
