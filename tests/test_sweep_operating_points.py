"""Tests for `multi/sweep_operating_points.py`, the threshold sweep that
scores the 4-class student and its three binary teachers under the SAME
decision rule -- written because `multi/evaluate_multiclass.py` scores them
under different ones (teacher `prob > 0.5`, student 4-way argmax), so its
single-point precision gap cannot distinguish a worse model from a looser
readout.

House style (matching tests/test_evaluate_multiclass.py and
tests/test_multiclass_inference.py): synthesise everything in-test, plain
asserts, no fixtures, no GPU, no real data -- the actual sweep needs a GPU and
the RootPainter annotation folders, so only the accumulation, interpolation
and reproduction-check logic is exercised here.
"""
import numpy as np
import pytest

from multi.sweep_operating_points import (
    DEFAULT_THRESHOLDS,
    READOUT_ARGMAX,
    READOUT_THRESHOLD,
    TEACHER_HEAD,
    TEACHER_OPERATING_POINT,
    SweepAccumulator,
    check_reproduction,
    interpolate_precision_at_recall,
    matched_recall_table,
)


def _annotation(foreground, background=None, defined=None):
    """Build the (foreground, background, defined) triple
    `decode_rootpainter_annotation` returns, from a 1-D foreground pattern."""
    foreground = np.asarray(foreground, dtype=bool).reshape(1, -1)
    background = np.asarray(background, dtype=bool).reshape(1, -1) if background is not None \
        else ~foreground
    defined = np.asarray(defined, dtype=bool).reshape(1, -1) if defined is not None \
        else np.ones_like(foreground)
    return foreground, background, defined


def test_threshold_grid_contains_the_published_operating_point():
    """0.5 anchors both the reproduction check and the matched-recall
    comparison -- if it ever drops out of the grid neither can be computed."""
    assert TEACHER_OPERATING_POINT in DEFAULT_THRESHOLDS
    assert list(DEFAULT_THRESHOLDS) == sorted(DEFAULT_THRESHOLDS)


def test_threshold_readout_at_0_5_matches_the_hardcoded_rule_it_replaces():
    """The t=0.5 row must be exactly evaluate_multiclass.py's `probs > 0.5`
    baseline: same comparison, same pixels, same scorer."""
    probs = np.array([[0.9, 0.6, 0.5, 0.4, 0.1]], dtype=np.float32)
    annotation = _annotation([True, True, True, False, False])

    accumulator = SweepAccumulator()
    accumulator.add_threshold_readouts("hypocot_v5", TEACHER_HEAD, "hypocotyl", probs, annotation)
    rows = accumulator.rows()

    row = next(r for r in rows if r["threshold"] == 0.5)
    # `> 0.5` predicts pixels 0 and 1 only: 2 tp, 1 fn (the 0.5 pixel, which
    # is NOT > 0.5), 0 fp.
    assert row["micro_recall"] == pytest.approx(2 / 3)
    assert row["micro_precision"] == pytest.approx(1.0)
    assert row["readout"] == READOUT_THRESHOLD


def test_lower_thresholds_gain_recall_and_lose_precision():
    """The whole premise of the sweep: recall is monotone non-increasing and
    precision monotone non-decreasing as the threshold rises, so a permissive
    readout inevitably looks high-recall/low-precision."""
    probs = np.array([[0.95, 0.8, 0.6, 0.45, 0.3, 0.15]], dtype=np.float32)
    annotation = _annotation([True, True, True, False, False, False])

    accumulator = SweepAccumulator()
    accumulator.add_threshold_readouts("student", "groupnorm", "cotyledon", probs, annotation)
    rows = sorted(accumulator.rows(), key=lambda r: r["threshold"])

    recalls = [r["micro_recall"] for r in rows]
    assert recalls == sorted(recalls, reverse=True)
    assert rows[0]["threshold"] == pytest.approx(0.05)
    assert rows[0]["micro_recall"] == pytest.approx(1.0)      # t=0.05 catches everything
    assert rows[0]["micro_precision"] == pytest.approx(0.5)   # ...including all 3 background pixels

    strict = next(r for r in rows if r["threshold"] == pytest.approx(0.9))
    assert strict["micro_precision"] == pytest.approx(1.0)    # only the 0.95 pixel survives
    assert strict["micro_recall"] == pytest.approx(1 / 3)

    # Above every probability in the image nothing is predicted at all, so
    # precision is undefined (nan) rather than 1.0 -- the case
    # interpolate_precision_at_recall has to drop.
    assert np.isnan(rows[-1]["micro_precision"])
    assert rows[-1]["micro_recall"] == pytest.approx(0.0)


def test_argmax_readout_is_scored_as_its_own_row_with_nan_threshold():
    """The student's published readout must appear in the same table as the
    sweep, flagged as a distinct readout rather than smuggled in at some
    threshold it does not correspond to."""
    # Class index 2 is hypocotyl (mask_merge.CLASS_NAMES).
    label_map = np.array([[2, 2, 0, 1]], dtype=np.uint8)
    annotation = _annotation([True, True, True, False])

    accumulator = SweepAccumulator()
    accumulator.add_argmax_readout("run_plain_head", "plain", "hypocotyl", label_map, annotation)
    rows = accumulator.rows()

    assert len(rows) == 1
    assert rows[0]["readout"] == READOUT_ARGMAX
    assert np.isnan(rows[0]["threshold"])
    assert rows[0]["micro_recall"] == pytest.approx(2 / 3)
    assert rows[0]["micro_precision"] == pytest.approx(1.0)


def test_argmax_readout_pools_every_image_into_one_row():
    """Regression: the argmax cell was originally keyed by a fresh
    `float("nan")` threshold, and NaN never compares equal to itself -- so
    every dict lookup missed, a new ClassScorer was allocated per image, and
    the sweep emitted one single-image row per image instead of one pooled
    row. Micro precision/recall computed over one image each is not the
    published number, so this silently broke the reproduction check."""
    annotation = _annotation([True, True, False, False])
    label_map = np.array([[2, 0, 2, 0]], dtype=np.uint8)  # 1 tp, 1 fn, 1 fp

    accumulator = SweepAccumulator()
    for _ in range(3):
        accumulator.add_argmax_readout("run_x", "groupnorm", "hypocotyl", label_map, annotation)

    rows = accumulator.rows()
    assert len(rows) == 1
    assert rows[0]["n_images"] == 3
    assert rows[0]["n_images_scored"] == 3
    assert rows[0]["micro_precision"] == pytest.approx(0.5)  # pooled 3 tp / (3 tp + 3 fp)
    assert rows[0]["micro_recall"] == pytest.approx(0.5)


def test_threshold_readout_pools_every_image_into_one_row_per_threshold():
    probs = np.array([[0.9, 0.1]], dtype=np.float32)
    annotation = _annotation([True, False])

    accumulator = SweepAccumulator()
    for _ in range(4):
        accumulator.add_threshold_readouts("m", "h", "cotyledon", probs, annotation)

    rows = accumulator.rows()
    assert len(rows) == len(DEFAULT_THRESHOLDS)
    assert all(r["n_images"] == 4 for r in rows)


def test_accumulator_keeps_models_classes_and_readouts_in_separate_cells():
    """One scorer per (model, head, class, readout, threshold): a shared cell
    would silently pool two models' pixels into one precision figure."""
    probs = np.array([[0.9, 0.1]], dtype=np.float32)
    annotation = _annotation([True, False])

    accumulator = SweepAccumulator(thresholds=(0.5,))
    accumulator.add_threshold_readouts("teacher", TEACHER_HEAD, "cotyledon", probs, annotation)
    accumulator.add_threshold_readouts("student", "groupnorm", "cotyledon", probs, annotation)
    accumulator.add_threshold_readouts("student", "groupnorm", "hypocotyl", probs, annotation)
    accumulator.add_argmax_readout("student", "groupnorm", "cotyledon",
                                   np.array([[1, 0]], dtype=np.uint8), annotation)

    rows = accumulator.rows()
    assert len(rows) == 4
    keys = {(r["model"], r["class"], r["readout"]) for r in rows}
    assert keys == {
        ("teacher", "cotyledon", READOUT_THRESHOLD),
        ("student", "cotyledon", READOUT_THRESHOLD),
        ("student", "hypocotyl", READOUT_THRESHOLD),
        ("student", "cotyledon", READOUT_ARGMAX),
    }
    assert all(r["n_images"] == 1 for r in rows)


def test_undefined_pixels_are_excluded_from_every_threshold():
    """The sweep must inherit evaluate_multiclass's "defined pixels only"
    rule at every operating point, not just at 0.5 -- it reuses ClassScorer
    precisely so this cannot drift."""
    probs = np.array([[0.9, 0.9]], dtype=np.float32)
    # Pixel 1 is an explicit background stroke the model calls foreground --
    # but it is undefined, so it must never become a false positive.
    annotation = _annotation(
        foreground=[True, False], background=[False, True], defined=[True, False],
    )

    accumulator = SweepAccumulator(thresholds=(0.5,))
    accumulator.add_threshold_readouts("m", "h", "cotyledon", probs, annotation)

    row = accumulator.rows()[0]
    assert row["micro_precision"] == pytest.approx(1.0)
    assert row["micro_recall"] == pytest.approx(1.0)


def test_interpolate_precision_at_recall_is_linear_between_bracketing_points():
    points = [
        (0.2, 0.30, 0.90),
        (0.5, 0.50, 0.70),
        (0.8, 0.90, 0.30),
    ]
    matched = interpolate_precision_at_recall(points, target_recall=0.80)

    assert matched["reachable"]
    # Halfway between the (0.50, 0.70) and (0.30, 0.90) points in recall.
    assert matched["precision"] == pytest.approx(0.40)
    assert matched["threshold"] == pytest.approx(0.35)


def test_interpolate_hits_a_sweep_point_exactly_when_the_target_is_one():
    points = [(0.2, 0.30, 0.90), (0.5, 0.50, 0.70)]
    matched = interpolate_precision_at_recall(points, target_recall=0.70)
    assert matched["reachable"]
    assert matched["precision"] == pytest.approx(0.50)
    assert matched["threshold"] == pytest.approx(0.5)


def test_interpolate_refuses_to_extrapolate_past_the_curves_maximum_recall():
    """The explicit requirement: if the student never reaches the teacher's
    recall, say so rather than invent a precision for it."""
    points = [(0.2, 0.30, 0.55), (0.5, 0.50, 0.40), (0.8, 0.90, 0.10)]
    matched = interpolate_precision_at_recall(points, target_recall=0.95)

    assert not matched["reachable"]
    assert np.isnan(matched["precision"])
    assert matched["max_recall"] == pytest.approx(0.55)
    assert matched["precision_at_max_recall"] == pytest.approx(0.30)
    assert "never reaches" in matched["note"]


def test_interpolate_drops_nan_points_rather_than_interpolating_through_them():
    """At a high threshold a model can predict no foreground anywhere, which
    makes its precision nan (ClassScorer._precision_recall). Interpolating
    through that point would poison the result."""
    points = [
        (0.2, 0.30, 0.90),
        (0.5, 0.50, 0.70),
        (0.95, float("nan"), 0.0),
    ]
    matched = interpolate_precision_at_recall(points, target_recall=0.80)
    assert matched["reachable"]
    assert matched["precision"] == pytest.approx(0.40)


def test_interpolate_with_no_usable_points_reports_rather_than_raises():
    matched = interpolate_precision_at_recall(
        [(0.5, float("nan"), float("nan"))], target_recall=0.5
    )
    assert not matched["reachable"]
    assert np.isnan(matched["precision"])
    assert matched["note"]


def _row(model, head, cname, readout, threshold, precision, recall):
    return {
        "model": model, "head": head, "class": cname, "readout": readout,
        "threshold": threshold, "n_images": 1, "n_images_scored": 1,
        "n_images_below_0.05_recall": 0,
        "micro_precision": precision, "micro_recall": recall,
        "macro_precision": precision, "macro_recall": recall,
    }


def _synthetic_rows():
    """A miniature sweep: one teacher and one student for one class, where
    the student's curve happens to pass through the teacher's recall."""
    rows = [
        _row("hypocot_v5", TEACHER_HEAD, "hypocotyl", READOUT_THRESHOLD, 0.5, 0.957, 0.960),
        _row("hypocot_v5", TEACHER_HEAD, "hypocotyl", READOUT_THRESHOLD, 0.9, 0.990, 0.800),
        _row("run_x", "groupnorm", "hypocotyl", READOUT_ARGMAX, float("nan"), 0.156, 0.925),
        _row("run_x", "groupnorm", "hypocotyl", READOUT_THRESHOLD, 0.5, 0.400, 0.940),
        _row("run_x", "groupnorm", "hypocotyl", READOUT_THRESHOLD, 0.9, 0.800, 0.860),
    ]
    return rows


def test_matched_recall_table_reports_student_precision_at_the_teachers_recall():
    table = matched_recall_table(_synthetic_rows())

    assert len(table) == 1
    entry = table[0]
    assert entry["class"] == "hypocotyl"
    assert entry["student"] == "run_x"
    assert entry["teacher"] == "hypocot_v5"
    assert entry["teacher_precision"] == pytest.approx(0.957)
    assert entry["teacher_recall"] == pytest.approx(0.960)
    # The student's argmax row is carried through unchanged for reference.
    assert entry["student_argmax_precision"] == pytest.approx(0.156)
    # Target recall 0.960 is above the student's best swept recall (0.940).
    assert not entry["reachable"]
    assert entry["student_max_recall"] == pytest.approx(0.940)


def test_matched_recall_table_interpolates_when_the_student_does_reach_it():
    rows = _synthetic_rows()
    # Give the student a permissive point that overshoots the teacher's recall.
    rows.append(_row("run_x", "groupnorm", "hypocotyl", READOUT_THRESHOLD, 0.05, 0.200, 0.980))

    entry = matched_recall_table(rows)[0]

    assert entry["reachable"]
    # Between (0.400, 0.940) at t=0.5 and (0.200, 0.980) at t=0.05: target
    # 0.960 is exactly halfway, so precision 0.300 at threshold 0.275.
    assert entry["student_precision_at_teacher_recall"] == pytest.approx(0.300)
    assert entry["student_threshold_at_teacher_recall"] == pytest.approx(0.275)


def test_matched_recall_table_only_uses_the_teachers_own_0_5_row():
    """A teacher row at some other threshold must not become the target --
    the published operating point is 0.5 and nothing else."""
    rows = [r for r in _synthetic_rows() if not (r["head"] == TEACHER_HEAD and r["threshold"] == 0.5)]
    assert matched_recall_table(rows) == []


def test_check_reproduction_passes_on_the_published_figures():
    published = {("hypocot_v5", "hypocotyl", READOUT_THRESHOLD): (0.957, 0.960)}
    checks = check_reproduction(_synthetic_rows(), published=published)

    assert len(checks) == 1
    label, expected, got, ok = checks[0]
    assert ok
    assert expected == (0.957, 0.960)
    assert got == (pytest.approx(0.957), pytest.approx(0.960))


def test_check_reproduction_fails_loudly_when_a_number_drifts():
    """The gate the whole report hangs on: a harness that does not reproduce
    the published operating points is measuring something else, and this must
    surface as a failure rather than a plausible-looking table."""
    published = {("hypocot_v5", "hypocotyl", READOUT_THRESHOLD): (0.800, 0.960)}
    (_, _, _, ok), = check_reproduction(_synthetic_rows(), published=published)
    assert not ok


def test_check_reproduction_matches_the_student_argmax_row_not_a_threshold_row():
    """The student's published numbers come from argmax, so the check must
    look the argmax row up -- picking a threshold row instead would compare
    against the wrong readout entirely."""
    published = {("run_x", "hypocotyl", READOUT_ARGMAX): (0.156, 0.925)}
    (_, _, got, ok), = check_reproduction(_synthetic_rows(), published=published)
    assert ok
    assert got[0] == pytest.approx(0.156)  # argmax row, not the 0.400 threshold row


def test_check_reproduction_skips_anchors_with_no_matching_row():
    published = {("not_a_model", "hypocotyl", READOUT_THRESHOLD): (0.5, 0.5)}
    assert check_reproduction(_synthetic_rows(), published=published) == []
