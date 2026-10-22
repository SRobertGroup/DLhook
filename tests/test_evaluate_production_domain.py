"""Tests for `multi/evaluate_production_domain.py`, which re-scores the
4-class student and its three binary teachers on production-style
per-seedling crops instead of the legacy 1024x1024 whole-plate frames every
published number was measured on.

The load-bearing claim of that harness is that the human ground truth survives
the re-cropping *pixel-aligned with the image*, so most of these tests are
about the crop geometry and the annotation carry-through rather than about
scoring (which is `ClassScorer`'s, already covered by
tests/test_evaluate_multiclass.py and tests/test_sweep_operating_points.py).

House style (matching tests/test_sweep_operating_points.py and
tests/test_evaluate_multiclass.py): synthesise everything in-test, plain
asserts, no fixtures, no GPU, no real data -- the actual evaluation needs a
GPU and the RootPainter annotation folders.
"""
import numpy as np
import pytest

from multi.evaluate_production_domain import (
    FRAMING_CROP,
    FRAMING_WHOLE_PLATE,
    MIN_COMPONENT_PIXELS,
    SEEDLING_MERGE_RADIUS,
    crop_annotation,
    crop_probability_map,
    crop_size_summary,
    cut_production_crops,
    find_seedling_boxes,
    legacy_comparison_table,
    load_legacy_rows,
    union_annotated_foreground,
)
from multi.src.recrop_geometry import compute_crop_box, crop_from_box
from multi.sweep_operating_points import READOUT_ARGMAX, READOUT_THRESHOLD, TEACHER_HEAD


def _blank(h=400, w=400):
    return np.zeros((h, w), dtype=bool)


def _annotation(foreground, background=None, defined=None):
    """The (foreground, background, defined) triple
    `decode_rootpainter_annotation` returns, defaulting to "everything the
    strokes did not mark is explicit background, and all of it is defined"."""
    background = ~foreground if background is None else background
    defined = np.ones_like(foreground) if defined is None else defined
    return foreground, background, defined


# Two seedlings far enough apart that no plausible merge radius fuses them:
# 160 px of clear space, against a 12 px radius.
SEEDLING_A = (slice(20, 181), slice(30, 91))   # rows 20..180, cols 30..90
SEEDLING_B = (slice(20, 181), slice(250, 311))


def _two_seedling_foreground():
    foreground = _blank()
    foreground[SEEDLING_A] = True
    foreground[SEEDLING_B] = True
    return foreground


def test_two_seedlings_yield_two_boxes_matching_compute_crop_box():
    """The core geometry assertion: each box is exactly what the production
    rule (`compute_crop_box`, the same function the GUI and recrop_plates.py
    use) returns for that seedling's extent -- not an approximation of it."""
    boxes = find_seedling_boxes(_two_seedling_foreground())

    assert len(boxes) == 2
    expected = [
        compute_crop_box(30, 90, 20, 180),
        compute_crop_box(250, 310, 20, 180),
    ]
    assert boxes == expected
    # Sanity: this case is decided by the padding, not by the 30 px floor, so
    # it genuinely exercises the padding fractions (0.4 width, 0.10 height).
    assert boxes[0]["half_w"] == 54   # 60 * (1 + 2*0.4) / 2
    assert boxes[0]["half_h"] == 96   # 160 * (1 + 2*0.10) / 2


def test_boxes_come_back_in_a_deterministic_order():
    """Connected-component label order is an implementation detail of OpenCV;
    the rows must not depend on it."""
    foreground = _blank()
    foreground[SEEDLING_B] = True
    foreground[SEEDLING_A] = True
    boxes = find_seedling_boxes(foreground)
    assert [b["cx"] for b in boxes] == sorted(b["cx"] for b in boxes)


def test_organ_fragments_of_one_seedling_merge_into_a_single_box():
    """A seedling's cotyledon dab, hypocotyl stroke and radicle stroke are
    three disjoint components. Labelling them directly would give three boxes
    for one seedling; the dilation exists precisely to prevent that."""
    foreground = _blank()
    foreground[20:30, 50:60] = True    # cotyledon dab
    foreground[40:90, 52:57] = True    # hypocotyl stroke, 10 px below it
    foreground[100:130, 52:56] = True  # radicle stroke, 10 px below that

    boxes = find_seedling_boxes(foreground)

    assert len(boxes) == 1
    # The box spans all three fragments and is taken from the UNDILATED
    # pixels, so it matches the fragments' own joint extent exactly.
    assert boxes[0] == compute_crop_box(50, 59, 20, 129)


def test_dilation_does_not_fuse_two_neighbouring_seedlings():
    """Neighbouring seedlings in these plate frames sit ~65 px apart. The
    merge radius has to close a seedling's own gaps without closing that one,
    or two seedlings land in one crop and the geometry stops being
    production-like."""
    foreground = _blank()
    foreground[20:120, 50:56] = True
    foreground[20:120, 115:121] = True  # 59 px of clear space between them

    assert len(find_seedling_boxes(foreground)) == 2
    # ...and the radius really is the thing keeping them apart.
    assert len(find_seedling_boxes(foreground, merge_radius=40)) == 1


def test_specks_below_the_minimum_component_size_are_not_seedlings():
    foreground = _blank()
    foreground[20:120, 50:56] = True
    foreground[300, 300] = True  # one stray annotated pixel, 180 px away

    assert len(find_seedling_boxes(foreground)) == 1
    assert len(find_seedling_boxes(foreground, min_component_pixels=1)) == 2
    assert MIN_COMPONENT_PIXELS > 1


def test_no_annotated_foreground_yields_no_boxes():
    assert find_seedling_boxes(_blank()) == []
    assert find_seedling_boxes(None) == []


def test_union_pools_every_available_class():
    cotyledon = _blank()
    cotyledon[20:30, 50:60] = True
    radicle = _blank()
    radicle[100:130, 52:56] = True

    union = union_annotated_foreground({
        "cotyledon": _annotation(cotyledon),
        "radicle": _annotation(radicle),
    })

    assert np.array_equal(union, cotyledon | radicle)


def test_union_excludes_foreground_outside_the_defined_region():
    """Mirrors `_per_image_stats`: a pixel flagged foreground but not defined
    is not ground truth, so it must not attract a crop box either."""
    foreground = _blank()
    foreground[20:120, 50:56] = True
    foreground[300:310, 300:310] = True
    defined = _blank()
    defined[:200, :] = True  # the second blob is undefined

    union = union_annotated_foreground({"hypocotyl": _annotation(foreground, defined=defined)})

    assert union[20:120, 50:56].all()
    assert not union[300:310, 300:310].any()
    assert find_seedling_boxes(union) == [compute_crop_box(50, 55, 20, 119)]


def test_union_of_nothing_is_none():
    assert union_annotated_foreground({}) is None
    assert union_annotated_foreground({"cotyledon": _annotation(_blank())}) is None


def test_cropped_annotation_keeps_exactly_the_in_box_foreground():
    """The whole reason this harness is allowed to claim it carries real human
    ground truth into the new geometry: the cropped annotation must hold
    precisely the original's pixels inside the box, no more and no fewer."""
    foreground = _two_seedling_foreground()
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    image[foreground] = 200

    records = cut_production_crops(image, {"hypocotyl": _annotation(foreground)})

    assert len(records) == 2
    for record in records:
        x1, y1, x2, y2 = record["bounds"]
        cropped_fg, cropped_bg, cropped_defined = record["annotations"]["hypocotyl"]
        assert np.count_nonzero(cropped_fg) == np.count_nonzero(foreground[y1:y2, x1:x2])
        assert np.array_equal(cropped_fg, foreground[y1:y2, x1:x2])
        assert np.array_equal(cropped_bg, (~foreground)[y1:y2, x1:x2])
        assert cropped_defined.all()
    # Between them the two crops carry every annotated pixel of the frame.
    total = sum(np.count_nonzero(r["annotations"]["hypocotyl"][0]) for r in records)
    assert total == np.count_nonzero(foreground)


def test_crop_uses_the_identical_slice_for_image_and_annotation():
    """A separately recomputed slice would drift from the image wherever the
    box is clamped at a frame edge; the bounds come from `crop_from_box`'s own
    return value for exactly that reason."""
    foreground = _blank()
    foreground[0:40, 0:20] = True  # hard against the top-left corner
    image = np.arange(400 * 400 * 3, dtype=np.uint8).reshape(400, 400, 3)

    (record,) = cut_production_crops(image, {"cotyledon": _annotation(foreground)})
    x1, y1, x2, y2 = record["bounds"]

    expected_crop, ex1, ey1, ex2, ey2 = crop_from_box(image, record["box"])
    assert (x1, y1, x2, y2) == (ex1, ey1, ex2, ey2)
    assert (x1, y1) == (0, 0)  # clamped, not negative
    assert np.array_equal(record["image"], expected_crop)
    assert record["annotations"]["cotyledon"][0].shape == record["image"].shape[:2]


def test_crop_annotation_is_a_plain_window_on_all_three_planes():
    foreground = np.zeros((10, 10), dtype=bool)
    foreground[2:5, 3:7] = True
    background = ~foreground
    defined = np.zeros((10, 10), dtype=bool)
    defined[:6, :] = True

    cropped = crop_annotation((foreground, background, defined), 2, 1, 8, 6)

    for plane, original in zip(cropped, (foreground, background, defined)):
        assert plane.shape == (5, 6)
        assert np.array_equal(plane, original[1:6, 2:8])


def test_every_class_annotation_is_carried_into_every_crop():
    """A crop must carry each class that was annotated on the source frame,
    even when that class has no strokes inside this particular crop -- its
    explicit background strokes there are real false-positive evidence."""
    cotyledon = _blank()
    cotyledon[20:181, 30:91] = True
    hypocotyl = _blank()
    hypocotyl[20:181, 250:311] = True
    image = np.zeros((400, 400, 3), dtype=np.uint8)

    records = cut_production_crops(image, {
        "cotyledon": _annotation(cotyledon),
        "hypocotyl": _annotation(hypocotyl),
    })

    assert len(records) == 2
    for record in records:
        assert set(record["annotations"]) == {"cotyledon", "hypocotyl"}


def test_crop_size_summary_reports_the_distribution_and_the_floor_fraction():
    summary = crop_size_summary([(60, 60), (60, 60), (74, 248), (100, 300)])

    assert summary["n_crops"] == 4
    assert summary["median_width"] == pytest.approx(67.0)
    assert summary["median_height"] == pytest.approx(154.0)
    assert summary["min_width"] == 60
    assert summary["max_height"] == 300
    # Two of the four crops were decided by compute_crop_box's 30 px half-size
    # floor rather than by the seedling -- the signal that the harness is not
    # reproducing production geometry.
    assert summary["fraction_at_min_box_floor"] == pytest.approx(0.5)


def test_crop_size_summary_of_nothing_does_not_raise():
    assert crop_size_summary([]) == {"n_crops": 0}


def _row(model, head, cname, readout, threshold, precision, recall):
    return {
        "model": model, "head": head, "class": cname, "readout": readout,
        "threshold": threshold, "n_images": 10, "n_images_scored": 10,
        "n_images_below_0.05_recall": 0,
        "micro_precision": precision, "micro_recall": recall,
        "macro_precision": precision, "macro_recall": recall,
    }


def test_legacy_comparison_pairs_each_model_with_its_own_operating_point():
    """Teachers are read at threshold 0.5 and students at argmax -- the two
    published operating points. Pairing a student's threshold row against a
    teacher's would compare different decision rules, which is the mistake
    sweep_operating_points.py exists to avoid."""
    production = [
        _row("hypocot_v5", TEACHER_HEAD, "hypocotyl", READOUT_THRESHOLD, 0.5, 0.97, 0.95),
        _row("hypocot_v5", TEACHER_HEAD, "hypocotyl", READOUT_THRESHOLD, 0.9, 0.99, 0.80),
        _row("run_x", "groupnorm", "hypocotyl", READOUT_ARGMAX, float("nan"), 0.62, 0.90),
        _row("run_x", "groupnorm", "hypocotyl", READOUT_THRESHOLD, 0.5, 0.70, 0.88),
    ]
    legacy = [
        _row("hypocot_v5", TEACHER_HEAD, "hypocotyl", READOUT_THRESHOLD, 0.5, 0.957, 0.960),
        _row("run_x", "groupnorm", "hypocotyl", READOUT_ARGMAX, float("nan"), 0.156, 0.925),
    ]

    table = legacy_comparison_table(production, legacy)

    assert [e["model"] for e in table] == ["hypocot_v5", "run_x"]
    student = table[1]
    assert student["role"] == "student"
    assert student["readout"] == READOUT_ARGMAX
    assert student["legacy_micro_precision"] == pytest.approx(0.156)
    assert student["production_micro_precision"] == pytest.approx(0.62)
    assert student["precision_delta"] == pytest.approx(0.62 - 0.156)
    assert student["legacy_source"] == "sweep csv"
    teacher = table[0]
    assert teacher["legacy_micro_precision"] == pytest.approx(0.957)
    assert teacher["production_micro_precision"] == pytest.approx(0.97)


def test_legacy_comparison_falls_back_to_the_published_anchors():
    production = [_row("hypocot_v5", TEACHER_HEAD, "hypocotyl", READOUT_THRESHOLD, 0.5, 0.97, 0.95)]
    published = {("hypocot_v5", "hypocotyl", READOUT_THRESHOLD): (0.957, 0.960)}

    (entry,) = legacy_comparison_table(production, legacy_rows=[], published=published)

    assert entry["legacy_source"] == "published"
    assert entry["legacy_micro_recall"] == pytest.approx(0.960)


def test_legacy_comparison_keeps_models_with_no_legacy_figure():
    """A newly-added checkpoint has no legacy row. It must still appear with a
    production number rather than vanishing from the table."""
    production = [_row("run_new", "plain", "radicle", READOUT_ARGMAX, float("nan"), 0.40, 0.30)]

    (entry,) = legacy_comparison_table(production, legacy_rows=[], published={})

    assert entry["legacy_source"] == "-"
    assert np.isnan(entry["legacy_micro_precision"])
    assert entry["production_micro_precision"] == pytest.approx(0.40)


def test_load_legacy_rows_returns_empty_for_a_missing_file(tmp_path):
    assert load_legacy_rows(tmp_path / "nope.csv") == []


def test_load_legacy_rows_parses_numeric_columns(tmp_path):
    path = tmp_path / "sweep.csv"
    path.write_text(
        "model,head,class,readout,threshold,n_images,n_images_scored,"
        "n_images_below_0.05_recall,micro_precision,micro_recall,macro_precision,macro_recall\n"
        "hypocot_v5,binary,hypocotyl,threshold,0.5,200,126,4,0.957,0.960,0.916,0.901\n",
        encoding="utf-8",
    )

    (row,) = load_legacy_rows(path)

    assert row["threshold"] == pytest.approx(0.5)
    assert row["micro_precision"] == pytest.approx(0.957)
    assert row["n_images_scored"] == 126
    assert isinstance(row["n_images_scored"], int)


def test_crop_probability_map_windows_a_teacher_map_and_a_student_stack_alike():
    """The whole-plate control cuts whole-frame probabilities with the SAME
    bounds the crop was cut with, so the two framings score identical pixels.
    A binary teacher's map is HxW and a student's stack is CxHxW; the window
    has to land on the last two axes either way."""
    teacher = np.arange(100, dtype=np.float32).reshape(10, 10)
    student = np.stack([teacher, teacher + 100, teacher + 200, teacher + 300])
    bounds = (2, 1, 8, 6)  # x1, y1, x2, y2

    assert np.array_equal(crop_probability_map(teacher, bounds), teacher[1:6, 2:8])
    cropped_student = crop_probability_map(student, bounds)
    assert cropped_student.shape == (4, 5, 6)
    assert np.array_equal(cropped_student[2], student[2][1:6, 2:8])


def test_crop_probability_map_lines_up_with_the_annotation_crop():
    """Pinned together rather than separately: a drift between these two
    windows would silently score each model against a shifted annotation."""
    foreground = _blank()
    foreground[20:181, 30:91] = True
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    probabilities = np.random.default_rng(0).random((400, 400)).astype(np.float32)

    (record,) = cut_production_crops(image, {"hypocotyl": _annotation(foreground)})
    cropped_probs = crop_probability_map(probabilities, record["bounds"])

    assert cropped_probs.shape == record["annotations"]["hypocotyl"][0].shape
    assert cropped_probs.shape == record["image"].shape[:2]


def test_the_two_framings_are_distinct_named_modes():
    assert FRAMING_CROP != FRAMING_WHOLE_PLATE


def test_merge_radius_default_sits_between_the_two_measured_scales():
    """Documented rationale, pinned: a seedling's organ strokes are a few px
    apart, neighbouring seedlings ~65 px. The default must close the first
    without closing the second (i.e. stay below half of 65)."""
    assert 0 < SEEDLING_MERGE_RADIUS < 65 / 2
