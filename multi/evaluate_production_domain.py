#!/usr/bin/env python
"""CLI: re-score the 4-class student and its three binary teachers in the
geometry the GUI actually runs them in -- tight, per-seedling crops -- instead
of the legacy 1024x1024 whole-plate frames every published number so far was
measured on.

Why this script exists
----------------------
The student's precision against human annotations is catastrophically below
its teachers' (hypocotyl micro-precision 0.156 vs 0.957). Two explanations
have already been tested and eliminated: the output-head architecture
(`multi/results/run_plain_head/`, no difference) and the readout asymmetry
(`multi/sweep_operating_points.py`, a full threshold sweep -- the student's
precision curve never overlaps the teacher's at any operating point).

The remaining structural explanation is a train/evaluate DOMAIN mismatch:

- the student was TRAINED on `cropped_training_set/` -- tight per-seedling
  crops cut at native plate resolution, median 74x248 px (manifest.csv), one
  seedling filling the frame;
- every evaluation so far RUNS on `training_dataset/dlhook/` -- 1024x1024
  whole-plate frames holding ~20 seedlings and large areas of empty agar.
  These are legacy output of a retired GUI code path (the pre-refactor GUI
  used a fixed 1024x1024 working size);
- the three teachers were trained in RootPainter *on those 1024^2 frames*, so
  they are in-domain there and the student is not;
- CURRENT PRODUCTION geometry is the tight per-seedling crop: the bounding box
  of the user's two clicks, padded 40% horizontally and 10% vertically, cut at
  native resolution with no resize (`seedling_measurment.py`, mirrored by
  `multi/src/recrop_geometry.compute_crop_box`) -- the same geometry family as
  `cropped_training_set/`.

So the student may be being condemned on a framing it will never see. This
script measures it in the framing it will.

How
---
For every human-annotated 1024^2 image it locates seedlings from the UNION of
the three classes' annotated foreground (connected components of the union,
merged by a modest dilation so one seedling yields one box rather than one box
per organ stroke -- see `SEEDLING_MERGE_RADIUS`), cuts each box with the
production rule via the shared `compute_crop_box`/`crop_from_box`, and applies
the IDENTICAL slice to the raw image and to every annotation RGBA, so the real
human ground truth survives into the new geometry pixel-aligned. Student and
teachers are then scored on those crops with the SAME `ClassScorer` and
`decode_rootpainter_annotation` the published evaluation uses, over the same
threshold grid as `multi/sweep_operating_points.py` plus the student's argmax
readout. Output columns are identical to `sweep_operating_points.csv`, so the
two CSVs can be concatenated and compared row for row.

THE CAVEAT -- read this before reading any number below
--------------------------------------------------------
Crops are cut around ANNOTATED foreground, so by construction there are no
pure-background crops: every crop contains at least one seedling, and the
large empty-agar regions of the plate -- where a false positive costs
precision and nothing else -- are simply never presented to any model. This
INFLATES PRECISION FOR EVERY MODEL, student and teacher alike. It is
symmetric, so the student-vs-teacher comparison stays fair, but the absolute
precision figures are optimistic relative to a run over whole plates, and must
not be quoted as production precision. Same text is printed at the top and
bottom of every run and written to
`multi/results/evaluate_production_domain_caveats.txt`.

Second caveat, specific to this annotation set: only 73 of the 471 annotated
images carry strokes for more than one class (most carry exactly one), so for
most seedlings the "union of the three classes' foreground" is really one
organ's strokes. The resulting box is a tight frame on that organ rather than
on the whole seedling, and it frequently lands on
`compute_crop_box`'s `min_box_half_size` floor (60x60). Compare the printed
crop-size distribution against `cropped_training_set/manifest.csv`
(median 74x248) before trusting the result: if the crops do not resemble the
training crops, this harness is not reproducing production geometry and its
numbers mean nothing.

Usage:
    python multi/evaluate_production_domain.py --config multi/configs/training_config.yaml \\
        --model multi/results/run_groupnorm_head_baseline/best.pt \\
        --model multi/results/run_plain_head/best.pt

    # Quick smoke run over a handful of annotated images:
    python multi/evaluate_production_domain.py --config multi/configs/training_config.yaml \\
        --model multi/results/run_plain_head/best.pt --limit 4
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import REPO_ROOT, ensure_repo_root_importable, load_config, resolved_path
from src.mask_merge import (
    CLASS_INDEX,
    FOREGROUND_CLASSES,
    decode_rootpainter_annotation,
    discover_annotation_files,
)
from src.recrop_geometry import (
    DEFAULT_PADDING_HEIGHT_FRACTION,
    DEFAULT_PADDING_WIDTH_FRACTION,
    compute_crop_box,
    crop_from_box,
)

ensure_repo_root_importable()

# Imported, never redefined -- exactly as sweep_operating_points.py imports
# ClassScorer: the whole point is that these rows pool tp/fp/fn the same way
# the published evaluation does, so the only thing that differs between the
# two tables is the geometry of the pixels fed in.
from multi.evaluate_multiclass import (  # noqa: E402
    MULTICLASS_PATCH_SIZE,
    _load_rgba,
)
from multi.sweep_operating_points import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    PUBLISHED,
    READOUT_ARGMAX,
    READOUT_THRESHOLD,
    REPORT_FIELDNAMES,
    TEACHER_HEAD,
    TEACHER_OPERATING_POINT,
    SweepAccumulator,
    _default_model_name,
    _print_matched_recall,
    _print_sweep_table,
    matched_recall_table,
)

# --- seedling localisation -------------------------------------------------
#
# RootPainter strokes are sparse dabs along one organ: measured over this
# annotation set, a hypocotyl stroke's connected component is ~37 px tall and
# a cotyledon dab ~9 px, and one seedling that has two classes annotated
# contributes two or more disjoint components. Dilating the union before
# labelling merges the strokes of one seedling; the box is then taken from the
# UNDILATED pixels inside each merged component, so the dilation only decides
# grouping and never inflates the box.
#
# 12 px is chosen against the two measured scales it sits between: strokes of
# one seedling's organs are contiguous or within a few px of each other, while
# neighbouring seedlings in these 1024^2 plate frames are ~65 px apart
# centre-to-centre. A radius of 12 (25x25 ellipse) closes the former and stays
# comfortably below half the latter, so it does not fuse two seedlings into
# one box.
SEEDLING_MERGE_RADIUS = 12
# Drop specks: single stray annotated pixels are not seedlings. Measured
# component areas are 20-150 px, so 10 keeps every real stroke.
MIN_COMPONENT_PIXELS = 10

# Where inference runs. Both framings score the IDENTICAL pixels -- the same
# crops of the same annotations -- and differ only in what the network was
# shown when it produced the probabilities for those pixels.
#   crop        : the model is handed the per-seedling crop (production).
#   whole_plate : the model is handed the 1024^2 frame and its probability map
#                 is then cut to the crop (the control -- see
#                 `score_whole_plate_framed`).
FRAMING_CROP = "crop"
FRAMING_WHOLE_PLATE = "whole_plate"

# How many per-seedling crops are pushed through the models in one call.
# Purely a batching knob -- `_segment_many` batches tiles across all images it
# is handed, so a larger number is faster and numerically identical.
DEFAULT_CROP_BATCH_SIZE = 24

# `cropped_training_set/manifest.csv`: the geometry the student was trained
# on, and therefore the target the crops this script cuts must resemble for
# the measurement to mean anything.
TRAINING_CROP_MEDIAN_WIDTH = 74
TRAINING_CROP_MEDIAN_HEIGHT = 248

CAVEAT = """\
CAVEAT -- THIS RUN'S PRECISION FIGURES ARE OPTIMISTIC FOR EVERY MODEL, AND
THE INFLATION IS NOT EVEN ACROSS MODELS.

Crops are cut around ANNOTATED foreground, so there are no pure-background
crops: every crop contains a seedling, and the large empty-agar regions of the
plate -- where a false positive costs precision and nothing else -- are never
presented to any model. Measured on this annotation set, the crops retain
~100% of the human FOREGROUND strokes but only 0.7-1.2% of the explicit
BACKGROUND strokes, i.e. ~99% of the evidence that can produce a false
positive is simply no longer scored. Precision therefore rises for every model
regardless of whether any model got better.

It is tempting to call that symmetric and therefore fair. It is not. The
inflation depends on WHERE a model's false positives sit: a model whose errors
hug the seedling keeps them inside the crop, while a model whose errors are
spread over the plate loses ~99% of them. The student is the latter and the
teachers are the former, so cropping helps the student far more.

MEASURED, 2026-09-18, with `--framing whole_plate` as the control: the student
reaches hypocotyl micro-precision 0.965 from WHOLE-PLATE inference scored on
the crop pixels, versus 0.955 from crop inference on the same pixels and 0.156
on the full legacy pixel set. The recovery from 0.156 is therefore entirely
the pixel-population change; per-seedling FRAMING contributes nothing (and is
marginally negative). Crop framing also costs recall badly at these crop sizes
-- cotyledon teacher recall 0.920 (control) -> 0.259 (crop framing) on
identical pixels. Always run both framings; the difference between them is the
real framing effect and everything else is accounting.

Secondary caveat: most annotated images carry strokes for only one of the
three classes, so for most seedlings the union-of-classes box frames one organ
rather than the whole seedling and often lands on compute_crop_box's 60x60
min_box_half_size floor. Check the crop-size distribution printed above
against cropped_training_set/manifest.csv (median {w}x{h}) before trusting
anything here: crops that do not resemble the training crops mean this harness
is not reproducing production geometry.""".format(
    w=TRAINING_CROP_MEDIAN_WIDTH, h=TRAINING_CROP_MEDIAN_HEIGHT
)


def union_annotated_foreground(annotations: dict) -> np.ndarray | None:
    """Boolean OR of every available class's DEFINED foreground strokes.

    `annotations` maps a class name to the (foreground, background, defined)
    triple `decode_rootpainter_annotation` returns. Foreground is masked by
    `defined` first, mirroring `_per_image_stats`: a pixel flagged foreground
    but outside the defined region is not ground truth and must not attract a
    crop box. Returns None when nothing is available or nothing is annotated.
    """
    union = None
    for foreground, _background, defined in annotations.values():
        marked = foreground & defined
        union = marked if union is None else (union | marked)
    if union is None or not union.any():
        return None
    return union


def find_seedling_boxes(
    union_foreground: np.ndarray,
    merge_radius: int = SEEDLING_MERGE_RADIUS,
    min_component_pixels: int = MIN_COMPONENT_PIXELS,
    padding_width_fraction: float = DEFAULT_PADDING_WIDTH_FRACTION,
    padding_height_fraction: float = DEFAULT_PADDING_HEIGHT_FRACTION,
) -> list[dict]:
    """One production-rule crop box per seedling found in `union_foreground`.

    Components are labelled on a DILATED copy (so the several strokes of one
    seedling group into one component) but each box is computed from that
    component's UNDILATED pixels, so `merge_radius` only affects grouping.
    Boxes come back in a deterministic top-to-bottom, left-to-right order.
    """
    if union_foreground is None or not np.any(union_foreground):
        return []

    mask = union_foreground.astype(np.uint8)
    if merge_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * merge_radius + 1, 2 * merge_radius + 1)
        )
        grouped = cv2.dilate(mask, kernel)
    else:
        grouped = mask

    n_labels, labels = cv2.connectedComponents(grouped, connectivity=8)

    boxes = []
    for label in range(1, n_labels):
        selection = (labels == label) & union_foreground
        if np.count_nonzero(selection) < min_component_pixels:
            continue
        ys, xs = np.nonzero(selection)
        boxes.append(compute_crop_box(
            float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()),
            padding_width_fraction=padding_width_fraction,
            padding_height_fraction=padding_height_fraction,
        ))
    boxes.sort(key=lambda b: (b["cy"], b["cx"]))
    return boxes


def crop_annotation(annotation, x1: int, y1: int, x2: int, y2: int):
    """Apply an already-computed image slice to one (foreground, background,
    defined) triple. Taking the bounds from `crop_from_box`'s return value
    rather than recomputing them is what guarantees the annotation stays
    pixel-aligned with the cropped image even where the box was clamped at an
    image edge."""
    foreground, background, defined = annotation
    window = (slice(y1, y2), slice(x1, x2))
    return foreground[window], background[window], defined[window]


def cut_production_crops(image: np.ndarray, annotations: dict, **box_kwargs) -> list[dict]:
    """Cut one production-style crop per seedling out of a whole-plate frame.

    Returns a list of {"box", "bounds", "image", "annotations"} records. The
    image crop comes from `crop_from_box` -- the same function the headless
    re-cropper uses -- and every annotation is cut with the bounds that call
    returned, never with an independently recomputed slice.
    """
    union = union_annotated_foreground(annotations)
    records = []
    for box in find_seedling_boxes(union, **box_kwargs):
        crop, x1, y1, x2, y2 = crop_from_box(image, box)
        if crop.size == 0:
            continue
        records.append({
            "box": box,
            "bounds": (x1, y1, x2, y2),
            "image": crop,
            "annotations": {
                cname: crop_annotation(triple, x1, y1, x2, y2)
                for cname, triple in annotations.items()
            },
        })
    return records


def crop_probability_map(prob: np.ndarray, bounds) -> np.ndarray:
    """Cut a whole-frame probability map down to one crop's bounds.

    Accepts an (H, W) binary-teacher map or a (num_classes, H, W) student
    stack -- the window is applied to the last two axes either way, so the
    whole-plate control feeds `accumulate` arrays shaped exactly like the
    crop-framed run's.
    """
    x1, y1, x2, y2 = bounds
    return prob[..., y1:y2, x1:x2]


def crop_size_summary(sizes) -> dict:
    """Descriptive stats for the (width, height) of every crop cut, so the run
    can be checked against `cropped_training_set/manifest.csv` before its
    scores are believed."""
    sizes = list(sizes)
    summary = {"n_crops": len(sizes)}
    if not sizes:
        return summary
    widths = np.array([w for w, _h in sizes], dtype=float)
    heights = np.array([h for _w, h in sizes], dtype=float)
    for name, values in (("width", widths), ("height", heights)):
        summary[f"min_{name}"] = float(values.min())
        summary[f"p25_{name}"] = float(np.percentile(values, 25))
        summary[f"median_{name}"] = float(np.median(values))
        summary[f"p75_{name}"] = float(np.percentile(values, 75))
        summary[f"max_{name}"] = float(values.max())
    summary["median_area"] = float(np.median(widths * heights))
    # The floor in compute_crop_box: a crop at exactly 60x60 in both axes is
    # one whose geometry was decided by min_box_half_size, not by the seedling.
    summary["fraction_at_min_box_floor"] = float(
        np.count_nonzero((widths <= 60) & (heights <= 60)) / len(sizes)
    )
    return summary


def _print_crop_size_summary(summary: dict, n_source_images: int) -> None:
    print("\n=== Production crops cut from the annotated plate frames ===")
    print(f"source images with at least one crop : {n_source_images}")
    print(f"crops                                : {summary.get('n_crops', 0)}")
    if not summary.get("n_crops"):
        return
    print(f"width   min/p25/median/p75/max        : "
          f"{summary['min_width']:.0f} / {summary['p25_width']:.0f} / "
          f"{summary['median_width']:.0f} / {summary['p75_width']:.0f} / {summary['max_width']:.0f}")
    print(f"height  min/p25/median/p75/max        : "
          f"{summary['min_height']:.0f} / {summary['p25_height']:.0f} / "
          f"{summary['median_height']:.0f} / {summary['p75_height']:.0f} / {summary['max_height']:.0f}")
    print(f"crops pinned at the 60x60 box floor   : {summary['fraction_at_min_box_floor']:.1%}")
    print(f"training-set crops, for comparison    : median "
          f"{TRAINING_CROP_MEDIAN_WIDTH}x{TRAINING_CROP_MEDIAN_HEIGHT} "
          f"(cropped_training_set/manifest.csv)")


# --- legacy-vs-production comparison ---------------------------------------

def _operating_point_row(rows, model, cname, head):
    """Each model's own headline operating point: threshold 0.5 for a binary
    teacher, argmax for a 4-class student -- exactly the two readouts the
    published table reports."""
    readout = READOUT_THRESHOLD if head == TEACHER_HEAD else READOUT_ARGMAX
    for row in rows:
        if row["model"] != model or row["class"] != cname or row["readout"] != readout:
            continue
        if readout == READOUT_THRESHOLD and not np.isclose(row["threshold"], TEACHER_OPERATING_POINT):
            continue
        return row
    return None


def load_legacy_rows(path) -> list[dict]:
    """Read a previous `sweep_operating_points.csv` back as row dicts with the
    numeric columns as floats. Returns [] when the file is absent, in which
    case `legacy_comparison_table` falls back to the hardcoded published
    figures."""
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            row = dict(raw)
            for key in ("threshold", "micro_precision", "micro_recall",
                        "macro_precision", "macro_recall"):
                row[key] = float(row[key]) if row.get(key) not in (None, "") else float("nan")
            for key in ("n_images", "n_images_scored", "n_images_below_0.05_recall"):
                row[key] = int(row[key]) if row.get(key) not in (None, "") else 0
            rows.append(row)
    return rows


def legacy_comparison_table(production_rows, legacy_rows=None, published=PUBLISHED) -> list[dict]:
    """The deliverable: for every (model, class), micro precision/recall at
    that model's own operating point in the legacy 1024^2 geometry next to the
    same figure in production crop geometry.

    Legacy numbers come from `legacy_rows` (a previous sweep CSV) when one is
    supplied, otherwise from the hardcoded `PUBLISHED` anchors in
    sweep_operating_points.py. Models with no legacy figure still appear, with
    nan in the legacy columns, rather than being dropped.
    """
    legacy_rows = legacy_rows or []
    table = []
    models = sorted(
        {(r["model"], r["head"]) for r in production_rows},
        key=lambda item: (item[1] != TEACHER_HEAD, item[0]),
    )
    for cname in FOREGROUND_CLASSES:
        for model, head in models:
            production = _operating_point_row(production_rows, model, cname, head)
            if production is None:
                continue
            legacy = _operating_point_row(legacy_rows, model, cname, head)
            if legacy is not None:
                legacy_p, legacy_r = legacy["micro_precision"], legacy["micro_recall"]
                legacy_source = "sweep csv"
            else:
                readout = READOUT_THRESHOLD if head == TEACHER_HEAD else READOUT_ARGMAX
                anchor = published.get((model, cname, readout))
                legacy_p, legacy_r = anchor if anchor else (float("nan"), float("nan"))
                legacy_source = "published" if anchor else "-"
            table.append({
                "class": cname,
                "model": model,
                "role": "teacher" if head == TEACHER_HEAD else "student",
                "readout": READOUT_THRESHOLD if head == TEACHER_HEAD else READOUT_ARGMAX,
                "legacy_micro_precision": legacy_p,
                "legacy_micro_recall": legacy_r,
                "legacy_source": legacy_source,
                "production_micro_precision": production["micro_precision"],
                "production_micro_recall": production["micro_recall"],
                "precision_delta": production["micro_precision"] - legacy_p,
                "recall_delta": production["micro_recall"] - legacy_r,
                "n_crops_scored": production["n_images_scored"],
            })
    return table


def _fmt(value) -> str:
    return "    nan" if value is None or np.isnan(value) else f"{value:7.3f}"


def _print_legacy_comparison(table) -> None:
    print("\n=== Legacy 1024^2 geometry vs production per-seedling crop geometry ===")
    print("(micro precision/recall at each model's own operating point: "
          "teacher prob>0.5, student argmax)")
    header = (f"{'class':<10} {'model':<30} {'role':<8} "
              f"{'legacyP':>7} {'legacyR':>7} {'prodP':>7} {'prodR':>7} "
              f"{'dP':>7} {'dR':>7} {'crops':>6}")
    print(header)
    print("-" * len(header))
    for entry in table:
        print(
            f"{entry['class']:<10} {entry['model']:<30} {entry['role']:<8} "
            f"{_fmt(entry['legacy_micro_precision'])} {_fmt(entry['legacy_micro_recall'])} "
            f"{_fmt(entry['production_micro_precision'])} {_fmt(entry['production_micro_recall'])} "
            f"{_fmt(entry['precision_delta'])} {_fmt(entry['recall_delta'])} "
            f"{entry['n_crops_scored']:>6}"
        )


# --- the run ---------------------------------------------------------------

def evaluate_production_domain(
    raw_dir,
    annotations_dirs: dict,
    weights: dict,
    model_paths: dict,
    thresholds=DEFAULT_THRESHOLDS,
    crop_batch_size: int = DEFAULT_CROP_BATCH_SIZE,
    limit: int | None = None,
    merge_radius: int = SEEDLING_MERGE_RADIUS,
    min_component_pixels: int = MIN_COMPONENT_PIXELS,
    framing: str = FRAMING_CROP,
    progress: bool = True,
):
    """Cut production-style crops out of every annotated plate frame and score
    every model on them. Returns (rows, crop_stats) where `rows` has exactly
    `sweep_operating_points.REPORT_FIELDNAMES` columns -- with `n_images`
    counting CROPS, not source frames -- and `crop_stats` is the crop-size
    summary the run must be sanity-checked against.

    Torch-dependent imports are local, mirroring evaluate_multiclass.evaluate,
    so the cropping and table helpers above stay importable without torch.
    """
    from models.UNetInference import MARGIN, get_predictor

    from src.multiclass_inference import MulticlassInference

    raw_dir = Path(raw_dir)
    annotation_index = {
        cname: discover_annotation_files(adir) for cname, adir in annotations_dirs.items()
    }
    all_filenames = sorted(set().union(*[set(idx) for idx in annotation_index.values()])) \
        if annotation_index else []
    if limit is not None:
        all_filenames = all_filenames[:limit]

    binary_predictors = {cname: get_predictor(str(path)) for cname, path in weights.items()}
    teacher_names = {cname: Path(weights[cname]).stem for cname in weights}

    students = {
        name: MulticlassInference(
            path,
            in_size=MULTICLASS_PATCH_SIZE + 2 * MARGIN,
            out_size=MULTICLASS_PATCH_SIZE,
            margin=MARGIN,
        )
        for name, path in model_paths.items()
    }

    accumulator = SweepAccumulator(thresholds)
    sizes: list[tuple[int, int]] = []
    n_source_images = 0
    pending: list[dict] = []

    def accumulate(record, teacher_probs: dict, student_probs: dict) -> None:
        """Feed one crop's readouts. `teacher_probs` maps a class to an HxW
        foreground-probability map the size of the crop; `student_probs` maps
        a student name to its (num_classes, H, W) stack, also crop-sized.
        Both framings funnel through here, so the only thing that can differ
        between them is where those probabilities were computed."""
        for cname, annotation in record["annotations"].items():
            if cname in teacher_probs:
                accumulator.add_threshold_readouts(
                    teacher_names[cname], TEACHER_HEAD, cname,
                    teacher_probs[cname], annotation,
                )
            for student_name, model in students.items():
                probs = student_probs[student_name]
                accumulator.add_threshold_readouts(
                    student_name, model.head, cname,
                    probs[CLASS_INDEX[cname]], annotation,
                )
                accumulator.add_argmax_readout(
                    student_name, model.head, cname,
                    np.argmax(probs, axis=0).astype(np.uint8), annotation,
                )

    def score_crop_framed(batch: list[dict]) -> None:
        """PRODUCTION framing: each model is handed the crop itself, exactly
        as the GUI hands it a per-seedling crop. Every model sees the
        identical crop list, so all rows are scored on the same pixels."""
        if not batch:
            return
        images = [record["image"] for record in batch]
        binary_probs = {
            cname: predictor._segment_many(images)
            for cname, predictor in binary_predictors.items()
        }
        student_probs = {
            name: model.segment_many_argmax(images, return_probs=True)
            for name, model in students.items()
        }
        for i, record in enumerate(batch):
            accumulate(
                record,
                {cname: probs[i] for cname, probs in binary_probs.items()},
                {name: probs[i] for name, probs in student_probs.items()},
            )

    def score_whole_plate_framed(image, records: list[dict]) -> None:
        """THE CONTROL. Inference runs on the whole 1024^2 plate frame, and
        the resulting probability maps are then cut with the crop bounds, so
        this run scores EXACTLY the same pixels as the crop-framed run while
        differing only in what context the network saw.

        This is what separates the two things the crop-framed run confounds:
        a model really behaving better when framed on one seedling, versus
        precision rising merely because cropping removed ~99% of the
        explicitly-annotated background from the scored pixel pool. If these
        two runs give the same numbers, the framing changes nothing and the
        entire production-vs-legacy precision gain is that accounting effect.
        """
        if not records:
            return
        binary_probs = {
            cname: predictor._segment_many([image])[0]
            for cname, predictor in binary_predictors.items()
        }
        student_probs = {
            name: model.segment_many_argmax([image], return_probs=True)[0]
            for name, model in students.items()
        }
        for record in records:
            bounds = record["bounds"]
            accumulate(
                record,
                {cname: crop_probability_map(probs, bounds)
                 for cname, probs in binary_probs.items()},
                {name: crop_probability_map(probs, bounds)
                 for name, probs in student_probs.items()},
            )

    for n_done, name in enumerate(all_filenames, start=1):
        raw_path = raw_dir / name
        if not raw_path.exists():
            warnings.warn(f"No matching raw image found for annotation {name}; skipping")
            continue
        image = cv2.imread(str(raw_path))
        if image is None:
            warnings.warn(f"Could not read raw image {raw_path}; skipping")
            continue

        annotations = {}
        for cname in FOREGROUND_CLASSES:
            ann_path = annotation_index.get(cname, {}).get(name)
            if ann_path is None:
                continue
            annotations[cname] = decode_rootpainter_annotation(_load_rgba(ann_path))
        if not annotations:
            continue

        records = cut_production_crops(
            image, annotations,
            merge_radius=merge_radius,
            min_component_pixels=min_component_pixels,
        )
        if not records:
            continue
        n_source_images += 1
        for record in records:
            crop_h, crop_w = record["image"].shape[:2]
            sizes.append((crop_w, crop_h))

        if framing == FRAMING_WHOLE_PLATE:
            score_whole_plate_framed(image, records)
        else:
            pending.extend(records)
            while len(pending) >= crop_batch_size:
                score_crop_framed(pending[:crop_batch_size])
                del pending[:crop_batch_size]

        if progress and n_done % 20 == 0:
            print(f"  {n_done}/{len(all_filenames)} images, {len(sizes)} crops", flush=True)

    score_crop_framed(pending)
    pending.clear()

    stats = crop_size_summary(sizes)
    stats["n_source_images"] = n_source_images
    stats["framing"] = framing
    return accumulator.rows(), stats


def _write_csv(rows, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_caveats(out_csv: Path, stats: dict) -> Path:
    """The caveat lives next to the CSV rather than inside it: the CSV must
    keep sweep_operating_points.csv's exact column set so the two are directly
    comparable, and a comment line would break every reader of it."""
    path = out_csv.with_name(out_csv.stem + "_caveats.txt")
    lines = [
        f"Caveats for {out_csv.name}",
        "=" * (len(out_csv.name) + 13),
        "",
        CAVEAT,
        "",
        "Crops cut in this run:",
    ]
    for key in sorted(stats):
        lines.append(f"  {key}: {stats[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "multi" / "configs" / "training_config.yaml"))
    parser.add_argument("--data-root", default=None, help="Root dir paths are resolved against")
    parser.add_argument("--model", action="append", default=None,
                        help="Path to a trained multiclass checkpoint; repeatable. "
                             "The head is auto-detected from the checkpoint keys.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N annotated images (union across classes)")
    parser.add_argument("--crop-batch-size", type=int, default=DEFAULT_CROP_BATCH_SIZE)
    parser.add_argument("--merge-radius", type=int, default=SEEDLING_MERGE_RADIUS,
                        help="Dilation radius (px) used to group one seedling's organ strokes "
                             "into a single crop box")
    parser.add_argument("--min-component-pixels", type=int, default=MIN_COMPONENT_PIXELS)
    parser.add_argument("--framing", choices=(FRAMING_CROP, FRAMING_WHOLE_PLATE),
                        default=FRAMING_CROP,
                        help="Where inference runs. 'crop' (default) hands each model the "
                             "per-seedling crop, as production does. 'whole_plate' is the "
                             "CONTROL: inference runs on the 1024^2 frame and the probability "
                             "maps are then cut to the same crops, so the two runs score "
                             "identical pixels and differ only in framing. Comparing them is "
                             "the only way to tell a real framing effect from precision rising "
                             "because cropping removed ~99%% of the scored background.")
    parser.add_argument("--thresholds", default=None,
                        help="Comma-separated threshold grid (default: "
                             + ",".join(str(t) for t in DEFAULT_THRESHOLDS) + ")")
    parser.add_argument("--legacy-csv", default=None,
                        help="A previous sweep_operating_points.csv to read the legacy "
                             "1024^2 numbers from (defaults to "
                             "multi/results/sweep_operating_points.csv if it exists)")
    parser.add_argument("--out-csv", default=None,
                        help="Defaults to multi/results/evaluate_production_domain.csv")
    args = parser.parse_args()

    thresholds = tuple(float(t) for t in args.thresholds.split(",")) if args.thresholds \
        else DEFAULT_THRESHOLDS

    config = load_config(args.config, args.data_root)
    evaluation_cfg = config.get("evaluation", {})

    raw_dir = resolved_path(config, "raw_data_dir", section="evaluation")
    annotations_dirs = {
        cname: resolved_path(config, f"annotations_{cname}", section="evaluation")
        for cname in FOREGROUND_CLASSES
        if f"annotations_{cname}" in evaluation_cfg
    }
    weights = {
        cname: resolved_path(config, f"weights_{cname}")
        for cname in FOREGROUND_CLASSES
        if f"weights_{cname}" in config["paths"]
    }
    model_paths = {_default_model_name(p): p for p in (args.model or [])}

    print(CAVEAT)
    print()

    rows, stats = evaluate_production_domain(
        raw_dir=raw_dir,
        annotations_dirs=annotations_dirs,
        weights=weights,
        model_paths=model_paths,
        thresholds=thresholds,
        crop_batch_size=args.crop_batch_size,
        limit=args.limit,
        merge_radius=args.merge_radius,
        min_component_pixels=args.min_component_pixels,
        framing=args.framing,
    )

    out_csv = Path(args.out_csv) if args.out_csv else \
        REPO_ROOT / "multi" / "results" / "evaluate_production_domain.csv"
    _write_csv(rows, out_csv)
    caveat_path = _write_caveats(out_csv, stats)

    legacy_path = Path(args.legacy_csv) if args.legacy_csv else \
        REPO_ROOT / "multi" / "results" / "sweep_operating_points.csv"
    legacy_rows = load_legacy_rows(legacy_path)

    _print_crop_size_summary(stats, stats.get("n_source_images", 0))
    _print_sweep_table(rows)
    _print_legacy_comparison(legacy_comparison_table(rows, legacy_rows))
    _print_matched_recall(matched_recall_table(rows))
    print()
    print(CAVEAT)
    print(f"\nReport: {out_csv}")
    print(f"Caveats: {caveat_path}")
    print(f"Legacy numbers read from: {legacy_path if legacy_rows else 'PUBLISHED constants'}")


if __name__ == "__main__":
    main()
