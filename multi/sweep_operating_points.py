#!/usr/bin/env python
"""CLI: score the 4-class student and its three binary teachers across a grid
of probability thresholds, so the comparison between them stops depending on
which decision rule each one happens to be read out with.

Why this script exists
----------------------
`multi/evaluate_multiclass.py` scores the two sides at *different* operating
points, by design and with a documented rationale ("take each network's own
top pick"):

    teacher :  pred_fg = binary_probs[cname][i] > 0.5      # >= 50% confidence
    student :  pred_fg = multiclass_labels[i] == CLASS_INDEX[cname]   # argmax

Over a 4-way softmax, argmax can select a class with as little as ~26%
probability, so the student's rule is systematically the more permissive of
the two. A permissive rule buys recall and spends precision; the published
result (student recall near the teachers', student precision far below them)
has exactly that shape, which means the single-point comparison cannot by
itself distinguish "the student is worse" from "the student is read out at a
looser operating point". Neither can it confirm the readout story -- the gap
may simply be real.

This script removes the question by sweeping BOTH sides over the same
threshold grid on the same pixels: for a threshold `t` and class `c` a pixel
is that class's foreground where `prob[c] > t`, for teacher and student
alike. The student's `argmax` readout is additionally scored as its own row,
so the already-published numbers appear in the same table as the sweep.

Nothing here re-implements scoring. `ClassScorer`, `decode_rootpainter_
annotation` and `discover_annotation_files` are imported from
evaluate_multiclass.py / src/mask_merge.py, so "defined pixels only", the
micro/macro split and the `n < 0.05 recall` count all behave identically to
the published evaluation -- that is what makes the two tables comparable.
`check_reproduction()` asserts that identity numerically: teacher rows at
t=0.5 and student `argmax` rows must land on the published figures, or the
sweep is not measuring the same thing and its other rows mean nothing.

The headline number
-------------------
`matched_recall_table()` answers the only question a threshold-free
comparison can't: at the recall each teacher reaches at its own 0.5
operating point, what precision does the student reach? The student's
precision/recall curve is interpolated to the teacher's recall. Where the
student's curve never reaches that recall at all, the row says so rather
than extrapolating.

Usage:
    python multi/sweep_operating_points.py --config multi/configs/training_config.yaml \\
        --model multi/results/run_groupnorm_head_baseline/best.pt \\
        --model multi/results/run_plain_head/best.pt

    # Quick smoke run over a handful of annotated images:
    python multi/sweep_operating_points.py --config multi/configs/training_config.yaml \\
        --model multi/results/run_plain_head/best.pt --limit 8
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

ensure_repo_root_importable()

# Imported, never redefined: the sweep is only meaningful if it pools tp/fp/fn
# exactly the way the published evaluation does. Imported by its package path
# (rather than the bare `evaluate_multiclass` the sys.path line above would
# also allow) so there is exactly one copy of ClassScorer in the process,
# whether this module is run as a script or imported as multi.sweep_operating_points.
from multi.evaluate_multiclass import (  # noqa: E402
    DEFAULT_CHUNK_SIZE,
    MULTICLASS_PATCH_SIZE,
    ClassScorer,
    _load_rgba,
)

# Both sides are swept over this identical grid. 0.5 must be a member: the
# teacher's row at 0.5 is the published operating point and the anchor of both
# the reproduction check and the matched-recall comparison.
DEFAULT_THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
TEACHER_OPERATING_POINT = 0.5

READOUT_THRESHOLD = "threshold"
READOUT_ARGMAX = "argmax"

# `head` column value for the three binary teachers. They are UNetGNRes too,
# but with a 2-class output and no multiclass head choice to record, so the
# column says what kind of model the row is rather than repeating "groupnorm".
TEACHER_HEAD = "binary"

REPORT_FIELDNAMES = [
    "model", "head", "class", "readout", "threshold",
    "n_images", "n_images_scored", "n_images_below_0.05_recall",
    "micro_precision", "micro_recall", "macro_precision", "macro_recall",
]

# The numbers this harness must reproduce before any other row in its output
# is worth reading -- multi/results/run_groupnorm_head_baseline/*.csv, quoted
# in multi/session_recap.md's "Update -- 2026-09-17" table. Keyed by
# (model-name predicate, class, readout): see check_reproduction.
PUBLISHED = {
    ("cotyledon_v5", "cotyledon", READOUT_THRESHOLD): (0.902, 0.921),
    ("hypocot_v5", "hypocotyl", READOUT_THRESHOLD): (0.957, 0.960),
    ("germ_v2", "radicle", READOUT_THRESHOLD): (0.654, 0.476),
    ("run_groupnorm_head_baseline", "cotyledon", READOUT_ARGMAX): (0.379, 0.865),
    ("run_groupnorm_head_baseline", "hypocotyl", READOUT_ARGMAX): (0.156, 0.925),
    ("run_groupnorm_head_baseline", "radicle", READOUT_ARGMAX): (0.186, 0.252),
}
REPRODUCTION_TOLERANCE = 0.002


class SweepAccumulator:
    """Owns one `ClassScorer` per (model, head, class, readout, threshold) cell
    and feeds them from probability maps.

    Kept separate from the I/O in `sweep()` so the accumulation rules are
    testable on small synthetic arrays without a checkpoint, a GPU or the
    annotation folders.
    """

    def __init__(self, thresholds=DEFAULT_THRESHOLDS):
        self.thresholds = tuple(thresholds)
        self.scorers: dict[tuple, ClassScorer] = {}

    def _scorer(self, model, head, cname, readout, threshold) -> ClassScorer:
        """`threshold` is None for the argmax readout, NOT float("nan").
        NaN never compares equal to itself, so a nan inside a dict key makes
        every lookup miss and silently allocates a fresh scorer per call --
        which pooled nothing and reported one row per image. The nan only
        appears again in `rows()`, where it is a value and not a key."""
        key = (model, head, cname, readout, threshold)
        scorer = self.scorers.get(key)
        if scorer is None:
            scorer = ClassScorer()
            self.scorers[key] = scorer
        return scorer

    def add_threshold_readouts(self, model, head, cname, prob_map, annotation) -> None:
        """Score one image's foreground-probability map for one class at every
        threshold in the grid. `prob_map` is float in [0, 1] and the same HxW
        as the annotation; `annotation` is the (foreground, background,
        defined) triple `decode_rootpainter_annotation` returns.

        `prob > t` is the same comparison evaluate_multiclass.py hardcodes at
        t=0.5, so the t=0.5 row of this sweep is that script's baseline row by
        construction, not by reimplementation.
        """
        foreground, background, defined = annotation
        for threshold in self.thresholds:
            pred_fg = prob_map > threshold
            scorer = self._scorer(model, head, cname, READOUT_THRESHOLD, float(threshold))
            scorer.add(pred_fg, foreground, background, defined)

    def add_argmax_readout(self, model, head, cname, label_map, annotation) -> None:
        """Score one image under the student's published readout: a pixel is
        this class's foreground exactly where the 4-way argmax picked it.
        `label_map` holds class indices (0..num_classes-1)."""
        foreground, background, defined = annotation
        pred_fg = label_map == CLASS_INDEX[cname]
        scorer = self._scorer(model, head, cname, READOUT_ARGMAX, None)
        scorer.add(pred_fg, foreground, background, defined)

    def rows(self) -> list[dict]:
        rows = []
        for (model, head, cname, readout, threshold), scorer in self.scorers.items():
            rows.append({
                "model": model,
                "head": head,
                "class": cname,
                "readout": readout,
                # The argmax readout has no threshold; reported as nan so the
                # column stays numeric for every row.
                "threshold": float("nan") if threshold is None else threshold,
                **scorer.summary(),
            })
        rows.sort(key=lambda r: (
            r["class"],
            r["model"],
            r["readout"] != READOUT_ARGMAX,
            0.0 if np.isnan(r["threshold"]) else r["threshold"],
        ))
        return rows


def interpolate_precision_at_recall(points, target_recall: float) -> dict:
    """Linearly interpolate a precision/recall curve to `target_recall`.

    `points` is an iterable of (threshold, precision, recall) triples for one
    (model, class) sweep, in any order. Points whose precision or recall is
    nan are dropped (a threshold at which the model predicts no foreground at
    all anywhere has undefined precision, which is common at the top of the
    grid and must not be interpolated through).

    Returns a dict with `reachable` False -- and no invented precision -- when
    the curve's maximum recall is below the target. Extrapolating past the end
    of a sweep is precisely the thing this script exists to avoid.
    """
    usable = [
        (float(t), float(p), float(r)) for (t, p, r) in points
        if not (np.isnan(p) or np.isnan(r))
    ]
    result = {
        "reachable": False,
        "precision": float("nan"),
        "threshold": float("nan"),
        "recall": float(target_recall),
        "max_recall": float("nan"),
        "precision_at_max_recall": float("nan"),
        "note": "",
    }
    if not usable:
        result["note"] = "no usable sweep points"
        return result

    usable.sort(key=lambda item: item[2])  # by recall, ascending
    max_threshold, max_precision, max_recall = usable[-1]
    result["max_recall"] = max_recall
    result["precision_at_max_recall"] = max_precision

    if target_recall > max_recall:
        result["note"] = (
            f"student curve never reaches recall {target_recall:.3f} "
            f"(max {max_recall:.3f} at threshold {max_threshold:g})"
        )
        return result

    if target_recall <= usable[0][2]:
        # Target sits at or below the lowest-recall sweep point. Its precision
        # is a valid, un-extrapolated reading of the curve at a recall no
        # lower than asked for, so report it and flag that no bracket existed.
        threshold, precision, recall = usable[0]
        result.update({
            "reachable": True, "precision": precision, "threshold": threshold,
            "recall": recall,
            "note": "target below sweep range; reporting lowest-recall point",
        })
        return result

    for (t_lo, p_lo, r_lo), (t_hi, p_hi, r_hi) in zip(usable, usable[1:]):
        if r_lo <= target_recall <= r_hi:
            span = r_hi - r_lo
            frac = 0.0 if span == 0 else (target_recall - r_lo) / span
            result.update({
                "reachable": True,
                "precision": p_lo + frac * (p_hi - p_lo),
                "threshold": t_lo + frac * (t_hi - t_lo),
            })
            return result

    result["note"] = "no bracketing pair found"
    return result


def _row_points(rows, model, cname):
    return [
        (r["threshold"], r["micro_precision"], r["micro_recall"])
        for r in rows
        if r["model"] == model and r["class"] == cname and r["readout"] == READOUT_THRESHOLD
    ]


def _find_row(rows, cname, readout, model_contains=None, threshold=None):
    for row in rows:
        if row["class"] != cname or row["readout"] != readout:
            continue
        if model_contains is not None and model_contains not in row["model"]:
            continue
        if threshold is not None and not np.isclose(row["threshold"], threshold):
            continue
        return row
    return None


def teacher_models(rows) -> list[str]:
    return sorted({r["model"] for r in rows if r["head"] == TEACHER_HEAD})


def student_models(rows) -> list[str]:
    return sorted({r["model"] for r in rows if r["head"] != TEACHER_HEAD})


def matched_recall_table(rows, operating_point: float = TEACHER_OPERATING_POINT) -> list[dict]:
    """The deliverable: for each (student, class), the student's precision at
    the recall its teacher achieves at the teacher's own `operating_point`
    threshold. Micro precision/recall throughout, since that is what the
    published table headlines."""
    table = []
    for cname in FOREGROUND_CLASSES:
        teacher_row = None
        for model in teacher_models(rows):
            candidate = _find_row(rows, cname, READOUT_THRESHOLD, model_contains=model,
                                  threshold=operating_point)
            if candidate is not None:
                teacher_row = candidate
                break
        if teacher_row is None:
            continue
        target_recall = teacher_row["micro_recall"]
        for student in student_models(rows):
            points = _row_points(rows, student, cname)
            if not points:
                continue
            matched = interpolate_precision_at_recall(points, target_recall)
            argmax_row = _find_row(rows, cname, READOUT_ARGMAX, model_contains=student)
            table.append({
                "class": cname,
                "student": student,
                "teacher": teacher_row["model"],
                "teacher_precision": teacher_row["micro_precision"],
                "teacher_recall": target_recall,
                "student_argmax_precision": argmax_row["micro_precision"] if argmax_row else float("nan"),
                "student_argmax_recall": argmax_row["micro_recall"] if argmax_row else float("nan"),
                "student_precision_at_teacher_recall": matched["precision"],
                "student_threshold_at_teacher_recall": matched["threshold"],
                "reachable": matched["reachable"],
                "student_max_recall": matched["max_recall"],
                "student_precision_at_max_recall": matched["precision_at_max_recall"],
                "note": matched["note"],
            })
    return table


def check_reproduction(rows, published=PUBLISHED, tolerance: float = REPRODUCTION_TOLERANCE):
    """Compare this sweep's anchor rows against the already-published figures.

    Returns a list of (label, expected, actual, ok) tuples, one per entry in
    `published` that a matching row exists for. A False anywhere means the
    harness is not measuring what evaluate_multiclass.py measured, and every
    other row it produced should be discarded rather than interpreted.
    """
    checks = []
    for (model_contains, cname, readout), (exp_p, exp_r) in sorted(published.items()):
        threshold = TEACHER_OPERATING_POINT if readout == READOUT_THRESHOLD else None
        row = _find_row(rows, cname, readout, model_contains=model_contains, threshold=threshold)
        if row is None:
            continue
        got_p, got_r = row["micro_precision"], row["micro_recall"]
        ok = bool(abs(got_p - exp_p) <= tolerance and abs(got_r - exp_r) <= tolerance)
        label = f"{model_contains} / {cname} / {readout}"
        checks.append((label, (exp_p, exp_r), (got_p, got_r), ok))
    return checks


def sweep(
    raw_dir,
    annotations_dirs: dict,
    weights: dict,
    model_paths: dict,
    thresholds=DEFAULT_THRESHOLDS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    limit: int | None = None,
    progress: bool = True,
) -> list[dict]:
    """Run every model over the annotation set once and score all readouts.

    `weights` maps FOREGROUND_CLASSES to a binary teacher checkpoint;
    `model_paths` maps a display name to a 4-class student checkpoint (the
    head is inferred from the checkpoint's own keys). Every model sees the
    same chunk of images, so all rows are scored on identical pixels.

    Torch-dependent imports are local, mirroring evaluate_multiclass.evaluate,
    so the pure scoring/interpolation helpers above stay importable without a
    working torch install.
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

    students = {}
    for name, path in model_paths.items():
        # Same geometry evaluate_multiclass.py uses: the checkpoints were
        # trained at data.patch_size 252, not the GUI's 572 tile.
        students[name] = MulticlassInference(
            path,
            in_size=MULTICLASS_PATCH_SIZE + 2 * MARGIN,
            out_size=MULTICLASS_PATCH_SIZE,
            margin=MARGIN,
        )

    accumulator = SweepAccumulator(thresholds)

    for start in range(0, len(all_filenames), chunk_size):
        chunk_names = all_filenames[start:start + chunk_size]
        images, valid_names = [], []
        for name in chunk_names:
            raw_path = raw_dir / name
            if not raw_path.exists():
                warnings.warn(f"No matching raw image found for annotation {name}; skipping")
                continue
            image = cv2.imread(str(raw_path))
            if image is None:
                warnings.warn(f"Could not read raw image {raw_path}; skipping")
                continue
            images.append(image)
            valid_names.append(name)
        if not images:
            continue

        binary_probs = {
            cname: predictor._segment_many(images) for cname, predictor in binary_predictors.items()
        }
        # One forward pass per student per chunk: the argmax readout is taken
        # from this same stitched softmax rather than a second inference call.
        # tests/test_multiclass_inference.py pins that
        # `segment_many_argmax(..., return_probs=True)` and the default
        # uint8 label output are the argmax of one another, so deriving it
        # here is equivalent to calling the default readout.
        student_probs = {
            name: model.segment_many_argmax(images, return_probs=True)
            for name, model in students.items()
        }

        for i, name in enumerate(valid_names):
            for cname in FOREGROUND_CLASSES:
                ann_path = annotation_index.get(cname, {}).get(name)
                if ann_path is None:
                    continue
                annotation = decode_rootpainter_annotation(_load_rgba(ann_path))

                if cname in binary_probs:
                    accumulator.add_threshold_readouts(
                        teacher_names[cname], TEACHER_HEAD, cname,
                        binary_probs[cname][i], annotation,
                    )

                for student_name, model in students.items():
                    probs = student_probs[student_name][i]
                    accumulator.add_threshold_readouts(
                        student_name, model.head, cname,
                        probs[CLASS_INDEX[cname]], annotation,
                    )
                    accumulator.add_argmax_readout(
                        student_name, model.head, cname,
                        np.argmax(probs, axis=0).astype(np.uint8), annotation,
                    )

        if progress:
            done = min(start + chunk_size, len(all_filenames))
            print(f"  {done}/{len(all_filenames)} images", flush=True)

    return accumulator.rows()


def _fmt(value) -> str:
    return "    nan" if value is None or np.isnan(value) else f"{value:7.3f}"


def _print_sweep_table(rows) -> None:
    header = (f"{'class':<10} {'model':<30} {'head':<10} {'readout':<9} {'thr':>5} "
              f"{'scored':>6} {'<0.05R':>6} {'microP':>7} {'microR':>7} {'macroP':>7} {'macroR':>7}")
    print(header)
    print("-" * len(header))
    for row in rows:
        thr = "  --" if np.isnan(row["threshold"]) else f"{row['threshold']:.2f}"
        print(
            f"{row['class']:<10} {row['model']:<30} {row['head']:<10} {row['readout']:<9} "
            f"{thr:>5} {row['n_images_scored']:>6} {row['n_images_below_0.05_recall']:>6} "
            f"{_fmt(row['micro_precision'])} {_fmt(row['micro_recall'])} "
            f"{_fmt(row['macro_precision'])} {_fmt(row['macro_recall'])}"
        )


def _print_reproduction_check(checks) -> bool:
    print("\n=== Correctness check: does this harness reproduce the published numbers? ===")
    header = f"{'anchor row':<52} {'expected P/R':>16} {'got P/R':>16}  ok"
    print(header)
    print("-" * len(header))
    all_ok = True
    for label, (exp_p, exp_r), (got_p, got_r), ok in checks:
        all_ok = all_ok and ok
        print(f"{label:<52} {exp_p:7.3f}/{exp_r:6.3f} {got_p:7.3f}/{got_r:6.3f}  "
              f"{'PASS' if ok else 'FAIL'}")
    if not checks:
        print("(no anchor rows present in this run -- nothing checked)")
        return False
    print("\nOVERALL:", "PASS" if all_ok else "FAIL -- do not interpret the sweep")
    return all_ok


def _print_matched_recall(table) -> None:
    print("\n=== Student precision at the teacher's own 0.5-threshold recall ===")
    header = (f"{'class':<10} {'student':<30} {'teacher':<14} {'teachP':>7} {'teachR':>7} "
              f"{'argmaxP':>8} {'matchedP':>9} {'@thr':>6}")
    print(header)
    print("-" * len(header))
    for entry in table:
        matched = "unreached" if not entry["reachable"] else f"{entry['student_precision_at_teacher_recall']:9.3f}"
        thr = "    --" if np.isnan(entry["student_threshold_at_teacher_recall"]) \
            else f"{entry['student_threshold_at_teacher_recall']:6.3f}"
        print(
            f"{entry['class']:<10} {entry['student']:<30} {entry['teacher']:<14} "
            f"{entry['teacher_precision']:7.3f} {entry['teacher_recall']:7.3f} "
            f"{entry['student_argmax_precision']:8.3f} {matched:>9} {thr}"
        )
        if entry["note"]:
            print(f"{'':<10} note: {entry['note']}")


def _write_csv(rows, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _default_model_name(path) -> str:
    """Name a student checkpoint by the run folder it lives in
    (multi/results/run_plain_head/best.pt -> "run_plain_head"), since every
    run writes a file called best.pt."""
    path = Path(path)
    return path.parent.name if path.parent.name not in ("", ".") else path.stem


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
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--thresholds", default=None,
                        help="Comma-separated threshold grid (default: "
                             + ",".join(str(t) for t in DEFAULT_THRESHOLDS) + ")")
    parser.add_argument("--out-csv", default=None,
                        help="Defaults to multi/results/sweep_operating_points.csv")
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

    rows = sweep(
        raw_dir=raw_dir,
        annotations_dirs=annotations_dirs,
        weights=weights,
        model_paths=model_paths,
        thresholds=thresholds,
        chunk_size=args.chunk_size,
        limit=args.limit,
    )

    out_csv = Path(args.out_csv) if args.out_csv else \
        REPO_ROOT / "multi" / "results" / "sweep_operating_points.csv"
    _write_csv(rows, out_csv)
    _print_sweep_table(rows)
    _print_reproduction_check(check_reproduction(rows))
    _print_matched_recall(matched_recall_table(rows))
    print(f"\nReport: {out_csv}")


if __name__ == "__main__":
    main()
