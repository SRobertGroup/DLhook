#!/usr/bin/env python
"""CLI: score a trained multiclass model, side-by-side with the three
shipped binary models it distills (cotyledon_v5, hypocot_v5, germ_v1),
against the 441 human RootPainter-annotated crops in `training_dataset/dlhook/`.

Those 441 crops are held out validation data: their RootPainter annotations
live outside the repo (see `evaluation:` in training_config.yaml) and are
never used to build a training patch. "Germination" in this repo means the
RADICLE -- germ_v1.pkl / dlhook_germ annotations score the radicle class.

Fair comparison, multiclass argmax vs binary threshold(0.5)
-------------------------------------------------------------
A binary model outputs one foreground probability per pixel; thresholding it
at 0.5 is exactly "pick whichever of {background, foreground} that model's
own 2-way softmax calls more likely" -- there is no tuned threshold here, it
is the network's own most-likely call. The multiclass model instead makes
one 4-way choice per pixel (background vs cotyledon vs hypocotyl vs radicle)
via `argmax` over its softmax. To score the multiclass model against one
class the same way -- "the network's own most-likely call" -- a pixel counts
as that class's foreground exactly where `argmax == class_index`, not where
that class's softmax channel merely exceeds 0.5 (which, unlike the binary
2-way case, is not equivalent to being the argmax in a 4-way softmax). Both
readouts are therefore "take the model's own top pick," applied consistently
to a 2-way and a 4-way head respectively -- the only threshold-free way to
compare them.

Because the multiclass model makes ONE mutually-exclusive choice per pixel
while the three binary models each decide independently (their foreground
calls can overlap on a pixel, the multiclass model's cannot), the two are
still not decision-theoretically identical -- this is an intrinsic property
of distilling three independent binary teachers into one shared-softmax
student, not an artifact of how this script scores them.

Score on DEFINED pixels only
-----------------------------
RootPainter strokes are sparse (median ~1% of pixels). `decode_rootpainter_
annotation` (reused from `src/mask_merge.py`) returns (foreground,
background, defined); pixels outside `defined` are excluded from every
count -- scoring them would be meaningless (see `_per_image_stats`).

Micro vs macro -- why both are a hard requirement
----------------------------------------------------
Micro precision/recall pools tp/fp/fn across every scored pixel in every
image; macro averages each image's own precision/recall. A model that
entirely misses one image in twelve (0 recall on that image) barely moves
micro recall (that image contributes few pixels to the pool) but should
tank macro recall (one of twelve images scores 0) -- pooling hides the
"misses whole images" failure mode that averaging surfaces. See the CLAUDE.md-
adjacent finding this script was written to reproduce: germ_v1 misses
roughly 1 image in 12 entirely (micro recall ~0.86, macro recall ~0.61 on
the same data).

Usage:
    python multi/evaluate_multiclass.py --config multi/configs/training_config.yaml \\
        --model multi/results/models/best.pt

    # Works even before any multiclass checkpoint exists:
    python multi/evaluate_multiclass.py --config multi/configs/training_config.yaml --baseline-only

    # Quick smoke test on a handful of annotated images:
    python multi/evaluate_multiclass.py --config multi/configs/training_config.yaml \\
        --baseline-only --limit 8
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

DEFAULT_CHUNK_SIZE = 8
LOW_RECALL_THRESHOLD = 0.05

# The 4-class checkpoint under test was trained at training_config.yaml's
# `data.patch_size: 252` (see that file's comment: the real crops are narrow
# strips, median 74x246px, so 252 -- the smallest valid UNetGNRes patch size
# -- was chosen over the GUI's default 572 to avoid reflect-padding ~94% of
# every tile with synthetic context). Score it at that same geometry rather
# than MulticlassInference's 572/560/6 default, which is UNetInference's
# live-GUI tile size, not this checkpoint's training size.
MULTICLASS_PATCH_SIZE = 252

REPORT_FIELDNAMES = [
    "model", "class", "n_images", "n_images_scored", "n_images_below_0.05_recall",
    "micro_precision", "micro_recall", "macro_precision", "macro_recall",
]


def _load_rgba(path: Path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.array(im)


def _precision_recall(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return precision, recall


def _per_image_stats(pred_fg: np.ndarray, foreground: np.ndarray, background: np.ndarray,
                      defined: np.ndarray):
    """tp/fp/fn for one image, one class, restricted to DEFINED pixels only
    -- `foreground`/`background` are masked by `defined` before anything
    else touches them, so an undefined pixel can never contribute to any
    count even if it happens to be flagged as foreground or background by
    some other bug upstream."""
    fg = foreground & defined
    bg = background & defined
    tp = int(np.count_nonzero(pred_fg & fg))
    fn = int(np.count_nonzero(~pred_fg & fg))
    fp = int(np.count_nonzero(pred_fg & bg))
    n_fg = tp + fn
    return tp, fp, fn, n_fg


class ClassScorer:
    """Accumulates per-image and pooled (micro) tp/fp/fn for one (model,
    class) pair, then reduces to micro + macro precision/recall plus the
    below-0.05-recall image count."""

    def __init__(self):
        self.micro_tp = self.micro_fp = self.micro_fn = 0
        self.per_image_precision: list[float] = []
        self.per_image_recall: list[float] = []
        self.n_images = 0
        self.n_low_recall = 0

    def add(self, pred_fg: np.ndarray, foreground: np.ndarray, background: np.ndarray,
            defined: np.ndarray) -> None:
        tp, fp, fn, n_fg = _per_image_stats(pred_fg, foreground, background, defined)
        self.micro_tp += tp
        self.micro_fp += fp
        self.micro_fn += fn
        self.n_images += 1
        if n_fg == 0:
            # No foreground-stroke pixels for this class in this image --
            # recall is undefined for it, so it is excluded from the macro
            # average and the below-threshold count (mirrors
            # mask_merge.validate_pseudo_labels's same convention).
            return
        precision, recall = _precision_recall(tp, fp, fn)
        self.per_image_precision.append(precision)
        self.per_image_recall.append(recall)
        if recall < LOW_RECALL_THRESHOLD:
            self.n_low_recall += 1

    def summary(self) -> dict:
        micro_precision, micro_recall = _precision_recall(self.micro_tp, self.micro_fp, self.micro_fn)
        macro_precision = float(np.mean(self.per_image_precision)) if self.per_image_precision else float("nan")
        macro_recall = float(np.mean(self.per_image_recall)) if self.per_image_recall else float("nan")
        return {
            "n_images": self.n_images,
            "n_images_scored": len(self.per_image_recall),
            "n_images_below_0.05_recall": self.n_low_recall,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
        }


def evaluate(
    raw_dir,
    annotations_dirs: dict,
    weights: dict,
    model_path=None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    limit: int | None = None,
) -> list[dict]:
    """Run the evaluation and return a list of report-row dicts (one per
    (model, class) pair). `weights` maps FOREGROUND_CLASSES to a binary
    checkpoint path (the baselines); `model_path`, if given, is a trained
    4-class checkpoint scored alongside them. Import of torch-dependent
    predictors is local so `--help` and pure scoring logic stay importable
    without a working torch install.
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
    multiclass_model = MulticlassInference(
        model_path,
        in_size=MULTICLASS_PATCH_SIZE + 2 * MARGIN,
        out_size=MULTICLASS_PATCH_SIZE,
        margin=MARGIN,
    ) if model_path is not None else None

    model_names = {cname: Path(weights[cname]).stem for cname in weights}
    scorers = {(model_names[cname], cname): ClassScorer() for cname in weights}
    if multiclass_model is not None:
        for cname in FOREGROUND_CLASSES:
            scorers[("multiclass", cname)] = ClassScorer()

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
        multiclass_labels = multiclass_model.segment_many_argmax(images) if multiclass_model else None

        for i, name in enumerate(valid_names):
            for cname in FOREGROUND_CLASSES:
                ann_path = annotation_index.get(cname, {}).get(name)
                if ann_path is None:
                    continue
                ann = _load_rgba(ann_path)
                foreground, background, defined = decode_rootpainter_annotation(ann)

                if cname in binary_probs:
                    pred_fg_baseline = binary_probs[cname][i] > 0.5
                    scorers[(model_names[cname], cname)].add(pred_fg_baseline, foreground, background, defined)

                if multiclass_labels is not None:
                    pred_fg_multi = multiclass_labels[i] == CLASS_INDEX[cname]
                    scorers[("multiclass", cname)].add(pred_fg_multi, foreground, background, defined)

    rows = []
    for (model_name, cname), scorer in scorers.items():
        row = {"model": model_name, "class": cname, **scorer.summary()}
        rows.append(row)
    rows.sort(key=lambda r: (r["class"], r["model"] != "multiclass", r["model"]))
    return rows


def _print_table(rows: list[dict]) -> None:
    header = f"{'class':<10} {'model':<14} {'n_img':>6} {'scored':>7} {'<0.05rec':>9} " \
             f"{'microP':>7} {'microR':>7} {'macroP':>7} {'macroR':>7}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['class']:<10} {row['model']:<14} {row['n_images']:>6} "
            f"{row['n_images_scored']:>7} {row['n_images_below_0.05_recall']:>9} "
            f"{row['micro_precision']:>7.3f} {row['micro_recall']:>7.3f} "
            f"{row['macro_precision']:>7.3f} {row['macro_recall']:>7.3f}"
        )


def _write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "multi" / "configs" / "training_config.yaml"),
                         help="Path to training_config.yaml")
    parser.add_argument("--data-root", default=None, help="Root dir paths are resolved against")
    parser.add_argument("--model", default=None,
                         help="Path to a trained multiclass checkpoint (state_dict .pt). "
                              "Required unless --baseline-only.")
    parser.add_argument("--baseline-only", action="store_true",
                         help="Score only the three binary models -- works even before any "
                              "multiclass checkpoint exists.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only evaluate the first N annotated images (union across classes)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--out-csv", default=None,
                         help="Defaults to multi/results/evaluate_multiclass.csv")
    args = parser.parse_args()

    if not args.baseline_only and not args.model:
        parser.error("--model is required unless --baseline-only is passed")

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

    rows = evaluate(
        raw_dir=raw_dir,
        annotations_dirs=annotations_dirs,
        weights=weights,
        model_path=None if args.baseline_only else args.model,
        chunk_size=args.chunk_size,
        limit=args.limit,
    )

    out_csv = Path(args.out_csv) if args.out_csv else REPO_ROOT / "multi" / "results" / "evaluate_multiclass.csv"
    _write_csv(rows, out_csv)
    _print_table(rows)
    print(f"\nReport: {out_csv}")


if __name__ == "__main__":
    main()
