#!/usr/bin/env python
"""Segment a flat folder of training crops (as produced by
multi/recrop_plates.py) into per-class uint8 foreground-probability maps,
using the three shipped binary RootPainter models. A later stage
(multi/src/mask_merge.py) merges these per-class probability maps into
multiclass labels; this script's only job is to write them.

This is a data-prep CLI, run from the dlhook_env (needs cv2/numpy/torch).
It imports models/UNetInference.py directly (not seedling_measurment.py/ui/*,
which construct Tk widgets and load CUDA/superres models at import time).

========================================================================
CRITICAL CONVENTION -- read this before touching any output of this script
========================================================================
Every PNG this script writes is a PROBABILITY MAP, not a binary mask:

    255 = HIGH probability of foreground (the class in question)
      0 = LOW probability

This is the OPPOSITE of this repo's legacy on-disk MASK convention, where
MaskStore.dump() (utils/mask_store.py:115-141) applies cv2.bitwise_not so
that *masks* written to disk read 0 = foreground. This script never calls
dump() and never inverts its output. Getting this backwards here would
silently poison the entire training set built from these probability maps
-- if you need the legacy mask convention somewhere downstream, invert
explicitly and deliberately at that boundary, not here.

Class -> weight file. Note: this repo's "germination" label means the
RADICLE, not the seed coat splitting open -- germ_v2.pkl is the radicle
model, despite the filename:
    cotyledon -> weights/RootPainter_weights/cotyledon_v5.pkl
    hypocotyl -> weights/RootPainter_weights/hypocot_v5.pkl
    radicle   -> weights/RootPainter_weights/germ_v2.pkl (RootPainter round
                 000022, adopted in place of germ_v1/round 000019 -- higher
                 micro precision against the val/ annotations, see
                 multi/configs/training_config.yaml's weights_radicle)
cotyledon_v3.pkl is deliberately never used here (byte-identical duplicate
of cotyledon_v2.pkl).

Use --classes to regenerate only a subset of the three per-class outputs
(e.g. --classes radicle after swapping the radicle weight file) without
touching the other classes' already-written PNGs -- --force alone reruns
ALL three models (and rewrites their PNGs, changing mtimes) for any crop
missing even one output.

This uses UNetInference._segment_many (float32 probabilities at native
H x W), NOT predict_files, which thresholds to a binary mask at 0.5 and
destroys exactly the probability information this script exists to keep.

Output naming: "{crop_stem}-{class}.png", e.g. for a crop named
"0-crop-MB_MB_1_64.png" the three outputs are
"0-crop-MB_MB_1_64-cotyledon.png", "...-hypocotyl.png", "...-radicle.png".

Resumable: a crop whose three class outputs already all exist is skipped
unless --force. Crops are decoded in IMAGE_CHUNK-sized chunks
(models/UNetInference.py) to bound host memory; each model's
_segment_many() still batches tiles across a whole chunk internally.

Usage:
    python multi/segment_crops.py --crops cropped_training_set --out cropped_training_set_seg
    python multi/segment_crops.py --crops cropped_training_set --out cropped_training_set_seg --cpu
    python multi/segment_crops.py --crops cropped_training_set --out cropped_training_set_seg --force
    python multi/segment_crops.py --crops cropped_training_set --out cropped_training_set_seg --classes radicle --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# multi/ on sys.path so `from src...` resolves to multi/src (matches
# multi/recrop_plates.py's convention); the repo root on sys.path so
# `models.UNetInference` is importable without importing
# seedling_measurment.py itself.
_MULTI_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MULTI_DIR.parent
sys.path.insert(0, str(_MULTI_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from src.recrop_geometry import list_image_files  # noqa: E402

WEIGHTS_DIR = _REPO_ROOT / "weights" / "RootPainter_weights"

# class name -> weight file. Order here is also the write order per chunk.
CLASS_WEIGHTS = {
    "cotyledon": WEIGHTS_DIR / "cotyledon_v5.pkl",
    "hypocotyl": WEIGHTS_DIR / "hypocot_v5.pkl",
    "radicle": WEIGHTS_DIR / "germ_v2.pkl",
}


def output_path(out_dir: Path, crop_stem: str, class_name: str) -> Path:
    return out_dir / f"{crop_stem}-{class_name}.png"


def needs_processing(out_dir: Path, crop_stem: str, force: bool, class_names=None) -> bool:
    """True unless every requested class's output already exists for this
    crop and --force was not given -- the resumability check. `class_names`
    defaults to all of CLASS_WEIGHTS (the historical all-three behaviour);
    pass a subset (e.g. ["radicle"]) to check/regenerate only those classes
    without disturbing the others' existing outputs."""
    if force:
        return True
    class_names = class_names if class_names is not None else CLASS_WEIGHTS
    return not all(
        output_path(out_dir, crop_stem, cname).exists() for cname in class_names
    )


def prob_to_uint8(prob: np.ndarray) -> np.ndarray:
    """Foreground-probability map (float, expected range [0, 1]) -> uint8
    PNG payload, 255 = HIGH probability. See the module docstring's
    CRITICAL CONVENTION section before changing this -- do not invert."""
    return np.clip(np.round(prob * 255.0), 0, 255).astype(np.uint8)


def segment_crops(crops_dir: Path, out_dir: Path, force: bool, chunk_size: int,
                   get_predictor_fn, image_chunk_default: int, class_names=None):
    """Segment every crop in `crops_dir`, writing uint8 probability maps to
    `out_dir`. `class_names` restricts which of CLASS_WEIGHTS are run/written
    (default: all three) -- only the selected classes' predictors are loaded
    and only their output PNGs are read for resumability or written, so the
    unselected classes' existing files are never opened, let alone rewritten.
    Returns (n_processed, n_skipped, n_failed_to_read)."""
    chunk_size = chunk_size or image_chunk_default
    class_names = list(class_names) if class_names is not None else list(CLASS_WEIGHTS)
    unknown = [c for c in class_names if c not in CLASS_WEIGHTS]
    if unknown:
        raise ValueError(f"Unknown class name(s): {unknown}; choose from {list(CLASS_WEIGHTS)}")

    filenames = list_image_files(str(crops_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    n_skipped = 0
    for fname in filenames:
        stem = Path(fname).stem
        if needs_processing(out_dir, stem, force, class_names):
            pending.append((fname, stem))
        else:
            n_skipped += 1

    print(f"[INFO] {len(filenames)} crops found in {crops_dir}; "
          f"{n_skipped} already done, {len(pending)} to process "
          f"(classes: {', '.join(class_names)}).")

    if not pending:
        return 0, n_skipped, 0

    # Constructed only once there is confirmed work, so a fully-resumed run
    # (nothing pending) never loads a model at all.
    predictors = {cname: get_predictor_fn(str(CLASS_WEIGHTS[cname])) for cname in class_names}

    n_processed = 0
    n_failed = 0
    for start in range(0, len(pending), chunk_size):
        chunk = pending[start:start + chunk_size]
        valid_stems = []
        images = []
        for fname, stem in chunk:
            img = cv2.imread(str(crops_dir / fname))
            if img is None:
                print(f"[WARNING] Could not read: {fname}")
                n_failed += 1
                continue
            valid_stems.append(stem)
            images.append(img)

        if not images:
            continue

        # Images are decoded once per chunk and reused across all three
        # models -- each _segment_many call batches this chunk's tiles
        # internally for that one model.
        for cname, predictor in predictors.items():
            prob_maps = predictor._segment_many(images)
            for stem, prob in zip(valid_stems, prob_maps):
                cv2.imwrite(str(output_path(out_dir, stem, cname)), prob_to_uint8(prob))

        n_processed += len(valid_stems)
        print(f"[INFO] Segmented {n_processed}/{len(pending)} crops so far...")

    return n_processed, n_skipped, n_failed


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crops", required=True,
                         help="Flat folder of crop images (e.g. multi/recrop_plates.py's --output-dir)")
    parser.add_argument("--out", required=True,
                         help="Output folder for per-class uint8 probability maps")
    parser.add_argument("--cpu", action="store_true",
                         help="Force CPU inference even if CUDA is available")
    parser.add_argument("--force", action="store_true",
                         help="Re-segment crops whose outputs already exist (default: skip -- resumable)")
    parser.add_argument("--chunk-size", type=int, default=None,
                         help="Crops decoded per chunk (default: models.UNetInference.IMAGE_CHUNK)")
    parser.add_argument("--classes", nargs="+", default=None, choices=list(CLASS_WEIGHTS),
                         help="Restrict to these class(es) only (default: all three). "
                              "Unselected classes' existing outputs are never opened or rewritten -- "
                              "use this to regenerate one class (e.g. after swapping its weight file) "
                              "without touching the others' mtimes.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    if args.cpu:
        # Must happen before any UNetInference/predictor is constructed --
        # UNetInference.__init__ picks its device from torch.cuda.is_available()
        # at construction time. On this build torch.cuda.is_available() can
        # return True even with CUDA_VISIBLE_DEVICES="", so forcing CPU
        # requires this monkeypatch, not just an env var.
        import torch
        torch.cuda.is_available = lambda: False

    from models.UNetInference import IMAGE_CHUNK, get_predictor

    crops_dir = Path(args.crops)
    out_dir = Path(args.out)

    if not crops_dir.is_dir():
        print(f"--crops directory not found: {crops_dir}")
        return 1

    missing_weights = [str(p) for p in CLASS_WEIGHTS.values() if not p.exists()]
    if missing_weights:
        print("Missing weight file(s): " + ", ".join(missing_weights))
        return 1

    n_processed, n_skipped, n_failed = segment_crops(
        crops_dir, out_dir, args.force, args.chunk_size, get_predictor, IMAGE_CHUNK,
        class_names=args.classes,
    )

    print(f"Done. Processed {n_processed} crop(s), skipped {n_skipped} already-done, "
          f"{n_failed} unreadable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
