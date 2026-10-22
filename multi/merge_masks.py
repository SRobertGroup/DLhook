#!/usr/bin/env python
"""CLI: merge the three shipped binary segmentation models (cotyledon,
hypocotyl, radicle) into multiclass pseudo-label PNGs, with human RootPainter
strokes overriding the automatic result where they exist.

This is a data-touching command meant to be run against the full dataset --
pass --limit for a quick smoke test.

Usage:
    python multi/merge_masks.py --config multi/configs/training_config.yaml [--data-root PATH]
                                 [--limit N] [--force] [--preview]

    # Disk-fed mode: read pre-computed per-class probability maps
    # ({crop_stem}-{cotyledon|hypocotyl|radicle}.png, 255 = high probability)
    # from a directory instead of running inference -- no torch import.
    python multi/merge_masks.py --config multi/configs/training_config.yaml
                                 --from-probs PATH/TO/PROB_DIR [--limit N] [--force] [--preview]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config, resolved_path
from src.mask_merge import (
    DEFAULT_CONFLICT_MARGIN,
    FOREGROUND_CLASSES,
    merge_dataset,
    merge_dataset_from_probs,
)


def _annotations_dirs(config: dict) -> dict:
    evaluation_cfg = config.get("evaluation", {})
    return {
        cname: resolved_path(config, f"annotations_{cname}", section="evaluation")
        for cname in FOREGROUND_CLASSES
        if f"annotations_{cname}" in evaluation_cfg
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to training_config.yaml")
    parser.add_argument("--data-root", default=None, help="Root dir paths are resolved against")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images")
    parser.add_argument("--force", action="store_true", help="Overwrite existing merged masks")
    parser.add_argument("--preview", action="store_true", help="Also write colorized preview PNGs")
    parser.add_argument(
        "--from-probs", default=None, metavar="DIR",
        help="Read per-class probability maps from DIR instead of running inference "
             "(leaves the inline inference path untouched)",
    )
    args = parser.parse_args()

    config = load_config(args.config, args.data_root)
    data_cfg = config["data"]
    annotations_dirs = _annotations_dirs(config)

    if args.from_probs:
        result = merge_dataset_from_probs(
            raw_dir=resolved_path(config, "raw_data_dir"),
            prob_dir=Path(args.from_probs),
            out_dir=resolved_path(config, "multiclass_masks_dir"),
            report_path=resolved_path(config, "mask_merge_report"),
            annotations_dirs=annotations_dirs,
            thresholds=data_cfg.get("class_thresholds"),
            conflict_margin=data_cfg.get("conflict_margin", DEFAULT_CONFLICT_MARGIN),
            preview_dir=resolved_path(config, "multiclass_preview_dir") if args.preview else None,
            limit=args.limit,
            force=args.force,
        )
    else:
        weights = {
            cname: resolved_path(config, f"weights_{cname}")
            for cname in FOREGROUND_CLASSES
            if f"weights_{cname}" in config["paths"]
        }

        result = merge_dataset(
            raw_dir=resolved_path(config, "raw_data_dir"),
            out_dir=resolved_path(config, "multiclass_masks_dir"),
            report_path=resolved_path(config, "mask_merge_report"),
            weights=weights,
            annotations_dirs=annotations_dirs,
            thresholds=data_cfg.get("class_thresholds"),
            conflict_margin=data_cfg.get("conflict_margin", DEFAULT_CONFLICT_MARGIN),
            preview_dir=resolved_path(config, "multiclass_preview_dir") if args.preview else None,
            limit=args.limit,
            force=args.force,
        )

    print(f"Processed: {len(result.processed)}")
    print(f"Skipped (already merged, use --force to redo): {len(result.skipped)}")
    print(f"Report: {result.report_path}")


if __name__ == "__main__":
    main()
