#!/usr/bin/env python
"""CLI: build train/val patch manifests from merged multiclass label maps.

This is a data-touching command meant to be run against the real dataset --
after multi/merge_masks.py has been run. Local development only exercises
the underlying logic against synthetic fixtures (see tests/test_patch_index.py).

Usage:
    python multi/build_patch_index.py --config multi/configs/training_config.yaml [--data-root PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config, resolved_path
from src.mask_merge import FOREGROUND_CLASSES
from src.patch_index import build_patch_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to training_config.yaml")
    parser.add_argument("--data-root", default=None, help="Root dir paths are resolved against")
    args = parser.parse_args()

    config = load_config(args.config, args.data_root)
    data_cfg = config["data"]

    evaluation_cfg = config.get("evaluation", {})
    annotations_dirs = {
        cname: resolved_path(config, f"annotations_{cname}", section="evaluation")
        for cname in FOREGROUND_CLASSES
        if f"annotations_{cname}" in evaluation_cfg
    }

    n_train, n_val = build_patch_index(
        masks_dir=resolved_path(config, "multiclass_masks_dir"),
        out_dir=resolved_path(config, "patch_index_dir"),
        patch_size=data_cfg["patch_size"],
        train_stride=data_cfg["train_stride"],
        val_fraction=data_cfg["val_fraction"],
        num_classes=data_cfg["num_classes"],
        class_names=data_cfg["class_names"],
        split_seed=data_cfg["split_seed"],
        min_foreground_fraction=data_cfg["min_foreground_fraction"],
        background_keep_ratio=data_cfg["background_keep_ratio"],
        annotations_dirs=annotations_dirs,
    )

    print(f"Train patches: {n_train}")
    print(f"Val patches:   {n_val}")


if __name__ == "__main__":
    main()
