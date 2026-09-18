"""Build train/val patch manifests (`train_patches.csv` / `val_patches.csv`)
from the merged multiclass masks produced by `mask_merge.merge_dataset`.

Split is image-level only: every patch cut from one image goes to the same
split as that image, so a model never sees train and val pixels from the
same source photo. Annotated images (the 441 with RootPainter strokes)
keep whatever train/val membership RootPainter already assigned them;
everything else is split by a seeded shuffle.
"""
from __future__ import annotations

import csv
import random
import warnings
from math import ceil
from pathlib import Path

import numpy as np
from PIL import Image

BACKGROUND_CLASS_INDEX = 0
IGNORE_VALUE = 255


def discover_annotation_split(annotations_dirs: dict) -> dict:
    """filename -> "train"/"val", merged across all classes in
    `annotations_dirs` (each a directory with train/ and val/ subfolders).
    Classes are visited in the order given; a filename already assigned by
    an earlier class keeps that split and a conflicting later assignment is
    dropped with a warning (mirrors the train-precedence rule used for the
    same-class train/val duplicate in mask_merge.discover_annotation_files).
    """
    assignment: dict[str, str] = {}
    for cname, adir in annotations_dirs.items():
        adir = Path(adir)
        for split in ("train", "val"):
            split_dir = adir / split
            if not split_dir.is_dir():
                continue
            for path in sorted(split_dir.iterdir()):
                if not path.is_file():
                    continue
                existing = assignment.get(path.name)
                if existing is None:
                    assignment[path.name] = split
                elif existing != split:
                    warnings.warn(
                        f"{path.name}: {cname} annotations say split={split!r} but it was "
                        f"already assigned split={existing!r} by another class; keeping "
                        f"{existing!r}."
                    )
    return assignment


def pad_to_min(array: np.ndarray, min_size: int):
    """Reflect-pad a 2D (mask) or 3D (image) array up to at least
    min_size x min_size, same geometry as UNetInference._pad_to_min."""
    h, w = array.shape[:2]
    h_pad, w_pad = max(0, min_size - h), max(0, min_size - w)
    if not (h_pad or w_pad):
        return array
    h_before, h_after = h_pad // 2, h_pad - h_pad // 2
    w_before, w_after = w_pad // 2, w_pad - w_pad // 2
    pad_widths = [(h_before, h_after), (w_before, w_after)]
    if array.ndim == 3:
        pad_widths.append((0, 0))
    return np.pad(array, pad_widths, mode="reflect")


def _tile_origins(height: int, width: int, patch_size: int, stride: int):
    """Patch top-left origins covering (height, width) with the given
    stride; the last row/column is shifted inward (not resized) so every
    patch stays exactly patch_size, same tiling strategy as inference."""
    n_x = max(1, ceil(max(1, width - patch_size + 1) / stride))
    n_y = max(1, ceil(max(1, height - patch_size + 1) / stride))
    xs = [min(i * stride, max(0, width - patch_size)) for i in range(n_x)]
    ys = [min(i * stride, max(0, height - patch_size)) for i in range(n_y)]
    xs = sorted(set(xs))
    ys = sorted(set(ys))
    return [(x, y) for x in xs for y in ys]


def build_patch_index(
    masks_dir,
    out_dir,
    patch_size: int,
    train_stride: int,
    val_fraction: float,
    num_classes: int,
    class_names: list,
    split_seed: int,
    min_foreground_fraction: float,
    background_keep_ratio: float,
    annotations_dirs: dict | None = None,
    image_glob: str = "*.png",
):
    """Returns (n_train, n_val) patch counts and writes
    `<out_dir>/train_patches.csv` and `<out_dir>/val_patches.csv`, each with
    columns: filename, x, y, patch_size, foreground_fraction.
    """
    if len(class_names) != num_classes:
        raise ValueError(
            f"class_names has {len(class_names)} entries but num_classes={num_classes}"
        )

    masks_dir = Path(masks_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted(masks_dir.glob(image_glob))
    if not mask_paths:
        warnings.warn(f"No masks found under {masks_dir} matching {image_glob!r}")

    annotated_split = discover_annotation_split(annotations_dirs) if annotations_dirs else {}

    unannotated = [p.name for p in mask_paths if p.name not in annotated_split]
    rng = random.Random(split_seed)
    shuffled = sorted(unannotated)
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * val_fraction)
    val_set = set(shuffled[:n_val])

    def split_for(name: str) -> str:
        if name in annotated_split:
            return annotated_split[name]
        return "val" if name in val_set else "train"

    keep_rng = random.Random(split_seed)
    rows = {"train": [], "val": []}

    for path in mask_paths:
        split = split_for(path.name)
        stride = train_stride if split == "train" else patch_size

        with Image.open(path) as im:
            mask = np.array(im.convert("L"))
        mask = pad_to_min(mask, patch_size)
        h, w = mask.shape

        for (x, y) in _tile_origins(h, w, patch_size, stride):
            patch = mask[y:y + patch_size, x:x + patch_size]
            foreground_px = np.count_nonzero(
                (patch != BACKGROUND_CLASS_INDEX) & (patch != IGNORE_VALUE)
            )
            foreground_fraction = foreground_px / patch.size

            if foreground_fraction < min_foreground_fraction:
                if background_keep_ratio < 1.0 and keep_rng.random() >= background_keep_ratio:
                    continue

            rows[split].append({
                "filename": path.name,
                "x": x,
                "y": y,
                "patch_size": patch_size,
                "foreground_fraction": round(foreground_fraction, 6),
            })

    fieldnames = ["filename", "x", "y", "patch_size", "foreground_fraction"]
    for split in ("train", "val"):
        csv_path = out_dir / f"{split}_patches.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows[split])

    return len(rows["train"]), len(rows["val"])
