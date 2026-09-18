"""torch Dataset over the patch manifests written by patch_index.py."""
from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import ensure_repo_root_importable
from .normalization import zscore_normalize
from .patch_index import IGNORE_VALUE, pad_to_min

ensure_repo_root_importable()
# MARGIN is the fixed 6px-per-side context UNetGNRes needs on every input --
# its MaxPool floor-division shrinks the output by 2*MARGIN relative to the
# input regardless of size (see models/UNetInference.py's IN_SIZE/OUT_SIZE/
# MARGIN comment, and multiclass_inference.py, which reuses the same
# constant for whole-image tiling at inference time). Reused here rather than
# hardcoded so training and inference can never drift apart.
from models.UNetInference import MARGIN, _pad_reflect  # noqa: E402  (path set up above)


class PatchDataset(Dataset):
    def __init__(self, csv_path, raw_dir, masks_dir, augment: bool = False,
                 augmentation_cfg: dict | None = None, seed: int | None = None,
                 raw_ext: str = ".png", binarize_mask: bool = False):
        self.raw_dir = Path(raw_dir)
        self.masks_dir = Path(masks_dir)
        self.augment = augment
        self.raw_ext = raw_ext
        self.binarize_mask = binarize_mask
        self.margin = MARGIN
        self.rng = random.Random(seed)

        cfg = dict(augmentation_cfg or {})
        if cfg.get("vertical_flip"):
            warnings.warn(
                "augmentation.vertical_flip=True is ignored: seedlings are "
                "gravitropically oriented, a vertical flip is not label-preserving."
            )
        self.rotation_degrees = float(cfg.get("rotation_degrees", 0)) if augment else 0.0
        self.horizontal_flip = bool(cfg.get("horizontal_flip", False)) if augment else False
        self.brightness_jitter = float(cfg.get("brightness_jitter", 0)) if augment else 0.0
        self.contrast_jitter = float(cfg.get("contrast_jitter", 0)) if augment else 0.0

        with open(csv_path, "r", newline="", encoding="utf-8") as fh:
            self.rows = list(csv.DictReader(fh))

    def __len__(self):
        return len(self.rows)

    def _load_patch(self, row):
        filename = row["filename"]
        x, y, size = int(row["x"]), int(row["y"]), int(row["patch_size"])

        raw_path = self.raw_dir / (Path(filename).stem + self.raw_ext)
        mask_path = self.masks_dir / filename

        with Image.open(raw_path) as im:
            raw = np.array(im.convert("RGB")).astype(np.float32)
        with Image.open(mask_path) as im:
            mask = np.array(im.convert("L"))

        # `size` (patch_size) is the LABEL region's size, and x/y are origins
        # into the patch_size-padded frame patch_index.py used to build the
        # CSV (see build_patch_index / _tile_origins). raw and mask share the
        # same original H, W, so pad_to_min with the same `size` keeps them
        # in the same coordinate frame.
        raw = pad_to_min(raw, size)
        mask = pad_to_min(mask, size)

        # UNetGNRes shrinks its output by 2*MARGIN relative to its input (the
        # 12px MaxPool floor-division loss confirmed against get_valid_patch_
        # sizes), so the image fed to the model must carry MARGIN pixels of
        # extra context beyond the labelled region on every side, while the
        # label patch stays exactly `size`. Reflect-pad the (already
        # patch_size-padded) raw frame by MARGIN -- identical to what
        # UNetInference._plan_tiles does to its base image before tiling --
        # so this only adds pixels at the true image edges. Because that
        # padding is added uniformly, indexing the padded array at the SAME
        # (x, y) origin as the label patch yields exactly the MARGIN-pixel-
        # larger, correctly centered crop (mirrors how UNetInference.
        # _segment_many uses one (x, y) pair to index both the margin-padded
        # input tile and the unpadded output tile).
        raw_margin_padded = _pad_reflect(raw, self.margin)
        raw_patch = raw_margin_padded[
            y:y + size + 2 * self.margin, x:x + size + 2 * self.margin, :
        ].copy()
        mask_patch = mask[y:y + size, x:x + size].copy()
        return raw_patch, mask_patch

    def _augment(self, raw_patch: np.ndarray, mask_patch: np.ndarray):
        if self.horizontal_flip and self.rng.random() < 0.5:
            raw_patch = raw_patch[:, ::-1, :].copy()
            mask_patch = mask_patch[:, ::-1].copy()

        if self.rotation_degrees:
            angle = self.rng.uniform(-self.rotation_degrees, self.rotation_degrees)
            # raw_patch (size + 2*margin) and mask_patch (size) are no longer
            # the same shape, but the margin is symmetric on every side, so
            # both share the same physical center. Rotating each around its
            # OWN center by the same angle keeps them aligned despite the
            # different canvas sizes.
            mask_h, mask_w = mask_patch.shape
            mask_matrix = cv2.getRotationMatrix2D((mask_w / 2, mask_h / 2), angle, 1.0)
            # NEAREST + a fixed ignore-value border: rotation must never
            # interpolate a fractional/blended value into the label map.
            mask_patch = cv2.warpAffine(
                mask_patch, mask_matrix, (mask_w, mask_h), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=IGNORE_VALUE,
            )

            raw_h, raw_w = raw_patch.shape[:2]
            raw_matrix = cv2.getRotationMatrix2D((raw_w / 2, raw_h / 2), angle, 1.0)
            raw_patch = cv2.warpAffine(
                raw_patch, raw_matrix, (raw_w, raw_h), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )

        if self.brightness_jitter or self.contrast_jitter:
            contrast_factor = 1.0 + self.rng.uniform(-self.contrast_jitter, self.contrast_jitter)
            brightness_factor = 1.0 + self.rng.uniform(-self.brightness_jitter, self.brightness_jitter)
            mean = raw_patch.mean()
            raw_patch = (raw_patch - mean) * contrast_factor + mean
            raw_patch = raw_patch * brightness_factor
            raw_patch = np.clip(raw_patch, 0, 255)

        return raw_patch, mask_patch

    def __getitem__(self, idx):
        row = self.rows[idx]
        raw_patch, mask_patch = self._load_patch(row)

        if self.augment:
            raw_patch, mask_patch = self._augment(raw_patch, mask_patch)

        if self.binarize_mask:
            mask_patch = np.where(
                mask_patch == IGNORE_VALUE, IGNORE_VALUE, (mask_patch > 0).astype(np.uint8)
            )

        # Per-patch normalisation (this patch's own mean/std, pooled over
        # the whole HxWxC array), not a fixed dataset-wide constant --
        # illumination varies a lot across the capture rig's plates and
        # timepoints. Shared with multiclass_inference.py via
        # normalization.zscore_normalize so the two paths cannot drift
        # apart again (see that module's docstring for the bug this closed).
        raw_patch = zscore_normalize(raw_patch)

        image_chw = np.ascontiguousarray(raw_patch.transpose(2, 0, 1)).astype(np.float32)
        label_hw = np.ascontiguousarray(mask_patch).astype(np.int64)
        return torch.from_numpy(image_chw), torch.from_numpy(label_hw)
