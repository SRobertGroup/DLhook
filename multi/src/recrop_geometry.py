"""Crop-box geometry shared by the headless plate re-cropper (recrop_plates.py).

This intentionally duplicates (rather than imports) the small amount of
geometry from seedling_measurment.py's `_compute_crop_box` /
`start_analysis` crop loop: that module constructs Tk widgets and loads
CUDA/superres models at import time, which is exactly what a headless,
GPU-optional batch script must avoid pulling in. Keeping the two in sync is a
matter of re-reading seedling_measurment.py when either changes -- see
CLAUDE.md's crop-box padding constants (`crop_padding_width_fraction`,
`crop_padding_height_fraction`, `min_box_half_size`), which this mirrors
exactly.
"""
from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np

# Mirrors seedling_measurment.py's Gui.__init__ defaults exactly.
DEFAULT_PADDING_WIDTH_FRACTION = 0.4
DEFAULT_PADDING_HEIGHT_FRACTION = 0.10
DEFAULT_MIN_BOX_HALF_SIZE = 30

# Mirrors seedling_measurment.py's VALID_IMAGE_EXTENSIONS. Extensions are
# matched case-insensitively (Camera_1 uses ".jpg", Highres_EMS_* uses
# ".JPG", simpler_AGJKV uses ".TIF").
VALID_IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def compute_crop_box(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    padding_width_fraction: float = DEFAULT_PADDING_WIDTH_FRACTION,
    padding_height_fraction: float = DEFAULT_PADDING_HEIGHT_FRACTION,
    min_box_half_size: float = DEFAULT_MIN_BOX_HALF_SIZE,
) -> dict:
    """Given a seedling's bounding extent (image coords), return the padded
    crop box {"cx", "cy", "half_w", "half_h"} -- the exact geometry
    `Gui._compute_crop_box` derives from a start/end point pair, generalised
    to any bounding box (here, a detected blob's extent instead of two
    clicked points).

    The padding is asymmetric on purpose: seedlings are imaged growing
    roughly vertically, so more slack is given sideways (for the hook
    swinging left/right) than vertically.
    """
    half_w = (max_x - min_x) * (1 + 2 * padding_width_fraction) / 2
    half_h = (max_y - min_y) * (1 + 2 * padding_height_fraction) / 2

    # Degenerate-case floor only, matching Gui._compute_crop_box: boxes
    # otherwise hug the detected extent plus padding, they are not floored
    # up to any fixed default size.
    half_w = max(half_w, min_box_half_size)
    half_h = max(half_h, min_box_half_size)

    return {
        "cx": round((min_x + max_x) / 2),
        "cy": round((min_y + max_y) / 2),
        "half_w": round(half_w),
        "half_h": round(half_h),
    }


def crop_from_box(img: np.ndarray, box: dict) -> tuple[np.ndarray, int, int, int, int]:
    """Apply a crop box to an image, clamped to image bounds -- the exact
    indexing of the GUI's start_analysis crop loop (seedling_measurment.py).
    Returns (crop, x1, y1, x2, y2)."""
    center_x, center_y = box["cx"], box["cy"]
    x1 = max(0, center_x - box["half_w"])
    y1 = max(0, center_y - box["half_h"])
    x2 = min(img.shape[1], center_x + box["half_w"])
    y2 = min(img.shape[0], center_y + box["half_h"])
    return img[y1:y2, x1:x2], x1, y1, x2, y2


def sanitize_series_name(name: str) -> str:
    """Filesystem/collision-safe stand-in for a series name: series names in
    example_data/ contain spaces, " - Copy" suffixes, and mixed case
    ("AGJKV system pictures", "F1_Plate_2_YS"). Collapse anything that is not
    alphanumeric/underscore/hyphen/dot into a single underscore."""
    cleaned = _SANITIZE_RE.sub("_", name.strip())
    return cleaned.strip("_") or "series"


def build_output_filename(crop_id: int, series: str, original_filename: str) -> str:
    """`{crop_id}-crop-{series}_{stem}.png` -- keeps the exact
    `{crop_id}-crop-{original_name}.png` convention used across the codebase
    (parsed elsewhere with `split("-", 1)`, so only the pure-integer prefix
    before the first "-" is load-bearing) while folding the series name into
    the "original_name" portion so identically-named frames from different
    series (e.g. every series' "IMG_086.png") never collide.

    Uses os.path.splitext rather than the legacy `image[:-4]` slice used in
    seedling_measurment.py, so it is not tripped up by extensions that are
    not exactly 4 characters (".jpeg", ".TIFF")."""
    stem, _ext = os.path.splitext(original_filename)
    series_part = sanitize_series_name(series)
    return f"{crop_id}-crop-{series_part}_{stem}.png"


def list_image_files(path: str) -> list[str]:
    """Sorted image filenames directly in `path` (no subfolders), matching
    VALID_IMAGE_EXTENSIONS case-insensitively -- mirrors
    seedling_measurment.py's `_list_image_files`."""
    return sorted(
        f for f in os.listdir(path)
        if f.lower().endswith(VALID_IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(path, f))
    )


def reference_frame_index(n_frames: int, fraction: float) -> int:
    """Pick a single representative frame index out of a sorted file list at
    roughly `fraction` through the series (default: late, since seedlings
    are small/faint near germination and are their most detectable -- large,
    clearly dark against the plate -- later on)."""
    if n_frames <= 0:
        raise ValueError("empty series has no reference frame")
    idx = int(round((n_frames - 1) * fraction))
    return min(max(idx, 0), n_frames - 1)
