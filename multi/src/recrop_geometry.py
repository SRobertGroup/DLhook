"""Crop-box geometry shared by the headless plate re-cropper (recrop_plates.py).

compute_crop_box and its padding constants live in utils/crop_layout.py (the one definition used
by the main window's box canvas and by the manual crop-drawing tool); they are re-exported here.
The rest (crop_from_box, file listing, output naming) mirrors seedling_measurement.py's
start_analysis crop loop without importing that module, which constructs Tk widgets and loads
CUDA models at import time -- exactly what a headless batch script must avoid.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Optional

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:      # some multi/ scripts import this before putting the repo root on sys.path
    sys.path.insert(0, _REPO_ROOT)

# The box geometry now lives in utils/crop_layout.py (the GUI imports it from there and must
# not import multi/); these names are re-exported so existing importers keep working.
from utils.crop_layout import (  # noqa: E402,F401
    DEFAULT_MIN_BOX_HALF_SIZE,
    DEFAULT_PADDING_HEIGHT_FRACTION,
    DEFAULT_PADDING_WIDTH_FRACTION,
    compute_crop_box,
)

# Mirrors seedling_measurement.py's VALID_IMAGE_EXTENSIONS. Extensions are
# matched case-insensitively (Camera_1 uses ".jpg", Highres_EMS_* uses
# ".JPG", simpler_AGJKV uses ".TIF").
VALID_IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def crop_from_box(img: np.ndarray, box: dict) -> tuple[np.ndarray, int, int, int, int]:
    """Apply a crop box to an image, clamped to image bounds -- the exact
    indexing of the GUI's start_analysis crop loop (seedling_measurement.py).
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
    seedling_measurement.py, so it is not tripped up by extensions that are
    not exactly 4 characters (".jpeg", ".TIFF")."""
    stem, _ext = os.path.splitext(original_filename)
    series_part = sanitize_series_name(series)
    return f"{crop_id}-crop-{series_part}_{stem}.png"


def list_image_files(path: str) -> list[str]:
    """Sorted image filenames directly in `path` (no subfolders), matching
    VALID_IMAGE_EXTENSIONS case-insensitively -- mirrors
    seedling_measurement.py's `_list_image_files`."""
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
