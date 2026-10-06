"""Landmark supervision from the hand-annotation clicks (ui/angle_annotator.py).

Each annotated crop has five clicks (utils/angle_annotation.py): the junction
where the cotyledons meet the hypocotyl, two points on the hypocotyl axis and
two on the cotyledon axis, plus an Overhook flag. They become three targets:

* a Gaussian HEATMAP at the junction -- the only anatomically fixed point;
* two unit-vector FIELDS (hypocotyl, cotyledon) pointing away from the junction,
  supervised only on the pixels within `ray_width` of the ray from the junction
  to the farthest click on that axis. The axis points are "anywhere along the
  axis", so nothing outside the ray is constrained -- a point-on-axis click
  must not become a target position;
* an image-level OVERHOOK label.

With a collar/root CSV (ui/root_annotator.py) a crop also gets the plant COLLAR
(root-hypocotyl transition) as a second Gaussian heatmap and the ROOT direction as
a unit-vector field along the ray collar -> root click, supervised like the other
axes. Crops without a visible radicle simply have no collar targets.

Reading them back (`readout_from_fields`) takes the junction as the heatmap
peak and each direction from the field at the junction, where both rays start
and the field is guaranteed to be supervised (readout_radius <= ray_width). The
angle is then `theta` = angle between the two directions, converted to the app's
bio convention exactly like the annotator does (180 - theta, or 180 + theta when
overhooked).

Everything here is pure numpy so it is unit-testable without torch.
"""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, replace

import numpy as np

from .config import ensure_repo_root_importable

ensure_repo_root_importable()

from utils.angle_annotation import orient_axis  # noqa: E402
from utils.landmark_readout import (  # noqa: E402,F401  (re-exported: pure readout lives in utils/)
    PEAK_FOUND, READOUT_RADIUS, bio_from_theta, readout_from_fields, theta_between,
)

HM_SIGMA = 3.0
RAY_WIDTH = 4.0
MIN_RAY_LENGTH = 8.0
PEAK_PRESENT = 0.5        # a rendered/predicted junction counts as "in the patch" above this
KP_CHANNELS = 5           # 1 junction logit + 2 x (x, y) direction fields
KP_CHANNELS_ROOT = 8      # + 1 collar logit + (x, y) root direction field


@dataclass(frozen=True)
class Landmark:
    filename: str          # crop file name, e.g. "3-crop-F1_Plate_2_YS_IMG_139.png"
    seedling: tuple        # (series, crop_id) -- crop_id repeats across series
    junction: tuple        # (x, y) in crop pixels, continuous (pixel i spans [i, i+1))
    hypo_dir: tuple        # unit vector, junction -> along the hypocotyl, away from it
    cotyl_dir: tuple       # unit vector, junction -> along the cotyledon, away from it
    hypo_len: float
    cotyl_len: float
    overhook: bool
    collar: tuple | None = None      # (x, y) crop pixels; None when no radicle is annotated
    root_dir: tuple | None = None    # unit vector collar -> along the root
    root_len: float = 0.0

    @property
    def theta(self) -> float:
        return theta_between(self.hypo_dir, self.cotyl_dir)


def _ray(junction, p1, p2):
    """(unit direction away from the junction, length to the farthest click)."""
    u = orient_axis(p1, p2, junction)
    length = max(float(np.dot(np.asarray(p, float) - np.asarray(junction, float), u)) for p in (p1, p2))
    return (float(u[0]), float(u[1])), max(length, MIN_RAY_LENGTH)


def _attach_roots(out, root_csv_path):
    """Add collar / root direction from a root_annotator CSV to the landmarks in `out`
    (frames with a visible radicle only). Returns how many crops got them."""
    n = 0
    with open(root_csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lm = out.get(row.get("crop_file"))
            if lm is None or row.get("status") != "annotated" or row.get("radicle_visible") != "1":
                continue
            try:
                collar = (float(row["collar_x"]), float(row["collar_y"]))
                root = (float(row["root_x"]), float(row["root_y"]))
            except (ValueError, KeyError):
                continue
            dx, dy = root[0] - collar[0], root[1] - collar[1]
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            out[lm.filename] = replace(lm, collar=collar, root_dir=(dx / length, dy / length),
                                       root_len=max(length, MIN_RAY_LENGTH))
            n += 1
    return n


def load_landmarks(csv_path, exclude_filenames=(), root_csv_path=None):
    """Landmarks from an annotation CSV (measured rows only), keyed by crop file.

    `exclude_filenames` is the leakage guard: pass the validation split's crop
    names (multi/processed/patch_index/val_patches.csv) so no held-out crop can
    become a training target. Returns (landmarks, report) where report counts
    loaded / excluded / invalid rows (an invalid row is one whose clicks cannot
    be oriented, e.g. both axis points on opposite sides of the junction).
    `root_csv_path` (a ui/root_annotator.py CSV) adds the collar and root direction to
    the crops it covers; report["with_root"] counts them."""
    exclude = set(exclude_filenames)
    out, report = {}, {"loaded": 0, "excluded": 0, "invalid": 0}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "measured":
                continue
            name = row["crop_file"]
            if name in exclude:
                report["excluded"] += 1
                continue
            try:
                pt = lambda p: (float(row[f"{p}_x"]), float(row[f"{p}_y"]))
                j = pt("junction")
                hypo_dir, hypo_len = _ray(j, pt("hypo1"), pt("hypo2"))
                cotyl_dir, cotyl_len = _ray(j, pt("cotyl1"), pt("cotyl2"))
            except (ValueError, KeyError):
                report["invalid"] += 1
                continue
            out[name] = Landmark(name, (row.get("series", ""), int(row["crop_id"])), j, hypo_dir,
                                 cotyl_dir, hypo_len, cotyl_len, row.get("overhook") == "1")
            report["loaded"] += 1
    if root_csv_path:
        report["with_root"] = _attach_roots(out, root_csv_path)
    return out, report


def split_by_seedling(landmarks, val_fraction=0.15, seed=0):
    """(train, val) dicts, split so that all frames of one seedling land on the
    same side -- frames of one seedling are near-duplicates."""
    keys = sorted({lm.seedling for lm in landmarks.values()})
    random.Random(seed).shuffle(keys)
    n_val = int(round(len(keys) * val_fraction)) if len(keys) >= 2 else 0
    if val_fraction > 0 and len(keys) >= 2:
        n_val = max(1, n_val)
    val_keys = set(keys[:n_val])
    train = {n: lm for n, lm in landmarks.items() if lm.seedling not in val_keys}
    val = {n: lm for n, lm in landmarks.items() if lm.seedling in val_keys}
    return train, val


def render_targets(h, w, lm: Landmark, sigma=HM_SIGMA, ray_width=RAY_WIDTH):
    """Targets on an h x w crop grid (before any padding or patch cut).

    Returns float32 arrays: hm (h, w), paf (4, h, w) = [hypo_x, hypo_y, cotyl_x,
    cotyl_y], paf_valid (2, h, w) = 1 where that axis' field is supervised."""
    yy, xx = np.mgrid[0:h, 0:w]
    px, py = xx + 0.5, yy + 0.5                       # pixel centres
    jx, jy = lm.junction
    hm = np.exp(-((px - jx) ** 2 + (py - jy) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)

    paf = np.zeros((4, h, w), np.float32)
    valid = np.zeros((2, h, w), np.float32)
    for k, (u, length) in enumerate(((lm.hypo_dir, lm.hypo_len), (lm.cotyl_dir, lm.cotyl_len))):
        t = np.clip((px - jx) * u[0] + (py - jy) * u[1], 0.0, length)
        dist = np.hypot(px - (jx + t * u[0]), py - (jy + t * u[1]))
        on_ray = dist <= ray_width
        valid[k][on_ray] = 1.0
        paf[2 * k][on_ray] = u[0]
        paf[2 * k + 1][on_ray] = u[1]

    collar_hm = np.zeros((h, w), np.float32)
    root_vec = np.zeros((2, h, w), np.float32)
    root_valid = np.zeros((1, h, w), np.float32)
    if lm.collar is not None and lm.root_dir is not None:
        cx, cy = lm.collar
        collar_hm = np.exp(-((px - cx) ** 2 + (py - cy) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
        u = lm.root_dir
        t = np.clip((px - cx) * u[0] + (py - cy) * u[1], 0.0, lm.root_len)
        on_ray = np.hypot(px - (cx + t * u[0]), py - (cy + t * u[1])) <= ray_width
        root_valid[0][on_ray] = 1.0
        root_vec[0][on_ray] = u[0]
        root_vec[1][on_ray] = u[1]
    return {"hm": hm, "paf": paf, "paf_valid": valid,
            "collar_hm": collar_hm, "root_vec": root_vec, "root_valid": root_valid}


def pad_targets_to_min(array, min_size):
    """Zero-pad a (H, W) or (C, H, W) target up to min_size x min_size with the
    same centred offsets as multi/src/patch_index.pad_to_min. Constant zeros,
    not reflection: reflecting a heatmap would duplicate the junction peak."""
    h, w = array.shape[-2:]
    h_pad, w_pad = max(0, min_size - h), max(0, min_size - w)
    if not (h_pad or w_pad):
        return array
    pad = [(h_pad // 2, h_pad - h_pad // 2), (w_pad // 2, w_pad - w_pad // 2)]
    if array.ndim == 3:
        pad = [(0, 0)] + pad
    return np.pad(array, pad, mode="constant")
