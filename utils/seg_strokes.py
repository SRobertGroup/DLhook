"""Partial segmentation strokes (ui/seg_annotator.py) -- pure numpy / cv2, no Tk.

A stroke map is a uint8 HxW array: UNLABELLED (255) where nothing was painted, BACKGROUND (0) for an
explicit background stroke, 1 / 2 / 3 for cotyledon / hypocotyl / radicle. Only the painted pixels
carry a label; everywhere else the training pseudo-label survives.

On disk the strokes use the RootPainter annotation format that multi/src/mask_merge.py already
burns over the pseudo-labels (`merge_dataset(annotations_dirs=...)`): one RGBA PNG per class, under
`<out>/<class>/<train|val>/<crop file>`, with R = 255 on that class's strokes, G = 255 on the
background strokes and A = 255 on either. A class file deliberately does NOT mark the other classes'
strokes as background: the merge applies the class files one after the other (background first,
then foreground, per class), so doing so would let a later class erase an earlier one.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

UNLABELLED = 255
BACKGROUND = 0
CLASS_VALUES = {"cotyledon": 1, "hypocotyl": 2, "radicle": 3}
SPLITS = ("train", "val")
MAX_UNDO = 50


class StrokeMap:
    """The strokes of one crop, with per-stroke undo."""

    def __init__(self, shape, labels=None):
        self.labels = (np.full(shape[:2], UNLABELLED, np.uint8) if labels is None
                       else np.asarray(labels, np.uint8).copy())
        self._undo = []

    def begin_stroke(self):
        """Snapshot for undo; call once at the start of each drag."""
        self._undo.append(self.labels.copy())
        del self._undo[:-MAX_UNDO]

    def paint(self, p0, p1, radius, value):
        """Paint a round-capped segment p0 -> p1 (image px, floats) with `value` (a class value,
        BACKGROUND, or UNLABELLED to erase)."""
        radius = max(1, int(round(radius)))
        a = (int(round(p0[0])), int(round(p0[1])))
        b = (int(round(p1[0])), int(round(p1[1])))
        cv2.line(self.labels, a, b, int(value), thickness=2 * radius - 1 if radius > 1 else 1)
        cv2.circle(self.labels, a, radius - 1 if radius > 1 else 0, int(value), -1)
        cv2.circle(self.labels, b, radius - 1 if radius > 1 else 0, int(value), -1)

    def undo(self):
        if self._undo:
            self.labels = self._undo.pop()
            return True
        return False

    @property
    def empty(self):
        return not bool((self.labels != UNLABELLED).any())

    def counts(self):
        """{name: painted pixels} for background and each class (only the non-zero ones)."""
        out = {}
        for name, value in (("background", BACKGROUND), *CLASS_VALUES.items()):
            n = int((self.labels == value).sum())
            if n:
                out[name] = n
        return out


def encode_class(labels, class_name):
    """RGBA uint8 RootPainter annotation of one class from a stroke map (see the module docstring)."""
    value = CLASS_VALUES[class_name]
    fg = labels == value
    bg = labels == BACKGROUND
    rgba = np.zeros((*labels.shape, 4), np.uint8)
    rgba[..., 0][fg] = 255
    rgba[..., 1][bg] = 255
    rgba[..., 3][fg | bg] = 255
    return rgba


def decode_files(rgba_by_class, shape):
    """Stroke map from {class name: RGBA array}: background where any file has G, a class where its
    file has R (foreground wins, as in mask_merge.apply_human_overrides)."""
    labels = np.full(shape[:2], UNLABELLED, np.uint8)
    for rgba in rgba_by_class.values():
        labels[(rgba[..., 1] == 255) & (rgba[..., 0] != 255)] = BACKGROUND
    for name, rgba in rgba_by_class.items():
        labels[rgba[..., 0] == 255] = CLASS_VALUES[name]
    return labels


def _path(out_dir, class_name, split, crop_file):
    return os.path.join(out_dir, class_name, split, crop_file)


def find_split(out_dir, crop_file):
    """'train' / 'val' when strokes exist for this crop, else None."""
    for split in SPLITS:
        if any(os.path.exists(_path(out_dir, c, split, crop_file)) for c in CLASS_VALUES):
            return split
    return None


def load_strokes(out_dir, crop_file, shape):
    """(StrokeMap, split or None) of a crop; an empty map when it has no strokes yet."""
    split = find_split(out_dir, crop_file)
    if split is None:
        return StrokeMap(shape), None
    from PIL import Image
    files = {}
    for name in CLASS_VALUES:
        path = _path(out_dir, name, split, crop_file)
        if os.path.exists(path):
            with Image.open(path) as im:
                files[name] = np.array(im.convert("RGBA"))
    return StrokeMap(shape, decode_files(files, shape)), split


def save_strokes(out_dir, crop_file, stroke_map, split="train"):
    """Write the three class files under `split` (removing any copy under the other split); an empty
    map deletes the crop's files. Returns the paths written."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    from PIL import Image
    for other in SPLITS:
        for name in CLASS_VALUES:
            path = _path(out_dir, name, other, crop_file)
            if os.path.exists(path) and (other != split or stroke_map.empty):
                os.remove(path)
    if stroke_map.empty:
        return []
    written = []
    for name in CLASS_VALUES:
        path = _path(out_dir, name, split, crop_file)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp.png"
        Image.fromarray(encode_class(stroke_map.labels, name), "RGBA").save(tmp)
        os.replace(tmp, path)
        written.append(path)
    return written


def annotations_dirs(out_dir):
    """{class: <out>/<class>} -- the `annotations_dirs` argument of mask_merge.merge_dataset."""
    return {name: os.path.join(out_dir, name) for name in CLASS_VALUES}
