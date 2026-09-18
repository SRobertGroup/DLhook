"""Shared input normalisation for the multiclass pipeline.

`multi/src/data_loader.py` (training) and `multi/src/multiclass_inference.py`
(inference) both feed raw pixel arrays into the same UNetGNRes checkpoint, so
they must draw those arrays from the same distribution -- a model fit to one
preprocessing is not meaningfully evaluated on a different one. This used to
be two separate implementations that quietly drifted apart: training did a
per-patch z-score while inference did a flat `/255.0`, so on a real crop
training fed mean 0.000/std 1.000 (range roughly [-8, +2]) while inference
fed mean 0.598/std 0.061 (range roughly [0.1, 0.7]) -- a ~16x difference in
standard deviation, i.e. the model had effectively never seen inputs shaped
like the ones it was being scored on. Factoring the single normalisation
function out here means both call sites share one implementation and cannot
silently diverge again.
"""
from __future__ import annotations

import numpy as np

DEFAULT_EPS = 1e-6


def zscore_normalize(array: np.ndarray, eps: float = DEFAULT_EPS) -> np.ndarray:
    """(x - x.mean()) / (x.std() + eps), with mean/std pooled as ONE scalar
    over every pixel and every channel of `array` together -- not computed
    per-channel. This matches exactly what
    `PatchDataset.__getitem__` (data_loader.py) does to its training patches
    (`raw_patch.mean()`, `raw_patch.std()` called on the full HxWxC array),
    so callers that want training-equivalent statistics must pass this
    function the same *shape and scope* of array training would have z-
    scored -- see multiclass_inference.py's `_run_batch` for how the
    inference side decides what that scope is when tiling a full image.
    """
    array = np.asarray(array, dtype=np.float32)
    mean = array.mean()
    std = array.std()
    return (array - mean) / (std + eps)
