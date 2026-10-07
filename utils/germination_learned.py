"""Learned germination onset -- the pure part (numpy only, no model loading).

A seedling has germinated from the first frame in which a radicle is visible. The
detector fitted by multi/germination_detector_fit.py on hand-marked onsets
(ui/germination_annotator.py) works in two steps:

1. per frame, a logistic regression on a few statistics of the four-class model's
   softmax maps (how much radicle, how much cotyledon / hypocotyl) gives the
   probability that a radicle is visible (`frame_features`, `GerminationModel`);
2. per seedling, ONE onset explains the whole series: the step function (no radicle
   before frame k, radicle from k on) with the lowest log-loss against those
   probabilities (`best_onset`). A radicle does not disappear again, so a single
   noisy frame cannot move the onset.

Measured on 75 seedlings it never saw: every onset found, 71% on the exact frame and
91% within one frame, where the area-near-the-seed rule found 4-6 of 75 (with a proxy
seed point). The model run that produces the softmax maps lives in
models/segmentation_backends.py (LearnedGerminationRunner); weights are a small JSON
written by the fitting script.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

FEATURE_NAMES = ["r_max", "r_top20", "r_log_mass", "r_log_n50", "r_log_n20", "c_log_mass", "h_log_mass"]


def frame_features(probs):
    """Features of one crop's softmax maps (4, H, W): class 3 = radicle, 1 = cotyledon, 2 = hypocotyl."""
    radicle = probs[3].ravel()
    top = np.sort(radicle)[-20:] if radicle.size else np.zeros(1)
    return [float(radicle.max()) if radicle.size else 0.0, float(top.mean()),
            float(np.log1p(radicle.sum())), float(np.log1p((radicle > 0.5).sum())),
            float(np.log1p((radicle > 0.2).sum())),
            float(np.log1p(probs[1].sum())), float(np.log1p(probs[2].sum()))]


@dataclass
class GerminationModel:
    """Standardised logistic regression over FEATURE_NAMES."""
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    bias: float
    checkpoint: str = ""

    @classmethod
    def load(cls, path):
        data = json.loads(open(path, encoding="utf-8").read())
        if data.get("features") != FEATURE_NAMES:
            raise ValueError(f"{path} was fitted on features {data.get('features')}, expected {FEATURE_NAMES}")
        return cls(np.asarray(data["mean"], float), np.asarray(data["std"], float),
                   np.asarray(data["weights"], float), float(data["bias"]), data.get("checkpoint", ""))

    def visible_probs(self, features):
        """Per-frame probability that a radicle is visible, from an (n_frames, n_features) array."""
        x = (np.asarray(features, float) - self.mean) / self.std
        return 1.0 / (1.0 + np.exp(-(x @ self.weights + self.bias)))


def best_onset(p):
    """Index k in 0..n minimising the step-function log-loss: frames < k not visible, >= k visible.
    k == n means no radicle anywhere in the series."""
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    before = np.concatenate([[0.0], np.cumsum(-np.log(1 - p))])
    after = np.concatenate([np.cumsum((-np.log(p))[::-1])[::-1], [0.0]])
    return int(np.argmin(before + after))


def onset_or_none(p):
    """best_onset, with 'no radicle in the series' (k == n) and an empty series as None."""
    if len(p) == 0:
        return None
    k = best_onset(p)
    return k if k < len(p) else None
