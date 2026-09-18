"""Model construction for multiclass training. Reuses
`models/unet.py:UNetGNRes` -- no second architecture -- parameterised by
`n_classes` (default 2, so every shipped binary checkpoint and the running
GUI still load it unmodified; see models/unet.py)."""
from __future__ import annotations

import warnings
from pathlib import Path

import torch

from .config import ensure_repo_root_importable

ensure_repo_root_importable()

from models.unet import (  # noqa: E402  (path set up above)
    HEAD_GROUPNORM,
    HEAD_PLAIN,
    UNetGNRes,
    head_from_state_dict,  # noqa: F401  (moved to models/unet.py; re-exported for existing callers)
)


def _strip_module_prefix(state_dict: dict) -> dict:
    """Shipped weights are plain state_dicts saved from a torch.nn.DataParallel
    wrapper, so every key is prefixed 'module.'."""
    return {
        (key[len("module."):] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def warm_start_from_checkpoint(model: UNetGNRes, checkpoint_path) -> list:
    """Load every matching layer EXCEPT conv_out (the classification head,
    whose shape is n_classes-dependent) from a binary RootPainter checkpoint
    -- e.g. so 4-class training starts from the cotyledon encoder instead of
    random init. Returns the list of loaded parameter keys.

    The `conv_out` skip is by key prefix, so it holds for either head: a
    plain-head model has no `conv_out.2.*` to receive the checkpoint's
    GroupNorm affine parameters, and those keys are skipped by name before
    the shape check ever runs."""
    checkpoint_path = Path(checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = _strip_module_prefix(state_dict)

    own_state = model.state_dict()
    loaded = []
    for key, value in state_dict.items():
        if key.startswith("conv_out"):
            continue
        if key not in own_state or own_state[key].shape != value.shape:
            continue
        own_state[key] = value
        loaded.append(key)

    model.load_state_dict(own_state)
    if not loaded:
        warnings.warn(f"warm_start_from={checkpoint_path}: no layers were loaded")
    return loaded


def build_model(cfg: dict) -> UNetGNRes:
    """cfg is the `model` section of the training config.

    `head` selects the classification head and defaults to "groupnorm", the
    only behaviour that existed before the key was added, so an older config
    without the key trains exactly the architecture it used to. "plain"
    drops the ReLU + per-class instance normalisation -- see
    models/unet.py:build_conv_out."""
    num_classes = cfg.get("num_classes", 2)
    im_channels = cfg.get("in_channels", 3)
    head = cfg.get("head") or HEAD_GROUPNORM
    model = UNetGNRes(im_channels=im_channels, n_classes=num_classes, head=head)

    warm_start_from = cfg.get("warm_start_from")
    if warm_start_from:
        warm_start_from_checkpoint(model, warm_start_from)

    return model
