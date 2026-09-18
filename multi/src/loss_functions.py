"""Loss functions for multiclass training. Both variants respect
`ignore_index=255` (the merge tool's "no consensus / unlabelled" value) so
those pixels never contribute gradient."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_IGNORE_INDEX = 255


def align_output_to_target(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Center-crop a model's spatial output down to `target`'s spatial size,
    if they differ.

    UNetGNRes's output is always a multiple of 16 (four MaxPool2d(2) stages
    -- see models/unet.py), so it is only ever exactly `input - 2*MARGIN`
    when `input` is itself in models/unet.py:get_valid_patch_sizes()'s family
    (input % 16 == 12, same residue as the historical IN_SIZE=572). PatchDataset
    feeds the model `patch_size + 2*MARGIN` (see multi/src/data_loader.py),
    which shifts that residue, so for every patch_size in that family the
    model's real output comes back exactly 4px larger (2px/side) than
    `patch_size` -- a constant, deterministic remainder of the architecture's
    granularity, not a sizing bug in the dataset. Crop that remainder off the
    model's own freshly computed output here, right before the loss (the
    original crash site, multi/train_unet_multiclass.py:65-66 ->
    multi/src/loss_functions.py:38) -- this never touches or discards any of
    the label's own hand/pseudo-labelled pixels.

    Raises if `output` is smaller than `target` in either spatial dimension:
    that would mean PatchDataset's margin was too small for this patch_size,
    which should never happen with MARGIN imported from
    models/UNetInference.py, but must fail loudly rather than silently
    misalign predictions against labels if it ever does.
    """
    oh, ow = output.shape[-2:]
    th, tw = target.shape[-2:]
    if (oh, ow) == (th, tw):
        return output
    if oh < th or ow < tw:
        raise RuntimeError(
            f"Model output {oh}x{ow} is smaller than the label {th}x{tw} -- "
            "PatchDataset's context margin is too small for this patch_size "
            "(see align_output_to_target's docstring)."
        )
    top, left = (oh - th) // 2, (ow - tw) // 2
    return output[..., top:top + th, left:left + tw]


class FocalLoss(nn.Module):
    """Multiclass focal loss (Lin et al. 2017) with per-class alpha and an
    ignore index. `alpha`, if given, must have one entry per class."""

    def __init__(self, gamma: float = 2.0, alpha=None, ignore_index: int = DEFAULT_IGNORE_INDEX):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.alpha is not None and self.alpha.numel() != logits.shape[1]:
            raise ValueError(
                f"focal_alpha has {self.alpha.numel()} entries but the model has "
                f"{logits.shape[1]} classes."
            )

        valid = target != self.ignore_index
        safe_target = target.clone()
        safe_target[~valid] = 0  # placeholder; masked out below, never touches the loss

        log_probs = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(log_probs, safe_target, reduction="none")  # (N, H, W)
        pt = log_probs.gather(1, safe_target.unsqueeze(1)).squeeze(1).exp()
        focal_term = (1.0 - pt).clamp(min=0.0) ** self.gamma
        loss = focal_term * ce

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[safe_target]
            loss = alpha_t * loss

        loss = loss * valid
        denom = valid.sum().clamp(min=1)
        return loss.sum() / denom


def build_loss(cfg: dict) -> nn.Module:
    """cfg is the `loss` section of the training config."""
    loss_type = cfg.get("type", "cross_entropy")
    ignore_index = cfg.get("ignore_index", DEFAULT_IGNORE_INDEX)
    class_weights = cfg.get("class_weights")
    weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32) if class_weights else None

    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=ignore_index)
    if loss_type == "focal":
        return FocalLoss(
            gamma=cfg.get("focal_gamma", 2.0),
            alpha=cfg.get("focal_alpha"),
            ignore_index=ignore_index,
        )
    raise ValueError(f"Unknown loss.type: {loss_type!r} (expected 'cross_entropy' or 'focal')")
