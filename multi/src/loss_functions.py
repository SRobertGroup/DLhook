"""Loss functions for multiclass training. Both variants respect
`ignore_index=255` (the merge tool's "no consensus / unlabelled" value) so
those pixels never contribute gradient."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ensure_repo_root_importable

ensure_repo_root_importable()

from models.unet import align_output_to_target  # noqa: E402,F401  (moved to models/unet.py; re-exported for existing callers)

DEFAULT_IGNORE_INDEX = 255


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
