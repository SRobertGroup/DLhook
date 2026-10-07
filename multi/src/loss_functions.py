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


DEFAULT_LANDMARK_WEIGHTS = {"hm": 1.0, "paf": 1.0, "collar": 1.0, "root": 1.0}


def landmark_loss(kp_logits: torch.Tensor, targets: dict,
                  weights: dict | None = None):
    """Loss for the optional landmark head (models/unet.py: forward_with_landmarks),
    already cropped to the label grid.

    kp_logits (N,5,H,W): channel 0 = junction logit, 1..4 = hypocotyl/cotyledon
    direction fields. `targets` holds hm (N,H,W), paf (N,4,H,W), paf_valid (N,2,H,W)
    and has_kp (N,).

    Every term is averaged over the samples with has_kp == 1 only -- a patch
    with no annotation, or one that does not contain the junction, contributes
    nothing here (its segmentation loss is separate). With no such sample the
    result is a zero that still carries a graph, so backward() is harmless.

    * junction: sigmoid(logit) against the Gaussian target, squared error
      weighted 1 + 20*hm so the peak is not drowned by background, normalised
      by the target's own mass (sum hm) rather than by pixel count, which would
      make the term vanishingly small on a 252x252 patch;
    * directions: L1 against the unit vector, only where paf_valid (the ray);
    * with an 8-channel head (kp_logits channels 5..7) and targets collar_hm /
      root_vec / root_valid / has_collar: the same heatmap loss on the collar and
      the same masked L1 on the root direction, each averaged over the samples that
      have a collar annotation (has_kp and has_collar).

    Returns (total, parts) with `parts` a dict of detached floats."""
    w = {**DEFAULT_LANDMARK_WEIGHTS, **(weights or {})}
    has = targets["has_kp"].to(kp_logits.dtype)
    n_has = has.sum()
    if float(n_has) == 0.0:
        return kp_logits.sum() * 0.0, {"hm": 0.0, "paf": 0.0}

    hm_t = targets["hm"].to(kp_logits.dtype)
    prob = torch.sigmoid(kp_logits[:, 0].float())
    sq = ((1.0 + 20.0 * hm_t.float()) * (prob - hm_t.float()) ** 2).sum(dim=(1, 2))
    hm_loss = ((sq / hm_t.float().sum(dim=(1, 2)).clamp(min=1e-6)) * has.float()).sum() / n_has

    valid = targets["paf_valid"].to(kp_logits.dtype).repeat_interleave(2, dim=1)
    l1 = ((kp_logits[:, 1:5].float() - targets["paf"].float()).abs() * valid.float()).sum(dim=(1, 2, 3))
    paf_loss = ((l1 / valid.float().sum(dim=(1, 2, 3)).clamp(min=1.0)) * has.float()).sum() / n_has

    total = w["hm"] * hm_loss + w["paf"] * paf_loss
    parts = {"hm": float(hm_loss.detach()), "paf": float(paf_loss.detach())}

    if kp_logits.shape[1] >= 8 and "collar_hm" in targets:
        has_c = (targets["has_collar"].to(kp_logits.dtype) * has).float()
        n_c = has_c.sum()
        if float(n_c) > 0.0:
            c_t = targets["collar_hm"].float()
            c_prob = torch.sigmoid(kp_logits[:, 5].float())
            c_sq = ((1.0 + 20.0 * c_t) * (c_prob - c_t) ** 2).sum(dim=(1, 2))
            collar_loss = ((c_sq / c_t.sum(dim=(1, 2)).clamp(min=1e-6)) * has_c).sum() / n_c
            r_valid = targets["root_valid"].float().repeat_interleave(2, dim=1)
            r_l1 = ((kp_logits[:, 6:8].float() - targets["root_vec"].float()).abs() * r_valid).sum(dim=(1, 2, 3))
            root_loss = ((r_l1 / r_valid.sum(dim=(1, 2, 3)).clamp(min=1.0)) * has_c).sum() / n_c
        else:
            collar_loss = root_loss = kp_logits[:, 5:8].sum() * 0.0
        total = total + w["collar"] * collar_loss + w["root"] * root_loss
        parts.update(collar=float(collar_loss.detach()), root=float(root_loss.detach()))
    return total, parts
