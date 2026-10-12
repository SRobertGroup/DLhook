#!/usr/bin/env python
"""Train the multiclass U-Net on patches indexed by multi/build_patch_index.py.

This trains against real data, which lives on the remote host. Local development
only exercises run_training() against tiny synthetic fixtures (see tests/test_training.py).

Usage:
    python multi/train_unet_multiclass.py --config multi/configs/training_config.yaml [--data-root PATH]
    python multi/train_unet_multiclass.py --config multi/configs/training_config.yaml --resume
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config, resolved_path
from src.data_loader import PatchDataset
from src.landmarks import (angle_balanced_weights, angle_bin, landmark_patch_rows, load_landmarks,
                           load_measurable_negatives, namespace_landmarks, readout_from_fields,
                           split_by_seedling, theta_between)
from src.loss_functions import align_output_to_target, build_loss, landmark_loss, measurable_loss
from src.model import build_model
from models.unet import strip_deprecated_landmark_keys


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def compute_lr(epoch: int, training_cfg: dict) -> float:
    """Linear warmup over `lr_warmup_epochs`, then cosine decay from lr_max to lr_min
    over the remaining epochs. A pure function of the epoch index (not scheduler
    state), so it resumes correctly without needing to be checkpointed."""
    lr_max = training_cfg["lr_max"]
    lr_min = training_cfg.get("lr_min", lr_max)
    warmup_epochs = training_cfg.get("lr_warmup_epochs", 0)

    if warmup_epochs > 0 and epoch < warmup_epochs:
        return lr_max * (epoch + 1) / warmup_epochs

    decay_span = max(1, training_cfg["num_epochs"] - warmup_epochs - 1)
    progress = min(1.0, (epoch - warmup_epochs) / decay_span)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def update_confusion_matrix(confusion: np.ndarray, target: torch.Tensor, pred: torch.Tensor,
                             num_classes: int, ignore_index: int = 255) -> None:
    """Accumulate one batch's (target, pred) pair into `confusion` in place.

    `confusion[i, j]` becomes the count of pixels whose true class is `i` and
    predicted class is `j`, so `np.diag(confusion)` is the per-class true
    positives. Pixels where `target == ignore_index` are dropped before
    counting -- the same "no consensus / unlabelled" convention the loss
    functions respect (src/loss_functions.py's DEFAULT_IGNORE_INDEX), so a
    class's metrics are never inflated or deflated by unlabelled pixels.
    """
    target_flat = target.detach().to("cpu").reshape(-1).numpy()
    pred_flat = pred.detach().to("cpu").reshape(-1).numpy()

    valid = target_flat != ignore_index
    target_flat = target_flat[valid].astype(np.int64)
    pred_flat = pred_flat[valid].astype(np.int64)

    counts = np.bincount(target_flat * num_classes + pred_flat, minlength=num_classes * num_classes)
    confusion += counts.reshape(num_classes, num_classes)


def compute_class_metrics(confusion: np.ndarray | None) -> dict:
    """Derive per-class IoU/Dice from a confusion matrix built by
    `update_confusion_matrix` (rows = true class, cols = predicted class).

    A class with union (TP+FP+FN) == 0 -- absent from both the predictions
    and the ground truth over the whole pass -- has an undefined 0/0
    IoU/Dice. That entry is reported as NaN rather than 0, and excluded
    (not zeroed) when averaging `mean_fg_dice`, so a foreground class that
    simply never appeared in a given validation split doesn't drag the mean
    down as if the model had missed it.

    `mean_fg_dice` averages Dice over every class except class 0
    (background): background is 95.9% of pixels in this dataset, and
    including it would reintroduce the exact loss-driven blindness to the
    rare foreground classes this metric exists to catch.
    """
    if confusion is None:
        return {"iou": [], "dice": [], "mean_fg_dice": 0.0}

    num_classes = confusion.shape[0]
    confusion = confusion.astype(np.float64)
    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp

    union = tp + fp + fn
    iou = np.full(num_classes, np.nan)
    np.divide(tp, union, out=iou, where=union > 0)

    dice_denom = 2 * tp + fp + fn
    dice = np.full(num_classes, np.nan)
    np.divide(2 * tp, dice_denom, out=dice, where=dice_denom > 0)

    fg_dice = dice[1:]
    fg_valid = ~np.isnan(fg_dice)
    mean_fg_dice = float(np.mean(fg_dice[fg_valid])) if fg_valid.any() else 0.0

    return {"iou": iou.tolist(), "dice": dice.tolist(), "mean_fg_dice": mean_fg_dice}


def run_epoch(model, loader, loss_fn, optimizer, device, desc: str,
              use_amp: bool = False, scaler: torch.amp.GradScaler | None = None,
              grad_clip_norm: float | None = None, num_classes: int | None = None,
              ignore_index: int = 255, landmark_weights: dict | None = None,
              measurable: dict | None = None) -> tuple[float, np.ndarray | None]:
    """Run one training epoch (`optimizer` given) or one validation pass
    (`optimizer=None`).

    When `num_classes` is given, also accumulates a `num_classes x
    num_classes` confusion matrix over the pass (see
    `update_confusion_matrix`) and returns it as the second element,
    otherwise the second element is None. Only the validation call in
    `run_training` passes `num_classes` -- the extra argmax + bincount per
    batch is cheap, but there's no need to pay it on every training batch
    too when only the validation metrics drive checkpoint selection.

    A loader built with landmarks yields (images, masks, targets); with
    `landmark_weights` given, the landmark head's loss (src/loss_functions.py:
    landmark_loss) is added to the segmentation loss. Without it the extra
    element is ignored and the model's plain forward() is used. With `measurable`
    ({"weight", "pos_weight"}) the image-level "measurable" head's loss (src/loss_functions.py:
    measurable_loss) is added too.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n_batches = 0.0, 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64) if num_classes else None
    progress = tqdm(loader, desc=desc, unit="batch", leave=False)
    with torch.set_grad_enabled(is_train):
        for batch in progress:
            images, masks, *extra = batch
            images, masks = images.to(device), masks.to(device)
            targets = {k: v.to(device) for k, v in extra[0].items()} if extra else None
            use_landmarks = targets is not None and landmark_weights is not None
            with torch.autocast(device_type=device.type, enabled=use_amp):
                measurable_logit = None
                if use_landmarks and measurable:
                    outputs, kp_out, measurable_logit = model.forward_with_landmarks(images, return_measurable=True)
                elif use_landmarks:
                    outputs, kp_out = model.forward_with_landmarks(images)
                else:
                    outputs = model(images)
                # See align_output_to_target's docstring: PatchDataset feeds
                # patch_size + 2*MARGIN, whose output the fixed architecture
                # rounds to a multiple of 16 -- a few pixels off patch_size
                # for every get_valid_patch_sizes() value. Crop that
                # architecture-forced remainder off the model's own output,
                # never off the label, right before the loss.
                outputs = align_output_to_target(outputs, masks)
                loss = loss_fn(outputs, masks)
                if use_landmarks:
                    kp_loss, _ = landmark_loss(align_output_to_target(kp_out, masks),
                                                targets, landmark_weights)
                    loss = loss + kp_loss
                if measurable_logit is not None:
                    m_loss, _ = measurable_loss(measurable_logit, targets, float(measurable.get("weight", 1.0)),
                                                float(measurable.get("pos_weight", 1.0)))
                    loss = loss + m_loss
            if is_train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()
            if confusion is not None:
                preds = outputs.detach().argmax(dim=1)
                update_confusion_matrix(confusion, masks, preds, num_classes, ignore_index)
            total_loss += loss.item()
            n_batches += 1
            progress.set_postfix(loss=f"{total_loss / n_batches:.4f}")
    return total_loss / max(n_batches, 1), confusion


def run_landmark_validation(model, loader, device, readout_radius: float = 3.0,
                            missing_penalty_deg: float = 90.0, missing_penalty_px: float = 30.0) -> dict:
    """Landmark metrics over a loader built with landmarks (augmentation off).

    For every patch that contains the annotated junction, the angle is read back
    from the predicted fields exactly as at inference (readout_from_fields) and
    compared with the same readout applied to the TARGET fields, so the number
    is the angle error the landmark head would give, not a proxy. A patch whose
    junction the model fails to find counts as `missing_penalty_deg` -- otherwise
    a head that predicts nothing would score a perfect (empty) mean.

    Returns landmark_angle_mae (deg), junction_px_err,
    landmark_found (fraction) and n; the means are NaN when the loader holds no
    patch with a junction."""
    model.eval()
    angle_err, junction_err, found, n = [], [], 0, 0
    collar_err, root_err = [], []
    with torch.no_grad():
        for images, masks, targets in loader:
            _, kp_out = model.forward_with_landmarks(images.to(device))
            kp_out = align_output_to_target(kp_out, masks).float().cpu()
            hm_p = torch.sigmoid(kp_out[:, 0]).numpy()
            vec_p = kp_out[:, 1:5].numpy()
            has_root = kp_out.shape[1] >= 8 and "collar_hm" in targets
            if has_root:
                collar_p = torch.sigmoid(kp_out[:, 5]).numpy()
                root_p = kp_out[:, 6:8].numpy()
            for i in range(images.shape[0]):
                if has_root and float(targets["has_collar"][i]) > 0.5 and float(targets["has_kp"][i]) > 0.5:
                    t_c = readout_from_fields(targets["hm"][i].numpy(), targets["paf"][i].numpy(),
                                              radius=readout_radius, collar_hm=targets["collar_hm"][i].numpy(),
                                              root_vec=targets["root_vec"][i].numpy())
                    if t_c is not None and t_c["collar"] is not None and t_c["root_dir"] is not None:
                        p_c = readout_from_fields(hm_p[i], vec_p[i], radius=readout_radius,
                                                  peak_threshold=0.0, collar_hm=collar_p[i], root_vec=root_p[i])
                        if p_c is None or p_c["collar"] is None or p_c["root_dir"] is None:
                            collar_err.append(missing_penalty_px)
                            root_err.append(missing_penalty_deg)
                        else:
                            collar_err.append(float(np.hypot(p_c["collar"][0] - t_c["collar"][0],
                                                              p_c["collar"][1] - t_c["collar"][1])))
                            root_err.append(theta_between(p_c["root_dir"], t_c["root_dir"]))
                if float(targets["has_kp"][i]) < 0.5:
                    continue
                truth = readout_from_fields(targets["hm"][i].numpy(), targets["paf"][i].numpy(),
                                            radius=readout_radius)
                if truth is None:
                    continue
                n += 1
                pred = readout_from_fields(hm_p[i], vec_p[i], radius=readout_radius)
                if pred is None:
                    angle_err.append(missing_penalty_deg)
                    continue
                found += 1
                angle_err.append(abs(pred["theta"] - truth["theta"]))
                junction_err.append(float(np.hypot(pred["junction"][0] - truth["junction"][0],
                                                    pred["junction"][1] - truth["junction"][1])))
    nan = float("nan")
    return {
        "landmark_angle_mae": float(np.mean(angle_err)) if angle_err else nan,
        "junction_px_err": float(np.mean(junction_err)) if junction_err else nan,
        "landmark_found": found / n if n else nan,
        "collar_px_err": float(np.mean(collar_err)) if collar_err else nan,
        "root_angle_err": float(np.mean(root_err)) if root_err else nan,
        "n": n,
    }


def run_wholecrop_landmark_validation(model, landmarks: dict, raw_dir, device, raw_ext: str = ".png",
                                      patch_size: int = 252, readout_radius: float = 3.0,
                                      missing_penalty_deg: float = 90.0, missing_penalty_px: float = 30.0,
                                      raw_paths: dict | None = None, per_crop: dict | None = None) -> dict:
    """Landmark metrics on WHOLE crops of the held-out annotated seedlings, read through
    MulticlassInference's own tiling and readout -- the conditions the model is used in.
    The patch metrics of run_landmark_validation score 252 px tiles, where padding,
    tiles that cut the hook and the miss penalty make them noisy enough to pick a poor
    checkpoint; this is the metric to select on ('wholecrop_theta_mae').

    Errors are against the annotated landmark itself (not a re-rendered target). A crop
    with no junction found counts as missing_penalty_deg; a missing collar as
    missing_penalty_px / missing_penalty_deg. `raw_paths` ({name: image path}) overrides
    raw_dir for crops from other folders (landmarks.extra_sources). With `per_crop` (a dict) it
    is filled with {name: {"err": theta error, "theta_gt", "theta_pred", "path"}} for every crop."""
    import cv2
    from models.UNetInference import MARGIN
    from models.multiclass_inference import LandmarkEnsemble, MulticlassInference

    models = list(model) if isinstance(model, (list, tuple)) else [model]    # several = averaged before the readout
    for m in models:
        m.eval()
    images, kept = [], []
    for name in sorted(landmarks):
        path = (raw_paths or {}).get(name) or Path(raw_dir) / (Path(name).stem + raw_ext)
        image = cv2.imread(str(path))
        if image is not None:
            images.append(image)
            kept.append(name)
    members = [MulticlassInference.from_model(m, device=device, in_size=patch_size + 2 * MARGIN,
                                              out_size=patch_size, margin=MARGIN) for m in models]
    inference = members[0] if len(members) == 1 else LandmarkEnsemble(members)
    readouts = []
    for start in range(0, len(images), 16):
        readouts += inference.predict_landmarks(images[start:start + 16], readout_radius=readout_radius)

    theta_err, junction_err, collar_err, root_err = [], [], [], []
    tip_err, base_err, tip_paths = [], [], []
    for name, r in zip(kept, readouts):
        lm = landmarks[name]
        if per_crop is not None:
            per_crop[name] = {"err": missing_penalty_deg if r is None else abs(r["theta"] - lm.theta),
                              "theta_gt": lm.theta, "theta_pred": None if r is None else r["theta"],
                              "path": "" if r is None else r.get("cotyl_path", ""),
                              "overhook_gt": lm.overhook, "overhook_pred": None if r is None else r.get("overhook"),
                              "hypo_deg": None if r is None else theta_between(r["hypo_dir"], lm.hypo_dir),
                              "cotyl_deg": None if r is None else theta_between(r["cotyl_dir"], lm.cotyl_dir)}
        if r is not None and r.get("tip") is not None and lm.cotyl_tip is not None:
            tip_err.append(float(np.hypot(r["tip"][0] - lm.cotyl_tip[0], r["tip"][1] - lm.cotyl_tip[1])))
            if r.get("base") is not None and lm.cotyl_base is not None:
                base_err.append(float(np.hypot(r["base"][0] - lm.cotyl_base[0], r["base"][1] - lm.cotyl_base[1])))
        if r is not None and "cotyl_path" in r:
            tip_paths.append(r["cotyl_path"] != "field")
        if r is None:
            theta_err.append(missing_penalty_deg)
        else:
            theta_err.append(abs(r["theta"] - lm.theta))
            junction_err.append(float(np.hypot(r["junction"][0] - lm.junction[0], r["junction"][1] - lm.junction[1])))
        if lm.collar is not None and lm.root_dir is not None:
            if r is None or r.get("collar") is None or r.get("root_dir") is None:
                collar_err.append(missing_penalty_px)
                root_err.append(missing_penalty_deg)
            else:
                collar_err.append(float(np.hypot(r["collar"][0] - lm.collar[0], r["collar"][1] - lm.collar[1])))
                root_err.append(theta_between(r["root_dir"], lm.root_dir))
    nan = float("nan")
    return {
        "wholecrop_theta_mae": float(np.mean(theta_err)) if theta_err else nan,
        "wholecrop_theta_median": float(np.median(theta_err)) if theta_err else nan,
        "wholecrop_junction_px": float(np.median(junction_err)) if junction_err else nan,
        "wholecrop_collar_px": float(np.median(collar_err)) if collar_err else nan,
        "wholecrop_root_deg": float(np.median(root_err)) if root_err else nan,
        "wholecrop_tip_px": float(np.median(tip_err)) if tip_err else nan,
        "wholecrop_base_px": float(np.median(base_err)) if base_err else nan,
        "wholecrop_tip_used": float(np.mean(tip_paths)) if tip_paths else nan,
        "wholecrop_n": len(kept),
    }


def _auc(labels, scores):
    """Area under the ROC curve (rank form, ties averaged); NaN with a single class."""
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def measurable_scores(model, items: dict, device, patch_size: int = 252) -> dict:
    """{name: measurable probability} of whole crops ({name: image path}), read through
    MulticlassInference's own tiling (tile with the strongest junction peak, flip-averaged) --
    exactly as the GUI reads it. Crops that cannot be read, or a model without the head, are absent."""
    import cv2
    from models.UNetInference import MARGIN
    from models.multiclass_inference import LandmarkEnsemble, MulticlassInference

    models = list(model) if isinstance(model, (list, tuple)) else [model]
    for m in models:
        m.eval()
    members = [MulticlassInference.from_model(m, device=device, in_size=patch_size + 2 * MARGIN,
                                              out_size=patch_size, margin=MARGIN) for m in models]
    inference = members[0] if len(members) == 1 else LandmarkEnsemble(members)
    names, images = [], []
    for name in sorted(items):
        image = cv2.imread(str(items[name]))
        if image is not None:
            names.append(name)
            images.append(image)
    out = {}
    for start in range(0, len(images), 16):
        fields = inference.predict_landmark_fields(images[start:start + 16])
        for name, (_, _, extra) in zip(names[start:start + 16], fields):
            if extra and "measurable" in extra:
                out[name] = float(extra["measurable"])
    return out


def run_measurable_validation(model, positives: dict, negatives: dict, device, patch_size: int = 252,
                              keep_fraction: float = 0.98) -> dict:
    """The "measurable" head on WHOLE held-out crops: `positives` / `negatives` are {name: image path}
    of measured / skipped crops. measurable_auc, and measurable_rejected -- the share of skipped crops
    rejected at the threshold that keeps `keep_fraction` of the measured ones (the operating point
    the GUI gate uses)."""
    scores = measurable_scores(model, {**positives, **negatives}, device, patch_size)
    pos = [scores[n] for n in positives if n in scores]
    neg = [scores[n] for n in negatives if n in scores]
    nan = float("nan")
    if not pos or not neg:
        return {"measurable_auc": nan, "measurable_rejected": nan, "measurable_n": len(pos) + len(neg)}
    threshold = float(np.quantile(pos, 1.0 - keep_fraction))     # keeps this share of measured crops
    return {"measurable_auc": _auc([1] * len(pos) + [0] * len(neg), pos + neg),
            "measurable_rejected": float(np.mean(np.asarray(neg) < threshold)),
            "measurable_n": len(pos) + len(neg)}


def load_extra_landmark_sources(config: dict, sources, existing: dict, patch_size: int, stride: int,
                                negatives: dict | None = None):
    """Landmark-only annotation sets from `landmarks.extra_sources` (a list of {csv, raw_dir,
    root_csv?, tag?}): crops outside the training patch index, e.g. a former test set or
    another crop folder. They are loaded WITHOUT the validation-split guard (listing a set
    here is the opt-in), keyed "<tag>/<crop file>" when a tag is given (needed whenever crop
    names can repeat those of another folder), and tiled with all-ignore labels so only the
    landmark loss sees them. A source may carry `seedling_groups: <json>` ({"series:crop_id":
    "[tag/]series:crop_id"}, written by multi/match_seedlings.py) naming the seedlings it shares
    a plant with in another set, so the held-out split keeps one plant on one side.
    With `negatives` (a dict, filled in place) each source's skipped crops are loaded too
    (load_measurable_negatives, same keys and seedling groups) and tiled like the measured ones:
    the negatives of the "measurable" head.
    Returns (landmarks, raw_paths, patch_rows)."""
    import json
    from dataclasses import replace

    from PIL import Image

    landmarks, raw_paths, sizes = {}, {}, {}
    for i, src in enumerate(sources or ()):
        holder = {"src": src, "_data_root": config.get("_data_root")}
        raw_dir = resolved_path(holder, "raw_dir", section="src")
        lms, report = load_landmarks(resolved_path(holder, "csv", section="src"),
                                     root_csv_path=(resolved_path(holder, "root_csv", section="src")
                                                    if src.get("root_csv") else None))
        tag = src.get("tag")
        keyed = namespace_landmarks(lms, tag) if tag else lms
        if src.get("seedling_groups"):
            groups = json.loads(resolved_path(holder, "seedling_groups", section="src").read_text(encoding="utf-8"))
            grouped = {}
            for key, lm in keyed.items():
                raw_series = lm.seedling[0][len(tag) + 1:] if tag else lm.seedling[0]
                target = groups.get(f"{raw_series}:{lm.seedling[1]}")
                if target:
                    t_series, _, t_id = target.rpartition(":")
                    lm = replace(lm, seedling=(t_series, int(t_id)))
                grouped[key] = lm
            keyed = grouped
        clash = sorted((set(keyed) & set(existing)) | (set(keyed) & set(landmarks)))
        if clash:
            raise ValueError(f"landmarks.extra_sources[{i}] ({src['csv']}): {len(clash)} crop names already "
                             f"used by another landmark set (e.g. {clash[0]!r}); give this source a tag")
        missing = 0
        for key, lm in keyed.items():
            path = raw_dir / (key.split("/", 1)[1] if tag else key)
            if not path.exists():
                missing += 1
                continue
            with Image.open(path) as im:
                w, h = im.size
            landmarks[key], raw_paths[key], sizes[key] = lm, path, (h, w)
        n_neg = 0
        if negatives is not None:
            groups = (json.loads(resolved_path(holder, "seedling_groups", section="src").read_text(encoding="utf-8"))
                      if src.get("seedling_groups") else None)
            for key, seedling in load_measurable_negatives(resolved_path(holder, "csv", section="src"), tag=tag,
                                                           seedling_groups=groups).items():
                path = raw_dir / (key.split("/", 1)[1] if tag else key)
                if key in keyed or key in existing or not path.exists():
                    continue
                with Image.open(path) as im:
                    w, h = im.size
                negatives[key], raw_paths[key], sizes[key] = seedling, path, (h, w)
                n_neg += 1
        print(f"Extra landmarks {src['csv']}: {len(keyed) - missing} loaded"
              + (f" as {tag}/..." if tag else "") + (f", {missing} crops missing in {raw_dir}" if missing else "")
              + (f", {report['with_root']} with collar/root" if "with_root" in report else "")
              + (f", {n_neg} skipped (measurable negatives)" if negatives is not None else ""))
    return landmarks, raw_paths, landmark_patch_rows(sizes, patch_size, stride, raw_paths)


LANDMARK_COLUMNS = ["landmark_angle_mae", "junction_px_err"]
WHOLECROP_COLUMNS = ["wholecrop_theta_mae", "wholecrop_theta_median", "wholecrop_junction_px"]
WHOLECROP_ROOT_COLUMNS = ["wholecrop_collar_px", "wholecrop_root_deg"]
WHOLECROP_COTYL_COLUMNS = ["wholecrop_tip_px", "wholecrop_base_px", "wholecrop_tip_used"]
ROOT_COLUMNS = ["collar_px_err", "root_angle_err"]          # only when landmarks.root_csv is set
MEASURABLE_COLUMNS = ["measurable_auc", "measurable_rejected"]   # only with model.measurable_head


def run_training(config: dict, resume: bool = False) -> dict:
    data_cfg, training_cfg, model_cfg = config["data"], config["training"], config["model"]
    device = resolve_device(training_cfg.get("device", "auto"))
    torch.manual_seed(training_cfg.get("seed", 0))

    raw_dir = resolved_path(config, "raw_data_dir")
    masks_dir = resolved_path(config, "multiclass_masks_dir")
    patch_index_dir = resolved_path(config, "patch_index_dir")

    # raw_ext/binarize_mask default to PatchDataset's own defaults (.png raw images,
    # mask values already class indices) when absent from data_cfg - the
    # Multiclass_Anatomy_rat project sets both explicitly (.tif raw images, {0,255}
    # per-tissue masks); this config key is a no-op for the root project.
    raw_ext = data_cfg.get("raw_ext", ".png")
    binarize_mask = data_cfg.get("binarize_mask", False)

    # Optional landmark supervision (config `landmarks:` section). Only crops in
    # the TRAIN split can become landmark targets -- the validation split is the
    # held-out test set -- and the annotated seedlings are split once more into
    # landmark-train / landmark-validation so model selection never sees a
    # seedling it trained its landmarks on.
    lm_cfg = config.get("landmarks") or {}
    use_landmarks = bool(lm_cfg.get("enabled"))
    # Train only the new landmark head for this many epochs before releasing the
    # rest of the network, so a randomly initialised head's gradients cannot
    # disturb the warm-started segmentation encoder.
    head_only_epochs = int(lm_cfg.get("head_only_epochs", 0)) if use_landmarks else 0
    # Modules trained during the head-only epochs: the landmark head, plus any listed in
    # landmarks.trainable_extra (e.g. ["up4."] to let the last decoder stage adapt while the
    # rest of the backbone stays frozen; segmentation is still guarded by min_mean_fg_dice).
    head_prefixes = ("conv_kp.", "kp_trunk.", "cls_measurable.") + tuple(lm_cfg.get("trainable_extra") or ())
    # landmarks.trainable_only replaces that list: e.g. ["cls_measurable."] trains the measurable head
    # alone on frozen landmark features, so the angle maps of a warm-started model stay bit-identical.
    if lm_cfg.get("trainable_only"):
        head_prefixes = tuple(lm_cfg["trainable_only"])
    use_measurable = use_landmarks and bool(model_cfg.get("measurable_head"))
    neg_train, neg_val, neg_paths, measurable_opts = {}, {}, {}, None
    lm_train, lm_val, lm_weights = None, {}, None
    lm_columns = (LANDMARK_COLUMNS + (ROOT_COLUMNS if lm_cfg.get("root_csv") else [])
                  + WHOLECROP_COLUMNS + (WHOLECROP_ROOT_COLUMNS if lm_cfg.get("root_csv") else [])
                  + (WHOLECROP_COTYL_COLUMNS if model_cfg.get("landmark_cotyl") else [])
                  + (MEASURABLE_COLUMNS if use_measurable else []))
    if use_landmarks:
        if not model_cfg.get("landmark_head"):
            raise ValueError("landmarks.enabled requires model.landmark_head: true")
        with open(patch_index_dir / "val_patches.csv", "r", newline="", encoding="utf-8") as fh:
            val_names = {row["filename"] for row in csv.DictReader(fh)}
        all_landmarks, lm_report = load_landmarks(resolved_path(config, "csv", section="landmarks"),
                                                  exclude_filenames=val_names,
                                                  root_csv_path=(resolved_path(config, "root_csv", section="landmarks")
                                                                 if lm_cfg.get("root_csv") else None))
        print(f"Landmarks: {lm_report['loaded']} loaded, {lm_report['excluded']} in the validation split "
              f"(dropped), {lm_report['invalid']} unusable"
              + (f", {lm_report['with_root']} with collar/root" if "with_root" in lm_report else ""))
        if lm_cfg.get("root_csv") and not model_cfg.get("landmark_root"):
            raise ValueError("landmarks.root_csv requires model.landmark_root: true")
        if model_cfg.get("landmark_cotyl") and not model_cfg.get("landmark_root"):
            raise ValueError("model.landmark_cotyl requires model.landmark_root: true")
        if not all_landmarks:
            raise ValueError("landmarks.enabled but no usable annotated train-split crop was found")
        negatives = {} if use_measurable else None
        extra_landmarks, lm_raw_paths, lm_extra_rows = load_extra_landmark_sources(
            config, lm_cfg.get("extra_sources"), all_landmarks, int(data_cfg.get("patch_size", 252)),
            int(data_cfg.get("train_stride", data_cfg.get("patch_size", 252))), negatives=negatives)
        all_landmarks = {**all_landmarks, **extra_landmarks}
        if use_measurable:
            # The main set's skipped crops are already in the patch index (they carry pseudo-label masks).
            main_neg = load_measurable_negatives(resolved_path(config, "csv", section="landmarks"),
                                                 exclude_filenames=val_names)
            for key, seedling in main_neg.items():
                if key not in all_landmarks and (raw_dir / key).exists():
                    negatives[key] = seedling
                    neg_paths[key] = raw_dir / key
            neg_paths.update({k: lm_raw_paths[k] for k in negatives if k in lm_raw_paths})
            negatives = {k: v for k, v in negatives.items() if k not in all_landmarks}
        # split_seed fixes which seedlings are held out, apart from training.seed, so ensemble members
        # trained with different seeds still select their checkpoint on the same plants
        lm_train, lm_val = split_by_seedling(all_landmarks, lm_cfg.get("val_fraction", 0.15),
                                             lm_cfg.get("split_seed", training_cfg.get("seed", 0)))
        lm_weights = lm_cfg.get("loss_weights") or {}
        print(f"Landmark seedling split: {len(lm_train)} train / {len(lm_val)} validation crops")
        if use_measurable:
            # Skipped crops follow their seedling: a held-out plant is held out for both heads.
            val_seedlings = {lm.seedling for lm in lm_val.values()}
            neg_train = {k: s for k, s in negatives.items() if s not in val_seedlings}
            neg_val = {k: s for k, s in negatives.items() if s in val_seedlings}
            pos_weight = lm_weights.get("measurable_pos", "auto")
            if pos_weight == "auto":       # frame-level class balance of the training split
                pos_weight = len(neg_train) / max(len(lm_train), 1)
            measurable_opts = {"weight": float(lm_weights.get("measurable", 1.0)), "pos_weight": float(pos_weight)}
            print(f"Measurable head: {len(lm_train)} measured / {len(neg_train)} skipped training crops, "
                  f"{len(lm_val)} / {len(neg_val)} held out; pos_weight {measurable_opts['pos_weight']:.2f}")

    train_dataset = PatchDataset(
        patch_index_dir / "train_patches.csv", raw_dir, masks_dir,
        augment=True, augmentation_cfg=config.get("augmentation", {}),
        seed=training_cfg.get("seed", 0),
        raw_ext=raw_ext, binarize_mask=binarize_mask,
        landmarks=lm_train, landmark_cfg=lm_cfg,
        extra_rows=([r for r in lm_extra_rows if r["filename"] in lm_train or r["filename"] in neg_train]
                    if use_landmarks else None),
        measurable_negatives=set(neg_train) if use_measurable else None,
        # landmarks.labelled_only: train on annotated crops only (measured + skipped), e.g. for a
        # measurable-head-only run, where unlabelled segmentation tiles teach nothing
        filenames=(set(lm_train) | set(neg_train)) if (use_landmarks and lm_cfg.get("labelled_only")) else None,
    )
    val_dataset = PatchDataset(
        patch_index_dir / "val_patches.csv", raw_dir, masks_dir, augment=False,
        raw_ext=raw_ext, binarize_mask=binarize_mask,
    )

    num_workers = training_cfg.get("num_workers", 0)
    oversample = float(lm_cfg.get("oversample", 1.0)) if use_landmarks else 1.0
    sampling = lm_cfg.get("sampling", "uniform") if use_landmarks else "uniform"
    if sampling not in ("uniform", "angle_balanced"):
        raise ValueError("landmarks.sampling must be 'uniform' or 'angle_balanced'")
    if oversample > 1.0:
        # Annotated crops are a small fraction of the patches: draw them more often. With
        # sampling: angle_balanced each angle bin (closed / middle / open hook) is drawn equally
        # often among the annotated crops, on top of the oversampling.
        balance = angle_balanced_weights(lm_train) if sampling == "angle_balanced" else {}
        if balance:
            print("Angle-balanced sampling of the annotated crops: "
                  + ", ".join(f"{label} {sum(1 for lm in lm_train.values() if angle_bin(lm.theta) == b)} crops x"
                              f"{next(balance[n] for n, lm in lm_train.items() if angle_bin(lm.theta) == b):.2f}"
                              for b, label in enumerate(("closed", "middle", "open"))
                              if any(angle_bin(lm.theta) == b for lm in lm_train.values())))
        sample_weights = [oversample * balance.get(row["filename"], 1.0) if row["filename"] in lm_train
                          else float(lm_cfg.get("negative_weight", 1.0)) if row["filename"] in neg_train else 1.0
                          for row in train_dataset.rows]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=training_cfg["batch_size"],
                                   sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=training_cfg["batch_size"],
                                   shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=training_cfg["batch_size"],
                             shuffle=False, num_workers=num_workers)
    lm_val_loader = None
    if use_landmarks and lm_val:
        lm_val_dataset = PatchDataset(
            patch_index_dir / "train_patches.csv", raw_dir, masks_dir, augment=False,
            raw_ext=raw_ext, binarize_mask=binarize_mask,
            landmarks=lm_val, filenames=set(lm_val), landmark_cfg=lm_cfg, extra_rows=lm_extra_rows,
        )
        lm_val_loader = DataLoader(lm_val_dataset, batch_size=training_cfg["batch_size"],
                                    shuffle=False, num_workers=num_workers)

    model = build_model(model_cfg).to(device)
    loss_fn = build_loss(config["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_cfg["lr_max"],
                                   weight_decay=training_cfg.get("weight_decay", 0.0))

    use_amp = bool(training_cfg.get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if device.type == "cuda" else None
    grad_clip_norm = training_cfg.get("grad_clip_norm")

    checkpoints_dir = resolved_path(config, "checkpoints_dir")
    logs_dir = resolved_path(config, "logs_dir")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = checkpoints_dir / "best.pt"
    last_checkpoint_path = checkpoints_dir / "last.pt"

    num_classes = data_cfg.get("num_classes", model_cfg.get("num_classes", 2))
    class_names = data_cfg.get("class_names") or [f"class{i}" for i in range(num_classes)]
    ignore_index = config.get("loss", {}).get("ignore_index", 255)

    # Which validation metric selects "best" (drives both the best.pt
    # checkpoint and early stopping): "mean_fg_dice" (default) or the old
    # "val_loss" behaviour. Background is 95.9% of all pixels here, so raw
    # val loss is dominated by a class nobody cares about getting right --
    # mean_fg_dice (Dice averaged over classes 1..num_classes-1, see
    # compute_class_metrics) is blind to background and rewards actually
    # learning the rare foreground classes instead.
    #
    # "landmark_angle_mae" (needs the landmarks section) selects on the angle
    # error read back from the landmark head on the landmark-validation
    # seedlings (lower is better), but only among epochs whose mean_fg_dice
    # stays at or above landmarks.min_mean_fg_dice, so fine-tuning for
    # landmarks cannot quietly wreck the segmentation.
    selection_metric = training_cfg.get("selection_metric", "mean_fg_dice")
    if selection_metric not in ("mean_fg_dice", "val_loss", "landmark_angle_mae", "wholecrop_theta_mae",
                                "measurable_auc"):
        raise ValueError(
            "training.selection_metric must be 'mean_fg_dice', 'val_loss', 'landmark_angle_mae', "
            "'wholecrop_theta_mae' or 'measurable_auc', "
            f"got {selection_metric!r}"
        )
    if selection_metric in ("landmark_angle_mae", "wholecrop_theta_mae") and not use_landmarks:
        raise ValueError(f"selection_metric {selection_metric!r} requires landmarks.enabled")
    if selection_metric == "measurable_auc" and not use_measurable:
        raise ValueError("selection_metric 'measurable_auc' requires model.measurable_head and landmarks.enabled")
    higher_is_better = selection_metric in ("mean_fg_dice", "measurable_auc")
    min_fg_dice = float(lm_cfg.get("min_mean_fg_dice", 0.0))

    metric_columns = (
        ["mean_fg_dice"]
        + [f"{name}_iou" for name in class_names]
        + [f"{name}_dice" for name in class_names]
        + (lm_columns if use_landmarks else [])
    )

    start_epoch = 0
    best_val_loss = float("inf")
    best_metric = float("-inf") if higher_is_better else float("inf")
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}
    for key in metric_columns:
        history.setdefault(key, [])

    if resume:
        if not last_checkpoint_path.exists():
            raise FileNotFoundError(
                f"--resume was passed but no checkpoint exists at {last_checkpoint_path}"
            )
        # weights_only=False on purpose: last.pt is this run's own resume file (optimizer/scaler state
        # and history), never a downloaded weight file.
        checkpoint = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(strip_deprecated_landmark_keys(checkpoint["model_state_dict"]))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        history = checkpoint.get("history", history)
        for key in metric_columns:
            history.setdefault(key, [])

        # An older checkpoint (predating selection_metric/best_metric), or one
        # written under a different selection_metric, has no directly
        # comparable "best" value for the metric this run is selecting on.
        # Rather than crash on a missing key, fall back: val_loss is always
        # comparable across runs, and any other metric just starts fresh so
        # the next epoch is free to become the new best.
        if "best_metric" in checkpoint and checkpoint.get("selection_metric") == selection_metric:
            best_metric = checkpoint["best_metric"]
        elif selection_metric == "val_loss":
            best_metric = best_val_loss
        else:
            best_metric = float("-inf") if higher_is_better else float("inf")
        print(f"Resuming from {last_checkpoint_path}: starting at epoch {start_epoch} "
              f"(best {selection_metric} so far: {best_metric:.4f})")

    # A fresh timestamped log is written each run (rather than appended-to across
    # resumes) so it stays self-contained even if checkpoints move between hosts;
    # any epochs already completed are replayed into it from the checkpoint's history.
    log_path = logs_dir / f"train_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "val_loss"] + metric_columns)
        n_prior = len(history["train_loss"])
        for prior_epoch in range(n_prior):
            row = [prior_epoch, history["train_loss"][prior_epoch], history["val_loss"][prior_epoch]]
            for key in metric_columns:
                values = history.get(key, [])
                # Older checkpoints, or a resume with a different num_classes/
                # class_names, may not have a value for every prior epoch --
                # leave those cells blank rather than crash or fabricate a number.
                row.append(values[prior_epoch] if prior_epoch < len(values) else "")
            writer.writerow(row)
        fh.flush()

        for epoch in range(start_epoch, training_cfg["num_epochs"]):
            epoch_start = time.time()
            lr = compute_lr(epoch, training_cfg)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            if use_landmarks:
                head_only = epoch < head_only_epochs
                for name, param in model.named_parameters():
                    param.requires_grad = (not head_only) or name.startswith(head_prefixes)

            extra_kwargs = {"landmark_weights": lm_weights, "measurable": measurable_opts} if use_landmarks else {}
            train_loss, _ = run_epoch(model, train_loader, loss_fn, optimizer, device,
                                       desc=f"Epoch {epoch} [train]", use_amp=use_amp, scaler=scaler,
                                       grad_clip_norm=grad_clip_norm, **extra_kwargs)
            val_loss, confusion = run_epoch(model, val_loader, loss_fn, None, device,
                                             desc=f"Epoch {epoch} [val]", use_amp=use_amp,
                                             num_classes=num_classes, ignore_index=ignore_index)
            metrics = compute_class_metrics(confusion)

            lm_metrics = {}
            if use_landmarks:
                lm_metrics = (run_landmark_validation(model, lm_val_loader, device,
                                                      readout_radius=lm_cfg.get("readout_radius", 3.0))
                              if lm_val_loader is not None else {})
                if lm_val:
                    lm_metrics.update(run_wholecrop_landmark_validation(
                        model, lm_val, raw_dir, device, raw_ext=raw_ext, raw_paths=lm_raw_paths,
                        patch_size=int(data_cfg.get("patch_size", 252)),
                        readout_radius=lm_cfg.get("readout_radius", 3.0)))
                if use_measurable:
                    positives = {k: (lm_raw_paths.get(k) or raw_dir / (Path(k).stem + raw_ext)) for k in lm_val}
                    lm_metrics.update(run_measurable_validation(
                        model, positives, {k: neg_paths[k] for k in neg_val if k in neg_paths}, device,
                        patch_size=int(data_cfg.get("patch_size", 252))))
                for key in lm_columns:
                    history[key].append(lm_metrics.get(key, float("nan")))

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["mean_fg_dice"].append(metrics["mean_fg_dice"])
            for i, name in enumerate(class_names):
                history[f"{name}_iou"].append(metrics["iou"][i])
                history[f"{name}_dice"].append(metrics["dice"][i])

            row = [epoch, train_loss, val_loss, metrics["mean_fg_dice"]]
            row += [metrics["iou"][i] for i in range(len(class_names))]
            row += [metrics["dice"][i] for i in range(len(class_names))]
            if use_landmarks:
                row += [lm_metrics.get(key, float("nan")) for key in lm_columns]
            writer.writerow(row)
            fh.flush()

            if selection_metric in ("landmark_angle_mae", "wholecrop_theta_mae"):
                mae = lm_metrics.get(selection_metric, float("nan"))
                eligible = not math.isnan(mae) and metrics["mean_fg_dice"] >= min_fg_dice
                current_metric = mae if eligible else float("inf")
            elif selection_metric == "measurable_auc":
                auc = lm_metrics.get("measurable_auc", float("nan"))
                current_metric = auc if not math.isnan(auc) else float("-inf")
            else:
                current_metric = metrics["mean_fg_dice"] if higher_is_better else val_loss
            improved = current_metric > best_metric if higher_is_better else current_metric < best_metric
            if improved:
                best_metric = current_metric
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_checkpoint_path)
            else:
                epochs_without_improvement += 1

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "best_val_loss": best_val_loss,
                "best_metric": best_metric,
                "selection_metric": selection_metric,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
            }, last_checkpoint_path)

            elapsed_min = (time.time() - epoch_start) / 60
            marker = " (best)" if improved else ""
            lm_text = ""
            if use_landmarks and lm_metrics:
                lm_text = (f" landmark_angle_mae={lm_metrics['landmark_angle_mae']:.1f}"
                           f" junction_px_err={lm_metrics['junction_px_err']:.1f}")
                if "wholecrop_theta_mae" in lm_metrics:
                    lm_text += (f" | whole-crop theta={lm_metrics['wholecrop_theta_mae']:.1f}"
                                f" (median {lm_metrics['wholecrop_theta_median']:.1f})"
                                f" junction={lm_metrics['wholecrop_junction_px']:.1f}px")
                    if "wholecrop_collar_px" in lm_columns:
                        lm_text += (f" collar={lm_metrics['wholecrop_collar_px']:.1f}px"
                                    f" root={lm_metrics['wholecrop_root_deg']:.1f}deg")
                if "collar_px_err" in lm_columns:
                    lm_text += (f" collar_px_err={lm_metrics['collar_px_err']:.1f}"
                                f" root_angle_err={lm_metrics['root_angle_err']:.1f}")
                if "measurable_auc" in lm_metrics:
                    lm_text += (f" | measurable AUC={lm_metrics['measurable_auc']:.3f}"
                                f" rejected@98%={lm_metrics['measurable_rejected']:.2f}")
            print(f"Epoch {epoch}: lr={lr:.2e} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"mean_fg_dice={metrics['mean_fg_dice']:.4f}{lm_text}{marker} [{elapsed_min:.1f} min]")

            if epochs_without_improvement >= training_cfg.get("early_stopping_patience", float("inf")):
                print(f"Early stopping: no improvement for {epochs_without_improvement} epochs")
                break

    if not best_checkpoint_path.exists():
        print(f"WARNING: no epoch satisfied the selection rule ({selection_metric}"
              f"{f', mean_fg_dice >= {min_fg_dice}' if selection_metric in ('landmark_angle_mae', 'wholecrop_theta_mae') else ''}); "
              "no best checkpoint was written -- see last.pt")
    history["best_val_loss"] = best_val_loss
    history["best_metric"] = best_metric
    history["selection_metric"] = selection_metric
    history["checkpoint_path"] = str(best_checkpoint_path)
    history["log_path"] = str(log_path)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to training_config.yaml")
    parser.add_argument("--data-root", default=None, help="Root dir paths are resolved against")
    parser.add_argument("--resume", action="store_true",
                         help="Resume from checkpoints_dir/last.pt instead of starting fresh")
    args = parser.parse_args()

    config = load_config(args.config, args.data_root)
    history = run_training(config, resume=args.resume)
    print(f"Best val loss: {history['best_val_loss']:.4f}")
    print(f"Best {history['selection_metric']}: {history['best_metric']:.4f}")
    print(f"Checkpoint: {history['checkpoint_path']}")
    print(f"Log: {history['log_path']}")


if __name__ == "__main__":
    main()
