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
from src.landmarks import load_landmarks, readout_from_fields, split_by_seedling, theta_between
from src.loss_functions import align_output_to_target, build_loss, landmark_loss
from src.model import build_model


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
              ignore_index: int = 255, landmark_weights: dict | None = None) -> tuple[float, np.ndarray | None]:
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
    element is ignored and the model's plain forward() is used.
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
                if use_landmarks:
                    outputs, kp_out, overhook_out = model.forward_with_landmarks(images)
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
                    kp_loss, _ = landmark_loss(align_output_to_target(kp_out, masks), overhook_out,
                                                targets, landmark_weights)
                    loss = loss + kp_loss
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

    Returns landmark_angle_mae (deg), junction_px_err, overhook_acc,
    landmark_found (fraction) and n; the means are NaN when the loader holds no
    patch with a junction."""
    model.eval()
    angle_err, junction_err, overhook_ok, found, n = [], [], [], 0, 0
    collar_err, root_err = [], []
    with torch.no_grad():
        for images, masks, targets in loader:
            _, kp_out, overhook_out = model.forward_with_landmarks(images.to(device))
            kp_out = align_output_to_target(kp_out, masks).float().cpu()
            overhook_p = torch.sigmoid(overhook_out.float()).cpu().numpy()
            hm_p = torch.sigmoid(kp_out[:, 0]).numpy()
            vec_p = kp_out[:, 1:5].numpy()
            has_root = kp_out.shape[1] >= 8 and "collar_hm" in targets
            if has_root:
                collar_p = torch.sigmoid(kp_out[:, 5]).numpy()
                root_p = kp_out[:, 6:8].numpy()
            for i in range(images.shape[0]):
                if has_root and float(targets["has_collar"][i]) > 0.5 and float(targets["has_kp"][i]) > 0.5:
                    t_c = readout_from_fields(targets["hm"][i].numpy(), targets["paf"][i].numpy(), 0.0,
                                              radius=readout_radius, collar_hm=targets["collar_hm"][i].numpy(),
                                              root_vec=targets["root_vec"][i].numpy())
                    if t_c is not None and t_c["collar"] is not None and t_c["root_dir"] is not None:
                        p_c = readout_from_fields(hm_p[i], vec_p[i], 0.0, radius=readout_radius,
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
                                            float(targets["overhook"][i]), radius=readout_radius)
                if truth is None:
                    continue
                n += 1
                overhook_ok.append(float((overhook_p[i] > 0.5) == (float(targets["overhook"][i]) > 0.5)))
                pred = readout_from_fields(hm_p[i], vec_p[i], overhook_p[i], radius=readout_radius)
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
        "overhook_acc": float(np.mean(overhook_ok)) if overhook_ok else nan,
        "landmark_found": found / n if n else nan,
        "collar_px_err": float(np.mean(collar_err)) if collar_err else nan,
        "root_angle_err": float(np.mean(root_err)) if root_err else nan,
        "n": n,
    }


def run_wholecrop_landmark_validation(model, landmarks: dict, raw_dir, device, raw_ext: str = ".png",
                                      patch_size: int = 252, readout_radius: float = 3.0,
                                      missing_penalty_deg: float = 90.0, missing_penalty_px: float = 30.0) -> dict:
    """Landmark metrics on WHOLE crops of the held-out annotated seedlings, read through
    MulticlassInference's own tiling and readout -- the conditions the model is used in.
    The patch metrics of run_landmark_validation score 252 px tiles, where padding,
    tiles that cut the hook and the miss penalty make them noisy enough to pick a poor
    checkpoint; this is the metric to select on ('wholecrop_theta_mae').

    Errors are against the annotated landmark itself (not a re-rendered target). A crop
    with no junction found counts as missing_penalty_deg; a missing collar as
    missing_penalty_px / missing_penalty_deg."""
    import cv2
    from models.UNetInference import MARGIN
    from models.multiclass_inference import MulticlassInference

    model.eval()
    images, kept = [], []
    for name in sorted(landmarks):
        image = cv2.imread(str(Path(raw_dir) / (Path(name).stem + raw_ext)))
        if image is not None:
            images.append(image)
            kept.append(name)
    inference = MulticlassInference.from_model(model, device=device, in_size=patch_size + 2 * MARGIN,
                                               out_size=patch_size, margin=MARGIN)
    readouts = []
    for start in range(0, len(images), 16):
        readouts += inference.predict_landmarks(images[start:start + 16], readout_radius=readout_radius)

    theta_err, junction_err, collar_err, root_err = [], [], [], []
    for name, r in zip(kept, readouts):
        lm = landmarks[name]
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
        "wholecrop_n": len(kept),
    }


LANDMARK_COLUMNS = ["landmark_angle_mae", "junction_px_err", "overhook_acc"]
WHOLECROP_COLUMNS = ["wholecrop_theta_mae", "wholecrop_theta_median", "wholecrop_junction_px"]
WHOLECROP_ROOT_COLUMNS = ["wholecrop_collar_px", "wholecrop_root_deg"]
ROOT_COLUMNS = ["collar_px_err", "root_angle_err"]          # only when landmarks.root_csv is set


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
    lm_train, lm_val, lm_weights = None, {}, None
    lm_columns = (LANDMARK_COLUMNS + (ROOT_COLUMNS if lm_cfg.get("root_csv") else [])
                  + WHOLECROP_COLUMNS + (WHOLECROP_ROOT_COLUMNS if lm_cfg.get("root_csv") else []))
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
        if not all_landmarks:
            raise ValueError("landmarks.enabled but no usable annotated train-split crop was found")
        lm_train, lm_val = split_by_seedling(all_landmarks, lm_cfg.get("val_fraction", 0.15),
                                             training_cfg.get("seed", 0))
        lm_weights = lm_cfg.get("loss_weights") or {}
        print(f"Landmark seedling split: {len(lm_train)} train / {len(lm_val)} validation crops")

    train_dataset = PatchDataset(
        patch_index_dir / "train_patches.csv", raw_dir, masks_dir,
        augment=True, augmentation_cfg=config.get("augmentation", {}),
        seed=training_cfg.get("seed", 0),
        raw_ext=raw_ext, binarize_mask=binarize_mask,
        landmarks=lm_train, landmark_cfg=lm_cfg,
    )
    val_dataset = PatchDataset(
        patch_index_dir / "val_patches.csv", raw_dir, masks_dir, augment=False,
        raw_ext=raw_ext, binarize_mask=binarize_mask,
    )

    num_workers = training_cfg.get("num_workers", 0)
    oversample = float(lm_cfg.get("oversample", 1.0)) if use_landmarks else 1.0
    if oversample > 1.0:
        # Annotated crops are a small fraction of the patches: draw them more often
        sample_weights = [oversample if row["filename"] in lm_train else 1.0 for row in train_dataset.rows]
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
            landmarks=lm_val, filenames=set(lm_val), landmark_cfg=lm_cfg,
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
    if selection_metric not in ("mean_fg_dice", "val_loss", "landmark_angle_mae", "wholecrop_theta_mae"):
        raise ValueError(
            "training.selection_metric must be 'mean_fg_dice', 'val_loss', 'landmark_angle_mae' or "
            "'wholecrop_theta_mae', "
            f"got {selection_metric!r}"
        )
    if selection_metric in ("landmark_angle_mae", "wholecrop_theta_mae") and not use_landmarks:
        raise ValueError(f"selection_metric {selection_metric!r} requires landmarks.enabled")
    higher_is_better = selection_metric == "mean_fg_dice"
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
        checkpoint = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
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
                    param.requires_grad = (not head_only) or name.startswith(("conv_kp.", "cls_overhook.", "kp_trunk."))

            extra_kwargs = {"landmark_weights": lm_weights} if use_landmarks else {}
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
                        model, lm_val, raw_dir, device, raw_ext=raw_ext,
                        patch_size=int(data_cfg.get("patch_size", 252)),
                        readout_radius=lm_cfg.get("readout_radius", 3.0)))
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
                           f" junction_px_err={lm_metrics['junction_px_err']:.1f}"
                           f" overhook_acc={lm_metrics['overhook_acc']:.2f}")
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
