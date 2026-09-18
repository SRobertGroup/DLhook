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
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config, resolved_path
from src.data_loader import PatchDataset
from src.loss_functions import align_output_to_target, build_loss
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
              ignore_index: int = 255) -> tuple[float, np.ndarray | None]:
    """Run one training epoch (`optimizer` given) or one validation pass
    (`optimizer=None`).

    When `num_classes` is given, also accumulates a `num_classes x
    num_classes` confusion matrix over the pass (see
    `update_confusion_matrix`) and returns it as the second element,
    otherwise the second element is None. Only the validation call in
    `run_training` passes `num_classes` -- the extra argmax + bincount per
    batch is cheap, but there's no need to pay it on every training batch
    too when only the validation metrics drive checkpoint selection.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n_batches = 0.0, 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64) if num_classes else None
    progress = tqdm(loader, desc=desc, unit="batch", leave=False)
    with torch.set_grad_enabled(is_train):
        for images, masks in progress:
            images, masks = images.to(device), masks.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                # See align_output_to_target's docstring: PatchDataset feeds
                # patch_size + 2*MARGIN, whose output the fixed architecture
                # rounds to a multiple of 16 -- a few pixels off patch_size
                # for every get_valid_patch_sizes() value. Crop that
                # architecture-forced remainder off the model's own output,
                # never off the label, right before the loss.
                outputs = align_output_to_target(outputs, masks)
                loss = loss_fn(outputs, masks)
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

    train_dataset = PatchDataset(
        patch_index_dir / "train_patches.csv", raw_dir, masks_dir,
        augment=True, augmentation_cfg=config.get("augmentation", {}),
        seed=training_cfg.get("seed", 0),
        raw_ext=raw_ext, binarize_mask=binarize_mask,
    )
    val_dataset = PatchDataset(
        patch_index_dir / "val_patches.csv", raw_dir, masks_dir, augment=False,
        raw_ext=raw_ext, binarize_mask=binarize_mask,
    )

    num_workers = training_cfg.get("num_workers", 0)
    train_loader = DataLoader(train_dataset, batch_size=training_cfg["batch_size"],
                               shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=training_cfg["batch_size"],
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
    selection_metric = training_cfg.get("selection_metric", "mean_fg_dice")
    if selection_metric not in ("mean_fg_dice", "val_loss"):
        raise ValueError(
            f"training.selection_metric must be 'mean_fg_dice' or 'val_loss', got {selection_metric!r}"
        )
    higher_is_better = selection_metric == "mean_fg_dice"

    metric_columns = (
        ["mean_fg_dice"]
        + [f"{name}_iou" for name in class_names]
        + [f"{name}_dice" for name in class_names]
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
            best_metric = float("-inf")
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

            train_loss, _ = run_epoch(model, train_loader, loss_fn, optimizer, device,
                                       desc=f"Epoch {epoch} [train]", use_amp=use_amp, scaler=scaler,
                                       grad_clip_norm=grad_clip_norm)
            val_loss, confusion = run_epoch(model, val_loader, loss_fn, None, device,
                                             desc=f"Epoch {epoch} [val]", use_amp=use_amp,
                                             num_classes=num_classes, ignore_index=ignore_index)
            metrics = compute_class_metrics(confusion)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["mean_fg_dice"].append(metrics["mean_fg_dice"])
            for i, name in enumerate(class_names):
                history[f"{name}_iou"].append(metrics["iou"][i])
                history[f"{name}_dice"].append(metrics["dice"][i])

            row = [epoch, train_loss, val_loss, metrics["mean_fg_dice"]]
            row += [metrics["iou"][i] for i in range(len(class_names))]
            row += [metrics["dice"][i] for i in range(len(class_names))]
            writer.writerow(row)
            fh.flush()

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
            print(f"Epoch {epoch}: lr={lr:.2e} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"mean_fg_dice={metrics['mean_fg_dice']:.4f}{marker} [{elapsed_min:.1f} min]")

            if epochs_without_improvement >= training_cfg.get("early_stopping_patience", float("inf")):
                print(f"Early stopping: no improvement for {epochs_without_improvement} epochs")
                break

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
