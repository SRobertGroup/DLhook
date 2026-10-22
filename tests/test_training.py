"""Tests for multi/train_unet_multiclass.py's per-class validation metrics and
metric-driven checkpoint selection.

`run_training` normally trains against real image/mask data that lives
outside the repo (see multi/configs/training_config.yaml), so these tests
never touch real images: the patch CSVs are written with zero rows and
`run_epoch` is monkeypatched to a scripted stand-in, which is enough because
the DataLoader is never actually iterated when `run_epoch` is replaced.
That keeps this suite fast, GPU-free, and focused purely on the selection/
bookkeeping logic around run_epoch rather than the training math itself
(covered separately by tests/test_multi_model.py and
tests/test_multi_data_loader.py).
"""
from __future__ import annotations

import math

import numpy as np
import torch

import multi.train_unet_multiclass as train_module
from multi.src.model import build_model
from multi.train_unet_multiclass import (
    compute_class_metrics,
    run_training,
    update_confusion_matrix,
)

CSV_HEADER = "filename,x,y,patch_size,foreground_fraction\n"


def _write_stub_patch_csv(path):
    """A one-row patch manifest. The row never needs to resolve to a real
    image: every test here monkeypatches run_epoch, so the DataLoader built
    from it is never iterated -- but torch's DataLoader(shuffle=True)
    constructs a RandomSampler eagerly and raises if the dataset is empty,
    so PatchDataset needs at least one row just to be constructable."""
    path.write_text(CSV_HEADER + "dummy.png,0,0,32,0.0\n", encoding="utf-8")


def _make_config(tmp_path, *, selection_metric="mean_fg_dice", num_epochs=2,
                  num_classes=2, class_names=None, run_dir="run"):
    """A minimal, self-contained training_config.yaml-equivalent dict. Points
    every path at empty/tiny fixtures under tmp_path/run_dir so run_training
    can build real Dataset/DataLoader/model/optimizer objects without any
    real image data (the loaders are never iterated once run_epoch is
    monkeypatched)."""
    root = tmp_path / run_dir
    (root / "raw").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    patch_index_dir = root / "patch_index"
    patch_index_dir.mkdir(parents=True)
    _write_stub_patch_csv(patch_index_dir / "train_patches.csv")
    _write_stub_patch_csv(patch_index_dir / "val_patches.csv")

    return {
        "_data_root": str(root),
        "paths": {
            "raw_data_dir": "raw",
            "multiclass_masks_dir": "masks",
            "patch_index_dir": "patch_index",
            "checkpoints_dir": "checkpoints",
            "logs_dir": "logs",
        },
        "data": {
            "num_classes": num_classes,
            "class_names": class_names or [f"class{i}" for i in range(num_classes)],
        },
        "model": {"num_classes": num_classes, "in_channels": 3},
        "loss": {"type": "cross_entropy", "ignore_index": 255},
        "training": {
            "batch_size": 1,
            "num_epochs": num_epochs,
            "lr_max": 1.0e-3,
            "device": "cpu",
            "num_workers": 0,
            "seed": 0,
            "early_stopping_patience": 1000,
            "selection_metric": selection_metric,
        },
    }


def _make_fake_run_epoch(val_script, mutate_model=True):
    """Stand-in for train_unet_multiclass.run_epoch: the training call is a
    no-op (loss 0.0, no confusion matrix) except it optionally stamps every
    model parameter with the epoch index, so a test can later tell which
    epoch's weights ended up in best.pt. The validation call returns the
    scripted (val_loss, confusion) pair for that epoch, read out of `desc`
    ("Epoch {n} [train]"/"[val]") exactly as run_training formats it."""

    def fake_run_epoch(model, loader, loss_fn, optimizer, device, desc,
                        use_amp=False, scaler=None, grad_clip_norm=None,
                        num_classes=None, ignore_index=255):
        epoch = int(desc.split()[1])
        is_train = optimizer is not None
        if is_train:
            if mutate_model:
                with torch.no_grad():
                    for p in model.parameters():
                        p.fill_(float(epoch))
            return 0.0, None
        val_loss, confusion = val_script[epoch]
        return val_loss, confusion

    return fake_run_epoch


# --- confusion matrix -------------------------------------------------------

def test_update_confusion_matrix_hand_built_case_excludes_ignore_pixels():
    num_classes = 3
    ignore_index = 255
    # Row 0 has one ignored pixel (target==255) that must be dropped
    # entirely -- it must not count towards ANY cell, including a
    # correct-by-coincidence one.
    target = torch.tensor([[0, 1, 2, 255], [1, 1, 0, 2]])
    pred = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 2]])

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    update_confusion_matrix(confusion, target, pred, num_classes, ignore_index)

    expected = np.array([
        [2, 0, 0],  # true=0: predicted 0 twice
        [1, 2, 0],  # true=1: predicted 0 once, predicted 1 twice
        [0, 1, 1],  # true=2: predicted 1 once, predicted 2 once
    ])
    assert np.array_equal(confusion, expected)
    # 8 pixels total, 1 ignored -> exactly 7 counted.
    assert confusion.sum() == 7


def test_update_confusion_matrix_accumulates_across_calls():
    num_classes = 2
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    update_confusion_matrix(confusion, torch.tensor([0, 1]), torch.tensor([0, 1]), num_classes)
    update_confusion_matrix(confusion, torch.tensor([1, 1]), torch.tensor([0, 1]), num_classes)
    assert np.array_equal(confusion, np.array([[1, 0], [1, 2]]))


# --- per-class IoU/Dice ------------------------------------------------------

def test_compute_class_metrics_matches_hand_computed_values():
    # true=0(bg): 5 correct, 1 predicted as class1 (FP for class1).
    # true=1: 1 predicted bg (FN), 3 correct.
    # true=2: never occurs in truth or in any prediction.
    confusion = np.array([
        [5, 0, 0],
        [1, 3, 0],
        [0, 0, 0],
    ])

    metrics = compute_class_metrics(confusion)

    # class1: TP=3, FP=col1-TP=3-3=0, FN=row1_sum-TP=4-3=1
    # -> IoU=3/(3+0+1)=0.75, Dice=2*3/(2*3+0+1)=6/7
    assert metrics["iou"][1] == 0.75
    assert metrics["dice"][1] == 6 / 7

    # class2: absent from both truth and prediction -> undefined, NaN --
    # and must not silently become 0 (which would poison a mean).
    assert math.isnan(metrics["iou"][2])
    assert math.isnan(metrics["dice"][2])

    # mean_fg_dice averages classes 1..N-1 only, and must skip the NaN
    # class2 entry rather than propagate it -- so it equals class1's Dice
    # alone, and must itself be a real (non-NaN) number.
    assert not math.isnan(metrics["mean_fg_dice"])
    assert metrics["mean_fg_dice"] == 6 / 7


def test_compute_class_metrics_all_foreground_classes_absent_defaults_to_zero_not_nan():
    confusion = np.array([[10, 0], [0, 0]])  # only background ever occurs
    metrics = compute_class_metrics(confusion)
    assert math.isnan(metrics["dice"][1])
    assert metrics["mean_fg_dice"] == 0.0


def test_compute_class_metrics_none_confusion_is_a_safe_default():
    metrics = compute_class_metrics(None)
    assert metrics["mean_fg_dice"] == 0.0
    assert metrics["iou"] == []


# --- checkpoint selection -----------------------------------------------------

def test_checkpoint_selection_prefers_mean_fg_dice_over_worse_val_loss(tmp_path, monkeypatch):
    """The exact scenario mean-foreground-Dice selection exists for: epoch 0
    has the lower (better) val loss but the worse Dice; epoch 1 has a higher
    (worse) val loss but much better Dice. Selecting on mean_fg_dice must
    keep epoch 1's checkpoint; selecting on val_loss must keep epoch 0's."""
    # dice = 2*TP / (2*TP + FP + FN)
    confusion_epoch0 = np.array([[10, 1], [1, 1]])  # TP=1,FP=1,FN=1 -> dice=0.5
    confusion_epoch1 = np.array([[10, 1], [1, 9]])  # TP=9,FP=1,FN=1 -> dice=0.9
    val_script = {0: (0.05, confusion_epoch0), 1: (0.20, confusion_epoch1)}

    probe_model = build_model({"num_classes": 2, "in_channels": 3})
    param_name = next(iter(probe_model.named_parameters()))[0]

    for selection_metric, expected_epoch, expected_best_metric, expected_best_val_loss in [
        ("mean_fg_dice", 1, 0.9, 0.20),
        ("val_loss", 0, 0.05, 0.05),
    ]:
        config = _make_config(tmp_path, selection_metric=selection_metric,
                               run_dir=f"run_{selection_metric}")
        monkeypatch.setattr(train_module, "run_epoch", _make_fake_run_epoch(val_script))

        history = run_training(config)

        assert history["selection_metric"] == selection_metric
        assert history["best_metric"] == expected_best_metric
        assert history["best_val_loss"] == expected_best_val_loss

        best_state = torch.load(history["checkpoint_path"], map_location="cpu", weights_only=False)
        assert torch.all(best_state[param_name] == float(expected_epoch)), (
            f"selection_metric={selection_metric!r}: expected best.pt to hold epoch "
            f"{expected_epoch}'s weights"
        )


def test_log_csv_has_per_class_columns(tmp_path, monkeypatch):
    confusion = np.array([[10, 1], [1, 1]])
    val_script = {0: (0.05, confusion), 1: (0.05, confusion)}
    config = _make_config(tmp_path, num_epochs=2, class_names=["background", "fg"],
                           run_dir="run_csv")
    monkeypatch.setattr(train_module, "run_epoch", _make_fake_run_epoch(val_script))

    history = run_training(config)

    header = open(history["log_path"], encoding="utf-8").readline().strip().split(",")
    assert header == [
        "epoch", "train_loss", "val_loss", "mean_fg_dice",
        "background_iou", "fg_iou", "background_dice", "fg_dice",
    ]


# --- resume ------------------------------------------------------------------

def test_resume_from_checkpoint_missing_best_metric_key_does_not_crash(tmp_path, monkeypatch):
    """An old checkpoint written before this feature existed has no
    'best_metric'/'selection_metric' keys (only the legacy 'best_val_loss').
    Resuming onto the new mean_fg_dice-selecting code must not KeyError --
    it should fall back gracefully and keep training."""
    config = _make_config(tmp_path, selection_metric="mean_fg_dice", num_epochs=2,
                           run_dir="run_resume")

    model = build_model(config["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr_max"])
    checkpoints_dir = tmp_path / "run_resume" / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    torch.save({
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": None,
        "best_val_loss": 0.5,
        # No "best_metric" / "selection_metric" -- the old checkpoint format.
        "epochs_without_improvement": 0,
        "history": {"train_loss": [0.3], "val_loss": [0.5]},
    }, checkpoints_dir / "last.pt")

    confusion = np.array([[10, 1], [1, 4]])  # TP=4,FP=1,FN=1 -> dice=2*4/(8+1+1)=0.8
    val_script = {1: (0.4, confusion)}
    monkeypatch.setattr(train_module, "run_epoch", _make_fake_run_epoch(val_script))

    history = run_training(config, resume=True)  # must not raise

    assert history["best_metric"] == 0.8
    assert history["selection_metric"] == "mean_fg_dice"
    # Resumed history must still carry the one pre-existing epoch plus the
    # newly run one.
    assert history["train_loss"] == [0.3, 0.0]
    assert history["val_loss"] == [0.5, 0.4]


def test_resume_with_val_loss_selection_falls_back_to_best_val_loss(tmp_path, monkeypatch):
    """Same old-format checkpoint, but resuming under selection_metric=
    'val_loss': best_val_loss IS directly comparable across runs, so it
    should seed best_metric instead of resetting to worst-possible."""
    config = _make_config(tmp_path, selection_metric="val_loss", num_epochs=2,
                           run_dir="run_resume_valloss")

    model = build_model(config["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr_max"])
    checkpoints_dir = tmp_path / "run_resume_valloss" / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    torch.save({
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": None,
        "best_val_loss": 0.1,
        "epochs_without_improvement": 0,
        "history": {"train_loss": [0.3], "val_loss": [0.1]},
    }, checkpoints_dir / "last.pt")

    confusion = np.array([[10, 1], [1, 4]])
    # Epoch 1's val loss (0.9) is worse than the prior best (0.1), so under
    # val_loss selection the resumed best_val_loss/best_metric must NOT move.
    val_script = {1: (0.9, confusion)}
    monkeypatch.setattr(train_module, "run_epoch", _make_fake_run_epoch(val_script))

    history = run_training(config, resume=True)

    assert history["best_metric"] == 0.1
    assert history["best_val_loss"] == 0.1
