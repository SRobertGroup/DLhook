"""Tests for the landmark supervision added to the multiclass pipeline: target
rendering and loading (multi/src/landmarks.py), the augmentation-consistent
dataset, the optional model head, the loss, the readout, the inference path and
a real two-epoch training run. All on synthetic data: no weights, GPU or images
from the project are needed."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from models.UNetInference import MARGIN
from models.multiclass_inference import MulticlassInference
from models.unet import UNetGNRes, has_landmark_head, head_from_state_dict
from multi.src.data_loader import PatchDataset
from multi.src.landmarks import (
    Landmark, bio_from_theta, load_landmarks, pad_targets_to_min, readout_from_fields,
    render_targets, split_by_seedling,
)
from multi.src.loss_functions import landmark_loss
from multi.src.model import build_model
from multi.src.patch_index import pad_to_min

SIZE = 108                       # a valid patch size (572 - 16 * 29)
W, H = 60, 120                   # a narrow crop like the real ones


def _lm(junction=(30.5, 30.5), hypo=((0.0, 1.0), 50.0), cotyl=((1.0, 0.0), 25.0),
        name="0-crop-A_f1.png", seedling=("A", 0)):
    return Landmark(name, seedling, junction, hypo[0], cotyl[0], hypo[1], cotyl[1])


# --- targets ---------------------------------------------------------------------

def test_render_targets_peak_rays_and_validity():
    t = render_targets(H, W, _lm())
    assert np.unravel_index(np.argmax(t["hm"]), t["hm"].shape) == (30, 30)
    assert t["hm"].max() == pytest.approx(1.0, abs=1e-3)
    # hypocotyl runs down from the junction, cotyledon to the right
    assert t["paf_valid"][0][60, 30] == 1 and tuple(t["paf"][0:2, 60, 30]) == (0.0, 1.0)
    assert t["paf_valid"][1][30, 45] == 1 and tuple(t["paf"][2:4, 30, 45]) == (1.0, 0.0)
    # nothing is supervised off the rays: not above the junction, not far to the side
    assert t["paf_valid"][0][10, 30] == 0 and t["paf_valid"][0][60, 5] == 0
    assert t["paf_valid"][1][30, 5] == 0 and t["paf_valid"][1][90, 45] == 0
    norms = np.hypot(t["paf"][0], t["paf"][1])[t["paf_valid"][0] == 1]
    assert np.allclose(norms, 1.0)


def test_pad_targets_uses_the_same_offsets_as_the_image_pad_but_with_zeros():
    arr = np.ones((H, W), np.float32)
    padded = pad_targets_to_min(arr, SIZE)
    ref = pad_to_min(arr, SIZE)
    assert padded.shape == ref.shape == (H, SIZE)
    cols = np.where(padded[0] == 1)[0]
    assert cols.min() == (SIZE - W) // 2 and cols.max() == (SIZE - W) // 2 + W - 1   # zeros, not reflection
    assert pad_targets_to_min(np.ones((4, H, W)), SIZE).shape == (4, H, SIZE)


# --- loading ---------------------------------------------------------------------

LM_FIELDS = ["series", "crop_id", "frame", "crop_file", "status", "overhook",
             "junction_x", "junction_y", "hypo1_x", "hypo1_y", "hypo2_x", "hypo2_y",
             "cotyl1_x", "cotyl1_y", "cotyl2_x", "cotyl2_y"]


def _write_landmark_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LM_FIELDS)
        w.writeheader()
        for r in rows:
            j, (h1, h2), (c1, c2) = r.get("junction", (30.5, 30.5)), r.get("hypo", ((30.5, 50.5), (30.5, 80.5))), \
                r.get("cotyl", ((45.5, 30.5), (55.5, 30.5)))
            w.writerow({"series": r.get("series", "A"), "crop_id": r.get("crop_id", 0), "frame": r["frame"],
                        "crop_file": r["crop_file"], "status": r.get("status", "measured"),
                        "overhook": int(r.get("overhook", 0)),
                        "junction_x": j[0], "junction_y": j[1], "hypo1_x": h1[0], "hypo1_y": h1[1],
                        "hypo2_x": h2[0], "hypo2_y": h2[1], "cotyl1_x": c1[0], "cotyl1_y": c1[1],
                        "cotyl2_x": c2[0], "cotyl2_y": c2[1]})


def test_load_landmarks_orients_rays_and_guards_the_validation_split(tmp_path):
    path = tmp_path / "gt.csv"
    _write_landmark_csv(path, [
        {"frame": "f1", "crop_file": "0-crop-A_f1.png", "overhook": 1},
        {"frame": "f2", "crop_file": "0-crop-A_f2.png", "hypo": ((30.5, 80.5), (30.5, 50.5))},   # click order reversed
        {"frame": "f3", "crop_file": "0-crop-A_f3.png"},                                          # in the val split
        {"frame": "f4", "crop_file": "0-crop-A_f4.png", "status": "skipped"},
        {"frame": "f5", "crop_file": "0-crop-A_f5.png", "hypo": ((30.5, 20.5), (30.5, 40.5))},    # straddles the junction
    ])
    landmarks, report = load_landmarks(str(path), exclude_filenames={"0-crop-A_f3.png"})
    assert report == {"loaded": 2, "excluded": 1, "invalid": 1}
    assert set(landmarks) == {"0-crop-A_f1.png", "0-crop-A_f2.png"}
    for lm in landmarks.values():                                   # away from the junction whatever the click order
        assert lm.hypo_dir == pytest.approx((0.0, 1.0)) and lm.cotyl_dir == pytest.approx((1.0, 0.0))
        assert lm.hypo_len == pytest.approx(50.0) and lm.cotyl_len == pytest.approx(25.0)
        assert lm.theta == pytest.approx(90.0)
    assert not hasattr(landmarks["0-crop-A_f1.png"], "overhook")             # the flag in the CSV is ignored


def test_split_by_seedling_keeps_all_frames_of_a_seedling_together():
    lms = {f"{c}-crop-A_{f}.png": _lm(name=f"{c}-crop-A_{f}.png", seedling=("A", c))
           for c in range(10) for f in range(3)}
    train, val = split_by_seedling(lms, 0.2, seed=1)
    assert len(train) + len(val) == 30 and val and train
    assert {lm.seedling for lm in train.values()}.isdisjoint({lm.seedling for lm in val.values()})
    assert (train, val) == split_by_seedling(lms, 0.2, seed=1)         # deterministic
    only_one = split_by_seedling({"x": _lm()}, 0.5, seed=0)
    assert len(only_one[0]) == 1 and not only_one[1]


# --- readout ---------------------------------------------------------------------

def _fields(theta_deg, hm_peak=0.9, size=40, junction=(20, 15)):
    """A constant-direction field pair at angle theta apart, with a junction peak."""
    hm = np.zeros((size, size), np.float32)
    hm[junction[1], junction[0]] = hm_peak
    a = math.radians(theta_deg)
    vec = np.zeros((4, size, size), np.float32)
    vec[0], vec[1] = 0.0, 1.0                      # hypocotyl straight down
    vec[2], vec[3] = math.sin(a), math.cos(a)      # cotyledon theta away from it
    return hm, vec


def test_readout_recovers_theta_and_the_bio_convention():
    hm, vec = _fields(35.0)
    r = readout_from_fields(hm, vec)
    assert r["theta"] == pytest.approx(35.0, abs=1e-3) and r["bio"] == pytest.approx(145.0, abs=1e-3)
    assert r["junction"] == pytest.approx((20.5, 15.5)) and "overhook" not in r and "overhook_prob" not in r
    assert r["bio"] == pytest.approx(180.0 - r["theta"]) and r["bio"] <= 180.0
    assert bio_from_theta(0.0) == 180.0 and bio_from_theta(90.0) == 90.0


def test_readout_returns_none_without_a_junction_or_a_direction():
    hm, vec = _fields(35.0, hm_peak=0.05)
    assert readout_from_fields(hm, vec) is None
    hm, vec = _fields(35.0)
    vec[:] = 0.0
    assert readout_from_fields(hm, vec) is None


def test_rendered_targets_read_back_to_the_annotated_angle():
    for cotyl_dir in ((1.0, 0.0), (math.sin(0.4), math.cos(0.4)), (0.0, 1.0)):
        lm = _lm(cotyl=(cotyl_dir, 25.0))
        t = render_targets(H, W, lm)
        r = readout_from_fields(t["hm"], t["paf"])
        assert r["theta"] == pytest.approx(lm.theta, abs=1.0)
        assert r["junction"] == pytest.approx(lm.junction, abs=1.0)
        assert r["bio"] == pytest.approx(bio_from_theta(lm.theta), abs=1.0)


# --- dataset ---------------------------------------------------------------------

def _make_dataset(root, landmarks, row_y=0, augment_cfg=None, augment=False, image_h=H):
    raw_dir, masks_dir = root / "raw", root / "masks"
    raw_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    rng = np.random.RandomState(0)
    Image.fromarray((rng.rand(image_h, W, 3) * 255).astype(np.uint8)).save(raw_dir / "img0.png")
    Image.fromarray(np.zeros((image_h, W), np.uint8)).save(masks_dir / "img0.png")
    csv_path = root / "patches.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["filename", "x", "y", "patch_size", "foreground_fraction"])
        w.writeheader()
        w.writerow({"filename": "img0.png", "x": 0, "y": row_y, "patch_size": SIZE, "foreground_fraction": 0.0})
    return PatchDataset(csv_path, raw_dir, masks_dir, augment=augment, augmentation_cfg=augment_cfg or {},
                        landmarks=landmarks, seed=0)


def test_dataset_without_landmarks_still_returns_image_and_label(tmp_path):
    item = _make_dataset(tmp_path, None)[0]
    assert len(item) == 2


def test_dataset_targets_follow_the_mask_grid(tmp_path):
    ds = _make_dataset(tmp_path, {"img0.png": _lm()})
    image, label, kp = ds[0]
    pad_left = (SIZE - W) // 2
    assert label.shape == (SIZE, SIZE) and image.shape[-1] == SIZE + 2 * MARGIN
    assert kp["hm"].shape == (SIZE, SIZE) and kp["paf"].shape == (4, SIZE, SIZE) and kp["paf_valid"].shape == (2, SIZE, SIZE)
    assert np.unravel_index(int(kp["hm"].argmax()), kp["hm"].shape) == (30, 30 + pad_left)
    assert float(kp["has_kp"]) == 1.0 and "overhook" not in kp


def test_dataset_crop_without_annotation_gives_empty_targets(tmp_path):
    kp = _make_dataset(tmp_path, {"other.png": _lm()})[0][2]
    assert float(kp["has_kp"]) == 0.0 and float(kp["hm"].max()) == 0.0 and float(kp["paf_valid"].max()) == 0.0


def test_patch_that_misses_the_junction_has_no_landmark_terms(tmp_path):
    ds = _make_dataset(tmp_path, {"img0.png": _lm()}, row_y=100, image_h=220)   # patch rows 100..207
    kp = ds[0][2]
    assert float(kp["has_kp"]) == 0.0


def test_horizontal_flip_mirrors_the_heatmap_and_negates_x_vectors(tmp_path):
    ds = _make_dataset(tmp_path, {"img0.png": _lm()}, augment=True, augment_cfg={"horizontal_flip": True})
    ds.rng.random = lambda: 0.0                                             # always flip
    kp = ds[0][2]
    pad_left = (SIZE - W) // 2
    assert np.unravel_index(int(kp["hm"].argmax()), kp["hm"].shape) == (30, SIZE - 1 - (30 + pad_left))
    cotyl_valid = np.argwhere(kp["paf_valid"][1].numpy() == 1)
    y, x = cotyl_valid[0]
    assert tuple(kp["paf"][2:4, y, x].numpy()) == (-1.0, 0.0)               # cotyledon now points left
    y, x = np.argwhere(kp["paf_valid"][0].numpy() == 1)[0]
    assert tuple(kp["paf"][0:2, y, x].numpy()) == (0.0, 1.0)                # straight down is unchanged


def test_rotation_moves_the_junction_and_rotates_the_vectors_like_the_mask(tmp_path):
    ds = _make_dataset(tmp_path, {"img0.png": _lm()}, augment=True, augment_cfg={"rotation_degrees": 15})
    ds.rng.uniform = lambda a, b: 10.0
    kp = ds[0][2]
    pad_left = (SIZE - W) // 2
    matrix = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), 10.0, 1.0)
    expected = matrix @ np.array([30 + pad_left, 30, 1.0])
    row, col = np.unravel_index(int(kp["hm"].argmax()), kp["hm"].shape)
    assert (col, row) == pytest.approx(tuple(expected), abs=1.01)
    expected_dir = matrix[:, :2] @ np.array([1.0, 0.0])                     # cotyledon (1, 0) rotated
    y, x = np.argwhere(kp["paf_valid"][1].numpy() == 1)[0]
    assert tuple(kp["paf"][2:4, y, x].numpy()) == pytest.approx(tuple(expected_dir), abs=1e-4)
    assert np.hypot(*kp["paf"][2:4, y, x].numpy()) == pytest.approx(1.0, abs=1e-4)


# --- model -----------------------------------------------------------------------

def test_landmark_head_is_optional_and_leaves_forward_untouched():
    torch.manual_seed(0)
    plain = UNetGNRes(n_classes=4).eval()
    with_head = UNetGNRes(n_classes=4, landmark_head=True).eval()
    missing, unexpected = with_head.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("conv_kp.") for k in missing)

    x = torch.randn(1, 3, SIZE + 2 * MARGIN, SIZE + 2 * MARGIN)
    with torch.no_grad():
        seg, kp = with_head.forward_with_landmarks(x)
        assert torch.equal(plain(x), with_head(x)) and torch.equal(seg, plain(x))
    assert kp.shape[1] == 5 and kp.shape[-2:] == seg.shape[-2:]
    with pytest.raises(RuntimeError):
        plain.forward_with_landmarks(x)


def test_landmark_keys_are_detected_without_disturbing_head_detection():
    plain, with_head = UNetGNRes(n_classes=4), UNetGNRes(n_classes=4, landmark_head=True)
    assert not has_landmark_head(plain.state_dict()) and has_landmark_head(with_head.state_dict())
    prefixed = {"module." + k: v for k, v in with_head.state_dict().items()}
    assert has_landmark_head(prefixed)
    assert head_from_state_dict(with_head.state_dict()) == head_from_state_dict(plain.state_dict())


def test_warm_start_can_keep_the_trained_segmentation_head(tmp_path):
    torch.manual_seed(0)
    trained = UNetGNRes(n_classes=4)
    ckpt = tmp_path / "trained.pt"
    torch.save(trained.state_dict(), ckpt)
    cfg = {"num_classes": 4, "landmark_head": True, "warm_start_from": str(ckpt)}

    torch.manual_seed(1)
    reset = build_model(dict(cfg))
    kept = build_model(dict(cfg, warm_start_keep_head=True))
    assert not torch.equal(reset.conv_out[0].weight, trained.conv_out[0].weight)     # default: head re-initialised
    assert torch.equal(kept.conv_out[0].weight, trained.conv_out[0].weight)          # opt-in: head kept
    assert torch.equal(kept.conv_in[0].weight, trained.conv_in[0].weight) and kept.landmark_head


# --- loss ------------------------------------------------------------------------

def _loss_inputs(n=2, size=16, has=(1.0, 1.0)):
    torch.manual_seed(0)
    hm = torch.zeros(n, size, size)
    hm[:, 8, 8] = 1.0
    paf = torch.zeros(n, 4, size, size)
    paf[:, 1, 8, :] = 1.0
    valid = torch.zeros(n, 2, size, size)
    valid[:, 0, 8, :] = 1.0
    targets = {"hm": hm, "paf": paf, "paf_valid": valid,
               "has_kp": torch.tensor(list(has))}
    return torch.randn(n, 5, size, size, requires_grad=True), targets


def test_landmark_loss_ignores_samples_without_a_junction():
    kp, targets = _loss_inputs(has=(1.0, 0.0))
    both, _ = landmark_loss(kp, targets)
    first_only, _ = landmark_loss(kp[:1], {k: v[:1] for k, v in targets.items()})
    assert float(both) == pytest.approx(float(first_only), rel=1e-5)

    kp0, targets0 = _loss_inputs(has=(0.0, 0.0))
    zero, parts = landmark_loss(kp0, targets0)
    assert float(zero) == 0.0 and parts == {"hm": 0.0, "paf": 0.0}
    zero.backward()                                                          # harmless: carries a graph


def test_landmark_loss_has_gradients_and_rewards_correct_predictions():
    kp, targets = _loss_inputs()
    wrong, parts = landmark_loss(kp, targets)
    wrong.backward()
    assert float(kp.grad.abs().sum()) > 0 and torch.isfinite(kp.grad).all()
    assert set(parts) == {"hm", "paf"}

    good_kp = torch.full((2, 5, 16, 16), -8.0)
    good_kp[:, 0, 8, 8] = 8.0                                               # junction peak
    good_kp[:, 1:5] = targets["paf"]
    good, _ = landmark_loss(good_kp, targets)
    assert float(good) < 0.05 * float(wrong)


# --- inference -------------------------------------------------------------------

def _checkpoint(tmp_path, landmark_head):
    torch.manual_seed(0)
    path = tmp_path / ("lm.pt" if landmark_head else "plain.pt")
    torch.save(UNetGNRes(n_classes=4, landmark_head=landmark_head).state_dict(), path)
    return path


def _crop():
    return (np.random.RandomState(0).rand(H, W, 3) * 255).astype(np.uint8)


def test_inference_reads_landmarks_and_keeps_the_segmentation_path(tmp_path):
    kwargs = dict(num_classes=4, in_size=SIZE + 2 * MARGIN, out_size=SIZE, margin=MARGIN, device=torch.device("cpu"))
    with_head = MulticlassInference(_checkpoint(tmp_path, True), **kwargs)
    plain = MulticlassInference(_checkpoint(tmp_path, False), **kwargs)
    assert with_head.has_landmarks and not plain.has_landmarks

    result = with_head.predict_landmarks([_crop(), _crop()])
    assert len(result) == 2 and all(r is None or {"theta", "bio", "junction"} <= set(r) for r in result)
    assert with_head.segment_many_argmax([_crop()])[0].shape == (H, W)       # segmentation unchanged
    with pytest.raises(RuntimeError):
        plain.predict_landmarks([_crop()])


def test_landmark_head_does_not_change_the_segmentation_output(tmp_path):
    torch.manual_seed(0)
    plain = UNetGNRes(n_classes=4)
    with_head = UNetGNRes(n_classes=4, landmark_head=True)
    with_head.load_state_dict(plain.state_dict(), strict=False)
    paths = {}
    for name, model in (("plain", plain), ("lm", with_head)):
        paths[name] = tmp_path / f"{name}.pt"
        torch.save(model.state_dict(), paths[name])
    kwargs = dict(num_classes=4, in_size=SIZE + 2 * MARGIN, out_size=SIZE, margin=MARGIN, device=torch.device("cpu"))
    a = MulticlassInference(paths["plain"], **kwargs).segment_many_argmax([_crop()], return_probs=True)[0]
    b = MulticlassInference(paths["lm"], **kwargs).segment_many_argmax([_crop()], return_probs=True)[0]
    assert np.array_equal(a, b)


# --- a real (tiny) training run ----------------------------------------------------

def _training_config(root):
    patch_index = root / "patch_index"
    patch_index.mkdir()
    for split, names in (("train", ["a.png", "b.png"]), ("val", ["v.png"])):
        with open(patch_index / f"{split}_patches.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["filename", "x", "y", "patch_size", "foreground_fraction"])
            for n in names:
                w.writerow([n, 0, 0, SIZE, 0.0])
    (root / "raw").mkdir()
    (root / "masks").mkdir()
    rng = np.random.RandomState(0)
    for n in ("a.png", "b.png", "v.png"):
        Image.fromarray((rng.rand(H, W, 3) * 255).astype(np.uint8)).save(root / "raw" / n)
        Image.fromarray(np.zeros((H, W), np.uint8)).save(root / "masks" / n)
    _write_landmark_csv(root / "landmarks.csv", [
        {"frame": "a", "crop_file": "a.png", "series": "S", "crop_id": 0},
        {"frame": "b", "crop_file": "b.png", "series": "S", "crop_id": 1},
        {"frame": "v", "crop_file": "v.png", "series": "S", "crop_id": 2},        # validation split: must be dropped
    ])
    return {
        "_data_root": str(root),
        "paths": {"raw_data_dir": "raw", "multiclass_masks_dir": "masks", "patch_index_dir": "patch_index",
                  "checkpoints_dir": "ckpt", "logs_dir": "logs"},
        "data": {"num_classes": 4, "class_names": ["background", "cotyledon", "hypocotyl", "radicle"]},
        "model": {"num_classes": 4, "in_channels": 3, "landmark_head": True},
        "loss": {"type": "cross_entropy", "ignore_index": 255},
        "augmentation": {"horizontal_flip": True, "rotation_degrees": 10},
        "landmarks": {"enabled": True, "csv": "landmarks.csv", "val_fraction": 0.5, "oversample": 3.0,
                      "min_mean_fg_dice": 0.0, "loss_weights": {"hm": 1.0, "paf": 1.0}},
        "training": {"batch_size": 2, "num_epochs": 2, "lr_max": 1e-3, "device": "cpu", "num_workers": 0,
                     "seed": 0, "early_stopping_patience": 100, "selection_metric": "landmark_angle_mae"},
    }


def test_training_run_with_landmarks_end_to_end(tmp_path, capsys):
    import multi.train_unet_multiclass as trainer
    history = trainer.run_training(_training_config(tmp_path))
    out = capsys.readouterr().out
    assert "1 in the validation split" in out                                   # the leakage guard fired
    assert history["selection_metric"] == "landmark_angle_mae"
    for key in ("landmark_angle_mae", "junction_px_err"):
        assert len(history[key]) == 2
    assert math.isfinite(history["landmark_angle_mae"][-1])

    state = torch.load(history["checkpoint_path"], map_location="cpu", weights_only=True)
    assert has_landmark_head(state)
    log = open(history["log_path"], newline="", encoding="utf-8").read().splitlines()[0]
    assert "landmark_angle_mae" in log


def test_selection_by_landmark_error_requires_the_landmarks_section(tmp_path):
    import multi.train_unet_multiclass as trainer
    config = _training_config(tmp_path)
    config["landmarks"]["enabled"] = False
    with pytest.raises(ValueError, match="requires landmarks.enabled"):
        trainer.run_training(config)


def test_landmarks_need_the_model_head(tmp_path):
    import multi.train_unet_multiclass as trainer
    config = _training_config(tmp_path)
    config["model"]["landmark_head"] = False
    with pytest.raises(ValueError, match="landmark_head"):
        trainer.run_training(config)


def test_junction_channel_starts_at_a_low_prior():
    torch.manual_seed(0)
    model = UNetGNRes(n_classes=4, landmark_head=True).eval()
    x = torch.randn(1, 3, SIZE + 2 * MARGIN, SIZE + 2 * MARGIN)
    with torch.no_grad():
        _, kp = model.forward_with_landmarks(x)
    assert float(torch.sigmoid(kp[:, 0]).mean()) < 0.1     # not 0.5: background must not swamp the peak


def test_head_only_epochs_leave_the_encoder_untouched(tmp_path):
    import multi.train_unet_multiclass as trainer
    config = _training_config(tmp_path)
    config["landmarks"]["head_only_epochs"] = 2            # every epoch of this 2-epoch run
    torch.manual_seed(0)
    initial = build_model(dict(config["model"]))
    history = trainer.run_training(config)
    # the final epoch's weights (best.pt may be epoch 0, whose one batch can miss every annotated patch)
    last = torch.load(Path(history["checkpoint_path"]).parent / "last.pt", map_location="cpu", weights_only=True)
    state = last["model_state_dict"]
    assert torch.equal(state["conv_in.0.weight"], initial.state_dict()["conv_in.0.weight"])      # encoder frozen
    assert not torch.equal(state["conv_kp.weight"], initial.state_dict()["conv_kp.weight"])      # head trained


def test_landmark_trunk_roundtrips_through_inference(tmp_path):
    import torch
    from models.unet import UNetGNRes, has_landmark_head, landmark_hidden_from_state_dict

    model = UNetGNRes(n_classes=4, head="plain", landmark_head=True, landmark_hidden=64)
    state = model.state_dict()
    assert has_landmark_head(state) and landmark_hidden_from_state_dict(state) == 64
    assert landmark_hidden_from_state_dict(UNetGNRes(n_classes=4, landmark_head=True).state_dict()) == 0
    seg, kp = model.forward_with_landmarks(torch.zeros(1, 3, 64, 64))
    assert kp.shape[1] == 5
    assert torch.equal(seg, model(torch.zeros(1, 3, 64, 64)))


# --- collar and root direction -----------------------------------------------------

def _lm_root(**kw):
    from dataclasses import replace
    return replace(_lm(**kw), collar=(30.5, 100.5), root_dir=(0.0, 1.0), root_len=15.0)


def test_root_targets_peak_at_the_collar_and_run_along_the_root():
    t = render_targets(H, W, _lm_root())
    assert np.unravel_index(np.argmax(t["collar_hm"]), t["collar_hm"].shape) == (100, 30)
    assert t["root_valid"][0][108, 30] == 1 and tuple(t["root_vec"][:, 108, 30]) == (0.0, 1.0)
    assert t["root_valid"][0][90, 30] == 0 and t["root_valid"][0][108, 5] == 0      # nothing above the collar or aside
    empty = render_targets(H, W, _lm())                                             # no radicle annotated
    assert empty["collar_hm"].max() == 0 and empty["root_valid"].max() == 0


def test_root_csv_attaches_the_collar_only_to_visible_radicle_frames(tmp_path):
    gt = tmp_path / "gt.csv"
    _write_landmark_csv(gt, [{"frame": "f1", "crop_file": "0-crop-A_f1.png"},
                             {"frame": "f2", "crop_file": "0-crop-A_f2.png"},
                             {"frame": "f3", "crop_file": "0-crop-A_f3.png"}])
    root = tmp_path / "root.csv"
    with open(root, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["crop_file", "status", "radicle_visible", "collar_x", "collar_y",
                                           "root_x", "root_y"])
        w.writeheader()
        w.writerow({"crop_file": "0-crop-A_f1.png", "status": "annotated", "radicle_visible": "1",
                    "collar_x": 30, "collar_y": 100, "root_x": 30, "root_y": 120})
        w.writerow({"crop_file": "0-crop-A_f2.png", "status": "annotated", "radicle_visible": "0"})
        w.writerow({"crop_file": "0-crop-A_f3.png", "status": "skipped", "radicle_visible": ""})
    landmarks, report = load_landmarks(str(gt), root_csv_path=str(root))
    assert report["with_root"] == 1 and report["loaded"] == 3
    lm = landmarks["0-crop-A_f1.png"]
    assert lm.collar == (30.0, 100.0) and lm.root_dir == pytest.approx((0.0, 1.0)) and lm.root_len == pytest.approx(20.0)
    assert landmarks["0-crop-A_f2.png"].collar is None and landmarks["0-crop-A_f3.png"].collar is None


def test_dataset_carries_and_flips_the_root_targets(tmp_path):
    ds = _make_dataset(tmp_path, {"img0.png": _lm_root(hypo=((0.0, 1.0), 30.0))}, augment=True,
                       augment_cfg={"horizontal_flip": True})
    ds.rng.random = lambda: 0.0                                                      # always flip
    kp = ds[0][2]
    pad_left = (SIZE - W) // 2
    assert float(kp["has_collar"]) == 1.0
    assert np.unravel_index(int(kp["collar_hm"].argmax()), kp["collar_hm"].shape) == (100, SIZE - 1 - (30 + pad_left))
    assert tuple(kp["root_vec"][:, 104, SIZE - 1 - (30 + pad_left)].numpy()) == (0.0, 1.0)   # straight down: unchanged


def test_rotation_rotates_the_root_vectors(tmp_path):
    ds = _make_dataset(tmp_path, {"img0.png": _lm_root(hypo=((0.0, 1.0), 30.0))}, augment=True,
                       augment_cfg={"rotation_degrees": 15})
    ds.rng.uniform = lambda a, b: 10.0
    kp = ds[0][2]
    matrix = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), 10.0, 1.0)
    expected = matrix[:, :2] @ np.array([0.0, 1.0])
    y, x = np.argwhere(kp["root_valid"][0].numpy() == 1)[0]
    assert tuple(kp["root_vec"][:, y, x].numpy()) == pytest.approx(tuple(expected), abs=1e-4)


def test_model_with_a_root_head_has_eight_channels_and_roundtrips(tmp_path):
    from models.unet import landmark_root_from_state_dict
    model = UNetGNRes(n_classes=4, head="plain", landmark_head=True, landmark_hidden=64, landmark_root=True)
    seg, kp = model.forward_with_landmarks(torch.zeros(1, 3, 64, 64))
    assert kp.shape[1] == 8
    assert torch.sigmoid(model.conv_kp.bias[5]) < 0.02 and torch.sigmoid(model.conv_kp.bias[0]) < 0.02
    assert landmark_root_from_state_dict(model.state_dict())
    assert not landmark_root_from_state_dict(UNetGNRes(n_classes=4, landmark_head=True).state_dict())
    ckpt = tmp_path / "root.pt"
    torch.save(model.state_dict(), ckpt)
    inference = MulticlassInference(str(ckpt), num_classes=4, in_size=252 + 2 * MARGIN, out_size=252, margin=MARGIN)
    image = (np.random.RandomState(0).rand(H, W, 3) * 255).astype(np.uint8)
    hm, vec, extra = inference.predict_landmark_fields([image])[0]
    assert extra["collar_hm"].shape == hm.shape == (H, W) and extra["root_vec"].shape == (2, H, W)


def test_readout_adds_the_collar_and_root_direction_without_changing_the_angle():
    hm, vec = _fields(60.0)
    collar_hm = np.zeros_like(hm)
    collar_hm[30, 10] = 0.9
    root_vec = np.zeros((2,) + hm.shape, np.float32)
    root_vec[1] = 1.0
    plain = readout_from_fields(hm, vec)
    with_root = readout_from_fields(hm, vec, collar_hm=collar_hm, root_vec=root_vec)
    assert with_root["collar"] == (10.5, 30.5) and with_root["root_dir"] == pytest.approx((0.0, 1.0))
    assert with_root["bio"] == plain["bio"] and "collar" not in plain
    none = readout_from_fields(hm, vec, collar_hm=np.zeros_like(hm), root_vec=root_vec)
    assert none["collar"] is None and none["root_dir"] is None


def test_root_loss_uses_only_samples_with_a_collar():
    kp, targets = _loss_inputs(has=(1.0, 1.0))
    n, size = kp.shape[0], kp.shape[-1]
    kp8 = torch.randn(n, 8, size, size, requires_grad=True)
    collar = torch.zeros(n, size, size)
    collar[:, 4, 4] = 1.0
    root = torch.zeros(n, 2, size, size)
    root[:, 1, 4:, 4] = 1.0
    valid = torch.zeros(n, 1, size, size)
    valid[:, 0, 4:, 4] = 1.0
    t8 = {**targets, "collar_hm": collar, "root_vec": root, "root_valid": valid, "has_collar": torch.tensor([1.0, 0.0])}
    total, parts = landmark_loss(kp8, t8)
    assert "collar" in parts and "root" in parts and parts["collar"] > 0
    total.backward()
    assert kp8.grad[:, 5:8].abs().sum() > 0
    # the collar/root terms come from the sample that has a collar only
    _, parts_first = landmark_loss(kp8[:1], {k: v[:1] for k, v in t8.items()})
    assert parts["collar"] == pytest.approx(parts_first["collar"], rel=1e-5)


# --- whole-crop validation / checkpoint selection ----------------------------------

def test_inference_can_wrap_a_live_model_and_matches_the_checkpoint_path(tmp_path):
    torch.manual_seed(0)
    model = UNetGNRes(n_classes=4, head="plain", landmark_head=True, landmark_hidden=64, landmark_root=True)
    model.eval()
    ckpt = tmp_path / "m.pt"
    torch.save(model.state_dict(), ckpt)
    geometry = dict(in_size=108 + 2 * MARGIN, out_size=108, margin=MARGIN)
    image = (np.random.RandomState(1).rand(H, W, 3) * 255).astype(np.uint8)
    loaded = MulticlassInference(str(ckpt), num_classes=4, device=torch.device("cpu"), **geometry)
    live = MulticlassInference.from_model(model, device=torch.device("cpu"), **geometry)
    a = loaded.predict_landmark_fields([image])[0]
    b = live.predict_landmark_fields([image])[0]
    assert np.allclose(a[0], b[0], atol=1e-5) and np.allclose(a[1], b[1], atol=1e-5)
    assert np.allclose(a[2]["collar_hm"], b[2]["collar_hm"], atol=1e-5)


def test_training_can_select_on_the_whole_crop_angle_error(tmp_path, capsys):
    import multi.train_unet_multiclass as trainer
    config = _training_config(tmp_path)
    root_csv = tmp_path / "root.csv"
    with open(root_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["crop_file", "status", "radicle_visible", "collar_x", "collar_y",
                                           "root_x", "root_y"])
        w.writeheader()
        for name in ("a.png", "b.png"):
            w.writerow({"crop_file": name, "status": "annotated", "radicle_visible": "1",
                        "collar_x": 30, "collar_y": 100, "root_x": 30, "root_y": 115})
    config["model"].update(landmark_root=True, landmark_hidden=64)
    config["landmarks"]["root_csv"] = "root.csv"
    config["training"]["selection_metric"] = "wholecrop_theta_mae"
    history = trainer.run_training(config)
    out = capsys.readouterr().out
    assert "whole-crop theta=" in out and "collar=" in out
    for key in ("wholecrop_theta_mae", "wholecrop_theta_median", "wholecrop_junction_px",
                "wholecrop_collar_px", "wholecrop_root_deg"):
        assert len(history[key]) == 2
    assert math.isfinite(history["wholecrop_theta_mae"][-1])
    assert history["selection_metric"] == "wholecrop_theta_mae"
    header = open(history["log_path"], newline="", encoding="utf-8").read().splitlines()[0]
    assert "wholecrop_theta_mae" in header and "wholecrop_root_deg" in header


def test_checkpoints_trained_with_the_overhook_head_still_load(tmp_path):
    from models.unet import strip_deprecated_landmark_keys
    model = UNetGNRes(n_classes=4, head="plain", landmark_head=True, landmark_hidden=64)
    state = model.state_dict()
    old = dict(state)
    old["cls_overhook.weight"] = torch.zeros(1, 64)
    old["cls_overhook.bias"] = torch.zeros(1)
    assert not any(k.startswith("cls_overhook") for k in strip_deprecated_landmark_keys(old))
    assert set(strip_deprecated_landmark_keys({"module." + k: v for k, v in old.items()})) == {"module." + k for k in state}
    ckpt = tmp_path / "old.pt"
    torch.save(old, ckpt)
    inference = MulticlassInference(str(ckpt), num_classes=4, device=torch.device("cpu"),
                                    in_size=108 + 2 * MARGIN, out_size=108, margin=MARGIN)
    assert inference.has_landmarks
    image = (np.random.RandomState(0).rand(H, W, 3) * 255).astype(np.uint8)
    hm, vec, extra = inference.predict_landmark_fields([image])[0]
    assert hm.shape == (H, W) and extra is None


def test_flip_tta_is_mirror_consistent_and_can_be_switched_off(tmp_path):
    torch.manual_seed(0)
    model = UNetGNRes(n_classes=4, head="plain", landmark_head=True, landmark_hidden=64, landmark_root=True)
    ckpt = tmp_path / "m.pt"
    torch.save(model.state_dict(), ckpt)
    inf = MulticlassInference(str(ckpt), num_classes=4, device=torch.device("cpu"),
                              in_size=108 + 2 * MARGIN, out_size=108, margin=MARGIN)
    image = (np.random.RandomState(3).rand(H, W, 3) * 255).astype(np.uint8)
    mirror = np.ascontiguousarray(image[:, ::-1])
    hm, vec, extra = inf.predict_landmark_fields([image])[0]
    hm_m, vec_m, extra_m = inf.predict_landmark_fields([mirror])[0]
    assert np.allclose(hm, hm_m[:, ::-1], atol=1e-5)                       # averaging makes it exactly symmetric
    assert np.allclose(vec[0], -vec_m[0][:, ::-1], atol=1e-5) and np.allclose(vec[1], vec_m[1][:, ::-1], atol=1e-5)
    assert np.allclose(extra["root_vec"][0], -extra_m["root_vec"][0][:, ::-1], atol=1e-5)
    single = inf.predict_landmark_fields([image], flip_tta=False)[0][0]
    assert not np.allclose(single, hm)


def test_trainable_extra_lets_the_last_decoder_stage_adapt(tmp_path):
    import multi.train_unet_multiclass as trainer
    config = _training_config(tmp_path)
    config["landmarks"]["head_only_epochs"] = 2
    config["landmarks"]["trainable_extra"] = ["up4."]
    torch.manual_seed(0)
    initial = build_model(dict(config["model"])).state_dict()
    history = trainer.run_training(config)
    state = torch.load(Path(history["checkpoint_path"]).parent / "last.pt", map_location="cpu",
                       weights_only=True)["model_state_dict"]
    assert torch.equal(state["conv_in.0.weight"], initial["conv_in.0.weight"])        # still frozen
    up4 = [k for k in state if k.startswith("up4.") and k.endswith("weight")]
    assert any(not torch.equal(state[k], initial[k]) for k in up4)                      # adapted


# --- extra landmark-only sources (other crop folders, a former test set) ----------------

def test_namespaced_landmarks_keep_folders_apart():
    from multi.src.landmarks import namespace_landmarks
    lm = _lm()
    out = namespace_landmarks({lm.filename: lm}, "new")
    (key, nlm), = out.items()
    assert key == f"new/{lm.filename}" and nlm.filename == key
    assert nlm.seedling == (f"new/{lm.seedling[0]}", lm.seedling[1]) and nlm.junction == lm.junction


def test_extra_rows_read_another_folder_with_an_all_ignore_mask(tmp_path):
    from multi.src.data_loader import PatchDataset
    from multi.src.landmarks import landmark_patch_rows, namespace_landmarks
    other = tmp_path / "other"
    other.mkdir()
    Image.fromarray(np.full((H, W, 3), 50, np.uint8)).save(other / "a.png")
    index = tmp_path / "train.csv"
    index.write_text("filename,x,y,patch_size,foreground_fraction\n", encoding="utf-8")
    rows = landmark_patch_rows({"new/a.png": (H, W)}, SIZE, SIZE, {"new/a.png": other / "a.png"})
    assert rows and all(r["no_mask"] == "1" for r in rows)
    lm = namespace_landmarks({"a.png": _lm(name="a.png")}, "new")
    ds = PatchDataset(index, tmp_path / "raw", tmp_path / "masks", landmarks=lm, extra_rows=rows)
    raw, mask, kp = ds[0]
    assert (mask == 255).all() and float(kp["has_kp"]) == 1.0


def test_training_with_an_extra_source_that_reuses_crop_names(tmp_path, capsys):
    import multi.train_unet_multiclass as trainer
    config = _training_config(tmp_path)
    other = tmp_path / "new_set"
    other.mkdir()
    rng = np.random.RandomState(1)
    for n in ("a.png", "c.png"):                     # a.png repeats a training crop name
        Image.fromarray((rng.rand(H, W, 3) * 255).astype(np.uint8)).save(other / n)
    _write_landmark_csv(tmp_path / "new.csv", [
        {"frame": "a", "crop_file": "a.png", "series": "S", "crop_id": 0},
        {"frame": "c", "crop_file": "c.png", "series": "S", "crop_id": 5},
    ])
    config["data"]["patch_size"] = SIZE
    config["landmarks"]["extra_sources"] = [{"csv": "new.csv", "raw_dir": "new_set"}]
    with pytest.raises(ValueError, match="give this source a tag"):
        trainer.run_training(config)
    config["landmarks"]["extra_sources"][0]["tag"] = "new"
    history = trainer.run_training(config)
    assert "Extra landmarks new.csv: 2 loaded as new/..." in capsys.readouterr().out
    assert len(history["landmark_angle_mae"]) == 2


def test_seedling_groups_keep_one_plant_on_one_side_of_the_split(tmp_path, capsys):
    import json
    import multi.train_unet_multiclass as trainer
    config = _training_config(tmp_path)
    config["data"]["patch_size"] = SIZE
    other = tmp_path / "open_set"
    other.mkdir()
    Image.fromarray((np.random.RandomState(2).rand(H, W, 3) * 255).astype(np.uint8)).save(other / "c.png")
    _write_landmark_csv(tmp_path / "open.csv", [{"frame": "c", "crop_file": "c.png", "series": "S", "crop_id": 1}])
    (tmp_path / "groups.json").write_text(json.dumps({"S:1": "S:1"}), encoding="utf-8")     # the plant of training crop b.png
    src = {"csv": "open.csv", "raw_dir": "open_set", "tag": "open", "seedling_groups": "groups.json"}
    lms, _, _ = trainer.load_extra_landmark_sources(config, [src], {}, SIZE, SIZE)
    assert lms["open/c.png"].seedling == ("S", 1)
    from multi.src.landmarks import load_landmarks, split_by_seedling
    base, _ = load_landmarks(tmp_path / "landmarks.csv", exclude_filenames={"v.png"})
    for seed in range(8):
        train, val = split_by_seedling({**base, **lms}, 0.5, seed)
        assert ("b.png" in train) == ("open/c.png" in train)


def test_match_seedlings_finds_the_same_plant_by_box(tmp_path):
    import json
    from multi import match_seedlings as ms
    def manifest(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["series", "crop_id", "x1", "y1", "x2", "y2"])
            w.writerows(rows)
    manifest(tmp_path / "ref.csv", [["S", 0, 0, 0, 100, 200], ["S", 1, 300, 0, 400, 200], ["T", 0, 0, 0, 100, 200]])
    manifest(tmp_path / "new.csv", [["S", 5, 10, 0, 110, 150], ["S", 6, 600, 0, 700, 200], ["U", 0, 0, 0, 100, 200]])
    new, ref = ms.manifest_boxes(tmp_path / "new.csv"), ms.manifest_boxes(tmp_path / "ref.csv")
    matches = ms.match_seedlings(new, [("old", ref)])
    assert list(matches) == [("S", 5)] and matches[("S", 5)][:3] == ("old", "S", 0)      # other series / far boxes: new plants
    assert ms.groups_json(matches) == {"S:5": "old/S:0"}
    assert ms.groups_json(ms.match_seedlings(new, [("", ref)])) == {"S:5": "S:0"}
    out = tmp_path / "g.json"
    assert ms.main(["--new", str(tmp_path / "new.csv"), "--ref", f"old={tmp_path / 'ref.csv'}", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == {"S:5": "old/S:0"}
