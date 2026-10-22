"""The learned "measurable" output: model head, negatives from skipped annotations, dataset targets,
loss, training wiring, inference aggregation and the gate that uses it."""
import csv
import math

import numpy as np
import pytest
import torch
from PIL import Image

from models.unet import UNetGNRes, measurable_from_state_dict


def _model(head=True):
    return UNetGNRes(n_classes=4, landmark_head=True, landmark_hidden=32, landmark_root=True,
                     measurable_head=head)


def test_head_output_and_default_two_tuple():
    x = torch.randn(2, 3, 76, 76)
    m = _model(head=True).eval()
    with torch.no_grad():
        seg, kp, logit = m.forward_with_landmarks(x, return_measurable=True)
        assert logit.shape == (2,) and seg.shape[0] == kp.shape[0] == 2
        assert len(m.forward_with_landmarks(x)) == 2                 # existing callers unchanged
        plain = _model(head=False).eval()
        assert plain.forward_with_landmarks(x, return_measurable=True)[2] is None
    assert measurable_from_state_dict(m.state_dict())
    assert not measurable_from_state_dict(plain.state_dict())
    assert measurable_from_state_dict({"module.cls_measurable.weight": 0})


def test_old_checkpoints_load_and_new_ones_build_the_head(tmp_path):
    from models.multiclass_inference import MulticlassInference

    for head in (False, True):
        path = tmp_path / f"m{int(head)}.pt"
        torch.save(_model(head).state_dict(), path)
        inf = MulticlassInference(str(path), num_classes=4, device=torch.device("cpu"), in_size=76, out_size=64,
                                  margin=6)
        assert getattr(inf._core(), "measurable_head", False) == head
        image = (np.random.RandomState(0).rand(150, 50, 3) * 255).astype(np.uint8)   # a tall crop: 3 tiles
        (_, _, extra), = inf.predict_landmark_fields([image])
        if head:
            assert 0.0 <= extra["measurable"] <= 1.0
        else:
            assert "measurable" not in (extra or {})


def test_flip_and_ensemble_average_the_probability():
    from models.multiclass_inference import merge_landmark_fields

    hm, vec = np.zeros((4, 4)), np.zeros((4, 4, 4))
    merged = merge_landmark_fields([(hm, vec, {"measurable": 0.2}), (hm, vec, {"measurable": 0.6}), (hm, vec, None)])
    assert merged[2]["measurable"] == pytest.approx(0.4)


def test_readout_carries_measurable_and_the_gate_uses_it():
    from utils.landmark_gate import MEASURABLE_THRESHOLD, annotate_readout
    from utils.landmark_readout import readout_from_fields

    hm = np.zeros((40, 40), np.float32)
    hm[20, 20] = 0.9
    vec = np.zeros((4, 40, 40), np.float32)
    vec[1], vec[2] = 1.0, 1.0
    r = readout_from_fields(hm, vec, measurable=0.93)
    assert r["measurable_p"] == pytest.approx(0.93)
    annotate_readout(r)
    assert r["gate_source"] == "measurable" and r["unreadable_p"] == pytest.approx(0.07) and not r["unreadable"]
    r2 = annotate_readout(readout_from_fields(hm, vec, measurable=1 - MEASURABLE_THRESHOLD - 0.01))
    assert r2["unreadable"]
    r3 = annotate_readout(readout_from_fields(hm, vec))                   # no head: the logistic gate
    assert r3["gate_source"] == "logistic" and "measurable_p" not in r3


# ----------------------------------------------------------------- labels
def _write_rows(path, rows):
    from tests.test_landmarks import _write_landmark_csv
    _write_landmark_csv(path, rows)


def test_negatives_from_skipped_and_reviewed_unmeasurable_rows(tmp_path):
    from multi.src.landmarks import load_measurable_negatives
    from utils.angle_annotation import CSV_FIELDS

    p = tmp_path / "a.csv"
    _write_rows(p, [
        {"frame": "f1", "crop_file": "0-crop-S_f1.png", "series": "S", "crop_id": 0},
        {"frame": "f2", "crop_file": "0-crop-S_f2.png", "series": "S", "crop_id": 0, "status": "skipped"},
        {"frame": "f3", "crop_file": "0-crop-S_f3.png", "series": "S", "crop_id": 0},     # reviewed: unmeasurable
        {"frame": "f4", "crop_file": "0-crop-S_f4.png", "series": "S", "crop_id": 0},     # reviewed: rotation
        {"frame": "f5", "crop_file": "1-crop-S_f5.png", "series": "S", "crop_id": 1, "status": "skipped"},
    ])
    with open(str(p) + ".review.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerow({"series": "S", "crop_id": 0, "frame": "f3", "crop_file": "0-crop-S_f3.png", "status": "unmeasurable"})
        w.writerow({"series": "S", "crop_id": 0, "frame": "f4", "crop_file": "0-crop-S_f4.png", "status": "rotation"})
    neg = load_measurable_negatives(str(p), exclude_filenames={"1-crop-S_f5.png"})
    assert neg == {"0-crop-S_f2.png": ("S", 0), "0-crop-S_f3.png": ("S", 0)}
    tagged = load_measurable_negatives(str(p), tag="new", seedling_groups={"S:1": "other:7"})
    assert tagged["new/0-crop-S_f2.png"] == ("new/S", 0) and tagged["new/1-crop-S_f5.png"] == ("other", 7)


def test_dataset_targets_for_measured_skipped_and_unlabelled_tiles(tmp_path):
    from multi.src.data_loader import PatchDataset
    from tests.test_landmarks import H, SIZE, W, _lm

    raw, masks = tmp_path / "raw", tmp_path / "masks"
    raw.mkdir(), masks.mkdir()
    for n in ("m.png", "s.png", "u.png"):
        Image.fromarray(np.full((H, W, 3), 80, np.uint8)).save(raw / n)
        Image.fromarray(np.zeros((H, W), np.uint8)).save(masks / n)
    index = tmp_path / "idx.csv"
    with open(index, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "x", "y", "patch_size", "foreground_fraction"])
        for n in ("m.png", "s.png", "u.png"):
            w.writerow([n, 0, 0, SIZE, 0.0])
    ds = PatchDataset(index, raw, masks, landmarks={"m.png": _lm(name="m.png")}, measurable_negatives={"s.png"})
    targets = {row["filename"]: ds[i][2] for i, row in enumerate(ds.rows)}
    assert (float(targets["m.png"]["measurable"]), float(targets["m.png"]["has_measurable"])) == (1.0, 1.0)
    assert (float(targets["s.png"]["measurable"]), float(targets["s.png"]["has_measurable"])) == (0.0, 1.0)
    assert float(targets["u.png"]["has_measurable"]) == 0.0
    assert "measurable" not in PatchDataset(index, raw, masks, landmarks={"m.png": _lm(name="m.png")})[0][2]


def test_measurable_loss():
    from multi.src.loss_functions import measurable_loss

    logit = torch.tensor([3.0, -3.0, 0.0], requires_grad=True)
    t = {"measurable": torch.tensor([1.0, 0.0, 1.0]), "has_measurable": torch.tensor([1.0, 1.0, 0.0])}
    loss, raw = measurable_loss(logit, t)
    assert raw == pytest.approx(float(torch.nn.functional.softplus(torch.tensor(-3.0))), rel=1e-4)
    loss.backward()
    assert logit.grad[2] == 0                                             # an ignored tile has no gradient
    assert measurable_loss(logit, t, weight=0)[1] == 0.0
    assert measurable_loss(logit, {**t, "has_measurable": torch.zeros(3)})[1] == 0.0
    assert measurable_loss(None, t)[1] == 0.0


def test_auc_helper():
    import multi.train_unet_multiclass as trainer

    assert trainer._auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert trainer._auc([1, 0], [0.5, 0.5]) == 0.5
    assert math.isnan(trainer._auc([1, 1], [0.2, 0.3]))


# ----------------------------------------------------------------- training
def _measurable_config(tmp_path, warm_start=None):
    from tests.test_landmarks import H, SIZE, W, _training_config, _write_landmark_csv

    config = _training_config(tmp_path)
    rng = np.random.RandomState(2)
    for n in ("s0.png", "s1.png"):                                        # skipped frames of seedlings 0 and 1
        Image.fromarray((rng.rand(H, W, 3) * 255).astype(np.uint8)).save(tmp_path / "raw" / n)
        Image.fromarray(np.zeros((H, W), np.uint8)).save(tmp_path / "masks" / n)
    with open(tmp_path / "patch_index" / "train_patches.csv", "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for n in ("s0.png", "s1.png"):
            w.writerow([n, 0, 0, SIZE, 0.0])
    _write_landmark_csv(tmp_path / "landmarks.csv", [
        {"frame": "a", "crop_file": "a.png", "series": "S", "crop_id": 0},
        {"frame": "b", "crop_file": "b.png", "series": "S", "crop_id": 1},
        {"frame": "v", "crop_file": "v.png", "series": "S", "crop_id": 2},
        {"frame": "s0", "crop_file": "s0.png", "series": "S", "crop_id": 0, "status": "skipped"},
        {"frame": "s1", "crop_file": "s1.png", "series": "S", "crop_id": 1, "status": "skipped"},
    ])
    config["model"].update(measurable_head=True, landmark_hidden=32, landmark_root=False)
    config["landmarks"]["loss_weights"]["measurable"] = 1.0
    config["training"]["selection_metric"] = "measurable_auc"
    if warm_start:
        config["model"]["warm_start_from"] = str(warm_start)
        config["model"]["warm_start_keep_head"] = True
        config["landmarks"].update(trainable_only=["cls_measurable."], head_only_epochs=999, labelled_only=True)
    return config


def test_training_run_with_the_measurable_head(tmp_path, capsys):
    import multi.train_unet_multiclass as trainer

    history = trainer.run_training(_measurable_config(tmp_path))
    out = capsys.readouterr().out
    assert "Measurable head: 1 measured / 1 skipped training crops, 1 / 1 held out" in out
    assert history["selection_metric"] == "measurable_auc" and len(history["measurable_auc"]) == 2
    state = torch.load(history["checkpoint_path"], map_location="cpu", weights_only=True)
    assert measurable_from_state_dict(state)


def test_head_only_run_keeps_the_angle_maps_bit_identical(tmp_path):
    import multi.train_unet_multiclass as trainer

    source = UNetGNRes(n_classes=4, landmark_head=True, landmark_hidden=32)
    src_path = tmp_path / "member.pt"
    torch.save(source.state_dict(), src_path)
    history = trainer.run_training(_measurable_config(tmp_path, warm_start=src_path))
    state = torch.load(history["checkpoint_path"], map_location="cpu", weights_only=True)
    for key, value in source.state_dict().items():
        assert torch.equal(state[key], value), key                       # nothing but cls_measurable moved
    assert measurable_from_state_dict(state)


def test_negative_weight_applies_without_oversampling(tmp_path, monkeypatch):
    import multi.train_unet_multiclass as trainer

    seen = {}
    real = trainer.WeightedRandomSampler

    def spy(weights, **kw):
        seen["weights"] = list(weights)
        return real(weights, **kw)
    monkeypatch.setattr(trainer, "WeightedRandomSampler", spy)
    config = _measurable_config(tmp_path)
    config["landmarks"].update(oversample=1.0, negative_weight=0.5, labelled_only=True)
    trainer.run_training(config)
    assert sorted(set(seen["weights"])) == [0.5, 1.0]                  # skipped crops drawn half as often
