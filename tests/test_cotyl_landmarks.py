"""Cotyledon base and tip heatmaps: targets, head, loss, readout, sampling and the monotone pass."""
import math

import numpy as np
import pytest
import torch

from models.unet import (
    KP_CHANNELS_COTYL, UNetGNRes, landmark_cotyl_from_state_dict, landmark_root_from_state_dict,
)
from multi.src.landmarks import (
    Landmark, angle_balanced_weights, angle_bin, load_landmarks, render_targets,
)
from multi.src.loss_functions import landmark_loss
from utils.landmark_readout import readout_from_fields
from utils.landmark_timeseries import monotone_opening

H, W = 120, 60


def gauss(x, y, sigma=3.0):
    yy, xx = np.mgrid[0:H, 0:W]
    return np.exp(-((xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2) / (2 * sigma ** 2)).astype(np.float32)


def lm(**kw):
    base = dict(filename="0-crop-a.png", seedling=("S", 0), junction=(30.5, 30.5), hypo_dir=(0.0, 1.0),
                cotyl_dir=(0.0, -1.0), hypo_len=40.0, cotyl_len=15.0, cotyl_base=(30.5, 28.5), cotyl_tip=(30.5, 15.5))
    return Landmark(**{**base, **kw})


# --- targets ---------------------------------------------------------------------

def test_base_and_tip_targets_peak_at_the_clicks_and_are_empty_without_them():
    t = render_targets(H, W, lm())
    assert np.unravel_index(np.argmax(t["base_hm"]), t["base_hm"].shape) == (28, 30)
    assert np.unravel_index(np.argmax(t["tip_hm"]), t["tip_hm"].shape) == (15, 30)
    none = render_targets(H, W, lm(cotyl_base=None, cotyl_tip=None))
    assert none["base_hm"].max() == 0 and none["tip_hm"].max() == 0


def test_load_landmarks_takes_base_and_tip_as_the_near_and_far_cotyledon_click(tmp_path):
    from tests.test_landmarks import _write_landmark_csv
    p = tmp_path / "gt.csv"
    _write_landmark_csv(p, [
        {"frame": "f1", "crop_file": "0-crop-A_f1.png", "cotyl": ((45.5, 30.5), (55.5, 30.5))},
        {"frame": "f2", "crop_file": "0-crop-A_f2.png", "cotyl": ((55.5, 30.5), (45.5, 30.5))},   # clicked far first
    ])
    lms, _ = load_landmarks(str(p))
    for name in lms:
        assert lms[name].cotyl_base == (45.5, 30.5) and lms[name].cotyl_tip == (55.5, 30.5)


def test_the_dataset_carries_base_and_tip_through_flip_and_rotation(tmp_path):
    from PIL import Image
    from multi.src.data_loader import PatchDataset
    import csv
    size = 108
    rng = np.random.RandomState(0)
    Image.fromarray((rng.rand(H, W, 3) * 255).astype(np.uint8)).save(tmp_path / "a.png")
    Image.fromarray(np.zeros((H, W), np.uint8)).save(tmp_path / "m.png")
    (tmp_path / "m").mkdir()
    Image.fromarray(np.zeros((H, W), np.uint8)).save(tmp_path / "m" / "a.png")
    index = tmp_path / "idx.csv"
    with open(index, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "x", "y", "patch_size", "foreground_fraction"])
        w.writerow(["a.png", 0, 0, size, 0.0])
    ds = PatchDataset(index, tmp_path, tmp_path / "m", landmarks={"a.png": lm(filename="a.png")})
    _, _, kp = ds[0]
    assert kp["tip_hm"].shape == (size, size) and kp["has_tip"] == 1.0
    ys, xs = np.unravel_index(int(kp["tip_hm"].argmax()), kp["tip_hm"].shape)
    ref = np.unravel_index(np.argmax(render_targets(H, W, lm())["tip_hm"]), (H, W))
    # the crop is padded up to the patch size, centred: nothing along the tall axis, 24 px across
    assert abs(ys - (ref[0] + max(0, size - H) // 2)) <= 1 and abs(xs - (ref[1] + max(0, size - W) // 2)) <= 1
    flipped = PatchDataset(index, tmp_path, tmp_path / "m", landmarks={"a.png": lm(filename="a.png")})
    t = {k: v[:, ::-1].copy() if v.ndim == 2 else v for k, v in
         {"tip_hm": kp["tip_hm"].numpy(), "base_hm": kp["base_hm"].numpy()}.items()}
    out = flipped._augment(np.zeros((size + 2 * 12, size + 2 * 12, 3), np.float32), np.zeros((size, size), np.uint8),
                           {**{k: np.zeros_like(v.numpy()) for k, v in kp.items() if k in
                               ("hm", "paf", "paf_valid", "collar_hm", "root_vec", "root_valid")},
                            "tip_hm": kp["tip_hm"].numpy(), "base_hm": kp["base_hm"].numpy()})
    assert out[2]["tip_hm"].shape == (size, size)


# --- head and checkpoints ---------------------------------------------------------

def test_the_ten_channel_head_builds_loads_and_is_detected():
    model = UNetGNRes(n_classes=4, head="plain", landmark_head=True, landmark_hidden=64, landmark_root=True,
                      landmark_cotyl=True)
    state = model.state_dict()
    assert state["conv_kp.weight"].shape[0] == KP_CHANNELS_COTYL == 10
    assert landmark_cotyl_from_state_dict(state) and landmark_root_from_state_dict(state)
    seg, kp = model.forward_with_landmarks(torch.zeros(1, 3, 64, 64))
    assert kp.shape[1] == 10 and torch.equal(seg, model(torch.zeros(1, 3, 64, 64)))
    old = UNetGNRes(n_classes=4, head="plain", landmark_head=True, landmark_hidden=64, landmark_root=True).state_dict()
    assert not landmark_cotyl_from_state_dict(old) and landmark_root_from_state_dict(old)
    with pytest.raises(ValueError):
        UNetGNRes(n_classes=4, landmark_head=True, landmark_cotyl=True)          # needs the root channels
    assert float(model.conv_kp.bias[8]) == pytest.approx(-4.6) == float(model.conv_kp.bias[9])


# --- loss ------------------------------------------------------------------------

def _targets(n=2, with_tip=True):
    hm = torch.from_numpy(gauss(30.5, 30.5))[None].repeat(n, 1, 1)
    tip = torch.from_numpy(gauss(30.5, 15.5))[None].repeat(n, 1, 1)
    z4, z2, z1 = torch.zeros(n, 4, H, W), torch.zeros(n, 2, H, W), torch.zeros(n, 1, H, W)
    return {"hm": hm, "paf": z4, "paf_valid": z2, "has_kp": torch.ones(n), "collar_hm": torch.zeros(n, H, W),
            "root_vec": z2, "root_valid": z1, "has_collar": torch.zeros(n),
            "base_hm": hm.clone(), "tip_hm": tip, "has_tip": torch.ones(n) if with_tip else torch.zeros(n)}


def test_the_tip_loss_trains_the_new_channels_and_ignores_tiles_without_the_tip():
    logits = torch.zeros(2, 10, H, W, requires_grad=True)
    total, parts = landmark_loss(logits, _targets())
    assert "cotyl" in parts and parts["cotyl"] > 0
    total.backward()
    assert logits.grad[:, 8:10].abs().sum() > 0
    logits2 = torch.zeros(2, 10, H, W, requires_grad=True)
    _, parts2 = landmark_loss(logits2, _targets(with_tip=False))
    assert parts2["cotyl"] == 0.0
    ten, eight = torch.zeros(1, 10, H, W), torch.zeros(1, 8, H, W)
    t = _targets(1)
    assert "cotyl" not in landmark_loss(eight, t)[1]                              # an 8-channel head is unchanged
    assert float(landmark_loss(ten[:, :8], t)[0]) == pytest.approx(float(landmark_loss(eight, t)[0]))


# --- readout ---------------------------------------------------------------------

def _fields(junction=(30.5, 60.5), hypo=(0.0, 1.0), cotyl_field=(0.0, 1.0), tip=None, base=None, decoy=None):
    hm = gauss(*junction)
    if decoy is not None:
        hm = np.maximum(hm, 0.9 * gauss(*decoy))
        hm = np.maximum(hm, 0.0)
    vec = np.zeros((4, H, W), np.float32)
    vec[0], vec[1], vec[2], vec[3] = hypo[0], hypo[1], cotyl_field[0], cotyl_field[1]
    extra = {}
    if tip is not None:
        extra["tip_hm"] = gauss(*tip)
        extra["base_hm"] = gauss(*(base or junction))
    return hm, vec, extra


def test_readout_uses_base_to_tip_when_the_field_is_wrong_and_falls_back_otherwise():
    # an open hook read at a junction that slid down the stem: the field points down the hypocotyl (closed)
    hm, vec, extra = _fields(tip=(30.5, 40.5), base=(30.5, 58.5))
    r = readout_from_fields(hm, vec, **extra)
    assert r["cotyl_path"] == "base_tip" and r["cotyl_dir"] == pytest.approx((0.0, -1.0), abs=1e-6)
    assert r["theta"] == pytest.approx(180.0, abs=1e-3) and r["tip"] == (30.5, 40.5)
    hm, vec, extra = _fields(tip=(30.5, 40.5))
    extra["base_hm"] = np.zeros((H, W), np.float32)                              # no base peak
    assert readout_from_fields(hm, vec, **extra)["cotyl_path"] == "junction_tip"
    extra = {"tip_hm": np.zeros((H, W), np.float32), "base_hm": np.zeros((H, W), np.float32)}
    r = readout_from_fields(hm, vec, **extra)
    assert r["cotyl_path"] == "field" and r["theta"] == pytest.approx(0.0, abs=1e-3) and r["tip"] is None
    assert readout_from_fields(hm, vec)["cotyl_path"] == "field"                  # a head without the channels
    close = _fields(tip=(30.5, 59.5), base=(30.5, 60.5))                          # tip 1 px from the junction
    assert readout_from_fields(close[0], close[1], **close[2])["cotyl_path"] == "field"


def test_the_tip_picks_the_junction_peak_it_makes_sense_with():
    # the strongest peak (y=100) is 85 px from the tip; the true junction (y=60) is 20 px away
    hm, vec, extra = _fields(junction=(30.5, 100.5), tip=(30.5, 40.5), base=(30.5, 58.5), decoy=(30.5, 60.5))
    assert hm[100, 30] > hm[60, 30]
    r = readout_from_fields(hm, vec, **extra)
    assert abs(r["junction"][1] - 60.5) <= 1.0
    r0 = readout_from_fields(hm, vec)                                            # without a tip: the strongest peak
    assert abs(r0["junction"][1] - 100.5) <= 1.0


# --- sampling --------------------------------------------------------------------

def test_angle_balanced_weights_give_every_bin_the_same_total():
    def with_theta(name, theta):
        u = (1.0, 0.0)
        v = (math.cos(math.radians(theta)), math.sin(math.radians(theta)))
        return Landmark(name, ("S", int(name)), (10.0, 10.0), u, v, 30.0, 15.0)
    lms = {str(i): with_theta(str(i), 10) for i in range(6)}                      # six closed
    lms.update({str(10 + i): with_theta(str(10 + i), 90) for i in range(2)})      # two middle
    lms["20"] = with_theta("20", 170)                                             # one open
    w = angle_balanced_weights(lms)
    totals = {}
    for name, lmk in lms.items():
        totals[angle_bin(lmk.theta)] = totals.get(angle_bin(lmk.theta), 0) + w[name]
    assert list(totals) == [0, 1, 2] and all(abs(v - 3.0) < 1e-9 for v in totals.values())
    assert abs(sum(w.values()) / len(w) - 1.0) < 1e-9 and angle_balanced_weights({}) == {}


# --- monotone pass ---------------------------------------------------------------

def test_monotone_opening_keeps_clean_series_and_repairs_lone_frames():
    clean = [176, 176, 170, 150, 120, 90, 60, 40, 30, 30]
    out = monotone_opening(clean, step=1.0)
    assert out == pytest.approx(clean, abs=1.0)
    lone = [176, 176, 90, 176, 170, 140, 60, 140, 30, 30]            # a lone open and a lone re-closing frame
    fixed = monotone_opening(lone, step=1.0)
    assert all(a >= b for a, b in zip(fixed, fixed[1:]))              # non-increasing
    assert fixed[2] >= 150 and fixed[7] <= 60                         # the lone frames are pulled to their neighbours
    long_open = [176, 176, 40, 40, 40, 40, 40]                        # a long open run stays open
    assert monotone_opening(long_open, step=1.0)[-1] <= 45
    gappy = [176, float("nan"), float("nan"), 100, float("nan"), 30]
    g = monotone_opening(gappy, step=1.0)
    assert len(g) == 6 and not any(math.isnan(v) for v in g) and all(a >= b for a, b in zip(g, g[1:]))
    assert monotone_opening([]) == [] and all(math.isnan(v) for v in monotone_opening([float("nan")] * 3))


# --- a real (tiny) training run ----------------------------------------------------

def test_two_epoch_training_run_with_base_tip_heads_and_balanced_sampling(tmp_path, capsys):
    import multi.train_unet_multiclass as trainer
    from tests.test_landmarks import _training_config
    config = _training_config(tmp_path)
    config["model"].update(landmark_root=True, landmark_cotyl=True)
    config["landmarks"].update(sampling="angle_balanced", oversample=4.0)
    history = trainer.run_training(config)
    assert "Angle-balanced sampling" in capsys.readouterr().out
    for key in ("wholecrop_tip_px", "wholecrop_base_px", "wholecrop_tip_used"):
        assert len(history[key]) == 2
    state = torch.load(history["checkpoint_path"], map_location="cpu", weights_only=True)
    assert state["conv_kp.weight"].shape[0] == KP_CHANNELS_COTYL
    config2 = _training_config(tmp_path / "again") if (tmp_path / "again").mkdir() is None else None
    config2["model"]["landmark_cotyl"] = True                                 # without landmark_root
    with pytest.raises(ValueError, match="landmark_cotyl requires model.landmark_root"):
        trainer.run_training(config2)
    config3 = _training_config(tmp_path / "third") if (tmp_path / "third").mkdir() is None else None
    config3["landmarks"]["sampling"] = "sideways"
    with pytest.raises(ValueError, match="landmarks.sampling"):
        trainer.run_training(config3)


def test_overhook_is_a_ray_crossing():
    from utils.landmark_readout import rays_cross
    hypo = ((0.0, 1.0))                                                # junction (0,0), stem runs down
    assert rays_cross((-3, 0), (0.2, 0.98), (0, 0), hypo)             # cotyledon swings back across the stem
    assert not rays_cross((-3, 0), (-0.2, 0.98), (0, 0), hypo)        # closed, diverging
    assert not rays_cross((-3, 0), (0.0, -1.0), (0, 0), hypo)         # open: points away from the stem
    assert not rays_cross((-3, 0), (0.0, 1.0), (0, 0), hypo)          # parallel


def test_ensemble_averages_the_maps_before_the_readout_and_mixes_plain_and_cotyl_heads():
    import numpy as np
    from models.multiclass_inference import LandmarkEnsemble, merge_landmark_fields
    hm = np.zeros((8, 8), np.float32)
    plain = (hm + 0.2, np.ones((4, 8, 8), np.float32), None)
    full = (hm + 0.6, np.zeros((4, 8, 8), np.float32),
            {"collar_hm": hm + 1.0, "root_vec": np.ones((2, 8, 8), np.float32), "base_hm": hm + 0.5, "tip_hm": hm + 0.9})
    other = (hm + 0.4, np.ones((4, 8, 8), np.float32), {"collar_hm": hm + 0.0, "root_vec": np.zeros((2, 8, 8), np.float32)})
    m_hm, m_vec, m_extra = merge_landmark_fields([plain, full, other])
    assert np.allclose(m_hm, 0.4) and np.allclose(m_vec, 2 / 3)
    assert np.allclose(m_extra["collar_hm"], 0.5) and np.allclose(m_extra["root_vec"], 0.5)     # members that have it
    assert np.allclose(m_extra["base_hm"], 0.5) and np.allclose(m_extra["tip_hm"], 0.9)         # only one member has them
    assert merge_landmark_fields([plain, plain])[2] is None

    class Member:
        def __init__(self, fields):
            self.fields = fields

        def predict_landmark_fields(self, images, flip_tta=None):
            return [self.fields for _ in images]

    out = LandmarkEnsemble([Member(plain), Member(full)]).predict_landmark_fields([object(), object()])
    assert len(out) == 2 and np.allclose(out[0][0], 0.4)
