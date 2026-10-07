import json

import numpy as np
import pytest

from utils.germination_detector import GerminationDetector
from utils.germination_learned import (
    FEATURE_NAMES, GerminationModel, best_onset, frame_features, onset_or_none,
)


def write_weights(path, checkpoint="", weights=None, bias=0.0):
    n = len(FEATURE_NAMES)
    path.write_text(json.dumps({"features": FEATURE_NAMES, "mean": [0.0] * n, "std": [1.0] * n,
                                "weights": weights or [0.0] * n, "bias": bias, "checkpoint": checkpoint}))
    return path


def test_frame_features_read_the_radicle_and_seedling_channels():
    probs = np.zeros((4, 10, 10), np.float32)
    probs[3, 2:4, 2:4] = 0.9                                           # 4 confident radicle pixels
    probs[1] = 0.1
    f = dict(zip(FEATURE_NAMES, frame_features(probs)))
    assert f["r_max"] == pytest.approx(0.9) and f["r_log_n50"] == pytest.approx(np.log1p(4))
    assert f["c_log_mass"] == pytest.approx(np.log1p(10.0), rel=1e-5) and f["h_log_mass"] == 0.0


def test_one_step_explains_the_series_and_a_lone_noisy_frame_does_not_move_it():
    assert best_onset([0.1, 0.1, 0.9, 0.9, 0.9]) == 2
    assert best_onset([0.1, 0.1, 0.9, 0.2, 0.9, 0.9]) == 2            # one dip after the onset
    assert best_onset([0.1, 0.8, 0.1, 0.1, 0.1]) == 5                 # one blip before: no radicle at all
    assert onset_or_none([0.1, 0.2]) is None and onset_or_none([]) is None
    assert onset_or_none([0.9, 0.9]) == 0


def test_model_loads_and_rejects_other_features(tmp_path):
    p = write_weights(tmp_path / "w.json", weights=[1.0] + [0.0] * (len(FEATURE_NAMES) - 1), bias=-0.5)
    m = GerminationModel.load(p)
    probs = m.visible_probs([[0.0] * len(FEATURE_NAMES), [1.0] + [0.0] * (len(FEATURE_NAMES) - 1)])
    assert probs[0] < 0.5 < probs[1]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"features": ["x"], "mean": [0], "std": [1], "weights": [0], "bias": 0}))
    with pytest.raises(ValueError):
        GerminationModel.load(bad)


def test_detector_records_a_learned_onset_and_manual_still_wins():
    d = GerminationDetector()
    assert d.method(0) is None
    d.record_learned(0, 3, [0.1, 0.2, 0.4, 0.9])
    assert d.get_time_zero(0) == 3 and d.method(0) == "learned"
    assert "learned onset frame=3" in d.describe(0)
    d.set_override(0, 1)
    assert d.get_time_zero(0) == 1 and d.method(0) == "manual"
    d.clear_override(0)
    d.detect(1, [[], []], (5, 5), 50)                                  # the area rule
    assert d.method(1) == "rule"


def test_method_resolution(tmp_path, monkeypatch):
    from models import segmentation_backends as sb
    ckpt = tmp_path / "m.pt"
    weights = write_weights(tmp_path / "w.json", checkpoint=str(ckpt))
    monkeypatch.delenv(sb.GERMINATION_ENV_VAR, raising=False)
    assert sb.resolve_germination_method(str(weights)) == "rule"      # checkpoint missing -> fall back
    ckpt.write_bytes(b"x")
    assert sb.resolve_germination_method(str(weights)) == "learned"
    assert sb.resolve_germination_method(str(tmp_path / "missing.json")) == "rule"
    monkeypatch.setenv(sb.GERMINATION_ENV_VAR, "rule")
    assert sb.resolve_germination_method(str(weights)) == "rule"
    monkeypatch.setenv(sb.GERMINATION_ENV_VAR, "magic")
    with pytest.raises(ValueError):
        sb.resolve_germination_method(str(weights))


def test_runner_reads_crops_through_the_four_class_model(tmp_path):
    import cv2
    import torch
    from models import segmentation_backends as sb
    from models.unet import UNetGNRes
    torch.manual_seed(0)
    ckpt = tmp_path / "m.pt"
    torch.save(UNetGNRes(n_classes=4).state_dict(), ckpt)
    weights = write_weights(tmp_path / "w.json", checkpoint=str(ckpt))
    paths = []
    for i in range(3):
        p = tmp_path / f"0-crop-f{i}.png"
        cv2.imwrite(str(p), (np.random.RandomState(i).rand(80, 40, 3) * 255).astype(np.uint8))
        paths.append(str(p))
    paths.insert(1, str(tmp_path / "missing.png"))
    runner = sb.LearnedGerminationRunner(str(weights))
    onset, probs = runner.onset(paths)
    assert len(probs) == 4 and probs[1] != probs[1]                   # unreadable crop -> NaN
    assert all(0 < p < 1 for i, p in enumerate(probs) if i != 1)
    assert onset is None or 0 <= onset < 4


def test_gui_uses_the_learned_onset_and_falls_back_to_the_rule(monkeypatch):
    import seedling_measurment as sm

    class Var:
        def get(self):
            return 0

    gui = sm.Gui.__new__(sm.Gui)
    gui.germination_detector = None
    gui.debug_var = Var()
    gui.cropped_filenames_by_crop = {0: ["0-crop-a.png", "0-crop-b.png", "0-crop-c.png"]}
    gui.germ_time_series_by_crop = {0: [[], [], []]}
    gui.crop_points_distributed = [[(10, 10)]]
    gui.crop_boxes = [{"cx": 50, "cy": 50, "half_w": 20, "half_h": 40}]

    class Runner:
        def __init__(self):
            self.paths = None

        def onset(self, paths):
            self.paths = paths
            return 1, [0.1, 0.8, 0.9]

    runner = Runner()
    monkeypatch.setattr(sm, "resolve_germination_method", lambda: "learned")
    monkeypatch.setattr(sm, "get_germination_runner", lambda: runner)
    gui._ensure_germination_detected(0)
    assert gui.germination_detector.get_time_zero(0) == 1 and gui.germination_detector.method(0) == "learned"
    assert [p.replace("\\", "/") for p in runner.paths] == ["data/images/0-crop-a.png", "data/images/0-crop-b.png",
                                                           "data/images/0-crop-c.png"]

    def broken():
        raise RuntimeError("no GPU")
    monkeypatch.setattr(sm, "get_germination_runner", broken)
    gui._ensure_germination_detected(0)                               # learned run fails -> area rule
    assert gui.germination_detector.method(0) == "rule"

    monkeypatch.setattr(sm, "resolve_germination_method", lambda: "rule")
    monkeypatch.setattr(sm, "get_germination_runner", lambda: pytest.fail("rule mode must not load the model"))
    gui._ensure_germination_detected(0)
    assert gui.germination_detector.method(0) == "rule"
