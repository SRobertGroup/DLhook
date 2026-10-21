"""multi/validate_measurable.py: the annotated set it scores and the head-vs-logistic report (the
ensemble reads are faked; no model runs)."""
import argparse
import json

import numpy as np
from PIL import Image

from tests.test_landmarks import _write_landmark_csv


def test_annotated_set_keeps_measured_and_skipped_frames_with_a_crop(tmp_path):
    from multi.validate_measurable import annotated_set

    crops = tmp_path / "crops"
    crops.mkdir()
    for n in ("0-crop-A_f1.png", "0-crop-A_f2.png"):
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(crops / n)
    csv_path = tmp_path / "a.csv"
    _write_landmark_csv(csv_path, [
        {"frame": "f1", "crop_file": "0-crop-A_f1.png"},
        {"frame": "f2", "crop_file": "0-crop-A_f2.png", "status": "skipped"},
        {"frame": "f3", "crop_file": "0-crop-A_f3.png"},                     # no crop on disk
    ])
    items = annotated_set(csv_path, crops)
    assert [(p.name, measured) for p, measured, _ in items] == [("0-crop-A_f1.png", True), ("0-crop-A_f2.png", False)]


def test_score_reports_both_gates_at_one_operating_point(tmp_path, monkeypatch):
    import multi.validate_measurable as vm

    items = [(tmp_path / f"{i}.png", i < 50, 20.0 if i < 50 else None) for i in range(100)]
    rng = np.random.RandomState(0)

    def readout(measured):
        return {"theta": 22.0, "peak": 0.9 if measured else 0.3, "tip_peak": 0.5, "member_theta_sd": 2.0}
    reads = [((0.9 if m else 0.1) + rng.uniform(-0.05, 0.05), readout(m)) for _, m, _ in items]
    monkeypatch.setattr(vm, "annotated_set", lambda *a: items)
    monkeypatch.setattr(vm, "_ensemble", lambda spec: None)
    monkeypatch.setattr(vm, "_read", lambda ens, paths: reads)
    out = tmp_path / "v.json"
    vm.score(argparse.Namespace(truth="t.csv", crops="c", checkpoints="x", threshold=0.5, keep=0.98, out=str(out)))
    result = json.loads(out.read_text(encoding="utf-8"))
    head = result["head"]
    assert head["auc"] == 1.0 and head["measured_kept"] == 1.0 and head["skipped_rejected"] == 1.0
    assert head["skipped_rejected_at_98%_kept"] == 1.0 and head["kept_theta_mae"] == 2.0
    assert set(result["logistic_gate"]) == set(head) and result["all_measured_theta_mae"] == 2.0
