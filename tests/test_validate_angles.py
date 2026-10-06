"""Tests for the pure parts of multi/validate_angles.py (truth loading, series
ordering, seed proxy, error statistics). The pipeline run itself needs weights
and a GPU-capable environment and is exercised by running the script."""
from __future__ import annotations

import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "multi"))
import validate_angles as va  # noqa: E402


def _write_truth(path, rows):
    fields = ["series", "seedling_id", "crop_id", "frame", "img_name", "crop_file", "status", "bio_angle", "overhook"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({f: r.get(f, "") for f in fields})


def test_load_truth_keeps_only_measured_rows(tmp_path):
    path = tmp_path / "gt.csv"
    _write_truth(path, [
        {"series": "A", "crop_id": 0, "frame": "f1", "crop_file": "0-crop-A_f1.png", "status": "measured", "bio_angle": "150.5", "overhook": "0"},
        {"series": "A", "crop_id": 0, "frame": "f2", "crop_file": "0-crop-A_f2.png", "status": "skipped"},
        {"series": "B", "crop_id": 1, "frame": "f1", "crop_file": "1-crop-B_f1.png", "status": "measured", "bio_angle": "200", "overhook": "1"},
    ])
    rows = va.load_truth(str(path))
    assert [(r["series"], r["crop_id"], r["gt"], r["gt_overhook"]) for r in rows] == [("A", 0, 150.5, 0), ("B", 1, 200.0, 1)]


def test_load_series_keeps_manifest_order_and_only_wanted_seedlings(tmp_path):
    manifest = tmp_path / "manifest.csv"
    bs = chr(92)
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["output_path", "series", "crop_id"])
        w.writeheader()
        for series, cid, f in (("A", 0, "f10"), ("A", 0, "f9"), ("A", 1, "f10"), ("B", 0, "f1")):
            w.writerow({"output_path": "cropped_training_set" + bs + f"{cid}-crop-{series}_{f}.png", "series": series, "crop_id": cid})
    truth = [{"series": "A", "crop_id": 0}, {"series": "B", "crop_id": 0}]
    series = va.load_series(str(manifest), truth)
    assert list(series) == [("A", 0), ("B", 0)]
    assert series[("A", 0)] == ["0-crop-A_f10.png", "0-crop-A_f9.png"]      # manifest order, not natural sort


def test_proxy_seed_point_sits_at_the_lower_padding_boundary():
    assert va.proxy_seed_point(76, 144) == (38, 132)
    assert va.proxy_seed_point(76, 144, pad_fraction=0.0) == (38, 144)


def test_error_stats_basic_and_missing_predictions():
    s = va.error_stats([160.0, 100.0, None, float("nan")], [150.0, 120.0, 90.0, 80.0])
    assert s["n"] == 4 and s["n_pred"] == 2 and s["coverage"] == 0.5
    assert s["mae"] == pytest.approx(15.0) and s["bias"] == pytest.approx(-5.0)       # errors +10, -20
    assert s["median_ae"] == pytest.approx(15.0) and s["rmse"] == pytest.approx((250) ** 0.5)
    assert s["within10"] == pytest.approx(0.5) and s["within20"] == pytest.approx(1.0)


def test_error_stats_overhook_agreement_and_empty():
    s = va.error_stats([190.0, 170.0], [200.0, 190.0])       # both gt overhooked; pred right on the first only
    assert s["overhook_agree"] == pytest.approx(0.5)
    empty = va.error_stats([None], [100.0])
    assert empty["n_pred"] == 0 and empty["mae"] is None and empty["coverage"] == 0.0


def test_a_mirrored_convention_shows_up_as_a_large_error():
    truth = [175.0, 150.0, 60.0, 20.0]                       # closed ... open, bio convention
    perfect = va.error_stats(truth, truth)
    flipped = va.error_stats([180 - t for t in truth], truth)
    assert perfect["mae"] == 0.0 and flipped["mae"] > 50


def test_per_series_and_table_format():
    rows = [{"series": "A", "gt": 100.0, "raw": 90.0, "raw_flip": 90.0, "recon": 100.0},
            {"series": "B", "gt": 100.0, "raw": None, "raw_flip": None, "recon": 120.0}]
    ps = va.per_series(rows, "recon")
    assert ps["A"]["mae"] == 0.0 and ps["B"]["mae"] == 20.0
    table = va.format_table({"binary": rows})
    assert "binary" in table and "recon" in table and "raw_flip" in table


def test_table_shows_landmark_readings_only_for_the_backend_that_has_them():
    pipeline = [{"series": "A", "gt": 100.0, "raw": 10.0, "raw_flip": 170.0, "recon": 100.0}]
    landmark = [{"series": "A", "gt": 100.0, "landmark": 95.0, "landmark_recon": 98.0}]
    table = va.format_table({"multiclass": pipeline, "landmark:run": landmark})
    lines = table.splitlines()
    assert any(l.startswith("landmark:run") and "landmark_recon" in l for l in lines)
    assert not any(l.startswith("multiclass") and "landmark" in l for l in lines)
    assert not any(l.startswith("landmark:run") and " raw " in l for l in lines)


def test_run_landmarks_scores_a_checkpoint_with_the_landmark_head(tmp_path):
    import cv2
    import numpy as np
    import torch
    from models.unet import UNetGNRes

    torch.manual_seed(0)
    ckpt = tmp_path / "lm.pt"
    torch.save(UNetGNRes(n_classes=4, landmark_head=True).state_dict(), ckpt)
    plain = tmp_path / "plain.pt"
    torch.save(UNetGNRes(n_classes=4).state_dict(), plain)

    crops = tmp_path / "crops"
    crops.mkdir()
    rng = np.random.RandomState(0)
    files = [f"0-crop-S_f{i}.png" for i in range(3)]
    for name in files:
        cv2.imwrite(str(crops / name), (rng.rand(120, 60, 3) * 255).astype(np.uint8))
    series = {("S", 0): files}

    out = va.run_landmarks(str(ckpt), series, crops)
    assert set(out) == {("S", 0, f) for f in files}
    assert all(set(v) == {"landmark", "landmark_recon", "landmark_temporal"} for v in out.values())
    with pytest.raises(SystemExit):
        va.run_landmarks(str(plain), series, crops)
