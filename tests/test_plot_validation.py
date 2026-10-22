import csv
import json

import numpy as np

from multi import plot_validation as pv
from multi.validate_angles import SERIES_FIELDS
from utils.germination_learned import FEATURE_NAMES


def write_series(path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SERIES_FIELDS)
        w.writeheader()
        for cid in range(6):
            for i in range(5):
                w.writerow({**{f: "" for f in SERIES_FIELDS}, "series": "S", "crop_id": cid, "frame_idx": i,
                            "crop_file": f"{cid}-crop-f{i}.png", "gt": 175 - 30 * i if i % 2 else "",
                            "landmark": 170 - 30 * i, "recon": 190 if i == 0 else 150})


def test_folding_and_pairing(tmp_path):
    p = tmp_path / "s.csv"
    write_series(p)
    series = pv.load_series_csv(p)
    gt, pred, names = pv.paired(series, "landmark")
    assert len(gt) == 12 and set(names) == {"S"} and np.allclose(pred - gt, -5)
    assert pv.fold([190, 170])[0] == 170
    onsets = {("S", c): 1 for c in range(6)}
    al = pv.aligned(series, onsets, "landmark")
    assert min(al) == 0 and len(al[0]) == 6                 # frame 0 is before germination: dropped


def test_all_figures_are_written(tmp_path):
    p = tmp_path / "s.csv"
    write_series(p)
    with open(tmp_path / "germ.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["series", "crop_id", "status", "onset_index"])
        for c in range(6):
            w.writerow(["S", c, "found", 1])
    with open(tmp_path / "feat.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["series", "crop_id", "frame_index"] + FEATURE_NAMES)
        for c in range(6):
            for i in range(5):
                w.writerow(["S", c, i] + [float(i >= 1) * 5] * len(FEATURE_NAMES))
    n = len(FEATURE_NAMES)
    (tmp_path / "w.json").write_text(json.dumps({"features": FEATURE_NAMES, "mean": [2.5] * n, "std": [1] * n,
                                                 "weights": [1.0] * n, "bias": 0.0}))
    with open(tmp_path / "pairs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["orig_theta", "rep_theta"])
        w.writerows([[10, 12], [90, 85]])
    out = tmp_path / "out"
    assert pv.main(["--series-csv", str(p), "--repeat", str(tmp_path / "pairs.csv"),
                    "--germination", str(tmp_path / "germ.csv"), "--features", str(tmp_path / "feat.csv"),
                    "--germination-weights", str(tmp_path / "w.json"), "--out-dir", str(out)]) == 0
    for name in ("scatter", "error_cdf", "error_by_opening", "per_series_error", "pooled_kinematics",
                 "germination_onset"):
        assert (out / f"{name}.png").exists(), name
    assert pv.learned_onsets(tmp_path / "feat.csv", tmp_path / "w.json")[("S", 0)][0] == 1
