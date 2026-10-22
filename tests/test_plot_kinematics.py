import csv

import numpy as np

from multi import plot_kinematics as pk
from multi.validate_angles import SERIES_FIELDS


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SERIES_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({**{f: "" for f in SERIES_FIELDS}, **r})


def rows():
    out = []
    for i in range(4):
        out.append({"series": "S", "crop_id": 0, "frame_idx": i, "crop_file": f"0-crop-f{i}.png",
                    "gt": 170 - 20 * i if i in (0, 3) else "", "gt_overhook": 0 if i in (0, 3) else "",
                    "landmark": 168 - 20 * i, "recon": 150})
    return out


def test_series_csv_is_loaded_in_frame_order_with_nan_for_blanks(tmp_path):
    p = tmp_path / "s.csv"
    write(p, list(reversed(rows())))
    series = pk.load_series_csv(p)
    e = series[("S", 0)]
    assert list(e["idx"]) == [0, 1, 2, 3]
    assert np.isnan(e["gt"][1]) and e["gt"][3] == 110
    assert pk.method_stats(series, "landmark") == (2, 2.0, 2.0)
    assert pk.method_stats(series, "recon")[0] == 2


def test_plots_are_written(tmp_path):
    p = tmp_path / "s.csv"
    write(p, rows())
    assert pk.main(["--series-csv", str(p), "--out-dir", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "scatter.png").exists()
    assert (tmp_path / "out" / "kinematics_S.png").exists()
