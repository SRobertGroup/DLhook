import csv

import numpy as np
import pytest

from multi import analyze_repeatability as ar
from utils.angle_annotation import CSV_FIELDS, repeat_queue


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({**{f: "" for f in CSV_FIELDS}, **r})


def row(series, crop_id, frame, bio=150.0, status="measured", overhook=0, jx=10.0, jy=20.0):
    return {"series": series, "crop_id": crop_id, "frame": frame, "crop_file": f"{crop_id}-crop-{frame}.png",
            "status": status, "bio_angle": bio if status == "measured" else "",
            "theta": abs(180 - bio) if status == "measured" else "", "overhook": overhook,
            "junction_x": jx, "junction_y": jy}


def test_repeat_queue_is_blind_random_capped_and_deterministic(tmp_path):
    rows = [row("S", c, f"f{i}") for c in range(10) for i in range(8)] + [row("S", 99, "x", status="skipped")]
    src = tmp_path / "a.csv"
    write(src, rows)
    q = repeat_queue([str(src)], n=30, seed=1, max_per_seedling=3)
    assert len(q) == 30 and max(sum(1 for i in q if i.crop_id == c) for c in range(10)) <= 3
    assert all(i.crop_id != 99 for i in q)                                    # skipped frames are never re-offered
    assert q == repeat_queue([str(src)], n=30, seed=1, max_per_seedling=3)
    assert q != repeat_queue([str(src)], n=30, seed=2, max_per_seedling=3)
    assert len({i.crop_id for i in q}) == 10                                  # spread over every seedling
    assert len(repeat_queue([str(src)], n=500, seed=1, max_per_seedling=0)) == 80   # everything measured, no cap


def test_repeat_queue_skips_missing_crops_and_duplicates(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write(a, [row("S", 0, "f1"), row("S", 0, "f2")])
    write(b, [row("S", 0, "f1"), row("S", 1, "g1")])                           # f1 appears in both files
    (tmp_path / "crops").mkdir()
    for name in ("0-crop-f1.png", "1-crop-g1.png"):
        (tmp_path / "crops" / name).write_bytes(b"x")
    q = repeat_queue([str(a), str(b)], n=10, seed=0, max_per_seedling=0, folder=str(tmp_path / "crops"))
    assert sorted(i.filename for i in q) == ["0-crop-f1.png", "1-crop-g1.png"]     # f2 has no crop file


def test_pairing_and_statistics(tmp_path):
    orig = tmp_path / "o.csv"
    rep = tmp_path / "r.csv"
    write(orig, [row("S", 0, "a", 150), row("S", 0, "b", 100), row("S", 0, "c", 170),
                 row("S", 0, "d", 80), row("S", 0, "e", 60, status="skipped")])
    write(rep, [row("S", 0, "a", 154, jx=13, jy=24), row("S", 0, "b", 96), row("S", 0, "c", 170, overhook=1),
                row("S", 0, "d", 80, status="skipped"), row("S", 0, "e", 61), row("S", 5, "zz", 10)])
    pairs, lost, gained = ar.pair_up(ar.load_rows([str(orig)]), ar.load_rows([str(rep)]))
    assert len(pairs) == 3 and lost == 1 and gained == 1                      # "zz" has no original at all
    s = ar.diff_stats([p["orig_bio"] for p in pairs], [p["rep_bio"] for p in pairs])
    assert s["n"] == 3 and s["bias"] == pytest.approx(0.0) and s["mae"] == pytest.approx(8 / 3)
    assert s["repeatability"] == pytest.approx(1.96 * np.std([4, -4, 0], ddof=1))
    first = next(p for p in pairs if p["key"][2] == "a")
    assert first["junction_px"] == pytest.approx(5.0)
    text = ar.summarise(pairs, lost, gained)
    assert "3 frames measured both times" in text and "overhook flag: same call 67%" in text


def test_cli_writes_the_report(tmp_path, capsys):
    orig, rep = tmp_path / "o.csv", tmp_path / "r.csv"
    write(orig, [row("S", 0, f"f{i}", 100 + 10 * i) for i in range(6)])
    write(rep, [row("S", 0, f"f{i}", 102 + 10 * i) for i in range(6)])
    assert ar.main(["--original", str(orig), "--repeat", str(rep), "--out-dir", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "summary.txt").exists() and (tmp_path / "out" / "bland_altman.png").exists()
    assert "bias   +2.0" in capsys.readouterr().out
