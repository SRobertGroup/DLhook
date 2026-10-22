"""Choose the robust series reconstruction's settings on held-out plants, and test them on others.

    python -m multi.tune_series          # after multi.run_series on the BerryBox v2 series (multi/results/series_eval)

Offline: every setting re-runs utils.landmark_timeseries.reconstruct_landmark_series_detail on the
per-frame readings the GUI exported (series.csv: raw_angle = theta, landmark_peak, measurable_p,
unreadable), so no GUI rerun is needed per setting.

Plants are split by seedling. The dense hand series (angle_series_bbval.csv) are divided within each
rig into A (choosing) and B (testing); the sparse annotation (angle_berrybox_val.csv) of every other
plant is a second test set. On A, among the settings that keep >= 95% of the measured frames and the
direction of change between consecutive measured frames (no worse than 2 points below no filter) and
leave no more spikes over all plants than a plain 5-reading median, the
simplest one within 0.3 deg of the best mean error is chosen. Its scores on A, B and the sparse set are
reported next to no filter and a plain 5-reading median. Writes report.txt and grid.csv to --out.
"""
import argparse
import csv
import itertools
import math
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from multi.evaluate_series import SERIES_DIR, load_series, load_truth, spikes  # noqa: E402
from utils.landmark_timeseries import reconstruct_landmark_series_detail  # noqa: E402
from utils.paths import repo_path  # noqa: E402

DENSE = "data/annotations/angles/angle_series_bbval.csv"
SPARSE = "data/annotations/angles/angle_berrybox_val.csv"
GRID = {"wide": [3, 4, 6], "reject_deg": [30.0, 45.0, 60.0], "narrow": [1, 3, 5], "weighted": [False, True],
        "island_max": [0, 2, 4], "max_gap": [3, 5]}
TOLERANCE_DEG = 0.3
MIN_COVERAGE = 0.95
TREND_DEG = 10.0


def readouts_of(s):
    """Per-frame readout dicts of one exported seedling series (as the GUI feeds the reconstruction)."""
    out = []
    for raw_bio, gated, peak, mp in zip(s["raw_bio"], s["unreadable"], s["peak"], s["measurable_p"]):
        if gated:
            out.append({"theta": None, "gated": True})
        elif np.isfinite(raw_bio):
            out.append({"theta": 180.0 - raw_bio, "peak": peak if np.isfinite(peak) else 1.0,
                        "measurable_p": mp if np.isfinite(mp) else None})
        else:
            out.append(None)
    return out


def split_dense(keys):
    """(A, B): alternate within each rig, in seedling order."""
    a, b = set(), set()
    for series in sorted({k[0] for k in keys}):
        for i, k in enumerate(sorted(k for k in keys if k[0] == series)):
            (a if i % 2 == 0 else b).add(k)
    return a, b


def score(recon, series, truth, keys):
    """dict(mae, off20, coverage, skipped_read, direction) of reconstructed series on `keys`."""
    err, n_meas, n_cov, n_skip, n_skip_read = [], 0, 0, 0, 0
    agree = total = 0
    for key in keys:
        bio = recon[key]
        meas = []
        for i, img in enumerate(series[key]["img"]):
            t = truth.get((key[0], key[1], img))
            if t is None:
                continue
            if t["status"] == "measured":
                n_meas += 1
                if np.isfinite(bio[i]):
                    n_cov += 1
                    err.append(abs(bio[i] - float(t["bio_angle"])))
                meas.append((float(t["bio_angle"]), bio[i]))
            else:
                n_skip += 1
                n_skip_read += int(np.isfinite(bio[i]))
        for (h0, p0), (h1, p1) in zip(meas, meas[1:]):
            if abs(h1 - h0) >= TREND_DEG and np.isfinite(p0) and np.isfinite(p1):
                total += 1
                agree += int(np.sign(h1 - h0) == np.sign(p1 - p0))
    e = np.array(err)
    return {"n": len(e), "mae": float(e.mean()) if len(e) else math.nan,
            "off20": float(np.mean(e > 20)) if len(e) else math.nan,
            "coverage": n_cov / max(n_meas, 1), "skipped_read": n_skip_read / max(n_skip, 1),
            "direction": agree / max(total, 1)}


def spike_rate(recon):
    n = s = 0
    for bio in recon.values():
        b = np.asarray(bio, float)
        n += int(np.isfinite(b).sum())
        s += int(spikes(b).sum())
    return s / max(n, 1)


def reconstruct_all(series, cache, method="robust", **kw):
    return {k: np.asarray(reconstruct_landmark_series_detail(cache[k], method=method, **kw)[0], float) for k in series}


def complexity(p):
    return int(p["island_max"] > 0) + int(p["narrow"] > 1) + int(p["weighted"]) + int(p["max_gap"] > 3)


def _line(label, s, sp=None):
    return (f"{label:34s} n {s['n']:4d}  MAE {s['mae']:5.1f}  >20deg {s['off20']:4.0%}  coverage {s['coverage']:4.0%}  "
            f"read on skipped {s['skipped_read']:4.0%}  direction {s['direction']:4.0%}"
            + (f"  spikes {sp:5.1%}" if sp is not None else ""))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--series-dir", default=SERIES_DIR)
    p.add_argument("--dense", default=DENSE)
    p.add_argument("--sparse", default=SPARSE)
    p.add_argument("--out", default=None, help="Output folder (default <series-dir>/tuning)")
    args = p.parse_args(argv)
    series = load_series(args.series_dir)
    dense, sparse = load_truth(args.dense), load_truth(args.sparse)
    dense_keys = {(k[0], k[1]) for k in dense if (k[0], k[1]) in series}
    a_keys, b_keys = split_dense(dense_keys)
    sparse_keys = {(k[0], k[1]) for k in sparse if (k[0], k[1]) in series} - dense_keys
    cache = {k: readouts_of(s) for k, s in series.items()}

    base = reconstruct_all(series, cache, method="none")
    rows = []
    names = list(GRID)
    for values in itertools.product(*GRID.values()):
        params = dict(zip(names, values))
        recon = reconstruct_all(series, cache, **params)
        sa = score(recon, series, dense, a_keys)
        rows.append({**params, **{f"A_{k}": v for k, v in sa.items()}, "spikes": spike_rate(recon)})
    base_a = score(base, series, dense, a_keys)
    median5 = dict(wide=0, reject_deg=999.0, narrow=5, weighted=False, island_max=0, max_gap=3)
    median5_spikes = spike_rate(reconstruct_all(series, cache, **median5))
    ok = [r for r in rows if r["A_coverage"] >= MIN_COVERAGE and r["A_direction"] >= base_a["direction"] - 0.02
          and r["spikes"] <= median5_spikes]
    best = min(r["A_mae"] for r in ok)
    near = [r for r in ok if r["A_mae"] <= best + TOLERANCE_DEG]
    chosen = min(near, key=lambda r: (complexity(r), r["A_mae"]))
    params = {k: chosen[k] for k in names}

    out = Path(repo_path(args.out or os.path.join(args.series_dir, "tuning")))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "grid.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["A_mae"]))
    variants = [("no filter (today)", base), ("running median 5 only", reconstruct_all(series, cache, **median5)),
                ("chosen: " + ", ".join(f"{k}={v}" for k, v in params.items()), reconstruct_all(series, cache, **params))]
    lines = [f"dense A (choosing): {len(a_keys)} seedlings {sorted(a_keys)}",
             f"dense B (testing):  {len(b_keys)} seedlings {sorted(b_keys)}",
             f"sparse (testing):   {len(sparse_keys)} other seedlings", "",
             f"{len(rows)} settings; {len(ok)} keep coverage >= {MIN_COVERAGE:.0%}, direction and spikes <= plain "
             f"median 5 ({median5_spikes:.1%}); best A MAE {best:.2f}; "
             f"{len(near)} within {TOLERANCE_DEG} deg; simplest chosen", ""]
    for label, recon in variants:
        lines.append(label)
        sp = spike_rate(recon)
        lines.append(_line("  dense A", score(recon, series, dense, a_keys), sp))
        lines.append(_line("  dense B", score(recon, series, dense, b_keys)))
        lines.append(_line("  sparse (other plants)", score(recon, series, sparse, sparse_keys)))
        lines.append("")
    lines.append("spikes: share of finite values > 20 deg from the median of their 2+2 neighbours, all 73 plants.")
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
