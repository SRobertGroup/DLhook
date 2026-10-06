"""Angle kinematics against the hand measurements, for the landmark head and the
ellipse pipeline.

Reads multi/results/angle_validation/angle_series.csv (written by
`python -m multi.validate_angles --landmark-checkpoint ...`: every frame of every
measured seedling, in time order, with each reading and the hand-measured angle
where there is one) and writes, to --out-dir:

  scatter.png               predicted vs measured, one panel per method
  kinematics_<series>.png   one row per seedling, left = landmark vs measurements,
                            right = ellipse pipeline vs measurements

Angles are in the bio convention (180 = closed, decreasing as the hook opens,
> 180 overhooked). The x axis is the frame's position in the series (the crops
folder has no timestamps), not hours.

    python -m multi.plot_kinematics [--series-csv ...] [--out-dir ...] [--min-measured 2]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "multi" / "results" / "angle_validation" / "angle_series.csv"
DEFAULT_OUT = REPO_ROOT / "multi" / "results" / "angle_validation" / "kinematics"

METHODS = (("landmark", "Landmark head", "tab:blue"), ("recon", "Ellipse pipeline", "tab:orange"))


def _num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def load_series_csv(path):
    """{(series, crop_id): {"idx", "gt", "gt_overhook", "<method>": ndarray}} with NaN for blanks."""
    grouped = OrderedDict()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            grouped.setdefault((row["series"], int(row["crop_id"])), []).append(row)
    out = OrderedDict()
    for key, rows in grouped.items():
        rows.sort(key=lambda r: int(r["frame_idx"]))
        entry = {"idx": np.array([int(r["frame_idx"]) for r in rows], dtype=float),
                 "gt": np.array([_num(r["gt"]) for r in rows]),
                 "gt_overhook": np.array([_num(r["gt_overhook"]) for r in rows])}
        for method, _, _ in METHODS:
            entry[method] = np.array([_num(r.get(method)) for r in rows])
        out[key] = entry
    return out


def method_stats(series, method):
    """(n, MAE, median error) of `method` over the frames that have both a reading and a measurement."""
    errors = []
    for entry in series.values():
        ok = ~np.isnan(entry["gt"]) & ~np.isnan(entry[method])
        errors += list(np.abs(entry[method][ok] - entry["gt"][ok]))
    if not errors:
        return 0, float("nan"), float("nan")
    return len(errors), float(np.mean(errors)), float(np.median(errors))


def _safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:60]


def plot_scatter(series, out_path, plt):
    fig, axes = plt.subplots(1, len(METHODS), figsize=(5.2 * len(METHODS), 5), squeeze=False)
    for ax, (method, title, color) in zip(axes[0], METHODS):
        pred, truth, over = [], [], []
        for entry in series.values():
            ok = ~np.isnan(entry["gt"]) & ~np.isnan(entry[method])
            pred += list(entry[method][ok])
            truth += list(entry["gt"][ok])
            over += list(entry["gt_overhook"][ok] == 1)
        pred, truth, over = np.array(pred), np.array(truth), np.array(over, dtype=bool)
        ax.plot([0, 240], [0, 240], color="grey", lw=0.8)
        ax.scatter(truth[~over], pred[~over], s=14, color=color, alpha=0.7, label="not overhooked")
        ax.scatter(truth[over], pred[over], s=22, color="tab:red", marker="D", alpha=0.8, label="overhooked (measured)")
        n, mae, med = method_stats(series, method)
        ax.set_title(f"{title}: n={n}, MAE {mae:.1f}°, median {med:.1f}°")
        ax.set_xlabel("measured angle (°)")
        ax.set_ylabel("predicted angle (°)")
        ax.set_xlim(0, 240)
        ax.set_ylim(0, 240)
        ax.set_aspect("equal")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_series(name, seedlings, out_path, plt, min_measured):
    rows = [(cid, e) for cid, e in seedlings if int(np.sum(~np.isnan(e["gt"]))) >= min_measured]
    if not rows:
        return False
    fig, axes = plt.subplots(len(rows), len(METHODS), figsize=(5.6 * len(METHODS), 2.3 * len(rows)),
                             squeeze=False, sharex="row")
    for r, (cid, entry) in enumerate(rows):
        measured = ~np.isnan(entry["gt"])
        over = measured & (entry["gt_overhook"] == 1)
        for c, (method, title, color) in enumerate(METHODS):
            ax = axes[r][c]
            ax.axhline(180, color="grey", lw=0.7, ls="--")
            ax.plot(entry["idx"], entry[method], "-o", ms=2.5, lw=1.1, color=color, label=title)
            ax.plot(entry["idx"][measured], entry["gt"][measured], "o", ms=6, mfc="none", mec="black", mew=1.2,
                    label="measured")
            ax.plot(entry["idx"][over], entry["gt"][over], "D", ms=5, color="tab:red", label="measured, overhook")
            ok = measured & ~np.isnan(entry[method])
            err = np.abs(entry[method][ok] - entry["gt"][ok])
            ax.set_ylim(-10, 250)
            ax.set_ylabel(f"seedling {cid + 1}\nangle (°)")
            ax.set_title(f"{title}  (MAE {np.mean(err):.1f}° on {int(ok.sum())} frames)" if ok.any()
                         else f"{title}  (no reading on a measured frame)", fontsize=9)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="lower left")
    for ax in axes[-1]:
        ax.set_xlabel("frame in series")
    fig.suptitle(name, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--series-csv", default=str(DEFAULT_CSV))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--min-measured", type=int, default=2,
                   help="plot a seedling only when it has at least this many measured frames (default 2)")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not Path(args.series_csv).exists():
        raise SystemExit(f"{args.series_csv} not found -- run `python -m multi.validate_angles "
                         f"--landmark-checkpoint <best.pt>` first")
    series = load_series_csv(args.series_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_scatter(series, out_dir / "scatter.png", plt)
    by_series = OrderedDict()
    for (name, cid), entry in series.items():
        by_series.setdefault(name, []).append((cid, entry))
    written = 0
    for name, seedlings in by_series.items():
        if plot_series(name, seedlings, out_dir / f"kinematics_{_safe_name(name)}.png", plt, args.min_measured):
            written += 1
    for method, title, _ in METHODS:
        n, mae, med = method_stats(series, method)
        print(f"{title:18s} n={n:4d}  MAE {mae:5.1f}  median {med:5.1f}")
    print(f"wrote scatter.png and {written} per-series figure(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
