"""Kinematics plots of headless series runs (multi/run_series.py), as the GUI's kinematics window draws them.

    python -m multi.plot_series_kinematics                                   # every series under multi/results/series_eval
    python -m multi.plot_series_kinematics --series-dir multi/results/series_eval --truth data/annotations/angles/angle_berrybox_val.csv

Per series (<series>/series.csv) it writes to <series-dir>/kinematics/:

  <series>_kinematics.png   every seedling's bio angle against time since its own germination (hours;
                            frame index when the series has no timestamps), points coloured by state,
                            one line colour per seedling, a black ring on temporally flagged frames,
                            ungerminated seedlings dashed on raw elapsed time -- ui/kinematics_window.py
  <series>_panels.png       one panel per seedling on the same axis, with the per-frame reading (grey;
                            an orange ring when the robust series filter rejected it),
                            the series (line), unreadable frames (red ticks) and, with --truth, the hand
                            angles (green diamonds) and frames judged unmeasurable (green crosses)

Seedlings are numbered as in the boxes file (run_series' seedlings.csv undoes the GUI's spatial
renumbering), so panel titles match the annotation CSVs.
"""
import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.paths import repo_path  # noqa: E402

SERIES_DIR = "multi/results/series_eval"
# as ui/kinematics_window.py (not imported: it selects the TkAgg backend)
STATE_COLORS = {"Closed": "tab:blue", "Opening": "tab:orange", "Manual": "tab:purple"}
UNKNOWN_STATE_COLOR = "0.5"
SEEDLING_LINE_COLORS = ["tab:green", "tab:red", "tab:brown", "tab:pink", "tab:olive", "tab:cyan", "tab:gray", "black"]
CURVE_ALPHA, POINT_ALPHA = 0.35, 0.6


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def load_runs(series_dir):
    """[(series name, folder, {file seedling id: [rows in frame order]})]."""
    out = []
    for sub in sorted(Path(repo_path(series_dir)).iterdir()):
        if not (sub / "series.csv").exists():
            continue
        name = sub.name.replace("_", " ") if sub.name.startswith("Plate") else sub.name
        to_file = {}
        if (sub / "seedlings.csv").exists():
            with open(sub / "seedlings.csv", newline="", encoding="utf-8") as fh:
                to_file = {int(r["seedling_id"]): int(r["box_index"]) + 1 for r in csv.DictReader(fh)}
        seeds = {}
        with open(sub / "series.csv", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                sid = int(r["seedling_id"])
                seeds.setdefault(to_file.get(sid, sid), []).append(r)
        out.append((name, sub.name, seeds))
    return out


def time_axis(rows):
    """(x per frame, uses_time): elapsed hours when every frame has a timestamp, else frame index."""
    t = np.array([_num(r["time_elapsed_min"]) for r in rows])
    if len(t) and np.isfinite(t).all():
        return t / 60.0, True
    return np.arange(len(rows), dtype=float), False


def aligned(rows):
    """(x, germination frame or None): x shifted to the seedling's germination, as the GUI does."""
    x, _ = time_axis(rows)
    germ = next((int(r["germination_frame"]) for r in rows if r["germination_frame"] not in ("", None)), None)
    if germ is None or germ >= len(x):
        return x, None
    return x - x[germ], germ


def load_truth(path):
    from utils.angle_review import read_rows
    if not path or not os.path.exists(repo_path(path)):
        return {}
    return {(r["series"], int(r["seedling_id"]), r["img_name"]): r for r in read_rows(repo_path(path))}


def plot_kinematics(name, seeds, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(10, 5.5))
    uses_time = False
    states, flagged_any = [], False
    for sid, rows in sorted(seeds.items()):
        x, germ = aligned(rows)
        uses_time = time_axis(rows)[1]
        y = np.array([_num(r["bio_angle"]) for r in rows])
        ok = np.isfinite(y)
        if not ok.any():
            continue
        st = [r["state"] for r in rows]
        colors = [STATE_COLORS.get(st[i], UNKNOWN_STATE_COLOR) for i in np.nonzero(ok)[0]]
        flag = np.array([bool(r["temporal_flag"]) and r["state"] != "Manual" for r in rows])[ok]
        color = SEEDLING_LINE_COLORS[(sid - 1) % len(SEEDLING_LINE_COLORS)]
        ax.plot(x[ok], y[ok], "--" if germ is None else "-", color=color, lw=1, alpha=CURVE_ALPHA, zorder=1)
        ax.scatter(x[ok], y[ok], c=colors, s=14, alpha=POINT_ALPHA, zorder=2,
                   edgecolors="black", linewidths=np.where(flag, 1.2, 0.0))
        flagged_any = flagged_any or flag.any()
        for s in set(st[i] for i in np.nonzero(ok)[0]):
            label = s if s in STATE_COLORS else "Unclassified"
            if label not in states:
                states.append(label)
    ax.axhline(180, color="0.85", lw=1, zorder=0)
    ax.set_xlabel("Time since germination (h)" if uses_time else "Frame index since germination")
    ax.set_ylabel("Angle (deg)")
    ax.set_title(f"{name}: {len(seeds)} seedlings (dashed: no germination frame, raw elapsed time)", fontsize=10)
    handles = [Line2D([], [], marker="o", ls="none", color=STATE_COLORS.get(s, UNKNOWN_STATE_COLOR), label=s) for s in states]
    if flagged_any:
        handles.append(Line2D([], [], marker="o", ls="none", mfc="white", mec="black", label="Temporal flag"))
    ax.legend(handles=handles, title="State", fontsize=8, title_fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_panels(name, seeds, truth, out_path, ncols=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids = sorted(seeds)
    nrows = math.ceil(len(ids) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows), sharey=True, squeeze=False)
    uses_time = False
    for ax, sid in zip(axes.flat, ids):
        rows = seeds[sid]
        x, germ = aligned(rows)
        uses_time = time_axis(rows)[1]
        raw = 180.0 - np.array([_num(r["raw_angle"]) for r in rows])
        y = np.array([_num(r["bio_angle"]) for r in rows])
        unread = np.array([r["unreadable"] == "1" for r in rows])
        ax.plot(x, raw, ".", color="0.65", ms=2.5)
        ax.plot(x, y, "-", color="tab:blue", lw=1.2)
        ax.plot(x[unread], np.full(unread.sum(), 4), "|", color="tab:red", ms=5)
        rej = np.array([r.get("series_edit") == "rejected" for r in rows])
        if rej.any():
            ax.plot(x[rej], raw[rej], "o", mfc="none", color="tab:orange", ms=4)
        hm = [(x[i], _num(t["bio_angle"])) for i, r in enumerate(rows)
              for t in [truth.get((name, sid, r["img_name"]))] if t and t["status"] == "measured"]
        hs = [x[i] for i, r in enumerate(rows) for t in [truth.get((name, sid, r["img_name"]))] if t and t["status"] != "measured"]
        if hm:
            ax.plot(*zip(*hm), "D", color="tab:green", ms=4)
        if hs:
            ax.plot(hs, [196] * len(hs), "x", color="tab:green", ms=4)
        if germ is not None:
            ax.axvline(0, color="tab:brown", ls=":", lw=0.8)
        ax.axhline(180, color="0.85", lw=0.8)
        ax.set_ylim(0, 205)
        ax.set_title(f"#{sid}" + ("" if germ is not None else " (no germination)"), fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(ids):]:
        ax.axis("off")
    fig.supxlabel("Time since germination (h)" if uses_time else "Frame index since germination", fontsize=9)
    fig.supylabel("Angle (deg)", fontsize=9)
    fig.suptitle(f"{name}: grey per-frame reading (orange ring: rejected by the series filter), blue series, "
                 "red unreadable, green hand (x: unmeasurable)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--series-dir", default=SERIES_DIR)
    p.add_argument("--truth", default="data/annotations/angles/angle_berrybox_val.csv",
                   help="Angle annotation CSV drawn on the panels ('' for none)")
    p.add_argument("--out", default=None, help="Output folder (default <series-dir>/kinematics)")
    args = p.parse_args(argv)
    runs = load_runs(args.series_dir)
    if not runs:
        p.error(f"no <series>/series.csv under {args.series_dir}; run multi.run_series first")
    truth = load_truth(args.truth)
    out = Path(repo_path(args.out or os.path.join(args.series_dir, "kinematics")))
    out.mkdir(parents=True, exist_ok=True)
    for name, folder, seeds in runs:
        plot_kinematics(name, seeds, out / f"{folder}_kinematics.png")
        plot_panels(name, seeds, truth, out / f"{folder}_panels.png")
        print(f"{name}: {len(seeds)} seedlings -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
