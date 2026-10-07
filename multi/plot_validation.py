"""Validation figures: angle accuracy against hand measurements and human repeatability,
pooled kinematics aligned on germination, and germination-onset error.

Inputs (all written by earlier steps, defaults point at the new-seedling test set):

  --series-csv   angle_series.csv from `python -m multi.validate_angles --backends binary
                 --landmark-checkpoint ...` (every frame, each reading, the measured angle)
  --repeat       pairs.csv from `python -m multi.analyze_repeatability` (blind re-measurements)
  --germination  hand-marked onsets (ui/germination_annotator.py)
  --features + --germination-weights
                 the per-frame feature cache of multi/germination_detector_fit.py and the
                 saved detector, to recompute the learned onsets without a GPU

Writes to --out-dir:

  scatter.png             predicted vs measured (landmark head, ellipse pipeline) and repeat vs
                          first measurement (human), identity line and +-10 deg band
  error_cdf.png           share of frames within a given error, all three
  error_by_opening.png    error by how open the hook is (measured angle bins)
  per_series_error.png    mean error per series (rig / plate)
  pooled_kinematics.png   all seedlings aligned on their hand-marked germination frame: median
                          and interquartile band of each method, with the measured angles
  germination_onset.png   onset error of the learned detector and of the area rule

Angles are bio (180 = closed, decreasing as the hook opens); the measured angle is
180 - theta and readings are folded below 180 (overhook dropped, as in validate_angles).
The time axis is crop steps (frames in the crops folder), not hours.

    python -m multi.plot_validation [--out-dir docs/img/validation]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi.plot_kinematics import load_series_csv  # noqa: E402

RESULTS = REPO_ROOT / "multi" / "results"
METHODS = (("landmark", "Landmark head", "tab:blue"), ("recon", "Ellipse pipeline", "tab:orange"))
HUMAN = ("human", "Human repeat", "tab:green")
ROUND_COLORS = ("tab:green", "tab:purple", "tab:brown", "tab:cyan")
OPENING_BINS = ((150, 181, "closed\n150-180"), (120, 150, "120-150"), (90, 120, "90-120"),
                (45, 90, "45-90"), (-1, 45, "open\n< 45"))


def fold(values):
    v = np.asarray(values, float)
    return 180.0 - np.abs(v - 180.0)


def paired(series, method):
    """(measured, predicted, series name) over frames with both, readings folded below 180."""
    gt, pred, names = [], [], []
    for (name, _), e in series.items():
        ok = ~np.isnan(e["gt"]) & ~np.isnan(e[method])
        gt += list(fold(e["gt"][ok]))
        pred += list(fold(e[method][ok]))
        names += [name] * int(ok.sum())
    return np.array(gt), np.array(pred), names


def load_repeat(path):
    """(first, repeat, round label) per pair, angles 180 - theta, from analyze_repeatability's
    pairs.csv (the `round` column is empty for a single-round file)."""
    first, rep, rounds = [], [], []
    if path and Path(path).exists():
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    first.append(180.0 - float(row["orig_theta"]))
                    rep.append(180.0 - float(row["rep_theta"]))
                except (KeyError, ValueError):
                    continue
                rounds.append(row.get("round", ""))
    return np.array(first), np.array(rep), np.array(rounds, dtype=object)


def stats(gt, pred):
    e = np.abs(pred - gt)
    if not len(e):
        return "n=0"
    return f"n={len(e)}  MAE {e.mean():.1f}°  median {np.median(e):.1f}°  within 10° {np.mean(e <= 10):.0%}"


def plot_scatter(panels, out_path, plt):
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.9), squeeze=False)
    for ax, (title, color, gt, pred, xlabel, ylabel, *groups) in zip(axes[0], panels):
        x = np.linspace(0, 185, 2)
        ax.fill_between(x, x - 10, x + 10, color="grey", alpha=0.15, lw=0, label="±10°")
        ax.plot(x, x, color="grey", lw=0.8)
        if groups and len(set(groups[0])) > 1:          # one colour per group (repeat rounds)
            for k, name in enumerate(dict.fromkeys(groups[0])):
                sel = groups[0] == name
                ax.scatter(gt[sel], pred[sel], s=12, alpha=0.6, edgecolor="none",
                           color=ROUND_COLORS[k % len(ROUND_COLORS)], label=f"{name} (n={int(sel.sum())})")
        else:
            ax.scatter(gt, pred, s=12, color=color, alpha=0.6, edgecolor="none")
        ax.set_xlim(0, 185)
        ax.set_ylim(0, 185)
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n{stats(gt, pred)}", fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_cdf(curves, out_path, plt):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    grid = np.linspace(0, 90, 181)
    for title, color, err in curves:
        if len(err):
            ax.plot(grid, [np.mean(err <= g) for g in grid], color=color, lw=2,
                    label=f"{title} (median {np.median(err):.1f}°)")
    for g in (10, 20, 30):
        ax.axvline(g, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("absolute angle error (°)")
    ax.set_ylabel("share of frames with error ≤ x")
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_by_opening(series, out_path, plt):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.38
    for k, (method, title, color) in enumerate(METHODS):
        gt, pred, _ = paired(series, method)
        err = np.abs(pred - gt)
        data = [err[(gt >= lo) & (gt < hi)] for lo, hi, _ in OPENING_BINS]
        pos = np.arange(len(OPENING_BINS)) + (k - 0.5) * width
        bp = ax.boxplot([d if len(d) else [np.nan] for d in data], positions=pos, widths=width * 0.9,
                        patch_artist=True, showfliers=False, medianprops={"color": "black"})
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.plot(pos, [d.mean() if len(d) else np.nan for d in data], "D", color=color, mec="black", ms=5,
                label=f"{title} (diamond = mean)")
        if k == 0:
            for p, d in zip(pos, data):
                ax.text(p + width / 2, -6, f"n={len(d)}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(OPENING_BINS)))
    ax.set_xticklabels([label for _, _, label in OPENING_BINS])
    ax.set_xlabel("measured hook angle (°, 180 = closed)")
    ax.set_ylabel("absolute angle error (°)")
    ax.set_ylim(-10, None)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_per_series(series, out_path, plt):
    per = OrderedDict()
    for method, _, _ in METHODS:
        gt, pred, names = paired(series, method)
        for name, e in zip(names, np.abs(pred - gt)):
            per.setdefault(name, {}).setdefault(method, []).append(e)
    names = sorted(per, key=lambda n: -np.mean(per[n].get("landmark", [0])))
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(names) + 1.2))
    y = np.arange(len(names))
    for k, (method, title, color) in enumerate(METHODS):
        vals = [np.mean(per[n][method]) if per[n].get(method) else np.nan for n in names]
        ax.barh(y + (k - 0.5) * 0.4, vals, height=0.4, color=color, alpha=0.8, label=title)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n[:28]} (n={len(per[n].get('landmark', []))})" for n in names], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("mean absolute angle error (°)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def aligned(series, onsets, method):
    """{offset from germination: [values]} for one method (or 'gt'), folded."""
    out = {}
    for key, e in series.items():
        if onsets.get(key) is None:
            continue
        rel = e["idx"] - onsets[key]
        vals = fold(e[method])
        for r, v in zip(rel, vals):
            if r >= 0 and not np.isnan(v):          # no hook before the radicle emerges
                out.setdefault(int(r), []).append(v)
    return out


def plot_pooled(series, onsets, out_path, plt, min_seedlings=5):
    measured = aligned(series, onsets, "gt")
    n_seedlings = sum(1 for k in series if onsets.get(k) is not None)
    fig, axes = plt.subplots(1, len(METHODS), figsize=(6.2 * len(METHODS), 4.4), squeeze=False, sharey=True)
    for ax, (method, title, color) in zip(axes[0], METHODS):
        vals = aligned(series, onsets, method)
        xs = sorted(x for x, v in vals.items() if len(v) >= min_seedlings)
        if xs:
            # NaN rows break the curve where too few seedlings have a reading, instead of bridging the gap
            grid = list(range(xs[0], xs[-1] + 1))
            q = np.array([np.percentile(vals[x], [25, 50, 75]) if x in xs else [np.nan] * 3 for x in grid])
            xs = grid
            ax.fill_between(xs, q[:, 0], q[:, 2], color=color, alpha=0.2, lw=0, label=f"{title}, IQR")
            ax.plot(xs, q[:, 1], "-o", color=color, ms=3, lw=2, label=f"{title}, median")
        mx = [x for x, v in measured.items() for _ in v]
        my = [y for v in measured.values() for y in v]
        ax.scatter(np.array(mx) + np.random.RandomState(0).uniform(-0.15, 0.15, len(mx)), my, s=9,
                   color="black", alpha=0.35, edgecolor="none", label="measured (each frame)")
        mxs = sorted(x for x, v in measured.items() if len(v) >= min_seedlings)
        if mxs:
            ax.plot(mxs, [np.median(measured[x]) for x in mxs], "--", color="black", lw=1.5,
                    label="measured, median")
        ax.axhline(180, color="grey", lw=0.7, ls=":")
        ax.axvline(0, color="tab:green", lw=0.8, ls="--")
        ax.set_xlabel("crop steps since germination (hand-marked onset = 0)")
        ax.set_title(f"{title} vs measurements ({n_seedlings} seedlings)", fontsize=10)
        ax.set_ylim(-5, 190)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(alpha=0.2)
    axes[0][0].set_ylabel("hook angle (°, 180 = closed)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def learned_onsets(features_csv, weights_json):
    """{(series, crop_id): (learned onset or None, area-rule onset or None, n_frames)} from the feature cache."""
    from multi.germination_detector_fit import threshold_onset
    from utils.germination_learned import FEATURE_NAMES, GerminationModel, best_onset

    model = GerminationModel.load(weights_json)
    rows = OrderedDict()
    with open(features_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.setdefault((row["series"], int(row["crop_id"])), []).append(row)
    out = {}
    for key, rs in rows.items():
        rs.sort(key=lambda r: int(r["frame_index"]))
        feats = np.array([[float(r[n]) for n in FEATURE_NAMES] for r in rs])
        k = best_onset(model.visible_probs(feats))
        area = np.expm1(feats[:, FEATURE_NAMES.index("r_log_n50")])
        out[key] = (k if k < len(rs) else None, threshold_onset(area, 5), len(rs))
    return out


def plot_germination(truth, detected, out_path, plt):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    offsets = np.arange(-3, 4)
    arms = (("learned detector", "tab:purple", 0), ("area rule (≥ 5 px radicle)", "tab:gray", 1))
    for k, (label, color, col) in enumerate(arms):
        errs, missed = [], 0
        for key, onset in truth.items():
            d = detected.get(key, (None, None, 0))[col]
            if d is None:
                missed += 1
            else:
                errs.append(int(np.clip(d - onset, offsets[0], offsets[-1])))
        n = len(errs) + missed
        counts = [np.mean(np.array(errs) == o) * len(errs) / n if n else 0 for o in offsets]
        exact = np.mean(np.array(errs) == 0) * len(errs) / n if n else 0
        ax.bar(offsets + (k - 0.5) * 0.4, counts, width=0.4, color=color, alpha=0.85,
               label=f"{label}: exact {exact:.0%}, not found {missed}/{n}")
    ax.set_xticks(offsets)
    ax.set_xticklabels(["≤-3"] + [str(o) for o in offsets[1:-1]] + ["≥+3"])
    ax.set_xlabel("detected - hand-marked onset (crop steps)")
    ax.set_ylabel("share of seedlings")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--series-csv", default=str(RESULTS / "new_seedlings" / "angles_v6_tta" / "angle_series.csv"))
    p.add_argument("--repeat", default=str(RESULTS / "repeatability_all" / "pairs.csv"),
                   help="pairs.csv of multi.analyze_repeatability (all rounds; one colour per round)")
    p.add_argument("--germination", default=str(REPO_ROOT / "germination_new.csv"))
    p.add_argument("--features", default=str(RESULTS / "germination_validation" / "frame_features_cropped_new_set.csv"))
    p.add_argument("--germination-weights", default=str(REPO_ROOT / "weights" / "germination_detector.json"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "docs" / "img" / "validation"))
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from multi.validate_germination import load_truth

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    series = load_series_csv(args.series_csv)
    first, rep, rounds = load_repeat(args.repeat)

    panels = [(title, color, *paired(series, m)[:2], "measured angle (°)", "predicted angle (°)")
              for m, title, color in METHODS]
    if len(first):
        n_rounds = len(set(rounds))
        title = HUMAN[1] + (f" ({n_rounds} blind rounds)" if n_rounds > 1 else " (blind)")
        panels.append((title, HUMAN[2], first, rep, "first measurement (°)", "blind repeat (°)", rounds))
    plot_scatter(panels, out / "scatter.png", plt)
    plot_cdf([(t, c, np.abs(p[1] - p[0])) for t, c, *p in panels], out / "error_cdf.png", plt)
    plot_by_opening(series, out / "error_by_opening.png", plt)
    plot_per_series(series, out / "per_series_error.png", plt)

    truth = {k: v["onset_index"] for k, v in load_truth(args.germination).items()
             if v["status"] == "found" and v["onset_index"] is not None} if Path(args.germination).exists() else {}
    plot_pooled(series, truth, out / "pooled_kinematics.png", plt)
    written = ["scatter", "error_cdf", "error_by_opening", "per_series_error", "pooled_kinematics"]
    if truth and Path(args.features).exists() and Path(args.germination_weights).exists():
        plot_germination(truth, learned_onsets(args.features, args.germination_weights),
                         out / "germination_onset.png", plt)
        written.append("germination_onset")

    for title, _, g, p, *_ in panels:
        print(f"{title:18s} {stats(g, p)}")
    print(f"wrote {', '.join(w + '.png' for w in written)} to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
