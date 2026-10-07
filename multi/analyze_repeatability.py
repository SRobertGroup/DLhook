#!/usr/bin/env python
"""How repeatable are the hand-measured hook angles? Compares a blind re-measurement with the originals.

    python -m ui.angle_annotator --folder cropped_training_set --out angle_repeat.csv \
        --repeat-of angle_landmarks_train.csv,angle_ground_truth.csv --repeat-n 200
    python -m multi.analyze_repeatability [--original a.csv,b.csv] [--repeat angle_repeat.csv]

Several rounds (each paired against its own originals -- crop names repeat across crop folders, so
rounds from different folders must not be pooled before pairing):

    python -m multi.analyze_repeatability \
        --round "round 1 (training crops)=angle_landmarks_train.csv,angle_ground_truth.csv:angle_repeat.csv" \
        --round "round 2 (new seedlings)=angle_new.csv:angle_repeat_new.csv" --out-dir multi/results/repeatability_all

Frames are paired by (series, crop_id, frame). For the pairs measured both times it reports, in the
app's bio convention (180 = closed, decreasing as the hook opens):

  * the difference repeat - original: bias, SD, mean/median absolute difference, the share within
    5 / 10 / 20 deg, and the 95% limits of agreement (Bland-Altman) = bias +- 1.96 SD; the
    repeatability coefficient 1.96 * SD(diff) is the difference two measurements of the same frame
    are expected to stay within 95% of the time;
  * the same for theta (the angle between the axes, which has no overhook sign), the junction click
    position (px) and the agreement of the Overhook flag;
  * how the error grows with the angle (bins of the original bio angle);
  * measurability: frames measured once and skipped the other time.

A model compared with ONE hand measurement cannot do better than that measurement's own error: if the
human repeatability is about as large as the model-vs-human error, the model is at the noise floor.
The plot uses the angle the landmark validation scores, 180 - theta (overhook dropped: it is a few
degrees past closed, below the click noise), with one colour per round. Writes pairs.csv (with a
`round` column), summary.txt (per round, then pooled) and bland_altman.png into --out-dir
(default multi/results/repeatability).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = _REPO_ROOT / "multi" / "results" / "repeatability"
DEFAULT_ORIGINALS = ("angle_landmarks_train.csv", "angle_ground_truth.csv")
BINS = ((0, 60), (60, 120), (120, 160), (160, 181), (181, 400))


def load_rows(paths):
    """{(series, crop_id, frame): row} over the CSVs (the first file wins on a duplicate key)."""
    rows = {}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows.setdefault((row.get("series", ""), int(row["crop_id"]), row["frame"]), row)
    return rows


def _f(row, name):
    return float(row[name]) if row.get(name) not in (None, "") else float("nan")


def pair_up(originals, repeats):
    """(pairs, only_original_measured, only_repeat_measured): pairs are dicts for frames measured both
    times; the other two count frames measured in one pass and skipped in the other."""
    pairs, lost, gained = [], 0, 0
    for key, r in repeats.items():
        o = originals.get(key)
        if o is None:
            continue
        o_ok, r_ok = o.get("status") == "measured", r.get("status") == "measured"
        if o_ok and r_ok:
            pairs.append({
                "key": key,
                "orig_bio": _f(o, "bio_angle"), "rep_bio": _f(r, "bio_angle"),
                "orig_theta": _f(o, "theta"), "rep_theta": _f(r, "theta"),
                "orig_overhook": int(o.get("overhook") or 0), "rep_overhook": int(r.get("overhook") or 0),
                "junction_px": float(np.hypot(_f(o, "junction_x") - _f(r, "junction_x"),
                                              _f(o, "junction_y") - _f(r, "junction_y"))),
            })
        elif o_ok:
            lost += 1
        elif r_ok:
            gained += 1
    return pairs, lost, gained


def diff_stats(a, b):
    """Statistics of b - a (arrays of equal length): bias, SD, MAE, median, within-N shares, limits."""
    d = np.asarray(b, float) - np.asarray(a, float)
    d = d[~np.isnan(d)]
    if not len(d):
        return {"n": 0}
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    return {"n": len(d), "bias": float(d.mean()), "sd": sd, "mae": float(np.abs(d).mean()),
            "median": float(np.median(np.abs(d))), "p90": float(np.percentile(np.abs(d), 90)),
            "within5": float(np.mean(np.abs(d) <= 5)), "within10": float(np.mean(np.abs(d) <= 10)),
            "within20": float(np.mean(np.abs(d) <= 20)),
            "loa_low": float(d.mean() - 1.96 * sd), "loa_high": float(d.mean() + 1.96 * sd),
            "repeatability": 1.96 * sd}


def _line(name, s):
    if not s.get("n"):
        return f"{name}: no pairs"
    return (f"{name:14s} n={s['n']:3d}  bias {s['bias']:+6.1f}  SD {s['sd']:5.1f}  MAE {s['mae']:5.1f}  "
            f"median |d| {s['median']:5.1f}  90% of |d| <= {s['p90']:5.1f}  within 5/10/20 deg "
            f"{s['within5']:.0%}/{s['within10']:.0%}/{s['within20']:.0%}  limits of agreement "
            f"[{s['loa_low']:+.1f}, {s['loa_high']:+.1f}]")


def summarise(pairs, lost, gained):
    lines = []
    ob = [p["orig_bio"] for p in pairs]
    rb = [p["rep_bio"] for p in pairs]
    lines.append(f"{len(pairs)} frames measured both times; measured only the first time: {lost}; "
                 f"only the second time: {gained}  (measurability agreement "
                 f"{len(pairs) / max(len(pairs) + lost + gained, 1):.0%})")
    lines.append("")
    lines.append("repeat - original (degrees, bio convention):")
    lines.append(_line("bio angle", diff_stats(ob, rb)))
    lines.append(_line("theta", diff_stats([p["orig_theta"] for p in pairs], [p["rep_theta"] for p in pairs])))
    lines.append(_line("180 - theta", diff_stats([folded(p, "orig") for p in pairs], [folded(p, "rep") for p in pairs])))
    j = [p["junction_px"] for p in pairs]
    if j:
        lines.append(f"junction click: median distance {np.median(j):.1f} px, mean {np.mean(j):.1f}, "
                     f"90% within {np.percentile(j, 90):.1f} px")
    same = np.mean([p["orig_overhook"] == p["rep_overhook"] for p in pairs]) if pairs else float("nan")
    orig_pos = [p for p in pairs if p["orig_overhook"]]
    kept = np.mean([p["rep_overhook"] for p in orig_pos]) if orig_pos else float("nan")
    lines.append(f"overhook flag: same call {same:.0%}; of the {len(orig_pos)} frames first called overhooked, "
                 f"{kept:.0%} were called overhooked again")
    lines.append("")
    lines.append("by the original bio angle:")
    for lo, hi in BINS:
        sel = [p for p in pairs if lo <= p["orig_bio"] < hi]
        label = f"{lo:3d}-{min(hi, 360):3d}" if hi < 400 else f"> 180   "
        lines.append(_line(label, diff_stats([p["orig_bio"] for p in sel], [p["rep_bio"] for p in sel])))
    return "\n".join(lines)


def write_pairs(path, pairs):
    fields = ["round", "series", "crop_id", "frame", "orig_bio", "rep_bio", "diff", "orig_theta", "rep_theta",
              "orig_overhook", "rep_overhook", "junction_px"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for p in pairs:
            series, crop_id, frame = p["key"]
            w.writerow({"round": p.get("round", ""), "series": series, "crop_id": crop_id, "frame": frame,
                        "orig_bio": f"{p['orig_bio']:.3f}", "rep_bio": f"{p['rep_bio']:.3f}",
                        "diff": f"{p['rep_bio'] - p['orig_bio']:.3f}",
                        "orig_theta": f"{p['orig_theta']:.3f}", "rep_theta": f"{p['rep_theta']:.3f}",
                        "orig_overhook": p["orig_overhook"], "rep_overhook": p["rep_overhook"],
                        "junction_px": f"{p['junction_px']:.2f}"})


def folded(pair, which):
    """The scored angle 180 - theta (no overhook sign) of one side of a pair."""
    return 180.0 - pair[f"{which}_theta"]


ROUND_COLORS = ("tab:green", "tab:purple", "tab:brown", "tab:cyan")


def plot(path, pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = list(dict.fromkeys(p.get("round", "") for p in pairs))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    x = np.linspace(0, 185, 2)
    axes[0].fill_between(x, x - 10, x + 10, color="grey", alpha=0.15, lw=0, label="+-10 deg")
    axes[0].plot(x, x, color="grey", lw=0.8)
    for k, name in enumerate(rounds):
        sel = [p for p in pairs if p.get("round", "") == name]
        a = np.array([folded(p, "orig") for p in sel])
        b = np.array([folded(p, "rep") for p in sel])
        st = diff_stats(a, b)
        color = ROUND_COLORS[k % len(ROUND_COLORS)]
        label = (f"{name}: " if name else "") + f"n={st['n']}, median |d| {st.get('median', float('nan')):.1f}"
        axes[0].scatter(a, b, s=14, alpha=0.65, color=color, edgecolor="none", label=label)
        axes[1].scatter((a + b) / 2, b - a, s=14, alpha=0.65, color=color, edgecolor="none")
    ax = axes[0]
    ax.set_xlabel("first measurement, 180 - theta (deg)")
    ax.set_ylabel("blind repeat, 180 - theta (deg)")
    ax.set_xlim(0, 185)
    ax.set_ylim(0, 185)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"repeat vs original, n={len(pairs)}")
    a = np.array([folded(p, "orig") for p in pairs])
    b = np.array([folded(p, "rep") for p in pairs])
    s = diff_stats(a, b)
    ax = axes[1]
    for y, style in ((s["bias"], "-"), (s["loa_low"], "--"), (s["loa_high"], "--")):
        ax.axhline(y, color="tab:red", ls=style, lw=1)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("mean of the two measurements (deg)")
    ax.set_ylabel("repeat - original (deg)")
    ax.set_title(f"Bland-Altman (all rounds): bias {s['bias']:+.1f}, "
                 f"95% limits [{s['loa_low']:+.0f}, {s['loa_high']:+.0f}]")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def parse_round(text):
    """'label=orig1.csv,orig2.csv:repeat.csv' (label optional) -> (label, [originals], repeat)."""
    label, _, rest = text.rpartition("=")
    originals, sep, repeat = rest.rpartition(":")
    if not sep or not originals:
        raise SystemExit(f"--round {text!r}: expected 'label=orig.csv[,orig2.csv]:repeat.csv'")
    return label.strip(), [x.strip() for x in originals.split(",") if x.strip()], repeat.strip()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--original", default=",".join(str(_REPO_ROOT / n) for n in DEFAULT_ORIGINALS))
    p.add_argument("--repeat", default=str(_REPO_ROOT / "angle_repeat.csv"))
    p.add_argument("--round", action="append", default=None, metavar="LABEL=ORIG[,ORIG]:REPEAT",
                   help="one repeat round, paired against its own originals (repeatable; replaces "
                        "--original/--repeat)")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = p.parse_args(argv)
    rounds = ([parse_round(r) for r in args.round] if args.round else
              [("", [x.strip() for x in args.original.split(",") if x.strip()], args.repeat)])
    pairs, lost, gained, sections = [], 0, 0, []
    for label, originals, repeat in rounds:
        r_pairs, r_lost, r_gained = pair_up(load_rows(originals), load_rows([repeat]))
        for pr in r_pairs:
            pr["round"] = label
        pairs += r_pairs
        lost += r_lost
        gained += r_gained
        if len(rounds) > 1:
            sections.append(f"=== {label or repeat}" + "\n" + summarise(r_pairs, r_lost, r_gained))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = summarise(pairs, lost, gained)
    if sections:
        text = "\n\n".join(sections + ["=== all rounds pooled" + "\n" + text])
    print(text)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    if pairs:
        write_pairs(out / "pairs.csv", pairs)
        plot(out / "bland_altman.png", pairs)
        print(f"\nwrote summary.txt, pairs.csv and bland_altman.png to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
