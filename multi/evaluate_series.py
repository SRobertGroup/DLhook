"""How good are the angle time series? A fix-agnostic evaluation of the GUI's series on held-out plants.

    python -m multi.run_series --images "example_data/Camera_2" --boxes data/crop_boxes/crop_final_round_berry.json \\
        --series Camera_2 --out multi/results/series_eval/Camera_2 --no-figures      # once per series
    python -m multi.evaluate_series                                                 # BerryBox v2 val, all series

Seedlings are numbered as in the boxes file (run_series' seedlings.csv undoes the GUI's spatial
renumbering), so they match the annotation CSVs.

Reads the GUI exports written by multi/run_series.py (series.csv: every frame of every seedling, the
per-frame landmark reading `raw_angle`, the reconstructed `bio_angle`, `unreadable`, landmarks) and the
hand annotations of the same plants (angle CSV: measured frames with a bio angle, skipped frames that
an annotator judged unmeasurable; root CSV: collar; repeat CSV: a second blind measurement), and reports:

  coverage     hand measured / skipped against the pipeline's readable / unreadable frames
  accuracy     per-frame reading (180 - raw_angle) and reconstructed angle against the hand angle,
               next to the human repeat error; split by hook state and by what the reconstruction did
  noise        frame-to-frame steps and single-frame spikes, raw and reconstructed, against the
               change rate implied by the hand measurements
  gaps         unreadable / missing runs inside the series (first to last readable frame)
  landmarks    junction and collar steps between frames, junction / collar error against the clicks,
               and how often an angle spike coincides with a landmark jump
  shape        direction of change between consecutive hand-measured frames, hand vs reconstructed

Writes report.txt, per_frame.csv and one figure per seedling (hand points over the series) to --out.
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
TRUTH = "data/annotations/angles/angle_berrybox_val.csv"
ROOT_TRUTH = "data/annotations/angles/angle_berrybox_val_root.csv"
REPEAT = "data/annotations/angles/angle_repeat_bbval.csv"
SPIKE_DEG = 20.0          # a frame this far from the median of its 2+2 neighbours is a spike
NEIGHBOURS = 2
JUMP_PX = 10.0            # a landmark moving this far between consecutive frames is a jump
TREND_DEG = 10.0          # hand change needed to call a direction between two measured frames


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f


# ---- loading -------------------------------------------------------------------------------------
def load_series(series_dir):
    """{(series, seedling_id): dict of per-frame arrays in frame order} from every <series>/series.csv."""
    out = {}
    for sub in sorted(Path(repo_path(series_dir)).iterdir()):
        path = sub / "series.csv"
        if not path.exists():
            continue
        series = sub.name.replace("_", " ") if sub.name.startswith("Plate") else sub.name
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        # the GUI numbers boxes spatially; annotations keep the boxes file's order (seedlings.csv maps them)
        to_file = {}
        if (sub / "seedlings.csv").exists():
            with open(sub / "seedlings.csv", newline="", encoding="utf-8") as fh:
                to_file = {int(r["seedling_id"]): int(r["box_index"]) + 1 for r in csv.DictReader(fh)}
        by_seed = {}
        for r in rows:
            sid = int(r["seedling_id"])
            by_seed.setdefault(to_file.get(sid, sid), []).append(r)
        for seed, rs in by_seed.items():
            raw = np.array([_num(r["raw_angle"]) for r in rs])
            out[(series, seed)] = {
                "img": [r["img_name"] for r in rs],
                "raw_bio": 180.0 - raw,
                "rec": np.array([_num(r["bio_angle"]) for r in rs]),
                "unreadable": np.array([r["unreadable"] == "1" for r in rs]),
                "state": [r["state"] for r in rs],
                "series_edit": [r.get("series_edit", "") for r in rs],
                "flag": [r["temporal_flag"] for r in rs],
                "junction": np.array([[_num(r["junction_x"]), _num(r["junction_y"])] for r in rs]),
                "collar": np.array([[_num(r["collar_x"]), _num(r["collar_y"])] for r in rs]),
                "peak": np.array([_num(r["landmark_peak"]) for r in rs]),
                "measurable_p": np.array([_num(r["measurable_p"]) for r in rs]),
                "germ": next((int(r["germination_frame"]) for r in rs if r["germination_frame"] not in ("", None)), None),
            }
    return out


def load_truth(path=TRUTH):
    """{(series, seedling_id, img_name): row} of an angle annotation CSV (review sidecar applied)."""
    from utils.angle_review import read_rows
    if not path or not os.path.isfile(repo_path(path)):
        return {}
    return {(r["series"], int(r["seedling_id"]), r["img_name"]): r for r in read_rows(repo_path(path))}


def load_root_truth(path=ROOT_TRUTH, truth=None):
    """{(series, seedling_id, img_name): collar (x, y)} of a root CSV, keyed through the angle CSV's
    crop files (the root CSV has no img_name)."""
    out = {}
    if not path or not os.path.isfile(repo_path(path)):
        return out
    by_crop = {r["crop_file"]: k for k, r in (truth or {}).items()}
    with open(repo_path(path), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = by_crop.get(r.get("crop_file"))
            if key and r.get("collar_x"):
                out[key] = (_num(r["collar_x"]), _num(r["collar_y"]))
    return out


# ---- metrics (pure) ------------------------------------------------------------------------------
def spikes(values, threshold=SPIKE_DEG, k=NEIGHBOURS):
    """Bool array: finite values further than `threshold` from the median of up to k finite neighbours
    on each side (frames with fewer than 2 finite neighbours are never spikes)."""
    v = np.asarray(values, float)
    out = np.zeros(len(v), bool)
    for i in range(len(v)):
        if not np.isfinite(v[i]):
            continue
        nb = np.concatenate([v[max(0, i - k):i], v[i + 1:i + 1 + k]])
        nb = nb[np.isfinite(nb)]
        if len(nb) >= 2 and abs(v[i] - np.median(nb)) > threshold:
            out[i] = True
    return out


def steps(values):
    """|difference| between consecutive finite values that are adjacent frames."""
    v = np.asarray(values, float)
    d = np.abs(np.diff(v))
    return d[np.isfinite(d)]


def runs(mask):
    """Lengths of the True runs of a bool array."""
    out, n = [], 0
    for m in mask:
        if m:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def interior_gaps(readable):
    """Run lengths of non-readable frames strictly between the first and last readable frame."""
    idx = np.nonzero(readable)[0]
    if len(idx) < 2:
        return []
    return runs(~readable[idx[0]:idx[-1] + 1])


def err_stats(pred, truth):
    p, t = np.asarray(pred, float), np.asarray(truth, float)
    ok = np.isfinite(p) & np.isfinite(t)
    e = p[ok] - t[ok]
    if not len(e):
        return "n    0"
    return (f"n {len(e):4d}  MAE {np.mean(np.abs(e)):5.1f}  median {np.median(np.abs(e)):5.1f}  "
            f"bias {np.mean(e):+5.1f}  >20deg {np.mean(np.abs(e) > 20):4.0%}")


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else math.nan


# ---- evaluation ----------------------------------------------------------------------------------
def evaluate(series, truth, root_truth, repeat):
    lines, per_frame = [], []
    seeds = sorted(series)
    n_frames = sum(len(s["img"]) for s in series.values())
    lines.append(f"{len(seeds)} seedlings, {n_frames} frames ({len({k[0] for k in seeds})} series)")

    # join hand rows onto frames
    for key in seeds:
        s = series[key]
        readable = np.isfinite(s["raw_bio"]) & ~s["unreadable"]
        s["readable"] = readable
        s["spike_raw"], s["spike_rec"] = spikes(s["raw_bio"]), spikes(s["rec"])
        jstep = np.r_[np.nan, np.hypot(*np.diff(s["junction"], axis=0).T)]
        cstep = np.r_[np.nan, np.hypot(*np.diff(s["collar"], axis=0).T)]
        s["jstep"], s["cstep"] = jstep, cstep
        for i, img in enumerate(s["img"]):
            t = truth.get((key[0], key[1], img))
            if t is None:
                continue
            row = {"series": key[0], "seedling_id": key[1], "frame_index": i, "img_name": img,
                   "hand_status": t["status"], "hand_bio": _num(t["bio_angle"]) if t["status"] == "measured" else math.nan,
                   "raw_bio": s["raw_bio"][i], "rec_bio": s["rec"][i], "readable": bool(readable[i]),
                   "state": s["state"][i], "flag": s["flag"][i], "spike_raw": bool(s["spike_raw"][i]),
                   "series_edit": s["series_edit"][i],
                   "jstep": jstep[i], "cstep": cstep[i], "measurable_p": s["measurable_p"][i]}
            if t["status"] == "measured":
                hj = (_num(t["junction_x"]), _num(t["junction_y"]))
                row["junction_err"] = math.dist(hj, s["junction"][i]) if np.isfinite(s["junction"][i]).all() else math.nan
            c = root_truth.get((key[0], key[1], img))
            row["collar_err"] = (math.dist(c, s["collar"][i]) if c and np.isfinite(s["collar"][i]).all() else math.nan)
            per_frame.append(row)
    meas = [r for r in per_frame if r["hand_status"] == "measured"]
    skip = [r for r in per_frame if r["hand_status"] != "measured"]

    # coverage
    lines += ["", "== coverage (annotated frames)",
              f"hand measured {len(meas)}: pipeline readable {sum(r['readable'] for r in meas)} "
              f"({np.mean([r['readable'] for r in meas]):.0%}), reconstructed value "
              f"{sum(np.isfinite(r['rec_bio']) for r in meas)} ({np.mean([np.isfinite(r['rec_bio']) for r in meas]):.0%})",
              f"hand skipped  {len(skip)}: pipeline readable {sum(r['readable'] for r in skip)} "
              f"({np.mean([r['readable'] for r in skip]):.0%}), reconstructed value "
              f"{sum(np.isfinite(r['rec_bio']) for r in skip)} ({np.mean([np.isfinite(r['rec_bio']) for r in skip]):.0%})"]

    # accuracy
    hb = [r["hand_bio"] for r in meas]
    lines += ["", "== accuracy against the hand angle (measured frames)",
              f"per-frame reading      {err_stats([r['raw_bio'] for r in meas], hb)}",
              f"reconstructed series   {err_stats([r['rec_bio'] for r in meas], hb)}"]
    rep = []
    for key, r in repeat.items():
        t = truth.get(key)
        if t and t["status"] == "measured" and r["status"] == "measured":
            rep.append((_num(r["bio_angle"]), _num(t["bio_angle"])))
    if rep:
        lines.append(f"human repeat           {err_stats([a for a, _ in rep], [b for _, b in rep])}")
    for label, lo, hi in (("closed (hand >= 150)", 150, 999), ("opening (90-150)", 90, 150), ("open (< 90)", -999, 90)):
        sel = [r for r in meas if lo <= r["hand_bio"] < hi]
        lines.append(f"  {label:22s} reading {err_stats([r['raw_bio'] for r in sel], [r['hand_bio'] for r in sel])}")
        lines.append(f"  {'':22s} series  {err_stats([r['rec_bio'] for r in sel], [r['hand_bio'] for r in sel])}")
    changed = [r for r in meas if np.isfinite(r["raw_bio"]) and np.isfinite(r["rec_bio"]) and abs(r["rec_bio"] - r["raw_bio"]) > 5]
    if changed:
        better = sum(abs(r["rec_bio"] - r["hand_bio"]) < abs(r["raw_bio"] - r["hand_bio"]) for r in changed)
        lines.append(f"frames the reconstruction moved by > 5 deg: {len(changed)}; closer to the hand on {better}, "
                     f"further on {len(changed) - better}")
    filled = [r for r in meas if not np.isfinite(r["raw_bio"]) and np.isfinite(r["rec_bio"])]
    lines.append(f"measured frames with no reading but an interpolated value: {len(filled)}  "
                 f"{err_stats([r['rec_bio'] for r in filled], [r['hand_bio'] for r in filled])}")
    for edit in ("rejected", "interpolated"):
        sel = [r for r in meas if r.get("series_edit") == edit]
        if sel:
            lines.append(f"measured frames the series marked {edit}: {len(sel)}  reading "
                         f"{err_stats([r['raw_bio'] for r in sel], [r['hand_bio'] for r in sel])}  series "
                         f"{err_stats([r['rec_bio'] for r in sel], [r['hand_bio'] for r in sel])}")
    sp = [r for r in meas if r["spike_raw"]]
    lines.append(f"measured frames whose reading is a spike: {len(sp)}  reading "
                 f"{err_stats([r['raw_bio'] for r in sp], [r['hand_bio'] for r in sp])}")

    # noise
    raw_steps = np.concatenate([steps(series[k]["raw_bio"]) for k in seeds])
    rec_steps = np.concatenate([steps(series[k]["rec"]) for k in seeds])
    hand_rate = []
    by_seed = {}
    for r in meas:
        by_seed.setdefault((r["series"], r["seedling_id"]), []).append(r)
    for rs in by_seed.values():
        rs.sort(key=lambda r: r["frame_index"])
        for a, b in zip(rs, rs[1:]):
            hand_rate.append(abs(b["hand_bio"] - a["hand_bio"]) / max(1, b["frame_index"] - a["frame_index"]))
    hand_rate = np.array(hand_rate)
    n_raw = sum(int(np.isfinite(series[k]["raw_bio"]).sum()) for k in seeds)
    n_sr = sum(int(series[k]["spike_raw"].sum()) for k in seeds)
    n_rec = sum(int(np.isfinite(series[k]["rec"]).sum()) for k in seeds)
    n_sc = sum(int(series[k]["spike_rec"].sum()) for k in seeds)
    lines += ["", "== noise (all frames)",
              f"step between consecutive frames, reading: median {np.median(raw_steps):.1f}  p90 {_pct(raw_steps, 90):.1f}  "
              f"p99 {_pct(raw_steps, 99):.1f} deg",
              f"step between consecutive frames, series:  median {np.median(rec_steps):.1f}  p90 {_pct(rec_steps, 90):.1f}  "
              f"p99 {_pct(rec_steps, 99):.1f} deg",
              f"hand change per frame (consecutive measured frames): median {np.median(hand_rate):.1f}  "
              f"p90 {_pct(hand_rate, 90):.1f} deg",
              f"spikes (> {SPIKE_DEG:.0f} deg from the median of {NEIGHBOURS}+{NEIGHBOURS} neighbours): reading {n_sr}/{n_raw} "
              f"({n_sr / max(n_raw, 1):.1%}), series {n_sc}/{n_rec} ({n_sc / max(n_rec, 1):.1%})"]

    # gaps
    gaps, unread, pre_germ = [], [], []
    for k in seeds:
        s = series[k]
        unread.append(1 - s["readable"].mean())
        gaps += interior_gaps(s["readable"])
    gaps = np.array(gaps)
    lines += ["", "== gaps",
              f"unreadable or missing per seedling: median {np.median(unread):.0%}  p90 {_pct(unread, 90):.0%}",
              f"interior gaps (between first and last readable frame): {len(gaps)}, "
              f"length median {np.median(gaps) if len(gaps) else 0:.0f}, > 3 frames (not interpolated): "
              f"{int((gaps > 3).sum())}, frames inside them {int(gaps.sum())}"]
    first_meas, first_read = [], []
    for key, rs in by_seed.items():
        s = series[key]
        first_meas.append(min(r["frame_index"] for r in rs))
        idx = np.nonzero(s["readable"])[0]
        first_read.append(idx[0] if len(idx) else np.nan)
    d = np.array(first_read, float) - np.array(first_meas, float)
    lines.append(f"first readable frame minus first hand-measured frame (sampled frames): median {np.nanmedian(d):+.0f}, "
                 f"earlier in {np.mean(d < 0):.0%} of seedlings")

    # landmarks
    js = np.concatenate([series[k]["jstep"][series[k]["readable"]] for k in seeds])
    cs = np.concatenate([series[k]["cstep"] for k in seeds])
    js, cs = js[np.isfinite(js)], cs[np.isfinite(cs)]
    lines += ["", "== landmark stability",
              f"junction step between frames (readable): median {np.median(js):.1f}  p90 {_pct(js, 90):.1f}  "
              f"> {JUMP_PX:.0f} px {np.mean(js > JUMP_PX):.1%}",
              f"collar step between frames: median {np.median(cs):.1f}  p90 {_pct(cs, 90):.1f}  "
              f"> {JUMP_PX:.0f} px {np.mean(cs > JUMP_PX):.1%}"]
    je = np.array([r.get("junction_err", math.nan) for r in meas])
    ce = np.array([r["collar_err"] for r in per_frame])
    lines.append(f"junction vs click: median {np.nanmedian(je):.1f} px, > 10 px {np.nanmean(je[np.isfinite(je)] > 10):.0%}; "
                 f"collar vs click: median {np.nanmedian(ce):.1f} px, > 10 px {np.nanmean(ce[np.isfinite(ce)] > 10):.0%} "
                 f"(n {int(np.isfinite(ce).sum())})")
    co = tot = 0
    for k in seeds:
        s = series[k]
        sel = s["spike_raw"]
        tot += int(sel.sum())
        jj = np.nan_to_num(s["jstep"], nan=0) > JUMP_PX
        nxt = np.r_[jj[1:], False]
        co += int((sel & (jj | nxt)).sum())
    lines.append(f"reading spikes with a junction jump > {JUMP_PX:.0f} px into or out of the frame: {co}/{tot} "
                 f"({co / max(tot, 1):.0%})")
    flags = [f for k in seeds for f in series[k]["flag"] if f]
    lines.append(f"temporal flags: {len(flags)} frames ({len(flags) / n_frames:.1%})")

    # shape
    agree = total = 0
    for key, rs in by_seed.items():
        for a, b in zip(rs, rs[1:]):
            dh = b["hand_bio"] - a["hand_bio"]
            dr = b["rec_bio"] - a["rec_bio"]
            if abs(dh) >= TREND_DEG and np.isfinite(dr):
                total += 1
                agree += int(np.sign(dh) == np.sign(dr))
    per_seed_mae = sorted(((np.nanmean([abs(r["rec_bio"] - r["hand_bio"]) for r in rs]), key) for key, rs in by_seed.items()),
                          reverse=True)
    lines += ["", "== shape",
              f"direction of change between consecutive measured frames (hand change >= {TREND_DEG:.0f} deg): "
              f"series agrees on {agree}/{total} ({agree / max(total, 1):.0%})",
              "worst seedlings by series MAE: " + ", ".join(f"{k[0]} #{k[1]} {m:.0f}" for m, k in per_seed_mae[:8]
                                                            if np.isfinite(m))]
    return "\n".join(lines), per_frame


def plot_seedlings(series, per_frame, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    hand = {}
    for r in per_frame:
        hand.setdefault((r["series"], r["seedling_id"]), []).append(r)
    for key, s in series.items():
        x = np.arange(len(s["img"]))
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
        a1.plot(x, s["raw_bio"], ".", color="#9a9a9a", ms=4, label="per-frame reading")
        a1.plot(x, s["rec"], "-", color="#1f77b4", lw=1.6, label="series (GUI)")
        u = s["unreadable"]
        a1.plot(x[u], np.full(u.sum(), 5), "|", color="#d62728", ms=8, label="unreadable")
        sp = s["spike_raw"]
        a1.plot(x[sp], s["raw_bio"][sp], "o", mfc="none", color="#ff7f0e", ms=7, label="spike")
        rs = hand.get(key, [])
        m = [r for r in rs if r["hand_status"] == "measured"]
        a1.plot([r["frame_index"] for r in m], [r["hand_bio"] for r in m], "D", color="#2ca02c", ms=6, label="hand")
        k = [r for r in rs if r["hand_status"] != "measured"]
        a1.plot([r["frame_index"] for r in k], [195] * len(k), "x", color="#2ca02c", ms=6, label="hand: skipped")
        if s["germ"] is not None:
            a1.axvline(s["germ"], color="#8c564b", ls=":", lw=1)
        a1.axhline(180, color="#bbbbbb", lw=0.8)
        a1.set_ylim(0, 205)
        a1.set_ylabel("bio angle (deg)")
        a1.legend(fontsize=7, ncol=4, loc="lower left")
        a1.set_title(f"{key[0]}  seedling {key[1]}", fontsize=10)
        a2.plot(x, s["jstep"], "-", color="#ff3b30", lw=1, label="junction step")
        a2.plot(x, s["cstep"], "-", color="#7b2cbf", lw=1, label="collar step")
        a2.axhline(JUMP_PX, color="#bbbbbb", lw=0.8)
        a2.set_ylim(0, 60)
        a2.set_ylabel("px / frame")
        a2.set_xlabel("frame")
        a2.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{key[0].replace(' ', '_')}_{key[1]:02d}.png"), dpi=90)
        plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--series-dir", default=SERIES_DIR, help="Folder of run_series outputs (<series>/series.csv)")
    p.add_argument("--truth", default=TRUTH)
    p.add_argument("--root-truth", default=ROOT_TRUTH)
    p.add_argument("--repeat", default=REPEAT)
    p.add_argument("--out", default=None, help="Output folder (default <series-dir>/evaluation)")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)
    series = load_series(args.series_dir)
    if not series:
        p.error(f"no <series>/series.csv under {args.series_dir}; run multi.run_series first")
    truth = load_truth(args.truth)
    text, per_frame = evaluate(series, truth, load_root_truth(args.root_truth, truth), load_truth(args.repeat))
    out = Path(repo_path(args.out or os.path.join(args.series_dir, "evaluation")))
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text(text + "\n", encoding="utf-8")
    with open(out / "per_frame.csv", "w", newline="", encoding="utf-8") as fh:
        keys = sorted({k for r in per_frame for k in r}, key=lambda k: list(per_frame[0]).index(k) if k in per_frame[0] else 99)
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(per_frame)
    if not args.no_figures:
        plot_seedlings(series, per_frame, out / "figures")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
