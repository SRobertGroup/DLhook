#!/usr/bin/env python
"""Compare the pipeline's hook angles with hand-measured ground truth.

Ground truth comes from the annotation tool (`python -m ui.angle_annotator`,
angle_ground_truth.csv). For every seedling that has at least one measured
frame, this runs the REAL pipeline -- the configured segmentation backend, then
Gui.process_single_frame and Gui._reconstruct_series_for_crop, with no Tk window
-- over that seedling's whole crop series (all its crops in cropped_training_set/
manifest.csv, in manifest = time order), and compares the result on the
annotated frames.

Three readings are compared per frame, because the pipeline has two stages:
    raw        the per-frame geometric value as AngleCalculator emits it
    raw_flip   180 - raw -- what Gui._reconstruct_series_for_crop feeds the
               reconstruction (it treats `raw` as 0 = closed)
    recon      the reconstructed bio angle the export writes (`bio_angle`)
Ground truth is in the bio convention (180 = closed, decreasing as the hook opens). The Overhook
flag is DROPPED by default (it is a few degrees past 180, below the repeatability of the hand clicks):
truth becomes 180 - theta and every reading is folded the same way (180 - |value - 180|), so an
ellipse reading of 200 and a truth of 160 agree. --keep-overhook restores the raw convention
(180 + theta for flagged frames). Whichever variant matches it tells you which convention the
pipeline really emits.

Assumptions you should know about:
* The training crops carry no seed (start) point. A proxy is used: horizontally
  centred, at the lower padding boundary of the crop (the crop is the click
  extent padded 10% top and bottom, so the seed coat sits ~1/12 of the height
  above the bottom edge). It only affects which part of the mask is cut away
  below the seed line. Change it with --seed-pad.
* Reconstruction runs over the every-5th-frame crop series, not the full time
  series, and without capture times.
* For the multiclass backend, frames outside the validation split were seen in
  training. Run the annotation tool with --split val (as recommended) and the
  annotated frames themselves are held out.

A landmark checkpoint (multi/configs/training_landmarks.yaml) can be scored the
same way with --landmark-checkpoint: its angle comes from the junction heatmap
and direction fields, not from ellipse fits, and is reported as `landmark`
(per frame) and `landmark_recon` (after the same reconstruction). Score it on the
held-out ground truth only -- never on frames you annotated for training.

Usage:
    python -m multi.validate_angles
    python -m multi.validate_angles --backends binary,multiclass --truth angle_ground_truth.csv
    python -m multi.validate_angles --backends multiclass --landmark-checkpoint multi/results/models_landmarks/best.pt
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import shutil
import sys
import tempfile
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ("raw", "raw_flip", "recon", "landmark", "landmark_recon", "landmark_temporal")


# --- pure helpers (unit-tested) ----------------------------------------------

FOLD_VARIANTS = tuple(v for v in VARIANTS if v != "raw")      # `raw` is the 0 = closed convention


def fold_angle(value):
    """Overhook dropped: reflect a bio angle above 180 back below it (180 - |value - 180|)."""
    return None if value is None or value != value else 180.0 - abs(value - 180.0)


def load_truth(path, fold=True):
    """Measured rows of an annotation CSV, with typed fields. With fold (default) the Overhook flag
    is dropped: gt = 180 - theta (theta is stored for every frame) and gt_overhook is 0."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["status"] != "measured":
                continue
            bio = float(row["bio_angle"])
            rows.append({
                "series": row["series"], "crop_id": int(row["crop_id"]), "frame": row["frame"],
                "crop_file": row["crop_file"], "gt": fold_angle(bio) if fold else bio,
                "gt_overhook": 0 if fold else int(row["overhook"] or 0),
            })
    return rows


def fold_predictions(pred):
    """fold_angle over every numeric reading of a {key: {reading: value}} prediction dict."""
    return {key: {k: (fold_angle(v) if k in FOLD_VARIANTS else v) for k, v in entry.items()}
            for key, entry in pred.items()}


def load_series(manifest_path, truth_rows):
    """{(series, crop_id): [crop_file, ...]} for the seedlings in `truth_rows`,
    each series' crops in manifest row order (recrop_plates writes them in
    time order)."""
    wanted = {(r["series"], r["crop_id"]) for r in truth_rows}
    series = OrderedDict()
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["series"], int(row["crop_id"]))
            if key in wanted:
                name = os.path.basename(row["output_path"].replace(chr(92), "/"))
                series.setdefault(key, []).append(name)
    return series


def proxy_seed_point(width, height, pad_fraction=1 / 12):
    """(x, y) in crop pixels: centred, at the lower padding boundary."""
    return int(round(width / 2)), int(round(height - height * pad_fraction))


def error_stats(pred, truth):
    """Error statistics of `pred` against `truth` (parallel lists; a missing
    prediction is None or NaN). Error = pred - truth, so a positive bias means
    the pipeline reads the hook as more closed than you did."""
    pairs = [(p, t) for p, t in zip(pred, truth) if p is not None and not np.isnan(p)]
    n_total = len(truth)
    if not pairs:
        return {"n": n_total, "n_pred": 0, "coverage": 0.0, "mae": None, "median_ae": None,
                "rmse": None, "bias": None, "within10": None, "within20": None, "overhook_agree": None}
    p = np.array([a for a, _ in pairs], float)
    t = np.array([b for _, b in pairs], float)
    err = p - t
    return {
        "n": n_total, "n_pred": len(pairs), "coverage": len(pairs) / n_total,
        "mae": float(np.mean(np.abs(err))), "median_ae": float(np.median(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))), "bias": float(np.mean(err)),
        "within10": float(np.mean(np.abs(err) <= 10)), "within20": float(np.mean(np.abs(err) <= 20)),
        "overhook_agree": float(np.mean((p > 180) == (t > 180))),     # only meaningful with --keep-overhook
    }


def summarize(rows, variant):
    return error_stats([r.get(variant) for r in rows], [r["gt"] for r in rows])


def per_series(rows, variant):
    groups = defaultdict(list)
    for r in rows:
        groups[r["series"]].append(r)
    return {s: summarize(g, variant) for s, g in sorted(groups.items())}


def _fmt(v, pct=False, nd=1):
    if v is None:
        return "-"
    return f"{100 * v:.0f}%" if pct else f"{v:.{nd}f}"


def format_table(results_by_backend):
    lines = ["backend      reading    frames  read    MAE   median  bias   <=10deg <=20deg"]
    for backend, rows in results_by_backend.items():
        for variant in VARIANTS:
            if not any(r.get(variant) is not None for r in rows):
                continue                       # this backend does not produce that reading
            s = summarize(rows, variant)
            lines.append(
                f"{backend:12s} {variant:9s} {s['n']:6d} {_fmt(s['coverage'], pct=True):>5s} "
                f"{_fmt(s['mae']):>6s} {_fmt(s['median_ae']):>7s} {_fmt(s['bias']):>6s} "
                f"{_fmt(s['within10'], pct=True):>7s} {_fmt(s['within20'], pct=True):>7s}")
    return "\n".join(lines)


# --- running the real pipeline -------------------------------------------------

def run_backend(name, series, crops_dir, seed_pad):
    """{(series, crop_id, crop_file): {"raw", "raw_flip", "recon", "state"}} for
    every crop of every seedling in `series`, from the real pipeline."""
    import cv2
    from models.segmentation_backends import get_backend
    from seedling_measurment import Gui
    from utils.mask_store import MaskStore

    keys = list(series)
    all_files = [f for k in keys for f in series[k]]
    gui = Gui.__new__(Gui)          # no Tk window, same seam as tests/test_gui_segmentation_routing.py
    gui.mask_store = MaskStore()
    gui.crop_boxes, gui.seedling_pairs = [], []
    gui.crop_points_distributed = [None] * len(keys)
    gui.crop_points_numberID = [None] * len(keys)
    gui.cotyl_time_series_by_crop, gui.germ_time_series_by_crop = {}, {}
    gui.frame_results_by_crop = {}

    print(f"[{name}] segmenting {len(all_files)} crops of {len(keys)} seedlings ...", flush=True)
    get_backend(name).predict_into(gui.mask_store, [str(crops_dir / f) for f in all_files])

    out = {}
    workdir = Path(tempfile.mkdtemp(prefix="validate_angles_"))
    previous_cwd = os.getcwd()
    try:
        # process_single_frame reads crops from the relative path data/images/
        # (weights were already loaded above, so changing directory is safe)
        (workdir / "data" / "images").mkdir(parents=True)
        for f in all_files:
            shutil.copyfile(crops_dir / f, workdir / "data" / "images" / f)
        os.chdir(workdir)

        errors = 0
        for k, key in enumerate(keys):
            files = series[key]
            first = cv2.imread(str(crops_dir / files[0]))
            h, w = first.shape[:2]
            gui.crop_points_distributed[k] = [proxy_seed_point(w, h, seed_pad)]
            gui.crop_points_numberID[k] = [k + 1]
            results = []
            for f in files:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        results.append(gui.process_single_frame(f, k))
                except Exception as exc:          # one bad crop must not sink the run
                    errors += 1
                    print(f"[{name}] {f}: {type(exc).__name__}: {exc}")
                    results.append(None)
            gui.frame_results_by_crop[k] = results
            with contextlib.redirect_stdout(io.StringIO()):
                gui._reconstruct_series_for_crop(k)

            seed_id = k + 1
            for f, res in zip(files, results):
                entry = {"raw": None, "raw_flip": None, "recon": None, "state": ""}
                if res is not None:
                    raw = res.get("raw_angle_dict", {}).get(seed_id)
                    recon = res["angle_dict"].get(seed_id)
                    ok = lambda v: isinstance(v, (int, float)) and not np.isnan(v)
                    entry["raw"] = float(raw) if ok(raw) else None
                    entry["raw_flip"] = 180.0 - raw if ok(raw) else None
                    entry["recon"] = float(recon) if ok(recon) else None
                    entry["state"] = res.get("state_dict", {}).get(seed_id, "")
                out[(key[0], key[1], f)] = entry
        if errors:
            print(f"[{name}] {errors} crop(s) raised and were treated as 'no reading'")
    finally:
        os.chdir(previous_cwd)
        shutil.rmtree(workdir, ignore_errors=True)
    return out


def run_landmarks(checkpoint, series, crops_dir):
    """{(series, crop_id, crop_file): {"landmark", "landmark_recon"}} from a
    landmark-head checkpoint, over the same crop series as run_backend. The
    per-frame angle is the landmark readout's bio value (None when no junction
    is found); landmark_recon runs the same reconstruction the pipeline applies."""
    from models.UNetInference import MARGIN
    from models.multiclass_inference import MulticlassInference
    from utils.angle_timeseries import reconstruct_series
    from utils.landmark_timeseries import reconstruct_landmark_series

    model = MulticlassInference(checkpoint, num_classes=4, in_size=252 + 2 * MARGIN, out_size=252, margin=MARGIN)
    if not model.has_landmarks:
        raise SystemExit(f"{checkpoint} has no landmark head (train with multi/configs/training_landmarks.yaml)")
    all_files = [f for files in series.values() for f in files]
    print(f"[landmark] reading {len(all_files)} crops of {len(series)} seedlings ...", flush=True)
    readouts = model.predict_files_landmarks([str(crops_dir / f) for f in all_files])

    out = {}
    for (series_name, crop_id), files in series.items():
        per_frame = [readouts[f]["bio"] if readouts.get(f) else None for f in files]
        with contextlib.redirect_stdout(io.StringIO()):
            recon, _ = reconstruct_series(per_frame)
        temporal = reconstruct_landmark_series([readouts.get(f) for f in files])
        for f, raw_value, recon_value, temporal_value in zip(files, per_frame, recon, temporal):
            out[(series_name, crop_id, f)] = {
                "landmark": raw_value,
                "landmark_recon": None if recon_value is None or np.isnan(recon_value) else float(recon_value),
                "landmark_temporal": None if np.isnan(temporal_value) else float(temporal_value),
            }
    return out


SERIES_FIELDS = ["series", "crop_id", "frame_idx", "crop_file", "gt", "gt_overhook",
                 "raw_flip", "recon", "landmark", "landmark_temporal"]


def write_series_csv(path, series, truth, full_preds):
    """Every frame of every measured seedling in time order, with each reading and
    (where the user measured it) the hand-measured angle -- the input of
    multi/plot_kinematics.py."""
    gt = {(t["series"], t["crop_id"], t["crop_file"]): t for t in truth}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SERIES_FIELDS)
        w.writeheader()
        for (s, c), files in series.items():
            for i, f in enumerate(files):
                p = full_preds.get((s, c, f), {})
                t = gt.get((s, c, f))
                w.writerow({"series": s, "crop_id": c, "frame_idx": i, "crop_file": f,
                            "gt": "" if t is None else t["gt"], "gt_overhook": "" if t is None else t["gt_overhook"],
                            **{k: ("" if p.get(k) is None else p[k]) for k in SERIES_FIELDS[6:]}})


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--truth", default=str(_REPO_ROOT / "angle_ground_truth.csv"))
    p.add_argument("--crops", default=str(_REPO_ROOT / "cropped_training_set"),
                   help="Folder of crops with its manifest.csv")
    p.add_argument("--backends", default="binary,multiclass", help="Comma-separated: binary, multiclass")
    p.add_argument("--landmark-checkpoint", default=None,
                   help="Also score a landmark-head checkpoint (best.pt from training_landmarks.yaml)")
    p.add_argument("--keep-overhook", action="store_true",
                   help="score against the raw bio angle (180 + theta for Overhook-flagged frames) instead of "
                        "dropping the flag (default: truth = 180 - theta, readings folded below 180)")
    p.add_argument("--seed-pad", type=float, default=1 / 12,
                   help="Seed point height above the crop's bottom edge, as a fraction of crop height (default 1/12)")
    p.add_argument("--out-dir", default=str(_REPO_ROOT / "multi" / "results" / "angle_validation"))
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    sys.path.insert(0, str(_REPO_ROOT))
    os.chdir(_REPO_ROOT)                   # backends resolve weight paths relative to the repo root
    crops_dir = Path(args.crops).resolve()      # run_backend changes directory: a relative path would break
    truth = load_truth(args.truth, fold=not args.keep_overhook)
    series = load_series(crops_dir / "manifest.csv", truth)
    print(f"{len(truth)} measured frames, {len(series)} seedlings, "
          f"{sum(len(v) for v in series.values())} crops to run")

    results_by_backend, csv_rows = OrderedDict(), []
    full_preds = {}                          # reading name -> {(series, crop_id, crop_file): value}, every frame
    for name in [b.strip() for b in args.backends.split(",") if b.strip()]:
        pred = run_backend(name, series, crops_dir, args.seed_pad)
        if not args.keep_overhook:
            pred = fold_predictions(pred)
        rows = []
        for t in truth:
            p = pred.get((t["series"], t["crop_id"], t["crop_file"]), {})
            rows.append({**t, "raw": p.get("raw"), "raw_flip": p.get("raw_flip"),
                         "recon": p.get("recon"), "state": p.get("state", "")})
        results_by_backend[name] = rows
        csv_rows += [{"backend": name, **r} for r in rows]
        if name == "multiclass" or not full_preds:
            for key, p in pred.items():
                full_preds.setdefault(key, {}).update(
                    {k: p.get(k) for k in ("raw_flip", "recon")})

    if args.landmark_checkpoint:
        pred = run_landmarks(args.landmark_checkpoint, series, crops_dir)
        if not args.keep_overhook:
            pred = fold_predictions(pred)
        name = "landmark:" + Path(args.landmark_checkpoint).parent.name
        rows = []
        for t in truth:
            p = pred.get((t["series"], t["crop_id"], t["crop_file"]), {})
            rows.append({**t, "landmark": p.get("landmark"), "landmark_recon": p.get("landmark_recon"),
                         "landmark_temporal": p.get("landmark_temporal")})
        results_by_backend[name] = rows
        csv_rows += [{"backend": name, **r} for r in rows]
        for key, p in pred.items():
            full_preds.setdefault(key, {}).update(
                {k: p.get(k) for k in ("landmark", "landmark_temporal")})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["backend", "series", "crop_id", "frame", "crop_file", "gt", "gt_overhook",
              "raw", "raw_flip", "recon", "state", "landmark", "landmark_recon", "landmark_temporal"]
    with open(out_dir / "angle_validation.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(csv_rows)

    write_series_csv(out_dir / "angle_series.csv", series, truth, full_preds)

    table = format_table(results_by_backend)
    print()
    print(table)
    for backend, rows in results_by_backend.items():
        variant = "landmark_temporal" if backend.startswith("landmark") else "recon"
        print(f"\nPer series, {backend} / {variant} (MAE deg, frames with a reading / frames):")
        for s, st in per_series(rows, variant).items():
            print(f"  {s[:36]:36s} {_fmt(st['mae']):>6s}   {st['n_pred']}/{st['n']}")
    (out_dir / "summary.txt").write_text(table + chr(10), encoding="utf-8")
    print(f"\nWrote {out_dir / 'angle_validation.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
