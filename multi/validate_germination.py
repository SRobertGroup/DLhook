#!/usr/bin/env python
"""Score germination detection against hand-marked onsets.

`python -m ui.germination_annotator` records, per seedling, the first frame in which a
radicle is visible. This script runs the app's GerminationDetector
(utils/germination_detector.py, unmodified) on the same seedlings' crops, with each
radicle-mask source, and compares the detected frame with the annotated one:

  germ_v1      weights/RootPainter_weights/germ_v1.pkl   (what the GUI deploys today)
  germ_v2      weights/RootPainter_weights/germ_v2.pkl   (the retrained binary teacher)
  multiclass   class 3 of weights/multiclass/dlhook_4class_v1.pt

Frames are the crops in --folder (every Nth source frame), so an error of 1 means one
crop step. The detector needs a seed point; there are no operator clicks here, so it uses
the same proxy as multi/compare_germination_areas.py (bottom-centre of the crop), the same
for every arm. Only seedlings annotated 'found' are scored; 'before_start' / 'none'
seedlings are reported separately as detected / not detected.

    python -m multi.validate_germination [--truth germination_train.csv] [--arms germ_v1,germ_v2,multiclass]

Outputs multi/results/germination_validation/{germination_validation.csv,summary.txt}.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = _REPO_ROOT / "multi" / "results" / "germination_validation"
ARMS = ("germ_v1", "germ_v2", "multiclass")


def load_truth(path):
    """{(series, crop_id): {"status", "onset_index"}} from a germination annotation CSV."""
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["status"] == "skipped":
                continue
            out[(row["series"], int(row["crop_id"]))] = {
                "status": row["status"],
                "onset_index": int(row["onset_index"]) if row["onset_index"] != "" else None}
    return out


def detect_onsets(masks_by_file, seedlings, folder, areas_out=None):
    """{(series, crop_id): detected frame index or None} -- the app's detector on one arm's masks.
    With areas_out (a list), also appends one (series, crop_id, frame_index, near_seed_area,
    total_area) row per frame, the signal the detector thresholds."""
    import cv2
    from multi.compare_germination_areas import germ_contours_from_mask
    from utils.germination_detector import GerminationDetector

    detector = GerminationDetector()
    out = {}
    for s in seedlings:
        files = [f[1] for f in s.frames]
        first = cv2.imread(os.path.join(folder, files[0]))
        h, w = first.shape[:2]
        contours = [germ_contours_from_mask(masks_by_file.get(f)) for f in files]
        out[s.key] = detector.detect(s.key, contours, (w / 2.0, h - 1.0), (w + h) / 2.0)
        if areas_out is not None:
            diag = detector.diagnostics[s.key]
            for i, near in enumerate(diag["areas"]):
                total = float(sum(cv2.contourArea(c) for c in contours[i]))
                areas_out.append((s.series, s.crop_id, i, float(near), total))
    return out


def score(truth, detected):
    """Summary dict for one arm: errors in frames over seedlings annotated 'found'."""
    errors, missed = [], 0
    for key, t in truth.items():
        if t["status"] != "found":
            continue
        d = detected.get(key)
        if d is None:
            missed += 1
        else:
            errors.append(d - t["onset_index"])
    errors = np.array(errors, dtype=float)
    n_found = len(errors) + missed
    out = {"n": n_found, "not_detected": missed, "detected": len(errors)}
    if len(errors):
        out.update(exact=float(np.mean(errors == 0)), within1=float(np.mean(np.abs(errors) <= 1)),
                   mae=float(np.mean(np.abs(errors))), median_abs=float(np.median(np.abs(errors))),
                   bias=float(np.mean(errors)), early=int(np.sum(errors < 0)), late=int(np.sum(errors > 0)))
    before = [detected.get(k) for k, t in truth.items() if t["status"] == "before_start"]
    none = [detected.get(k) for k, t in truth.items() if t["status"] == "none"]
    out["before_start_detected"] = f"{sum(v is not None for v in before)}/{len(before)}"
    out["none_false_alarm"] = f"{sum(v is not None for v in none)}/{len(none)}"
    return out


def format_summary(scores):
    lines = [f"{'arm':12s} {'n':>4s} {'found':>6s} {'exact':>6s} {'+-1':>6s} {'MAE':>6s} {'median':>7s} {'bias':>6s} "
             f"{'early':>6s} {'late':>5s}  before_start  none_false_alarm"]
    for arm, s in scores.items():
        if "mae" not in s:
            lines.append(f"{arm:12s} {s['n']:4d} {s['detected']:6d}   (no detection on any scored seedling)")
            continue
        lines.append(f"{arm:12s} {s['n']:4d} {s['detected']:6d} {s['exact']:6.0%} {s['within1']:6.0%} {s['mae']:6.2f} "
                     f"{s['median_abs']:7.1f} {s['bias']:+6.2f} {s['early']:6d} {s['late']:5d}  "
                     f"{s['before_start_detected']:>12s}  {s['none_false_alarm']:>16s}")
    return "\n".join(lines)


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--truth", default=str(_REPO_ROOT / "germination_train.csv"))
    p.add_argument("--folder", default=str(_REPO_ROOT / "cropped_training_set"))
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--germ-v1", default="weights/RootPainter_weights/germ_v1.pkl")
    p.add_argument("--germ-v2", default="weights/RootPainter_weights/germ_v2.pkl")
    p.add_argument("--multiclass", default="weights/multiclass/dlhook_4class_v1.pt")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    sys.path.insert(0, str(_REPO_ROOT))
    os.chdir(_REPO_ROOT)
    from multi.compare_germination_areas import run_binary_arm, run_multiclass_arm
    from utils.germination_annotation import load_seedlings
    from utils.angle_annotation import default_manifest

    truth = load_truth(args.truth)
    seedlings = [s for s in load_seedlings(args.folder, default_manifest(args.folder)) if s.key in truth]
    print(f"{len(truth)} annotated seedlings, {len(seedlings)} with crops, "
          f"{sum(len(s) for s in seedlings)} frames")
    paths = [os.path.join(args.folder, f[1]) for s in seedlings for f in s.frames]

    runners = {"germ_v1": lambda: run_binary_arm(args.germ_v1, paths),
               "germ_v2": lambda: run_binary_arm(args.germ_v2, paths),
               "multiclass": lambda: run_multiclass_arm(args.multiclass, paths)}
    scores, detected_by_arm, areas_by_arm = OrderedDict(), OrderedDict(), OrderedDict()
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        print(f"[{arm}] segmenting {len(paths)} crops ...", flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            masks = runners[arm]()
        # arms key their masks by file name or path depending on the backend; normalise to the base name
        masks = {os.path.basename(k): v for k, v in masks.items()}
        arm_areas = []
        detected_by_arm[arm] = detect_onsets(masks, seedlings, args.folder, arm_areas)
        areas_by_arm[arm] = arm_areas
        scores[arm] = score(truth, detected_by_arm[arm])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "germination_validation.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["series", "crop_id", "status", "n_frames", "annotated_index"] + [f"{a}_index" for a in detected_by_arm])
        for s in seedlings:
            t = truth[s.key]
            w.writerow([s.series, s.crop_id, t["status"], len(s), "" if t["onset_index"] is None else t["onset_index"]]
                       + ["" if detected_by_arm[a][s.key] is None else detected_by_arm[a][s.key] for a in detected_by_arm])
    with open(out_dir / "radicle_areas.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "series", "crop_id", "frame_index", "near_seed_area", "total_area"])
        for arm, rows in areas_by_arm.items():
            w.writerows((arm, *r) for r in rows)
    text = format_summary(scores)
    (out_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print()
    print(text)
    print(f"\nwrote {out_dir / 'germination_validation.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
