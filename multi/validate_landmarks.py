#!/usr/bin/env python
"""Score landmark checkpoints on whole crops against hand-clicked landmarks.

Reads an angle annotation CSV (junction + axis clicks, ui/angle_annotator.py; its review sidecar
`<csv>.review.csv` is applied) and optionally its collar/root CSV (ui/root_annotator.py), runs each
checkpoint on the annotated crops through the inference tiling and readout (the same
run_wholecrop_landmark_validation the trainer selects on), and reports theta (angle between the
axes), junction, collar and root-direction errors, plus the cotyledon base and tip errors and how
often the tip drove the angle for a head that has them. Unlike multi/validate_angles.py it needs no
manifest and no seedling series: one crop, one comparison.

    python -m multi.validate_landmarks --truth data/annotations/angles/angle_new.csv --root-truth data/annotations/angles/angle_new_root.csv \
        --folder data/crops/cropped_new_set --checkpoints multi/results/models_landmarks_v4/best.pt,multi/results/models_landmarks_v6/best.pt

A "+" inside one checkpoint entry (v6/best.pt+v11/best.pt) scores those checkpoints as an ensemble: their
junction heatmaps, direction fields and optional maps are averaged before the readout
(models.multiclass_inference.LandmarkEnsemble).

--per-crop-out DIR also writes one CSV per checkpoint (crop_file, err, theta_gt, theta_pred, path):
`err` is the file `python -m ui.angle_reviewer --errors` sorts by.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--truth", required=True, help="angle annotation CSV (measured rows are used)")
    p.add_argument("--root-truth", default=None, help="collar/root CSV from ui/root_annotator.py")
    p.add_argument("--folder", required=True, help="folder holding the annotated crops")
    p.add_argument("--checkpoints", required=True, help="comma-separated landmark checkpoints")
    p.add_argument("--patch-size", type=int, default=252)
    p.add_argument("--per-crop-out", default=None, help="folder for one per-crop errors CSV per checkpoint")
    args = p.parse_args(argv)
    # resolve before the chdir below: relative paths mean relative to where the command was run
    truth, root_truth, folder = (str(Path(v).resolve()) if v else None
                                 for v in (args.truth, args.root_truth, args.folder))
    per_crop_out = Path(args.per_crop_out).resolve() if args.per_crop_out else None
    # one checkpoint, or several joined by "+" to average their maps before the readout (an ensemble)
    checkpoints = ["+".join(str(Path(c.strip()).resolve()) for c in spec.split("+") if c.strip())
                   for spec in args.checkpoints.split(",") if spec.strip()]
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_REPO_ROOT / "multi"))
    os.chdir(_REPO_ROOT)
    import torch
    from models.UNetInference import MARGIN
    from models.multiclass_inference import MulticlassInference
    from multi.src.landmarks import load_landmarks
    from multi.train_unet_multiclass import run_wholecrop_landmark_validation

    landmarks, report = load_landmarks(truth, root_csv_path=root_truth)
    print(f"{report['loaded']} annotated crops ({report['invalid']} unusable)"
          + (f", {report['with_root']} with collar/root" if "with_root" in report else ""))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'checkpoint':40s} {'theta MAE':>9s} {'median':>7s} {'junction px':>11s} {'collar px':>9s} "
          f"{'root deg':>8s} {'tip px':>7s} {'base px':>7s} {'tip used':>8s}")
    for ckpt in checkpoints:
        parts = ckpt.split("+")
        model = [MulticlassInference(c, num_classes=4, device=device, in_size=args.patch_size + 2 * MARGIN,
                                     out_size=args.patch_size, margin=MARGIN)._core() for c in parts]
        model = model[0] if len(model) == 1 else model
        per_crop = {} if per_crop_out else None
        m = run_wholecrop_landmark_validation(model, landmarks, folder, device, patch_size=args.patch_size,
                                              per_crop=per_crop)
        label = ("ens " + "+".join(Path(c).parent.name.replace("models_landmarks_", "") for c in parts) if len(parts) > 1
                 else str(Path(ckpt).parent.name or ckpt)[:40])
        fmt = lambda v: "   n/a" if v != v else f"{v:6.1f}"
        used = m["wholecrop_tip_used"]
        print(f"{label:40s} {fmt(m['wholecrop_theta_mae']):>9s} {fmt(m['wholecrop_theta_median']):>7s} "
              f"{fmt(m['wholecrop_junction_px']):>11s} {fmt(m['wholecrop_collar_px']):>9s} "
              f"{fmt(m['wholecrop_root_deg']):>8s} {fmt(m['wholecrop_tip_px']):>7s} "
              f"{fmt(m['wholecrop_base_px']):>7s} {('   n/a' if used != used else f'{used:7.0%}'):>8s}")
        if per_crop_out:
            per_crop_out.mkdir(parents=True, exist_ok=True)
            out = per_crop_out / (f"{label.replace(' ', '_')}.csv" if len(parts) > 1 else f"{Path(ckpt).parent.name}.csv")
            with open(out, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["crop_file", "err", "theta_gt", "theta_pred", "path", "overhook_gt", "overhook_pred", "hypo_deg", "cotyl_deg"])
                for name, d in sorted(per_crop.items(), key=lambda kv: -kv[1]["err"]):
                    w.writerow([name, f"{d['err']:.2f}", f"{d['theta_gt']:.2f}",
                                "" if d["theta_pred"] is None else f"{d['theta_pred']:.2f}", d["path"], d["overhook_gt"], d["overhook_pred"],
                                "" if d["hypo_deg"] is None else f"{d['hypo_deg']:.2f}", "" if d["cotyl_deg"] is None else f"{d['cotyl_deg']:.2f}"])
            print(f"   per-crop errors -> {out}")
    print("\njunction / collar / root / tip / base are medians; a crop where nothing is found counts as 90 deg / 30 px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
