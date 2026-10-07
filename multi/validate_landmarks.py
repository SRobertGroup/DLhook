#!/usr/bin/env python
"""Score landmark checkpoints on whole crops against hand-clicked landmarks.

Reads an angle annotation CSV (junction + axis clicks, ui/angle_annotator.py) and optionally its
collar/root CSV (ui/root_annotator.py), runs each checkpoint on the annotated crops through the
inference tiling and readout (the same run_wholecrop_landmark_validation the trainer selects on),
and reports theta (angle between the axes), junction, collar and root-direction errors. Unlike
multi/validate_angles.py it needs no manifest and no seedling series: one crop, one comparison.

    python -m multi.validate_landmarks --truth angle_new.csv --root-truth angle_new_root.csv \
        --folder cropped_new_set --checkpoints multi/results/models_landmarks_v4/best.pt,multi/results/models_landmarks_v6/best.pt
"""
from __future__ import annotations

import argparse
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
    args = p.parse_args(argv)
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_REPO_ROOT / "multi"))
    os.chdir(_REPO_ROOT)
    import torch
    from models.UNetInference import MARGIN
    from models.multiclass_inference import MulticlassInference
    from multi.src.landmarks import load_landmarks
    from multi.train_unet_multiclass import run_wholecrop_landmark_validation

    landmarks, report = load_landmarks(args.truth, root_csv_path=args.root_truth)
    print(f"{report['loaded']} annotated crops ({report['invalid']} unusable)"
          + (f", {report['with_root']} with collar/root" if "with_root" in report else ""))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'checkpoint':40s} {'theta MAE':>9s} {'median':>7s} {'junction px':>11s} {'collar px':>9s} {'root deg':>8s}")
    for ckpt in [c.strip() for c in args.checkpoints.split(",") if c.strip()]:
        model = MulticlassInference(ckpt, num_classes=4, device=device, in_size=args.patch_size + 2 * MARGIN,
                                    out_size=args.patch_size, margin=MARGIN)._core()
        m = run_wholecrop_landmark_validation(model, landmarks, args.folder, device, patch_size=args.patch_size)
        label = str(Path(ckpt).parent.name or ckpt)[:40]
        fmt = lambda v: "   n/a" if v != v else f"{v:6.1f}"
        print(f"{label:40s} {fmt(m['wholecrop_theta_mae']):>9s} {fmt(m['wholecrop_theta_median']):>7s} "
              f"{fmt(m['wholecrop_junction_px']):>11s} {fmt(m['wholecrop_collar_px']):>9s} {fmt(m['wholecrop_root_deg']):>8s}")
    print("\njunction / collar / root are medians; a crop where nothing is found counts as 90 deg / 30 px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
