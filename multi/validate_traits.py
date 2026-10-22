"""Score the pipeline's hypocotyl traits against hand measurements (ui/trait_annotator.py).

    python -m multi.validate_traits --truth data/annotations/traits/cropped_berrybox_val.csv
    python -m multi.validate_traits --truth ... --checkpoint weights/multiclass/dlhook_4class_v1.pt+weights/multiclass/dlhook_4class_v2.pt --binary
    python -m multi.validate_traits --proxy                     # no hand traits: clicked J -> H2 -> collar, every folder

With --truth, every measured frame is scored for each mask (the four-class hypocotyl of each
--checkpoint, default the GUI's; --binary adds hypocot_v5) under two anchorings:

  hand     junction / collar = the ends of your midline -- isolates the mask
  model    junction / collar = the landmark ensemble's reading -- what the GUI exports

Per trait (length, diameter, mean curvature): n, bias (pipeline - hand), MAE, median ratio, Pearson r
and the share of frames off by more than 10%. Writes summary.txt and per_crop.csv to --out-dir.

--proxy keeps the check used for the v2 decision where no hand traits exist: length with the clicked
junction and collar against the clicked polyline J -> H2 -> collar, per annotated crop folder.
"""
import argparse
import csv
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.hypocotyl_traits import hypocotyl_midline, hypocotyl_traits  # noqa: E402
from utils.landmark_marks import annotated_sources, clicked_for_folder  # noqa: E402
from utils.paths import repo_path  # noqa: E402

OUT_DIR = "multi/results/traits_validation"
BINARY = "weights/RootPainter_weights/hypocot_v5.pkl"
TRAITS = [("length_px", "length"), ("diameter_px", "diameter"), ("curvature_mean", "curvature")]
OFF_FRACTION = 0.10


# ---- masks ---------------------------------------------------------------------------------------
def hypocotyl_masks(images, spec=None, binary=False):
    """{name: [uint8 0/255 hypocotyl mask per image]} for each checkpoint of `spec` ('+'-joined,
    default the GUI's four-class checkpoint) and, with `binary`, hypocot_v5."""
    from models.multiclass_inference import MulticlassInference
    from models.segmentation_backends import default_multiclass_checkpoint
    from models.UNetInference import MARGIN, get_predictor
    out = {}
    for ckpt in [c for c in (spec or default_multiclass_checkpoint()).split("+") if c]:
        model = MulticlassInference(repo_path(ckpt), num_classes=4, in_size=252 + 2 * MARGIN, out_size=252,
                                    margin=MARGIN)
        out[os.path.basename(ckpt)] = [(lab == 2).astype(np.uint8) * 255 for lab in model.segment_many_argmax(images)]
    if binary:
        out["hypocot_v5"] = [(p > 0.5).astype(np.uint8) * 255
                             for p in get_predictor(repo_path(BINARY))._segment_many(images)]
    return out


# ---- statistics (pure) ---------------------------------------------------------------------------
def compare(pred, truth, off=OFF_FRACTION):
    """dict(n, bias, mae, ratio, r, off) over the pairs where both are finite."""
    p, t = np.asarray(pred, float), np.asarray(truth, float)
    ok = np.isfinite(p) & np.isfinite(t)
    p, t = p[ok], t[ok]
    if not len(p):
        return {"n": 0, "bias": math.nan, "mae": math.nan, "ratio": math.nan, "r": math.nan, "off": math.nan}
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = p / t
    r = float(np.corrcoef(p, t)[0, 1]) if len(p) > 2 and p.std() > 0 and t.std() > 0 else math.nan
    return {"n": int(len(p)), "bias": float(np.mean(p - t)), "mae": float(np.mean(np.abs(p - t))),
            "ratio": float(np.nanmedian(ratio)), "r": r, "off": float(np.mean(np.abs(ratio - 1) > off))}


def _fmt(s):
    return (f"n {s['n']:4d}  bias {s['bias']:+8.3f}  MAE {s['mae']:7.3f}  ratio {s['ratio']:6.3f}  "
            f"r {s['r']:5.2f}  off>10% {s['off']:4.0%}") if s["n"] else "n    0"


# ---- --truth -------------------------------------------------------------------------------------
def score_truth(truth_csv, folder, spec=None, binary=False, landmark_model=None):
    """(summary text, per-crop rows). `landmark_model`: crop path -> readout (default the GUI's)."""
    from utils.trait_annotation import hand_traits, read_measured
    rows_in = [r for r in read_measured(truth_csv) if os.path.exists(os.path.join(repo_path(folder), r[0]))]
    if not rows_in:
        return f"no measured frame of {truth_csv} found in {folder}", []
    images = [cv2.imread(os.path.join(repo_path(folder), crop), cv2.IMREAD_COLOR) for crop, *_ in rows_in]
    masks = hypocotyl_masks(images, spec, binary)
    if landmark_model is None:
        from utils.landmark_marks import landmark_model as _lm
        landmark_model = _lm()
    per_crop = []
    for k, (crop, mid, transects, _) in enumerate(rows_in):
        hand = hand_traits(mid, transects)
        model = landmark_model(os.path.join(repo_path(folder), crop)) or {}
        anchors = {"hand": (mid[0], mid[-1]), "model": (model.get("junction"), model.get("collar"))}
        for name, ms in masks.items():
            for anchor, (j, c) in anchors.items():
                t = hypocotyl_traits(ms[k], j, c) if j is not None else {}
                row = {"crop_file": crop, "mask": name, "anchor": anchor}
                for key, _ in TRAITS:
                    row[f"hand_{key}"], row[f"pipe_{key}"] = hand[key], t.get(key, math.nan)
                per_crop.append(row)
    lines = [f"{len(rows_in)} measured frames ({truth_csv}, {folder})", ""]
    for name in masks:
        for anchor in ("hand", "model"):
            sel = [r for r in per_crop if r["mask"] == name and r["anchor"] == anchor]
            lines.append(f"== {name}, junction/collar from the {anchor}")
            for key, label in TRAITS:
                s = compare([r[f"pipe_{key}"] for r in sel], [r[f"hand_{key}"] for r in sel])
                lines.append(f"  {label:10s} {_fmt(s)}")
            lines.append("")
    lines.append("bias = pipeline - hand (px, curvature 1/px); ratio = median pipeline / hand.")
    return "\n".join(lines), per_crop


# ---- --proxy -------------------------------------------------------------------------------------
def score_proxy(spec=None, binary=False, folders=None):
    """Length with the clicked junction and collar against the clicked polyline J -> H2 -> collar."""
    seen, lines = [], []
    for _, _, folder in annotated_sources():
        if folder in seen or (folders and os.path.basename(os.path.normpath(folder)) not in folders):
            continue
        seen.append(folder)
        items = sorted((c, a) for c, a in clicked_for_folder(folder).items() if a.get("collar") and a.get("hypo2"))
        if not items:
            continue
        images = [cv2.imread(os.path.join(repo_path(folder), c), cv2.IMREAD_COLOR) for c, _ in items]
        ref = [math.dist(a["junction"], a["hypo2"]) + math.dist(a["hypo2"], a["collar"]) for _, a in items]
        lines.append(f"## {folder}: {len(items)} frames with junction, H2 and collar")
        lines.append(f"{'mask':24s} {'NaN':>4s} {'ratio':>6s} {'<0.9':>5s} {'<0.75':>6s} {'>1.15':>6s} "
                     f"{'start gap':>9s} {'>5px':>5s}")
        for name, ms in hypocotyl_masks(images, spec, binary).items():
            ratios, gaps, nan = [], [], 0
            for (c, a), m, r in zip(items, ms, ref):
                length = hypocotyl_traits(m, a["junction"], a["collar"])["length_px"]
                if not np.isfinite(length):
                    nan += 1
                    continue
                gaps.append(math.dist(hypocotyl_midline(m, a["junction"], a["collar"])[0], a["junction"]))
                ratios.append(length / r)
            q, g = np.array(ratios), np.array(gaps)
            lines.append(f"{name:24s} {nan:4d} {np.median(q):6.3f} {np.mean(q < 0.9):5.0%} {np.mean(q < 0.75):6.0%} "
                         f"{np.mean(q > 1.15):6.0%} {np.median(g):9.1f} {np.mean(g > 5):5.0%}")
        lines.append("")
    lines.append("ratio: median length / clicked J -> H2 -> collar; start gap: px from the junction to the midline start.")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--truth", help="Traits CSV from ui/trait_annotator.py")
    p.add_argument("--folder", default="data/crops/cropped_berrybox_val", help="Crop folder of --truth")
    p.add_argument("--proxy", action="store_true", help="No hand traits: clicked-polyline length, every folder")
    p.add_argument("--folders", default=None, help="--proxy: comma-separated folder names (default all)")
    p.add_argument("--checkpoint", default=None, metavar="CKPT[+CKPT]", help="Four-class checkpoints (default the GUI's)")
    p.add_argument("--binary", action="store_true", help="Also score the binary hypocot_v5 mask")
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args(argv)
    if not args.truth and not args.proxy:
        p.error("give --truth CSV or --proxy")
    out = _REPO_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.proxy:
        text = score_proxy(args.checkpoint, args.binary, args.folders.split(",") if args.folders else None)
        (out / "proxy_length.txt").write_text(text, encoding="utf-8")
    else:
        text, rows = score_truth(args.truth, args.folder, args.checkpoint, args.binary)
        (out / "summary.txt").write_text(text, encoding="utf-8")
        if rows:
            with open(out / "per_crop.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
