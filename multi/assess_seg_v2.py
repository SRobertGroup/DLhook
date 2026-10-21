"""Is a 4-class v2 worth it, and which crops need human strokes? (Stage 0 of the v2 plan.)

    python -m multi.assess_seg_v2                       # v1 against the binary teachers, every crop folder
    python -m multi.assess_seg_v2 --checkpoint weights/multiclass/dlhook_4class_v2.pt --out-dir ...   # rerun on v2

For each crop folder it reads the hand-annotated landmark frames (clicked junction and hypocotyl
points; collar from the *_root.csv where there is one) plus a random sample of the other crops, runs
the binary teachers (cotyledon_v5, hypocot_v5, at 0.5) and the four-class student (argmax), and
measures per crop:

  gap_teacher / gap_student    px from the clicked junction to the nearest hypocotyl pixel (M-9:
                               the hook region the trait mask misses)
  cover_teacher / cover_student share of the clicked junction -> H2 segment inside the hypocotyl mask
  parts_student / parts_teacher hypocotyl pieces of at least MIN_PART px ("cut in half" when > 1)
  dice_cot / dice_hypo         student against teacher (domain gap; v1's home domain ~0.84 / 0.95)

Writes <out-dir>/per_crop.csv, summary.txt (per folder) and tricky_crops.csv -- the crops ranked by
how much a human stroke would help, with the reason, for `python -m ui.seg_annotator --queue`.

    python -m multi.assess_seg_v2 --score-strokes v1.pt+v2.pt     # checkpoints against your --split val strokes

--score-strokes scores each checkpoint (argmax) on the painted pixels of the held-out strokes
(default: data/annotations/seg_strokes/cropped_berrybox_val, split val): per class precision and recall,
pooled over pixels and averaged per seedling, plus the rig breakdown; written to strokes_score.txt.
"""
import argparse
import csv
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

TEACHERS = {"cotyledon": "weights/RootPainter_weights/cotyledon_v5.pkl",
            "hypocotyl": "weights/RootPainter_weights/hypocot_v5.pkl"}
MIN_PART = 15            # px: smaller hypocotyl blobs are noise, not a second piece
GAP_PX = 5.0             # a junction gap above this is the M-9 problem
COVER_TOL_PX = 1         # the J -> H2 segment counts as covered within this distance of the mask
OUT_DIR = "multi/results/seg_v2_assessment"


# ---- metrics (pure) ------------------------------------------------------------------------------
def junction_gap(mask, junction):
    """px from `junction` (x, y) to the nearest foreground pixel of `mask`; inf when empty."""
    m = np.asarray(mask) > 0
    if not m.any():
        return float("inf")
    h, w = m.shape
    x = int(np.clip(round(junction[0]), 0, w - 1))
    y = int(np.clip(round(junction[1]), 0, h - 1))
    if m[y, x]:
        return 0.0
    dist = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)
    return float(dist[y, x])


def segment_coverage(mask, p0, p1, tol=COVER_TOL_PX):
    """Share of the points along p0 -> p1 lying within `tol` px of the mask's foreground."""
    m = np.asarray(mask) > 0
    if tol > 0:
        m = cv2.dilate(m.astype(np.uint8), np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)) > 0
    h, w = m.shape
    n = max(2, int(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) + 1)
    xs = np.clip(np.round(np.linspace(p0[0], p1[0], n)).astype(int), 0, w - 1)
    ys = np.clip(np.round(np.linspace(p0[1], p1[1], n)).astype(int), 0, h - 1)
    return float(m[ys, xs].mean())


def n_parts(mask, min_area=MIN_PART):
    m = (np.asarray(mask) > 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return int((stats[1:, cv2.CC_STAT_AREA] >= min_area).sum())


def dice(a, b):
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    s = a.sum() + b.sum()
    return float("nan") if s == 0 else float(2 * (a & b).sum() / s)


def tricky_score(row):
    """(score, reason): how much a human stroke would help this crop."""
    score, reasons = 0.0, []
    gs, gt = row.get("gap_student"), row.get("gap_teacher")
    if gs is not None and np.isfinite(gs) and gs > GAP_PX:
        score += min(gs, 40) / GAP_PX
        reasons.append(f"junction gap {gs:.0f} px")
    if gt is not None and np.isfinite(gt) and gt > GAP_PX:
        score += min(gt, 40) / (2 * GAP_PX)
        reasons.append(f"teacher gap {gt:.0f} px")
    if row.get("parts_student", 1) > 1:
        score += 2.0
        reasons.append(f"hypocotyl in {row['parts_student']} pieces")
    for key, label, weight in (("dice_hypo", "hypocotyl", 4.0), ("dice_cot", "cotyledon", 2.0)):
        d = row.get(key)
        if d is not None and np.isfinite(d) and d < 0.7:
            score += weight * (1 - d)
            reasons.append(f"{label} disagrees with teacher (Dice {d:.2f})")
    return round(score, 3), "; ".join(reasons)


# ---- data ----------------------------------------------------------------------------------------
from utils.landmark_marks import EXTRA_SETS, LANDMARK_CONFIG, annotated_sources, load_annotations  # noqa: E402,F401


def crop_plan(n_unannotated, max_annotated, seed):
    """{folder: {crop_file: annotation or None}} over every annotated source folder."""
    plan = {}
    for csv_path, root_csv, folder in annotated_sources():
        ann = load_annotations(csv_path, root_csv, folder)
        plan.setdefault(folder, {})
        rng = random.Random(f"{seed}:{folder}:{csv_path}")
        items = sorted(ann.items())
        if max_annotated and len(items) > max_annotated:
            items = rng.sample(items, max_annotated)
        plan[folder].update(items)
    for folder, crops in plan.items():
        rng = random.Random(f"{seed}:{folder}")
        others = sorted(n for n in os.listdir(_REPO_ROOT / folder)
                        if n.endswith(".png") and "-crop-" in n and n not in crops)
        for name in rng.sample(others, min(n_unannotated, len(others))):
            crops[name] = None
    return plan


# ---- run -----------------------------------------------------------------------------------------
def _student(checkpoint):
    from models.multiclass_inference import MulticlassInference
    from models.UNetInference import MARGIN
    from utils.paths import repo_path
    return MulticlassInference(repo_path(checkpoint), num_classes=4, in_size=252 + 2 * MARGIN,
                               out_size=252, margin=MARGIN)


def measure(images, annotations, student, teachers):
    """Rows of metrics for a batch of crops (BGR images, annotation dicts or None)."""
    labels = student.segment_many_argmax(images)
    probs = {name: t._segment_many(images) for name, t in teachers.items()}
    rows = []
    for k, (image, ann, lab) in enumerate(zip(images, annotations, labels)):
        t_cot, t_hyp = probs["cotyledon"][k] > 0.5, probs["hypocotyl"][k] > 0.5
        s_cot, s_hyp = lab == 1, lab == 2
        row = {"annotated": int(ann is not None), "dice_cot": dice(s_cot, t_cot), "dice_hypo": dice(s_hyp, t_hyp),
               "parts_student": n_parts(s_hyp), "parts_teacher": n_parts(t_hyp)}
        if ann is not None:
            j = ann["junction"]
            row.update(gap_teacher=junction_gap(t_hyp, j), gap_student=junction_gap(s_hyp, j))
            if ann.get("hypo2"):
                row.update(cover_teacher=segment_coverage(t_hyp, j, ann["hypo2"]),
                           cover_student=segment_coverage(s_hyp, j, ann["hypo2"]))
        rows.append(row)
    return rows


def _med(values):
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def _share(values, test):
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.mean([test(x) for x in v])) if v else float("nan")


def summarise(rows):
    lines = ["folder                         crops  annot  gapT_med gapS_med gapT>5  gapS>5  coverT coverS  "
             "splitS splitT  diceCot diceHyp"]
    for folder in sorted({r["folder"] for r in rows}):
        rs = [r for r in rows if r["folder"] == folder]
        a = [r for r in rs if r["annotated"]]
        lines.append(
            f"{Path(folder).name:30s} {len(rs):5d}  {len(a):5d}  "
            f"{_med([r.get('gap_teacher') for r in a]):8.1f} {_med([r.get('gap_student') for r in a]):8.1f} "
            f"{_share([r.get('gap_teacher') for r in a], lambda x: x > GAP_PX):6.0%}  "
            f"{_share([r.get('gap_student') for r in a], lambda x: x > GAP_PX):6.0%}  "
            f"{_med([r.get('cover_teacher') for r in a]):6.2f} {_med([r.get('cover_student') for r in a]):6.2f}  "
            f"{_share([r['parts_student'] for r in rs], lambda x: x > 1):6.0%} "
            f"{_share([r['parts_teacher'] for r in rs], lambda x: x > 1):6.0%}  "
            f"{_med([r['dice_cot'] for r in rs]):7.2f} {_med([r['dice_hypo'] for r in rs]):7.2f}")
    lines += ["", "gapT/gapS: junction -> nearest hypocotyl pixel (teacher hypocot_v5 / student), annotated frames;",
              f"gap>5: share above {GAP_PX:.0f} px; cover: share of the clicked J -> H2 segment inside the mask;",
              "split: hypocotyl in more than one piece; dice: student vs teacher (v1 home domain ~0.84 / 0.95)."]
    return "\n".join(lines)


def stroke_confusion(labels, strokes):
    """{class: (tp, fp, fn)} of a predicted label map against a stroke map (painted pixels only)."""
    from utils.seg_strokes import UNLABELLED
    painted = strokes != UNLABELLED
    out = {}
    for name, value in (("cotyledon", 1), ("hypocotyl", 2), ("radicle", 3)):
        pred, true = (labels == value) & painted, (strokes == value) & painted
        out[name] = (int((pred & true).sum()), int((pred & ~true).sum()), int((~pred & true).sum()))
    return out


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return p, r


def _sum3(items):
    return tuple(sum(x[i] for x in items) for i in range(3))


def score_strokes(spec, folder="data/crops/cropped_berrybox_val",
                  strokes="data/annotations/seg_strokes/cropped_berrybox_val", split="val"):
    """Text report: each checkpoint of `spec` ('+'-joined) against the painted pixels of `split` strokes."""
    from utils.seg_strokes import load_strokes
    crops = sorted({n for c in ("cotyledon", "hypocotyl", "radicle")
                    for n in (os.listdir(_REPO_ROOT / strokes / c / split)
                              if (_REPO_ROOT / strokes / c / split).is_dir() else [])})
    manifest = {}
    mp = _REPO_ROOT / folder / "manifest.csv"
    if mp.exists():
        with open(mp, newline="", encoding="utf-8") as fh:
            manifest = {os.path.basename(r["output_path"].replace("\\", "/")): (r["series"], r["crop_id"])
                        for r in csv.DictReader(fh)}
    images = [cv2.imread(str(_REPO_ROOT / folder / n), cv2.IMREAD_COLOR) for n in crops]
    maps = [load_strokes(str(_REPO_ROOT / strokes), n, im.shape)[0].labels for n, im in zip(crops, images)]
    lines = [f"{len(crops)} crops, {len({manifest.get(n) for n in crops})} seedlings ({folder}, {split} strokes)", ""]
    for ckpt in [c for c in spec.split("+") if c]:
        labels = _student(ckpt).segment_many_argmax(images)
        per_crop = [stroke_confusion(lab, sm) for lab, sm in zip(labels, maps)]
        lines.append(f"== {ckpt}")
        for name in ("cotyledon", "hypocotyl", "radicle"):
            tp, fp, fn = _sum3([c[name] for c in per_crop])
            p, r = _prf(tp, fp, fn)
            by_seed = {}
            for n, c in zip(crops, per_crop):
                by_seed.setdefault(manifest.get(n, n), []).append(c[name])
            per_seed = [_prf(*_sum3(cs)) for cs in by_seed.values()]
            lines.append(f"  {name:10s} pooled P {p:.3f} R {r:.3f} | per seedling P "
                         f"{np.nanmean([x[0] for x in per_seed]):.3f} R {np.nanmean([x[1] for x in per_seed]):.3f}"
                         f"  (stroke px {tp + fn})")
        rigs = {}
        for n, c in zip(crops, per_crop):
            rigs.setdefault(manifest.get(n, ("?", ""))[0], []).append(c["hypocotyl"])
        lines.append("  hypocotyl recall by rig: " + ", ".join(
            f"{rig} {_prf(*_sum3(cs))[1]:.2f} ({len(cs)})" for rig, cs in sorted(rigs.items())))
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", default="weights/multiclass/dlhook_4class_v1.pt")
    p.add_argument("--unannotated", type=int, default=150, help="Random non-annotated crops per folder")
    p.add_argument("--max-annotated", type=int, default=400, help="Cap of annotated frames per source CSV")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--score-strokes", default=None, metavar="CKPT[+CKPT]",
                   help="Score these checkpoints on the held-out strokes instead of the assessment")
    args = p.parse_args(argv)
    if args.score_strokes:
        text = score_strokes(args.score_strokes)
        out = _REPO_ROOT / args.out_dir
        out.mkdir(parents=True, exist_ok=True)
        (out / "strokes_score.txt").write_text(text, encoding="utf-8")
        print(text)
        return 0

    from models.UNetInference import get_predictor
    from utils.paths import repo_path
    teachers = {name: get_predictor(repo_path(path)) for name, path in TEACHERS.items()}
    student = _student(args.checkpoint)
    rows = []
    for folder, crops in crop_plan(args.unannotated, args.max_annotated, args.seed).items():
        names = sorted(crops)
        print(f"{folder}: {len(names)} crops ({sum(v is not None for v in crops.values())} annotated)", flush=True)
        for start in range(0, len(names), args.batch):
            chunk = names[start:start + args.batch]
            images = [cv2.imread(str(_REPO_ROOT / folder / n), cv2.IMREAD_COLOR) for n in chunk]
            keep = [(n, im) for n, im in zip(chunk, images) if im is not None]
            for (name, _), row in zip(keep, measure([im for _, im in keep], [crops[n] for n, _ in keep],
                                                    student, teachers)):
                row.update(folder=folder, crop_file=name)
                row["score"], row["reason"] = tricky_score(row)
                rows.append(row)

    out = _REPO_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    fields = ["folder", "crop_file", "annotated", "gap_teacher", "gap_student", "cover_teacher", "cover_student",
              "parts_student", "parts_teacher", "dice_cot", "dice_hypo", "score", "reason"]
    with open(out / "per_crop.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tricky = sorted((r for r in rows if r["score"] > 0), key=lambda r: -r["score"])
    with open(out / "tricky_crops.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["folder", "crop_file", "score", "reason"], extrasaction="ignore")
        w.writeheader()
        w.writerows(tricky)
    text = summarise(rows) + f"\n\ncheckpoint: {args.checkpoint}\ntricky crops: {len(tricky)} of {len(rows)}"
    (out / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
