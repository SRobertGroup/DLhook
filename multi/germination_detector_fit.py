#!/usr/bin/env python
"""Learn germination onset from the hand-marked onsets (ui/germination_annotator.py).

Per frame, a few statistics of the four-class model's softmax maps (how much radicle,
how much seedling) go into a logistic regression for "is a radicle visible in this
frame?". Per seedling, the frames are then explained by ONE onset: the step function
(no radicle before frame k, radicle from k on) that best fits the per-frame
probabilities. That is the temporal part -- a radicle never disappears again, so a
single noisy frame cannot flip the answer.

Scored by cross-validation over whole seedlings, against the app's current rule
(utils/germination_detector.py thresholding the radicle area) on the same frames:

    python -m multi.germination_detector_fit [--truth germination_train.csv] [--folds 5]

The fitted weights go to weights/germination_detector.json (use --no-save to skip).
Nothing in the GUI uses it yet. Features are cached in
multi/results/germination_validation/frame_features.csv (delete it to recompute).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.germination_learned import FEATURE_NAMES, best_onset, frame_features  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = _REPO_ROOT / "multi" / "results" / "germination_validation"


def extract_features(seedlings, folder, checkpoint, cache):
    """{(series, crop_id): array (n_frames, n_features)}, cached on disk."""
    if cache.exists():
        out = {}
        with open(cache, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        grouped = {}
        for r in rows:
            grouped.setdefault((r["series"], int(r["crop_id"])), []).append(r)
        for key, rs in grouped.items():
            rs.sort(key=lambda r: int(r["frame_index"]))
            out[key] = np.array([[float(r[n]) for n in FEATURE_NAMES] for r in rs])
        if all(s.key in out and len(out[s.key]) == len(s) for s in seedlings):
            return out
    import cv2
    from models.UNetInference import MARGIN
    from models.multiclass_inference import MulticlassInference

    model = MulticlassInference(checkpoint, num_classes=4, in_size=264, out_size=252, margin=MARGIN)
    out, rows = {}, []
    for n, s in enumerate(seedlings):
        images = [cv2.imread(os.path.join(folder, f[1])) for f in s.frames]
        probs = model.segment_many_argmax(images, return_probs=True)
        feats = np.array([frame_features(p) for p in probs])
        out[s.key] = feats
        for i, row in enumerate(feats):
            rows.append({"series": s.series, "crop_id": s.crop_id, "frame_index": i,
                         **{name: f"{v:.6f}" for name, v in zip(FEATURE_NAMES, row)}})
        if n % 20 == 0:
            print(f"  features {n + 1}/{len(seedlings)}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["series", "crop_id", "frame_index"] + FEATURE_NAMES)
        w.writeheader()
        w.writerows(rows)
    return out


def fit_logistic(x, y, l2=1.0, iters=200):
    """Standardised logistic regression with L2, by LBFGS (torch). Returns (mean, std, weights, bias)."""
    import torch
    mean, std = x.mean(0), x.std(0) + 1e-6
    xt = torch.tensor((x - mean) / std, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(xt.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xt @ w + b, yt) + l2 * 1e-3 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return mean, std, w.detach().numpy(), float(b.detach())


def predict_probs(model, x):
    mean, std, w, b = model
    z = ((x - mean) / std) @ w + b
    return 1.0 / (1.0 + np.exp(-z))


def threshold_onset(areas, thr, window=3, hits=2):
    """The app's rule (utils/germination_detector.py) on a per-frame area series, seed-independent."""
    n = len(areas)
    for i in range(n):
        if areas[i] >= thr and sum(areas[k] >= thr for k in range(i, min(n, i + window))) >= hits:
            return i
    return None


def summarise(errors, missed, n):
    e = np.array(errors, float)
    if not len(e):
        return f"found 0/{n}"
    return (f"found {len(e)}/{n}   exact {np.mean(e == 0):4.0%}   +-1 {np.mean(np.abs(e) <= 1):4.0%}   "
            f"MAE {np.mean(np.abs(e)):.2f}   bias {np.mean(e):+.2f}   early {int((e < 0).sum())}  late {int((e > 0).sum())}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--truth", default=str(_REPO_ROOT / "germination_train.csv"))
    p.add_argument("--folder", default=str(_REPO_ROOT / "cropped_training_set"))
    p.add_argument("--checkpoint", default="weights/multiclass/dlhook_4class_v1.pt")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--group", choices=("seedling", "series"), default="seedling",
                   help="cross-validate over held-out seedlings (default) or held-out whole series")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--apply", default=None, metavar="WEIGHTS_JSON",
                   help="Do not fit: score saved weights (e.g. weights/germination_detector.json) on --truth / "
                        "--folder -- the out-of-sample test on seedlings the detector never saw")
    p.add_argument("--out", default=str(_REPO_ROOT / "weights" / "germination_detector.json"))
    args = p.parse_args(argv)
    sys.path.insert(0, str(_REPO_ROOT))
    os.chdir(_REPO_ROOT)
    from multi.validate_germination import load_truth
    from utils.angle_annotation import default_manifest
    from utils.germination_annotation import load_seedlings

    truth = {k: v for k, v in load_truth(args.truth).items() if v["status"] == "found"}
    seedlings = [s for s in load_seedlings(args.folder, default_manifest(args.folder)) if s.key in truth]
    print(f"{len(seedlings)} seedlings with a marked onset, {sum(len(s) for s in seedlings)} frames")
    folder_name = Path(args.folder).name
    cache = FEATURE_DIR / ("frame_features.csv" if folder_name == "cropped_training_set"
                           else f"frame_features_{folder_name}.csv")
    feats = extract_features(seedlings, args.folder, args.checkpoint, cache)

    def labels(s):
        return (np.arange(len(s)) >= truth[s.key]["onset_index"]).astype(float)

    n = len(seedlings)
    if args.apply:
        saved = json.loads(Path(args.apply).read_text(encoding="utf-8"))
        if saved["features"] != FEATURE_NAMES:
            raise SystemExit(f"{args.apply} was fitted on different features: {saved['features']}")
        model = (np.array(saved["mean"]), np.array(saved["std"]), np.array(saved["weights"]), float(saved["bias"]))
        errors, frame_ok = [], []
        for s in seedlings:
            probs = predict_probs(model, feats[s.key])
            frame_ok.append(((probs > 0.5) == labels(s).astype(bool)).mean())
            k = best_onset(probs)
            if k < len(s):
                errors.append(k - truth[s.key]["onset_index"])
        print(f"\nsaved detector ({args.apply}, fitted on {saved.get('trained_on')}) on {os.path.basename(args.truth)}:")
        print("  " + summarise(errors, n - len(errors), n))
        print(f"  per-frame accuracy {np.mean(frame_ok):.1%}")
        area = {s.key: np.expm1(feats[s.key][:, FEATURE_NAMES.index("r_log_n50")]) for s in seedlings}
        for thr in (5, 20, 50):
            errs = [threshold_onset(area[s.key], thr) for s in seedlings]
            errs = [d - truth[s.key]["onset_index"] for s, d in zip(seedlings, errs) if d is not None]
            print(f"area rule, radicle pixels >= {thr:3d} anywhere in the crop:")
            print("  " + summarise(errs, n - len(errs), n))
        return 0

    rng = np.random.RandomState(0)
    if args.group == "series":
        # leave-series-out: every fold holds out whole series (rigs, plates), the honest test for a new experiment
        names = sorted({s.series for s in seedlings})
        rng.shuffle(names)
        fold_of = {name: i % args.folds for i, name in enumerate(names)}
        folds = [np.array([i for i, s in enumerate(seedlings) if fold_of[s.series] == f]) for f in range(args.folds)]
        folds = [h for h in folds if len(h)]
    else:
        order = rng.permutation(len(seedlings))
        folds = [order[i::args.folds] for i in range(args.folds)]
    learned_err, rule_err, rule_missed = [], {}, {}
    probs_all = {}
    for f, held in enumerate(folds):
        held_set = set(held.tolist())
        train = [seedlings[i] for i in range(len(seedlings)) if i not in held_set]
        model = fit_logistic(np.concatenate([feats[s.key] for s in train]),
                             np.concatenate([labels(s) for s in train]), args.l2)
        for i in held:
            s = seedlings[i]
            probs_all[s.key] = predict_probs(model, feats[s.key])
            k = best_onset(probs_all[s.key])
            learned_err.append(k - truth[s.key]["onset_index"] if k < len(s) else None)

    detected = [e for e in learned_err if e is not None]
    print("\nlearned detector (cross-validated, held-out " + ("whole series" if args.group == "series" else "seedlings") + "):")
    print("  " + summarise(detected, n - len(detected), n))
    total_mass_area = {s.key: np.expm1(feats[s.key][:, FEATURE_NAMES.index("r_log_n50")]) for s in seedlings}
    for thr in (5, 20, 50):
        errs = []
        for s in seedlings:
            d = threshold_onset(total_mass_area[s.key], thr)
            if d is not None:
                errs.append(d - truth[s.key]["onset_index"])
        print(f"area rule, radicle pixels >= {thr:3d} anywhere in the crop:")
        print("  " + summarise(errs, n - len(errs), n))
    frame_ok = np.mean([((probs_all[s.key] > 0.5) == labels(s).astype(bool)).mean() for s in seedlings])
    print(f"\nper-frame accuracy of the classifier: {frame_ok:.1%}")

    model = fit_logistic(np.concatenate([feats[s.key] for s in seedlings]),
                         np.concatenate([labels(s) for s in seedlings]), args.l2)
    print("weights (standardised features): " + ", ".join(f"{n}={w:+.2f}" for n, w in zip(FEATURE_NAMES, model[2])))
    if not args.no_save:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "features": FEATURE_NAMES, "mean": model[0].tolist(), "std": model[1].tolist(),
            "weights": model[2].tolist(), "bias": model[3], "checkpoint": args.checkpoint,
            "trained_on": os.path.basename(args.truth), "n_seedlings": n}, indent=1), encoding="utf-8")
        print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
