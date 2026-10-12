"""Score the learned "measurable" head against the readability gate it replaces (audit M-10).

    python -m multi.validate_measurable --fit           # threshold from the training runs' held-out seedlings
    python -m multi.validate_measurable                 # score on BerryBox v2 val (no model has seen it)

The landmark ensemble with the head (default: the three multi/results/models_landmarks_all_*_m
checkpoints) reads every annotated frame of a set; a frame the annotator measured is a positive, one
they skipped (or the review marked unmeasurable) a negative.

--fit rebuilds the held-out seedlings of the training configs (same sources, same split_seed as
training, multi/src/landmarks.py) and writes multi/results/measurable_fit.json: the threshold on
1 - measurable_p that keeps --keep (default 98%) of their measured crops. That value goes into
utils/landmark_gate.py:MEASURABLE_THRESHOLD.

Scoring reports, for the head and for the logistic gate (the GUI's gate for models without the
head; same landmark maps, so a like-for-like comparison): AUC, skipped frames rejected and measured
frames kept at each gate's own threshold, the angle error of the measured frames kept, and the skipped
frames rejected at the threshold that keeps --keep of this set's measured frames (one operating point
for both gates).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT, _REPO_ROOT / "multi"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DEFAULT_CHECKPOINTS = "+".join(f"multi/results/models_landmarks_all_{m}_m/best.pt" for m in "abc")
DEFAULT_CONFIG = "multi/configs/training_landmarks_all_a_m.yaml"
FIT_PATH = _REPO_ROOT / "multi" / "results" / "measurable_fit.json"


def _ensemble(spec):
    from models.UNetInference import MARGIN
    from models.multiclass_inference import LandmarkEnsemble
    return LandmarkEnsemble.from_checkpoints(spec, num_classes=4, in_size=252 + 2 * MARGIN, out_size=252, margin=MARGIN)


def held_out_pools(config_path):
    """({name: path} measured, {name: path} skipped) crops of the held-out seedlings of a training
    config -- the same pools the training run validated the measurable head on."""
    import csv

    from src.config import load_config, resolved_path
    from src.landmarks import load_landmarks, load_measurable_negatives, split_by_seedling
    from train_unet_multiclass import load_extra_landmark_sources

    config = load_config(config_path)
    lm_cfg, data_cfg = config["landmarks"], config["data"]
    raw_dir = resolved_path(config, "raw_data_dir")
    with open(resolved_path(config, "patch_index_dir") / "val_patches.csv", newline="", encoding="utf-8") as fh:
        val_names = {row["filename"] for row in csv.DictReader(fh)}
    landmarks, _ = load_landmarks(resolved_path(config, "csv", section="landmarks"), exclude_filenames=val_names)
    negatives = {}
    extra, raw_paths, _ = load_extra_landmark_sources(config, lm_cfg.get("extra_sources"), landmarks,
                                                      int(data_cfg.get("patch_size", 252)), 252, negatives=negatives)
    landmarks = {**landmarks, **extra}
    for key, seedling in load_measurable_negatives(resolved_path(config, "csv", section="landmarks"),
                                                   exclude_filenames=val_names).items():
        if key not in landmarks and (raw_dir / key).exists():
            negatives[key] = seedling
            raw_paths[key] = raw_dir / key
    _, lm_val = split_by_seedling(landmarks, lm_cfg.get("val_fraction", 0.15),
                                  lm_cfg.get("split_seed", config["training"].get("seed", 0)))
    val_seedlings = {lm.seedling for lm in lm_val.values()}
    pos = {k: raw_paths.get(k) or raw_dir / k for k in lm_val}
    neg = {k: raw_paths[k] for k, s in negatives.items() if s in val_seedlings and k not in landmarks}
    return pos, neg


def annotated_set(csv_path, crops_dir):
    """[(crop path, is_measured, truth theta or None)] for the measured / skipped frames of a set."""
    from utils.angle_review import UNMEASURABLE, read_rows

    out, seen = [], set()
    for row in read_rows(csv_path):
        status = row.get("status")
        name = row.get("crop_file")
        if status not in ("measured", "skipped", UNMEASURABLE) or not name or name in seen:
            continue
        seen.add(name)
        path = Path(crops_dir) / name
        if not path.exists():
            continue
        theta = float(row["theta"]) if status == "measured" and row.get("theta") not in (None, "") else None
        out.append((path, status == "measured", theta))
    return out


def _read(ensemble, paths, batch=16):
    """Per path: (measurable p or None, readout or None) from one pass of the ensemble."""
    import cv2

    from utils.landmark_gate import annotate_readout
    from utils.landmark_readout import READOUT_RADIUS, readout_from_fields

    out = []
    for start in range(0, len(paths), batch):
        chunk = paths[start:start + batch]
        images = [cv2.imread(str(p)) for p in chunk]
        per_member = [m.predict_landmark_fields(images) for m in ensemble.members]
        from models.multiclass_inference import merge_landmark_fields
        for group in zip(*per_member):
            hm, vec, extra = merge_landmark_fields(list(group))
            readout = readout_from_fields(hm, vec, radius=READOUT_RADIUS, **(extra or {}))
            if readout is not None:
                thetas = []
                for m_hm, m_vec, m_extra in group:
                    r = readout_from_fields(m_hm, m_vec, radius=READOUT_RADIUS, **(m_extra or {}))
                    thetas.append(None if r is None else float(r["theta"]))
                annotate_readout(readout, (extra or {}).get("tip_hm"), thetas)
            out.append(((extra or {}).get("measurable"), readout))
    return out


def _auc(labels, scores):
    from train_unet_multiclass import _auc as auc
    return auc(labels, scores)


def fit(args):
    pos, neg = held_out_pools(args.config)
    ensemble = _ensemble(args.checkpoints)
    names = sorted(pos) + sorted(neg)
    reads = _read(ensemble, [pos.get(n) or neg[n] for n in names])
    scores = {n: m for n, (m, _) in zip(names, reads) if m is not None}
    p_pos = np.array([scores[n] for n in pos if n in scores])
    p_neg = np.array([scores[n] for n in neg if n in scores])
    # unreadable when 1 - p >= threshold; keep `keep` of the measured crops
    threshold = float(np.quantile(1.0 - p_pos, args.keep))
    result = {"threshold": threshold, "keep_fraction": args.keep,
              "auc": _auc([1] * len(p_pos) + [0] * len(p_neg), list(p_pos) + list(p_neg)),
              "skipped_rejected": float(np.mean(1.0 - p_neg >= threshold)) if len(p_neg) else None,
              "n_measured": int(len(p_pos)), "n_skipped": int(len(p_neg)),
              "checkpoints": args.checkpoints, "config": args.config}
    FIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {FIT_PATH} -- set utils/landmark_gate.py: MEASURABLE_THRESHOLD = {threshold:.4f}")


def score(args):
    from utils.landmark_gate import GATE_THRESHOLD, unreadable_probability

    threshold = args.threshold
    if threshold is None:
        threshold = json.loads(FIT_PATH.read_text(encoding="utf-8"))["threshold"] if FIT_PATH.exists() else 0.5
    items = annotated_set(args.truth, args.crops)
    ensemble = _ensemble(args.checkpoints)
    reads = _read(ensemble, [p for p, _, _ in items])

    rows = []
    for (path, measured, truth), (m, readout) in zip(items, reads):
        logistic = (unreadable_probability(readout["peak"], readout.get("tip_peak"), readout["member_theta_sd"])
                    if readout else 1.0)
        err = None
        if measured and truth is not None and readout is not None:
            err = abs(float(readout["theta"]) - truth)
        rows.append({"measured": measured, "head": None if m is None else 1.0 - m, "logistic": logistic, "err": err})

    def report(key, thr):
        have = [r for r in rows if r[key] is not None]
        lab = [r["measured"] for r in have]
        auc = _auc(lab, [-r[key] for r in have])
        meas = [r for r in have if r["measured"]]
        skip = [r for r in have if not r["measured"]]
        kept = [r for r in meas if r[key] < thr]
        errs = [r["err"] for r in kept if r["err"] is not None]
        # same operating point for every gate: the threshold on this set that keeps args.keep of the measured
        at_keep = float(np.quantile([r[key] for r in meas], args.keep)) if meas else math.nan
        return {"auc": auc, "threshold": thr,
                f"skipped_rejected_at_{args.keep:.0%}_kept": (float(np.mean([r[key] > at_keep for r in skip]))
                                                             if skip else math.nan),
                "skipped_rejected": float(np.mean([r[key] >= thr for r in skip])) if skip else math.nan,
                "measured_kept": len(kept) / len(meas) if meas else math.nan,
                "kept_theta_mae": float(np.mean(errs)) if errs else math.nan,
                "n_measured": len(meas), "n_skipped": len(skip)}

    result = {"set": str(args.truth), "head": report("head", threshold),
              "logistic_gate": report("logistic", GATE_THRESHOLD)}
    all_errs = [r["err"] for r in rows if r["err"] is not None]
    result["all_measured_theta_mae"] = float(np.mean(all_errs)) if all_errs else math.nan
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoints", default=DEFAULT_CHECKPOINTS, help="Ensemble checkpoints joined by '+'")
    p.add_argument("--fit", action="store_true", help="Fit the threshold on the training runs' held-out seedlings")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="Training config whose held-out seedlings --fit uses")
    p.add_argument("--keep", type=float, default=0.98, help="Share of measured crops the threshold keeps (--fit; the scoring's common operating point)")
    p.add_argument("--truth", default="data/annotations/angles/angle_berrybox_val.csv")
    p.add_argument("--crops", default="data/crops/cropped_berrybox_val")
    p.add_argument("--threshold", type=float, default=None, help="Head threshold (default: the --fit result)")
    p.add_argument("--out", default="multi/results/measurable_validation.json")
    args = p.parse_args(argv)
    fit(args) if args.fit else score(args)


if __name__ == "__main__":
    main()
