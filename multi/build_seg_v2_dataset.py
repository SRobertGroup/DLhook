"""Build the training set of the four-class v2 (one tagged folder, labels with your strokes on top).

    python -m multi.build_seg_v2_dataset            # then: python multi/build_patch_index.py --config multi/configs/training_config_v2.yaml

For every source crop folder (SOURCES; BerryBox v2 val is held out -- its strokes only score v2):
  1. teacher probability maps (multi/segment_crops.py, skips crops already done) in <folder>_seg/;
  2. pseudo-labels with the same thresholds as v1 (mask_merge.merge_dataset_from_probs), with the
     folder's strokes from ui/seg_annotator.py burnt on top (data/annotations/seg_strokes/<folder>);
  3. on hand-annotated frames WITHOUT strokes, an ignore band (255) from the clicked junction to the
     teacher hypocotyl: the teachers leave the hook region out (multi/assess_seg_v2.py), so the student
     is not taught that it is background -- only strokes say what it is;
  4. crops and labels copied to data/crops/cropped_seg_v2/{images,labels}/<tag>__<crop>, with a merged
     manifest.csv (series prefixed with the tag, so `data.split_level: seedling` keeps plants apart).
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT, _REPO_ROOT / "multi"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SOURCES = [  # (crop folder, tag) -- the landmark configs' tags
    ("data/crops/cropped_training_set", "train"),
    ("data/crops/cropped_new_set", "new"),
    ("data/crops/cropped_open_set", "open"),
    ("data/crops/cropped_open_set_2", "open2"),
    ("data/crops/cropped_validation_set", "val"),
    ("data/crops/cropped_berrybox_train", "bb"),
    ("data/crops/cropped_round5", "r5"),
]
OUT = "data/crops/cropped_seg_v2"
STROKES = "data/annotations/seg_strokes"
THRESHOLDS = {"cotyledon": (0.05, 0.5), "hypocotyl": (0.05, 0.5), "radicle": (0.02, 0.3)}   # training_config.yaml
CONFLICT_MARGIN = 0.15
BAND_HALF_WIDTH = 4          # px each side of the junction -> hypocotyl line
MIN_GAP = 2.0                # px: smaller gaps need no band
HYPOCOTYL, IGNORE = 2, 255


def ignore_band(label, junction, half_width=BAND_HALF_WIDTH, min_gap=MIN_GAP):
    """Set to IGNORE the background pixels of a thick line from `junction` to the nearest hypocotyl
    pixel of `label` (in place). Cotyledon, hypocotyl and radicle pixels are never touched. Returns the
    number of pixels changed (0 when there is no hypocotyl or the gap is below `min_gap`)."""
    hyp = label == HYPOCOTYL
    if not hyp.any():
        return 0
    ys, xs = np.nonzero(hyp)
    d = np.hypot(xs - junction[0], ys - junction[1])
    k = int(np.argmin(d))
    if d[k] < min_gap:
        return 0
    band = np.zeros(label.shape, np.uint8)
    a = (int(round(junction[0])), int(round(junction[1])))
    cv2.line(band, a, (int(xs[k]), int(ys[k])), 1, thickness=2 * half_width + 1)
    hit = (band > 0) & (label == 0)
    label[hit] = IGNORE
    return int(hit.sum())


def _manifest(folder):
    path = _REPO_ROOT / folder / "manifest.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {os.path.basename(r["output_path"].replace("\\", "/")): r for r in csv.DictReader(fh)}


def _junctions(folder):
    """{crop: (x, y)} of the hand-clicked junctions of a folder (every annotated source)."""
    from utils.landmark_marks import annotated_sources, load_annotations
    out = {}
    for csv_path, root_csv, src in annotated_sources():
        if Path(src) == Path(folder):
            out.update({crop: a["junction"] for crop, a in load_annotations(csv_path, root_csv, src).items()})
    return out


def build(sources=SOURCES, out=OUT, strokes_root=STROKES, segment=True, python=sys.executable):
    from src.mask_merge import merge_dataset_from_probs
    from utils.seg_strokes import annotations_dirs

    out = _REPO_ROOT / out
    images, labels, work = out / "images", out / "labels", out / "_work"
    for d in (images, labels):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    rows, report = [], []
    for folder, tag in sources:
        src = _REPO_ROOT / folder
        seg = src.parent / (src.name + "_seg")
        if segment:
            subprocess.run([python, str(_REPO_ROOT / "multi" / "segment_crops.py"), "--crops", str(src),
                            "--out", str(seg)], check=True)
        stroke_dir = _REPO_ROOT / strokes_root / src.name
        has_strokes = stroke_dir.is_dir()
        lab_dir = work / tag
        merge_dataset_from_probs(raw_dir=src, prob_dir=seg, out_dir=lab_dir, report_path=work / f"{tag}_merge.csv",
                                 thresholds=THRESHOLDS, conflict_margin=CONFLICT_MARGIN, force=True,
                                 annotations_dirs=annotations_dirs(str(stroke_dir)) if has_strokes else None)
        stroked = set()
        if has_strokes:
            for c in ("cotyledon", "hypocotyl", "radicle"):
                for split in ("train", "val"):
                    p = stroke_dir / c / split
                    if p.is_dir():
                        stroked |= set(os.listdir(p))
        junctions, manifest = _junctions(folder), _manifest(folder)
        n_band = n_crops = 0
        for lab_path in sorted(lab_dir.glob("*.png")):
            name = lab_path.name
            label = cv2.imread(str(lab_path), cv2.IMREAD_UNCHANGED)
            if name not in stroked and name in junctions:
                n_band += int(ignore_band(label, junctions[name]) > 0)
            new = f"{tag}__{name}"
            cv2.imwrite(str(labels / new), label)
            shutil.copy2(src / name, images / new)
            m = manifest.get(name, {})
            rows.append({"output_path": f"{OUT}/images/{new}", "series": f"{tag}/{m.get('series', '')}",
                         "source_frame": m.get("source_frame", ""), "crop_id": m.get("crop_id", "")})
            n_crops += 1
        report.append(f"{tag:6s} {folder}: {n_crops} crops, {len(stroked)} with strokes, {n_band} with an ignore band")
        print(report[-1], flush=True)
    with open(images / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["output_path", "series", "source_frame", "crop_id"])
        w.writeheader()
        w.writerows(rows)
    shutil.rmtree(work, ignore_errors=True)
    (out / "build_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--no-segment", action="store_true", help="Reuse existing <folder>_seg maps (fail if missing)")
    args = p.parse_args(argv)
    rows = build(segment=not args.no_segment)
    print(f"{len(rows)} crops -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
