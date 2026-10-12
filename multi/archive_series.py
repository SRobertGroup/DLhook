#!/usr/bin/env python
"""Archive a duplicate series from the crop sets, optionally keeping some seedlings under another series.

Written for `simpler_AGJKV`: 25 frames byte-identical to frames of `AGJKV system pictures`, so 14 of
its 17 seedlings are the same plants as AGJKV seedlings already in the data (docs/AUDIT.md, section 8).
The other 3 are real plants of that rig that have no twin; they are kept and renamed into the
target series, which is exact because the frames are identical files.

    python -m multi.archive_series --dry-run          # count what would change
    python -m multi.archive_series                    # do it

Per crop set (training / new / open) the tool

  * moves the crops of the archived seedlings (and, for the training set, their segmentation labels,
    probability maps, previews) into `<archive>/<set>/...`, and drops their rows from the manifest, the
    patch index, the angle / root / review / repeat / germination CSVs and the boxes file;
  * renames each kept seedling into the target series with the next free crop id: crop files, manifest
    rows, CSV rows (series, seedling_id, crop_id, frame, onset_frame, crop_file), the patch index, and the
    box list (appended to the target series, so the target's existing crop ids do not move);
  * copies every file it changes to `<archive>/_backup/` first, so the whole thing can be undone by
    copying that tree back over the repo.

`example_data/<series>` is never touched.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

SEG_SUFFIXES = ("cotyledon", "hypocotyl", "radicle")

DEFAULT_SETS = {
    "training": {
        "dir": "data/crops/cropped_training_set", "boxes": "data/crop_boxes/crop_boxes.json", "keep": [0],
        "csvs": ["data/annotations/angles/angle_landmarks_train.csv", "data/annotations/angles/angle_landmarks_train_root.csv",
                 "data/annotations/angles/angle_landmarks_train.csv.review.csv", "data/annotations/angles/angle_ground_truth.csv", "data/annotations/angles/angle_repeat.csv",
                 "data/annotations/germination/germination_train.csv"],
        "crop_dirs": ["data/crops/cropped_training_set_labels", "multi/processed/masks_multiclass_preview"],
        "seg_dir": "data/crops/cropped_training_set_seg",
        "patch_index": ["multi/processed/patch_index/train_patches.csv", "multi/processed/patch_index/val_patches.csv"],
    },
    "new": {
        "dir": "data/crops/cropped_new_set", "boxes": "data/crop_boxes/crop_boxes_new_filtered.json", "keep": [3],
        "csvs": ["data/annotations/angles/angle_new.csv", "data/annotations/angles/angle_new_root.csv", "data/annotations/angles/angle_repeat_new.csv", "data/annotations/germination/germination_new.csv"],
        "crop_dirs": [], "seg_dir": None, "patch_index": [],
    },
    "open": {
        "dir": "data/crops/cropped_open_set", "boxes": "data/crop_boxes/crop_boxes_open.json", "keep": [0],
        "csvs": ["data/annotations/angles/angle_open.csv"], "crop_dirs": [], "seg_dir": None, "patch_index": [],
    },
}


def prefix(series: str) -> str:
    """Series as it appears in crop file names: spaces become underscores."""
    return series.replace(" ", "_")


def read_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader), reader.fieldnames


def write_csv(path: Path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


class Plan:
    """Everything one run would do, collected first so a dry run can print it."""

    def __init__(self, root: Path, archive: Path, dry_run: bool):
        self.root, self.archive, self.dry_run = root, archive, dry_run
        self.moves = []          # (src, dst)
        self.writes = {}         # path -> (rows, fields) | ("json", obj)
        self.counts = {}

    def count(self, key, n=1):
        self.counts[key] = self.counts.get(key, 0) + n

    def backup_then_write(self, path: Path):
        rel = path.relative_to(self.root)
        dst = self.archive / "_backup" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(path, dst)


def crop_name(crop_id, series_prefix, frame_stem):
    return f"{crop_id}-crop-{series_prefix}_{frame_stem}.png"


def plan_set(plan: Plan, name: str, cfg: dict, series: str, target: str):
    root = plan.root
    set_dir = root / cfg["dir"]
    manifest_path = set_dir / "manifest.csv"
    rows, fields = read_csv(manifest_path)
    keep = set(cfg["keep"])
    target_ids = [int(r["crop_id"]) for r in rows if r["series"] == target]
    next_id = (max(target_ids) + 1) if target_ids else 0
    mapping = {}                                           # old crop_id -> new crop_id (kept seedlings)
    for old in sorted({int(r["crop_id"]) for r in rows if r["series"] == series}):
        if old in keep:
            mapping[old] = next_id
            next_id += 1
    sp, tp = prefix(series), prefix(target)

    # --- manifest + files ------------------------------------------------------------------
    new_rows, file_map = [], {}                            # file_map: old crop file -> new name or None (archive)
    for r in rows:
        if r["series"] != series:
            new_rows.append(r)
            continue
        old = int(r["crop_id"])
        stem = Path(r["source_frame"]).stem
        old_file = crop_name(old, sp, stem)
        if old in mapping:
            new_file = crop_name(mapping[old], tp, stem)
            file_map[old_file] = new_file
            new_rows.append({**r, "series": target, "crop_id": mapping[old],
                             "output_path": f"{cfg['dir']}\\{new_file}"})
            plan.count(f"{name}: crops renamed")
        else:
            file_map[old_file] = None
            plan.count(f"{name}: crops archived")
    plan.writes[manifest_path] = (new_rows, fields)

    def plan_file(directory: Path, old_file: str, new_file, label: str):
        src = directory / old_file
        if not src.exists():
            plan.count(f"{name}: {label} missing")
            return
        dst = (directory / new_file) if new_file else plan.archive / name / directory.name / old_file
        if new_file and dst.exists():
            raise SystemExit(f"refusing to rename over an existing file: {dst}")
        plan.moves.append((src, dst))

    for old_file, new_file in file_map.items():
        plan_file(set_dir, old_file, new_file, "crop")
        for d in cfg["crop_dirs"]:
            plan_file(root / d, old_file, new_file, Path(d).name)
        if cfg["seg_dir"]:
            stem, new_stem = old_file[:-4], (new_file[:-4] if new_file else None)
            for suf in SEG_SUFFIXES:
                plan_file(root / cfg["seg_dir"], f"{stem}-{suf}.png",
                          f"{new_stem}-{suf}.png" if new_stem else None, "seg")

    # --- patch index ----------------------------------------------------------------------
    for rel in cfg["patch_index"]:
        p = root / rel
        if not p.exists():
            continue
        prow, pfields = read_csv(p)
        out = []
        for r in prow:
            if r["filename"] in file_map:
                if file_map[r["filename"]]:
                    out.append({**r, "filename": file_map[r["filename"]]})
                plan.count(f"{name}: patch rows touched")
            else:
                out.append(r)
        plan.writes[p] = (out, pfields)

    # --- annotation CSVs -------------------------------------------------------------------
    for rel in cfg["csvs"]:
        p = root / rel
        if not p.exists():
            continue
        crows, cfields = read_csv(p)
        out = []
        for r in crows:
            if r.get("series") != series:
                out.append(r)
                continue
            old = int(r["crop_id"])
            if old not in mapping:
                plan.count(f"{rel}: rows archived")
                continue
            new = mapping[old]
            r = dict(r, series=target, crop_id=new)
            if "seedling_id" in r:
                r["seedling_id"] = new + 1
            for col in ("frame", "onset_frame"):
                if r.get(col, "").startswith(sp + "_"):
                    r[col] = tp + "_" + r[col][len(sp) + 1:]
            if r.get("crop_file"):
                r["crop_file"] = re.sub(rf"^{old}-crop-{re.escape(sp)}_", f"{new}-crop-{tp}_", r["crop_file"])
            out.append(r)
            plan.count(f"{rel}: rows renamed")
        plan.writes[p] = (out, cfields)

    # --- boxes ----------------------------------------------------------------------------
    # The manifest is the record of what was cut. A boxes file is only rewritten while it still agrees
    # with it (data/crop_boxes/crop_boxes_new_filtered.json no longer does); a stale one is left alone and reported.
    bp = root / cfg["boxes"]
    if bp.exists():
        boxes = json.loads(bp.read_text(encoding="utf-8"))
        man = {}
        for r in rows:
            man.setdefault((r["series"], int(r["crop_id"])), (int(r["cx"]), int(r["cy"]), int(r["half_w"]), int(r["half_h"])))
        agree = all(tuple(b[k] for k in ("cx", "cy", "half_w", "half_h")) == man.get((s_, i), None)
                    for s_ in (series, target) for i, b in enumerate(boxes.get(s_, [])))
        if not agree or len(boxes.get(target, [])) != len(set(target_ids)):
            print(f"  WARNING: {cfg['boxes']} no longer matches {cfg['dir']}/manifest.csv; left unchanged")
            plan.count(f"{name}: boxes file stale (unchanged)")
        else:
            boxes.pop(series, None)
            tgt = boxes.setdefault(target, [])
            for old, new_id in sorted(mapping.items()):
                x = man[(series, old)]
                tgt.append(dict(zip(("cx", "cy", "half_w", "half_h"), x)))
                plan.count(f"{name}: boxes moved")
            plan.writes[bp] = ("json", boxes)
    return mapping


def run(root: Path, series: str, target: str, archive: Path, dry_run: bool, sets=None):
    sets = sets or DEFAULT_SETS
    plan = Plan(root, archive, dry_run)
    mappings = {name: plan_set(plan, name, cfg, series, target) for name, cfg in sets.items()}
    print(f"\n{'DRY RUN: ' if dry_run else ''}archive {series!r} -> {archive}  (kept seedlings go to {target!r})")
    for name, mp in mappings.items():
        print(f"  {name}: kept {mp or 'none'}")
    for k, v in sorted(plan.counts.items()):
        print(f"  {k}: {v}")
    print(f"  files to move: {len(plan.moves)}   files to rewrite: {len(plan.writes)}")
    if dry_run:
        return mappings
    for path in plan.writes:
        plan.backup_then_write(path)
    for src, dst in plan.moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    for path, payload in plan.writes.items():
        if payload[0] == "json":
            path.write_text(json.dumps(payload[1], indent=2), encoding="utf-8")
        else:
            write_csv(path, *payload)
    print("done; originals of every rewritten file are under", archive / "_backup")
    return mappings


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--series", default="simpler_AGJKV")
    p.add_argument("--target-series", default="AGJKV system pictures", help="series the kept seedlings move into")
    p.add_argument("--archive", default="_archive_simpler_AGJKV")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    run(root, args.series, args.target_series, root / args.archive, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
