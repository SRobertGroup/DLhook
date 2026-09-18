#!/usr/bin/env python
"""Headless batch re-cropper: harvest additional per-seedling training crops
from full plate frames in example_data/<series>/, using the exact same
crop-box geometry the GUI uses (see multi/src/recrop_geometry.py), without
constructing any Tk widgets or loading CUDA/superres models.

This is a data-prep CLI, run from the dlhook_env (needs cv2/numpy/skimage/
PIL -- the same stack utils/preprocess_model_input.py uses). It does NOT
import seedling_measurment.py/ui/* (Tk + heavy model imports at class-init
time) or utils/mask_store.py -- per CLAUDE.md, those are off limits.

Automatic row/seedling detection has been removed. Crop geometry (cx, cy,
half_w, half_h per seedling per series) must now come from --boxes-file, a
JSON file of {"series_name": [{"cx":.., "cy":.., "half_w":.., "half_h":..}, ...]}.
Draw crop boxes with the manual crop-drawing tool (`python -m multi.draw_crops`)
and pass its output here with --boxes-file -- that is the only way to supply
crop geometry now.

Only every Nth frame of each series is cropped by default (--every-n, default
5), since consecutive time-series frames are near-duplicates -- sampling
keeps the training set diverse and cuts inference cost a lot. Frames are
sampled from the TIME-ORDERED list (EXIF capture time where usable, natural
sort otherwise -- see order_series_for_sampling), not a lexicographic one:
example_data/MB's filenames are not zero-padded ("MB_1_0.jpg" .. "MB_1_64.jpg"),
so plain sorting would misorder them. The last frame of each series is
always included regardless of stride, since that is the frame --boxes-file's
geometry was drawn on.

Usage:
    python multi/recrop_plates.py --boxes-file my_boxes.json --dry-run
    python multi/recrop_plates.py --boxes-file my_boxes.json --series F1_Plate_2_YS,HR_Plate_5
    python multi/recrop_plates.py --boxes-file my_boxes.json --output-dir training_dataset/dlhook_recrop
    python multi/recrop_plates.py --boxes-file my_boxes.json --every-n 1   # every frame, no sampling

See `--help` for the remaining flags.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

# multi/ on sys.path so `from src...` resolves to multi/src (matches
# multi/build_patch_index.py's convention); the repo root on sys.path so
# `utils.preprocess_model_input` (a plain, Tk-free utility) is importable
# without importing seedling_measurment.py itself.
_MULTI_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MULTI_DIR.parent
sys.path.insert(0, str(_MULTI_DIR))
sys.path.insert(0, str(_REPO_ROOT))

import cv2  # noqa: E402

from src.recrop_geometry import (  # noqa: E402
    build_output_filename,
    crop_from_box,
    list_image_files,
)
from utils.preprocess_model_input import order_series, preprocess_images  # noqa: E402

PROTECTED_DIR = (_REPO_ROOT / "training_dataset" / "dlhook").resolve()

MANIFEST_FIELDS = [
    "output_path", "series", "source_frame", "crop_id",
    "cx", "cy", "half_w", "half_h",
    "x1", "y1", "x2", "y2",
    "crop_width", "crop_height",
]


class CollisionError(Exception):
    """Two planned crops would write to the same output path."""


class SeriesPlan:
    """Everything needed to crop one series: its frames and its fixed set of
    crop boxes (reused for every frame, like the GUI), sourced from
    --boxes-file."""

    def __init__(self, name, path, frames, boxes, source):
        self.name = name
        self.path = path
        self.frames = frames
        self.boxes = boxes
        self.source = source


def discover_series(input_dir: Path, only=None) -> dict:
    """Immediate subdirectories of `input_dir` that contain at least one
    image file -- each such directory is one plate series. `only`, if given,
    restricts to these series names (case-sensitive, matching the directory
    basename)."""
    series = {}
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input dir not found: {input_dir}")

    for entry in sorted(input_dir.iterdir()):
        if not entry.is_dir():
            continue
        if only is not None and entry.name not in only:
            continue
        if list_image_files(str(entry)):
            series[entry.name] = entry

    return series


def load_boxes_file(path: str) -> dict:
    """A JSON file of {"series_name": [{"cx":.., "cy":.., "half_w":.., "half_h":..}, ...]},
    the only way to supply crop boxes now that automatic detection is gone
    (e.g. saved from the manual crop-drawing tool, `python -m multi.draw_crops`,
    or from a previous GUI session)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    for series_name, boxes in data.items():
        for box in boxes:
            missing = {"cx", "cy", "half_w", "half_h"} - set(box)
            if missing:
                raise ValueError(
                    f"boxes-file entry for series {series_name!r} is missing {missing}: {box!r}"
                )
    return data


_NATSORT_RE = re.compile(r'(\d+)')


def natural_sort_key(filename: str):
    """True natural-sort key: alternating text/digit runs, digit runs
    compared numerically. Unlike order_series's own EXIF-unusable fallback
    (utils.preprocess_model_input.filename_sort_key), which keys on only the
    FIRST run of digits in the name, this handles non-zero-padded series
    correctly -- e.g. example_data/MB's "MB_1_0.jpg" .. "MB_1_64.jpg" all
    share the same first digit run ("1"), so filename_sort_key degenerates
    to a lexicographic tiebreak (0, 1, 10, 11, ..., 2, 20, ...) while this
    key sorts "MB_1_9.jpg" before "MB_1_10.jpg" as intended."""
    parts = _NATSORT_RE.split(filename)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def order_series_for_sampling(path: str, filenames: list) -> list:
    """Time-order a series' frames for deterministic --every-n sampling.

    Prefers embedded EXIF/TIFF capture times via
    utils.preprocess_model_input.order_series. When those aren't usable
    (any frame missing a timestamp, or every frame sharing one) -- the same
    "usable" check order_series applies internally before it falls back --
    this uses a full natural sort (natural_sort_key) instead of trusting
    order_series's own fallback, which is exactly the first-digit-run key
    that misorders non-zero-padded series like MB (see natural_sort_key)."""
    if not filenames:
        return []
    ordered, capture_times = order_series(path, filenames)
    times = list(capture_times.values())
    usable = all(t is not None for t in times) and len(set(times)) > 1
    if usable:
        return ordered
    return sorted(filenames, key=natural_sort_key)


def sample_every_n(ordered_frames: list, every_n: int) -> list:
    """Keep every Nth frame of a time-ordered list (0-indexed: positions
    0, N, 2N, ...), always including the last frame regardless of stride --
    that's the frame --boxes-file's geometry was drawn on, so it must be
    the one crop guaranteed to match. `every_n=1` returns every frame.
    Preserves time order; raises ValueError for every_n < 1."""
    if every_n < 1:
        raise ValueError(f"--every-n must be >= 1, got {every_n}")
    if not ordered_frames:
        return []
    indices = set(range(0, len(ordered_frames), every_n))
    indices.add(len(ordered_frames) - 1)
    return [ordered_frames[i] for i in sorted(indices)]


def read_preprocessed_gray(path: str):
    """Read a frame and apply the exact same preprocessing
    (grayscale + median blur + contrast + normalize) the GUI applies before
    cropping -- so these crops land in the same pixel domain as the existing
    training_dataset/dlhook crops."""
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"failed to read image: {path}")
    return preprocess_images().preprocess(img)


def plan_series(name: str, path: Path, frames, boxes_map: dict, max_frames=None):
    """Build a SeriesPlan for one series from its saved crop boxes. Caller
    is responsible for having already confirmed `name` is in `boxes_map`."""
    if max_frames is not None:
        frames = frames[:max_frames]

    return SeriesPlan(name, path, frames, boxes_map[name], "saved boxes file")


def box_sizes(boxes):
    return [(2 * b["half_w"], 2 * b["half_h"]) for b in boxes]


def build_plan_entries(series_plan: SeriesPlan):
    """(frame, box, crop_id) triples for one series."""
    entries = []
    for frame in series_plan.frames:
        for crop_id, box in enumerate(series_plan.boxes):
            entries.append((frame, box, crop_id))
    return entries


def check_collisions(planned):
    """`planned` is an iterable of (series, frame, crop_id, output_path).
    Raises CollisionError, listing every conflicting group, if any
    output_path is claimed by more than one entry -- including the
    case where two series' sanitized names collide."""
    by_path = {}
    for series, frame, crop_id, output_path in planned:
        key = os.path.normcase(str(output_path))
        by_path.setdefault(key, []).append((series, frame, crop_id, output_path))

    conflicts = {k: v for k, v in by_path.items() if len(v) > 1}
    if conflicts:
        lines = ["Refusing to run: colliding output filenames detected:"]
        for key, entries in conflicts.items():
            lines.append(f"  {entries[0][3]}:")
            for series, frame, crop_id, _ in entries:
                lines.append(f"    series={series!r} frame={frame!r} crop_id={crop_id}")
        raise CollisionError("\n".join(lines))


def _fmt_size_stats(sizes):
    if not sizes:
        return "no boxes detected"
    widths = sorted(w for w, _h in sizes)
    heights = sorted(h for _w, h in sizes)

    def stats(values):
        n = len(values)
        return f"min={values[0]} median={values[n // 2]} max={values[-1]}"

    return f"width[{stats(widths)}] height[{stats(heights)}]"


def run_dry_run(series_plans, output_dir: Path):
    total_crops = 0
    all_sizes = []
    planned_for_collisions = []

    print(f"{'series':<28} {'frames':>7} {'boxes':>6} {'crops':>8}  source")
    for plan in series_plans:
        boxes = plan.boxes
        n_boxes = len(boxes)
        n_crops = n_boxes * len(plan.frames)
        sizes = box_sizes(boxes)
        for frame in plan.frames:
            for crop_id in range(n_boxes):
                out_name = build_output_filename(crop_id, plan.name, frame)
                planned_for_collisions.append((plan.name, frame, crop_id, output_dir / out_name))

        total_crops += n_crops
        all_sizes.extend(sizes)
        print(f"{plan.name:<28} {len(plan.frames):>7} {n_boxes:>6} {n_crops:>8}  {plan.source}")
        print(f"    size (w,h) px: {_fmt_size_stats(sizes)}")

    print()
    print(f"TOTAL new crops across {len(series_plans)} series: {total_crops}")
    print(f"Overall size distribution: {_fmt_size_stats(all_sizes)}")

    try:
        check_collisions(planned_for_collisions)
        print("No output-filename collisions detected.")
    except CollisionError as exc:
        print(str(exc))
        return 1
    return 0


def run_real(series_plans, output_dir: Path, manifest_path: Path, overwrite: bool,
             use_superres: bool):
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Plan pass: resolve every (series, frame, box, crop_id, out_path)
    # up front and refuse loudly on any collision -- against another planned
    # entry, or against a file already on disk -- before writing anything.
    plan_rows = []  # (series, path, frame, box, crop_id, out_path)
    collision_check = []
    for plan in series_plans:
        entries = build_plan_entries(plan)
        for frame, box, crop_id in entries:
            out_name = build_output_filename(crop_id, plan.name, frame)
            out_path = output_dir / out_name
            plan_rows.append((plan.name, plan.path, frame, box, crop_id, out_path))
            collision_check.append((plan.name, frame, crop_id, out_path))

    check_collisions(collision_check)  # raises CollisionError; caller handles

    if not overwrite:
        existing = [p for _, _, _, _, _, p in plan_rows if p.exists()]
        if existing:
            lines = ["Refusing to run: output file(s) already exist (pass --overwrite to replace):"]
            lines.extend(f"  {p}" for p in existing[:20])
            if len(existing) > 20:
                lines.append(f"  ... and {len(existing) - 20} more")
            raise CollisionError("\n".join(lines))

    model_superres = None
    if use_superres:
        import torch
        from models.superres.superresolution_predict import RealesrganSuperresolution
        if torch.cuda.is_available():
            model_superres = RealesrganSuperresolution()
        else:
            print("[WARN] --superres requested but CUDA is not available; skipping super-resolution.")

    # --- Write pass. Frames are read+preprocessed once and reused for every
    # box drawn from them.
    manifest_rows = []
    frame_cache_key = None
    frame_gray = None
    for series, series_path, frame, box, crop_id, out_path in plan_rows:
        cache_key = (series_path, frame)
        if cache_key != frame_cache_key:
            frame_gray = read_preprocessed_gray(str(series_path / frame))
            frame_cache_key = cache_key

        crop, x1, y1, x2, y2 = crop_from_box(frame_gray, box)

        if model_superres is not None:
            orig_h, orig_w = crop.shape[:2]
            crop_x4 = model_superres.enhance(crop)
            crop = cv2.resize(crop_x4, (orig_w, orig_h), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(out_path), crop)
        manifest_rows.append({
            "output_path": str(out_path),
            "series": series,
            "source_frame": frame,
            "crop_id": crop_id,
            "cx": box["cx"], "cy": box["cy"], "half_w": box["half_w"], "half_h": box["half_h"],
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "crop_width": x2 - x1, "crop_height": y2 - y1,
        })

    # This run only rewrites rows for the series in `series_plans` -- if the
    # manifest file already exists (e.g. it's shared with other, previously-
    # cropped series, as cropped_training_set/manifest.csv is), preserve every
    # other series' rows rather than clobbering them. Rows belonging to a
    # series THIS run touches are dropped and replaced by manifest_rows above
    # (this is also what makes an --overwrite rerun of an existing series
    # correctly replace just that series' rows).
    this_run_series = {plan.name for plan in series_plans}
    preserved_rows = []
    if manifest_path.exists():
        with open(manifest_path, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            preserved_rows = [row for row in reader if row.get("series") not in this_run_series]

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(preserved_rows + manifest_rows)

    print(f"Wrote {len(manifest_rows)} crops to {output_dir}")
    if preserved_rows:
        print(f"Preserved {len(preserved_rows)} existing manifest rows from other series.")
    print(f"Manifest: {manifest_path}")
    return 0


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default=None,
                         help="Directory of plate series subfolders (default: <repo>/example_data)")
    parser.add_argument("--output-dir", default=None,
                         help="Output directory for new crops (default: <repo>/training_dataset/dlhook_recrop). "
                              "Never training_dataset/dlhook itself -- refused outright.")
    parser.add_argument("--manifest-path", default=None,
                         help="Manifest CSV path (default: <output-dir>/manifest.csv)")
    parser.add_argument("--series", default=None,
                         help="Comma-separated series (subfolder) names to restrict to (default: all)")
    parser.add_argument("--boxes-file", default=None,
                         help="Required. JSON {series: [{cx,cy,half_w,half_h}, ...]} giving crop boxes per "
                              "series -- the only way to supply crop geometry now that automatic detection "
                              "has been removed. Produce one with the manual crop-drawing tool, "
                              "`python -m multi.draw_crops`.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Report counts/sizes per series; write nothing")
    parser.add_argument("--overwrite", action="store_true",
                         help="Allow replacing existing output files (default: refuse loudly)")
    parser.add_argument("--superres", action="store_true",
                         help="Run Real-ESRGAN super-resolution per crop when CUDA is available (default: off -- slow, not needed for training data)")
    parser.add_argument("--max-frames-per-series", type=int, default=None,
                         help="Cap frames processed per series (debugging/dry-run speed), applied after --every-n sampling")
    parser.add_argument("--every-n", type=int, default=5,
                         help="Crop only every Nth frame of each series (time-ordered, not lexicographic), plus "
                              "always the last frame regardless of stride (default: 5). --every-n 1 means every frame.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    input_dir = Path(args.input_dir) if args.input_dir else _REPO_ROOT / "example_data"
    output_dir = Path(args.output_dir) if args.output_dir else _REPO_ROOT / "training_dataset" / "dlhook_recrop"
    manifest_path = Path(args.manifest_path) if args.manifest_path else output_dir / "manifest.csv"

    if output_dir.resolve() == PROTECTED_DIR:
        print(f"Refusing to write to {PROTECTED_DIR} -- that is the annotated training set "
              "RootPainter's masks are keyed to by exact filename. Use a different --output-dir.")
        return 1

    if not args.boxes_file:
        print(
            "No --boxes-file given. Automatic seedling/row detection has been removed -- "
            "crop geometry must now come from a boxes JSON file. Draw crop boxes with the "
            "manual crop-drawing tool (`python -m multi.draw_crops`) and pass its output "
            "here with --boxes-file."
        )
        return 1

    if args.every_n < 1:
        print(f"--every-n must be >= 1, got {args.every_n}")
        return 1

    boxes_map = load_boxes_file(args.boxes_file)
    only = set(args.series.split(",")) if args.series else None

    try:
        series_dirs = discover_series(input_dir, only)
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    if not series_dirs:
        print(f"No series with image files found under {input_dir}" + (f" matching {only}" if only else ""))
        return 1

    missing = sorted(name for name in series_dirs if name not in boxes_map)
    if missing:
        print(
            "No crop boxes in --boxes-file for series: " + ", ".join(missing) + ". "
            "Draw crop boxes for these series with the manual crop-drawing tool "
            "(`python -m multi.draw_crops`) and add them to the boxes file."
        )
        return 1

    series_plans = []
    for name, path in series_dirs.items():
        raw_frames = list_image_files(str(path))
        ordered_frames = order_series_for_sampling(str(path), raw_frames)
        sampled_frames = sample_every_n(ordered_frames, args.every_n)
        plan = plan_series(name, path, sampled_frames, boxes_map, args.max_frames_per_series)
        series_plans.append(plan)

    if args.dry_run:
        return run_dry_run(series_plans, output_dir)

    try:
        return run_real(series_plans, output_dir, manifest_path, args.overwrite, args.superres)
    except CollisionError as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
