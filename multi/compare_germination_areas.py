#!/usr/bin/env python
"""Read-only diagnostic: does switching the GUI's germination (radicle) mask
from the shipped binary model to the multiclass student shift the detected
germination frame?

Detected germination sets kinematic time-zero (GerminationDetector.detect,
utils/germination_detector.py) -- a silent shift there corrupts every
downstream angle trace. This script is the gate run *before* the default
backend is flipped (models/segmentation_backends.py's DEFAULT_BACKEND), and
it changes nothing: it only measures.

THREE ARMS, NOT TWO
-------------------
- germ_v1  -- weights/RootPainter_weights/germ_v1.pkl, what the GUI deploys
              today (BinaryBackend's label "4").
- germ_v2  -- weights/RootPainter_weights/germ_v2.pkl, the retrained teacher
              the multiclass student was actually distilled from.
- multiclass -- class index 3 of weights/multiclass/dlhook_4class_v1.pt.

Two arms would be uninterpretable: germ_v1 vs multiclass confounds "the
multiclass architecture" with "the germ_v1 -> germ_v2 teacher swap that
happened along the way". The middle arm (germ_v2) separates those two
causes. If germ_v2 and multiclass agree with each other but both differ
from germ_v1, the cause is the teacher swap, not the architecture -- any
conclusion drawn from this script's output must respect that; it does not
draw that conclusion for you.

THE SEED-POINT PROBLEM
-----------------------
GerminationDetector.detect() needs a `seed_point` -- in the GUI this is the
operator's click (crop_points_distributed[crop_id][0],
seedling_measurment.py:867). This script has no clicks, so it uses a
documented PROXY, identical across all three arms and controlled by
--seed-point:
  - "bottom-center" (default): (half_w, 2*half_h - 1) of the seedling's own
    fixed crop box -- i.e. bottom-center of the crop, where a germinating
    radicle is expected to sit relative to the seed coat.
  - "center": (half_w, half_h) -- the crop's geometric center.
A shared proxy seed point can shift every arm's ABSOLUTE detected frame
together (e.g. if the real click sits further from the proxy than from the
seed coat). It CANNOT manufacture a DIFFERENCE between arms run against the
same proxy -- that difference is the finding this script exists to surface,
the absolute frame number is not. This is also why the script additionally
reports a seed-point-INDEPENDENT measure: total germ contour area anywhere
in the crop, per frame, with no proximity filtering at all
(GerminationDetector.diagnostics[...]'s max_area_anywhere/frames_with_any_germ,
via describe()/.diagnostics -- not recomputed here).

REPRODUCING THE LIVE CONTOUR RECIPE
------------------------------------
`germ_contours_from_mask` below reproduces process_single_frame's germ path
exactly (seedling_measurment.py):
  - masks are 255 = foreground in memory already (no bitwise_not needed to
    get there) -- lines 1026-1033.
  - cv2.threshold(img_germ, 127, 255, cv2.THRESH_BINARY) -- line 1033.
  - NO zoom_out_mask on germ: that helper is applied to cotyledon/hypocotyl
    only and would displace the germ blob relative to the seed point --
    lines 1055-1059.
  - NO bitwise_not on germ: re-inverting made every germ contour describe
    the crop's background -- lines 1026-1032.
  - cv2.findContours(germ_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
    with NO len(c) >= 5 vertex floor -- that floor is only applied to the
    cotyledon/hypocotyl contours (a cv2.fitEllipse prerequisite); germ only
    ever takes contourArea, so a small emerging radicle blob with fewer than
    5 contour points is kept -- line 1067.
  - crop_size = (2*half_w + 2*half_h) / 2 -- line 873.
The real utils.germination_detector.GerminationDetector.detect is imported
and called unmodified; this script never reimplements its logic.

READ-ONLY
---------
This script does not modify utils/germination_detector.py, any detector
constant, or any file under Root_Painter_Sync/. It writes only: its own
--csv output (default multi/results/germination_comparison.csv, gitignored)
and, transiently, cropped/preprocessed frame PNGs under a temp directory
(cleaned up on exit unless --tmp-dir is given explicitly).

Usage:
    python multi/compare_germination_areas.py
    python multi/compare_germination_areas.py --series F1_Plate_2_YS --every-n 2 --limit-seedlings 3
    python multi/compare_germination_areas.py --seed-point center --csv multi/results/center_run.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# multi/ on sys.path (for `from src...`, matching recrop_plates.py's own
# convention) and the repo root on sys.path (for `models.*` / `utils.*`),
# without importing seedling_measurment.py itself (Tk + heavy model imports
# at class-init time).
_MULTI_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MULTI_DIR.parent
sys.path.insert(0, str(_MULTI_DIR))
sys.path.insert(0, str(_REPO_ROOT))

import recrop_plates  # noqa: E402
from src.recrop_geometry import build_output_filename, crop_from_box, list_image_files  # noqa: E402

from models.UNetInference import MARGIN, get_predictor  # noqa: E402
from models.multiclass_inference import MulticlassInference  # noqa: E402
from utils.germination_detector import GerminationDetector  # noqa: E402

ARMS = ("germ_v1", "germ_v2", "multiclass")
SEED_POINT_MODES = ("bottom-center", "center")

DEFAULT_BOXES_FILE = "crop_boxes.json"
DEFAULT_SERIES = "F1_Plate_2_YS"
DEFAULT_GERM_V1_WEIGHTS = "weights/RootPainter_weights/germ_v1.pkl"
DEFAULT_GERM_V2_WEIGHTS = "weights/RootPainter_weights/germ_v2.pkl"
DEFAULT_MULTICLASS_CHECKPOINT = "weights/multiclass/dlhook_4class_v1.pt"
DEFAULT_CSV_PATH = str(_MULTI_DIR / "results" / "germination_comparison.csv")

# The multiclass checkpoint's own training geometry (see
# models/segmentation_backends.py's MulticlassBackend docstring): the real
# crops are narrow strips, and 572 (UNetInference's live-GUI default) would
# reflect-pad ~94% of a tile with synthetic context. MARGIN is imported from
# models.UNetInference rather than hardcoded, per that module's own contract.
MULTICLASS_IN_SIZE = 264
MULTICLASS_OUT_SIZE = 252
MULTICLASS_NUM_CLASSES = 4

GERM_LABEL = "4"          # BinaryBackend's germination/radicle label.
GERM_CLASS_INDEX = 3      # Multiclass label map's radicle class.

CSV_FIELDNAMES = [
    "series", "seedling_id", "frame_index", "frame_name", "arm",
    "area_near_seed", "total_area", "detected_frame",
]


# --- contour recipe (see module docstring) ----------------------------------

def germ_contours_from_mask(mask):
    """`mask` is 255 = foreground (or None if that label had no mask for this
    frame). Reproduces process_single_frame's germ contour extraction
    exactly -- see the module docstring for the seedling_measurment.py line
    numbers this mirrors. Deliberately no len(c) >= 5 vertex floor."""
    if mask is None:
        return []
    _, bin_germ = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return list(cv2.findContours(bin_germ, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])


def class_index_mask(label_map, class_index):
    """Binary mask (255 = foreground) for exactly one class of a multiclass
    label map. Mirrors MulticlassBackend.predict_into's per-class split
    (models/segmentation_backends.py) applied to the radicle class only."""
    return (label_map == class_index).astype(np.uint8) * 255


def seed_point_for_box(box, mode):
    """Documented proxy seed point in the crop's own (box-local) pixel
    space -- see the module docstring's "THE SEED-POINT PROBLEM". Assumes
    the crop is not clamped by the source image's edges, i.e. its actual
    saved size is exactly (2*half_w, 2*half_h) -- true for every box in
    crop_boxes.json's example series, which sit well inside their plates."""
    width, height = 2 * box["half_w"], 2 * box["half_h"]
    if mode == "bottom-center":
        return (width / 2.0, height - 1.0)
    if mode == "center":
        return (width / 2.0, height / 2.0)
    raise ValueError(f"unknown --seed-point mode {mode!r} (expected one of {SEED_POINT_MODES})")


# --- the three arms ----------------------------------------------------------

def run_binary_arm(weight_path, crop_paths):
    """germ_v1 / germ_v2 arm: the exact call the live GUI's BinaryBackend
    makes for label "4" (models/segmentation_backends.py), just against
    whichever checkpoint the caller names."""
    predictor = get_predictor(weight_path)
    return predictor.predict_files(crop_paths, label=GERM_LABEL)


def run_multiclass_arm(checkpoint_path, crop_paths):
    """multiclass arm: one 4-class checkpoint, run at its own training
    geometry (264/252/MARGIN, not UNetInference's 572/560/6 default), with
    only class index 3 (radicle) extracted -- see class_index_mask."""
    predictor = MulticlassInference(
        checkpoint_path,
        num_classes=MULTICLASS_NUM_CLASSES,
        in_size=MULTICLASS_IN_SIZE,
        out_size=MULTICLASS_OUT_SIZE,
        margin=MARGIN,
    )
    label_maps = predictor.predict_files_labelmaps(crop_paths)
    return {
        file_name: class_index_mask(label_map, GERM_CLASS_INDEX)
        for file_name, label_map in label_maps.items()
    }


def run_arm(arm, crop_paths, args):
    if arm == "germ_v1":
        return run_binary_arm(args.germ_v1_weights, crop_paths)
    if arm == "germ_v2":
        return run_binary_arm(args.germ_v2_weights, crop_paths)
    if arm == "multiclass":
        return run_multiclass_arm(args.multiclass_checkpoint, crop_paths)
    raise ValueError(f"unknown arm {arm!r}")


# --- crop preparation ---------------------------------------------------------

def build_crop_files(series_dir, series_name, frames, boxes, tmp_dir):
    """Write one preprocessed+cropped PNG per (frame, seedling box), matching
    seedling_measurment.py's start_analysis crop loop (:667-696): grayscale +
    median blur + contrast + normalize preprocessing
    (recrop_plates.read_preprocessed_gray reproduces that exact preprocessing),
    then crop_from_box per seedling box. No super-resolution: the GUI only
    applies it when CUDA is available and it does not change which pixels
    the germ label segments, so this comparison omits it for reproducibility
    across machines.

    Returns {crop_id: [path_for_frames[0], path_for_frames[1], ...]}, one
    list per seedling in frame (time) order.
    """
    paths_by_crop_id = {crop_id: [] for crop_id in range(len(boxes))}
    for frame_name in frames:
        frame_gray = recrop_plates.read_preprocessed_gray(str(series_dir / frame_name))
        for crop_id, box in enumerate(boxes):
            crop, _x1, _y1, _x2, _y2 = crop_from_box(frame_gray, box)
            out_name = build_output_filename(crop_id, series_name, frame_name)
            out_path = tmp_dir / out_name
            cv2.imwrite(str(out_path), crop)
            paths_by_crop_id[crop_id].append(str(out_path))
    return paths_by_crop_id


# --- comparison ---------------------------------------------------------------

def classify_pair(frame_a, frame_b):
    """One of "both_none", "one_none", "agree_exact", "agree_within_1"
    (differs by exactly 1, not exact), "differ" (differs by more than 1)."""
    if frame_a is None and frame_b is None:
        return "both_none"
    if frame_a is None or frame_b is None:
        return "one_none"
    diff = abs(frame_a - frame_b)
    if diff == 0:
        return "agree_exact"
    if diff == 1:
        return "agree_within_1"
    return "differ"


def compare_arms(detected_frames_by_arm, arms=ARMS):
    """detected_frames_by_arm: {arm: {seed_id: frame_or_None}}.

    Returns {(arm_a, arm_b): {"agree_exact": n, "agree_within_1": n,
    "differ": n, "both_none": n, "one_none_seedlings": [seed_id, ...]}} for
    every unordered pair of `arms`, over the union of seed_ids either arm
    reports for that pair."""
    summary = {}
    for arm_a, arm_b in itertools.combinations(arms, 2):
        frames_a = detected_frames_by_arm.get(arm_a, {})
        frames_b = detected_frames_by_arm.get(arm_b, {})
        seed_ids = sorted(set(frames_a) | set(frames_b))
        counts = {"agree_exact": 0, "agree_within_1": 0, "differ": 0, "both_none": 0}
        one_none_seedlings = []
        for seed_id in seed_ids:
            category = classify_pair(frames_a.get(seed_id), frames_b.get(seed_id))
            if category == "one_none":
                one_none_seedlings.append(seed_id)
            else:
                counts[category] += 1
        summary[(arm_a, arm_b)] = {**counts, "one_none_seedlings": one_none_seedlings}
    return summary


def overall_agreement_within_1(detected_frames_by_arm, arms=ARMS):
    """The plan's gate: do all three arms' detected frames agree within ±1
    on every seedling that all three of them actually detected?

    Returns (all_agree, checked_seedlings, disagreeing_seedlings,
    incomplete_seedlings) -- `incomplete` lists seedlings where at least one
    arm returned None (excluded from the ±1 check itself, reported
    separately, same as compare_arms's one_none_seedlings)."""
    seed_ids = sorted(set().union(*(detected_frames_by_arm.get(arm, {}).keys() for arm in arms)))
    checked, disagreeing, incomplete = [], [], []
    for seed_id in seed_ids:
        values = [detected_frames_by_arm.get(arm, {}).get(seed_id) for arm in arms]
        if any(v is None for v in values):
            incomplete.append(seed_id)
            continue
        checked.append(seed_id)
        if max(values) - min(values) > 1:
            disagreeing.append(seed_id)
    return len(disagreeing) == 0, checked, disagreeing, incomplete


# --- output --------------------------------------------------------------------

def write_csv(csv_path, rows):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} per-frame rows to {csv_path}")


def print_table(boxes, detected_frames_by_arm, diagnostics_by_arm):
    print()
    header = f"{'seedling':>8}" + "".join(f"{arm:>14}" for arm in ARMS)
    print(header)
    for crop_id in range(len(boxes)):
        seed_id = crop_id + 1
        row = f"{seed_id:>8}"
        for arm in ARMS:
            frame = detected_frames_by_arm[arm].get(seed_id)
            row += f"{str(frame):>14}"
        print(row)

    print()
    print("Seed-point-independent diagnostics (max_area_anywhere px / frames_with_any_germ),"
          " from GerminationDetector.diagnostics -- not filtered by the seed-point proxy above:")
    header2 = f"{'seedling':>8}" + "".join(f"{arm:>22}" for arm in ARMS)
    print(header2)
    for crop_id in range(len(boxes)):
        seed_id = crop_id + 1
        row = f"{seed_id:>8}"
        for arm in ARMS:
            diag = diagnostics_by_arm[arm].get(seed_id, {})
            cell = f"{diag.get('max_area_anywhere', 0.0):.0f}/{diag.get('frames_with_any_germ', 0)}"
            row += f"{cell:>22}"
        print(row)


def print_summary(pair_summary):
    print()
    print("Pairwise agreement on detected frame:")
    for (arm_a, arm_b), stats in pair_summary.items():
        print(f"  {arm_a} vs {arm_b}: "
              f"agree_exact={stats['agree_exact']} agree_within_1={stats['agree_within_1']} "
              f"differ={stats['differ']} both_none={stats['both_none']}")
        if stats["one_none_seedlings"]:
            print(f"    ONE ARM DETECTED, THE OTHER DID NOT for seedling(s): "
                  f"{stats['one_none_seedlings']}")

    print()
    print("Reminder: if germ_v2 and multiclass mostly agree with each other while germ_v1 "
          "differs from both, that pattern points to the germ_v1 -> germ_v2 teacher swap, "
          "not the multiclass architecture. If germ_v1 and germ_v2 agree while multiclass "
          "differs from both, that points to the architecture instead. This script reports "
          "the pairwise numbers above; it does not draw that conclusion for you.")


def print_verdict(agreement_result, series):
    all_agree, checked, disagreeing, incomplete = agreement_result
    print()
    # An empty `checked` set makes "all agree" VACUOUSLY true -- every arm
    # returned None for every seedling, so there was nothing to compare and
    # the gate would wave the backend flip through on zero evidence. That is
    # the opposite of what this gate is for, so it is reported as
    # INCONCLUSIVE, never as a pass.
    if not checked:
        print(f"VERDICT for series={series!r}: INCONCLUSIVE -- no seedling had a detected "
              f"frame under all three arms, so there was nothing to compare. This is NOT "
              f"an agreement result and must not be read as one.")
        if incomplete:
            print(f"  Every seedling was excluded for that reason: {incomplete}")
        return
    print(f"VERDICT for series={series!r}: detected frames "
          f"{'AGREE' if all_agree else 'DO NOT AGREE'} within +/-1 across all three arms "
          f"(checked {len(checked)} seedling(s) where every arm detected a frame).")
    if len(checked) < len(checked) + len(incomplete):
        print(f"  CAUTION: only {len(checked)} of {len(checked) + len(incomplete)} seedlings "
              f"could be checked. A pass over a small subset is weak evidence -- read it "
              f"alongside the one-arm-None call-outs above, which are themselves "
              f"disagreements between arms.")
    if disagreeing:
        print(f"  Seedling(s) disagreeing by more than 1 frame across arms: {disagreeing}")
    if incomplete:
        print(f"  Seedling(s) excluded from the check because at least one arm returned "
              f"None (see the one-arm-None call-outs above): {incomplete}")


# --- CLI -----------------------------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--boxes-file", default=DEFAULT_BOXES_FILE,
                         help=f"JSON {{series: [{{cx,cy,half_w,half_h}}, ...]}} (default: {DEFAULT_BOXES_FILE})")
    parser.add_argument("--series", default=DEFAULT_SERIES,
                         help=f"Series (example_data/ subfolder) to compare (default: {DEFAULT_SERIES})")
    parser.add_argument("--input-dir", default=None,
                         help="Directory of plate series subfolders (default: <repo>/example_data)")
    parser.add_argument("--every-n", type=int, default=1,
                         help="Use only every Nth frame, time-ordered, plus the last frame (default: 1, every frame)")
    parser.add_argument("--limit-seedlings", type=int, default=None,
                         help="Only compare the first N seedlings' boxes (default: all)")
    parser.add_argument("--seed-point", choices=SEED_POINT_MODES, default="bottom-center",
                         help="Proxy seed-point placement within each crop box (default: bottom-center)")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH,
                         help=f"Per-frame-per-arm areas CSV output path (default: {DEFAULT_CSV_PATH})")
    parser.add_argument("--germ-v1-weights", default=DEFAULT_GERM_V1_WEIGHTS)
    parser.add_argument("--germ-v2-weights", default=DEFAULT_GERM_V2_WEIGHTS)
    parser.add_argument("--multiclass-checkpoint", default=DEFAULT_MULTICLASS_CHECKPOINT)
    parser.add_argument("--tmp-dir", default=None,
                         help="Directory to write intermediate crop PNGs into (default: an auto-cleaned temp dir)")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    boxes_map = recrop_plates.load_boxes_file(args.boxes_file)
    if args.series not in boxes_map:
        print(f"No crop boxes for series {args.series!r} in {args.boxes_file} "
              f"(available: {sorted(boxes_map)})")
        return 1
    boxes = boxes_map[args.series]
    if args.limit_seedlings is not None:
        boxes = boxes[:args.limit_seedlings]
    if not boxes:
        print("No seedling boxes to compare (empty boxes list / --limit-seedlings 0).")
        return 1

    input_dir = Path(args.input_dir) if args.input_dir else _REPO_ROOT / "example_data"
    series_dir = input_dir / args.series
    if not series_dir.is_dir():
        print(f"series directory not found: {series_dir}")
        return 1

    raw_frames = list_image_files(str(series_dir))
    ordered_frames = recrop_plates.order_series_for_sampling(str(series_dir), raw_frames)
    try:
        frames = recrop_plates.sample_every_n(ordered_frames, args.every_n)
    except ValueError as exc:
        print(str(exc))
        return 1
    if not frames:
        print(f"No image frames found under {series_dir}")
        return 1

    print(f"series={args.series!r} seedlings={len(boxes)} frames={len(frames)} of "
          f"{len(ordered_frames)} (every_n={args.every_n}) seed_point_mode={args.seed_point!r}")
    print("Seed point is a documented PROXY (no operator click available here): "
          f"'{args.seed_point}' of each seedling's own fixed crop box, IDENTICAL across all "
          "three arms below. A shared proxy can shift every arm's absolute detected frame "
          "together, but cannot manufacture a difference BETWEEN arms -- see the module "
          "docstring.")

    owns_tmp_dir = args.tmp_dir is None
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else Path(tempfile.mkdtemp(prefix="dlhook_germ_compare_"))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Writing {len(frames) * len(boxes)} preprocessed crop(s) to {tmp_dir} ...")
        paths_by_crop_id = build_crop_files(series_dir, args.series, frames, boxes, tmp_dir)

        detected_frames_by_arm = {}
        diagnostics_by_arm = {}
        csv_rows = []

        for arm in ARMS:
            print(f"Running arm: {arm} ...")
            all_paths = [p for paths in paths_by_crop_id.values() for p in paths]
            masks_by_filename = run_arm(arm, all_paths, args)

            detected_frames_by_arm[arm] = {}
            diagnostics_by_arm[arm] = {}

            for crop_id, box in enumerate(boxes):
                seed_id = crop_id + 1
                crop_paths = paths_by_crop_id[crop_id]
                contours_by_frame = [
                    germ_contours_from_mask(masks_by_filename.get(Path(p).name))
                    for p in crop_paths
                ]
                seed_point = seed_point_for_box(box, args.seed_point)
                crop_size = (2 * box["half_w"] + 2 * box["half_h"]) / 2

                detector = GerminationDetector()
                frame_idx = detector.detect(seed_id, contours_by_frame, seed_point, crop_size)
                diag = detector.diagnostics[seed_id]

                detected_frames_by_arm[arm][seed_id] = frame_idx
                diagnostics_by_arm[arm][seed_id] = diag

                for frame_index, (frame_name, contours) in enumerate(zip(frames, contours_by_frame)):
                    total_area = sum(cv2.contourArea(c) for c in contours)
                    csv_rows.append({
                        "series": args.series,
                        "seedling_id": seed_id,
                        "frame_index": frame_index,
                        "frame_name": frame_name,
                        "arm": arm,
                        "area_near_seed": diag["areas"][frame_index],
                        "total_area": total_area,
                        "detected_frame": frame_idx,
                    })
    finally:
        if owns_tmp_dir:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    write_csv(args.csv, csv_rows)
    print_table(boxes, detected_frames_by_arm, diagnostics_by_arm)
    print_summary(compare_arms(detected_frames_by_arm))
    print_verdict(overall_agreement_within_1(detected_frames_by_arm), args.series)
    return 0


if __name__ == "__main__":
    sys.exit(main())
