"""Merge the three shipped binary segmentation models (cotyledon, hypocotyl,
radicle/germination) into one single-channel multiclass pseudo-label PNG per
raw image, with human RootPainter strokes overriding the automatic result.

Class indices: 0 background, 1 cotyledon, 2 hypocotyl, 3 radicle, 255 ignore.
"cotyledon_v3.pkl" and RootPainter label "3" are not used anywhere in this
module -- see CLAUDE.md ("Label '3' exists on disk but is deliberately never
run").
"""
from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import ensure_repo_root_importable

ensure_repo_root_importable()

# NOTE: `from models.UNetInference import get_predictor` is deliberately NOT
# imported at module level. It is imported inside the two functions that
# actually run inference (`merge_dataset`, `validate_pseudo_labels`) so that
# the disk-fed path (`merge_dataset_from_probs`) can be used without pulling
# in torch at all.

CLASS_NAMES = ["background", "cotyledon", "hypocotyl", "radicle"]
FOREGROUND_CLASSES = ["cotyledon", "hypocotyl", "radicle"]
CLASS_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IGNORE_VALUE = 255

DEFAULT_THRESHOLDS = {
    "cotyledon": (0.05, 0.5),
    "hypocotyl": (0.05, 0.5),
    "radicle": (0.02, 0.3),
}
DEFAULT_CONFLICT_MARGIN = 0.15
DEFAULT_CHUNK_SIZE = 16

# Preview palette (BGR, for cv2.imwrite), keyed by class index / IGNORE_VALUE.
PREVIEW_COLORS_BGR = {
    0: (0, 0, 0),
    1: (60, 20, 220),      # cotyledon - red
    2: (255, 144, 30),     # hypocotyl - blue-ish
    3: (50, 205, 50),      # radicle - green
    IGNORE_VALUE: (0, 255, 255),  # ignore - yellow
}


def decode_rootpainter_annotation(rgba: np.ndarray):
    """Decode a RootPainter RGBA annotation image.

    ch0 (R) == 255 -> foreground stroke; ch1 (G) == 255 -> explicit
    background stroke; ch3 (A) == 0 -> undefined/unlabelled (blue is always
    0 and carries no information). Some files on disk have no alpha channel
    at all (RGB only) -- for those, "defined" falls back to `foreground |
    background` since there is no alpha to read.

    Returns (foreground, background, defined), each an HxW bool array.
    """
    if rgba.ndim != 3 or rgba.shape[2] < 3:
        raise ValueError(f"Expected an HxWx3 or HxWx4 array, got shape {rgba.shape}")

    foreground = rgba[..., 0] == 255
    background = rgba[..., 1] == 255
    if rgba.shape[2] >= 4:
        defined = rgba[..., 3] > 0
    else:
        defined = foreground | background

    return foreground, background, defined


def discover_annotation_files(class_dir: Path) -> dict:
    """Map filename -> Path for every annotation under `class_dir/{train,val}`.

    Train takes precedence: a filename appearing in both splits (the known
    dlhook_hypocotyl defect, `0-crop-IMG_1104.png`) is kept from train and
    dropped from val, with a warning -- not hardcoded to that one filename,
    so the same rule applies to any future duplicate.
    """
    class_dir = Path(class_dir)
    index: dict[str, Path] = {}
    for split in ("train", "val"):
        split_dir = class_dir / split
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.iterdir()):
            if not path.is_file():
                continue
            if path.name in index:
                warnings.warn(
                    f"{path.name} appears in both train/ and val/ under {class_dir}; "
                    f"keeping the train/ copy, dropping {split}/."
                )
                continue
            index[path.name] = path
    return index


def _merge_probs_to_labels(prob_maps: dict, thresholds: dict, conflict_margin: float):
    """Combine per-class foreground-probability maps into one label array
    following the vote/conflict/ignore rules described in the module
    docstring. Returns (label uint8 HxW, order list[str] of classes used)."""
    order = [c for c in FOREGROUND_CLASSES if c in prob_maps]
    if not order:
        raise ValueError("prob_maps must contain at least one foreground class")

    probs = np.stack([prob_maps[c] for c in order], axis=0).astype(np.float32)
    t_lo = np.array([thresholds[c][0] for c in order], dtype=np.float32).reshape(-1, 1, 1)
    t_hi = np.array([thresholds[c][1] for c in order], dtype=np.float32).reshape(-1, 1, 1)

    fg_hi = probs > t_hi
    bg_lo = probs < t_lo
    ignore_band = ~fg_hi & ~bg_lo

    num_fg = fg_hi.sum(axis=0)
    label = np.zeros(probs.shape[1:], dtype=np.uint8)  # default: background

    # 0 classes clear t_hi: ignore if any class is in the ambiguous middle
    # band, otherwise every class agreed on background (already the default).
    zero_fg_ignore = (num_fg == 0) & ignore_band.any(axis=0)
    label[zero_fg_ignore] = IGNORE_VALUE

    # Exactly 1 class clears t_hi: it wins outright, no conflict possible.
    one_fg = num_fg == 1
    if np.any(one_fg):
        winner_idx = np.argmax(fg_hi, axis=0)
        for i, cname in enumerate(order):
            sel = one_fg & (winner_idx == i)
            if np.any(sel):
                label[sel] = CLASS_INDEX[cname]

    # 2+ classes clear t_hi: highest probability wins, unless the top two
    # are within `conflict_margin` of each other, in which case -> ignore.
    multi_fg = num_fg >= 2
    if np.any(multi_fg):
        masked_probs = np.where(fg_hi, probs, -np.inf)
        sorted_probs = np.sort(masked_probs, axis=0)
        top1 = sorted_probs[-1]
        top2 = sorted_probs[-2]
        with np.errstate(invalid="ignore"):
            # top1/top2 are both -inf wherever multi_fg is False (fewer than
            # 2 classes cleared t_hi); the resulting nan there is never used.
            margin = top1 - top2
        winner_idx = np.argmax(masked_probs, axis=0)

        conflict = multi_fg & (margin <= conflict_margin)
        clear_winner = multi_fg & ~conflict
        label[conflict] = IGNORE_VALUE
        for i, cname in enumerate(order):
            sel = clear_winner & (winner_idx == i)
            if np.any(sel):
                label[sel] = CLASS_INDEX[cname]

    return label, order


def apply_human_overrides(label: np.ndarray, class_name: str, foreground: np.ndarray,
                           background: np.ndarray) -> bool:
    """Burn one class's human strokes into `label` in place. Foreground
    strokes win over background strokes on the same class (should not
    co-occur, but stroke drawing tools can produce stray overlaps).
    Undefined pixels (foreground == background == False) are left
    untouched, i.e. the pseudo-label survives there. Returns True if any
    pixel was touched."""
    bg_only = background & ~foreground
    touched = bool(foreground.any() or bg_only.any())
    label[bg_only] = CLASS_INDEX["background"]
    label[foreground] = CLASS_INDEX[class_name]
    return touched


@dataclass
class MergeResult:
    processed: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    out_dir: Path = None
    report_path: Path = None


REPORT_FIELDNAMES = [
    "filename", *[f"n_{name}" for name in CLASS_NAMES], "n_ignore", "ignore_fraction",
    "human_strokes_applied",
]


def _report_row(filename: str, label: np.ndarray, human_applied: bool) -> dict:
    """Build one row of the merge report, shared by both the inline
    (`merge_dataset`) and disk-fed (`merge_dataset_from_probs`) paths so
    they emit identical CSV columns."""
    counts = {name: int(np.count_nonzero(label == idx)) for name, idx in CLASS_INDEX.items()}
    n_ignore = int(np.count_nonzero(label == IGNORE_VALUE))
    total = label.size
    return {
        "filename": filename,
        **{f"n_{name}": counts[name] for name in CLASS_NAMES},
        "n_ignore": n_ignore,
        "ignore_fraction": n_ignore / total,
        "human_strokes_applied": human_applied,
    }


def _write_report(rows: list, report_path: Path) -> None:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _apply_annotation_overrides(label: np.ndarray, filename: str, annotation_index: dict) -> bool:
    """Burn every class's human strokes (if any exist for `filename`) into
    `label` in place. Returns True if any pixel was touched."""
    human_applied = False
    for cname, index in annotation_index.items():
        ann_path = index.get(filename)
        if ann_path is None:
            continue
        ann = np.array(_load_rgba(ann_path))
        fg, bg, _defined = decode_rootpainter_annotation(ann)
        if apply_human_overrides(label, cname, fg, bg):
            human_applied = True
    return human_applied


def _build_annotation_index(annotations_dirs: dict | None) -> dict:
    annotation_index = {}
    if annotations_dirs:
        for cname, adir in annotations_dirs.items():
            annotation_index[cname] = discover_annotation_files(adir)
    return annotation_index


def merge_dataset(
    raw_dir,
    out_dir,
    report_path,
    weights: dict,
    annotations_dirs: dict | None = None,
    thresholds: dict | None = None,
    conflict_margin: float = DEFAULT_CONFLICT_MARGIN,
    preview_dir=None,
    limit: int | None = None,
    force: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    image_glob: str = "*.png",
) -> MergeResult:
    """Produce one multiclass pseudo-label PNG per raw image.

    `weights` maps a subset of FOREGROUND_CLASSES to a model weight path
    (e.g. {"cotyledon": ".../cotyledon_v5.pkl", ...}). `annotations_dirs`
    maps the same class names to a directory containing `train/` and `val/`
    RootPainter annotation subfolders whose strokes override the automatic
    result (see `apply_human_overrides`).
    """
    from models.UNetInference import get_predictor  # local: keep torch out of the disk-fed path

    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    report_path = Path(report_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir = Path(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)

    merged_thresholds = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        merged_thresholds.update(thresholds)

    class_order = [c for c in FOREGROUND_CLASSES if c in weights]
    predictors = {c: get_predictor(str(weights[c])) for c in class_order}

    annotation_index = _build_annotation_index(annotations_dirs)

    all_paths = sorted(raw_dir.glob(image_glob))
    if limit is not None:
        all_paths = all_paths[:limit]

    todo, skipped = [], []
    for path in all_paths:
        out_path = out_dir / path.name
        if out_path.exists() and not force:
            skipped.append(path.name)
        else:
            todo.append(path)

    rows = []
    processed = []
    for start in range(0, len(todo), chunk_size):
        chunk_paths = todo[start:start + chunk_size]
        images, valid_paths = [], []
        for path in chunk_paths:
            image = cv2.imread(str(path))
            if image is None:
                warnings.warn(f"Could not read {path}; skipping")
                continue
            images.append(image)
            valid_paths.append(path)
        if not images:
            continue

        prob_maps_by_class = {
            cname: predictors[cname]._segment_many(images) for cname in class_order
        }

        for i, path in enumerate(valid_paths):
            prob_maps = {cname: prob_maps_by_class[cname][i] for cname in class_order}
            label, order = _merge_probs_to_labels(prob_maps, merged_thresholds, conflict_margin)

            human_applied = _apply_annotation_overrides(label, path.name, annotation_index)

            out_path = out_dir / path.name
            cv2.imwrite(str(out_path), label)
            if preview_dir is not None:
                cv2.imwrite(str(preview_dir / path.name), _colorize(label))

            rows.append(_report_row(path.name, label, human_applied))
            processed.append(path.name)

    _write_report(rows, report_path)

    return MergeResult(processed=processed, skipped=skipped, out_dir=out_dir, report_path=report_path)


def merge_dataset_from_probs(
    raw_dir,
    prob_dir,
    out_dir,
    report_path,
    thresholds: dict | None = None,
    conflict_margin: float = DEFAULT_CONFLICT_MARGIN,
    preview_dir=None,
    limit: int | None = None,
    force: bool = False,
    image_glob: str = "*.png",
    annotations_dirs: dict | None = None,
) -> MergeResult:
    """Disk-fed twin of `merge_dataset`: produce one multiclass pseudo-label
    PNG per raw image, reading per-class probability maps from `prob_dir`
    instead of running inference.

    For each raw image `{stem}.png` under `raw_dir`, this expects a
    `{stem}-{class}.png` uint8 probability map under `prob_dir` for every
    class in FOREGROUND_CLASSES (cotyledon, hypocotyl, radicle), where
    **255 = high probability** -- the OPPOSITE of the legacy on-disk mask
    convention where 0 = foreground (see MaskStore.dump() / CLAUDE.md). Do
    not invert. A crop missing any one of its three probability maps is
    skipped with a warning rather than treated as all-background for the
    missing class -- a missing file and a confident-background file must
    not look the same.

    Reuses `_merge_probs_to_labels` verbatim (no reimplementation of the
    vote/conflict/ignore algorithm) on `prob_uint8.astype(np.float32) /
    255.0`, so given the same effective probabilities this path and
    `merge_dataset` produce byte-identical label PNGs (see
    tests/test_multi_mask_merge.py). This function imports no torch --
    `models.UNetInference` is never touched.
    """
    raw_dir = Path(raw_dir)
    prob_dir = Path(prob_dir)
    out_dir = Path(out_dir)
    report_path = Path(report_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir = Path(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)

    merged_thresholds = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        merged_thresholds.update(thresholds)

    annotation_index = _build_annotation_index(annotations_dirs)

    all_paths = sorted(raw_dir.glob(image_glob))
    if limit is not None:
        all_paths = all_paths[:limit]

    todo, skipped = [], []
    for path in all_paths:
        out_path = out_dir / path.name
        if out_path.exists() and not force:
            skipped.append(path.name)
        else:
            todo.append(path)

    rows = []
    processed = []
    for path in todo:
        stem = path.stem
        prob_maps = {}
        missing = []
        for cname in FOREGROUND_CLASSES:
            prob_path = prob_dir / f"{stem}-{cname}.png"
            prob_u8 = cv2.imread(str(prob_path), cv2.IMREAD_GRAYSCALE) if prob_path.exists() else None
            if prob_u8 is None:
                missing.append(prob_path.name)
                continue
            prob_maps[cname] = prob_u8.astype(np.float32) / 255.0

        if missing:
            warnings.warn(
                f"{path.name}: missing probability map(s) {missing} in {prob_dir}; skipping "
                "this crop (a missing file is not treated as all-background)."
            )
            skipped.append(path.name)
            continue

        label, order = _merge_probs_to_labels(prob_maps, merged_thresholds, conflict_margin)
        human_applied = _apply_annotation_overrides(label, path.name, annotation_index)

        out_path = out_dir / path.name
        cv2.imwrite(str(out_path), label)
        if preview_dir is not None:
            cv2.imwrite(str(preview_dir / path.name), _colorize(label))

        rows.append(_report_row(path.name, label, human_applied))
        processed.append(path.name)

    _write_report(rows, report_path)

    return MergeResult(processed=processed, skipped=skipped, out_dir=out_dir, report_path=report_path)


def _load_rgba(path: Path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.array(im)


def _colorize(label: np.ndarray) -> np.ndarray:
    out = np.zeros((*label.shape, 3), dtype=np.uint8)
    for value, color in PREVIEW_COLORS_BGR.items():
        out[label == value] = color
    return out


def validate_pseudo_labels(
    weights: dict,
    annotations_dirs: dict,
    raw_dir,
    threshold: float = 0.5,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    limit_per_class: int | None = None,
) -> dict:
    """Acceptance gate: for each class, measure the raw thresholded model
    output (NOT the merged/overridden result -- that would trivially equal
    the strokes) against every human-annotated image for that class.

    Returns {class_name: {"recall": float, "contamination": float,
    "n_images": int, "n_foreground_px": int, "n_background_px": int}}.
    Recall = fraction of human foreground-stroke pixels the model also
    calls foreground at `threshold`. Contamination = fraction of human
    background-stroke pixels the model wrongly calls foreground.
    """
    from models.UNetInference import get_predictor  # local: keep torch out of the disk-fed path

    raw_dir = Path(raw_dir)
    results = {}
    for cname, adir in annotations_dirs.items():
        if cname not in weights:
            continue
        index = discover_annotation_files(adir)
        filenames = sorted(index.keys())
        if limit_per_class is not None:
            filenames = filenames[:limit_per_class]

        predictor = get_predictor(str(weights[cname]))

        tp = fn = fp_bg = n_bg = 0
        n_images = 0
        for start in range(0, len(filenames), chunk_size):
            chunk_names = filenames[start:start + chunk_size]
            images, anns = [], []
            for name in chunk_names:
                raw_path = raw_dir / name
                if not raw_path.exists():
                    warnings.warn(f"No matching raw image found for annotation {name}; skipping")
                    continue
                image = cv2.imread(str(raw_path))
                if image is None:
                    warnings.warn(f"Could not read raw image {raw_path}; skipping")
                    continue
                images.append(image)
                anns.append(np.array(_load_rgba(index[name])))
            if not images:
                continue

            prob_maps = predictor._segment_many(images)
            for prob, ann in zip(prob_maps, anns):
                fg, bg, _defined = decode_rootpainter_annotation(ann)
                pred_fg = prob > threshold
                tp += int(np.count_nonzero(pred_fg & fg))
                fn += int(np.count_nonzero(~pred_fg & fg))
                fp_bg += int(np.count_nonzero(pred_fg & bg))
                n_bg += int(np.count_nonzero(bg))
                n_images += 1

        n_fg = tp + fn
        results[cname] = {
            "recall": tp / n_fg if n_fg else float("nan"),
            "contamination": fp_bg / n_bg if n_bg else float("nan"),
            "n_images": n_images,
            "n_foreground_px": n_fg,
            "n_background_px": n_bg,
        }
    return results


