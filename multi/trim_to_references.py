"""Trim a new set of crop boxes so none of them touches an earlier set (the reference boxes).

For a validation set that must not contain a plant (or part of one) that was trained on. The
reference boxes are fixed, so only YOUR box moves: for every reference box it overlaps, one edge
is pulled back to the reference box's edge -- the side (left, right, top, bottom) that removes the
least area. A box that would lose more than --max-trim of its area, or shrink below the minimum
size (--min-half, default a 50 px wide box), is left unchanged and listed as needing a decision by hand (move it, or drop it).

    python -m multi.trim_to_references --boxes crop_boxes_validation.json \
        --out crop_boxes_validation_trimmed.json \
        --existing training=cropped_training_set --existing new=cropped_new_set \
        --existing open=cropped_open_set [--example-data example_data --preview-dir trim_preview]

`--existing` is the same `[LABEL=]PATH` as multi.draw_crops (crop folder, manifest.csv or boxes
.json). Without clicks there is no knowing where the seedling is inside its own box, so check the
result: `--preview-dir` writes one image per trimmed box (original orange, trimmed green,
references blue, on the series' last frame), and the trimmed file opens in
`python -m multi.draw_crops --out <trimmed file> --existing ...` for a final look.
Exit status 1 if any box is still touching a reference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi.draw_crops import load_reference_boxes  # noqa: E402
from utils.crop_layout import box_to_rect, rect_to_box, rects_overlap  # noqa: E402


def _area(r):
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


# draw_crops floors a drawn box at a 30 px half size (60 px) as a drawing convenience; nothing in the
# pipeline needs it (the crop is cut from the box as given), and most overlaps are strips of 5-11 px
# off a 60-66 px wide box, so trimming may go down to 50 px.
DEFAULT_MIN_HALF = 25


def trim_one(box: dict, refs: list, max_trim: float = 0.4, min_half: float = DEFAULT_MIN_HALF):
    """(trimmed box, status) for one box against the reference boxes of its series.

    status is "clear" (no overlap, box returned as is), "trimmed", or "manual" (could not be freed
    within `max_trim` of its area / the minimum size; the box is returned as is)."""
    rect = list(box_to_rect(box))
    original_area = _area(rect)
    touched = False
    for _ in range(4 * max(len(refs), 1)):                      # each pass removes at least one overlap
        hits = [box_to_rect(r) for r in refs if rects_overlap(rect, box_to_rect(r))]
        if not hits:
            break
        o = hits[0]
        touched = True
        candidates = []                                          # (area left, edge index, new value)
        for edge, value, keep in ((0, o[2], lambda r, v: (v, r[1], r[2], r[3])),
                                  (2, o[0], lambda r, v: (r[0], r[1], v, r[3])),
                                  (1, o[3], lambda r, v: (r[0], v, r[2], r[3])),
                                  (3, o[1], lambda r, v: (r[0], r[1], r[2], v))):
            cut = keep(rect, value)
            if cut[2] - cut[0] > 0 and cut[3] - cut[1] > 0:
                candidates.append((_area(cut), edge, value))
        if not candidates:
            return box, "manual"
        _, edge, value = max(candidates)
        rect[edge] = value
    else:
        return box, "manual"
    if not touched:
        return box, "clear"
    if (rect[2] - rect[0]) / 2 < min_half or (rect[3] - rect[1]) / 2 < min_half \
            or _area(rect) < (1 - max_trim) * original_area:
        return box, "manual"
    trimmed = rect_to_box(*rect)
    if any(rects_overlap(box_to_rect(trimmed), box_to_rect(r)) for r in refs):    # rounding must not re-open it
        return box, "manual"
    return trimmed, "trimmed"


def trim_set(boxes: dict, references: list, max_trim: float = 0.4, min_half: float = DEFAULT_MIN_HALF):
    """({series: [box, ...]}, [(series, index, status, area_kept_fraction)]) for every box of `boxes`.
    `references` is [(label, {series: [box, ...]})]."""
    out, report = {}, []
    for series, series_boxes in boxes.items():
        refs = [r for _, per in references for r in per.get(series, [])]
        out[series] = []
        for i, box in enumerate(series_boxes):
            new, status = trim_one(box, refs, max_trim, min_half)
            out[series].append(new)
            kept = _area(box_to_rect(new)) / _area(box_to_rect(box)) if _area(box_to_rect(box)) else 1.0
            report.append((series, i, status, kept))
    return out, report


def write_previews(boxes, trimmed, report, references, example_data, out_dir):
    """One PNG per changed or manual box: the plate region around it with the original box
    (orange), the trimmed box (green) and the reference boxes of that series (blue)."""
    import cv2
    from multi.draw_crops import discover_series_last_frames
    frames = {name: Path(d) / f for name, d, f in discover_series_last_frames(example_data)}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    n = 0
    for series, i, status, _ in report:
        if status == "clear" or series not in frames:
            continue
        if series not in cache:
            cache = {series: cv2.imread(str(frames[series]))}
        img = cache[series]
        if img is None:
            continue
        box = boxes[series][i]
        x1, y1, x2, y2 = (int(v) for v in box_to_rect(box))
        pad = 120
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(img.shape[1], x2 + pad), min(img.shape[0], y2 + pad)
        view = img[cy1:cy2, cx1:cx2].copy()

        def draw(rect, color, thick):
            a = (int(rect[0]) - cx1, int(rect[1]) - cy1)
            b = (int(rect[2]) - cx1, int(rect[3]) - cy1)
            cv2.rectangle(view, a, b, color, thick)

        for _, per in references:
            for r in per.get(series, []):
                draw(box_to_rect(r), (255, 128, 0), 2)
        draw(box_to_rect(box), (0, 140, 255), 2)
        if status == "trimmed":
            draw(box_to_rect(trimmed[series][i]), (0, 200, 0), 3)
        scale = min(1.0, 900 / max(view.shape[:2]))
        if scale < 1.0:
            view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.putText(view, f"{series[:28]} #{i + 1} {status}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255), 2)
        safe = "".join(c if c.isalnum() else "_" for c in series)[:30]
        cv2.imwrite(str(out_dir / f"{safe}_{i + 1:02d}_{status}.png"), view)
        n += 1
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--boxes", required=True, help="the boxes .json to trim")
    p.add_argument("--out", required=True, help="where to write the trimmed boxes .json")
    p.add_argument("--existing", action="append", required=True, metavar="[LABEL=]PATH")
    p.add_argument("--max-trim", type=float, default=0.4,
                   help="leave a box for manual handling if trimming would remove more than this share of "
                        "its area (default 0.4)")
    p.add_argument("--min-half", type=float, default=DEFAULT_MIN_HALF,
                   help="smallest half width / half height a trimmed box may have, px (default 25)")
    p.add_argument("--example-data", default=None, help="plate folders, for --preview-dir (default <repo>/example_data)")
    p.add_argument("--preview-dir", default=None)
    args = p.parse_args(argv)

    boxes = json.loads(Path(args.boxes).read_text(encoding="utf-8"))
    references = [load_reference_boxes(spec) for spec in args.existing]
    trimmed, report = trim_set(boxes, references, args.max_trim, args.min_half)
    Path(args.out).write_text(json.dumps(trimmed, indent=2), encoding="utf-8")

    counts = {s: sum(1 for r in report if r[2] == s) for s in ("clear", "trimmed", "manual")}
    print(f"{len(report)} boxes: {counts['clear']} already clear, {counts['trimmed']} trimmed, "
          f"{counts['manual']} need a decision by hand -> {args.out}")
    for series, i, status, kept in report:
        if status == "trimmed":
            print(f"  trimmed  {series[:34]:34s} #{i + 1:2d}  keeps {kept:.0%} of its area")
    for series, i, status, kept in report:
        if status == "manual":
            print(f"  MANUAL   {series[:34]:34s} #{i + 1:2d}  (still touching a reference box)")
    if args.preview_dir:
        n = write_previews(boxes, trimmed, report, references,
                           args.example_data or str(REPO_ROOT / "example_data"), args.preview_dir)
        print(f"wrote {n} preview image(s) to {args.preview_dir}")
    return 1 if counts["manual"] else 0


if __name__ == "__main__":
    sys.exit(main())
