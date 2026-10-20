"""Turn the analysis window's "Save edits for retraining" records into landmark training data.

    python -m multi.edits_to_csv <edits_dir> --out-prefix data/annotations/angles/angle_edits [--crop-dir data/crops/cropped_edits]

reads every record in <edits_dir> (utils/landmark_edits.write_edit) and writes

  <out-prefix>.csv        the angle annotator's CSV (utils/angle_annotation.CSV_FIELDS)
  <out-prefix>_root.csv   the collar / root CSV (utils/root_annotation.CSV_FIELDS), for the records
                          that carry both a collar and a root point
  <crop-dir>/             the crops under the curated naming "<crop_id>-crop-<series>_<stem>.png"
                          (default <out-prefix>_crops) and a manifest.csv with recrop_plates' columns

so that a `landmarks.extra_sources` entry {csv, root_csv, raw_dir: <crop-dir>, tag} trains on them
with no new loader. Names: series = the raw image folder of the GUI session, frame =
"<series>_<stem>", img_name = the raw file name; see utils/landmark_edits.py. Re-edits of one frame
keep the latest record.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi.recrop_plates import MANIFEST_FIELDS  # noqa: E402
from utils import angle_annotation as aa  # noqa: E402
from utils import root_annotation as ra  # noqa: E402
from utils.landmark_edits import read_records  # noqa: E402
from utils.root_annotation import root_direction  # noqa: E402


def curated_names(record):
    """(frame, crop file) of a record in the curated naming."""
    parsed = aa.parse_crop_filename(os.path.basename(record["crop_file"]))
    stem = parsed[1] if parsed else os.path.splitext(os.path.basename(record["crop_file"]))[0]
    frame = f"{record['series']}_{stem}"
    return frame, f"{int(record['crop_id'])}-crop-{frame}.png"


def _fmt(v):
    return "" if v is None else f"{float(v):.2f}"


def angle_row(record):
    frame, crop_file = curated_names(record)
    row = {f: "" for f in aa.CSV_FIELDS}
    row.update(series=record["series"], seedling_id=int(record["crop_id"]) + 1, crop_id=int(record["crop_id"]),
               frame=frame, img_name=record.get("source_frame", ""), crop_file=crop_file, status="measured",
               bio_angle=f"{float(record['bio_angle']):.3f}", theta=f"{float(record['theta']):.3f}",
               overhook=int(bool(record["overhook"])), annotated_at=record.get("timestamp", ""))
    for prefix in ("junction", "hypo1", "hypo2", "cotyl1", "cotyl2"):
        row[f"{prefix}_x"], row[f"{prefix}_y"] = _fmt(record[f"{prefix}_x"]), _fmt(record[f"{prefix}_y"])
    return row


def root_row(record):
    """Collar / root row, or None when the record lacks the collar or the root point."""
    if any(record.get(k) is None for k in ("collar_x", "collar_y", "root_x", "root_y")):
        return None
    try:
        ux, uy = root_direction((record["collar_x"], record["collar_y"]), (record["root_x"], record["root_y"]))
    except ValueError:
        return None
    frame, crop_file = curated_names(record)
    row = {f: "" for f in ra.CSV_FIELDS}
    row.update(series=record["series"], seedling_id=int(record["crop_id"]) + 1, crop_id=int(record["crop_id"]),
               frame=frame, crop_file=crop_file, status="annotated", radicle_visible="1",
               collar_x=_fmt(record["collar_x"]), collar_y=_fmt(record["collar_y"]),
               root_x=_fmt(record["root_x"]), root_y=_fmt(record["root_y"]),
               root_dir_x=f"{ux:.4f}", root_dir_y=f"{uy:.4f}", annotated_at=record.get("timestamp", ""))
    return row


def _manifest_row(record, output_path, crop_w, crop_h):
    box = record.get("crop_box") or {}
    row = {f: "" for f in MANIFEST_FIELDS}
    row.update(output_path=output_path, series=record["series"], source_frame=record.get("source_frame", ""),
               crop_id=int(record["crop_id"]), crop_width=crop_w, crop_height=crop_h)
    if all(k in box for k in ("cx", "cy", "half_w", "half_h")):
        cx, cy, hw, hh = (int(box[k]) for k in ("cx", "cy", "half_w", "half_h"))
        row.update(cx=cx, cy=cy, half_w=hw, half_h=hh, x1=max(0, cx - hw), y1=max(0, cy - hh),
                   x2=cx + hw, y2=cy + hh)
    return row


def _write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def convert(edits_dir, out_prefix, crop_dir=None):
    """Write the CSVs, crop folder and manifest. Returns {"angle": n, "root": n, "crops": n, "skipped": n,
    "csv", "root_csv", "crop_dir", "manifest"}."""
    import cv2

    edits_dir = str(edits_dir)
    crop_dir = str(crop_dir) if crop_dir else f"{out_prefix}_crops"
    latest = {}
    for rec in read_records(edits_dir):                # oldest first: later records win
        frame, _ = curated_names(rec)
        latest[(rec["series"], int(rec["crop_id"]), frame)] = rec

    os.makedirs(crop_dir, exist_ok=True)
    angle_rows, root_rows, manifest_rows, skipped = [], [], [], 0
    for rec in latest.values():
        src = os.path.join(edits_dir, rec["crop_image"]) if rec.get("crop_image") else None
        if src is None or not os.path.exists(src):
            skipped += 1
            continue
        _, crop_file = curated_names(rec)
        shutil.copyfile(src, os.path.join(crop_dir, crop_file))
        img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2] if img is not None else ("", "")
        angle_rows.append(angle_row(rec))
        rr = root_row(rec)
        if rr is not None:
            root_rows.append(rr)
        manifest_rows.append(_manifest_row(rec, os.path.join(crop_dir, crop_file), w, h))

    key = lambda r: (r["series"], int(r["crop_id"]), aa._natural_key(r["frame"]))   # noqa: E731
    angle_rows.sort(key=key)
    root_rows.sort(key=key)
    csv_path, root_path = f"{out_prefix}.csv", f"{out_prefix}_root.csv"
    _write_csv(csv_path, aa.CSV_FIELDS, angle_rows)
    if root_rows:
        _write_csv(root_path, ra.CSV_FIELDS, root_rows)

    # manifest: keep rows of an existing manifest that this run does not replace
    manifest_path = os.path.join(crop_dir, "manifest.csv")
    kept = []
    if os.path.exists(manifest_path):
        new_names = {os.path.basename(r["output_path"]) for r in manifest_rows}
        with open(manifest_path, newline="", encoding="utf-8") as fh:
            kept = [r for r in csv.DictReader(fh) if os.path.basename(r["output_path"]) not in new_names]
    _write_csv(manifest_path, MANIFEST_FIELDS, kept + manifest_rows)
    return {"angle": len(angle_rows), "root": len(root_rows), "crops": len(manifest_rows), "skipped": skipped,
            "csv": csv_path, "root_csv": root_path if root_rows else None, "crop_dir": crop_dir,
            "manifest": manifest_path}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("edits_dir", help="folder written by 'Save edits for retraining'")
    p.add_argument("--out-prefix", required=True, help="writes <prefix>.csv and <prefix>_root.csv")
    p.add_argument("--crop-dir", default=None, help="crop folder (default <out-prefix>_crops)")
    args = p.parse_args(argv)
    out = convert(args.edits_dir, args.out_prefix, args.crop_dir)
    print(f"{out['angle']} angle rows -> {out['csv']}; {out['root']} collar/root rows"
          + (f" -> {out['root_csv']}" if out["root_csv"] else "")
          + f"; {out['crops']} crops + manifest in {out['crop_dir']}"
          + (f"; {out['skipped']} records skipped (crop image missing)" if out["skipped"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
