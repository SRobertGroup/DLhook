"""Records of manual landmark edits made in the analysis window, kept for retraining -- pure, no Tk.

`data/` is wiped at startup and exit, so each "Apply landmarks" with "Save edits for retraining"
ticked writes a self-contained JSON record plus a copy of the crop into
a folder of the user's choice. `python -m multi.edits_to_csv` turns that folder into annotator-schema
CSVs and a crop folder the landmark trainer reads (landmarks.extra_sources).

Naming. A GUI session has no series name: the folder of the raw images is the series
(`series_name_from_path`, whitespace -> "_"). GUI crops are "<crop_id>-crop-<stem>.png"; the curated
training sets name a crop "<crop_id>-crop-<series>_<stem>.png" with frame "<series>_<stem>" and
img_name "<stem><ext>". Records keep the GUI names (series, source_frame = "<stem><ext>", crop_file);
the converter derives the curated names from them.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
from datetime import datetime

import numpy as np

SCHEMA_VERSION = 1


def series_name_from_path(path):
    """Series name of a GUI session: the raw image folder's name, whitespace replaced by '_'."""
    name = os.path.basename(os.path.normpath(str(path))) if path else ""
    name = name if name not in ("", ".") else "series"
    return re.sub(r"\s+", "_", name)


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    return v


def _xy_fields(record, prefix, point):
    record[f"{prefix}_x"], record[f"{prefix}_y"] = (
        (round(float(point[0]), 2), round(float(point[1]), 2)) if point is not None else (None, None))


def edit_record(series, source_frame, crop_file, crop_box, seed_id, crop_id, manual, model_readout, overhook):
    """JSON-serialisable record of one manual edit.

    `manual` is `LandmarkClickSession.result()` (the raw clicks are its hypo1/hypo2/cotyl1 = base/
    cotyl2 = tip/root_point entries); `model_readout` the reading it replaced (or None); `overhook`
    the final flag the user committed. Field names follow the annotator CSV."""
    rec = {
        "version": SCHEMA_VERSION,
        "series": series,
        "source_frame": source_frame,        # "<stem><ext>" of the raw image, e.g. "RPV_14_21.jpg"
        "crop_file": crop_file,              # the GUI crop name, "<crop_id>-crop-<stem>.png"
        "crop_box": _jsonable(crop_box) if crop_box is not None else None,
        "seed_id": int(seed_id),
        "crop_id": int(crop_id),
        "status": "measured",
    }
    _xy_fields(rec, "junction", manual["junction"])
    _xy_fields(rec, "hypo1", manual["hypo1"])
    _xy_fields(rec, "hypo2", manual["hypo2"])
    _xy_fields(rec, "cotyl1", manual["cotyl1"])
    _xy_fields(rec, "cotyl2", manual["cotyl2"])
    _xy_fields(rec, "collar", manual.get("collar"))
    _xy_fields(rec, "root", manual.get("root_point"))
    rec["theta"] = round(float(manual["theta"]), 3)
    rec["bio_angle"] = round(180.0 + manual["theta"] if overhook else 180.0 - manual["theta"], 3)
    rec["overhook"] = int(bool(overhook))
    rec["model_readout"] = _jsonable(model_readout) if model_readout else None
    rec["timestamp"] = datetime.now().isoformat(timespec="seconds")
    return rec


def record_stem(record):
    """File stem of a record: unique per (series, crop, frame), so re-editing a frame overwrites it."""
    base = os.path.splitext(os.path.basename(record["crop_file"]))[0]
    return f"{record['series']}_{base}"


def write_edit(directory, record, crop_path, edited_mask=None):
    """Write <directory>/<stem>.json, copy the crop to <directory>/crops/<stem>.png and, for
    `edited_mask` (one array, or {label: array}; 255 = foreground), save it under masks/.
    Returns the JSON path."""
    import cv2

    stem = record_stem(record)
    os.makedirs(os.path.join(directory, "crops"), exist_ok=True)
    rec = dict(record)
    if crop_path is not None and os.path.exists(crop_path):
        shutil.copyfile(crop_path, os.path.join(directory, "crops", f"{stem}.png"))
        rec["crop_image"] = f"crops/{stem}.png"
    masks = {}
    if edited_mask is not None:
        masks = dict(edited_mask) if isinstance(edited_mask, dict) else {"": edited_mask}
    if masks:
        os.makedirs(os.path.join(directory, "masks"), exist_ok=True)
        rec["edited_masks"] = {}
        for label, mask in masks.items():
            name = f"{stem}_{label}.png" if label else f"{stem}.png"
            cv2.imwrite(os.path.join(directory, "masks", name), np.asarray(mask, dtype=np.uint8))
            rec["edited_masks"][label or "mask"] = f"masks/{name}"
    path = os.path.join(directory, f"{stem}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(rec), fh, indent=1)
    return path


def read_records(directory):
    """All records in `directory` (the *.json written by write_edit), oldest first."""
    out = []
    for name in sorted(os.listdir(directory)):
        if name.lower().endswith(".json"):
            with open(os.path.join(directory, name), encoding="utf-8") as fh:
                rec = json.load(fh)
            if isinstance(rec, dict) and rec.get("status") == "measured" and "crop_file" in rec:
                out.append(rec)
    out.sort(key=lambda r: r.get("timestamp", ""))
    return out
