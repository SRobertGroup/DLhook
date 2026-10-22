"""Save / load the main window's crop boxes as JSON, in the multi/draw_crops.py and
multi/recrop_plates.py format::

    {"series_name": [{"cx": 428, "cy": 2783, "half_w": 120, "half_h": 90, "collar": [430, 2800]}, ...]}

"collar" (image pixels) is optional per box; recrop_plates.load_boxes_file ignores it. Pure
Python, no Tk. A file may hold several series (the one being saved replaces only its own entry).
"""
from __future__ import annotations

import json
import os
import tempfile

BOX_KEYS = ("cx", "cy", "half_w", "half_h")


def boxes_to_entries(boxes, collars=None) -> list:
    """JSON entries for `boxes` (dicts with cx/cy/half_w/half_h), with each box's collar override
    (an (x, y) pair or None, `collars` parallel to `boxes`) added as "collar": [x, y]."""
    entries = []
    for i, box in enumerate(boxes):
        entry = {k: int(round(box[k])) for k in BOX_KEYS}
        collar = collars[i] if collars is not None and i < len(collars) else box.get("collar")
        if collar is not None:
            entry["collar"] = [int(round(collar[0])), int(round(collar[1]))]
        entries.append(entry)
    return entries


def entries_to_boxes(entries):
    """(boxes, collars) from JSON entries; raises ValueError on a malformed entry."""
    boxes, collars = [], []
    for entry in entries:
        missing = [k for k in BOX_KEYS if k not in entry]
        if missing:
            raise ValueError(f"crop box entry is missing {missing}: {entry!r}")
        boxes.append({k: int(round(entry[k])) for k in BOX_KEYS})
        collar = entry.get("collar")
        if collar is not None:
            if len(collar) != 2:
                raise ValueError(f"collar must be [x, y]: {entry!r}")
            collars.append((int(round(collar[0])), int(round(collar[1]))))
        else:
            collars.append(None)
    return boxes, collars


def save_boxes_json(path, series_name, entries) -> None:
    """Write `entries` under `series_name`, keeping the other series of an existing file.
    Atomic: a temp file in the same directory, then os.replace."""
    data = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, ValueError):
            data = {}
    data[series_name] = entries
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".crop_boxes_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_boxes_json(path, series_name, choose=None):
    """(series_used, boxes, collars) from a boxes file.

    Series choice: `series_name` when the file has it; else the file's only series; else
    `choose(names)` (a callable returning one of the names, or None to cancel). Returns None when
    the choice was cancelled. Raises ValueError for a malformed file or when no series can be
    chosen."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or not data:
        raise ValueError("not a crop boxes file: expected {series_name: [boxes, ...]}")
    names = list(data)
    if series_name in data:
        used = series_name
    elif len(names) == 1:
        used = names[0]
    elif choose is not None:
        used = choose(names)
        if used is None:
            return None
        if used not in data:
            raise ValueError(f"series {used!r} is not in the file (has: {', '.join(names)})")
    else:
        raise ValueError(f"series {series_name!r} is not in the file (has: {', '.join(names)})")
    boxes, collars = entries_to_boxes(data[used])
    return used, boxes, collars
