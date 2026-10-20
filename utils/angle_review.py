"""Review of earlier angle annotations: the pure logic behind ui/angle_reviewer.py.

Fixing mistakes and marking frames where no angle can be measured, without overwriting the
original CSV (it stays the first-round record the repeatability study compares against).
Decisions go to a sidecar, `<csv>.review.csv`, with the annotator's own columns and one of these
statuses per reviewed frame:

  measured      corrected clicks: replace the original row
  confirmed     checked and left as it was
  rotation      the cotyledons pass in front of or behind the hypocotyl: no angle can be measured
  unmeasurable  no angle for another reason (out of the crop, hidden, ...)

`read_rows(csv)` is what the trainers and the scorers read: the original rows with the sidecar
applied (measured replaces; rotation / unmeasurable take the frame out of the measured set but keep
it, with that status, so it can be told apart from a plain skip; confirmed changes nothing).

Two queues feed the tool: the measured frames the model disagrees with most (`disagreement_queue`),
and the frames you skipped after the seedling germinated (`skipped_after_onset_queue`) -- before
germination there is no hook to measure, so those skips are not worth a second look.
"""
from __future__ import annotations

import csv
import os

from utils.angle_annotation import (
    CSV_FIELDS, AnnotationStore, WorkItem, _natural_key, discover_crops,
)

REVIEW_SUFFIX = ".review.csv"
ROTATION, UNMEASURABLE, CONFIRMED = "rotation", "unmeasurable", "confirmed"
NOT_MEASURABLE = (ROTATION, UNMEASURABLE)


def review_path(csv_path):
    return str(csv_path) + REVIEW_SUFFIX


def _key(row):
    return (row.get("series", ""), int(row["crop_id"]), row["frame"])


def read_rows(csv_path, apply_review=True):
    """The rows of an annotation CSV (as dicts, file order) with its review sidecar applied."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = [{**{f: "" for f in CSV_FIELDS}, **row} for row in csv.DictReader(fh)]
    sidecar = review_path(csv_path)
    if not apply_review or not os.path.exists(sidecar):
        return rows
    with open(sidecar, newline="", encoding="utf-8") as fh:
        reviewed = {_key(r): {**{f: "" for f in CSV_FIELDS}, **r} for r in csv.DictReader(fh)}
    out = []
    for row in rows:
        r = reviewed.get(_key(row))
        if r is None or r["status"] == CONFIRMED:
            out.append(row)
        elif r["status"] == "measured" or r["status"] in NOT_MEASURABLE:
            out.append(r)
        else:
            out.append(row)
    return out


def rotation_keys(csv_path):
    """{(series, crop_id, frame)} marked rotation in the sidecar of `csv_path`."""
    sidecar = review_path(csv_path)
    if not os.path.exists(sidecar):
        return set()
    with open(sidecar, newline="", encoding="utf-8") as fh:
        return {_key(r) for r in csv.DictReader(fh) if r["status"] == ROTATION}


class ReviewStore(AnnotationStore):
    """The sidecar. Same atomic, resumable CSV as the annotation store; `points_for` falls back
    to the original clicks so a frame opens with what you clicked the first time."""

    def __init__(self, csv_path):
        super().__init__(review_path(csv_path))
        self.original = AnnotationStore(csv_path)

    def points_for(self, item):
        corrected = super().points_for(item)
        return corrected if corrected is not None else self.original.points_for(item)

    def original_row(self, item):
        return self.original.get(item)

    def save_confirmed(self, item):
        self._put(item, self._base_row(item, CONFIRMED))

    def save_rotation(self, item):
        self._put(item, self._base_row(item, ROTATION))

    def save_unmeasurable(self, item):
        self._put(item, self._base_row(item, UNMEASURABLE))

    def counts_by_status(self, queue):
        statuses = [(self.get(i) or {}).get("status") for i in queue]
        return {s: statuses.count(s) for s in ("measured", CONFIRMED, ROTATION, UNMEASURABLE)}


def _items(rows, folder):
    return [WorkItem(int(r["crop_id"]), r["frame"], r["crop_file"], r.get("series", ""), r.get("img_name", ""))
            for r in rows if os.path.exists(os.path.join(folder, r["crop_file"]))]


def disagreement_queue(csv_path, folder, errors=None, only_errors=False):
    """Measured frames of `csv_path` (sidecar applied), the ones the model disagrees with most first.

    `errors` is {crop_file: angle error in degrees} (multi.validate_landmarks --per-crop-out). Frames
    without an error come last, the most open hooks (largest theta) first, since those fail most;
    with `only_errors` they are left out (a review limited to a list of suspect frames)."""
    rows = [r for r in read_rows(csv_path) if r["status"] == "measured"]
    errors = errors or {}
    if only_errors:
        rows = [r for r in rows if r["crop_file"] in errors]
    rows.sort(key=lambda r: (r["crop_file"] not in errors, -errors.get(r["crop_file"], 0.0),
                             -float(r["theta"] or 0), r.get("series", ""), int(r["crop_id"]),
                             _natural_key(r["frame"])))
    return _items(rows, folder)


def load_errors(path):
    """{crop_file: error} from a per-crop errors CSV with `crop_file` and `err` columns."""
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["crop_file"]: float(r["err"]) for r in csv.DictReader(fh) if r.get("err") not in (None, "")}


def onsets(germination_csv):
    """{(series, crop_id): first frame index with a visible radicle} -- 0 for a radicle already
    visible in the first frame, absent for a seedling with no usable onset."""
    out = {}
    with open(germination_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (r.get("series", ""), int(r["crop_id"]))
            if r["status"] == "found" and r.get("onset_index") not in (None, ""):
                out[key] = int(r["onset_index"])
            elif r["status"] == "before_start":
                out[key] = 0
    return out


def skipped_after_onset_queue(csv_path, germination_csv, folder, manifest_path=None):
    """Frames skipped in `csv_path` that come at or after their seedling's germination frame, in time
    order per seedling. The frame index is the position in the seedling's timeline, the one the
    germination annotator used (utils.angle_annotation.discover_crops)."""
    crops = discover_crops(folder, manifest_path)
    onset = onsets(germination_csv)
    position = {(s, c, frame): i for (s, c), frames in crops.items() for i, (frame, _, _) in enumerate(frames)}
    rows = []
    for r in read_rows(csv_path):
        if r["status"] != "skipped":
            continue
        key = _key(r)
        seedling = (key[0], key[1])
        if seedling in onset and key in position and position[key] >= onset[seedling]:
            rows.append(r)
    rows.sort(key=lambda r: (r.get("series", ""), int(r["crop_id"]), _natural_key(r["frame"])))
    return _items(rows, folder)
