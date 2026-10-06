"""Germination-onset annotation: the pure logic behind ui/germination_annotator.py.

The angle and collar/root annotators only offer frames where the hook can be
measured, which is after germination, so none of their frames sit in the
germination window. This tool works on whole seedling TIMELINES instead: for each
seedling you scrub through its frames in time order and mark the first frame in
which a radicle is visible (germination starts as soon as one is). One decision
per seedling, but it yields a label for every frame:

  status        meaning                                       frame labels (radicle_visible)
  found         onset at frame k                              0 before k, 1 from k on
  before_start  a radicle is already visible in frame 0       1 everywhere (onset is censored)
  none          no radicle in any frame                       0 everywhere (onset is after the series)
  skipped       cannot tell / unusable seedling               not exported

Optionally the collar and one root-axis point are clicked on the onset frame, the
same landmarks as utils/root_annotation.py.

Frames are the crops you point it at: cropped_training_set holds every 5th source
frame, so the onset is resolved to that spacing. For a finer onset, re-crop the
early frames with multi/recrop_plates.py --every-n 1 into another folder.
"""
from __future__ import annotations

import csv
import os
import tempfile
from datetime import datetime

import hashlib

from utils.angle_annotation import _natural_key, discover_crops
from utils.root_annotation import RootClickSession

STATUSES = ("found", "before_start", "none", "skipped")
CSV_FIELDS = ["series", "seedling_id", "crop_id", "status", "onset_frame", "onset_index", "n_frames",
              "collar_x", "collar_y", "root_x", "root_y", "annotated_at"]


class Seedling:
    """One timeline: its frames as (frame, filename, img_name) in time order."""

    def __init__(self, series, crop_id, frames):
        self.series = series
        self.crop_id = crop_id
        self.frames = list(frames)

    @property
    def key(self):
        return (self.series, self.crop_id)

    @property
    def seedling_id(self):
        return self.crop_id + 1

    @property
    def label(self):
        return f"Seedling {self.seedling_id}" + (f"  [{self.series}]" if self.series else "")

    def __len__(self):
        return len(self.frames)


def in_test_subset(series, crop_id, test_fraction=0.2, seed=0):
    """Deterministic seedling-level hold-out: True for about `test_fraction` of the
    seedlings, the same ones on every run."""
    digest = hashlib.sha1(f"{seed}:{series}:{crop_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2 ** 32 < test_fraction


def load_seedlings(folder, manifest_path=None, subset=None, test_fraction=0.2, seed=0, min_frames=2):
    """Full timelines of the seedlings in `folder`, ordered by (series, crop_id).

    subset 'train' / 'test' keeps the seedlings outside / inside a deterministic
    hold-out. The split is per SEEDLING, not per image: the multiclass training
    split is per image and leaves a seedling with a few scattered frames, which
    cannot show when a radicle first appears."""
    crops = discover_crops(folder, manifest_path)
    out = []
    for (series, crop_id), frames in crops.items():
        if len(frames) < min_frames:
            continue
        if subset and in_test_subset(series, crop_id, test_fraction, seed) != (subset == "test"):
            continue
        out.append(Seedling(series, crop_id, frames))
    return out


class GerminationStore:
    """One row per seedling, rewritten atomically on every change (resumable)."""

    def __init__(self, path):
        self.path = path
        self.rows = {}
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    row = {**{f: "" for f in CSV_FIELDS}, **row}
                    self.rows[(row["series"], int(row["crop_id"]))] = row

    def get(self, seedling):
        return self.rows.get(seedling.key)

    def onset_index(self, seedling):
        row = self.get(seedling)
        if row and row["status"] == "found" and row["onset_index"] != "":
            return int(row["onset_index"])
        return None

    def points_for(self, seedling):
        row = self.get(seedling)
        if not row or row["collar_x"] == "":
            return None
        return [(float(row["collar_x"]), float(row["collar_y"])), (float(row["root_x"]), float(row["root_y"]))]

    def counts(self, seedlings):
        statuses = [(self.get(s) or {}).get("status") for s in seedlings]
        return {name: statuses.count(name) for name in STATUSES}

    def save_found(self, seedling, index, session=None):
        if not 0 <= index < len(seedling):
            raise ValueError("onset frame out of range")
        row = self._base_row(seedling, "found")
        row.update(onset_frame=seedling.frames[index][0], onset_index=index)
        if session is not None and session.points:
            if not session.complete:
                raise ValueError("Place both the collar and a root point, or neither")
            session.direction()                     # rejects a root point on top of the collar
            (cx, cy), (rx, ry) = session.points
            row.update(collar_x=f"{cx:.2f}", collar_y=f"{cy:.2f}", root_x=f"{rx:.2f}", root_y=f"{ry:.2f}")
        self._put(seedling, row)

    def save_status(self, seedling, status):
        if status not in ("before_start", "none", "skipped"):
            raise ValueError(f"use save_found for status {status!r}")
        self._put(seedling, self._base_row(seedling, status))

    def _base_row(self, seedling, status):
        row = {f: "" for f in CSV_FIELDS}
        row.update(series=seedling.series, seedling_id=seedling.seedling_id, crop_id=seedling.crop_id,
                   status=status, n_frames=len(seedling),
                   annotated_at=datetime.now().isoformat(timespec="seconds"))
        return row

    def _put(self, seedling, row):
        self.rows[seedling.key] = row
        self._save_atomic()

    def _save_atomic(self):
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        ordered = sorted(self.rows.values(), key=lambda r: (r["series"], int(r["crop_id"])))
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".germination_gt_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(ordered)
            os.replace(tmp, self.path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise


FRAME_LABEL_FIELDS = ["series", "seedling_id", "crop_id", "frame", "crop_file", "frame_index",
                      "radicle_visible", "offset_from_onset", "status"]


def frame_labels(seedlings, store):
    """One dict per labelled frame (see the module docstring for the rules).
    `offset_from_onset` is frame_index - onset_index for 'found' seedlings (negative
    = before germination, 0 = the onset frame) and blank otherwise, so a training set
    can concentrate on the frames around germination."""
    out = []
    for s in seedlings:
        row = store.get(s)
        if not row or row["status"] == "skipped":
            continue
        status = row["status"]
        onset = store.onset_index(s)
        for i, (frame, filename, _) in enumerate(s.frames):
            if status == "found":
                visible, offset = int(i >= onset), i - onset
            else:
                visible, offset = (1 if status == "before_start" else 0), ""
            out.append({"series": s.series, "seedling_id": s.seedling_id, "crop_id": s.crop_id, "frame": frame,
                        "crop_file": filename, "frame_index": i, "radicle_visible": visible,
                        "offset_from_onset": offset, "status": status})
    return out


def write_frame_labels(path, seedlings, store):
    rows = frame_labels(seedlings, store)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FRAME_LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
