"""Collar and root-direction annotation: the pure logic behind ui/root_annotator.py.

The hook-angle annotator (utils/angle_annotation.py) records the junction, the
hypocotyl axis and the cotyledon axis. This module adds the other end of the
seedling, on the SAME frames:

  radicle_visible   is a radicle (root) visible at all in this frame?
  collar            the root-hypocotyl transition (the plant collar)
  root point        one point on the root axis, away from the collar

so a landmark model can later learn the collar position and the root direction,
and germination can be dated as the first frame with a visible radicle.

A frame with radicle_visible = 0 stores no points. The work list is the set of
frames already MEASURED in an angle CSV (angle_landmarks_train.csv,
angle_ground_truth.csv), in time order -- not shuffled, because judging when a
radicle first appears needs the frames before it.
"""
from __future__ import annotations

import csv
import math
import os
import tempfile
from datetime import datetime

from utils.angle_annotation import WorkItem, _natural_key

CSV_FIELDS = ["series", "seedling_id", "crop_id", "frame", "crop_file", "status", "radicle_visible",
              "collar_x", "collar_y", "root_x", "root_y", "root_dir_x", "root_dir_y", "annotated_at"]
_POINT_NAMES = ("collar", "root")
_PROMPTS = (
    "Click the COLLAR: where the root meets the hypocotyl",
    "Click a point on the ROOT axis, away from the collar",
)
MIN_ROOT_LENGTH_PX = 3.0


def root_direction(collar, root_point):
    """Unit vector (x, y) from the collar to the root point; ValueError when they coincide."""
    dx, dy = root_point[0] - collar[0], root_point[1] - collar[1]
    length = math.hypot(dx, dy)
    if length < MIN_ROOT_LENGTH_PX:
        raise ValueError("The root point is on top of the collar -- click further along the root")
    return dx / length, dy / length


class RootClickSession:
    """Two clicks: the collar, then one point on the root axis."""

    def __init__(self, points=None):
        self.points = list(points) if points else []

    @property
    def complete(self):
        return len(self.points) == len(_POINT_NAMES)

    def prompt(self):
        return "Both points placed -- Enter to save" if self.complete else _PROMPTS[len(self.points)]

    def add(self, x, y):
        if not self.complete:
            self.points.append((float(x), float(y)))

    def undo(self):
        if self.points:
            self.points.pop()

    def reset(self):
        self.points = []

    def named_points(self):
        return dict(zip(_POINT_NAMES, self.points))

    def direction(self):
        return root_direction(*self.points)


def measured_work_items(angles_csv):
    """The frames with status 'measured' in an angle CSV, as WorkItems in time
    order (series, seedling, natural frame order)."""
    items = []
    with open(angles_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") == "measured":
                items.append(WorkItem(int(row["crop_id"]), row["frame"], row["crop_file"],
                                      row.get("series", ""), row.get("img_name", "")))
    items.sort(key=lambda i: (i.series, i.crop_id, _natural_key(i.frame)))
    return items


def angle_points(angles_csv):
    """{(series, crop_id, frame): {"junction", "hypo_1", ... } in crop pixels} from an
    angle CSV, for drawing the existing clicks as context."""
    names = ("junction", "hypo1", "hypo2", "cotyl1", "cotyl2")
    out = {}
    with open(angles_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") == "measured":
                out[(row["series"], int(row["crop_id"]), row["frame"])] = {
                    n: (float(row[f"{n}_x"]), float(row[f"{n}_y"])) for n in names}
    return out


class RootStore:
    """Collar / root CSV, one row per reviewed frame, rewritten atomically on every change."""

    def __init__(self, path):
        self.path = path
        self.rows = {}
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    row = {**{f: "" for f in CSV_FIELDS}, **row}
                    self.rows[(row["series"], int(row["crop_id"]), row["frame"])] = row

    def get(self, item):
        return self.rows.get((item.series, item.crop_id, item.frame))

    def points_for(self, item):
        row = self.get(item)
        if not row or row["status"] != "annotated" or row["radicle_visible"] != "1":
            return None
        return [(float(row["collar_x"]), float(row["collar_y"])), (float(row["root_x"]), float(row["root_y"]))]

    def counts(self, queue):
        """(with radicle, without radicle, skipped) over the queue."""
        with_r = without_r = skipped = 0
        for item in queue:
            row = self.get(item)
            if not row:
                continue
            if row["status"] == "skipped":
                skipped += 1
            elif row["radicle_visible"] == "1":
                with_r += 1
            else:
                without_r += 1
        return with_r, without_r, skipped

    def save_radicle(self, item, session):
        """A frame with a visible radicle: needs both clicks."""
        if not session.complete:
            raise ValueError("Place the collar and one root point first")
        ux, uy = session.direction()
        (cx, cy), (rx, ry) = session.points
        row = self._base_row(item, "annotated", 1)
        row.update(collar_x=f"{cx:.2f}", collar_y=f"{cy:.2f}", root_x=f"{rx:.2f}", root_y=f"{ry:.2f}",
                   root_dir_x=f"{ux:.4f}", root_dir_y=f"{uy:.4f}")
        self._put(item, row)

    def save_no_radicle(self, item):
        self._put(item, self._base_row(item, "annotated", 0))

    def save_skipped(self, item):
        self._put(item, self._base_row(item, "skipped", ""))

    def _base_row(self, item, status, visible):
        row = {f: "" for f in CSV_FIELDS}
        row.update(series=item.series, seedling_id=item.seedling_id, crop_id=item.crop_id, frame=item.frame,
                   crop_file=item.filename, status=status, radicle_visible=str(visible),
                   annotated_at=datetime.now().isoformat(timespec="seconds"))
        return row

    def _put(self, item, row):
        self.rows[(item.series, item.crop_id, item.frame)] = row
        self._save_atomic()

    def _save_atomic(self):
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        ordered = sorted(self.rows.values(), key=lambda r: (r["series"], int(r["crop_id"]), _natural_key(r["frame"])))
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".root_gt_", suffix=".tmp")
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
