"""Hand measurement of the hypocotyl traits (length, curvature, diameter) -- pure, no Tk.

The annotator (ui/trait_annotator.py) clicks the hypocotyl MIDLINE from the junction to the collar
(as many points as the shape needs) and three edge-to-edge TRANSECTS across the stem. From these:

  length_px                 arc length of a centripetal Catmull-Rom spline through the midline clicks,
                            resampled and smoothed exactly like the pipeline's midline
                            (hypocotyl_traits._resample_smooth)
  curvature_mean / _max     hypocotyl_traits._curvature_stats on that same curve, so hand and model
                            curvature are computed the same way
  diameter_px               median transect length (NaN without a transect)

Rows are saved by TraitStore (one per frame, atomic writes, resumable), keyed like the angle
annotator's CSV so the two can be joined; multi/validate_traits.py scores the pipeline against them.
"""
from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from datetime import datetime

import numpy as np

from utils.angle_annotation import _natural_key
from utils.hypocotyl_traits import _arc_length, _curvature_stats, _resample_smooth

MIDLINE, DIAMETER = "midline", "diameter"
N_TRANSECTS = 3
MIN_MIDLINE_POINTS = 2
SPLINE_STEP_PX = 0.5          # spline sampling before the 1 px resample
CSV_FIELDS = ["series", "seedling_id", "crop_id", "frame", "img_name", "crop_file", "status",
              "midline", "transects", "length_px", "curvature_mean", "curvature_max", "diameter_px",
              "annotated_at"]


# ---- geometry ------------------------------------------------------------------------------------
def catmull_rom(points, step=SPLINE_STEP_PX, alpha=0.5):
    """Centripetal Catmull-Rom spline through `points` ((N, 2), N >= 2) as a dense (M, 2) polyline
    passing through every point; two points give the straight segment."""
    p = np.asarray(points, dtype=float)
    keep = np.concatenate([[True], np.hypot(*np.diff(p, axis=0).T) > 1e-9])
    p = p[keep]                                              # repeated clicks would divide by zero
    if len(p) < 2:
        return p
    if len(p) == 2:
        n = max(2, int(math.ceil(np.hypot(*(p[1] - p[0])) / step)) + 1)
        t = np.linspace(0.0, 1.0, n)[:, None]
        return p[0] * (1 - t) + p[1] * t
    ext = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])  # reflected end points
    out = [p[:1]]
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        t0 = 0.0
        t1 = t0 + np.hypot(*(p1 - p0)) ** alpha
        t2 = t1 + np.hypot(*(p2 - p1)) ** alpha
        t3 = t2 + np.hypot(*(p3 - p2)) ** alpha
        n = max(2, int(math.ceil(np.hypot(*(p2 - p1)) / step)) + 1)
        t = np.linspace(t1, t2, n)[1:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.vstack(out)


def hand_midline(points):
    """The smoothed 1 px midline through the clicks, as the pipeline builds its own; None if too short."""
    if points is None or len(points) < MIN_MIDLINE_POINTS:
        return None
    return _resample_smooth(catmull_rom(points))


def hand_traits(midline_points, transects=()):
    """dict(length_px, curvature_mean, curvature_max, diameter_px, n_transects) from the clicks."""
    out = {"length_px": float("nan"), "curvature_mean": float("nan"), "curvature_max": float("nan"),
           "diameter_px": float("nan"), "n_transects": 0}
    mid = hand_midline(midline_points)
    if mid is not None:
        out["length_px"] = _arc_length(mid)
        out["curvature_mean"], out["curvature_max"] = _curvature_stats(mid)
    widths = [math.dist(a, b) for a, b in transects or ()]
    out["n_transects"] = len(widths)
    if widths:
        out["diameter_px"] = float(np.median(widths))
    return out


# ---- click session -------------------------------------------------------------------------------
class TraitClickSession:
    """Midline clicks (junction first, collar last) and up to N_TRANSECTS edge-to-edge pairs.
    `mode` decides where the next click goes; undo removes the last click of the current mode."""

    def __init__(self, midline=None, transects=None):
        self.midline = [tuple(p) for p in midline or []]
        self.transects = [tuple(map(tuple, t)) for t in transects or []]
        self.pending = None                   # first edge of a transect being clicked
        self.mode = MIDLINE

    def toggle_mode(self):
        self.mode = DIAMETER if self.mode == MIDLINE else MIDLINE
        self.pending = None

    def add(self, x, y):
        p = (float(x), float(y))
        if self.mode == MIDLINE:
            self.midline.append(p)
        elif len(self.transects) < N_TRANSECTS:
            if self.pending is None:
                self.pending = p
            else:
                self.transects.append((self.pending, p))
                self.pending = None

    def undo(self):
        if self.mode == MIDLINE:
            if self.midline:
                self.midline.pop()
        elif self.pending is not None:
            self.pending = None
        elif self.transects:
            self.transects.pop()

    def reset(self):
        self.midline, self.transects, self.pending, self.mode = [], [], None, MIDLINE

    @property
    def empty(self):
        return not self.midline and not self.transects and self.pending is None

    @property
    def complete(self):
        """Enough to save: a midline (the transects are optional, the diameter is then empty)."""
        return len(self.midline) >= MIN_MIDLINE_POINTS

    @property
    def diameter_done(self):
        return len(self.transects) >= N_TRANSECTS

    def prompt(self):
        if self.mode == MIDLINE:
            if not self.midline:
                return "Click the JUNCTION (start of the midline)"
            return f"Midline: {len(self.midline)} points -- click along the stem to the COLLAR (Tab: diameter)"
        if self.diameter_done:
            return "Diameter: 3 transects done (Enter to save)"
        edge = "second" if self.pending is not None else "first"
        return f"Diameter transect {len(self.transects) + 1}/{N_TRANSECTS}: click the {edge} edge (Tab: midline)"

    def traits(self):
        return hand_traits(self.midline, self.transects)


# ---- store ---------------------------------------------------------------------------------------
def _num(v):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def parse_points(text):
    """JSON list of points -> list of (x, y) tuples (nested for transects); [] when empty."""
    if not text:
        return []
    data = json.loads(text)
    return [tuple(map(tuple, p)) if isinstance(p[0], (list, tuple)) else tuple(p) for p in data]


class TraitStore:
    """Traits CSV, one row per (series, crop_id, frame); rewritten atomically on every change."""

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

    def session_for(self, item):
        row = self.get(item)
        if not row or row["status"] != "measured":
            return TraitClickSession()
        return TraitClickSession(parse_points(row["midline"]), parse_points(row["transects"]))

    def counts(self, queue):
        statuses = [(self.get(i) or {}).get("status") for i in queue]
        return statuses.count("measured"), statuses.count("skipped")

    def save_measured(self, item, session):
        t = session.traits()
        row = self._base_row(item, "measured")
        row.update(midline=json.dumps([[round(x, 2), round(y, 2)] for x, y in session.midline]),
                   transects=json.dumps([[[round(a[0], 2), round(a[1], 2)], [round(b[0], 2), round(b[1], 2)]]
                                         for a, b in session.transects]),
                   length_px=_num(t["length_px"]), curvature_mean=_num(t["curvature_mean"]),
                   curvature_max=_num(t["curvature_max"]), diameter_px=_num(t["diameter_px"]))
        self._put(item, row)
        return t

    def save_skipped(self, item):
        self._put(item, self._base_row(item, "skipped"))

    def _base_row(self, item, status):
        row = {f: "" for f in CSV_FIELDS}
        row.update(series=item.series, seedling_id=item.seedling_id, crop_id=item.crop_id, frame=item.frame,
                   img_name=item.img_name, crop_file=item.filename, status=status,
                   annotated_at=datetime.now().isoformat(timespec="seconds"))
        return row

    def _put(self, item, row):
        self.rows[(item.series, item.crop_id, item.frame)] = row
        self._save_atomic()

    def _save_atomic(self):
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        ordered = sorted(self.rows.values(), key=lambda r: (r["series"], int(r["crop_id"]), _natural_key(r["frame"])))
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".traits_", suffix=".tmp")
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


def read_measured(path):
    """[(crop_file, midline points, transects, row)] of the measured rows of a traits CSV."""
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") == "measured":
                out.append((row["crop_file"], parse_points(row["midline"]), parse_points(row["transects"]), row))
    return out
