"""Logic behind the hand-annotation tool for ground-truth hook angles
(ui/angle_annotator.py) -- no Tk, so it is unit-testable.

Method. The annotator clicks five points on a seedling crop:

    1. the JUNCTION where the cotyledons meet the hypocotyl,
    2-3. two points on the hypocotyl axis,
    4-5. two points on the cotyledon axis.

Each axis is turned into a direction vector pointing AWAY from the junction
(the sign is decided by which side of the junction the two clicked points lie
on, not by click order). The geometric angle theta between the two vectors is
~0 for a closed hook (cotyledon folded back along the hypocotyl) and grows as
the cotyledon opens. It is converted to the app's "bio" convention with the
same formula as the in-app manual angle (ui/analysis_window.py
_place_manual_angle_point): 180 - theta, or 180 + theta for an overhooked hook,
so 180 = closed, decreasing as the hook opens, > 180 = overhooked -- directly
comparable with the exported `bio_angle` column.
"""
from __future__ import annotations

import csv
import os
import random
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime

import numpy as np

# Crop files are written by the GUI as "{crop_id}-crop-{frame}.png"; parse with
# split-on-first-dash semantics (never by first character -- breaks at id >= 10).
_CROP_RE = re.compile(r"^(\d+)-crop-(.+)\.png$", re.IGNORECASE)

STEPS = ("junction", "hypo_1", "hypo_2", "cotyl_1", "cotyl_2")
PROMPTS = {
    "junction": "Click the JUNCTION where the cotyledons meet the hypocotyl",
    "hypo_1": "HYPOCOTYL axis: first point (anywhere along it)",
    "hypo_2": "HYPOCOTYL axis: second point (further along it)",
    "cotyl_1": "COTYLEDON axis: first point (anywhere along it)",
    "cotyl_2": "COTYLEDON axis: second point (further along it)",
}

CSV_FIELDS = [
    "series", "seedling_id", "crop_id", "frame", "img_name", "crop_file", "status",
    "bio_angle", "theta", "overhook",
    "junction_x", "junction_y", "hypo1_x", "hypo1_y", "hypo2_x", "hypo2_y",
    "cotyl1_x", "cotyl1_y", "cotyl2_x", "cotyl2_y", "annotated_at",
]
_POINT_FIELDS = [("junction", "junction"), ("hypo_1", "hypo1"), ("hypo_2", "hypo2"),
                 ("cotyl_1", "cotyl1"), ("cotyl_2", "cotyl2")]


# --- geometry ---------------------------------------------------------------

def orient_axis(p, q, junction):
    """Unit vector along the line p-q, pointing away from `junction`."""
    p, q, j = (np.asarray(v, dtype=float) for v in (p, q, junction))
    v = q - p
    length = np.linalg.norm(v)
    if length < 1e-6:
        raise ValueError("the two points of an axis must be different")
    midpoint_offset = (p + q) / 2 - j
    side = float(np.dot(v, midpoint_offset))
    if abs(side) < 1e-9:
        raise ValueError("axis is ambiguous: click its points on one side of the junction")
    return (v if side > 0 else -v) / length


def hook_angle(junction, hypo_points, cotyl_points, overhook=False):
    """Returns (theta, bio_angle) in degrees for one annotated frame."""
    u = orient_axis(hypo_points[0], hypo_points[1], junction)
    v = orient_axis(cotyl_points[0], cotyl_points[1], junction)
    theta = float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))
    return theta, (180.0 + theta if overhook else 180.0 - theta)


class AngleClickSession:
    """The five-click state machine for one frame."""

    def __init__(self, points=None):
        self.points = [tuple(map(float, p)) for p in (points or [])][:len(STEPS)]

    @property
    def complete(self):
        return len(self.points) == len(STEPS)

    @property
    def step(self):
        return None if self.complete else STEPS[len(self.points)]

    def prompt(self):
        if self.complete:
            return "All 5 points placed -- Enter to save, Backspace to undo a point"
        return f"Point {len(self.points) + 1}/5 -- {PROMPTS[self.step]}"

    def add(self, x, y):
        if not self.complete:
            self.points.append((float(x), float(y)))

    def undo(self):
        if self.points:
            self.points.pop()

    def reset(self):
        self.points = []

    def named_points(self):
        return dict(zip(STEPS, self.points))

    def result(self, overhook=False):
        """(theta, bio_angle); raises ValueError if incomplete or degenerate."""
        if not self.complete:
            raise ValueError("place all 5 points first")
        j, h1, h2, c1, c2 = self.points
        return hook_angle(j, (h1, h2), (c1, c2), overhook)


# --- which frames to annotate -----------------------------------------------

def _natural_key(text):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def parse_crop_filename(name):
    """'3-crop-IMG_112.png' -> (3, 'IMG_112'), or None if not a crop file."""
    m = _CROP_RE.match(name)
    return (int(m.group(1)), m.group(2)) if m else None


def discover_crops(folder, manifest_path=None):
    """{(series, crop_id): [(frame, filename, img_name), ...]} -- frames in
    natural order.

    Plain crop folder (the GUI's data/images/): series is "" and img_name "".

    With a recrop_plates manifest (multi/recrop_plates.py writes one next to the
    crops, columns series / source_frame / crop_id / output_path): crop_id is
    only unique WITHIN a series there, so seedlings are keyed by
    (series, crop_id); img_name is the source frame with its extension, i.e.
    the exported CSV's `img_name`. Rows whose crop file is missing are dropped."""
    found = {}
    if manifest_path:
        with open(manifest_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                filename = os.path.basename(row["output_path"].replace("\\", "/"))
                parsed = parse_crop_filename(filename)
                if parsed and os.path.exists(os.path.join(folder, filename)):
                    found.setdefault((row["series"], int(row["crop_id"])), []).append(
                        (parsed[1], filename, row["source_frame"]))
    else:
        for name in os.listdir(folder):
            parsed = parse_crop_filename(name)
            if parsed:
                found.setdefault(("", parsed[0]), []).append((parsed[1], name, ""))
    return {key: sorted(frames, key=lambda f: _natural_key(f[0]))
            for key, frames in sorted(found.items(), key=lambda kv: (kv[0][0], kv[0][1]))}


def default_manifest(folder):
    """<folder>/manifest.csv if it looks like a recrop_plates manifest."""
    path = os.path.join(folder, "manifest.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh), [])
    return path if {"series", "source_frame", "crop_id", "output_path"} <= set(header) else None


def load_split_filenames(patch_index_dir, split):
    """Crop file names in the multiclass training pipeline's `split` ('train'
    or 'val'; multi/processed/patch_index/{split}_patches.csv). The split is
    per image, so 'val' crops were never seen in training -- the unbiased set
    to annotate when validating the multiclass model."""
    path = os.path.join(patch_index_dir, f"{split}_patches.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["filename"] for row in csv.DictReader(fh)}


def filter_crops(crops, allowed_filenames):
    kept = {key: [f for f in frames if f[1] in allowed_filenames] for key, frames in crops.items()}
    return {key: frames for key, frames in kept.items() if frames}


@dataclass(frozen=True)
class WorkItem:
    crop_id: int
    frame: str
    filename: str
    series: str = ""
    img_name: str = ""

    @property
    def seedling_id(self):
        return self.crop_id + 1  # crop_id is 0-based, seedling_id 1-based

    @property
    def seedling_key(self):
        return (self.series, self.crop_id)

    @property
    def seedling_label(self):
        return f"Seedling {self.seedling_id}" + (f"  [{self.series}]" if self.series else "")


def build_queue(crops, per_seedling, seed):
    """Reproducible random sample of `per_seedling` frames per seedling
    (None / too large -> every frame), grouped by seedling and shuffled within
    it so the annotator is blind to time order."""
    queue = []
    for (series, crop_id), frames in crops.items():
        rng = random.Random(f"{seed}:{series}:{crop_id}")
        frames = list(frames)
        if per_seedling and per_seedling < len(frames):
            chosen = rng.sample(frames, per_seedling)
        else:
            chosen = frames
            rng.shuffle(chosen)
        queue.extend(WorkItem(crop_id, frame, filename, series, img_name)
                     for frame, filename, img_name in chosen)
    return queue


def load_queue(folder, per_seedling, seed, manifest_path=None, split=None, patch_index_dir=None):
    """The frames to annotate, exactly as the annotator builds them (the
    sample is deterministic given these arguments)."""
    crops = discover_crops(folder, manifest_path)
    if split:
        crops = filter_crops(crops, load_split_filenames(patch_index_dir, split))
    return build_queue(crops, per_seedling, seed)


def repeat_queue(source_csvs, n=200, seed=0, max_per_seedling=4, folder=None):
    """A blind re-measurement sample: `n` frames that were already MEASURED in the given
    annotation CSVs, drawn at random over their seedlings (at most `max_per_seedling` of each,
    so the sample spreads over seedlings instead of piling up on the ones with many frames), in a
    fully shuffled order. Nothing of the earlier measurement is carried into the queue, so the
    annotator sees only the crop. With `folder`, frames whose crop file is missing are skipped.
    Deterministic for a given seed."""
    rows, seen = [], set()
    for path in source_csvs:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = (row.get("series", ""), int(row["crop_id"]), row["frame"])
                if row.get("status") != "measured" or key in seen:
                    continue
                if folder is not None and not os.path.exists(os.path.join(folder, row["crop_file"])):
                    continue
                seen.add(key)
                rows.append(row)
    rows.sort(key=lambda r: (r.get("series", ""), int(r["crop_id"]), _natural_key(r["frame"])))
    random.Random(f"repeat:{seed}").shuffle(rows)
    taken, queue = {}, []
    for row in rows:
        seedling = (row.get("series", ""), int(row["crop_id"]))
        if max_per_seedling and taken.get(seedling, 0) >= max_per_seedling:
            continue
        taken[seedling] = taken.get(seedling, 0) + 1
        queue.append(WorkItem(int(row["crop_id"]), row["frame"], row["crop_file"], row.get("series", ""),
                              row.get("img_name", "")))
        if len(queue) >= n:
            break
    return queue


# --- persistence ------------------------------------------------------------

class AnnotationStore:
    """Ground-truth CSV, one row per (seedling, frame), rewritten atomically
    on every change so a crash never leaves a partial file and the tool is
    resumable. Skipped frames are kept (status 'skipped', empty angle) so they
    are not offered again."""

    def __init__(self, path):
        self.path = path
        self.rows = {}
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    row = {**{f: "" for f in CSV_FIELDS}, **row}  # tolerate files from older versions
                    self.rows[(row["series"], int(row["crop_id"]), row["frame"])] = row

    def get(self, item):
        return self.rows.get((item.series, item.crop_id, item.frame))

    def points_for(self, item):
        row = self.get(item)
        if not row or row["status"] != "measured":
            return None
        return [(float(row[f"{p}_x"]), float(row[f"{p}_y"])) for _, p in _POINT_FIELDS]

    def counts(self, queue):
        statuses = [(self.get(i) or {}).get("status") for i in queue]
        return statuses.count("measured"), statuses.count("skipped")

    def save_measured(self, item, session, overhook):
        theta, bio = session.result(overhook)
        row = self._base_row(item, "measured")
        row.update(bio_angle=f"{bio:.3f}", theta=f"{theta:.3f}", overhook=int(bool(overhook)))
        for (_, prefix), (x, y) in zip(_POINT_FIELDS, session.points):
            row[f"{prefix}_x"], row[f"{prefix}_y"] = f"{x:.2f}", f"{y:.2f}"
        self._put(item, row)
        return theta, bio

    def save_skipped(self, item):
        self._put(item, self._base_row(item, "skipped"))

    def skip_unrecorded(self, queue):
        """Record every frame of `queue` with no row yet as skipped, in one
        write. Returns how many were added. For files made before the tool
        recorded frames passed over with Next as skipped."""
        todo = [item for item in queue if self.get(item) is None]
        for item in todo:
            self.rows[(item.series, item.crop_id, item.frame)] = self._base_row(item, "skipped")
        if todo:
            self._save_atomic()
        return len(todo)

    def _base_row(self, item, status):
        row = {f: "" for f in CSV_FIELDS}
        row.update(series=item.series, seedling_id=item.seedling_id, crop_id=item.crop_id,
                   frame=item.frame, img_name=item.img_name, crop_file=item.filename, status=status,
                   annotated_at=datetime.now().isoformat(timespec="seconds"))
        return row

    def _put(self, item, row):
        self.rows[(item.series, item.crop_id, item.frame)] = row
        self._save_atomic()

    def _save_atomic(self):
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        ordered = sorted(self.rows.values(), key=lambda r: (r["series"], int(r["crop_id"]), _natural_key(r["frame"])))
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".angle_gt_", suffix=".tmp")
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
