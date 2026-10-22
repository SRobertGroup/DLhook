"""Junction and collar reference points per crop: what the annotators clicked, what the landmark model
reads, and the corrections made in the stroke annotator (ui/seg_annotator.py).

    marks = FolderMarks("data/crops/cropped_berrybox_val")
    marks.get("0-crop-Camera_2_RPV_14_101.png")   # {"junction": (x, y) | None, "collar": ..., sources}
    marks.correct(crop, "junction", (x, y)); marks.save()

Priority for each point: corrected > clicked (angle CSV junction, root CSV collar) > model. The
corrections live in data/annotations/landmark_marks/<folder>.csv, one row per crop, and only the
`corrected` points of that file are used for retraining (multi/src/landmarks.apply_marks via
`landmarks.marks_dir`). The clicked annotations are found through the landmark training config's
sources plus EXTRA_SETS (annotated_sources), shared with multi/assess_seg_v2.py.
"""
from __future__ import annotations

import csv
import math
import os
from datetime import datetime

from utils.paths import data_path, repo_path

LANDMARK_CONFIG = "multi/configs/training_landmarks_all_a_mt.yaml"
EXTRA_SETS = [  # annotated sets that are not landmark training sources
    ("data/annotations/angles/angle_berrybox_val.csv", "data/annotations/angles/angle_berrybox_val_root.csv",
     "data/crops/cropped_berrybox_val"),
    ("data/annotations/angles/angle_round5.csv", None, "data/crops/cropped_round5"),
]
POINTS = ("junction", "collar")
CLICKED, MODEL, CORRECTED = "clicked", "model", "corrected"
CSV_FIELDS = ["crop_file", "junction_x", "junction_y", "collar_x", "collar_y", "root_dir_x", "root_dir_y",
              "junction_source", "collar_source", "edited_at"]


def _float(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _point(row, name):
    x, y = _float(row.get(f"{name}_x")), _float(row.get(f"{name}_y"))
    return None if x is None or y is None else (x, y)


def annotated_sources(config_path=LANDMARK_CONFIG):
    """[(csv, root_csv or None, crop folder)] -- the landmark config's sources plus EXTRA_SETS
    (repo-relative paths)."""
    import yaml
    with open(repo_path(config_path), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    lm = cfg["landmarks"]
    out = [(lm["csv"], lm.get("root_csv"), cfg["paths"]["raw_data_dir"])]
    out += [(s["csv"], s.get("root_csv"), s["raw_dir"]) for s in lm.get("extra_sources") or []]
    return out + EXTRA_SETS


def load_annotations(csv_path, root_csv, folder):
    """{crop_file: {"junction", "hypo2", "collar", "root_dir"}} of the measured frames whose crop exists
    (collar / root_dir from the root CSV, None where it has none)."""
    from utils.angle_review import read_rows
    out = {}
    if not os.path.exists(repo_path(csv_path)):
        return out
    roots = {}
    if root_csv and os.path.exists(repo_path(root_csv)):
        with open(repo_path(root_csv), newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if _point(row, "collar"):
                    roots[row["crop_file"]] = (_point(row, "collar"), _point(row, "root_dir"))
    for row in read_rows(repo_path(csv_path)):
        crop = row.get("crop_file")
        if row.get("status") != "measured" or not crop or not os.path.exists(os.path.join(repo_path(folder), crop)):
            continue
        j = _point(row, "junction")
        if j is None:
            continue
        collar, root_dir = roots.get(crop, (None, None))
        out[crop] = {"junction": j, "hypo2": _point(row, "hypo2"), "collar": collar, "root_dir": root_dir}
    return out


def clicked_for_folder(folder, sources=None):
    """{crop_file: annotation} merged over every annotated source whose crop folder is `folder`."""
    target = os.path.normcase(os.path.abspath(repo_path(folder)))
    out = {}
    for csv_path, root_csv, src in (sources if sources is not None else annotated_sources()):
        if os.path.normcase(os.path.abspath(repo_path(src))) == target:
            out.update(load_annotations(csv_path, root_csv, src))
    return out


def default_marks_path(folder):
    return data_path("annotations", "landmark_marks", os.path.basename(os.path.normpath(folder)) + ".csv")


class MarksStore:
    """The corrections CSV of one crop folder: {crop_file: row dict}. Saved atomically."""

    def __init__(self, path):
        self.path = path
        self.rows = {}
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as fh:
                self.rows = {r["crop_file"]: {**{f: "" for f in CSV_FIELDS}, **r} for r in csv.DictReader(fh)}

    def corrected(self, crop):
        """{"junction": (x, y) or None, "collar": ..., "root_dir": ...} -- only the corrected points."""
        row = self.rows.get(crop)
        out = {"junction": None, "collar": None, "root_dir": None}
        if row is None:
            return out
        for name in POINTS:
            if row.get(f"{name}_source") == CORRECTED:
                out[name] = _point(row, name)
        out["root_dir"] = _point(row, "root_dir")
        return out

    def set(self, crop, name, xy, root_dir=None):
        """Record a corrected point (None removes the correction)."""
        row = self.rows.setdefault(crop, {**{f: "" for f in CSV_FIELDS}, "crop_file": crop})
        row[f"{name}_x"], row[f"{name}_y"] = ("", "") if xy is None else (f"{xy[0]:.2f}", f"{xy[1]:.2f}")
        row[f"{name}_source"] = "" if xy is None else CORRECTED
        if root_dir is not None:
            row["root_dir_x"], row["root_dir_y"] = f"{root_dir[0]:.4f}", f"{root_dir[1]:.4f}"
        row["edited_at"] = datetime.now().isoformat(timespec="seconds")
        if not any(row.get(f"{p}_source") == CORRECTED for p in POINTS):
            del self.rows[crop]

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            for crop in sorted(self.rows):
                w.writerow(self.rows[crop])
        os.replace(tmp, self.path)


class FolderMarks:
    """Junction / collar per crop of one folder, corrected > clicked > model.

    `model` is a callable crop path -> readout dict (junction, collar, root_dir) or None, called lazily
    once per crop; None disables the model fallback. `clicked` defaults to clicked_for_folder."""

    def __init__(self, folder, path=None, clicked=None, model=None):
        self.folder = folder
        self.store = MarksStore(path or default_marks_path(folder))
        self.clicked = clicked_for_folder(folder) if clicked is None else clicked
        self.model = model
        self._model_cache = {}

    def _model(self, crop):
        if self.model is None:
            return None
        if crop not in self._model_cache:
            try:
                self._model_cache[crop] = self.model(os.path.join(self.folder, crop))
            except Exception:                    # a failed reading only loses the fallback
                self._model_cache[crop] = None
        return self._model_cache[crop]

    def get(self, crop, use_model=True):
        """{"junction", "collar", "root_dir", "junction_source", "collar_source"} (points None when unknown)."""
        corrected = self.store.corrected(crop)
        clicked = self.clicked.get(crop) or {}
        out = {"root_dir": corrected["root_dir"] or clicked.get("root_dir")}
        model = None
        for name in POINTS:
            if corrected[name] is not None:
                out[name], out[f"{name}_source"] = corrected[name], CORRECTED
            elif clicked.get(name) is not None:
                out[name], out[f"{name}_source"] = tuple(clicked[name]), CLICKED
            else:
                if model is None and use_model:
                    model = self._model(crop) or {}
                xy = (model or {}).get(name)
                out[name] = None if xy is None else (float(xy[0]), float(xy[1]))
                out[f"{name}_source"] = MODEL if xy is not None else ""
        if out["root_dir"] is None:
            rd = (model or {}).get("root_dir") if model else None
            out["root_dir"] = None if rd is None else (float(rd[0]), float(rd[1]))
        return out

    def correct(self, crop, name, xy):
        """Move `name` to `xy` (recorded as corrected, with the best known root direction)."""
        rd = self.get(crop)["root_dir"] if name == "collar" else None
        self.store.set(crop, name, xy, root_dir=rd)

    def restore(self, crop, name, previous):
        """Undo helper: put back a previous get() value (drops the correction when it was not one)."""
        if previous.get(f"{name}_source") == CORRECTED:
            self.store.set(crop, name, previous[name])
        else:
            self.store.set(crop, name, None)

    def save(self):
        self.store.save()


def landmark_model(checkpoints=None):
    """crop path -> readout with the GUI's landmark ensemble (loaded on first call); None when the
    readability gate rejects the crop (no seedling to read, so no marker to show)."""
    def read(path):
        from models.segmentation_backends import get_landmark_runner
        readout = get_landmark_runner(checkpoints).predict_paths([path]).get(os.path.basename(path))
        return None if not readout or readout.get("unreadable") else readout
    return read
