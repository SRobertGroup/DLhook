"""Landmark click session and overlay shapes shared by the analysis window and the annotators -- no Tk.

A landmark edit is five clicks (junction J, two points on the hypocotyl axis H1/H2, the cotyledon
base C1 and tip C2) plus two optional ones (the collar and one point along the root). `result()`
turns them into a readout-shaped dict (the same keys the landmark model's readout carries, see
utils/landmark_readout.py) so a manual edit and a model reading are interchangeable downstream.
`landmark_shapes()` draws either clicked points or a readout as `ZoomableImageCanvas.set_overlay`
shapes; ui/angle_annotator.py uses it too, so both tools look the same.

Angle convention: theta is the angle between the two axes (0 = closed), bio = 180 - theta, or
180 + theta when the hook is flagged overhooked (180 = closed, > 180 = overhooked).
"""
from __future__ import annotations

import numpy as np

from utils.angle_annotation import orient_axis
from utils.landmark_readout import rays_cross, theta_between
from utils.root_annotation import root_direction

STEPS = ("junction", "hypo_1", "hypo_2", "cotyl_base", "cotyl_tip", "collar", "root")
N_ANGLE_POINTS = 5

PROMPTS = {
    "junction": "Click the junction J",
    "hypo_1": "Click a point on the hypocotyl (H1)",
    "hypo_2": "Click a second point on the hypocotyl (H2)",
    "cotyl_base": "Click the cotyledon BASE (C1)",
    "cotyl_tip": "Click the cotyledon TIP (C2)",
    "collar": "Click the collar (optional - Enter to skip)",
    "root": "Click a point along the root (optional - Enter to skip)",
}

# Colours, as the annotators draw them (RGB hex for Tk).
HYPO_COLOR = "#ff9f1c"
COTYL_COLOR = "#2ec4b6"
JUNCTION_COLOR = "#ff3b30"
COLLAR_COLOR = "#7b2cbf"
ROOT_COLOR = "#e63946"

_LABELS = {"junction": "J", "hypo_1": "H1", "hypo_2": "H2", "cotyl_1": "C1", "cotyl_2": "C2"}
_COLORS = {"junction": JUNCTION_COLOR, "hypo_1": HYPO_COLOR, "hypo_2": HYPO_COLOR,
           "cotyl_1": COTYL_COLOR, "cotyl_2": COTYL_COLOR}
_ALIASES = {"cotyl_base": "cotyl_1", "cotyl_tip": "cotyl_2"}


class LandmarkClickSession:
    """Click state machine for one frame: 5 angle points, then an optional collar and root point."""

    def __init__(self):
        self.points = []
        self.skipped = False

    @property
    def angle_complete(self):
        return len(self.points) >= N_ANGLE_POINTS

    @property
    def complete(self):
        return len(self.points) >= len(STEPS) or (self.skipped and self.angle_complete)

    @property
    def step(self):
        return None if self.complete else STEPS[len(self.points)]

    def prompt(self):
        if self.complete:
            return "All landmarks placed - Apply landmarks (Backspace undoes a point)"
        return PROMPTS[self.step]

    def add(self, x, y):
        if not self.complete:
            self.points.append((float(x), float(y)))

    def undo(self):
        self.skipped = False
        if self.points:
            self.points.pop()

    def reset(self):
        self.points = []
        self.skipped = False

    def skip_optional(self):
        """Give up the remaining optional steps (collar / root); only once the angle is placed."""
        if self.angle_complete:
            self.skipped = True

    def named_points(self):
        return dict(zip(STEPS, self.points))

    def result(self, overhook=None):
        """Readout-shaped dict of the clicks. `overhook` overrides the automatic flag (None = use
        `overhook_auto`). Raises ValueError on degenerate clicks or when the angle is not complete."""
        if not self.angle_complete:
            raise ValueError("place the first 5 points (J, H1, H2, C1, C2) first")
        j, h1, h2, base, tip = self.points[:N_ANGLE_POINTS]
        hypo_dir = orient_axis(h1, h2, j)
        cotyl_dir = orient_axis(base, tip, j)
        theta = theta_between(hypo_dir, cotyl_dir)
        along = np.asarray(tip, float) - np.asarray(base, float)
        n = float(np.hypot(*along))
        auto = bool(n > 1e-6 and rays_cross(base, along / n, j, hypo_dir))
        flag = auto if overhook is None else bool(overhook)
        collar = self.points[5] if len(self.points) > 5 else None
        root_point = self.points[6] if len(self.points) > 6 else None
        root_dir = None
        if collar is not None and root_point is not None:
            ux, uy = root_direction(collar, root_point)
            root_dir = (float(ux), float(uy))
        return {
            "junction": j,
            "hypo_dir": (float(hypo_dir[0]), float(hypo_dir[1])),
            "cotyl_dir": (float(cotyl_dir[0]), float(cotyl_dir[1])),
            "base": base, "tip": tip,
            "theta": theta,
            "overhook_auto": auto,
            "overhook": flag,
            "bio": 180.0 + theta if flag else 180.0 - theta,
            "collar": collar, "root_dir": root_dir,
            "peak": 1.0, "edit": "manual",
            # the raw clicks, named like the annotator CSV (retraining records)
            "hypo1": h1, "hypo2": h2, "cotyl1": base, "cotyl2": tip, "root_point": root_point,
        }


def _xy(p):
    return (float(p[0]), float(p[1]))


def _shapes_from_points(points, arrow_min):
    pts = {_ALIASES.get(k, k): v for k, v in points.items()}
    shapes = []
    for name, xy in pts.items():
        if name in _LABELS:
            shapes.append({"kind": "point", "xy": xy, "color": _COLORS[name], "label": _LABELS[name]})

    if "junction" in pts:
        j = pts["junction"]
        for a, b, color in (("hypo_1", "hypo_2", HYPO_COLOR), ("cotyl_1", "cotyl_2", COTYL_COLOR)):
            if a in pts and b in pts:
                shapes.append({"kind": "line", "from": pts[a], "to": pts[b], "color": color, "dash": True})
                try:  # arrow from the junction along the auto-oriented axis
                    ux, uy = orient_axis(pts[a], pts[b], j)
                except ValueError:
                    continue
                length = max(float(arrow_min), 1.2 * max(abs(pts[a][0] - j[0]) + abs(pts[a][1] - j[1]),
                                                         abs(pts[b][0] - j[0]) + abs(pts[b][1] - j[1])))
                shapes.append({"kind": "line", "from": j, "to": (j[0] + ux * length, j[1] + uy * length),
                               "color": color, "arrow": True})
    if "collar" in pts:
        shapes.append({"kind": "point", "xy": pts["collar"], "color": COLLAR_COLOR, "label": "collar"})
    if "root" in pts:
        shapes.append({"kind": "point", "xy": pts["root"], "color": ROOT_COLOR, "label": "root"})
        if "collar" in pts:
            shapes.append({"kind": "line", "from": pts["collar"], "to": pts["root"],
                           "color": ROOT_COLOR, "arrow": True})
    return shapes


def _shapes_from_readout(readout, arrow_min):
    if not readout or readout.get("junction") is None:
        return []
    j = _xy(readout["junction"])
    shapes = [{"kind": "point", "xy": j, "color": JUNCTION_COLOR, "label": "J"}]
    for key, color in (("hypo_dir", HYPO_COLOR), ("cotyl_dir", COTYL_COLOR)):
        d = readout.get(key)
        if d is not None:
            shapes.append({"kind": "line", "from": j,
                           "to": (j[0] + float(d[0]) * arrow_min, j[1] + float(d[1]) * arrow_min),
                           "color": color, "arrow": True})
    if readout.get("base") is not None:
        shapes.append({"kind": "point", "xy": _xy(readout["base"]), "color": COTYL_COLOR, "label": "C1"})
    if readout.get("tip") is not None:
        shapes.append({"kind": "point", "xy": _xy(readout["tip"]), "color": COTYL_COLOR, "label": "C2"})
    collar = readout.get("collar")
    if collar is not None:
        c = _xy(collar)
        shapes.append({"kind": "point", "xy": c, "color": COLLAR_COLOR, "label": "collar"})
        d = readout.get("root_dir")
        if d is not None:
            shapes.append({"kind": "line", "from": c,
                           "to": (c[0] + float(d[0]) * arrow_min, c[1] + float(d[1]) * arrow_min),
                           "color": ROOT_COLOR, "arrow": True})
    return shapes


MUTED_COLOR = "#9a9a9a"      # a readout the readability gate rejected
FLAG_COLOR = "#ffb000"       # a readout the time-series checks flagged (utils/landmark_temporal.py)


def landmark_shapes(points=None, readout=None, *, arrow_min=40, color=None):
    """Overlay shapes for `ZoomableImageCanvas.set_overlay`.

    `points` = {name: (x, y)} of clicked points (a session's `named_points()`, or the annotator's
    junction/hypo_1/hypo_2/cotyl_1/cotyl_2): labelled points, dashed axes, arrows from the junction
    along each oriented axis, and the collar -> root arrow. Otherwise `readout` (a model or manual
    readout dict): junction point, hypocotyl / cotyledon arrows along the stored directions,
    base / tip points and the collar with its root arrow when present. `color`, when given, replaces
    every shape's colour (MUTED_COLOR for a gated model readout; default unchanged). Neither -> []."""
    if points:
        shapes = _shapes_from_points(points, arrow_min)
    elif readout:
        shapes = _shapes_from_readout(readout, arrow_min)
    else:
        return []
    if color is not None:
        shapes = [dict(s, color=color) for s in shapes]
    return shapes


# --- turning a model reading into a manual one --------------------------------------------------
HYPO_POINTS_PX = (10.0, 40.0)      # synthetic H1 / H2 distance from the junction along hypo_dir
COTYL_BASE_PX = 2.0                # synthetic C1 (base) distance from the junction along cotyl_dir
COTYL_MIN_PX = 20.0                # synthetic C2 (tip) distance when the reading has no tip
ROOT_POINT_PX = 30.0               # synthetic root point distance from the collar along root_dir


def session_from_readout(readout):
    """A LandmarkClickSession whose clicks reproduce `readout` (model or manual): junction, two
    hypocotyl points along hypo_dir, cotyledon base / tip along cotyl_dir (the tip at the reading's
    tip distance when it has one), then its collar and a root point along root_dir. Its result()
    (overhook=False) has the reading's theta and bio = 180 - theta, as the model reports them.
    None when the reading has no junction or directions."""
    if not readout or readout.get("junction") is None or readout.get("hypo_dir") is None \
            or readout.get("cotyl_dir") is None:
        return None
    j = np.asarray(readout["junction"], float)
    h = np.asarray(readout["hypo_dir"], float)
    c = np.asarray(readout["cotyl_dir"], float)
    if np.hypot(*h) < 1e-9 or np.hypot(*c) < 1e-9:
        return None
    h, c = h / np.hypot(*h), c / np.hypot(*c)
    tip = readout.get("tip")
    reach = max(COTYL_MIN_PX, float(np.hypot(*(np.asarray(tip, float) - j)))) if tip is not None else COTYL_MIN_PX
    session = LandmarkClickSession()
    for p in (j, j + HYPO_POINTS_PX[0] * h, j + HYPO_POINTS_PX[1] * h, j + COTYL_BASE_PX * c, j + reach * c):
        session.add(float(p[0]), float(p[1]))
    collar = readout.get("collar")
    if collar is not None:
        session.add(float(collar[0]), float(collar[1]))
        root_dir = readout.get("root_dir")
        if root_dir is not None:
            r = np.asarray(collar, float) + ROOT_POINT_PX * np.asarray(root_dir, float)
            session.add(float(r[0]), float(r[1]))
    return session


def previous_collar(frame_results, seed_id, index):
    """(frame index, collar, root_dir) of the nearest frame before `index` whose collar can be
    trusted: a Manual frame with a collar, or a model reading the gate kept and the time-series
    checks did not flag for its collar. None when there is none."""
    for k in range(index - 1, -1, -1):
        result = frame_results[k]
        if result is None:
            continue
        readout = (result.get("landmarks") or {}).get(seed_id)
        if not readout or readout.get("collar") is None:
            continue
        manual = (result.get("state_dict") or {}).get(seed_id) == "Manual" or readout.get("edit") == "manual"
        trusted = manual or readout.get("collar_edit") == "manual" or (
            not readout.get("gated") and "collar" not in (readout.get("temporal_flag") or ""))
        if trusted:
            return k, tuple(float(v) for v in readout["collar"]), readout.get("root_dir")
    return None
