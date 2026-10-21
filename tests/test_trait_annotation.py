import math
import os
import types

import cv2
import numpy as np
import pytest

from utils.angle_annotation import WorkItem
from utils.trait_annotation import (DIAMETER, MIDLINE, TraitClickSession, TraitStore, catmull_rom, hand_traits,
                                    read_measured)


def test_straight_midline_length_and_flat_curvature():
    t = hand_traits([(10, 10), (10, 50), (10, 100)])
    assert t["length_px"] == pytest.approx(90, abs=0.5)
    assert t["curvature_mean"] < 1e-3
    assert math.isnan(t["diameter_px"]) and t["n_transects"] == 0


def test_arc_curvature_and_length():
    r = 60.0
    angles = np.linspace(0, math.pi / 2, 7)
    pts = [(r * math.cos(a), r * math.sin(a)) for a in angles]
    t = hand_traits(pts)
    assert t["length_px"] == pytest.approx(r * math.pi / 2, rel=0.03)
    assert t["curvature_mean"] == pytest.approx(1 / r, rel=0.1)


def test_spline_passes_through_the_clicks():
    pts = [(0, 0), (10, 5), (20, 20), (22, 40)]
    curve = catmull_rom(pts)
    for p in pts:
        assert np.min(np.hypot(curve[:, 0] - p[0], curve[:, 1] - p[1])) < 1e-6


def test_diameter_is_the_median_transect():
    t = hand_traits([(0, 0), (0, 40)], [((0, 0), (8, 0)), ((0, 0), (10, 0)), ((0, 0), (30, 0))])
    assert t["diameter_px"] == pytest.approx(10) and t["n_transects"] == 3


def test_session_modes_undo_and_completion():
    s = TraitClickSession()
    assert s.empty and not s.complete and s.mode == MIDLINE
    s.add(1, 1)
    s.add(1, 20)
    assert s.complete
    s.toggle_mode()
    assert s.mode == DIAMETER
    s.add(0, 10)
    assert s.pending == (0.0, 10.0) and not s.transects
    s.add(5, 10)
    assert s.transects == [((0.0, 10.0), (5.0, 10.0))]
    for _ in range(4):
        s.add(0, 0)
        s.add(1, 0)
    assert len(s.transects) == 3 and s.diameter_done                  # capped at three
    s.undo()
    assert len(s.transects) == 2
    s.toggle_mode()
    s.undo()
    assert s.midline == [(1.0, 1.0)]
    s.reset()
    assert s.empty


def test_store_round_trip(tmp_path):
    path = str(tmp_path / "t.csv")
    item = WorkItem(crop_id=2, frame="F_01", filename="2-crop-F_01.png", series="S")
    s = TraitClickSession([(1, 1), (1, 30), (4, 60)], [((0, 10), (6, 10))])
    store = TraitStore(path)
    t = store.save_measured(item, s)
    again = TraitStore(path)
    s2 = again.session_for(item)
    assert s2.midline == [(1.0, 1.0), (1.0, 30.0), (4.0, 60.0)] and s2.transects == [((0.0, 10.0), (6.0, 10.0))]
    (crop, mid, tr, row), = read_measured(path)
    assert crop == "2-crop-F_01.png" and float(row["length_px"]) == pytest.approx(t["length_px"], abs=1e-3)
    other = WorkItem(crop_id=3, frame="F_01", filename="3-crop-F_01.png", series="S")
    again.save_skipped(other)
    assert TraitStore(path).counts([item, other]) == (1, 1)


def test_validate_traits_compare():
    from multi.validate_traits import compare
    s = compare([10, 21, float("nan")], [10, 20, 5])
    assert s["n"] == 2 and s["bias"] == pytest.approx(0.5) and s["off"] == pytest.approx(0.0)
    assert compare([], [])["n"] == 0


@pytest.fixture(scope="module")
def tk_root():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    yield root
    root.destroy()


def test_annotator_measure_save_resume_and_blind_model(tmp_path, tk_root):
    from ui.trait_annotator import TraitAnnotator, measurable_queue

    folder = str(tmp_path / "crops")
    os.makedirs(folder)
    queue = []
    for i in range(2):
        name = f"{i}-crop-S_{i:03d}.png"
        cv2.imwrite(os.path.join(folder, name), np.full((80, 40, 3), 90, np.uint8))
        queue.append(WorkItem(crop_id=i, frame=f"S_{i:03d}", filename=name))
    assert measurable_queue(queue, {queue[1].filename: {"collar": (1, 2)}}) == [queue[1]]
    calls = []

    def model(image, j, c):
        calls.append((j, c))
        return np.array([[20.0, 5.0], [20.0, 60.0]]), hand_traits([j, c])

    store = TraitStore(str(tmp_path / "t.csv"))
    app = TraitAnnotator(tk_root, folder, queue, store, model_traits=model)
    try:
        app.canvas.canvas_to_image = lambda x, y: (x, y)
        for p in ((20, 5), (20, 30), (21, 60)):
            app._on_click(types.SimpleNamespace(x=p[0], y=p[1]))
        app.toggle_model()
        assert not app.show_model and not calls                               # blind until saved
        app.toggle_mode()
        app._on_click(types.SimpleNamespace(x=15, y=30))
        app._on_click(types.SimpleNamespace(x=25, y=30))
        assert app.confirm() and app.index == 1
        app.move(-1)
        assert app.saved and len(app.session.transects) == 1
        app.toggle_model()
        assert app.show_model and calls == [((20.0, 5.0), (21.0, 60.0))]
        assert any(s["kind"] == "polyline" and s["color"] == "#ffe600" for s in app.canvas._overlay)
        app.move(1)
        app._on_click(types.SimpleNamespace(x=3, y=3))
        app.move(1)                                                           # unsaved clicks: refused
        assert app.index == 1 and store.get(queue[1]) is None
    finally:
        app.destroy()
    again = TraitAnnotator(tk_root, folder, queue, TraitStore(str(tmp_path / "t.csv")))
    try:
        assert again.index == 1                                               # resumes at the first unrecorded frame
    finally:
        again.destroy()
