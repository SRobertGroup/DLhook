"""Main window box handling: the seed point (collar) flow, boxes JSON save/load, the shared
CropBoxCanvas (collar override, overlap highlight) and the spatial renumbering on every change.

Tk tests are skipped without a display, like tests/test_landmark_edits.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import seedling_measurement as sm
from ui.gui import crop_boxes
from seedling_measurement import Gui
from utils import crop_boxes_io
from utils.mask_store import MaskStore

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------- shared geometry
def test_compute_crop_box_is_one_object_across_modules():
    import multi.draw_crops as dc
    import multi.src.recrop_geometry as rg
    import utils.crop_layout as cl

    assert rg.compute_crop_box is cl.compute_crop_box
    assert dc.compute_crop_box is cl.compute_crop_box
    assert rg.DEFAULT_MIN_BOX_HALF_SIZE is cl.DEFAULT_MIN_BOX_HALF_SIZE or rg.DEFAULT_MIN_BOX_HALF_SIZE == 30
    assert cl.compute_crop_box(100, 140, 200, 300) == {"cx": 120, "cy": 250, "half_w": 36, "half_h": 60}


def test_draw_crops_reexports_the_shared_canvas_helpers():
    import multi.draw_crops as dc
    import ui.crop_box_canvas as cbc

    assert dc.CropBoxCanvas is cbc.CropBoxCanvas
    assert dc.round_box is cbc.round_box and dc.resize_rect_from_corners is cbc.resize_rect_from_corners
    assert cbc.round_box({"cx": 1.4, "cy": 2.6, "half_w": 30.2, "half_h": 31.9, "collar": [5.5, 6.4]}) == {
        "cx": 1, "cy": 3, "half_w": 30, "half_h": 32, "collar": [6, 6]}


# --------------------------------------------------------------------- boxes JSON
BOXES = [{"cx": 100, "cy": 200, "half_w": 40, "half_h": 60}, {"cx": 300, "cy": 200, "half_w": 35, "half_h": 55}]
COLLARS = [(110, 230), None]


def test_boxes_json_round_trip_with_collar(tmp_path):
    path = tmp_path / "boxes.json"
    crop_boxes_io.save_boxes_json(path, "plate_A", crop_boxes_io.boxes_to_entries(BOXES, COLLARS))
    data = json.loads(path.read_text())
    assert data == {"plate_A": [dict(BOXES[0], collar=[110, 230]), BOXES[1]]}
    series, boxes, collars = crop_boxes_io.load_boxes_json(path, "plate_A")
    assert (series, boxes, collars) == ("plate_A", BOXES, COLLARS)


def test_recrop_plates_accepts_a_file_saved_by_the_gui(tmp_path):
    from multi.recrop_plates import load_boxes_file

    path = tmp_path / "boxes.json"
    crop_boxes_io.save_boxes_json(path, "plate_A", crop_boxes_io.boxes_to_entries(BOXES, COLLARS))
    loaded = load_boxes_file(str(path))
    assert list(loaded) == ["plate_A"] and loaded["plate_A"][0]["collar"] == [110, 230]


def test_save_keeps_the_other_series_of_an_existing_file(tmp_path):
    path = tmp_path / "boxes.json"
    path.write_text(json.dumps({"other": [BOXES[0]]}))
    crop_boxes_io.save_boxes_json(path, "mine", crop_boxes_io.boxes_to_entries(BOXES))
    assert set(json.loads(path.read_text())) == {"other", "mine"}


def test_load_series_choice(tmp_path):
    path = tmp_path / "boxes.json"
    path.write_text(json.dumps({"a": [BOXES[0]], "b": BOXES}))
    assert crop_boxes_io.load_boxes_json(path, "b")[1] == BOXES                        # matching name
    assert crop_boxes_io.load_boxes_json(path, "zzz", choose=lambda names: "a")[0] == "a"  # asked
    assert crop_boxes_io.load_boxes_json(path, "zzz", choose=lambda names: None) is None   # cancelled
    with pytest.raises(ValueError):
        crop_boxes_io.load_boxes_json(path, "zzz")                                       # nobody to ask
    only = tmp_path / "one.json"
    only.write_text(json.dumps({"someplate": BOXES}))
    assert crop_boxes_io.load_boxes_json(only, "different_folder")[:2] == ("someplate", BOXES)  # single series


def test_load_rejects_malformed_entries(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"a": [{"cx": 1, "cy": 2}]}))
    with pytest.raises(ValueError):
        crop_boxes_io.load_boxes_json(path, "a")


# --------------------------------------------------------------------- Gui._seed_point
def _gui_with_box(**attrs):
    g = Gui.__new__(Gui)
    g.crop_boxes = [{"cx": 100, "cy": 100, "half_w": 30, "half_h": 40}]      # crop origin (70, 60)
    g.collar_overrides = [None]
    g.landmark_store = {}
    for k, v in attrs.items():
        setattr(g, k, v)
    return g


def test_seed_point_priority_override_then_landmark_then_none():
    g = _gui_with_box(landmark_store={"0-crop-a.png": {"collar": (12.4, 33.6), "theta": 20.0}})
    assert g._seed_point(0, "0-crop-a.png") == (12, 34)                           # landmark collar, crop coords
    assert g._seed_point(0, "0-crop-b.png") is None                                # no readout for this frame
    g.landmark_store["0-crop-c.png"] = {"collar": None}
    assert g._seed_point(0, "0-crop-c.png") is None                                # readout without a collar
    g.collar_overrides = [(80, 90)]                                                # image coords
    assert g._seed_point(0, "0-crop-a.png") == (10, 30)                            # override wins, minus the origin
    assert g._seed_point(0, "0-crop-b.png") == (10, 30)


def test_seed_point_uses_the_clamped_crop_origin():
    g = _gui_with_box()
    g.crop_boxes = [{"cx": 20, "cy": 50, "half_w": 30, "half_h": 40}]            # would start at x=-10: cut starts at 0
    g.collar_overrides = [(25, 60)]
    assert g._seed_point(0, "0-crop-a.png") == (25, 50)


def test_seed_point_works_on_a_bare_headless_gui():
    g = Gui.__new__(Gui)                                                            # no collar_overrides at all
    assert g._seed_point(0, "0-crop-a.png") is None
    g.seed_point_proxies = {0: (38, 132)}
    assert g._seed_point(0, "0-crop-a.png") == (38, 132)


# --------------------------------------------------------------------- no seed line without a collar
def _frame_gui(tmp_path, monkeypatch, source):
    os.makedirs(tmp_path / "data" / "images")
    cv2.imwrite(str(tmp_path / "data" / "images" / "0-crop-a.png"), np.full((120, 60, 3), 90, np.uint8))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sm, "resolve_angle_source", lambda: source)
    g = Gui.__new__(Gui)
    g.mask_store = MaskStore()
    g.cotyl_time_series_by_crop, g.germ_time_series_by_crop = {}, {}
    g.crop_boxes = [{"cx": 30, "cy": 60, "half_w": 30, "half_h": 60}]
    g.collar_overrides = [None]
    g.landmark_store = {}
    bar = np.zeros((120, 60), np.uint8)
    bar[20:112, 25:35] = 255
    blob = np.zeros((120, 60), np.uint8)
    blob[8:24, 12:48] = 255
    g.mask_store.put_raw("0-crop-a.png", "1", blob)
    g.mask_store.put_raw("0-crop-a.png", "2", bar)
    g.mask_store.put_raw("0-crop-a.png", "hypocotyl_binary", bar.copy())
    return g


def test_ellipse_path_masks_nothing_without_a_collar(tmp_path, monkeypatch, capsys):
    g = _frame_gui(tmp_path, monkeypatch, "ellipse")
    free = g.process_single_frame("0-crop-a.png", 0)
    assert free["seed_ids"] == [1]
    assert free["hypo_mask"][100:110, 28:32].any()                                 # nothing was cut off below
    g.collar_overrides = [(30, 60)]                                                # crop (30, 60): masks rows >= 60
    cut = g.process_single_frame("0-crop-a.png", 0)
    assert not cut["hypo_mask"][70:, :].any() and cut["hypo_mask"][30:50, 28:32].any()


def test_landmark_path_uses_the_per_frame_collar_and_masks_nothing_without_it(tmp_path, monkeypatch):
    g = _frame_gui(tmp_path, monkeypatch, "landmarks")
    base = {"junction": (30.5, 20.5), "theta": 25.0, "hypo_dir": (0.0, 1.0), "cotyl_dir": (0.5, -0.5), "peak": 0.9}
    g.landmark_store = {"0-crop-a.png": dict(base)}                                # readout without a collar
    free = g.process_single_frame("0-crop-a.png", 0)
    assert free["hypo_mask"][100:110, 28:32].any()
    g.landmark_store = {"0-crop-a.png": dict(base, collar=(30.0, 60.0))}
    cut = g.process_single_frame("0-crop-a.png", 0)
    assert not cut["hypo_mask"][70:, :].any()


def test_germination_rule_falls_back_to_the_bottom_centre(monkeypatch):
    seen = {}

    class Detector:
        germination_frame = {}

        def detect(self, crop_id, contours, seed_point, crop_size):
            seen["seed"] = seed_point

    g = _gui_with_box(germination_detector=Detector(), cropped_filenames_by_crop={0: ["0-crop-a.png"]},
                      germ_time_series_by_crop={0: [[]]}, frame_results_by_crop={}, file_list=["a.png"],
                      debug_var=SimpleNamespace(get=lambda: 0))
    monkeypatch.setattr(sm, "resolve_germination_method", lambda: "rule")
    g._ensure_germination_detected(0)
    assert seen["seed"] == (30, 79)                                                # (half_w, 2 * half_h - 1)
    g.collar_overrides = [(80, 90)]
    g._ensure_germination_detected(0)
    assert seen["seed"] == (10, 30)


# --------------------------------------------------------------------- normalise / renumber
def test_normalise_orders_spatially_keeps_collars_attached_and_clamps():
    g = Gui.__new__(Gui)
    g.width1, g.height1 = 1000, 800
    boxes = [
        {"cx": 700, "cy": 200, "half_w": 50, "half_h": 60, "collar": [705, 230]},   # right
        {"cx": 200, "cy": 650, "half_w": 50, "half_h": 60},                         # second row
        {"cx": 200, "cy": 200, "half_w": 50, "half_h": 60},                         # left
        {"cx": 990, "cy": 100, "half_w": 40, "half_h": 40, "collar": [1100, 100]},  # sticks out of the image
        {"cx": 2000, "cy": 100, "half_w": 40, "half_h": 40},                        # fully outside: dropped
    ]
    out, collars, order = g._normalise_boxes(boxes)
    assert order == [3, 2, 0, 1]                                                   # top row, middle row l->r, bottom row
    assert [b["cx"] for b in out] == [975, 200, 700, 200]
    assert out[0]["cx"] + out[0]["half_w"] <= 1000                                 # clamped into the image
    assert collars[0] == (1000, 100)                                               # collar pulled into its box
    assert collars[2] == (705, 230) and collars[1] is None and collars[3] is None  # collars stay with their boxes


class _FakeCanvas:
    def __init__(self):
        self.selected_index = None
        self.boxes = []

    def set_boxes(self, boxes, selected=None):
        self.boxes, self.selected_index = [dict(b) for b in boxes], selected

    def get_boxes(self):
        return [dict(b) for b in self.boxes]


def _change_gui(monkeypatch, answer):
    g = Gui.__new__(Gui)
    g.width1, g.height1 = 1000, 800
    g.root = None
    g._busy = False
    g.canvas = _FakeCanvas()
    g.crop_boxes, g.collar_overrides, g.transformed_mid_points = [], [], []
    g.frame_results_by_crop, g.cropped_filenames_by_crop = {}, {}
    g._update_button_states = lambda: None
    g._update_status = lambda: None
    g.cleared = []
    g._clear_results = lambda: (g.cleared.append(1), g.frame_results_by_crop.clear())
    asked = []
    monkeypatch.setattr(sm.messagebox, "askyesno", lambda *a, **k: (asked.append(1), answer)[1])
    return g, asked


def test_box_change_renumbers_selection_follows_the_box(monkeypatch):
    g, asked = _change_gui(monkeypatch, True)
    g._on_boxes_changed([{"cx": 700, "cy": 200, "half_w": 50, "half_h": 60}])
    g.canvas.selected_index = 1                                                    # the user draws a box left of it
    g._on_boxes_changed([{"cx": 700, "cy": 200, "half_w": 50, "half_h": 60},
                         {"cx": 200, "cy": 200, "half_w": 50, "half_h": 60}])
    assert [b["cx"] for b in g.crop_boxes] == [200, 700] and g.canvas.selected_index == 0
    assert g.transformed_mid_points == [(200, 200), (700, 200)] and not asked


def test_changing_boxes_after_results_asks_and_can_revert(monkeypatch):
    g, asked = _change_gui(monkeypatch, False)
    g._on_boxes_changed([{"cx": 200, "cy": 200, "half_w": 50, "half_h": 60}])
    g.frame_results_by_crop = {0: [{"angle_dict": {}}]}
    edited = [{"cx": 250, "cy": 200, "half_w": 50, "half_h": 60}]
    g.canvas.boxes = edited
    g._on_boxes_changed(edited)                                                    # declined
    assert asked == [1] and g.crop_boxes[0]["cx"] == 200 and g.canvas.boxes[0]["cx"] == 200
    assert g.frame_results_by_crop and not g.cleared

    g2, asked2 = _change_gui(monkeypatch, True)
    g2._on_boxes_changed([{"cx": 200, "cy": 200, "half_w": 50, "half_h": 60}])
    g2.frame_results_by_crop = {0: [{"angle_dict": {}}]}
    g2._on_boxes_changed(edited)                                                   # accepted
    assert g2.cleared == [1] and g2.crop_boxes[0]["cx"] == 250 and not g2.frame_results_by_crop


def test_no_edits_while_the_analysis_runs(monkeypatch):
    g, asked = _change_gui(monkeypatch, True)
    g._on_boxes_changed([{"cx": 200, "cy": 200, "half_w": 50, "half_h": 60}])
    g._busy = True
    g._on_boxes_changed([])
    assert len(g.crop_boxes) == 1 and len(g.canvas.boxes) == 1


# --------------------------------------------------------------------- Tk: canvas and the whole window
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


def _canvas(root, **kw):
    from ui.crop_box_canvas import CropBoxCanvas

    changes = []
    c = CropBoxCanvas(root, width=600, height=400, on_change=changes.append, **kw)
    c.set_image(np.full((200, 300, 3), 200, np.uint8), reset_view=True)
    return c, changes


def _ev(c, img_x, img_y, shift=False):
    x, y = c._img_to_canvas(img_x, img_y)
    return SimpleNamespace(x=x, y=y, state=1 if shift else 0)


def _draw(c, a, b):
    c._on_press(_ev(c, *a))
    c._on_drag(_ev(c, *b))
    c._on_release(_ev(c, *b))


def test_canvas_draw_move_resize_delete(tk_root):
    c, changes = _canvas(tk_root)
    _draw(c, (50, 20), (90, 120))
    assert len(c.boxes) == 1 and c.selected_index == 0 and len(changes) == 1
    box = c.boxes[0]
    assert (box["cx"], box["cy"]) == (70, 70) and box["half_w"] == 36 and box["half_h"] == 60   # padded extent
    c._on_press(_ev(c, 70, 70))
    c._on_drag(_ev(c, 90, 80))
    c._on_release(_ev(c, 90, 80))
    assert (c.boxes[0]["cx"], c.boxes[0]["cy"]) == (90, 80)
    c.delete_selected()
    assert c.boxes == [] and len(changes) == 3


def test_canvas_collar_override_set_move_clear(tk_root):
    c, changes = _canvas(tk_root, allow_collar=True)
    _draw(c, (50, 20), (90, 120))
    assert "collar" not in c.boxes[0]
    c._on_press(_ev(c, 70, 100, shift=True))
    c._on_release(_ev(c, 70, 100, shift=True))
    assert c.boxes[0]["collar"] == [70, 100] and changes[-1][0]["collar"] == [70, 100]
    c._on_press(_ev(c, 70, 60))                                                    # move the box: collar follows
    c._on_drag(_ev(c, 80, 70))
    c._on_release(_ev(c, 80, 70))
    assert c.boxes[0]["collar"] == [80, 110]
    c._on_press(_ev(c, 80, 110, shift=True))                                       # shift-drag moves just the collar
    c._on_drag(_ev(c, 85, 105, shift=True))
    c._on_release(_ev(c, 85, 105, shift=True))
    assert c.boxes[0]["collar"] == [85, 105]
    c._on_right_press(_ev(c, 85, 105, shift=True))
    assert "collar" not in c.boxes[0]
    c.set_collar(0, 5000, 5000)                                                    # clamped into the box
    assert c.boxes[0]["collar"] == [c.boxes[0]["cx"] + c.boxes[0]["half_w"], c.boxes[0]["cy"] + c.boxes[0]["half_h"]]
    assert c.clear_collar() and not c.clear_collar()


def test_canvas_shift_click_is_inert_without_allow_collar(tk_root):
    c, _ = _canvas(tk_root)                                                        # draw_crops configuration
    _draw(c, (50, 20), (90, 120))
    c._on_press(_ev(c, 70, 100, shift=True))
    c._on_release(_ev(c, 70, 100, shift=True))
    assert "collar" not in c.boxes[0]


def test_canvas_highlights_overlapping_boxes(tk_root):
    c, _ = _canvas(tk_root, highlight_overlaps=True, show_labels=True)
    c.set_boxes([{"cx": 60, "cy": 100, "half_w": 40, "half_h": 60},
                 {"cx": 90, "cy": 100, "half_w": 40, "half_h": 60},
                 {"cx": 240, "cy": 100, "half_w": 30, "half_h": 60}])
    assert c.overlapping_pairs() == [(0, 1)]
    outlines = [c.itemcget(i, "outline") for i in c.find_withtag("crop_box") if c.type(i) == "rectangle"]
    assert outlines.count(c.COLOR_CONFLICT) == 2 and c.COLOR_NORMAL in outlines
    texts = [c.itemcget(i, "text") for i in c.find_withtag("crop_box") if c.type(i) == "text"]
    assert sorted(texts) == ["1", "2", "3"]
    plain, _ = _canvas(tk_root)                                                    # draw_crops: no highlight, no labels
    plain.set_boxes(c.get_boxes())
    assert c.COLOR_CONFLICT not in [plain.itemcget(i, "outline") for i in plain.find_withtag("crop_box")
                                    if plain.type(i) == "rectangle"]
    assert not [i for i in plain.find_withtag("crop_box") if plain.type(i) == "text"]


def test_main_window_end_to_end(tk_root, tmp_path, monkeypatch):
    """Open a folder, draw boxes, collar override, save / load, folder replace -- the whole window."""
    monkeypatch.chdir(REPO_ROOT)                                                   # Gui reads data/logo/*
    series = tmp_path / "plate_one"
    series.mkdir()
    for i in range(3):
        cv2.imwrite(str(series / f"frame_{i:02d}.png"), np.full((300, 400, 3), 120 + i, np.uint8))
    monkeypatch.setattr(sm, "askdirectory", lambda *a, **k: str(series))
    monkeypatch.setattr(sm.messagebox, "askyesno", lambda *a, **k: True)
    top = __import__("tkinter").Toplevel(tk_root)
    try:
        gui = Gui(top)
        assert str(gui.button_start_analysis["state"]) == "disabled"
        gui.add_files()
        assert gui.file_list == ["frame_00.png", "frame_01.png", "frame_02.png"]
        assert (gui.width1, gui.height1) == (400, 300) and gui.current_frame_name == "frame_02.png"
        assert gui.series_name == "plate_one" and gui.listbox.curselection() == (2,)

        _draw(gui.canvas, (250, 40), (270, 200))                                   # drawn first, ends up second (right)
        _draw(gui.canvas, (60, 40), (80, 200))
        assert [b["cx"] for b in gui.crop_boxes] == [70, 260] and gui.canvas.selected_index == 0
        assert str(gui.button_start_analysis["state"]) == "normal"
        gui.canvas._on_press(_ev(gui.canvas, 70, 170, shift=True))
        gui.canvas._on_release(_ev(gui.canvas, 70, 170, shift=True))
        assert gui.collar_overrides == [(70, 170), None]
        b0 = gui.crop_boxes[0]
        ox, oy = max(0, b0["cx"] - b0["half_w"]), max(0, b0["cy"] - b0["half_h"])
        assert gui._seed_point(0, "0-crop-x.png") == (70 - ox, 170 - oy)           # collar in crop coordinates
        assert gui._seed_point(1, "1-crop-x.png") is None
        assert "2 seedling(s)" in gui.status_var.get() and "1 collar override(s)" in gui.status_var.get()

        gui.listbox.selection_clear(0, "end")                                      # switching frames keeps boxes and zoom
        gui.listbox.select_set(0)
        gui._on_frame_selected()
        assert gui.current_frame_name == "frame_00.png" and len(gui.canvas.boxes) == 2

        out = tmp_path / "saved.json"
        monkeypatch.setattr(crop_boxes, "asksaveasfilename", lambda *a, **k: str(out))
        gui.save_boxes()
        assert json.loads(out.read_text())["plate_one"][0]["collar"] == [70, 170]

        monkeypatch.setattr(crop_boxes, "askopenfilename", lambda *a, **k: str(out))
        gui._set_boxes([])
        assert gui.crop_boxes == []
        gui.load_boxes()
        assert len(gui.crop_boxes) == 2 and gui.collar_overrides == [(70, 170), None]

        gui.canvas.selected_index = 0
        gui.canvas.delete_selected()
        assert len(gui.crop_boxes) == 1 and gui.collar_overrides == [None]
        gui.root.update_idletasks()
        gui.canvas._on_configure(SimpleNamespace(width=900, height=500))           # window grows: canvas follows
        assert (gui.canvas.display_width, gui.canvas.display_height) == (900, 500)
    finally:
        top.destroy()
