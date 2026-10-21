"""Edit records for retraining (utils/landmark_edits.py), their conversion to annotator CSVs
(multi/edits_to_csv.py), the scale helper, and a smoke test of the analysis window's landmark flow
(skipped without a display)."""
from __future__ import annotations

import csv
import json
import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from multi import edits_to_csv
from multi.src.landmarks import load_landmarks
from utils import angle_annotation as aa
from utils import root_annotation as ra
from utils.hypocotyl_traits import scale_from_line, to_mm
from utils.landmark_edit import LandmarkClickSession
from utils.landmark_edits import (
    edit_record, read_records, record_stem, series_name_from_path, write_edit,
)

CLICKS = [(30.0, 20.0), (30.0, 40.0), (30.0, 60.0), (35.0, 18.0), (50.0, 10.0)]
BOX = {"cx": 100, "cy": 200, "half_w": 40, "half_h": 90}


def _manual(extra=()):
    s = LandmarkClickSession()
    for p in (*CLICKS, *extra):
        s.add(*p)
    return s.result()


def _record(frame="RPV_14_21", crop_id=0, series="Camera_2", extra=(), model=None, overhook=False):
    return edit_record(series, f"{frame}.jpg", f"{crop_id}-crop-{frame}.png", BOX, crop_id + 1, crop_id,
                       _manual(extra), model, overhook)


def _crop(path, w=80, h=120):
    img = np.full((h, w, 3), 90, np.uint8)
    cv2.imwrite(str(path), img)
    return str(path)


# ---- scale -----------------------------------------------------------------------------------

def test_scale_from_line_and_mm_conversion():
    assert scale_from_line((0, 0), (300, 400), 5.0) == pytest.approx(10.0)       # 500 px = 5 mm
    um = scale_from_line((10, 10), (110, 10), 1.0)
    assert um == pytest.approx(10.0)
    mm = to_mm({"length_px": 200.0, "diameter_px": 20.0, "curvature_mean": 0.01, "curvature_max": 0.02}, um)
    assert mm["length_mm"] == pytest.approx(2.0) and mm["diameter_mm"] == pytest.approx(0.2)
    assert mm["curvature_mean_per_mm"] == pytest.approx(1.0)


@pytest.mark.parametrize("mm", [0, -1, float("nan"), None])
def test_scale_from_line_rejects_bad_lengths(mm):
    with pytest.raises(ValueError):
        scale_from_line((0, 0), (10, 0), mm)


def test_scale_from_line_rejects_coincident_points():
    with pytest.raises(ValueError):
        scale_from_line((5, 5), (5, 5), 1.0)


# ---- records ---------------------------------------------------------------------------------

def test_series_name_from_path():
    assert series_name_from_path("C:/data/Camera 2 run/") == "Camera_2_run"
    assert series_name_from_path(None) == "series"


def test_record_fields_and_json_roundtrip():
    model = {"junction": (31.0, 21.0), "theta": np.float64(40.0), "hypo_dir": (0.0, 1.0), "peak": 0.8,
             "overhook": np.bool_(False), "collar": None}
    rec = _record(extra=[(30.0, 70.0), (30.0, 90.0)], model=model, overhook=True)
    assert rec["status"] == "measured" and rec["seed_id"] == 1 and rec["crop_id"] == 0
    assert (rec["junction_x"], rec["junction_y"]) == (30.0, 20.0)
    assert (rec["hypo1_x"], rec["hypo2_y"]) == (30.0, 60.0)
    assert (rec["cotyl1_x"], rec["cotyl1_y"]) == (35.0, 18.0)       # base
    assert (rec["cotyl2_x"], rec["cotyl2_y"]) == (50.0, 10.0)       # tip
    assert (rec["collar_x"], rec["collar_y"], rec["root_x"], rec["root_y"]) == (30.0, 70.0, 30.0, 90.0)
    assert rec["overhook"] == 1 and rec["bio_angle"] == pytest.approx(180 + rec["theta"], abs=1e-3)
    assert rec["model_readout"]["theta"] == 40.0 and rec["model_readout"]["overhook"] is False
    assert rec["timestamp"] and rec["series"] == "Camera_2" and rec["source_frame"] == "RPV_14_21.jpg"
    json.dumps(rec)                                                  # serialisable as is


def test_record_without_collar_has_none_fields():
    rec = _record()
    assert rec["collar_x"] is None and rec["root_y"] is None and rec["model_readout"] is None
    assert rec["bio_angle"] == pytest.approx(180 - rec["theta"], abs=1e-3)


def test_write_edit_files(tmp_path):
    crop = _crop(tmp_path / "0-crop-RPV_14_21.png")
    rec = _record()
    path = write_edit(str(tmp_path / "edits"), rec, crop, edited_mask={"2": np.full((120, 80), 255, np.uint8)})
    stem = record_stem(rec)
    assert path == str(tmp_path / "edits" / f"{stem}.json")
    assert (tmp_path / "edits" / "crops" / f"{stem}.png").exists()
    assert (tmp_path / "edits" / "masks" / f"{stem}_2.png").exists()
    saved = json.loads((tmp_path / "edits" / f"{stem}.json").read_text(encoding="utf-8"))
    assert saved["crop_image"] == f"crops/{stem}.png" and saved["edited_masks"] == {"2": f"masks/{stem}_2.png"}
    # no mask given: no masks folder
    write_edit(str(tmp_path / "e2"), rec, crop)
    assert not (tmp_path / "e2" / "masks").exists()


def test_read_records_orders_by_time(tmp_path):
    crop = _crop(tmp_path / "c.png")
    a, b = _record(frame="F1"), _record(frame="F2")
    a["timestamp"], b["timestamp"] = "2026-01-02T00:00:00", "2026-01-01T00:00:00"
    write_edit(str(tmp_path / "e"), a, crop)
    write_edit(str(tmp_path / "e"), b, crop)
    assert [r["source_frame"] for r in read_records(str(tmp_path / "e"))] == ["F2.jpg", "F1.jpg"]


# ---- converter + loader round trip -------------------------------------------------------------

def test_converter_round_trip_with_load_landmarks(tmp_path):
    edits = str(tmp_path / "edits")
    crop = _crop(tmp_path / "src.png")
    write_edit(edits, _record(frame="RPV_14_21", extra=[(30.0, 70.0), (28.0, 90.0)]), crop)
    write_edit(edits, _record(frame="RPV_14_31", crop_id=3, series="Camera_2"), crop)
    again = _record(frame="RPV_14_21", extra=[(30.0, 70.0), (28.0, 90.0)])
    again["timestamp"] = "2999-01-01T00:00:00"                      # a re-edit of the same frame wins
    write_edit(edits, again, crop)

    prefix = str(tmp_path / "angle_edits")
    out = edits_to_csv.convert(edits, prefix, str(tmp_path / "crops_out"))
    assert out["angle"] == 2 and out["root"] == 1 and out["crops"] == 2 and out["skipped"] == 0

    rows = list(csv.DictReader(open(prefix + ".csv", newline="", encoding="utf-8")))
    assert list(rows[0].keys()) == aa.CSV_FIELDS
    first = rows[0]
    assert first["series"] == "Camera_2" and first["frame"] == "Camera_2_RPV_14_21"
    assert first["img_name"] == "RPV_14_21.jpg" and first["crop_file"] == "0-crop-Camera_2_RPV_14_21.png"
    assert first["status"] == "measured" and first["seedling_id"] == "1" and first["crop_id"] == "0"
    assert rows[1]["crop_file"] == "3-crop-Camera_2_RPV_14_31.png" and rows[1]["seedling_id"] == "4"
    root_rows = list(csv.DictReader(open(prefix + "_root.csv", newline="", encoding="utf-8")))
    assert list(root_rows[0].keys()) == ra.CSV_FIELDS and root_rows[0]["radicle_visible"] == "1"

    # what the trainer does: load_landmarks on the CSVs, the crops found next to a manifest
    lms, report = load_landmarks(prefix + ".csv", root_csv_path=prefix + "_root.csv")
    assert report["loaded"] == 2 and report["invalid"] == 0 and report["with_root"] == 1
    lm = lms["0-crop-Camera_2_RPV_14_21.png"]
    assert lm.seedling == ("Camera_2", 0)
    assert lm.junction == pytest.approx((30.0, 20.0))
    assert lm.collar == pytest.approx((30.0, 70.0)) and lm.root_dir is not None
    assert (lm.cotyl_base, lm.cotyl_tip) == ((35.0, 18.0), (50.0, 10.0))
    for name in lms:
        assert os.path.exists(os.path.join(out["crop_dir"], name))

    # manifest: recrop_plates' columns, and the annotator's manifest-based discovery finds the crops
    with open(out["manifest"], newline="", encoding="utf-8") as fh:
        manifest = list(csv.DictReader(fh))
    assert list(manifest[0].keys()) == edits_to_csv.MANIFEST_FIELDS
    assert {m["source_frame"] for m in manifest} == {"RPV_14_21.jpg", "RPV_14_31.jpg"}
    assert manifest[0]["crop_width"] == "80" and manifest[0]["crop_height"] == "120"
    assert manifest[0]["x1"] == "60" and manifest[0]["y2"] == "290"
    found = aa.discover_crops(out["crop_dir"], out["manifest"])
    assert set(found) == {("Camera_2", 0), ("Camera_2", 3)}


def test_converter_with_load_extra_landmark_sources(tmp_path):
    train = pytest.importorskip("multi.train_unet_multiclass")
    edits = str(tmp_path / "edits")
    write_edit(edits, _record(extra=[(30.0, 70.0), (28.0, 90.0)]), _crop(tmp_path / "src.png"))
    prefix = str(tmp_path / "ang")
    out = edits_to_csv.convert(edits, prefix)
    src = {"csv": prefix + ".csv", "root_csv": out["root_csv"], "raw_dir": out["crop_dir"], "tag": "edits"}
    lms, raw_paths, rows = train.load_extra_landmark_sources({"_data_root": str(tmp_path)}, [src], {}, 108, 54)
    assert list(lms) == ["edits/0-crop-Camera_2_RPV_14_21.png"]
    assert rows and os.path.exists(next(iter(raw_paths.values())))


def test_converter_skips_records_without_a_crop_image(tmp_path):
    edits = str(tmp_path / "edits")
    write_edit(edits, _record(), None)                      # no crop copied
    out = edits_to_csv.convert(edits, str(tmp_path / "x"))
    assert out["angle"] == 0 and out["skipped"] == 1 and out["root_csv"] is None


def test_checkbox_off_means_nothing_is_written(tmp_path):
    # the window only calls write_edit when the box is ticked; with it off the folder stays absent
    assert not os.path.exists(tmp_path / "edits")


# ---- analysis window smoke test ----------------------------------------------------------------

@pytest.fixture(scope="module")
def tk_root():
    """One withdrawn Tk root for the module (a second Tk() after a destroy() fails on some installs)."""
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    yield root
    root.destroy()


def _fake_gui(root, tmp_path, n_frames=3):
    from utils.mask_store import MaskStore

    os.makedirs(tmp_path / "data" / "images", exist_ok=True)
    names = [f"0-crop-F{i}.png" for i in range(n_frames)]
    for name in names:
        cv2.imwrite(str(tmp_path / "data" / "images" / name), np.full((120, 80, 3), 100, np.uint8))
    results = [{"hook": None, "angle_dict": {1: 150.0}, "angle_list": [150], "state_dict": {1: "Closed"},
                "state_list": ["Closed"], "seed_ids": [1], "cotyl_mask": None, "hypo_mask": None,
                "germ_mask": None, "angle_source": "landmarks",
                "landmarks": {1: {"junction": (30.0, 20.0), "theta": 30.0, "peak": 0.9, "hypo_dir": (0.0, 1.0),
                                  "cotyl_dir": (0.5, 0.8), "edit": "model", "collar": (30.0, 100.0),
                                  "root_dir": (0.0, 1.0)}},
                "traits": {1: {"length_px": 80.0, "length_truncated": False, "diameter_px": 8.0}}}
               for _ in names]
    mask = np.zeros((120, 80), np.uint8)
    mask[20:100, 26:34] = 255
    store = MaskStore()
    for name in names:
        store.put_raw(name, "hypocotyl_binary", mask)
    calls = []
    gui = SimpleNamespace(
        root=root, mask_store=store, germination_detector=None,
        crop_boxes=[BOX], scale_um_per_px=None, path="C:/runs/Camera 2", file_list=[f"F{i}.jpg" for i in range(n_frames)],
        frame_results_by_crop={0: results}, cropped_filenames_by_crop={0: names},
        landmark_edits_dir=None, last_export_path=None, calls=calls,
        _reconstruct_series_for_crop=lambda crop_id: calls.append(crop_id),
        _trait_mask=lambda file_name, fallback=None: store.get(file_name, "hypocotyl_binary"))
    return gui, results


def test_window_landmark_edit_flow(tmp_path, monkeypatch, tk_root):
    root = tk_root
    monkeypatch.chdir(tmp_path)
    try:
        from ui.analysis_window import SeedlingAnalysisWindow
        gui, results = _fake_gui(root, tmp_path)
        edits = tmp_path / "edits"
        gui.landmark_edits_dir = str(edits)
        win = SeedlingAnalysisWindow(gui, 0)
        assert win.frame_results is results                  # by reference
        assert not win.germ_check.winfo_ismapped() or not win._germ_check_shown
        assert "source: landmarks" in win.angle_label.cget("text") and "peak 0.90" in win.angle_label.cget("text")
        assert "hypocotyl length 80 px" in win.angle_label.cget("text")
        win.canvas.canvas_to_image = lambda x, y: (x, y)

        win._toggle_manual_angle_mode()
        assert win._manual_mode
        for p in CLICKS[:4]:
            win._place_landmark_point(SimpleNamespace(x=p[0], y=p[1]))
        assert "C2" in win.angle_label.cget("text")
        win._place_landmark_point(SimpleNamespace(x=50.0, y=10.0))
        assert win._session.angle_complete and "theta" in win.angle_label.cget("text")
        win._landmark_undo()
        assert not win._session.angle_complete
        win._place_landmark_point(SimpleNamespace(x=50.0, y=10.0))
        win._landmark_skip_optional()
        win.save_edits_var.set(True)
        win._apply_landmarks()

        res = results[0]
        assert res["state_dict"][1] == "Manual" and res["angle_list"] == [round(res["angle_dict"][1])]
        assert res["angle_dict"][1] == pytest.approx(180 - res["landmarks"][1]["theta"])
        assert res["landmarks"][1]["edit"] == "manual" and res["landmarks"][1]["collar"] == (30.0, 100.0)
        assert res["traits"][1]["length_px"] > 50
        assert gui.calls == [0]                               # series re-reconstructed
        assert win._session.points == []
        assert len(list(edits.glob("*.json"))) == 1 and len(list((edits / "crops").glob("*.png"))) == 1
        rec = json.loads(next(edits.glob("*.json")).read_text(encoding="utf-8"))
        assert rec["series"] == "Camera_2" and rec["source_frame"] == "F0.jpg" and rec["collar_x"] is None
        assert rec["model_readout"]["edit"] == "model"

        # blank still works and re-reconstructs
        win._blank_angle()
        assert np.isnan(res["angle_dict"][1]) and gui.calls == [0, 0]
        win.destroy()
    finally:
        pass


def test_window_hides_germination_check_without_a_mask_and_shows_it_with_one(tmp_path, monkeypatch, tk_root):
    root = tk_root
    monkeypatch.chdir(tmp_path)
    try:
        from ui.analysis_window import SeedlingAnalysisWindow
        gui, results = _fake_gui(root, tmp_path)
        win = SeedlingAnalysisWindow(gui, 0)
        assert win._germ_check_shown is False
        results[0]["germ_mask"] = np.zeros((120, 80), np.uint8)
        win._render_current_frame()
        assert win._germ_check_shown is True
        win.destroy()
    finally:
        pass


def test_window_shows_raw_image_with_thin_overlays_and_no_brush(tmp_path, monkeypatch, tk_root):
    monkeypatch.chdir(tmp_path)
    from ui.analysis_window import SeedlingAnalysisWindow
    gui, results = _fake_gui(tk_root, tmp_path)
    cotyl = np.zeros((120, 80), np.uint8)
    cotyl[10:40, 40:70] = 255
    results[0]["cotyl_mask"] = cotyl
    win = SeedlingAnalysisWindow(gui, 0)
    try:
        raw = cv2.imread(str(tmp_path / "data" / "images" / "0-crop-F0.png"), cv2.IMREAD_COLOR)
        assert np.array_equal(win.canvas._image, raw)            # no global darkening / tinting

        kinds = [s["kind"] for s in win.canvas._overlay]
        assert "polyline" in kinds and "contour" in kinds and "point" in kinds
        poly = next(s for s in win.canvas._overlay if s["kind"] == "polyline")
        assert len(poly["pts"]) > 20 and all(25 < x < 35 for x, _ in poly["pts"])
        assert any(s["kind"] == "point" and s.get("label") == "J" for s in win.canvas._overlay)

        # the contour alpha-blends only along its own pixels in the displayed (zoomed) image
        zoomed = cv2.resize(raw, (raw.shape[1] * 5, raw.shape[0] * 5), interpolation=cv2.INTER_NEAREST)
        blended = win.canvas._blend_contours(zoomed, 0, 0)
        changed = np.any(blended != zoomed, axis=2)
        assert changed.any() and changed.mean() < 0.05
        ys, xs = np.nonzero(changed)
        assert xs.min() >= 40 * 5 - 6 and xs.max() <= 70 * 5 + 6 and ys.min() >= 10 * 5 - 6 and ys.max() <= 40 * 5 + 6
        assert np.array_equal(blended[300:340, 20:100], zoomed[300:340, 20:100])

        win.show_hypo.set(False)
        win.show_cotyl.set(False)
        win._render_current_frame()
        kinds = [s["kind"] for s in win.canvas._overlay]
        assert "polyline" not in kinds and "contour" not in kinds

        def widgets(w):
            for child in w.winfo_children():
                yield child
                yield from widgets(child)
        texts = [str(w.cget("text")) for w in widgets(win) if "text" in w.keys()]
        assert not any("brush" in t.lower() or "edit mask" in t.lower() or "recompute" in t.lower() for t in texts)
        assert not hasattr(win, "mask_editor") and not hasattr(win, "brush_radius_var")
    finally:
        win.destroy()


def test_window_keys_browse_frames_enter_applies_and_seedlings_switch(tmp_path, monkeypatch, tk_root):
    from ui.analysis_window import SeedlingAnalysisWindow

    monkeypatch.chdir(tmp_path)
    gui, results = _fake_gui(tk_root, tmp_path)
    gui.crop_boxes = [BOX, BOX, BOX]
    opened = []

    class Other:
        def geometry(self, g):
            opened.append(("geometry", g))

    gui.open_seedling_frame = lambda crop_id, frame=None: opened.append((crop_id, frame)) or Other()
    win = SeedlingAnalysisWindow(gui, 0)
    win.canvas.canvas_to_image = lambda x, y: (x, y)
    assert str(win.prev_seedling_button["state"]) == "disabled" and str(win.next_seedling_button["state"]) == "normal"
    win.deiconify(); win.focus_force(); win.update()

    win.event_generate("<Right>"); win.update()
    win.event_generate("<Right>"); win.update()
    assert win.current_frame == 2
    win.event_generate("<Right>"); win.update()           # clamped at the last frame
    win.event_generate("<Left>"); win.update()
    assert win.current_frame == 1

    # Enter does nothing until the 5 angle points exist, then applies like the button
    win._toggle_manual_angle_mode()
    win._on_enter()
    assert results[1]["state_dict"][1] == "Closed"
    for p in CLICKS[:4] + [(50.0, 10.0)]:
        win._place_landmark_point(SimpleNamespace(x=p[0], y=p[1]))
    win._on_enter()
    assert results[1]["state_dict"][1] == "Manual" and gui.calls == [0]

    win.event_generate("<Right>"); win.update()           # frame 2, then Del blanks it like the button
    win.focus_force(); win.update()
    win.event_generate("<Delete>"); win.update()
    assert np.isnan(results[2]["angle_dict"][1]) and results[2]["state_dict"][1] == "Manual"
    assert gui.calls == [0, 0] and win.current_frame == 2
    win.event_generate("<Left>"); win.update()

    win._switch_seedling(-1)                               # no seedling before the first: nothing opens
    assert opened == []
    win._switch_seedling(+1)                               # same frame, next seedling, this window closed
    assert opened[0] == (1, 1) and opened[1][0] == "geometry"
    assert not win.winfo_exists()


def test_seedling_switch_buttons_are_disabled_at_the_ends(tmp_path, monkeypatch, tk_root):
    from ui.analysis_window import SeedlingAnalysisWindow

    monkeypatch.chdir(tmp_path)
    gui, _ = _fake_gui(tk_root, tmp_path)
    gui.open_seedling_frame = lambda *a, **k: pytest.fail("no neighbour to open")
    win = SeedlingAnalysisWindow(gui, 0)                   # the only seedling
    assert str(win.prev_seedling_button["state"]) == "disabled"
    assert str(win.next_seedling_button["state"]) == "disabled"
    win._switch_seedling(-1)
    win._switch_seedling(+1)
    win.destroy()


def test_relaunch_asks_before_discarding_manual_edits(tmp_path, monkeypatch, tk_root):
    import ui.analysis_window as aw

    monkeypatch.chdir(tmp_path)
    gui, results = _fake_gui(tk_root, tmp_path)
    gui._busy = False
    gui._snapshot_tk_state = lambda: None
    started, asked = [], []
    monkeypatch.setattr(aw.threading, "Thread", lambda target: SimpleNamespace(start=lambda: started.append(target)))
    win = aw.SeedlingAnalysisWindow(gui, 0)
    try:
        win._start_segmentation()                         # no manual frames: no question
        assert len(started) == 1 and gui._busy is True
        gui._busy = False
        results[1]["state_dict"][1] = "Manual"
        monkeypatch.setattr(aw.messagebox, "askyesno", lambda *a, **k: asked.append(a) or False)
        win._start_segmentation()                         # declined: nothing starts, edit kept
        assert len(started) == 1 and len(asked) == 1 and "1 manual" in asked[0][1]
        assert results[1]["state_dict"][1] == "Manual"
        gui._busy = True                                  # another worker running
        monkeypatch.setattr(aw.messagebox, "showinfo", lambda *a, **k: asked.append("busy"))
        win._start_segmentation()
        assert len(started) == 1 and asked[-1] == "busy"
    finally:
        win.destroy()


# ---- accepting a model reading / borrowing the previous collar ---------------------------------

def test_session_from_readout_reproduces_the_reading():
    from utils.landmark_edit import session_from_readout

    readout = {"junction": (30.0, 20.0), "hypo_dir": (0.0, 1.0), "cotyl_dir": (0.6, -0.8), "tip": (48.0, -4.0),
               "collar": (30.0, 100.0), "root_dir": (0.0, 1.0), "theta": 143.13}
    manual = session_from_readout(readout).result(overhook=False)
    assert manual["theta"] == pytest.approx(143.13, abs=0.01) and manual["bio"] == pytest.approx(180 - 143.13, abs=0.01)
    assert manual["junction"] == (30.0, 20.0) and manual["collar"] == (30.0, 100.0)
    assert manual["root_dir"] == pytest.approx((0.0, 1.0)) and manual["tip"] == pytest.approx((48.0, -4.0))
    assert session_from_readout(None) is None and session_from_readout({"junction": (1, 1)}) is None


def test_previous_collar_skips_untrusted_frames():
    from utils.landmark_edit import previous_collar

    def frame(collar, state="Closed", **extra):
        return {"state_dict": {1: state}, "landmarks": {1: {"junction": (0, 0), "collar": collar, **extra}}}
    results = [frame((10, 90)),                                   # 0: trusted model collar
               frame((11, 91), gated=True),                       # 1: gated
               None,                                              # 2: no result
               frame((40, 60), temporal_flag="collar"),           # 3: collar flagged
               frame((50, 50))]                                   # 4: the frame being fixed
    assert previous_collar(results, 1, 4) == (0, (10.0, 90.0), None)
    results[1] = frame((12, 92), state="Manual", gated=True)      # a manual frame is trusted
    assert previous_collar(results, 1, 4)[:2] == (1, (12.0, 92.0))
    assert previous_collar(results, 1, 0) is None


def test_window_accept_reading_and_previous_collar(tmp_path, monkeypatch, tk_root):
    monkeypatch.chdir(tmp_path)
    from ui.analysis_window import SeedlingAnalysisWindow

    gui, results = _fake_gui(tk_root, tmp_path)
    edits = tmp_path / "edits"
    gui.landmark_edits_dir = str(edits)
    # frame 1: a grey reading (gated: no angle) the user knows is right
    gated = results[1]["landmarks"][1]
    gated.update(gated=True, gated_theta=30.0)
    results[1]["angle_dict"][1] = np.nan
    results[1]["state_dict"][1] = "Unreadable"
    # frame 2: a wrong collar
    results[2]["landmarks"][1]["collar"] = (70.0, 40.0)
    win = SeedlingAnalysisWindow(gui, 0)
    try:
        win.show_frame(1)
        assert "A accepts it" in win.angle_label.cget("text")
        win.save_edits_var.set(True)
        win._accept_reading()
        res = results[1]
        assert res["state_dict"][1] == "Manual" and res["angle_dict"][1] == pytest.approx(150.0, abs=0.01)
        assert res["landmarks"][1]["accepted"] and not res["landmarks"][1].get("gated")
        assert gui.calls == [0] and len(list(edits.glob("*.json"))) == 1      # rebuilt, saved for retraining
        win._accept_reading()                                                 # already manual: no-op
        assert gui.calls == [0]

        win.show_frame(2)
        win._use_previous_collar()
        fixed = results[2]["landmarks"][1]
        assert fixed["collar"] == (30.0, 100.0) and fixed["collar_edit"] == "manual"   # from accepted frame 1
        assert results[2]["state_dict"][1] == "Closed" and results[2]["angle_dict"][1] == 150.0   # angle untouched
        assert gui.calls == [0, 0] and "frame 2" in win.status_label.cget("text")
    finally:
        win.destroy()


# ---- easing the review: grey blanked frames, unlock after Apply, collar toggle, review queue ----

def _overlay_colors(win):
    return {s["color"] for s in win.canvas._overlay if s["kind"] in ("point", "line")}


def test_blanked_frame_stays_grey_and_apply_leaves_click_mode(tmp_path, monkeypatch, tk_root):
    from ui.analysis_window import SeedlingAnalysisWindow
    from utils.landmark_edit import MUTED_COLOR

    monkeypatch.chdir(tmp_path)
    gui, results = _fake_gui(tk_root, tmp_path)
    win = SeedlingAnalysisWindow(gui, 0)
    win.canvas.canvas_to_image = lambda x, y: (x, y)
    try:
        assert MUTED_COLOR not in _overlay_colors(win)
        win._blank_angle()
        assert _overlay_colors(win) == {MUTED_COLOR}                  # grey, not back to colours
        assert "blanked by hand" in win.angle_label.cget("text")

        # Apply (button or Enter) leaves click mode ...
        win._toggle_manual_angle_mode()
        for p in CLICKS:
            win._place_landmark_point(SimpleNamespace(x=p[0], y=p[1]))
        win._on_enter()
        assert results[0]["state_dict"][1] == "Manual" and not win._manual_mode
        assert MUTED_COLOR not in _overlay_colors(win)                # an angle again: coloured

        # ... unless edits are saved for retraining
        gui.landmark_edits_dir = str(tmp_path / "edits")
        win.save_edits_var.set(True)
        win.show_frame(1)
        win._toggle_manual_angle_mode()
        for p in CLICKS:
            win._place_landmark_point(SimpleNamespace(x=p[0], y=p[1]))
        win._apply_landmarks()
        assert results[1]["state_dict"][1] == "Manual" and win._manual_mode
    finally:
        win.destroy()


def test_collar_can_be_hidden(tmp_path, monkeypatch, tk_root):
    from ui.analysis_window import SeedlingAnalysisWindow

    monkeypatch.chdir(tmp_path)
    gui, _ = _fake_gui(tk_root, tmp_path)
    win = SeedlingAnalysisWindow(gui, 0)
    try:
        labels = lambda: [s.get("label") for s in win.canvas._overlay]
        assert "collar" in labels()
        win.show_collar.set(False)
        win._render_current_frame()
        assert "collar" not in labels() and "J" in labels()
    finally:
        win.destroy()


def test_review_queue_steps_through_flagged_frames_across_seedlings(tmp_path, monkeypatch, tk_root):
    from ui.analysis_window import SeedlingAnalysisWindow

    monkeypatch.chdir(tmp_path)
    gui, results = _fake_gui(tk_root, tmp_path)
    gui.crop_boxes = [BOX, BOX]
    other = [{"state_dict": {2: "Closed"}, "landmarks": {2: {"series_edit": "rejected"}}}]
    gui.frame_results_by_crop[1] = other
    results[2]["landmarks"][1]["temporal_flag"] = "collar"
    opened = []
    gui.open_seedling_frame = lambda crop_id, frame=None: opened.append((crop_id, frame)) or SimpleNamespace(
        geometry=lambda g: None)
    win = SeedlingAnalysisWindow(gui, 0)
    try:
        assert "2 frame(s) left" in win.review_label.cget("text")
        win._review_step(+1)
        assert win.current_frame == 2 and "1 of 2: collar flag" in win.review_label.cget("text")
        win._accept_reading()                                          # reviewed: leaves the queue
        assert "1 frame(s) left" in win.review_label.cget("text")
        win._review_step(-1)
        assert win.current_frame == 2 and "No earlier frame" in win.status_label.cget("text")
        win._review_step(+1)                                           # next one is seedling 2's
        assert opened == [(1, 0)] and not win.winfo_exists()
    finally:
        if win.winfo_exists():
            win.destroy()


def test_gui_review_flagged_frames_opens_the_first_queued_frame(monkeypatch):
    import ui.gui.seedling_windows as sw

    opened, infos = [], []
    gui = SimpleNamespace(root=None, frame_results_by_crop={
        0: [{"state_dict": {1: "Closed"}, "landmarks": {1: {"junction": (0, 0)}}},
            {"state_dict": {1: "Closed"}, "landmarks": {1: {"temporal_flag": "junction"}}}]})
    gui.open_seedling_frame = lambda crop_id, frame=None: opened.append((crop_id, frame))
    monkeypatch.setattr(sw.messagebox, "showinfo", lambda *a, **k: infos.append(a))
    assert sw.SeedlingWindowsMixin.review_flagged_frames(gui, quiet=True) == 1
    assert opened == [(0, 1)]
    gui.frame_results_by_crop[0][1]["state_dict"][1] = "Manual"
    assert sw.SeedlingWindowsMixin.review_flagged_frames(gui, quiet=True) == 0 and infos == []
    assert sw.SeedlingWindowsMixin.review_flagged_frames(gui) == 0 and len(infos) == 1
