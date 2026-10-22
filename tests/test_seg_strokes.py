"""Partial segmentation strokes: the stroke map, its RootPainter encoding (read back by the existing
merge code), the canvas "mask" overlay and the annotator window."""
import csv
import os
import types

import cv2
import numpy as np
import pytest

from utils.seg_strokes import (BACKGROUND, CLASS_VALUES, UNLABELLED, StrokeMap, annotations_dirs,
                               decode_files, encode_class, find_split, load_strokes, save_strokes)

COT, HYP = CLASS_VALUES["cotyledon"], CLASS_VALUES["hypocotyl"]


def _painted():
    s = StrokeMap((40, 30))
    s.begin_stroke()
    s.paint((5, 5), (5, 15), 2, COT)
    s.begin_stroke()
    s.paint((20, 10), (20, 35), 2, HYP)
    s.begin_stroke()
    s.paint((10, 30), (12, 30), 1, BACKGROUND)
    return s


def test_paint_undo_and_counts():
    s = _painted()
    assert set(s.counts()) == {"cotyledon", "hypocotyl", "background"}
    assert s.labels[10, 5] == COT and s.labels[20, 20] == HYP and s.labels[0, 29] == UNLABELLED
    s.undo()
    assert "background" not in s.counts()
    s.begin_stroke()
    s.paint((20, 20), (20, 20), 3, UNLABELLED)          # erase
    assert s.labels[20, 20] == UNLABELLED
    while s.undo():
        pass
    assert s.empty


def test_encoding_round_trips_and_matches_the_merge_decoder():
    from multi.src.mask_merge import decode_rootpainter_annotation

    s = _painted()
    files = {name: encode_class(s.labels, name) for name in CLASS_VALUES}
    fg, bg, defined = decode_rootpainter_annotation(files["hypocotyl"])
    assert np.array_equal(fg, s.labels == HYP) and np.array_equal(bg, s.labels == BACKGROUND)
    assert not (fg & (s.labels == COT)).any()                  # other classes are NOT background
    assert np.array_equal(decode_files(files, s.labels.shape), s.labels)


def test_merge_burns_all_classes_over_the_pseudo_label(tmp_path):
    """The real merge override (class files applied one after the other) keeps every class."""
    from multi.src.mask_merge import _apply_annotation_overrides, _build_annotation_index

    s = _painted()
    save_strokes(str(tmp_path), "0-crop-A_1.png", s, "train")
    label = np.full(s.labels.shape, 7, np.uint8)               # stand-in pseudo-label
    index = _build_annotation_index(annotations_dirs(str(tmp_path)))
    assert _apply_annotation_overrides(label, "0-crop-A_1.png", index)
    painted = s.labels != UNLABELLED
    assert np.array_equal(label[painted], s.labels[painted])    # 0 background, 1 cotyledon, 2 hypocotyl
    assert (label[~painted] == 7).all()                         # unpainted: pseudo-label survives


def test_save_load_splits_and_delete(tmp_path):
    out = str(tmp_path)
    s = _painted()
    save_strokes(out, "c.png", s, "val")
    assert find_split(out, "c.png") == "val"
    back, split = load_strokes(out, "c.png", s.labels.shape)
    assert split == "val" and np.array_equal(back.labels, s.labels)
    save_strokes(out, "c.png", s, "train")                      # moved, not duplicated
    assert find_split(out, "c.png") == "train" and not os.path.exists(os.path.join(out, "hypocotyl", "val", "c.png"))
    save_strokes(out, "c.png", StrokeMap(s.labels.shape), "train")   # empty: removed
    assert find_split(out, "c.png") is None
    with pytest.raises(ValueError):
        save_strokes(out, "c.png", s, "test")


def test_canvas_mask_overlay_blends_only_the_mask():
    from ui.zoomable_canvas import ZoomableImageCanvas

    mask = np.zeros((10, 10), bool)
    mask[2:4, 2:4] = True
    fake = types.SimpleNamespace(zoom=2.0, _overlay=[{"kind": "mask", "mask": mask, "color": "#ff0000", "alpha": 1.0}])
    out = ZoomableImageCanvas._blend_masks(fake, np.zeros((20, 20, 3), np.uint8), 0, 0, 10, 10)
    assert tuple(out[5, 5]) == (255, 0, 0) and tuple(out[0, 0]) == (0, 0, 0)
    assert out[..., 0].astype(bool).sum() == 16                 # 2x2 px at zoom 2 -> 4x4, nearest


# ---- the annotator window ------------------------------------------------------------------------

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


def _crops(folder, n=3):
    os.makedirs(folder, exist_ok=True)
    names = [f"{i}-crop-S_{i:03d}.png" for i in range(n)]
    for name in names:
        cv2.imwrite(os.path.join(folder, name), np.full((60, 40, 3), 90, np.uint8))
    return names


def test_queue_csv_keeps_this_folders_rows(tmp_path):
    from ui.seg_annotator import read_queue_csv

    folder = tmp_path / "cropped_x"
    names = _crops(str(folder))
    q = tmp_path / "q.csv"
    with open(q, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["folder", "crop_file", "reason"])
        w.writeheader()
        w.writerow({"folder": "data/crops/cropped_x", "crop_file": names[2], "reason": "junction gap 12 px"})
        w.writerow({"folder": "data/crops/other", "crop_file": names[0], "reason": "x"})
        w.writerow({"folder": "data/crops/cropped_x", "crop_file": "missing.png", "reason": "x"})
        w.writerow({"folder": "data/crops/cropped_x", "crop_file": names[1], "reason": ""})
    assert read_queue_csv(str(q), str(folder)) == [(names[2], "junction gap 12 px"), (names[1], "")]


def test_annotator_paint_save_resume(tmp_path, tk_root):
    from ui.seg_annotator import SegAnnotator

    folder, out = str(tmp_path / "crops"), str(tmp_path / "strokes")
    names = _crops(folder)
    items = [(n, "reason") for n in names]
    pred = np.zeros((60, 40), np.uint8)
    pred[10:50, 18:22] = 2
    app = SegAnnotator(tk_root, folder, items, out, split="train", predict=lambda image: pred)
    try:
        app.canvas.canvas_to_image = lambda x, y: (x, y)
        assert any(s["kind"] == "contour" for s in app.canvas._overlay)        # prediction drawn
        app.tool_var.set("2")
        app._press(types.SimpleNamespace(x=20, y=5))
        app._drag(types.SimpleNamespace(x=20, y=12))
        assert app.dirty and (app.strokes.labels == HYP).any()
        assert any(s["kind"] == "mask" for s in app.canvas._overlay)           # strokes drawn
        app.next()                                                             # saves, moves on
        assert find_split(out, names[0]) == "train" and app.index == 1
        app.tool_var.set("b")
        app._press(types.SimpleNamespace(x=5, y=5))
        app.skip()                                                             # discarded
        assert find_split(out, names[1]) is None and app.index == 2
    finally:
        app.destroy()
    again = SegAnnotator(tk_root, folder, items, out, split="val", predict=None)
    try:
        assert again.index == 2                                                # resumes after visited crops
        again.previous()
        again.previous()
        assert again.index == 0 and again.stroke_split == "train"              # keeps its saved split
        assert (again.strokes.labels == HYP).any()
    finally:
        again.destroy()


def test_annotator_markers_move_undo_save(tmp_path, tk_root):
    from ui.seg_annotator import SegAnnotator
    from utils.landmark_marks import FolderMarks, MarksStore

    folder, out = str(tmp_path / "crops"), str(tmp_path / "strokes")
    names = _crops(folder, 2)
    marks_csv = str(tmp_path / "marks.csv")
    clicked = {names[0]: {"junction": (20.0, 10.0), "collar": None, "root_dir": (0.0, 1.0)}}
    model = lambda path: {"junction": (1.0, 1.0), "collar": (20.0, 50.0), "root_dir": (0.0, 1.0)}
    marks = FolderMarks(folder, path=marks_csv, clicked=clicked, model=model)
    app = SegAnnotator(tk_root, folder, [(n, "") for n in names], out, marks=marks)
    try:
        app.canvas.canvas_to_image = lambda x, y: (x, y)
        labels = [s["label"] for s in app.canvas._overlay if s["kind"] == "point"]
        assert labels == ["J (clicked)", "collar (model)"]
        app.arm("collar")
        app._press(types.SimpleNamespace(x=22, y=45))                          # moves the collar, paints nothing
        assert app.strokes.empty and app.current_marks()["collar"] == (22.0, 45.0)
        app.arm("junction")
        app._press(types.SimpleNamespace(x=21, y=12))
        app.undo()                                                             # undoes the junction move only
        m = app.current_marks()
        assert m["junction"] == (20.0, 10.0) and m["junction_source"] == "clicked"
        assert m["collar_source"] == "corrected"
        app.next()
        row = MarksStore(marks_csv).rows[names[0]]
        assert row["collar_source"] == "corrected" and row["junction_source"] == ""
        assert row["root_dir_y"] == "1.0000"
        app.arm("junction")
        app._press(types.SimpleNamespace(x=5, y=5))
        app.skip()                                                             # unsaved move discarded
        assert names[1] not in MarksStore(marks_csv).rows
    finally:
        app.destroy()
