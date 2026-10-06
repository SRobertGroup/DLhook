"""Tests for utils/angle_annotation.py: the five-click hook-angle maths, frame
sampling and the ground-truth CSV store."""
from __future__ import annotations

import csv
import math

import cv2
import numpy as np
import pytest

from utils.angle_annotation import (
    AngleClickSession, AnnotationStore, CSV_FIELDS, WorkItem, build_queue,
    default_manifest, discover_crops, filter_crops, hook_angle, load_split_filenames,
    orient_axis, parse_crop_filename,
)

BACKSLASH = chr(92)  # manifests written on Windows use backslash paths
J = (100.0, 100.0)  # junction


def _polar(angle_deg, r):
    """Image-space point at `r` from the junction, angle measured from +x."""
    a = math.radians(angle_deg)
    return (J[0] + r * math.cos(a), J[1] + r * math.sin(a))


# --- geometry ---------------------------------------------------------------

def test_orient_axis_points_away_from_junction_whatever_the_click_order():
    p, q = (100.0, 150.0), (100.0, 200.0)      # both below the junction
    assert np.allclose(orient_axis(p, q, J), (0, 1))
    assert np.allclose(orient_axis(q, p, J), (0, 1))


def test_closed_hook_is_180_and_opening_decreases():
    hypo = (_polar(90, 30), _polar(90, 60))                 # down the hypocotyl
    closed = hook_angle(J, hypo, (_polar(90, 20), _polar(90, 50)))
    assert closed == pytest.approx((0.0, 180.0))
    quarter = hook_angle(J, hypo, (_polar(0, 20), _polar(0, 50)))
    assert quarter == pytest.approx((90.0, 90.0))
    straight = hook_angle(J, hypo, (_polar(-90, 20), _polar(-90, 50)))
    assert straight == pytest.approx((180.0, 0.0))


def test_overhook_reads_above_180_like_the_in_app_manual_angle():
    hypo = (_polar(90, 30), _polar(90, 60))
    cotyl = (_polar(60, 20), _polar(60, 50))                # 30 deg off the hypocotyl
    theta, bio = hook_angle(J, hypo, cotyl, overhook=True)
    assert theta == pytest.approx(30.0)
    assert bio == pytest.approx(210.0)
    assert hook_angle(J, hypo, cotyl)[1] == pytest.approx(150.0)


def test_degenerate_axes_are_rejected():
    with pytest.raises(ValueError):
        orient_axis((10, 10), (10, 10), J)
    with pytest.raises(ValueError):                        # points straddle the junction symmetrically
        orient_axis((100, 90), (100, 110), J)


# --- click session ----------------------------------------------------------

def test_session_walks_five_steps_and_undoes():
    s = AngleClickSession()
    assert s.step == "junction" and not s.complete
    for i, name in enumerate(["junction", "hypo_1", "hypo_2", "cotyl_1", "cotyl_2"]):
        assert s.step == name
        s.add(i, i)
    assert s.complete and s.step is None
    s.add(99, 99)                                           # extra clicks ignored
    assert len(s.points) == 5
    s.undo()
    assert s.step == "cotyl_2"
    s.reset()
    assert s.points == []
    with pytest.raises(ValueError):
        s.result()


def test_session_result_matches_hook_angle():
    s = AngleClickSession([J, _polar(90, 30), _polar(90, 60), _polar(0, 20), _polar(0, 50)])
    assert s.result() == pytest.approx((90.0, 90.0))


# --- frame discovery and sampling -------------------------------------------

def test_parse_crop_filename_handles_multi_digit_ids():
    assert parse_crop_filename("12-crop-IMG_001.png") == (12, "IMG_001")
    assert parse_crop_filename("3-crop-a-b.png") == (3, "a-b")
    assert parse_crop_filename("notes.txt") is None
    assert parse_crop_filename("crop-1.png") is None


def _make_folder(tmp_path, ids=(0, 1, 10), frames=12):
    for cid in ids:
        for f in range(frames):
            cv2.imwrite(str(tmp_path / f"{cid}-crop-IMG_{f:03d}.png"), np.zeros((20, 10, 3), np.uint8))
    (tmp_path / "readme.txt").write_text("x")
    return tmp_path


def test_discover_crops_groups_and_naturally_sorts(tmp_path):
    crops = discover_crops(str(_make_folder(tmp_path, frames=11)))
    assert list(crops) == [("", 0), ("", 1), ("", 10)]
    frames = [f for f, _, _ in crops[("", 0)]]
    assert frames == [f"IMG_{i:03d}" for i in range(11)]


def test_build_queue_is_reproducible_and_grouped(tmp_path):
    crops = discover_crops(str(_make_folder(tmp_path)))
    a = build_queue(crops, 5, seed=3)
    b = build_queue(crops, 5, seed=3)
    assert a == b
    assert build_queue(crops, 5, seed=4) != a
    assert len(a) == 15 and all(isinstance(i, WorkItem) for i in a)
    assert [i.crop_id for i in a] == [0] * 5 + [1] * 5 + [10] * 5       # grouped, ids ascending
    assert len({(i.crop_id, i.frame) for i in a}) == 15                  # no repeats


def test_build_queue_all_frames_when_n_is_large_or_none(tmp_path):
    crops = discover_crops(str(_make_folder(tmp_path, ids=(0,), frames=7)))
    for n in (None, 7, 100):
        q = build_queue(crops, n, seed=0)
        assert sorted(i.frame for i in q) == [f"IMG_{k:03d}" for k in range(7)]


def test_seedling_id_is_one_based():
    assert WorkItem(0, "f", "0-crop-f.png").seedling_id == 1


# --- store ------------------------------------------------------------------

def test_store_roundtrip_and_resume(tmp_path):
    out = tmp_path / "gt.csv"
    item = WorkItem(2, "IMG_005", "2-crop-IMG_005.png")
    other = WorkItem(2, "IMG_006", "2-crop-IMG_006.png")
    session = AngleClickSession([J, _polar(90, 30), _polar(90, 60), _polar(0, 20), _polar(0, 50)])

    store = AnnotationStore(str(out))
    theta, bio = store.save_measured(item, session, overhook=False)
    store.save_skipped(other)
    assert (theta, bio) == pytest.approx((90.0, 90.0))

    reloaded = AnnotationStore(str(out))                    # a fresh process
    row = reloaded.get(item)
    assert row["status"] == "measured" and float(row["bio_angle"]) == pytest.approx(90.0)
    assert row["seedling_id"] == "3" and row["crop_id"] == "2" and row["overhook"] == "0"
    assert reloaded.points_for(item) == pytest.approx(session.points, abs=0.01)
    assert reloaded.get(other)["status"] == "skipped" and reloaded.points_for(other) is None
    assert reloaded.counts([item, other, WorkItem(2, "x", "x")]) == (1, 1)

    with open(out, newline="") as fh:
        assert csv.DictReader(fh).fieldnames == CSV_FIELDS


def test_store_overwrites_an_existing_frame(tmp_path):
    store = AnnotationStore(str(tmp_path / "gt.csv"))
    item = WorkItem(0, "f", "0-crop-f.png")
    store.save_skipped(item)
    store.save_measured(item, AngleClickSession([J, _polar(90, 30), _polar(90, 60), _polar(90, 20), _polar(90, 50)]), True)
    assert len(AnnotationStore(str(tmp_path / "gt.csv")).rows) == 1
    assert AnnotationStore(str(tmp_path / "gt.csv")).get(item)["bio_angle"] == "180.000"


# --- manifest (multi-series crop sets) ----------------------------------------

def _make_manifest_folder(tmp_path):
    """Two series that BOTH have a crop_id 0, like cropped_training_set/."""
    rows = []
    for series, frames in (("0mM", ("A_000", "A_015", "A_030")), ("75mM", ("B_000", "B_015"))):
        for f in frames:
            name = f"0-crop-{series}_{f}.png"
            cv2.imwrite(str(tmp_path / name), np.zeros((20, 10, 3), np.uint8))
            rows.append({"output_path": "cropped_training_set" + BACKSLASH + name, "series": series,
                         "source_frame": f"{f}.tif", "crop_id": 0})
    rows.append({"output_path": "cropped_training_set" + BACKSLASH + "0-crop-gone.png", "series": "0mM",
                 "source_frame": "gone.tif", "crop_id": 0})            # file missing on disk
    with open(tmp_path / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["output_path", "series", "source_frame", "crop_id"])
        w.writeheader()
        w.writerows(rows)
    return tmp_path


def test_manifest_keeps_same_crop_id_in_different_series_apart(tmp_path):
    folder = _make_manifest_folder(tmp_path)
    manifest = default_manifest(str(folder))
    assert manifest is not None
    crops = discover_crops(str(folder), manifest)
    assert list(crops) == [("0mM", 0), ("75mM", 0)]
    assert [f for f, _, _ in crops[("0mM", 0)]] == ["0mM_A_000", "0mM_A_015", "0mM_A_030"]
    assert crops[("75mM", 0)][0][2] == "B_000.tif"                      # img_name = source frame


def test_default_manifest_ignores_other_csvs(tmp_path):
    (tmp_path / "manifest.csv").write_text("foo,bar\n1,2\n")
    assert default_manifest(str(tmp_path)) is None
    assert default_manifest(str(tmp_path / "nowhere")) is None


def test_store_does_not_mix_series(tmp_path):
    folder = _make_manifest_folder(tmp_path)
    queue = build_queue(discover_crops(str(folder), default_manifest(str(folder))), None, seed=0)
    first, other = [i for i in queue if i.series == "0mM"][0], [i for i in queue if i.series == "75mM"][0]
    store = AnnotationStore(str(tmp_path / "gt.csv"))
    store.save_skipped(first)
    assert store.get(first) and store.get(other) is None
    row = AnnotationStore(str(tmp_path / "gt.csv")).get(first)
    assert row["series"] == "0mM" and row["img_name"].endswith(".tif")
    assert first.seedling_label == "Seedling 1  [0mM]"


def test_split_filter_keeps_only_listed_files(tmp_path):
    folder = _make_manifest_folder(tmp_path)
    index = tmp_path / "idx"
    index.mkdir()
    (index / "val_patches.csv").write_text("filename,x,y\n0-crop-75mM_B_000.png,0,0\n0-crop-75mM_B_000.png,0,200\n")
    crops = filter_crops(discover_crops(str(folder), default_manifest(str(folder))),
                         load_split_filenames(str(index), "val"))
    assert list(crops) == [("75mM", 0)]
    assert [f[1] for f in crops[("75mM", 0)]] == ["0-crop-75mM_B_000.png"]


def test_old_csv_without_series_column_still_loads(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text("seedling_id,crop_id,frame,crop_file,status\n3,2,IMG_1,2-crop-IMG_1.png,skipped\n")
    store = AnnotationStore(str(path))
    assert store.get(WorkItem(2, "IMG_1", "2-crop-IMG_1.png"))["status"] == "skipped"


# --- recording skipped frames -----------------------------------------------

def test_skip_unrecorded_marks_only_missing_frames(tmp_path):
    crops = discover_crops(str(_make_folder(tmp_path, ids=(0,), frames=4)))
    queue = build_queue(crops, None, seed=0)
    store = AnnotationStore(str(tmp_path / "gt.csv"))
    store.save_measured(queue[0], AngleClickSession([J, _polar(90, 30), _polar(90, 60), _polar(0, 20), _polar(0, 50)]), False)

    assert store.skip_unrecorded(queue) == 3
    assert store.skip_unrecorded(queue) == 0                          # idempotent
    assert store.counts(queue) == (1, 3)
    assert AnnotationStore(str(tmp_path / "gt.csv")).get(queue[0])["status"] == "measured"   # not overwritten


def _make_app(tmp_path):
    tk = pytest.importorskip("tkinter")
    from ui.angle_annotator import AngleAnnotatorApp
    folder = _make_folder(tmp_path, ids=(0,), frames=3)
    try:
        return AngleAnnotatorApp(str(folder), str(tmp_path / "gt.csv"), per_seedling=None, seed=0)
    except tk.TclError:
        pytest.skip("no display available for Tk")


def test_next_records_an_unmeasured_frame_as_skipped(tmp_path):
    app = _make_app(tmp_path)
    try:
        first = app.item
        app._move(1)
        assert app.store.get(first)["status"] == "skipped" and app.index == 1
        second = app.item
        app._move(-1)                                                  # going back records nothing
        assert app.index == 0 and app.store.get(second) is None
    finally:
        app.destroy()


def test_next_refuses_to_discard_five_unsaved_points(tmp_path):
    app = _make_app(tmp_path)
    try:
        for p in [J, _polar(90, 30), _polar(90, 60), _polar(0, 20), _polar(0, 50)]:
            app.session.add(*p)
        app._move(1)
        assert app.index == 0 and app.store.get(app.item) is None
    finally:
        app.destroy()
