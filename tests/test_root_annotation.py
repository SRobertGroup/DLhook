import csv
import math

import pytest

from utils.angle_annotation import CSV_FIELDS as ANGLE_FIELDS, WorkItem
from utils.root_annotation import (
    RootClickSession, RootStore, angle_points, measured_work_items, root_direction,
)


def item(frame="f1", crop_id=0, series="S"):
    return WorkItem(crop_id, frame, f"{crop_id}-crop-{frame}.png", series, f"{frame}.tif")


def write_angles(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ANGLE_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({**{f: "" for f in ANGLE_FIELDS}, **r})


def test_root_direction_is_a_unit_vector_from_the_collar():
    assert root_direction((10, 10), (10, 30)) == (0.0, 1.0)
    ux, uy = root_direction((0, 0), (3, 4))
    assert math.isclose(ux, 0.6) and math.isclose(uy, 0.8)
    with pytest.raises(ValueError):
        root_direction((5, 5), (5.5, 5.5))


def test_session_takes_two_clicks_then_stops():
    s = RootClickSession()
    assert not s.complete and "COLLAR" in s.prompt()
    s.add(1, 2)
    assert "ROOT" in s.prompt()
    s.add(1, 20)
    s.add(9, 9)                                    # a third click is ignored
    assert s.complete and len(s.points) == 2 and s.direction() == (0.0, 1.0)
    s.undo()
    assert not s.complete
    s.reset()
    assert s.points == []


def test_work_items_are_the_measured_frames_in_time_order(tmp_path):
    p = tmp_path / "angles.csv"
    write_angles(p, [
        {"series": "S", "crop_id": 0, "frame": "S_10", "crop_file": "0-crop-S_10.png", "status": "measured"},
        {"series": "S", "crop_id": 0, "frame": "S_9", "crop_file": "0-crop-S_9.png", "status": "measured"},
        {"series": "S", "crop_id": 0, "frame": "S_8", "crop_file": "0-crop-S_8.png", "status": "skipped"},
    ])
    assert [i.frame for i in measured_work_items(p)] == ["S_9", "S_10"]      # natural order, skipped dropped


def test_angle_points_are_read_for_context(tmp_path):
    p = tmp_path / "angles.csv"
    write_angles(p, [{"series": "S", "crop_id": 1, "frame": "f", "status": "measured", "junction_x": "5", "junction_y": "6",
                      "hypo1_x": "1", "hypo1_y": "2", "hypo2_x": "3", "hypo2_y": "4", "cotyl1_x": "7", "cotyl1_y": "8",
                      "cotyl2_x": "9", "cotyl2_y": "10"}])
    pts = angle_points(p)[("S", 1, "f")]
    assert pts["junction"] == (5.0, 6.0) and pts["cotyl2"] == (9.0, 10.0)


def test_store_roundtrip_resume_and_counts(tmp_path):
    path = tmp_path / "root.csv"
    store = RootStore(path)
    a, b, c = item("f1"), item("f2"), item("f3")
    session = RootClickSession([(10, 10), (10, 40)])
    store.save_radicle(a, session)
    store.save_no_radicle(b)
    store.save_skipped(c)
    assert store.counts([a, b, c]) == (1, 1, 1)

    again = RootStore(path)                         # a fresh process resumes from the file
    assert again.points_for(a) == [(10.0, 10.0), (10.0, 40.0)]
    assert again.points_for(b) is None and again.get(c)["status"] == "skipped"
    row = again.get(a)
    assert row["radicle_visible"] == "1" and row["root_dir_y"] == "1.0000"


def test_saving_a_radicle_needs_both_points(tmp_path):
    store = RootStore(tmp_path / "root.csv")
    with pytest.raises(ValueError):
        store.save_radicle(item(), RootClickSession([(1, 1)]))
    assert not (tmp_path / "root.csv").exists()
