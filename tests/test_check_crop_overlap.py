import json

from multi import check_crop_overlap as co


def box(cx, cy, hw=10, hh=20):
    return {"cx": cx, "cy": cy, "half_w": hw, "half_h": hh}


def test_overlapping_and_touching_boxes():
    old = {"S": [box(100, 100)]}
    new = {"S": [box(105, 100), box(120, 100), box(300, 300)]}       # overlaps, touches (x 110..130 vs 90..110), far
    hits = co.find_cross_overlaps(old, new)
    assert [(s, i, j) for s, i, j, _ in hits] == [("S", 0, 0)]       # touching is not overlapping
    assert 0.7 < hits[0][3] < 0.8


def test_contained_box_is_a_full_overlap_and_other_series_are_ignored():
    old = {"A": [box(100, 100, 30, 30)], "B": [box(500, 500)]}
    new = {"A": [box(100, 100, 5, 5)], "B": [box(100, 100)], "C": [box(100, 100)]}
    hits = co.find_cross_overlaps(old, new)
    assert hits == [("A", 0, 0, 1.0)]


def test_cli_reports_and_sets_the_exit_status(tmp_path, capsys):
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    old.write_text(json.dumps({"S": [box(100, 100)]}))
    new.write_text(json.dumps({"S": [box(500, 500)], "T": [box(1, 1)]}))
    assert co.main(["--old", str(old), "--new", str(new)]) == 0
    assert "no new box intercepts" in capsys.readouterr().out
    new.write_text(json.dumps({"S": [box(100, 100)]}))
    assert co.main(["--old", str(old), "--new", str(new)]) == 1
    assert "new box 0 (seedling 1) x old box 0 (seedling 1)" in capsys.readouterr().out


def test_classification_defaults_and_filtered_output():
    old = {"S": [box(100, 100)]}
    new = {"S": [box(100, 100),          # duplicate (full overlap)
                 box(119, 100),          # partial: 2 px sliver shared
                 box(300, 300)]}         # free
    c = co.classify_new_boxes(old, new)["S"]
    assert [e["status"] for e in c] == ["duplicate", "partial", "free"]
    assert c[1]["partners"] == [0] and 0 < c[1]["fraction"] < 0.5
    assert [co.default_decision(e["status"]) for e in c] == ["drop", None, "keep"]

    kept = co.build_filtered({"S": c}, {})["S"]                              # undecided partial is not kept
    assert kept == [box(300, 300)]
    decisions = {"S": {co.box_key(box(119, 100)): "keep", co.box_key(box(300, 300)): "drop"}}
    assert co.build_filtered({"S": c}, decisions)["S"] == [box(119, 100)]


def test_new_boxes_that_overlap_each_other_are_reported_unless_dropped():
    from multi.review_crop_overlap import new_box_clashes
    entries = [{"index": 0, "box": box(100, 100)}, {"index": 1, "box": box(105, 100)},
               {"index": 2, "box": box(300, 300)}]
    clashes = new_box_clashes(entries, lambda e: "keep")
    assert set(clashes) == {0, 1} and clashes[0][0][0] == 1 and 0.7 < clashes[0][0][1] < 0.8
    assert new_box_clashes(entries, lambda e: "drop" if e["index"] == 1 else "keep") == {}


def test_review_app_draws_saves_and_deletes_a_new_seedling(tmp_path):
    import os
    import numpy as np
    import cv2
    import pytest
    tk = pytest.importorskip("tkinter")
    from multi.review_crop_overlap import ReviewApp
    plate = tmp_path / "plates" / "S"
    plate.mkdir(parents=True)
    cv2.imwrite(str(plate / "f1.png"), np.zeros((400, 400, 3), np.uint8))
    series = [("S", str(plate), "f1.png")]
    out = str(tmp_path / "filtered.json")
    try:
        app = ReviewApp({"S": [box(100, 100)]}, {"S": [box(300, 300)]}, series, out, 0.5)
    except tk.TclError:
        pytest.skip("no display")
    try:
        assert json.load(open(out)) == {"S": [box(300, 300)]}                  # the free file box is kept
        app._on_draw(box(105, 100))                                           # overlaps the old box
        assert app.entries[-1]["drawn"] and app.entries[-1]["status"] == "duplicate"
        app._on_draw(box(300, 300, 8, 8))                                     # inside the other new box
        assert app.clashes and json.load(open(app.added_path))["S"] == [box(105, 100), box(300, 300, 8, 8)]
        assert json.load(open(out))["S"] == [box(300, 300), box(300, 300, 8, 8)]   # duplicate dropped, free drawn kept
        app._delete_drawn()
        assert json.load(open(app.added_path))["S"] == [box(105, 100)]
        app._pick(0)
        app._delete_drawn()                                                    # a file box cannot be deleted
        assert len(app.new["S"]) == 1 and len(app.entries) == 2
    finally:
        app.destroy()
