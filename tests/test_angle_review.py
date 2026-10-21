"""The angle review sidecar and its queues (utils/angle_review.py)."""
import csv

from utils.angle_annotation import CSV_FIELDS, AnnotationStore, AngleClickSession, WorkItem
from utils.angle_review import (
    ReviewStore, disagreement_queue, load_errors, onsets, read_rows, review_path, rotation_keys,
    skipped_after_onset_queue,
)


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({**{f: "" for f in CSV_FIELDS}, **r})


def measured(crop_id, frame, theta, series="S", **kw):
    return {"series": series, "crop_id": crop_id, "frame": frame, "crop_file": f"{crop_id}-crop-{frame}.png",
            "status": "measured", "theta": theta, "bio_angle": 180 - theta, "junction_x": 10, "junction_y": 10,
            "hypo1_x": 10, "hypo1_y": 30, "hypo2_x": 10, "hypo2_y": 50, "cotyl1_x": 12, "cotyl1_y": 10,
            "cotyl2_x": 25, "cotyl2_y": 10, **kw}


def skipped(crop_id, frame, series="S"):
    return {"series": series, "crop_id": crop_id, "frame": frame, "crop_file": f"{crop_id}-crop-{frame}.png",
            "status": "skipped"}


def item(crop_id, frame, series="S"):
    return WorkItem(crop_id, frame, f"{crop_id}-crop-{frame}.png", series)


def test_the_sidecar_corrects_marks_and_never_touches_the_original(tmp_path):
    p = tmp_path / "gt.csv"
    write(p, [measured(0, "f1", 90), measured(0, "f2", 40), measured(0, "f3", 20), measured(0, "f4", 10),
              skipped(0, "f5")])
    before = p.read_bytes()
    store = ReviewStore(str(p))
    session = AngleClickSession([(10, 10), (10, 30), (10, 50), (10, 12), (10, 30)])      # a straight-down cotyledon
    store.save_measured(item(0, "f1"), session, False)
    store.save_confirmed(item(0, "f2"))
    store.save_rotation(item(0, "f3"))
    store.save_unmeasurable(item(0, "f4"))
    assert p.read_bytes() == before and review_path(str(p)) == str(p) + ".review.csv"
    status = {r["frame"]: r["status"] for r in read_rows(str(p))}
    assert status == {"f1": "measured", "f2": "measured", "f3": "rotation", "f4": "unmeasurable", "f5": "skipped"}
    theta = {r["frame"]: float(r["theta"]) for r in read_rows(str(p)) if r["status"] == "measured"}
    assert theta["f1"] == 0.0 and theta["f2"] == 40.0                    # corrected; kept as it was
    assert read_rows(str(p), apply_review=False)[2]["status"] == "measured"
    assert rotation_keys(str(p)) == {("S", 0, "f3")}


def test_the_store_opens_a_frame_with_the_original_clicks_until_it_is_corrected(tmp_path):
    p = tmp_path / "gt.csv"
    write(p, [measured(0, "f1", 90)])
    store = ReviewStore(str(p))
    assert store.points_for(item(0, "f1")) == AnnotationStore(str(p)).points_for(item(0, "f1"))
    assert store.get(item(0, "f1")) is None                               # not reviewed yet
    store.save_rotation(item(0, "f1"))
    assert store.points_for(item(0, "f1")) == AnnotationStore(str(p)).points_for(item(0, "f1"))
    assert ReviewStore(str(p)).counts_by_status([item(0, "f1")])["rotation"] == 1     # resumable


def test_disagreement_queue_puts_the_worst_frames_first_and_open_hooks_first_without_errors(tmp_path):
    p = tmp_path / "gt.csv"
    write(p, [measured(0, "a", 10), measured(0, "b", 160), measured(1, "c", 90)])
    for name in ("0-crop-a.png", "0-crop-b.png", "1-crop-c.png"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "e.csv").write_text("crop_file,err\n0-crop-a.png,5\n1-crop-c.png,80\n", encoding="utf-8")
    errors = load_errors(str(tmp_path / "e.csv"))
    assert [i.frame for i in disagreement_queue(str(p), str(tmp_path), errors)] == ["c", "a", "b"]   # b has no error: last
    assert [i.frame for i in disagreement_queue(str(p), str(tmp_path))] == ["b", "c", "a"]           # open hooks first
    assert [i.frame for i in disagreement_queue(str(p), str(tmp_path), errors, only_errors=True)] == ["c", "a"]


def test_skipped_queue_keeps_only_frames_after_germination(tmp_path):
    p = tmp_path / "gt.csv"
    frames = ["f0", "f1", "f2", "f3"]
    write(p, [skipped(0, f, series="") for f in frames] + [skipped(1, f, series="") for f in frames])  # a plain crop folder
    for c in (0, 1):
        for f in frames:
            (tmp_path / f"{c}-crop-{f}.png").write_bytes(b"x")
    g = tmp_path / "g.csv"
    g.write_text("series,seedling_id,crop_id,status,onset_frame,onset_index,n_frames\n"
                 ",1,0,found,f2,2,4\n,2,1,none,,,4\n", encoding="utf-8")
    assert onsets(str(g)) == {("", 0): 2}
    queue = skipped_after_onset_queue(str(p), str(g), str(tmp_path))
    assert [(i.crop_id, i.frame) for i in queue] == [(0, "f2"), (0, "f3")]        # seedling 1 never germinated: none
    ReviewStore(str(p)).save_rotation(item(0, "f2", series=""))
    assert [(i.crop_id, i.frame) for i in skipped_after_onset_queue(str(p), str(g), str(tmp_path))] == [(0, "f3")]


def test_the_trainers_see_the_review(tmp_path):
    from multi.src.landmarks import load_landmarks
    p = tmp_path / "gt.csv"
    write(p, [measured(0, "f1", 90), measured(0, "f2", 40)])
    store = ReviewStore(str(p))
    store.save_rotation(item(0, "f1"))
    lms, report = load_landmarks(str(p))
    assert set(lms) == {"0-crop-f2.png"} and report["loaded"] == 1
    from multi.validate_angles import load_truth
    assert [r["frame"] for r in load_truth(str(p))] == ["f2"]
