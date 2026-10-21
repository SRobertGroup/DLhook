import csv
import os

from utils.landmark_marks import FolderMarks, MarksStore, clicked_for_folder


def _write(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_priority_corrected_then_clicked_then_model(tmp_path):
    clicked = {"a.png": {"junction": (10.0, 10.0), "collar": None, "root_dir": None}}
    calls = []

    def model(path):
        calls.append(os.path.basename(path))
        return {"junction": (1.0, 1.0), "collar": (5.0, 50.0), "root_dir": (0.0, 1.0)}

    m = FolderMarks(str(tmp_path), path=str(tmp_path / "m.csv"), clicked=clicked, model=model)
    g = m.get("a.png")
    assert g["junction"] == (10.0, 10.0) and g["junction_source"] == "clicked"
    assert g["collar"] == (5.0, 50.0) and g["collar_source"] == "model" and g["root_dir"] == (0.0, 1.0)
    m.correct("a.png", "junction", (12.0, 11.0))
    assert m.get("a.png")["junction_source"] == "corrected"
    m.get("a.png")
    assert calls == ["a.png"]                                    # the model runs once per crop
    m.save()
    again = FolderMarks(str(tmp_path), path=str(tmp_path / "m.csv"), clicked=clicked, model=None)
    g = again.get("a.png")
    assert g["junction"] == (12.0, 11.0) and g["collar"] is None and g["collar_source"] == ""


def test_restore_and_removing_the_last_correction(tmp_path):
    m = FolderMarks(str(tmp_path), path=str(tmp_path / "m.csv"), clicked={}, model=None)
    before = m.get("a.png")
    m.correct("a.png", "collar", (3.0, 4.0))
    m.restore("a.png", "collar", before)
    assert "a.png" not in m.store.rows
    m.save()
    assert MarksStore(str(tmp_path / "m.csv")).rows == {}


def test_clicked_for_folder_reads_angle_and_root_csvs(tmp_path):
    folder = tmp_path / "crops"
    folder.mkdir()
    (folder / "0-crop-x.png").write_bytes(b"")
    angle, root = tmp_path / "a.csv", tmp_path / "a_root.csv"
    _write(angle, ["series", "crop_id", "frame", "crop_file", "status", "junction_x", "junction_y", "hypo2_x", "hypo2_y"],
           [{"series": "S", "crop_id": 0, "frame": "x", "crop_file": "0-crop-x.png", "status": "measured",
             "junction_x": 3, "junction_y": 4, "hypo2_x": 3, "hypo2_y": 30}])
    _write(root, ["crop_file", "collar_x", "collar_y", "root_dir_x", "root_dir_y"],
           [{"crop_file": "0-crop-x.png", "collar_x": 3, "collar_y": 60, "root_dir_x": 0, "root_dir_y": 1}])
    got = clicked_for_folder(str(folder), sources=[(str(angle), str(root), str(folder))])
    assert got["0-crop-x.png"]["junction"] == (3.0, 4.0)
    assert got["0-crop-x.png"]["collar"] == (3.0, 60.0) and got["0-crop-x.png"]["root_dir"] == (0.0, 1.0)


def test_load_landmarks_applies_corrected_marks(tmp_path):
    from multi.src.landmarks import load_landmarks

    fields = ["series", "crop_id", "frame", "crop_file", "status", "junction_x", "junction_y", "hypo1_x", "hypo1_y",
              "hypo2_x", "hypo2_y", "cotyl1_x", "cotyl1_y", "cotyl2_x", "cotyl2_y"]
    row = {"series": "S", "crop_id": 0, "frame": "f", "status": "measured", "junction_x": 20, "junction_y": 20,
           "hypo1_x": 20, "hypo1_y": 30, "hypo2_x": 20, "hypo2_y": 60, "cotyl1_x": 22, "cotyl1_y": 18,
           "cotyl2_x": 30, "cotyl2_y": 10}
    angle = tmp_path / "a.csv"
    _write(angle, fields, [dict(row, crop_file="a.png"), dict(row, crop_file="b.png")])
    store = MarksStore(str(tmp_path / "m.csv"))
    store.set("a.png", "junction", (21.0, 22.0))
    store.set("a.png", "collar", (20.0, 80.0), root_dir=(0.0, 1.0))
    store.set("b.png", "collar", (20.0, 80.0))                          # no root direction anywhere
    store.save()
    lms, report = load_landmarks(str(angle), marks_csv=str(tmp_path / "m.csv"))
    plain, _ = load_landmarks(str(angle))
    assert lms["a.png"].junction == (21.0, 22.0) and lms["a.png"].hypo_dir == plain["a.png"].hypo_dir
    assert lms["a.png"].collar == (20.0, 80.0) and lms["a.png"].root_dir == (0.0, 1.0)
    assert lms["b.png"].collar is None
    assert report["marks_junction"] == 1 and report["marks_collar"] == 1 and report["marks_collar_no_root"] == 1


def test_marks_csv_for_follows_the_config(tmp_path):
    from multi.train_unet_multiclass import marks_csv_for

    (tmp_path / "cropped_x.csv").write_text("crop_file\n", encoding="utf-8")
    assert marks_csv_for({"landmarks": {}}, "data/crops/cropped_x") is None
    cfg = {"landmarks": {"marks_dir": str(tmp_path)}}
    assert marks_csv_for(cfg, "data/crops/cropped_x") == str(tmp_path / "cropped_x.csv")
    assert marks_csv_for(cfg, "data/crops/cropped_y") is None
