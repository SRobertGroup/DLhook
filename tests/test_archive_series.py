"""multi/archive_series.py: archive a duplicate series, keep some seedlings under another series."""
import csv
import json

from multi.archive_series import run

MANIFEST = ["output_path", "series", "source_frame", "crop_id", "cx", "cy", "half_w", "half_h",
            "x1", "y1", "x2", "y2", "crop_width", "crop_height"]
ANGLE = ["series", "seedling_id", "crop_id", "frame", "img_name", "crop_file", "status", "theta"]
SETS = {"training": {"dir": "crops", "boxes": "boxes.json", "keep": [1], "csvs": ["angle.csv"],
                     "crop_dirs": ["labels"], "seg_dir": "seg", "patch_index": ["patches.csv"]}}


def write(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({f: r.get(f, "") for f in fields})


def read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build(root):
    for d in ("crops", "labels", "seg"):
        (root / d).mkdir()
    man, angle, patches = [], [], []
    spec = [("AGJKV system pictures", 0, "AGJKV_system_pictures"), ("dup", 0, "dup"), ("dup", 1, "dup")]
    for series, cid, pre in spec:
        name = f"{cid}-crop-{pre}_f1.png"
        (root / "crops" / name).write_bytes(b"c")
        (root / "labels" / name).write_bytes(b"l")
        for suf in ("cotyledon", "hypocotyl", "radicle"):
            (root / "seg" / f"{name[:-4]}-{suf}.png").write_bytes(b"s")
        man.append({"output_path": f"crops\\{name}", "series": series, "source_frame": "f1.TIF", "crop_id": cid,
                    "cx": 10 + cid, "cy": 20, "half_w": 5, "half_h": 8})
        angle.append({"series": series, "seedling_id": cid + 1, "crop_id": cid, "frame": f"{pre}_f1",
                      "img_name": "f1.TIF", "crop_file": name, "status": "measured", "theta": 30})
        patches.append({"filename": name, "x": 0, "y": 0})
    write(root / "crops" / "manifest.csv", MANIFEST, man)
    write(root / "angle.csv", ANGLE, angle)
    write(root / "patches.csv", ["filename", "x", "y"], patches)
    (root / "boxes.json").write_text(json.dumps({
        "AGJKV system pictures": [{"cx": 10, "cy": 20, "half_w": 5, "half_h": 8}],
        "dup": [{"cx": 10, "cy": 20, "half_w": 5, "half_h": 8}, {"cx": 11, "cy": 20, "half_w": 5, "half_h": 8}]}))


def test_dry_run_changes_nothing(tmp_path):
    build(tmp_path)
    before = sorted(p.name for p in (tmp_path / "crops").iterdir())
    run(tmp_path, "dup", "AGJKV system pictures", tmp_path / "arch", dry_run=True, sets=SETS)
    assert sorted(p.name for p in (tmp_path / "crops").iterdir()) == before and not (tmp_path / "arch").exists()


def test_archives_one_seedling_and_renames_the_kept_one_into_the_target_series(tmp_path):
    build(tmp_path)
    run(tmp_path, "dup", "AGJKV system pictures", tmp_path / "arch", dry_run=False, sets=SETS)
    crops = sorted(p.name for p in (tmp_path / "crops").iterdir())
    assert crops == ["0-crop-AGJKV_system_pictures_f1.png", "1-crop-AGJKV_system_pictures_f1.png", "manifest.csv"]
    assert (tmp_path / "arch" / "training" / "crops" / "0-crop-dup_f1.png").exists()          # the archived one (dup 0)
    assert (tmp_path / "arch" / "training" / "seg" / "0-crop-dup_f1-radicle.png").exists()
    assert (tmp_path / "labels" / "1-crop-AGJKV_system_pictures_f1.png").exists()
    assert (tmp_path / "seg" / "1-crop-AGJKV_system_pictures_f1-hypocotyl.png").exists()
    man = read(tmp_path / "crops" / "manifest.csv")
    assert [(r["series"], r["crop_id"], r["output_path"]) for r in man] == [
        ("AGJKV system pictures", "0", "crops\\0-crop-AGJKV_system_pictures_f1.png"),
        ("AGJKV system pictures", "1", "crops\\1-crop-AGJKV_system_pictures_f1.png")]
    angle = read(tmp_path / "angle.csv")
    assert [(r["series"], r["seedling_id"], r["crop_id"], r["frame"], r["crop_file"]) for r in angle] == [
        ("AGJKV system pictures", "1", "0", "AGJKV_system_pictures_f1", "0-crop-AGJKV_system_pictures_f1.png"),
        ("AGJKV system pictures", "2", "1", "AGJKV_system_pictures_f1", "1-crop-AGJKV_system_pictures_f1.png")]
    assert [r["filename"] for r in read(tmp_path / "patches.csv")] == [
        "0-crop-AGJKV_system_pictures_f1.png", "1-crop-AGJKV_system_pictures_f1.png"]
    boxes = json.loads((tmp_path / "boxes.json").read_text())
    assert "dup" not in boxes and boxes["AGJKV system pictures"][1] == {"cx": 11, "cy": 20, "half_w": 5, "half_h": 8}
    assert (tmp_path / "arch" / "_backup" / "angle.csv").exists()                              # originals kept for an undo
    assert len(read(tmp_path / "arch" / "_backup" / "angle.csv")) == 3
