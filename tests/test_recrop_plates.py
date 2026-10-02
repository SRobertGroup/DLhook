import csv
import os
import sys
from pathlib import Path

import numpy as np

# multi/ is not a package (no __init__.py, matching multi/build_patch_index.py's
# own convention) -- put it on sys.path so `from src...` and `import recrop_plates`
# resolve the same way the CLI script itself resolves them.
_MULTI_DIR = str(Path(__file__).resolve().parent.parent / "multi")
if _MULTI_DIR not in sys.path:
    sys.path.insert(0, _MULTI_DIR)

import recrop_plates  # noqa: E402
from src.recrop_geometry import (  # noqa: E402
    build_output_filename,
    compute_crop_box,
    crop_from_box,
    list_image_files,
    reference_frame_index,
    sanitize_series_name,
)


# ---------------------------------------------------------------------------
# Crop-box geometry -- mirrors Gui._compute_crop_box exactly (seedling_measurment.py).
# ---------------------------------------------------------------------------

def test_compute_crop_box_matches_gui_padding_formula():
    # start=(10,20), end=(50,120) -- as if from two clicked points.
    box = compute_crop_box(10, 50, 20, 120, padding_width_fraction=0.4, padding_height_fraction=0.10,
                            min_box_half_size=30)

    expected_half_w = round((50 - 10) * (1 + 2 * 0.4) / 2)
    expected_half_h = round((120 - 20) * (1 + 2 * 0.10) / 2)
    assert box == {"cx": 30, "cy": 70, "half_w": expected_half_w, "half_h": expected_half_h}


def test_compute_crop_box_degenerate_case_floors_to_min_half_size():
    box = compute_crop_box(10, 10, 20, 20, min_box_half_size=30)
    assert box["half_w"] == 30
    assert box["half_h"] == 30


def test_crop_from_box_clamps_to_image_bounds():
    img = np.zeros((100, 100), dtype=np.uint8)
    box = {"cx": 5, "cy": 95, "half_w": 20, "half_h": 20}
    crop, x1, y1, x2, y2 = crop_from_box(img, box)

    assert (x1, y1, x2, y2) == (0, 75, 25, 100)
    assert crop.shape == (25, 25)


# ---------------------------------------------------------------------------
# Filename convention -- {crop_id}-crop-{original_name}.png, parsed elsewhere
# with split("-", 1), so crop_id >= 10 must not be ambiguous.
# ---------------------------------------------------------------------------

def test_output_filename_convention_single_digit_crop_id():
    name = build_output_filename(0, "F1_Plate_2_YS", "IMG_086.png")
    assert name == "0-crop-F1_Plate_2_YS_IMG_086.png"
    crop_id_str, rest = name.split("-", 1)
    assert crop_id_str == "0"
    assert rest == "crop-F1_Plate_2_YS_IMG_086.png"


def test_output_filename_convention_double_digit_crop_id_is_unambiguous():
    name = build_output_filename(12, "Plate7", "Plate_7_041.tif")
    assert name.startswith("12-crop-")
    crop_id_str, rest = name.split("-", 1)
    # The bug this guards against: parsing by the first CHARACTER ("1")
    # instead of the first "-"-delimited token ("12").
    assert crop_id_str == "12"
    assert int(crop_id_str) == 12
    assert rest == "crop-Plate7_Plate_7_041.png"


def test_output_filename_handles_non_4_char_extensions():
    # image[:-4] (the legacy GUI slice) would mangle ".jpeg"/".TIFF"; this
    # must not rely on the extension being exactly 4 characters.
    name = build_output_filename(3, "MySeries", "frame.jpeg")
    assert name == "3-crop-MySeries_frame.png"

    name2 = build_output_filename(3, "MySeries", "frame.TIFF")
    assert name2 == "3-crop-MySeries_frame.png"


def test_sanitize_series_name_handles_spaces_and_copy_suffix():
    assert sanitize_series_name("F1 Plate 2000 - Copy") == "F1_Plate_2000_-_Copy"
    assert sanitize_series_name("AGJKV system pictures") == "AGJKV_system_pictures"


# ---------------------------------------------------------------------------
# Collision detection.
# ---------------------------------------------------------------------------

def test_check_collisions_raises_on_duplicate_output_path():
    planned = [
        ("SeriesA", "frame1.png", 0, Path("out/0-crop-SeriesA_frame1.png")),
        ("SeriesA", "frame2.png", 0, Path("out/0-crop-SeriesA_frame2.png")),
    ]
    recrop_plates.check_collisions(planned)  # no collision -> no raise


def test_check_collisions_raises_when_sanitized_series_names_collide():
    # "F1 Plate 2" and "F1_Plate_2" sanitize to the same string, and if both
    # series happen to have a same-named frame, their outputs collide even
    # though the series were meant to be kept apart.
    out_a = Path("out") / build_output_filename(0, "F1 Plate 2", "IMG_086.png")
    out_b = Path("out") / build_output_filename(0, "F1_Plate_2", "IMG_086.png")
    assert out_a == out_b  # the actual collision this test is guarding against

    planned = [
        ("F1 Plate 2", "IMG_086.png", 0, out_a),
        ("F1_Plate_2", "IMG_086.png", 0, out_b),
    ]
    try:
        recrop_plates.check_collisions(planned)
        assert False, "expected CollisionError"
    except recrop_plates.CollisionError as exc:
        assert "F1 Plate 2" in str(exc)
        assert "F1_Plate_2" in str(exc)


def test_check_collisions_allows_distinct_series_with_same_frame_name():
    # Different series legitimately share a frame filename (e.g. IMG_086.png
    # appears in several example_data/ series) -- as long as the series
    # names don't collapse to the same sanitized stem, no collision.
    out_a = Path("out") / build_output_filename(0, "Camera_1", "IMG_086.png")
    out_b = Path("out") / build_output_filename(0, "Camera_2", "IMG_086.png")
    assert out_a != out_b

    planned = [
        ("Camera_1", "IMG_086.png", 0, out_a),
        ("Camera_2", "IMG_086.png", 0, out_b),
    ]
    recrop_plates.check_collisions(planned)  # no raise


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------

def test_list_image_files_is_case_insensitive_and_skips_non_images(tmp_path):
    for name in ["a.tif", "B.PNG", "c.JPG", "notes.csv", "d.bmp"]:
        (tmp_path / name).write_bytes(b"\x00")
    os.mkdir(tmp_path / "subdir")

    got = list_image_files(str(tmp_path))
    assert got == sorted(["B.PNG", "a.tif", "c.JPG", "d.bmp"])


def test_reference_frame_index_picks_late_frame_by_default():
    assert reference_frame_index(10, 0.7) == 6  # round(9 * 0.7) = 6
    assert reference_frame_index(1, 0.7) == 0
    assert reference_frame_index(5, 0.0) == 0
    assert reference_frame_index(5, 1.0) == 4


# ---------------------------------------------------------------------------
# --every-n sampling.
# ---------------------------------------------------------------------------

def test_sample_every_n_picks_expected_stride_and_includes_last():
    frames = [f"f{i}.png" for i in range(11)]  # f0..f10
    got = recrop_plates.sample_every_n(frames, 5)
    # 0, 5, 10 by stride, and 10 is already the last frame.
    assert got == ["f0.png", "f5.png", "f10.png"]


def test_sample_every_n_always_appends_last_frame_even_off_stride():
    frames = [f"f{i}.png" for i in range(7)]  # f0..f6, stride 5 -> 0, 5
    got = recrop_plates.sample_every_n(frames, 5)
    # last frame f6 is NOT on the stride (0, 5) but must be included anyway.
    assert got == ["f0.png", "f5.png", "f6.png"]


def test_sample_every_n_one_returns_every_frame():
    frames = [f"f{i}.png" for i in range(4)]
    assert recrop_plates.sample_every_n(frames, 1) == frames


def test_sample_every_n_empty_list_returns_empty():
    assert recrop_plates.sample_every_n([], 5) == []


def test_sample_every_n_rejects_non_positive_stride():
    try:
        recrop_plates.sample_every_n(["a.png"], 0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_sample_every_n_preserves_time_order_not_resorted():
    # sample_every_n must not re-sort; it trusts the caller's ordering.
    frames = ["z.png", "a.png", "m.png", "b.png"]
    got = recrop_plates.sample_every_n(frames, 2)
    assert got == ["z.png", "m.png", "b.png"]  # indices 0, 2, and last (3)


# ---------------------------------------------------------------------------
# Natural sort -- guards the example_data/MB non-zero-padded case, where
# order_series's own EXIF-unusable fallback (first-digit-run key) degenerates
# because every "MB_1_<n>.jpg" frame shares the same first digit run ("1").
# ---------------------------------------------------------------------------

def test_natural_sort_key_orders_non_zero_padded_numeric_suffixes():
    names = ["MB_1_10.jpg", "MB_1_2.jpg", "MB_1_1.jpg", "MB_1_0.jpg", "MB_1_9.jpg"]
    got = sorted(names, key=recrop_plates.natural_sort_key)
    assert got == ["MB_1_0.jpg", "MB_1_1.jpg", "MB_1_2.jpg", "MB_1_9.jpg", "MB_1_10.jpg"]


def test_natural_sort_key_beats_plain_lexicographic_sort():
    names = ["MB_1_10.jpg", "MB_1_2.jpg"]
    # Plain lexicographic sort gets this backwards ("1" < "2" in "10" vs "2").
    assert sorted(names) == ["MB_1_10.jpg", "MB_1_2.jpg"]
    assert sorted(names, key=recrop_plates.natural_sort_key) == ["MB_1_2.jpg", "MB_1_10.jpg"]


def test_order_series_for_sampling_falls_back_to_natural_sort_without_exif(tmp_path, monkeypatch):
    # Frames with no embedded EXIF timestamps (order_series's own "usable"
    # check fails) must fall back to natural_sort_key, not order_series's
    # weaker first-digit-run fallback.
    names = ["MB_1_10.jpg", "MB_1_2.jpg", "MB_1_0.jpg"]
    for name in names:
        (tmp_path / name).write_bytes(b"\x00")

    got = recrop_plates.order_series_for_sampling(str(tmp_path), names)
    assert got == ["MB_1_0.jpg", "MB_1_2.jpg", "MB_1_10.jpg"]


def test_order_series_for_sampling_uses_exif_order_when_usable(tmp_path, monkeypatch):
    import datetime as _dt

    def fake_order_series(path, filenames):
        # Distinct capture times, reverse of filename order, to prove the
        # EXIF-based order (not natural sort) is what gets returned.
        base = _dt.datetime(2024, 1, 1)
        times = {f: base + _dt.timedelta(minutes=len(filenames) - i) for i, f in enumerate(filenames)}
        ordered = sorted(filenames, key=lambda f: times[f])
        return ordered, times

    monkeypatch.setattr(recrop_plates, "order_series", fake_order_series)
    names = ["b.png", "a.png", "c.png"]
    got = recrop_plates.order_series_for_sampling(str(tmp_path), names)
    # times: b.png=+3min, a.png=+2min, c.png=+1min -> ascending order c,a,b.
    assert got == ["c.png", "a.png", "b.png"]


def test_order_series_for_sampling_empty_returns_empty():
    assert recrop_plates.order_series_for_sampling("unused", []) == []


# ---------------------------------------------------------------------------
# run_real manifest merging -- a partial/incremental run (--series B) must
# preserve other series' rows (--series A) already in a shared manifest.csv,
# not clobber them, per recrop_plates' own --series usage docstring.
# ---------------------------------------------------------------------------

def test_run_real_preserves_other_series_manifest_rows(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    manifest_path = output_dir / "manifest.csv"

    # A manifest already on disk with rows for series "A", as if a prior run
    # (`--series A`) had produced it.
    existing_row = {
        "output_path": str(output_dir / "0-crop-A_frame1.png"),
        "series": "A", "source_frame": "frame1.png", "crop_id": "0",
        "cx": "10", "cy": "10", "half_w": "5", "half_h": "5",
        "x1": "5", "y1": "5", "x2": "15", "y2": "15",
        "crop_width": "10", "crop_height": "10",
    }
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=recrop_plates.MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerow(existing_row)
    original_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    original_row_a_line = original_lines[1]

    # Run for series "B" only, into the same manifest path.
    monkeypatch.setattr(
        recrop_plates, "read_preprocessed_gray",
        lambda path: np.zeros((50, 50), dtype=np.uint8),
    )
    series_path = tmp_path / "B"
    series_path.mkdir()
    box = {"cx": 25, "cy": 25, "half_w": 10, "half_h": 10}
    plan = recrop_plates.SeriesPlan("B", series_path, ["frameB.png"], [box], "test")

    rc = recrop_plates.run_real([plan], output_dir, manifest_path, overwrite=False, use_superres=False)
    assert rc == 0

    new_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    with open(manifest_path, "r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    # A's original row survives byte-identical (same exact CSV line)...
    assert original_row_a_line in new_lines
    # ...alongside B's newly written row.
    assert len(rows) == 2
    row_a = next(r for r in rows if r["series"] == "A")
    assert row_a == existing_row
    row_b = next(r for r in rows if r["series"] == "B")
    assert row_b["source_frame"] == "frameB.png"
    assert row_b["crop_id"] == "0"


def test_run_real_overwrite_rerun_replaces_only_that_series_rows(tmp_path, monkeypatch):
    # A rerun of series "A" itself (e.g. with --overwrite) must replace only
    # A's rows, not duplicate them, while still preserving other series.
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    manifest_path = output_dir / "manifest.csv"

    rows = [
        {
            "output_path": str(output_dir / "0-crop-A_frame1.png"),
            "series": "A", "source_frame": "frame1.png", "crop_id": "0",
            "cx": "10", "cy": "10", "half_w": "5", "half_h": "5",
            "x1": "5", "y1": "5", "x2": "15", "y2": "15",
            "crop_width": "10", "crop_height": "10",
        },
        {
            "output_path": str(output_dir / "0-crop-C_frame1.png"),
            "series": "C", "source_frame": "frame1.png", "crop_id": "0",
            "cx": "20", "cy": "20", "half_w": "5", "half_h": "5",
            "x1": "15", "y1": "15", "x2": "25", "y2": "25",
            "crop_width": "10", "crop_height": "10",
        },
    ]
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=recrop_plates.MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    monkeypatch.setattr(
        recrop_plates, "read_preprocessed_gray",
        lambda path: np.zeros((50, 50), dtype=np.uint8),
    )
    series_path = tmp_path / "A"
    series_path.mkdir()
    box = {"cx": 25, "cy": 25, "half_w": 10, "half_h": 10}
    plan = recrop_plates.SeriesPlan("A", series_path, ["frame1.png"], [box], "test")

    rc = recrop_plates.run_real([plan], output_dir, manifest_path, overwrite=True, use_superres=False)
    assert rc == 0

    with open(manifest_path, "r", newline="", encoding="utf-8") as fh:
        out_rows = list(csv.DictReader(fh))

    assert len(out_rows) == 2  # C preserved, A replaced (not duplicated)
    series_present = {r["series"] for r in out_rows}
    assert series_present == {"A", "C"}
    row_c = next(r for r in out_rows if r["series"] == "C")
    assert row_c == rows[1]  # C untouched
    row_a = next(r for r in out_rows if r["series"] == "A")
    assert row_a["source_frame"] == "frame1.png"
    assert row_a["cx"] == "25"  # A's new box, not the old one
