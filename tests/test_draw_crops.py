import json
import os
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from multi.draw_crops import (  # noqa: E402
    BoxStore,
    _capture_times_are_usable,
    discover_series_last_frames,
    natural_sort_key,
    pick_last_frame,
    resize_rect_from_corners,
    round_box,
)
from multi.recrop_plates import load_boxes_file  # noqa: E402


def _touch(path, content=b"not a real image, just a placeholder for extension-based discovery"):
    with open(path, "wb") as fh:
        fh.write(content)


# ---------------------------------------------------------------------------
# natural_sort_key / pick_last_frame -- the MB non-zero-padded trap.
# ---------------------------------------------------------------------------

def test_natural_sort_key_orders_non_zero_padded_numbers_correctly():
    names = ["MB_1_9.jpg", "MB_1_64.jpg", "MB_1_0.jpg", "MB_1_10.jpg"]
    ordered = sorted(names, key=natural_sort_key)
    assert ordered == ["MB_1_0.jpg", "MB_1_9.jpg", "MB_1_10.jpg", "MB_1_64.jpg"]
    # A plain lexicographic sort gets this wrong -- guard the regression this
    # test exists for.
    assert sorted(names) != ordered


def test_pick_last_frame_falls_back_to_natural_sort_without_exif(tmp_path):
    # Dummy (non-image) files: order_series's embedded-timestamp lookup fails
    # to open them and returns None for every frame, so pick_last_frame must
    # fall through to natural_sort_key -- not order_series's own legacy
    # first-digit-run + lexicographic fallback, which mis-sorts exactly this
    # filename pattern (see CLAUDE.md / task notes on example_data/MB).
    series_dir = tmp_path / "MB_like"
    series_dir.mkdir()
    names = [f"MB_1_{i}.jpg" for i in (0, 1, 9, 10, 63, 64)]
    for name in names:
        _touch(series_dir / name)

    last = pick_last_frame(str(series_dir), names)
    assert last == "MB_1_64.jpg"


def test_capture_times_are_usable_requires_all_present_and_not_all_identical():
    import datetime
    t0 = datetime.datetime(2024, 1, 1)
    t1 = datetime.datetime(2024, 1, 2)

    assert _capture_times_are_usable({"a": t0, "b": t1}) is True
    assert _capture_times_are_usable({"a": t0, "b": None}) is False  # missing one
    assert _capture_times_are_usable({"a": t0, "b": t0}) is False    # all identical
    assert _capture_times_are_usable({}) is False


# ---------------------------------------------------------------------------
# discover_series_last_frames -- trap #1 (stray files skipped) and trap #2
# (non-recursive frame listing).
# ---------------------------------------------------------------------------

def test_discover_series_last_frames_skips_stray_files_at_top_level(tmp_path):
    example_data = tmp_path / "example_data"
    example_data.mkdir()
    _touch(example_data / "some_notes.csv")
    _touch(example_data / "readme.txt")

    series_a = example_data / "SeriesA"
    series_a.mkdir()
    for i in range(3):
        _touch(series_a / f"frame_{i}.png")

    result = discover_series_last_frames(str(example_data))
    names = [name for name, _path, _frame in result]
    assert names == ["SeriesA"]  # the stray .csv/.txt are not series


def test_discover_series_last_frames_is_non_recursive(tmp_path):
    example_data = tmp_path / "example_data"
    example_data.mkdir()

    plate_dir = example_data / "F1_Plate_2_YS"
    plate_dir.mkdir()
    # Top-level plate frames, deliberately NOT zero-padded so a lexicographic
    # (or subfolder-polluted) scan would pick the wrong "last" frame.
    for i in (0, 1, 9, 10):
        _touch(plate_dir / f"F1 Plate 20{i}.tif")

    # A mask subfolder, like example_data/F1_Plate_2_YS/segmented_s0/ --
    # holds many files whose names would sort after the plate frames above.
    mask_dir = plate_dir / "segmented_s0"
    mask_dir.mkdir()
    for i in range(20):
        _touch(mask_dir / f"zzz_mask_{i:04d}.png")

    result = discover_series_last_frames(str(example_data))
    assert len(result) == 1
    name, dir_path, last_frame = result[0]
    assert name == "F1_Plate_2_YS"
    assert last_frame == "F1 Plate 2010.tif"  # highest-numbered TOP-LEVEL frame
    assert "segmented" not in last_frame


def test_discover_series_last_frames_raises_clear_error_for_missing_dir(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        discover_series_last_frames(str(tmp_path / "does_not_exist"))


# ---------------------------------------------------------------------------
# Pure resize geometry -- must NOT re-apply compute_crop_box padding.
# ---------------------------------------------------------------------------

def test_resize_rect_from_corners_is_a_plain_unpadded_rectangle():
    box = resize_rect_from_corners(10, 20, 50, 120, min_half_size=5)
    assert box == {"cx": 30, "cy": 70, "half_w": 20, "half_h": 50}


def test_resize_rect_from_corners_floors_at_min_half_size():
    box = resize_rect_from_corners(10, 10, 12, 11, min_half_size=30)
    assert box["half_w"] == 30
    assert box["half_h"] == 30


def test_round_box_rounds_every_field_to_int():
    box = round_box({"cx": 10.4, "cy": 10.6, "half_w": 5.5, "half_h": 5.49})
    assert box == {"cx": 10, "cy": 11, "half_w": 6, "half_h": 5}
    assert all(isinstance(v, int) for v in box.values())


# ---------------------------------------------------------------------------
# BoxStore -- atomic save/load round trip, resumability.
# ---------------------------------------------------------------------------

def test_box_store_round_trips_through_disk(tmp_path):
    out_path = tmp_path / "crop_boxes.json"
    store = BoxStore(str(out_path))
    assert store.get("SeriesA") == []  # nothing yet

    boxes = [{"cx": 100, "cy": 200, "half_w": 30, "half_h": 40}]
    store.set("SeriesA", boxes)

    assert out_path.exists()
    # No leftover temp file after an atomic write.
    assert list(tmp_path.glob(".crop_boxes_*.tmp")) == []

    reloaded = BoxStore(str(out_path))
    assert reloaded.get("SeriesA") == boxes


def test_box_store_mutations_are_independent_copies(tmp_path):
    # get()/set() must not hand back references the caller can mutate behind
    # the store's back.
    store = BoxStore(str(tmp_path / "crop_boxes.json"))
    boxes = [{"cx": 1, "cy": 2, "half_w": 3, "half_h": 4}]
    store.set("S", boxes)
    boxes[0]["cx"] = 999  # mutate the caller's copy after the fact

    assert store.get("S")[0]["cx"] == 1

    fetched = store.get("S")
    fetched[0]["cx"] = 12345
    assert store.get("S")[0]["cx"] == 1


def test_box_store_preserves_other_series_on_update(tmp_path):
    store = BoxStore(str(tmp_path / "crop_boxes.json"))
    store.set("SeriesA", [{"cx": 1, "cy": 1, "half_w": 1, "half_h": 1}])
    store.set("SeriesB", [{"cx": 2, "cy": 2, "half_w": 2, "half_h": 2}])

    store.set("SeriesA", [])  # clear SeriesA's boxes

    reloaded = BoxStore(str(store.path))
    assert reloaded.get("SeriesA") == []
    assert reloaded.get("SeriesB") == [{"cx": 2, "cy": 2, "half_w": 2, "half_h": 2}]


# ---------------------------------------------------------------------------
# Output contract -- must be accepted by recrop_plates.load_boxes_file
# unmodified.
# ---------------------------------------------------------------------------

def test_box_store_output_is_accepted_by_recrop_plates_load_boxes_file(tmp_path):
    out_path = tmp_path / "crop_boxes.json"
    store = BoxStore(str(out_path))
    store.set("F1_Plate_2_YS", [
        {"cx": 428, "cy": 2783, "half_w": 120, "half_h": 90},
        {"cx": 900, "cy": 2600, "half_w": 100, "half_h": 80},
    ])
    store.set("MB", [{"cx": 50, "cy": 60, "half_w": 30, "half_h": 30}])

    loaded = load_boxes_file(str(out_path))  # raises on any schema problem

    assert loaded == json.loads(out_path.read_text(encoding="utf-8"))
    assert set(loaded["F1_Plate_2_YS"][0]) == {"cx", "cy", "half_w", "half_h"}
    assert len(loaded["F1_Plate_2_YS"]) == 2
    assert len(loaded["MB"]) == 1


def test_box_store_rejects_downstream_when_a_key_is_missing(tmp_path):
    # Sanity check that load_boxes_file's validation is actually exercised --
    # BoxStore itself doesn't validate, so a hand-edited/corrupt file must
    # still be caught by the existing contract.
    import pytest
    out_path = tmp_path / "crop_boxes.json"
    out_path.write_text(json.dumps({"S": [{"cx": 1, "cy": 1, "half_w": 1}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="half_h"):
        load_boxes_file(str(out_path))
