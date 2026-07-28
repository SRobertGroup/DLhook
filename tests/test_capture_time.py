import datetime
import os

import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import ImageFileDirectory_v2

from utils.preprocess_model_input import (
    compute_time_deltas,
    filename_sort_key,
    get_image_creation_time,
    metadata_has_gaps,
    order_series,
)

DATETIME_IFD0 = 0x0132  # TIFF tag 306


def _write_tiff(path, capture_time=None):
    """A 2x2 TIFF, optionally carrying tag 306 (DateTime)."""
    image = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8))
    if capture_time is None:
        image.save(path)
        return
    ifd = ImageFileDirectory_v2()
    ifd[DATETIME_IFD0] = capture_time.strftime("%Y:%m:%d %H:%M:%S")
    image.save(path, tiffinfo=ifd)


def test_embedded_timestamp_wins_over_newer_filesystem_time(tmp_path):
    captured = datetime.datetime(2021, 3, 4, 5, 6, 7)
    path = tmp_path / "frame.tif"
    _write_tiff(path, captured)
    # The file was written just now, so every filesystem time is far newer than
    # the embedded tag -- the embedded one must still win.
    assert get_image_creation_time(str(path)) == captured


def test_falls_back_to_filesystem_when_nothing_embedded(tmp_path):
    path = tmp_path / "frame.png"
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(path)

    stat = os.stat(path)
    expected = datetime.datetime.fromtimestamp(min(stat.st_ctime, stat.st_mtime))
    assert get_image_creation_time(str(path)) == expected


def test_placeholder_timestamp_is_not_treated_as_a_capture_time(tmp_path):
    """"0000:00:00 00:00:00" means "unknown", not the year zero."""
    path = tmp_path / "frame.tif"
    image = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8))
    ifd = ImageFileDirectory_v2()
    ifd[DATETIME_IFD0] = "0000:00:00 00:00:00"
    image.save(path, tiffinfo=ifd)

    resolved = get_image_creation_time(str(path))
    assert resolved is not None
    assert resolved.year > 2000  # i.e. it fell through to the filesystem


def test_order_series_uses_capture_time_when_filename_keys_collide(tmp_path):
    """Every name here yields the same legacy sort key (13), and lexicographic
    order is the reverse of acquisition order -- only timestamps can order it."""
    times = {
        "RPV_13_3.tif": datetime.datetime(2024, 7, 26, 10, 0),
        "RPV_13_2.tif": datetime.datetime(2024, 7, 26, 10, 20),
        "RPV_13_1.tif": datetime.datetime(2024, 7, 26, 10, 40),
    }
    for name, captured in times.items():
        _write_tiff(tmp_path / name, captured)
    assert len({filename_sort_key(n) for n in times}) == 1

    ordered, capture_times = order_series(str(tmp_path), sorted(times))

    assert ordered == ["RPV_13_3.tif", "RPV_13_2.tif", "RPV_13_1.tif"]
    assert capture_times == times


def test_order_series_keeps_files_without_digits(tmp_path):
    """The old sort dropped digit-less names from file_list entirely."""
    for name in ("frame_2.tif", "frame_10.tif", "overview.tif"):
        _write_tiff(tmp_path / name)  # no embedded time -> legacy filename key

    ordered, _ = order_series(str(tmp_path), ["overview.tif", "frame_10.tif", "frame_2.tif"])

    assert sorted(ordered) == ["frame_10.tif", "frame_2.tif", "overview.tif"]
    # Numeric (not lexicographic) among the numbered frames, digit-less last.
    assert ordered == ["frame_2.tif", "frame_10.tif", "overview.tif"]


def test_elapsed_minutes_are_never_negative_for_an_unsorted_list():
    metadata = [
        {"filename": "b.tif", "creation_time": datetime.datetime(2024, 1, 1, 12, 30)},
        {"filename": "a.tif", "creation_time": datetime.datetime(2024, 1, 1, 12, 0)},
    ]
    deltas = compute_time_deltas(metadata)

    assert [d["elapsed_minutes"] for d in deltas] == [30.0, 0.0]
    assert metadata_has_gaps(metadata) is True  # out of order -> untrustworthy


def test_fallback_interval_fills_missing_timestamps():
    metadata = [
        {"filename": "a.tif", "creation_time": None},
        {"filename": "b.tif", "creation_time": None},
    ]
    deltas = compute_time_deltas(metadata, fallback_interval_minutes=15.0)

    assert [d["elapsed_minutes"] for d in deltas] == [0.0, 15.0]


def test_cancelled_fallback_leaves_elapsed_undetermined():
    metadata = [{"filename": "a.tif", "creation_time": None}]

    deltas = compute_time_deltas(metadata, fallback_interval_minutes=None)

    assert deltas[0]["elapsed_minutes"] is None
