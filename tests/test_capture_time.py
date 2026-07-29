import datetime
import os

import numpy as np
import pytest
from PIL import Image
from PIL.TiffImagePlugin import ImageFileDirectory_v2

from utils.preprocess_model_input import (
    compute_time_deltas,
    filename_sort_key,
    get_capture_time,
    metadata_has_gaps,
    order_series,
)

DATETIME_IFD0 = 0x0132  # TIFF tag 306


def _write_tiff(path, capture_time=None):
    """A 2x2 TIFF, optionally carrying tag 306 (DateTime).

    The extension check is not pedantry: Pillow picks the format from the
    filename, so saving to "x.jpg" here silently writes a JPEG and drops
    `tiffinfo` entirely -- which is exactly how an earlier version of this file
    ended up asserting against files that carried no timestamp at all.
    """
    assert str(path).endswith(".tif"), "this helper only writes TIFFs"
    image = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8))
    ifd = ImageFileDirectory_v2()
    if capture_time is None:
        # Mirrors example_data/F1_Plate_2_YS and
        # example_data/dark_grown_arabidopsis_kinematics: an ImageJ-written TIFF
        # whose only text tag is the ImageJ version -- no capture time anywhere.
        ifd[0x010E] = "ImageJ=1.54f\n"  # ImageDescription
    else:
        ifd[DATETIME_IFD0] = capture_time.strftime("%Y:%m:%d %H:%M:%S")
    image.save(path, tiffinfo=ifd)


def _metadata(*times):
    return [{"filename": f"f{i}.tif", "creation_time": t} for i, t in enumerate(times)]


def test_embedded_timestamp_is_read(tmp_path):
    captured = datetime.datetime(2021, 3, 4, 5, 6, 7)
    path = tmp_path / "frame.tif"
    _write_tiff(path, captured)

    assert get_capture_time(str(path)) == captured


def test_no_embedded_timestamp_gives_none_not_a_filesystem_time(tmp_path):
    """The F1_Plate_2_YS case. Falling back to st_ctime/st_mtime here is what
    made the elapsed-time axis collapse: all 50 of those TIFFs share two
    filesystem timestamps, and all 168 Camera_1 JPEGs share exactly one."""
    path = tmp_path / "frame.tif"
    _write_tiff(path)

    assert get_capture_time(str(path)) is None


def test_placeholder_timestamp_is_not_treated_as_a_capture_time(tmp_path):
    """"0000:00:00 00:00:00" means "unknown", not the year zero -- which, under
    an earliest-wins rule, would otherwise win every comparison."""
    path = tmp_path / "frame.tif"
    image = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8))
    ifd = ImageFileDirectory_v2()
    ifd[DATETIME_IFD0] = "0000:00:00 00:00:00"
    image.save(path, tiffinfo=ifd)

    assert get_capture_time(str(path)) is None


def test_order_series_uses_capture_time_over_the_filename(tmp_path):
    """The Camera_1 case: every name yields the same legacy sort key, so ordering
    would otherwise fall to the lexicographic tiebreak.

    The names are deliberately chosen so alphabetical order is NOT acquisition
    order -- otherwise this test passes whether the timestamps are read or not,
    which is how an earlier version of it stayed green while `capture_times`
    came back entirely None.
    """
    times = {
        "plate_1_c.tif": datetime.datetime(2026, 7, 8, 15, 22, 44),  # earliest, sorts last
        "plate_1_a.tif": datetime.datetime(2026, 7, 8, 16, 23, 50),
        "plate_1_b.tif": datetime.datetime(2026, 7, 9, 1, 30, 0),    # latest, sorts middle
    }
    for name, captured in times.items():
        _write_tiff(tmp_path / name, captured)
    assert len({filename_sort_key(n) for n in times}) == 1
    assert sorted(times) == ["plate_1_a.tif", "plate_1_b.tif", "plate_1_c.tif"]

    ordered, capture_times = order_series(str(tmp_path), sorted(times))

    assert ordered == ["plate_1_c.tif", "plate_1_a.tif", "plate_1_b.tif"]
    assert capture_times == times


CAMERA_1 = os.path.join("example_data", "Camera_1", "RPV_13_0.jpg")


@pytest.mark.skipif(not os.path.exists(CAMERA_1),
                    reason="example_data/Camera_1 is not tracked in git; local-only fixture")
def test_exif_sub_ifd_is_read_from_a_real_raspberry_pi_jpeg():
    """The actual root cause, pinned against a real file: this JPEG keeps
    DateTimeOriginal/DateTimeDigitized/DateTime in the Exif sub-IFD, with IFD0
    holding nothing but the pointer to it. Image.getexif() exposes IFD0 only, so
    every version of this code before the sub-IFD read found nothing here."""
    assert get_capture_time(CAMERA_1) == datetime.datetime(2026, 7, 8, 15, 22, 44)


def test_order_series_falls_back_to_filename_key_without_timestamps(tmp_path):
    """The F1_Plate_2_YS case: no timestamps at all, so ordering falls back to
    the filename -- and digit-less names must not be dropped as they once were."""
    for name in ("frame_2.tif", "frame_10.tif", "overview.tif"):
        _write_tiff(tmp_path / name)

    ordered, capture_times = order_series(
        str(tmp_path), ["overview.tif", "frame_10.tif", "frame_2.tif"])

    assert ordered == ["frame_2.tif", "frame_10.tif", "overview.tif"]
    assert all(t is None for t in capture_times.values())


def test_identical_timestamps_count_as_gaps():
    """A whole folder stamped with one time passes a monotonic check but is not
    a usable axis -- it used to silently put every frame at elapsed 0."""
    same = datetime.datetime(2024, 1, 1, 12, 0)

    assert metadata_has_gaps(_metadata(same, same, same)) is True


def test_missing_and_backwards_timestamps_count_as_gaps():
    t0 = datetime.datetime(2024, 1, 1, 12, 0)
    t1 = datetime.datetime(2024, 1, 1, 12, 30)

    assert metadata_has_gaps(_metadata(t0, None, t1)) is True
    assert metadata_has_gaps(_metadata(t1, t0)) is True
    assert metadata_has_gaps(_metadata(t0, t1)) is False


def test_untrusted_timestamps_are_replaced_by_the_interval_not_mixed_with_it():
    """The bug this pass fixed: timestamps that were present but untrustworthy
    kept being used, silently discarding the interval the user was just asked
    for -- so the prompt did nothing in exactly the case that triggers it."""
    same = datetime.datetime(2024, 1, 1, 12, 0)
    metadata = _metadata(same, same, same)
    assert metadata_has_gaps(metadata) is True

    deltas = compute_time_deltas(metadata, fallback_interval_minutes=20.0,
                                 trust_timestamps=False)

    assert [d["elapsed_minutes"] for d in deltas] == [0.0, 20.0, 40.0]


def test_elapsed_minutes_are_never_negative_for_an_unsorted_list():
    metadata = _metadata(datetime.datetime(2024, 1, 1, 12, 30),
                         datetime.datetime(2024, 1, 1, 12, 0))

    deltas = compute_time_deltas(metadata)

    assert [d["elapsed_minutes"] for d in deltas] == [30.0, 0.0]


def test_real_capture_times_are_used_when_trustworthy():
    """Camera_1's shape: hourly frames with one genuine multi-hour gap, which
    must survive as a real gap on the axis rather than being evened out."""
    base = datetime.datetime(2026, 7, 8, 15, 22, 44)
    metadata = _metadata(base,
                         base + datetime.timedelta(minutes=61),
                         base + datetime.timedelta(minutes=61 + 1110))
    assert metadata_has_gaps(metadata) is False

    deltas = compute_time_deltas(metadata)

    assert [d["elapsed_minutes"] for d in deltas] == [0.0, 61.0, 1171.0]


def test_cancelled_prompt_leaves_elapsed_undetermined():
    metadata = _metadata(None)

    deltas = compute_time_deltas(metadata, fallback_interval_minutes=None,
                                 trust_timestamps=False)

    assert deltas[0]["elapsed_minutes"] is None
