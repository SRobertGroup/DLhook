import logging
import os
import re
import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.util import img_as_float
from PIL import Image, ImageOps, ImageEnhance
import datetime

log = logging.getLogger(__name__)



"""
This class preprocess the images which will be passed on as input to the deep learning model
The preprocessing is applied to reduce noise and sharpen the edges of the seedlings
"""
class preprocess_images():
    def __init__(self):
        pass
    def preprocess(self,img_x):
        height, width, _= img_x.shape
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2GRAY)
        im1 = cv2.medianBlur(img_x, 3)
        im1=Image.fromarray(im1)
        enh = ImageEnhance.Contrast(im1)
        im2=enh.enhance(1.2)
        im1=np.array(im2)
        norm_img=np.zeros((height,width))
        final=cv2.normalize(im1, norm_img, 0, 255, cv2.NORM_MINMAX)
        return final

# Tag numbers, used instead of ExifTags.TAGS name lookups because the tag a
# camera actually writes the capture time into differs by container: the Exif
# sub-IFD (pointed to by 0x8769) holds DateTimeOriginal/DateTimeDigitized, while
# IFD0 -- the only thing Image.getexif() itself exposes -- holds a plain
# DateTime. Looking for the name "DateTimeOriginal" in getexif()'s own items (as
# this module used to) therefore essentially never matches on a real JPEG.
EXIF_IFD_POINTER = 0x8769
DATETIME_ORIGINAL = 0x9003
DATETIME_DIGITIZED = 0x9004
DATETIME_IFD0 = 0x0132  # also TIFF tag 306


def _parse_exif_datetime(value):
    """Parse an EXIF/TIFF datetime string, or None if it isn't one. Writers
    emit placeholders like "0000:00:00 00:00:00" for "unknown", which must not
    be mistaken for a real (and absurdly early) capture time."""
    if isinstance(value, bytes):
        value = value.decode("ascii", "ignore")
    if not isinstance(value, str):
        return None
    value = value.strip().rstrip("\x00")
    try:
        return datetime.datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def _metadata_capture_times(path):
    """Every capture timestamp embedded in the file itself, as a list (possibly
    empty) of datetimes -- Exif sub-IFD DateTimeOriginal/DateTimeDigitized,
    IFD0 DateTime, and, for TIFFs, tag 306 (which is not reachable through
    getexif() on every Pillow/TIFF combination)."""
    candidates = []
    try:
        with Image.open(path) as image:
            exif_data = image.getexif()
            if exif_data:
                # DATETIME_IFD0 is looked for in BOTH IFDs, not just IFD0:
                # example_data/Camera_1's rpicam-apps JPEGs put Make/Model/
                # Software/DateTime inside the Exif sub-IFD, leaving IFD0 holding
                # nothing but the pointer to it.
                sub_ifd = exif_data.get_ifd(EXIF_IFD_POINTER) or {}
                raw_values = [
                    sub_ifd.get(DATETIME_ORIGINAL),
                    sub_ifd.get(DATETIME_DIGITIZED),
                    sub_ifd.get(DATETIME_IFD0),
                    exif_data.get(DATETIME_IFD0),
                ]
            else:
                raw_values = []

            if image.format == "TIFF":
                raw_values.append(getattr(image, "tag_v2", {}).get(DATETIME_IFD0))

        for raw in raw_values:
            # TIFF ASCII tags come back as a 1-tuple of strings.
            if isinstance(raw, (tuple, list)):
                raw = raw[0] if raw else None
            parsed = _parse_exif_datetime(raw)
            if parsed is not None:
                candidates.append(parsed)
    except Exception as e:
        log.warning(f"Failed to read embedded timestamps for {path}: {e}")

    return candidates


def get_capture_time(path):
    """The image's capture time: the earliest timestamp embedded in the file
    itself, or None if it carries none.

    THERE IS NO FILESYSTEM FALLBACK, deliberately. st_ctime/st_mtime record when
    this *copy of the file* was written, not when the image was taken, and both
    real datasets in example_data/ prove how badly that misleads: all 168
    Camera_1 JPEGs share ONE filesystem timestamp, and the 50 F1_Plate_2_YS
    TIFFs share TWO (not in frame order). An elapsed-time axis built from those
    silently collapses to a single point, which is worse than having no axis --
    at least None is detectable, and lets the caller ask the user for the frame
    interval instead (see metadata_has_gaps/compute_time_deltas)."""
    embedded = _metadata_capture_times(path)
    return min(embedded) if embedded else None


_DIGIT_RUNS = re.compile(r"(\d+)")


def filename_sort_key(file_name):
    """Natural ordering key: every run of digits in the basename compared as a number, so
    "RPV_16_2.jpg" sorts before "RPV_16_10.jpg" and "F1 Plate 30009.tif" before "F1 Plate 30010.tif".
    Names without digits sort after the numbered ones. (It used to key on the FIRST run of digits
    only, which is the same for every frame of "RPV_16_<n>.jpg", so the series fell back to
    the text order 0, 1, 10, 100, 101, ...)"""
    parts = _DIGIT_RUNS.split(os.path.splitext(file_name)[0])
    return (len(parts) == 1, tuple(int(p) if i % 2 else p.lower() for i, p in enumerate(parts)))


def _interpolate_missing_times(ordered, capture_times):
    """Fill the missing capture times of a series whose other timestamps increase along `ordered`:
    linear between the nearest timed neighbours, and the median step beyond the first / last one."""
    timed = [i for i, f in enumerate(ordered) if capture_times[f] is not None]
    steps = [(capture_times[ordered[b]] - capture_times[ordered[a]]) / (b - a) for a, b in zip(timed, timed[1:])]
    step = sorted(steps)[len(steps) // 2]
    filled = dict(capture_times)
    for i, f in enumerate(ordered):
        if filled[f] is not None:
            continue
        before = max((k for k in timed if k < i), default=None)
        after = min((k for k in timed if k > i), default=None)
        if before is not None and after is not None:
            t0, t1 = capture_times[ordered[before]], capture_times[ordered[after]]
            filled[f] = t0 + (t1 - t0) * ((i - before) / (after - before))
        elif before is not None:
            filled[f] = capture_times[ordered[before]] + step * (i - before)
        else:
            filled[f] = capture_times[ordered[after]] - step * (after - i)
    return filled


def order_series(path, filenames):
    """
    Put a time series into acquisition order, returning
    (ordered_filenames, {filename: capture_time or None}).

    ORDER comes from the timestamps embedded in the files themselves, when every
    frame has one and they aren't all identical. Filesystem times are never used
    (see get_capture_time). Otherwise the order is the natural order of the file
    names (filename_sort_key: every number in the name counts).

    A few frames without a timestamp -- typically a corrupt file, which also cannot
    be read -- no longer throw the others away: when the frames that do have one
    increase along the file-name order, that order is kept and the missing times
    are interpolated from the neighbours (logged as a warning). Before, a single
    unreadable frame sent the whole series to the file-name fallback, which for
    names such as "RPV_16_<n>.jpg" was the text order 0, 1, 10, 100, ...
    """
    capture_times = {f: get_capture_time(os.path.join(path, f)) for f in filenames}

    times = list(capture_times.values())
    usable = all(t is not None for t in times) and len(set(times)) > 1
    if usable:
        return sorted(filenames, key=lambda f: (capture_times[f], f)), capture_times

    ordered = sorted(filenames, key=lambda f: (filename_sort_key(f), f))
    timed = [capture_times[f] for f in ordered if capture_times[f] is not None]
    missing = [f for f in ordered if capture_times[f] is None]
    if missing and len(timed) >= 2 and all(b > a for a, b in zip(timed, timed[1:])):
        capture_times = _interpolate_missing_times(ordered, capture_times)
        log.warning(f"{len(missing)} frame(s) carry no capture time ({', '.join(missing[:5])}"
                    f"{', ...' if len(missing) > 5 else ''}); their time is interpolated from the neighbours")
    return ordered, capture_times


def metadata_has_gaps(metadata_list):
    """
    True if the timestamps can't be trusted to describe the real acquisition
    times: any frame missing one, or the series not STRICTLY increasing.

    Strictly, not merely non-decreasing: two frames of a growth time series
    cannot share a capture instant, so duplicates mean the timestamps aren't
    per-frame capture times at all. Requiring only non-decreasing used to let
    the worst case through silently -- a whole folder stamped with one identical
    time passes a monotonic check, then every frame lands at elapsed_minutes=0
    and the kinematics x-axis collapses to a single point with no warning.
    """
    times = [m.get("creation_time") for m in metadata_list]
    if any(t is None for t in times):
        return True
    return any(b <= a for a, b in zip(times, times[1:]))


def compute_time_deltas(metadata_list, fallback_interval_minutes=None, trust_timestamps=True):
    """
    Convert per-image timestamps into elapsed minutes since the earliest valid
    timestamp in the series. If an entry's own timestamp is missing and
    fallback_interval_minutes is given, its elapsed time is estimated instead
    as index * fallback_interval_minutes.

    Pass trust_timestamps=False (i.e. metadata_has_gaps said so) to ignore the
    timestamps entirely and build the whole axis from the interval. Without
    that, a series whose timestamps are present but untrustworthy -- all
    identical, say -- kept using them and silently discarded the interval the
    user had just been prompted for, which is precisely the case the prompt
    exists to rescue.

    Returns a new list of dicts (input dicts are not mutated), each augmented
    with an "elapsed_minutes" key (float, or None if it can't be determined).
    """
    # Anchored on the earliest timestamp, not the first one in list order: those
    # coincide for a time-sorted series, but anchoring on list order would emit
    # negative elapsed times for any caller that passes an unsorted list.
    if trust_timestamps:
        valid_times = [m["creation_time"] for m in metadata_list if m.get("creation_time") is not None]
        origin = min(valid_times) if valid_times else None
    else:
        origin = None

    result = []
    for idx, entry in enumerate(metadata_list):
        entry = dict(entry)
        ts = entry.get("creation_time") if trust_timestamps else None
        if ts is not None and origin is not None:
            entry["elapsed_minutes"] = (ts - origin).total_seconds() / 60
        elif fallback_interval_minutes is not None:
            entry["elapsed_minutes"] = idx * fallback_interval_minutes
        else:
            entry["elapsed_minutes"] = None
        result.append(entry)
    return result