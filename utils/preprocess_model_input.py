import os
import re
import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.util import img_as_float
from PIL import Image, ImageOps, ImageEnhance
import datetime



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
                sub_ifd = exif_data.get_ifd(EXIF_IFD_POINTER) or {}
                raw_values = [
                    sub_ifd.get(DATETIME_ORIGINAL),
                    sub_ifd.get(DATETIME_DIGITIZED),
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
        print(f"[WARNING] Failed to read embedded timestamps for {path}: {e}")

    return candidates


def _filesystem_times(path):
    """The file's own creation/modification times, as a list of datetimes."""
    try:
        stat = os.stat(path)
    except OSError as e:
        print(f"[ERROR] Could not stat {path}: {e}")
        return []
    return [datetime.datetime.fromtimestamp(ts) for ts in (stat.st_ctime, stat.st_mtime)]


def get_init_time(path):
    """Fallback when nothing is embedded in the file: the earlier of its
    creation and modification times. (On Windows st_ctime is creation time and
    survives a copy, while st_mtime does not; on Linux st_ctime is inode-change
    time. Taking the minimum is the closest either platform gets to "when this
    image first existed".)"""
    times = _filesystem_times(path)
    return min(times) if times else None


def get_image_creation_time(path):
    """The image's capture time: the earliest timestamp embedded in the file
    itself, falling back to the filesystem only when the file carries none.

    Embedded tags win outright rather than competing with the filesystem times,
    because copying/re-encoding a series resets st_mtime (and, off Windows,
    st_ctime) to the copy date -- letting those compete would mean the earliest
    "time" for a copied series is whichever file the OS happened to touch
    first, which is not acquisition order at all."""
    embedded = _metadata_capture_times(path)
    if embedded:
        return min(embedded)
    return get_init_time(path)


def filename_sort_key(file_name):
    """Legacy ordering key: the first run of digits in the basename, or +inf for
    a name with no digits (which used to be dropped from the series entirely)."""
    number_match = re.search(r'\d+', os.path.splitext(file_name)[0])
    return int(number_match.group()) if number_match else float("inf")


def order_series(path, filenames):
    """
    Put a time series into acquisition order, returning
    (ordered_filenames, {filename: capture_time or None}).

    ORDER comes from timestamps embedded in the files themselves, when every
    frame has one and they aren't all identical. Filesystem times are
    deliberately NOT allowed to decide order: copying or re-encoding a folder
    rewrites them in whatever sequence the OS touched the files, which is not
    acquisition order and gives no warning when it's wrong. Without embedded
    timestamps this falls back to the legacy first-integer-in-the-filename key
    -- also imperfect (every frame of example_data/Camera_1 yields the same key
    13 from "RPV_13_0.jpg", so the sort degenerates to a lexicographic no-op
    that puts "..._100" before "..._2") but at least deliberate.

    The returned capture times still fall back to the filesystem per file, since
    an approximate elapsed-minutes x-axis beats none at all -- and if those turn
    out non-monotonic in the chosen order, metadata_has_gaps() will say so and
    the caller can ask the user for a frame interval instead.
    """
    embedded = {}
    capture_times = {}
    for file_name in filenames:
        file_path = os.path.join(path, file_name)
        candidates = _metadata_capture_times(file_path)
        embedded[file_name] = min(candidates) if candidates else None
        capture_times[file_name] = (embedded[file_name] if embedded[file_name] is not None
                                     else get_init_time(file_path))

    times = list(embedded.values())
    usable = all(t is not None for t in times) and len(set(times)) > 1
    if usable:
        ordered = sorted(filenames, key=lambda f: (embedded[f], f))
    else:
        ordered = sorted(filenames, key=lambda f: (filename_sort_key(f), f))

    return ordered, capture_times


def metadata_has_gaps(metadata_list):
    """
    True if any image is missing a usable creation_time, or if the recovered
    timestamps aren't monotonically non-decreasing (a sign the metadata can't
    be trusted to reflect real acquisition order/spacing).

    The non-monotonic branch stays reachable after order_series(): that only
    sorts by *embedded* timestamps, so a series ordered by filename key can
    still carry filesystem-derived times that jump around -- exactly the case
    where the caller should stop trusting them and ask for a frame interval.
    """
    times = [m.get("creation_time") for m in metadata_list]
    if any(t is None for t in times):
        return True
    return any(b < a for a, b in zip(times, times[1:]))


def compute_time_deltas(metadata_list, fallback_interval_minutes=None):
    """
    Convert per-image timestamps into elapsed minutes since the earliest valid
    timestamp in the series. If an entry's own timestamp is missing and
    fallback_interval_minutes is given, its elapsed time is estimated instead
    as index * fallback_interval_minutes.

    Returns a new list of dicts (input dicts are not mutated), each augmented
    with an "elapsed_minutes" key (float, or None if it can't be determined).
    """
    # Anchored on the earliest timestamp, not the first one in list order: those
    # coincide for a time-sorted series, but anchoring on list order would emit
    # negative elapsed times for any caller that passes an unsorted list.
    valid_times = [m["creation_time"] for m in metadata_list if m.get("creation_time") is not None]
    origin = min(valid_times) if valid_times else None

    result = []
    for idx, entry in enumerate(metadata_list):
        entry = dict(entry)
        ts = entry.get("creation_time")
        if ts is not None and origin is not None:
            entry["elapsed_minutes"] = (ts - origin).total_seconds() / 60
        elif fallback_interval_minutes is not None:
            entry["elapsed_minutes"] = idx * fallback_interval_minutes
        else:
            entry["elapsed_minutes"] = None
        result.append(entry)
    return result