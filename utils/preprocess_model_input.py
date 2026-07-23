import os
import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.util import img_as_float
from PIL import Image, ImageOps, ImageEnhance
from PIL.ExifTags import TAGS
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

def get_init_time(path):
    """Fallback: Get file modification time if EXIF is unavailable."""
    try:
        ts = os.path.getctime(path)
        return datetime.datetime.fromtimestamp(ts)
    except Exception as e:
        print(f"[ERROR] Could not read modification time for {path}: {e}")
        return None

def get_image_creation_time(path):
    """Try to get EXIF creation time, else fallback to file modification time."""
    try:
        image = Image.open(path)
        exif_data = image.getexif()

        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "DateTimeOriginal":
                    try:
                        return datetime.datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                    except ValueError as ve:
                        print(f"[WARNING] DateTimeOriginal format issue for {path}: {ve}")
                        break

    except Exception as e:
        print(f"[WARNING] Failed to get image creation time for {path}: {e}")

    return get_init_time(path)


def metadata_has_gaps(metadata_list):
    """
    True if any image is missing a usable creation_time, or if the recovered
    timestamps aren't monotonically non-decreasing (a sign the metadata can't
    be trusted to reflect real acquisition order/spacing).
    """
    times = [m.get("creation_time") for m in metadata_list]
    if any(t is None for t in times):
        return True
    return any(b < a for a, b in zip(times, times[1:]))


def compute_time_deltas(metadata_list, fallback_interval_minutes=None):
    """
    Convert per-image timestamps into elapsed minutes since the first valid
    timestamp in the series. If an entry's own timestamp is missing and
    fallback_interval_minutes is given, its elapsed time is estimated instead
    as index * fallback_interval_minutes.

    Returns a new list of dicts (input dicts are not mutated), each augmented
    with an "elapsed_minutes" key (float, or None if it can't be determined).
    """
    first_valid = next(
        (m["creation_time"] for m in metadata_list if m.get("creation_time") is not None),
        None,
    )

    result = []
    for idx, entry in enumerate(metadata_list):
        entry = dict(entry)
        ts = entry.get("creation_time")
        if ts is not None and first_valid is not None:
            entry["elapsed_minutes"] = (ts - first_valid).total_seconds() / 60
        elif fallback_interval_minutes is not None:
            entry["elapsed_minutes"] = idx * fallback_interval_minutes
        else:
            entry["elapsed_minutes"] = None
        result.append(entry)
    return result