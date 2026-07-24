import cv2
import numpy as np


def _area_near_point(contours, point, proximity_radius):
    """Total area of contours whose centroid falls within proximity_radius of point."""
    px, py = point
    total_area = 0.0
    for contour in contours:
        if len(contour) < 5:
            continue
        area = cv2.contourArea(contour)
        if area <= 0:
            continue
        m = cv2.moments(contour)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        if np.hypot(cx - px, cy - py) <= proximity_radius:
            total_area += area
    return total_area


class GerminationDetector:
    """
    Detects each seedling's germination time-zero from the germ_v1 mask's
    time series: the first frame at which the germ mask shows an
    area-near-seed-coat at or above `area_threshold`, for two consecutive
    frames in a row (a single-frame blip is treated as segmentation noise,
    not germination).

    `proximity_radius`/`area_threshold` are expressed in the same pixel space
    as the seed coat point and germ contours -- each seedling's own crop
    keeps its native, box-derived pixel size (crops are not resized to a
    shared fixed working size), so by default these scale per seedling from
    the `crop_size` passed into detect() rather than a single hardcoded
    constant. There's no ground-truth germination timing available to
    calibrate exact defaults against either way, so these remain tunable.
    """

    def __init__(self, proximity_radius=None, area_threshold=None):
        # Overrides, if given, apply to every seedling regardless of its own
        # crop size. Otherwise detect() derives them per call from that
        # seedling's own crop_size, since crops are no longer resized to a
        # shared fixed working size (each seedling's crop keeps its own
        # box-derived pixel dimensions).
        self._proximity_radius_override = proximity_radius
        self._area_threshold_override = area_threshold

        self.germination_frame = {}
        self._overrides = {}

    def detect(self, seedling_id, germ_contours_by_frame, seed_point, crop_size):
        """
        germ_contours_by_frame: list of contour-lists, one per frame, in
        time order, for this seedling only.
        seed_point: (x, y) seed coat location for this seedling.
        crop_size: this seedling's own crop's pixel size (e.g. average of its
        width/height), used to scale proximity_radius/area_threshold to it.

        Stores and returns the detected time-zero frame index, or None if
        the threshold is never met for two consecutive frames.
        """
        proximity_radius = (self._proximity_radius_override if self._proximity_radius_override is not None
                             else crop_size * 0.12)
        area_threshold = (self._area_threshold_override if self._area_threshold_override is not None
                           else (proximity_radius ** 2) * np.pi * 0.05)

        areas = [_area_near_point(contours, seed_point, proximity_radius)
                 for contours in germ_contours_by_frame]

        frame_idx = None
        for t in range(len(areas) - 1):
            if areas[t] >= area_threshold and areas[t + 1] >= area_threshold:
                frame_idx = t
                break

        self.germination_frame[seedling_id] = frame_idx
        return frame_idx

    def set_override(self, seedling_id, frame_idx):
        """Manual user override of a detected (or missing) time-zero frame."""
        self._overrides[seedling_id] = frame_idx

    def get_time_zero(self, seedling_id):
        """User override if set, else the detected frame index (or None)."""
        if seedling_id in self._overrides:
            return self._overrides[seedling_id]
        return self.germination_frame.get(seedling_id)
