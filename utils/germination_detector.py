import cv2
import numpy as np


def _area_near_point(contours, point, proximity_radius):
    """Total area of contours whose centroid falls within proximity_radius of
    point. No minimum vertex count: that floor exists in the angle path only
    because cv2.fitEllipse needs 5 points, and applying it here discarded small
    but real emerging-radicle blobs."""
    px, py = point
    total_area = 0.0
    for contour in contours:
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


# Fraction of a seedling's own crop size used as the search radius around its
# seed coat point. Raised from 0.12 after measuring the real dumped germ masks
# (example_data/F1_Plate_2_YS/segmented_s*/): the germinating blob's centroid
# sits 18-39 px from the seed point on a ~220 px crop, and 0.12 (26 px) excluded
# the farther ones outright. 0.18 (~40 px) covers all of them.
PROXIMITY_RADIUS_FRACTION = 0.18

# Fraction of that search disk's area which counts as "germination is visible
# here". Lowered from 0.05 after the same measurement, and this was the main
# reason germination never fired: at 0.05 the threshold works out to ~90-109 px,
# which is the same order as the ENTIRE germ mask anywhere in the crop (measured
# maxima: 17, 389, 230, 296 px), so it could essentially never be met near the
# seed. At 0.01 it is ~50 px on a 220 px crop -- comfortably above segmentation
# speckle, and low enough to fire on the real data.
AREA_THRESHOLD_FRACTION = 0.01

# Germination is confirmed when at least CONFIRM_HITS of the CONFIRM_WINDOW
# frames starting at a candidate frame clear the threshold. This replaces a
# strict "two consecutive frames" rule: the germ_v1 mask is present in only
# ~40-55% of frames on real data (19, 27 and 23 of 50 for the three seedlings
# that have any), so consecutive hits are often unavailable even where
# germination is plainly visible. Still requires more than one frame, so a lone
# speckle is not mistaken for germination.
CONFIRM_WINDOW = 3
CONFIRM_HITS = 2


class GerminationDetector:
    """
    Detects each seedling's germination time-zero from the germ_v1 mask's
    time series: the first frame at which the germ mask shows an
    area-near-seed-coat at or above `area_threshold`, confirmed by
    CONFIRM_HITS-of-CONFIRM_WINDOW frames (so a single-frame blip is treated as
    segmentation noise, not germination).

    `proximity_radius`/`area_threshold` are expressed in the same pixel space
    as the seed coat point and germ contours -- each seedling's own crop
    keeps its native, box-derived pixel size (crops are not resized to a
    shared fixed working size), so by default these scale per seedling from
    the `crop_size` passed into detect() rather than a single hardcoded
    constant.

    SENSITIVITY, AND WHAT IT CAN'T FIX. The defaults above were chosen by
    sweeping radius/threshold/confirmation against the four real seedlings whose
    germ masks are dumped under example_data/F1_Plate_2_YS/: three of them then
    detect at frames 8-11, stable across a plateau of nearby settings, where the
    previous defaults detected in one of four. The fourth cannot be fixed by any
    setting -- germ_v1 produced almost no germ mask for it at all (17 px total
    across 50 frames), which is a segmentation limitation, not a threshold one.
    detect() reports that case distinctly (see describe()) so it is not confused
    with "the threshold was not met"; the user's manual override
    (set_override, wired to SeedlingAnalysisWindow) is the answer for it.

    There is still NO ground-truth germination timing to validate the chosen
    frame against -- only that a frame is now found, and that the choice is
    insensitive to the exact settings. Treat the detected frame as a starting
    point the user can correct, not an answer.
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
        # Per-seedling record of what detect() actually measured, so a wrong
        # time-zero can be diagnosed (and the thresholds above calibrated)
        # from real numbers instead of guesswork. See describe().
        self.diagnostics = {}

    def detect(self, seedling_id, germ_contours_by_frame, seed_point, crop_size):
        """
        germ_contours_by_frame: list of contour-lists, one per frame, in
        time order, for this seedling only.
        seed_point: (x, y) seed coat location for this seedling.
        crop_size: this seedling's own crop's pixel size (e.g. average of its
        width/height), used to scale proximity_radius/area_threshold to it.

        Stores and returns the detected time-zero frame index, or None if no
        frame is confirmed by the CONFIRM_HITS-of-CONFIRM_WINDOW rule.
        """
        proximity_radius = (self._proximity_radius_override if self._proximity_radius_override is not None
                             else crop_size * PROXIMITY_RADIUS_FRACTION)
        area_threshold = (self._area_threshold_override if self._area_threshold_override is not None
                           else (proximity_radius ** 2) * np.pi * AREA_THRESHOLD_FRACTION)

        areas = [_area_near_point(contours, seed_point, proximity_radius)
                 for contours in germ_contours_by_frame]
        # Total germ area anywhere in the crop, which separates "the model found
        # no germinating tissue at all" from "it found some, but not near the
        # seed / not enough of it" -- two failures with different remedies.
        total_areas = [sum(cv2.contourArea(c) for c in contours)
                       for contours in germ_contours_by_frame]

        n = len(areas)
        frame_idx = None
        for t in range(n):
            if areas[t] < area_threshold:
                continue
            hits = sum(1 for k in range(t, min(n, t + CONFIRM_WINDOW))
                       if areas[k] >= area_threshold)
            if hits >= CONFIRM_HITS:
                frame_idx = t
                break

        self.germination_frame[seedling_id] = frame_idx
        self.diagnostics[seedling_id] = {
            "proximity_radius": proximity_radius,
            "area_threshold": area_threshold,
            "areas": areas,
            "max_area_anywhere": max(total_areas) if total_areas else 0.0,
            "frames_with_any_germ": sum(1 for a in total_areas if a > 0),
            "frame": frame_idx,
        }
        return frame_idx

    def describe(self, seedling_id):
        """One-line summary of the last detect() call for this seedling: the
        thresholds it used, the per-frame near-seed areas it compared against
        them, and how much germ mask existed anywhere in the crop.

        That last number is what distinguishes the two ways this fails. If
        max_area_anywhere is ~0 the model segmented no germinating tissue and no
        threshold can help (use the manual override). If it is substantial but
        the near-seed areas are not, the blob is being found away from the seed
        point, or the threshold is too high for it."""
        diag = self.diagnostics.get(seedling_id)
        if diag is None:
            return f"seedling {seedling_id}: never detected"
        areas = ", ".join(f"{a:.0f}" for a in diag["areas"])
        verdict = ""
        if diag["frame"] is None:
            if diag["max_area_anywhere"] < diag["area_threshold"]:
                verdict = "  -> NO GERM MASK to detect (segmentation, not threshold)"
            else:
                verdict = "  -> germ mask exists but not near the seed / below threshold"
        return (f"seedling {seedling_id}: frame={diag['frame']} "
                f"radius={diag['proximity_radius']:.1f} "
                f"threshold={diag['area_threshold']:.0f} "
                f"max_germ_anywhere={diag['max_area_anywhere']:.0f} "
                f"frames_with_germ={diag['frames_with_any_germ']}/{len(diag['areas'])} "
                f"areas=[{areas}]{verdict}")

    def set_override(self, seedling_id, frame_idx):
        """Manual user override of a detected (or missing) time-zero frame."""
        self._overrides[seedling_id] = frame_idx

    def clear_override(self, seedling_id):
        """Drop a manual override, falling back to whatever detect() found."""
        self._overrides.pop(seedling_id, None)

    def has_override(self, seedling_id):
        """True if this seedling's time-zero was set by the user rather than detected."""
        return seedling_id in self._overrides

    def get_time_zero(self, seedling_id):
        """User override if set, else the detected frame index (or None)."""
        if seedling_id in self._overrides:
            return self._overrides[seedling_id]
        return self.germination_frame.get(seedling_id)
