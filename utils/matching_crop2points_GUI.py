
class MatchCropPoints:
    """
    Converts each seedling's start point (image coordinates) into its position
    relative to that seedling's own crop box. Crop boxes are 1:1 with
    seedlings by construction (Phase 2), so no spatial bucketing is needed --
    box i and start point i simply belong to the same seedling. No rescaling
    is applied: each saved crop keeps its box's own native pixel dimensions
    (crops are not resized to any fixed working size), so the point's
    position relative to the box's own origin is exactly its position in the
    saved crop file.
    """

    def __init__(self, crop_boxes, seedling_start_points):
        self.relative_points = []

        for box, (px, py) in zip(crop_boxes, seedling_start_points):
            x1 = box["cx"] - box["half_w"]
            y1 = box["cy"] - box["half_h"]

            rx = px - x1
            ry = py - y1
            self.relative_points.append([(int(round(rx)), int(round(ry)))])

    def return_crop_points(self):
        """Returns one relative seedling point (as a single-element list) per crop box."""
        return self.relative_points
