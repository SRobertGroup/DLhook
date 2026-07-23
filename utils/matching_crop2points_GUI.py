
class MatchCropPoints:
    """
    Converts each seedling's start point (image coordinates) into its position
    relative to that seedling's own crop box, rescaled to the fixed working
    size the segmentation pipeline pads crops to. Crop boxes are 1:1 with
    seedlings by construction (Phase 2), so no spatial bucketing is needed --
    box i and start point i simply belong to the same seedling.
    """

    def __init__(self, crop_boxes, seedling_start_points, working_size=1024):
        self.working_size = working_size
        self.relative_points = []

        for box, (px, py) in zip(crop_boxes, seedling_start_points):
            x1 = box["cx"] - box["half_w"]
            y1 = box["cy"] - box["half_h"]
            scale_x = self.working_size / (2 * box["half_w"])
            scale_y = self.working_size / (2 * box["half_h"])

            rx = (px - x1) * scale_x
            ry = (py - y1) * scale_y
            self.relative_points.append([(int(rx), int(ry))])

    def return_crop_points(self):
        """Returns one relative seedling point (as a single-element list) per crop box."""
        return self.relative_points
