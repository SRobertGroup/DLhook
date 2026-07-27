import cv2


class MaskEditor:
    """
    Wraps one binary mask (uint8, 0/255) with circular-brush add/erase
    strokes and an undo stack. Undo is per-stroke (one entry per
    begin_stroke()/end_stroke() pair), not per pixel -- matches the app's
    existing Z-key-undoes-last-action convention (Phase 2's point placement).
    """

    def __init__(self, mask, brush_radius=5):
        self.mask = mask.copy()
        self.brush_radius = brush_radius
        self._undo_stack = []
        self._stroke_backup = None

    def set_brush_radius(self, radius):
        self.brush_radius = max(1, int(radius))

    def begin_stroke(self):
        self._stroke_backup = self.mask.copy()

    def paint(self, x, y, add):
        color = 255 if add else 0
        cv2.circle(self.mask, (int(round(x)), int(round(y))), self.brush_radius, color, thickness=-1)

    def end_stroke(self):
        if self._stroke_backup is not None:
            self._undo_stack.append(self._stroke_backup)
            self._stroke_backup = None

    def undo(self):
        if not self._undo_stack:
            return False
        self.mask = self._undo_stack.pop()
        return True

    def get_mask(self):
        return self.mask
