import cv2


class MaskEditor:
    """
    Wraps one binary mask (uint8, 0/255) with circular-brush add/erase
    strokes and an undo stack. Undo is per-stroke (one entry per
    begin_stroke()/end_stroke() pair), not per pixel -- matches the app's
    existing Z-key-undoes-last-action convention (Phase 2's point placement).

    A stroke is a begin_stroke() / paint()* / end_stroke() sequence; paint()
    joins consecutive positions within one stroke, so callers can feed it raw
    mouse-motion positions without leaving gaps between them.
    """

    def __init__(self, mask, brush_radius=5):
        self.mask = mask.copy()
        self.brush_radius = brush_radius
        self._undo_stack = []
        self._stroke_backup = None
        # Previous position within the current stroke, so consecutive positions
        # can be joined instead of stamped in isolation. See paint().
        self._last_point = None

    def set_brush_radius(self, radius):
        self.brush_radius = max(1, int(radius))

    def begin_stroke(self):
        self._stroke_backup = self.mask.copy()
        self._last_point = None

    def paint(self, x, y, add):
        """Paints (or erases) at (x, y), joined to the previous position in this
        stroke by a line of the same width.

        The join is what makes a stroke continuous: the caller only gets a
        <B1-Motion> event every few tens of milliseconds, so a quick drag jumps
        tens of pixels between events. Stamping one circle per event left gaps
        along the path -- barely noticeable while adding, but in erase mode the
        skipped pixels stay behind as an unerased trace following the cursor.
        """
        color = 255 if add else 0
        point = (int(round(x)), int(round(y)))
        if self._last_point is not None and self._last_point != point:
            # Round caps from the circles at each end; cv2.line's own caps are flat.
            cv2.line(self.mask, self._last_point, point, color, thickness=2 * self.brush_radius)
        cv2.circle(self.mask, point, self.brush_radius, color, thickness=-1)
        self._last_point = point

    def end_stroke(self):
        self._last_point = None
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
