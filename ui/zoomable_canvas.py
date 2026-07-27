import tkinter as tk

import cv2
from PIL import Image, ImageTk

MAX_ZOOM_MULTIPLIER = 8
ZOOM_STEP = 1.2


class ZoomableImageCanvas(tk.Canvas):
    """
    A tk.Canvas that displays a numpy BGR image with mouse-wheel zoom,
    centered on the cursor, and canvas<->image coordinate mapping that
    accounts for the current zoom level -- generalizes the app's existing
    fixed-scale canvas_to_image_coords helper so Phase 7's brush strokes land
    on the right mask pixel regardless of zoom.

    No click-drag pan gesture by design: minimum zoom always shows the whole
    image (like "fit to window"), so there's no off-image blank space to pan
    into, and left-click-drag is reserved entirely for brush painting.
    """

    def __init__(self, parent, width=600, height=600, **kwargs):
        super().__init__(parent, width=width, height=height, bg="#FFFFFF", **kwargs)
        self.display_width = width
        self.display_height = height

        self._image = None
        self._photo = None
        self._base_zoom = 1.0
        self.zoom = 1.0
        self._center_x = 0
        self._center_y = 0

        # Brush preview circle, in image-space (so it stays correctly placed
        # across zoom changes) -- (img_x, img_y, img_radius, outline_color) or
        # None when hidden. Stored rather than only drawn on the spot because
        # _redraw() does delete("all") on every zoom/frame change and must
        # re-draw it after the image to keep it on top.
        self._brush_preview = None

        self.bind("<MouseWheel>", self._on_mousewheel)  # Windows/macOS
        self.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, ZOOM_STEP))  # Linux scroll up
        self.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 1 / ZOOM_STEP))  # Linux scroll down

    def set_image(self, image):
        """image: HxWx3 BGR numpy array, in the same pixel space as any
        overlay/mask the caller wants to paint or map coordinates against.

        Zoom/center are only initialized on the very first image, then left
        alone on every later call -- so navigating between a seedling's
        frames (all the same size) keeps whatever region the user zoomed
        into, instead of resetting to fit-to-window on every frame change."""
        is_first_image = self._image is None
        h, w = image.shape[:2]
        self._image = image

        if is_first_image:
            self._base_zoom = min(self.display_width / w, self.display_height / h)
            self.zoom = self._base_zoom
            self._center_x = w / 2
            self._center_y = h / 2

        self._redraw()

    def zoom_in(self):
        self._zoom_at(self.display_width / 2, self.display_height / 2, ZOOM_STEP)

    def zoom_out(self):
        self._zoom_at(self.display_width / 2, self.display_height / 2, 1 / ZOOM_STEP)

    def reset_zoom(self):
        if self._image is None:
            return
        h, w = self._image.shape[:2]
        self._base_zoom = min(self.display_width / w, self.display_height / h)
        self.zoom = self._base_zoom
        self._center_x = w / 2
        self._center_y = h / 2
        self._redraw()

    def set_brush_preview(self, img_x, img_y, img_radius, outline="#ffff00"):
        """Shows/moves the brush-size preview circle, in image-space
        coordinates so it tracks the cursor correctly at any zoom level."""
        self._brush_preview = (img_x, img_y, img_radius, outline)
        self._draw_brush_preview()

    def clear_brush_preview(self):
        self._brush_preview = None
        self.delete("brush_preview")

    def _draw_brush_preview(self):
        self.delete("brush_preview")
        if self._brush_preview is None or self._image is None:
            return
        img_x, img_y, img_radius, outline = self._brush_preview
        x1, y1, _, _ = self._visible_bounds()
        cx = (img_x - x1) * self.zoom
        cy = (img_y - y1) * self.zoom
        r = max(1.0, img_radius * self.zoom)
        self.create_oval(cx - r, cy - r, cx + r, cy + r,
                          outline=outline, width=2, tags="brush_preview")

    def canvas_to_image(self, canvas_x, canvas_y):
        """Maps a canvas pixel coordinate to the image pixel coordinate
        currently shown there -- Phase 7's brush uses this to know which
        mask pixel the cursor is over. Uses the uniform zoom factor on both
        axes (matching _redraw's letterboxed draw), not a per-axis stretch,
        so this stays correct for non-square images/crops."""
        x1, y1, _, _ = self._visible_bounds()
        return x1 + canvas_x / self.zoom, y1 + canvas_y / self.zoom

    def _on_mousewheel(self, event):
        factor = ZOOM_STEP if event.delta > 0 else 1 / ZOOM_STEP
        self._zoom_at(event.x, event.y, factor)

    def _zoom_at(self, canvas_x, canvas_y, factor):
        if self._image is None:
            return

        img_x, img_y = self.canvas_to_image(canvas_x, canvas_y)

        max_zoom = self._base_zoom * MAX_ZOOM_MULTIPLIER
        new_zoom = min(max(self.zoom * factor, self._base_zoom), max_zoom)
        if new_zoom == self.zoom:
            return
        self.zoom = new_zoom
        self._center_x = img_x
        self._center_y = img_y
        self._clamp_center()
        self._redraw()

    def _clamp_center(self):
        h, w = self._image.shape[:2]
        half_view_w = (self.display_width / 2) / self.zoom
        half_view_h = (self.display_height / 2) / self.zoom

        if w > 2 * half_view_w:
            self._center_x = min(max(self._center_x, half_view_w), w - half_view_w)
        else:
            self._center_x = w / 2

        if h > 2 * half_view_h:
            self._center_y = min(max(self._center_y, half_view_h), h - half_view_h)
        else:
            self._center_y = h / 2

    def _visible_bounds(self):
        """Image-space (x1, y1, x2, y2) of the region currently shown."""
        half_view_w = (self.display_width / 2) / self.zoom
        half_view_h = (self.display_height / 2) / self.zoom
        return (self._center_x - half_view_w, self._center_y - half_view_h,
                self._center_x + half_view_w, self._center_y + half_view_h)

    def _redraw(self):
        if self._image is None:
            return

        h, w = self._image.shape[:2]
        x1, y1, x2, y2 = self._visible_bounds()
        ix1, iy1 = max(0, int(round(x1))), max(0, int(round(y1)))
        ix2, iy2 = min(w, int(round(x2))), min(h, int(round(y2)))
        crop = self._image[iy1:iy2, ix1:ix2]
        if crop.size == 0:
            return

        # Resize using the actual zoom factor on both axes -- not a stretch
        # to fill the canvas -- so a non-square crop/image is never
        # distorted. Whatever clipping happened above the image's real
        # edges just leaves a blank margin (drawn at an offset) instead of
        # being stretched back out to fill the square canvas.
        out_w = max(1, int(round(crop.shape[1] * self.zoom)))
        out_h = max(1, int(round(crop.shape[0] * self.zoom)))
        resized = cv2.resize(crop, (out_w, out_h))
        offset_x = (ix1 - x1) * self.zoom
        offset_y = (iy1 - y1) * self.zoom

        # No BGR->RGB conversion, matching the rest of the app's existing
        # (uncorrected) display convention.
        self._photo = ImageTk.PhotoImage(image=Image.fromarray(resized))
        self.delete("all")
        self.create_image(offset_x, offset_y, image=self._photo, anchor=tk.NW)
        self._draw_brush_preview()
