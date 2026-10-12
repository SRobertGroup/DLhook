import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

MAX_ZOOM_MULTIPLIER = 8
ZOOM_STEP = 1.2

# Manual angle markers are drawn at a fixed size in SCREEN pixels, so they stay
# the same crisp dot/line at every zoom level instead of growing with the image.
ANGLE_MARKER_RADIUS_PX = 3
ANGLE_MARKER_WIDTH_PX = 2
POLYLINE_WIDTH_PX = 2
CONTOUR_WIDTH_PX = 2


def _hex_to_rgb(color):
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


class ZoomableImageCanvas(tk.Canvas):
    """
    A tk.Canvas that displays a numpy BGR image with mouse-wheel zoom,
    centered on the cursor, and canvas<->image coordinate mapping that
    accounts for the current zoom level -- generalizes the app's existing
    fixed-scale canvas_to_image_coords helper so clicks land on the right
    image pixel regardless of zoom.

    No click-drag pan gesture by design: minimum zoom always shows the whole
    image (like "fit to window"), so there's no off-image blank space to pan
    into, and left click is reserved for placing landmarks.
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

        # Manual angle markers, same idea: image-space points + a color, drawn
        # as canvas items rather than burned into the image, so they don't get
        # magnified into blobs by the zoom upscale.

        # Generic image-space overlay (points / lines / arrows), see set_overlay.
        self._overlay = []

        self.bind("<MouseWheel>", self._on_mousewheel)  # Windows/macOS
        self.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, ZOOM_STEP))  # Linux scroll up
        self.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 1 / ZOOM_STEP))  # Linux scroll down

    def set_image(self, image, reset_view=False, overlay=None):
        """image: HxWx3 BGR numpy array, in the same pixel space as any
        overlay/mask the caller wants to paint or map coordinates against.

        Zoom/center are only initialized on the very first image, then left
        alone on every later call -- so navigating between a seedling's
        frames (all the same size) keeps whatever region the user zoomed
        into, instead of resetting to fit-to-window on every frame change.
        reset_view=True forces a fresh fit-to-window, for a caller switching
        to an image of a different size."""
        if overlay is not None:
            self._overlay = list(overlay)       # drawn by the _redraw below, not a second one
        if overlay is not None:
            self._overlay = list(overlay)       # drawn by the _redraw below, not a second one
        if overlay is not None:
            self._overlay = list(overlay)       # drawn by the _redraw below, not a second one
        if overlay is not None:
            self._overlay = list(overlay)       # drawn by the _redraw below, not a second one
        if overlay is not None:
            self._overlay = list(overlay)       # drawn by the _redraw below, not a second one
        if overlay is not None:
            self._overlay = list(overlay)       # drawn by the _redraw below, not a second one
        if overlay is not None:
            self._overlay = list(overlay)       # drawn by the _redraw below, not a second one
        if overlay is not None:
            self._overlay = list(overlay)       # drawn by the _redraw below, not a second one
        if overlay is not None:
            self._overlay = list(overlay)       # drawn by the _redraw below, not a second one
        if overlay is not None:
            self._overlay = list(overlay)       # drawn by the _redraw below, not a second one
        if overlay is not None:
            self._overlay = list(overlay)       # drawn by the _redraw below, not a second one
        is_first_image = self._image is None or reset_view
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

    def set_overlay(self, shapes):
        """Image-space annotation overlay, drawn as fixed-screen-size canvas
        items (crisp at any zoom, like the angle markers). `shapes` is a list
        of dicts:
            {"kind": "point", "xy": (x, y), "color": c, "label": "J"}
            {"kind": "line", "from": (x, y), "to": (x, y), "color": c,
             "arrow": bool, "dash": bool}
            {"kind": "polyline", "pts": [(x, y), ...], "color": c}   (fixed screen width)
            {"kind": "contour", "pts": [(x, y), ...], "color": c, "alpha": 0.5}
        A "contour" is alpha-blended into the displayed (zoomed) image along its own pixels only
        (Tk canvas lines have no alpha), so it stays 1-2 screen px wide at any zoom and the raw
        image is untouched elsewhere."""
        had_contour = any(s["kind"] == "contour" for s in self._overlay)
        self._overlay = list(shapes)
        if had_contour or any(s["kind"] == "contour" for s in self._overlay):
            self._redraw()          # contours are blended into the image itself
        else:
            self._draw_overlay()

    def _draw_overlay(self):
        self.delete("overlay")
        if not self._overlay or self._image is None:
            return
        x1, y1, _, _ = self._visible_bounds()

        def to_canvas(p):
            return (p[0] - x1) * self.zoom, (p[1] - y1) * self.zoom

        for shape in self._overlay:
            color = shape.get("color", "#ff3b30")
            if shape["kind"] == "line":
                (ax, ay), (bx, by) = to_canvas(shape["from"]), to_canvas(shape["to"])
                self.create_line(ax, ay, bx, by, fill=color, width=ANGLE_MARKER_WIDTH_PX,
                                 arrow=tk.LAST if shape.get("arrow") else None,
                                 arrowshape=(10, 12, 5), dash=(4, 3) if shape.get("dash") else None,
                                 tags="overlay")
            elif shape["kind"] == "polyline":
                pts = [c for p in shape["pts"] for c in to_canvas(p)]
                if len(pts) >= 4:
                    self.create_line(*pts, fill=color, width=POLYLINE_WIDTH_PX, tags="overlay")
            elif shape["kind"] == "point":
                cx, cy = to_canvas(shape["xy"])
                r = ANGLE_MARKER_RADIUS_PX + 1
                self.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline="white", tags="overlay")
                if shape.get("label"):
                    self.create_text(cx + r + 3, cy - r - 3, text=shape["label"], anchor=tk.SW,
                                     fill=color, font=("Helvetica", 10, "bold"), tags="overlay")

    def canvas_to_image(self, canvas_x, canvas_y):
        """Maps a canvas pixel coordinate to the image pixel coordinate
        currently shown there. Uses the uniform zoom factor on both
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

    def _blend_contours(self, resized, ix1, iy1):
        """Alpha-blend every "contour" overlay shape into the zoomed image, on the line's own pixels
        only (anti-aliased, CONTOUR_WIDTH_PX wide at screen scale)."""
        contours = [s for s in self._overlay if s["kind"] == "contour" and len(s["pts"]) >= 2]
        if not contours:
            return resized
        out = resized.astype(np.float32)
        for shape in contours:
            layer = np.zeros(resized.shape[:2], np.uint8)
            pts = (np.asarray(shape["pts"], np.float64) - (ix1, iy1)) * self.zoom
            cv2.polylines(layer, [np.round(pts * 16).astype(np.int32).reshape(-1, 1, 2)], True, 255,
                          CONTOUR_WIDTH_PX, cv2.LINE_AA, shift=4)
            a = (layer.astype(np.float32) / 255.0 * float(shape.get("alpha", 0.5)))[..., None]
            rgb = np.array(_hex_to_rgb(shape.get("color", "#00ff00")), np.float32)
            out = out * (1 - a) + rgb * a
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)

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
        # Explicit interpolation: the default INTER_LINEAR is a 2x2 tap filter,
        # visibly blocky on the large upscales this canvas always does (a
        # ~100 px seedling crop fitted to 600 px is already ~6x before the user
        # zooms at all).
        interpolation = cv2.INTER_LANCZOS4 if self.zoom > 1 else cv2.INTER_AREA
        resized = cv2.resize(crop, (out_w, out_h), interpolation=interpolation)
        offset_x = (ix1 - x1) * self.zoom
        offset_y = (iy1 - y1) * self.zoom

        # The stored image is BGR (cv2.imread, as every caller passes it); PIL/Tk expect RGB.
        # Converted here, on the displayed pixels only, so red shows as red; the contour colours
        # are blended after the conversion, in RGB.
        if resized.ndim == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        resized = self._blend_contours(resized, ix1, iy1)
        self._photo = ImageTk.PhotoImage(image=Image.fromarray(resized))
        self.delete("all")
        self.create_image(offset_x, offset_y, image=self._photo, anchor=tk.NW)
        self._draw_overlay()
