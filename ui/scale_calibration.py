import tkinter as tk
from tkinter import messagebox, simpledialog

from ui.zoomable_canvas import ZoomableImageCanvas
from utils.hypocotyl_traits import scale_from_line

LINE_COLOR = "#ff3b30"


class ScaleCalibrationWindow(tk.Toplevel):
    """Click two points on an image over a known length (a ruler, a grid square), enter that length
    in mm, and `on_done(um_per_px)` is called with the scale. Crops are native-resolution slices of
    the raw frame (the preprocessing does not resize), so the scale measured on the raw image holds
    for the crops."""

    def __init__(self, parent, image, on_done, current_um_per_px=None):
        super().__init__(parent)
        self.title("Calibrate scale")
        self.on_done = on_done
        self.points = []
        self.canvas = ZoomableImageCanvas(self, width=800, height=600)
        self.canvas.grid(row=0, column=0, columnspan=4, padx=8, pady=8)
        self.canvas.set_image(image)
        self.canvas.bind("<Button-1>", self._on_click)
        current = f"current: {current_um_per_px:.3f} um/px" if current_um_per_px else "no scale set yet"
        self.label = tk.Label(self, text="", justify=tk.LEFT)
        self.label.grid(row=1, column=0, columnspan=4, padx=8, sticky="w")
        self._base_text = f"Click the two ends of a known length (mouse wheel zooms) -- {current}"
        tk.Button(self, text="Undo point", command=self._undo).grid(row=2, column=0, padx=8, pady=8)
        tk.Button(self, text="Reset", command=self._reset).grid(row=2, column=1, pady=8)
        tk.Button(self, text="OK (enter length)", command=self._confirm).grid(row=2, column=2, pady=8)
        tk.Button(self, text="Cancel", command=self.destroy).grid(row=2, column=3, padx=8, pady=8)
        self._refresh()

    def _on_click(self, event):
        h, w = self.canvas._image.shape[:2]
        x, y = self.canvas.canvas_to_image(event.x, event.y)
        if not (0 <= x < w and 0 <= y < h):
            return
        if len(self.points) >= 2:
            self.points = []
        self.points.append((float(x), float(y)))
        self._refresh()

    def _undo(self):
        if self.points:
            self.points.pop()
        self._refresh()

    def _reset(self):
        self.points = []
        self._refresh()

    def _refresh(self):
        shapes = [{"kind": "point", "xy": p, "color": LINE_COLOR, "label": str(i + 1)}
                  for i, p in enumerate(self.points)]
        text = self._base_text
        if len(self.points) == 2:
            shapes.append({"kind": "line", "from": self.points[0], "to": self.points[1], "color": LINE_COLOR})
            dx, dy = self.points[1][0] - self.points[0][0], self.points[1][1] - self.points[0][1]
            text += f"\nline length: {(dx * dx + dy * dy) ** 0.5:.1f} px"
        self.canvas.set_overlay(shapes)
        self.label.configure(text=text)

    def _confirm(self):
        if len(self.points) != 2:
            messagebox.showwarning("Calibrate scale", "Place two points first.", parent=self)
            return
        mm = simpledialog.askfloat("Known length", "Length of the line in mm:", parent=self, minvalue=1e-6)
        if mm is None:
            return
        try:
            scale = scale_from_line(self.points[0], self.points[1], mm)
        except ValueError as exc:
            messagebox.showwarning("Calibrate scale", str(exc), parent=self)
            return
        self.on_done(scale)
        self.destroy()
