"""The physical scale (um per pixel): entry and calibration dialog.

A mixin of seedling_measurement.Gui: the methods run on the Gui instance and use its state."""
import os
from tkinter import messagebox

import cv2
import numpy as np


class ScaleMixin:
    def set_scale(self, um_per_px):
        """Set (or clear, with None) the scale and show it in the entry."""
        self.scale_um_per_px = float(um_per_px) if um_per_px else None
        self.scale_var.set("" if self.scale_um_per_px is None else f"{self.scale_um_per_px:.4g}")

    def _apply_scale_entry(self, show_error=False):
        """Validate the Scale entry: empty clears the scale, a positive number sets it; anything else
        reverts the entry to the current value."""
        text = self.scale_var.get().strip().replace(",", ".")
        if not text:
            self.scale_um_per_px = None
            return
        try:
            value = float(text)
            if not (value > 0 and np.isfinite(value)):
                raise ValueError
        except ValueError:
            if show_error:
                messagebox.showwarning("Scale", "The scale must be a positive number of micrometres per pixel.")
            self.set_scale(self.scale_um_per_px)
            return
        self.scale_um_per_px = value

    def _open_scale_calibration(self):
        if not self.file_list or not self.path:
            messagebox.showinfo("Calibrate scale", "Open an image directory first.")
            return
        # The crops are native-resolution slices of the raw frame (preprocess_images changes
        # contrast and grey level, never the size), so pixels on the raw image are crop pixels.
        image = cv2.imread(os.path.join(self.path, self.file_list[0]), cv2.IMREAD_COLOR)
        if image is None:
            messagebox.showwarning("Calibrate scale", f"Cannot read {self.file_list[0]}")
            return
        from ui.scale_calibration import ScaleCalibrationWindow
        ScaleCalibrationWindow(self.root, image, self.set_scale, self.scale_um_per_px)
