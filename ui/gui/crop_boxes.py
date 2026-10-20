"""Crop-box editing, saving/loading and the per-frame seed point (collar).

A mixin of seedling_measurement.Gui: the methods run on the Gui instance and use its state."""
import os

import cv2
from tkinter import messagebox, simpledialog
from tkinter.filedialog import askopenfilename, asksaveasfilename

from ui.crop_box_canvas import clamp_point_to_box
from utils.apicalhook_angle import mask_below_seed_line
from utils.crop_boxes_io import boxes_to_entries, load_boxes_json, save_boxes_json
from utils.crop_layout import clamp_to_image, spatial_order
from utils.paths import data_path


def _boxes_dir():
    """data/crop_boxes/ (where the draw_crops boxes files live) when it exists, else the dialog's default."""
    folder = data_path("crop_boxes")
    return {"initialdir": folder} if os.path.isdir(folder) else {}


class CropBoxMixin:
    def _normalise_boxes(self, boxes):
        """Canvas boxes -> (boxes, collars, order): rounded, clamped to the image, boxes that fall
        outside it dropped, in spatial order. `order[new_index] = index in the input list`."""
        kept = []
        for i, raw in enumerate(boxes):
            box = {k: int(round(raw[k])) for k in ("cx", "cy", "half_w", "half_h")}
            if self.width1 and self.height1:
                box = clamp_to_image(box, self.width1, self.height1)
            if box["half_w"] < 1 or box["half_h"] < 1:
                continue
            collar = raw.get("collar")
            if collar is not None:
                x, y = clamp_point_to_box(collar[0], collar[1], box)
                collar = (int(round(x)), int(round(y)))
            kept.append((i, box, collar))
        order = spatial_order([box for _, box, _ in kept])
        kept = [kept[k] for k in order]
        return ([box for _, box, _ in kept], [collar for _, _, collar in kept], [i for i, _, _ in kept])

    def _boxes_for_canvas(self):
        return [dict(box, collar=list(collar)) if collar is not None else dict(box)
                for box, collar in zip(self.crop_boxes, self.collar_overrides)]

    def _push_boxes_to_canvas(self, selected=None):
        self.canvas.set_boxes(self._boxes_for_canvas(), selected=selected)

    def _commit_boxes(self, boxes, collars):
        self.crop_boxes = boxes
        self.collar_overrides = collars
        self.transformed_mid_points = [(box["cx"], box["cy"]) for box in boxes]
        self._update_button_states()
        self._update_status()

    def _on_boxes_changed(self, boxes):
        """Canvas on_change callback (after every completed draw / move / resize / delete / collar
        edit). Normalises the boxes (clamp, spatial order -- which renumbers the labels), and, when
        results already exist, asks first: crop ids index into every result store, so changing the
        boxes invalidates them."""
        selected = self.canvas.selected_index
        if self._busy:                      # no edits while the analysis is cutting crops
            self._push_boxes_to_canvas()
            return
        new_boxes, new_collars, order = self._normalise_boxes(boxes)
        new_selected = order.index(selected) if selected in order else None
        if new_boxes != self.crop_boxes or new_collars != self.collar_overrides:
            if self._has_results():
                if not messagebox.askyesno(
                        "Clear results?",
                        "Seedling results already exist. Changing the boxes (or a collar) clears them "
                        "-- including manual landmark edits and blanked frames.\n\nContinue?", parent=self.root):
                    self._push_boxes_to_canvas()
                    return
                self._clear_results()
            self._commit_boxes(new_boxes, new_collars)
        self._push_boxes_to_canvas(selected=new_selected)

    def _set_boxes(self, boxes):
        """Replace all boxes (dicts with optional "collar") through the same normalise / confirm /
        commit path as a canvas edit."""
        self.canvas.set_boxes(boxes)
        self._on_boxes_changed(self.canvas.get_boxes())

    def save_boxes(self):
        if not self.crop_boxes:
            messagebox.showinfo("Save boxes", "Draw at least one box first.", parent=self.root)
            return
        path = asksaveasfilename(
            parent=self.root, title="Save crop boxes", defaultextension=".json",
            filetypes=[("JSON", "*.json")], initialfile="crop_boxes.json", **_boxes_dir())
        if not path:
            return
        try:
            save_boxes_json(path, self.series_name or "series",
                            boxes_to_entries(self.crop_boxes, self.collar_overrides))
        except OSError as exc:
            messagebox.showerror("Save boxes", f"Cannot write {path}:\n{exc}", parent=self.root)
            return
        self.status_var.set(f"Saved {len(self.crop_boxes)} box(es) for '{self.series_name}' to {path}")

    def _choose_series(self, names):
        return simpledialog.askstring(
            "Load boxes",
            f"'{self.series_name}' is not in this file. Type the series to load:\n\n" + "\n".join(names),
            parent=self.root)

    def load_boxes(self):
        if not self.file_list:
            messagebox.showinfo("Load boxes", "Open an image directory first.", parent=self.root)
            return
        path = askopenfilename(parent=self.root, title="Load crop boxes", filetypes=[("JSON", "*.json")],
                               **_boxes_dir())
        if not path:
            return
        try:
            loaded = load_boxes_json(path, self.series_name, choose=self._choose_series)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            messagebox.showerror("Load boxes", f"Cannot read crop boxes from {path}:\n{exc}", parent=self.root)
            return
        if loaded is None:
            return
        series, boxes, collars = loaded
        self._set_boxes([dict(box, collar=list(c)) if c is not None else dict(box)
                         for box, c in zip(boxes, collars)])
        self.status_var.set(f"Loaded {len(self.crop_boxes)} box(es) of '{series}' from {path}")

    def _crop_origin(self, crop_id):
        """Top-left of this seedling's crop in image pixels, exactly as the crop loop cuts it."""
        box = self.crop_boxes[crop_id]
        return max(0, box["cx"] - box["half_w"]), max(0, box["cy"] - box["half_h"])

    def _seed_point(self, crop_id, file_name):
        """The seedling's seed / collar point in CROP coordinates for this frame, or None.

        Priority: the user's collar override for this box (image coordinates, shifted by the crop
        origin), else this frame's collar from the landmark model (already in crop coordinates),
        else a headless proxy (seed_point_proxies), else None. None means "no seed line": the
        callers then neither mask below it nor use it for anything else."""
        overrides = getattr(self, "collar_overrides", None) or []
        if crop_id < len(overrides) and overrides[crop_id] is not None:
            ox, oy = self._crop_origin(crop_id)
            return (int(round(overrides[crop_id][0] - ox)), int(round(overrides[crop_id][1] - oy)))
        readout = (getattr(self, "landmark_store", None) or {}).get(file_name)
        collar = readout.get("collar") if readout else None
        if collar is not None:
            return (int(round(collar[0])), int(round(collar[1])))
        proxy = (getattr(self, "seed_point_proxies", None) or {}).get(crop_id)
        return tuple(proxy) if proxy is not None else None

    @staticmethod
    def _bottom_centre(width, height):
        """Stand-in point for ApicalHook / the germination rule when there is no seed point: it
        sits on the crop's last row, where the seed line masks nothing."""
        return (int(width) // 2, int(height) - 1)

    @staticmethod
    def _above_seed_mask(shape, seed_point):
        """255 where a pixel is kept (on or above the seed line), or None (keep everything) when
        there is no seed point."""
        if seed_point is None:
            return None
        return cv2.bitwise_not(mask_below_seed_line(shape[:2], [seed_point]))

    #Transform the angle to the correct format
    #function for multithreading analysis in background
