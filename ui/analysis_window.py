import os
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
import cv2

from utils.gui_thread_safety import ProgressReporter
from utils.mask_editor import MaskEditor
from utils.postprocmask import resolve_mask_path
from ui.zoomable_canvas import ZoomableImageCanvas

# BGR colors, matching ApicalVisualizer's existing green-for-cotyledon convention.
COTYL_COLOR = (0, 255, 0)
HYPO_COLOR = (0, 128, 255)

# Germ is intentionally not shown/editable here (confirmed with user) -- it's
# still segmented and used by GerminationDetector, just not exposed in this UI.
LABEL_BY_TARGET = {"cotyl": "1", "hypo": "2"}

CANVAS_SIZE = 600
DEFAULT_BRUSH_RADIUS = 20


class SeedlingAnalysisWindow(tk.Toplevel):
    """
    Per-seedling preview/edit window (Phases 5-7). Lets the user run
    segmentation for just this one seedling's cropped time series, step
    through frames with per-label mask overlays, zoom in for a closer look,
    and brush-edit the cotyledon/hypocotyl mask before recomputing that
    frame's angle -- all without waiting on the full multi-seedling
    "Start Analysis" batch, which stays unchanged as the separate step used
    to actually commit a run.
    """

    def __init__(self, gui, crop_id):
        super().__init__(gui.root)
        self.gui = gui
        self.crop_id = crop_id
        self.title(f"Seedling {crop_id} - analysis")

        self.cropped_filenames = []
        self.frame_results = []
        self.current_frame = 0

        self.show_cotyl = tk.BooleanVar(value=True)
        self.show_hypo = tk.BooleanVar(value=True)
        self.edit_target = tk.StringVar(value="cotyl")
        self.brush_radius_var = tk.StringVar(value=str(DEFAULT_BRUSH_RADIUS))
        self.brush_mode = "add"
        self.mask_editor = None
        self._painting = False

        # Manual angle override (ported from the removed legacy review window):
        # a 3-click vector angle, with the same Overhook sign-flip convention,
        # scoped to this seedling/current frame.
        self.overhook_var = tk.IntVar()
        self._manual_mode = False
        self._manual_angle_points = []
        self._manual_angle_value = None

        self.progress_reporter = ProgressReporter()

        self._build_widgets()

        # If this seedling was already segmented (a previous preview session,
        # or Export Results lazily segmenting it), pick up the shared results
        # instead of asking the user to launch segmentation again.
        if crop_id in gui.frame_results_by_crop:
            self.cropped_filenames = gui.cropped_filenames_by_crop[crop_id]
            self.frame_results = gui.frame_results_by_crop[crop_id]
            self._update_nav_state()
            self._load_mask_editor_for_current()
            self._render_current_frame()

        self.after(100, self._pump_progress)

    def _build_widgets(self):
        self.canvas = ZoomableImageCanvas(self, width=CANVAS_SIZE, height=CANVAS_SIZE)
        self.canvas.grid(row=0, column=0, columnspan=4, padx=8, pady=8)
        self.canvas.bind("<Enter>", lambda event: self.canvas.focus_set())
        self.canvas.bind("<Button-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<KeyPress-a>", lambda event: self._set_brush_mode("add"))
        self.canvas.bind("<KeyPress-A>", lambda event: self._set_brush_mode("add"))
        self.canvas.bind("<KeyPress-e>", lambda event: self._set_brush_mode("erase"))
        self.canvas.bind("<KeyPress-E>", lambda event: self._set_brush_mode("erase"))
        self.canvas.bind("<KeyPress-z>", self._undo_stroke)
        self.canvas.bind("<KeyPress-Z>", self._undo_stroke)

        zoom_frame = tk.Frame(self)
        zoom_frame.grid(row=0, column=4, padx=4, pady=8, sticky="n")
        tk.Button(zoom_frame, text="Zoom +", width=8, command=self.canvas.zoom_in).pack(pady=2)
        tk.Button(zoom_frame, text="Zoom -", width=8, command=self.canvas.zoom_out).pack(pady=2)
        tk.Button(zoom_frame, text="Reset zoom", width=8, command=self.canvas.reset_zoom).pack(pady=2)
        tk.Label(zoom_frame, text="(mouse wheel\nalso zooms)", justify=tk.CENTER).pack(pady=(8, 0))

        self.launch_button = tk.Button(self, text="Launch segmentation", command=self._start_segmentation)
        self.launch_button.grid(row=1, column=0, padx=4, pady=4, sticky="w")

        self.progress = ttk.Progressbar(self, orient=tk.HORIZONTAL, length=220)
        self.progress.grid(row=1, column=1, columnspan=2, padx=4, pady=4, sticky="w")

        self.status_label = tk.Label(self, text="")
        self.status_label.grid(row=1, column=3, padx=4, pady=4, sticky="w")

        overlay_frame = tk.Frame(self)
        overlay_frame.grid(row=2, column=0, columnspan=4)
        tk.Label(overlay_frame, text="Show:").pack(side=tk.LEFT)
        tk.Checkbutton(overlay_frame, text="Cotyledon", variable=self.show_cotyl,
                        command=self._render_current_frame).pack(side=tk.LEFT)
        tk.Checkbutton(overlay_frame, text="Hypocotyl", variable=self.show_hypo,
                        command=self._render_current_frame).pack(side=tk.LEFT)

        nav_frame = tk.Frame(self)
        nav_frame.grid(row=3, column=0, columnspan=4, pady=8)
        self.prev_button = tk.Button(nav_frame, text="<--", width=4, command=self._show_previous_frame, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=4)
        self.frame_label = tk.Label(nav_frame, text="No frames yet -- launch segmentation")
        self.frame_label.pack(side=tk.LEFT, padx=8)
        self.next_button = tk.Button(nav_frame, text="-->", width=4, command=self._show_next_frame, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=4)

        self.angle_label = tk.Label(self, text="")
        self.angle_label.grid(row=4, column=0, columnspan=4, pady=4)

        edit_frame = tk.Frame(self)
        edit_frame.grid(row=5, column=0, columnspan=4, pady=(0, 8))
        tk.Label(edit_frame, text="Edit mask:").pack(side=tk.LEFT)
        tk.Radiobutton(edit_frame, text="Cotyledon", variable=self.edit_target, value="cotyl",
                        command=self._on_edit_target_change).pack(side=tk.LEFT)
        tk.Radiobutton(edit_frame, text="Hypocotyl", variable=self.edit_target, value="hypo",
                        command=self._on_edit_target_change).pack(side=tk.LEFT)
        tk.Label(edit_frame, text="  Brush size:").pack(side=tk.LEFT)
        brush_entry = tk.Entry(edit_frame, width=4, textvariable=self.brush_radius_var)
        brush_entry.pack(side=tk.LEFT)
        brush_entry.bind("<Return>", self._on_brush_radius_change)
        brush_entry.bind("<FocusOut>", self._on_brush_radius_change)
        self.brush_mode_label = tk.Label(edit_frame, text="Brush: ADD (A=add, E=erase, Z=undo)")
        self.brush_mode_label.pack(side=tk.LEFT, padx=(8, 0))
        self.recompute_button = tk.Button(edit_frame, text="Recompute angle", command=self._recompute_angle)
        self.recompute_button.pack(side=tk.LEFT, padx=(8, 0))

        manual_frame = tk.Frame(self)
        manual_frame.grid(row=6, column=0, columnspan=4, pady=(0, 8))
        tk.Label(manual_frame, text="Manual angle:").pack(side=tk.LEFT)
        tk.Checkbutton(manual_frame, text="Overhook", variable=self.overhook_var).pack(side=tk.LEFT)
        self.place_angle_button = tk.Button(manual_frame, text="Place angle", width=12,
                                             command=self._toggle_manual_angle_mode)
        self.place_angle_button.pack(side=tk.LEFT, padx=(8, 0))
        self.replace_angle_button = tk.Button(manual_frame, text="Replace angle", width=12,
                                               command=self._replace_angle)
        self.replace_angle_button.pack(side=tk.LEFT, padx=(4, 0))
        self.blank_angle_button = tk.Button(manual_frame, text="Blank this frame", width=14,
                                             command=self._blank_angle)
        self.blank_angle_button.pack(side=tk.LEFT, padx=(4, 0))

    # --- Segmentation (background thread) -----------------------------

    def _start_segmentation(self):
        self.launch_button["state"] = tk.DISABLED
        threading.Thread(target=self._run_segmentation).start()

    def _run_segmentation(self):
        """Runs on a background thread -- must not touch Tk widgets directly."""
        self.progress_reporter.report(value=0)
        cropped_filenames = self.gui.segment_single_seedling(self.crop_id, progress_reporter=self.progress_reporter)

        results = []
        total = len(cropped_filenames)
        for idx, file_name in enumerate(cropped_filenames):
            result = self.gui.process_single_frame(file_name, self.crop_id)
            results.append(result)
            self.progress_reporter.report(value=int(((idx + 1) / total) * 100))

        # Reconciles the Phase 8 "known gap": germination detection used to
        # only ever run inside the full "Start Analysis" batch, so a
        # preview-only seedling never got a germination time-zero at all.
        self.gui._ensure_germination_detected(self.crop_id)

        # Store by reference (not a copy) in the Gui's shared per-crop results,
        # so later in-place edits here (Recompute angle / Replace angle / Blank
        # this frame) are visible to Export Results even after this window closes.
        self.gui.frame_results_by_crop[self.crop_id] = results
        self.gui.cropped_filenames_by_crop[self.crop_id] = cropped_filenames

        self.progress_reporter.report(cropped_filenames=cropped_filenames, frame_results=results, done=True)

    def _pump_progress(self):
        for event in self.progress_reporter.drain():
            if "value" in event:
                self.progress["value"] = event["value"]
            if "message" in event:
                self.status_label.configure(text=event["message"])
            if event.get("done"):
                self.cropped_filenames = event["cropped_filenames"]
                self.frame_results = event["frame_results"]
                self.current_frame = 0
                self.launch_button["state"] = tk.NORMAL
                self.status_label.configure(text="Done")
                self._update_nav_state()
                self._load_mask_editor_for_current()
                self._render_current_frame()
        self.after(100, self._pump_progress)

    # --- Frame navigation ------------------------------------------------

    def _update_nav_state(self):
        has_frames = len(self.frame_results) > 0
        self.prev_button["state"] = tk.NORMAL if has_frames and self.current_frame > 0 else tk.DISABLED
        self.next_button["state"] = tk.NORMAL if has_frames and self.current_frame < len(self.frame_results) - 1 else tk.DISABLED

    def _show_previous_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self._reset_manual_angle()
            self._update_nav_state()
            self._load_mask_editor_for_current()
            self._render_current_frame()

    def _show_next_frame(self):
        if self.current_frame < len(self.frame_results) - 1:
            self.current_frame += 1
            self._reset_manual_angle()
            self._update_nav_state()
            self._load_mask_editor_for_current()
            self._render_current_frame()

    # --- Mask brush editing (Phase 7) ------------------------------------

    def _on_edit_target_change(self):
        self._load_mask_editor_for_current()
        self._render_current_frame()

    def _get_brush_radius(self):
        try:
            return max(1, int(self.brush_radius_var.get()))
        except ValueError:
            return DEFAULT_BRUSH_RADIUS

    def _on_brush_radius_change(self, event=None):
        if self.mask_editor is not None:
            self.mask_editor.set_brush_radius(self._get_brush_radius())

    def _load_mask_editor_for_current(self):
        """(Re)loads the brush-editable mask for the current frame + edit
        target from disk (the postprocessed version if one already exists,
        else the raw prediction) -- called whenever the frame or the edit
        target changes, never mid-stroke."""
        if not self.cropped_filenames:
            self.mask_editor = None
            return

        file_name = self.cropped_filenames[self.current_frame]
        label = LABEL_BY_TARGET[self.edit_target.get()]
        path = resolve_mask_path(file_name, label)
        raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            self.mask_editor = None
            return

        _, binary = cv2.threshold(raw, 127, 255, cv2.THRESH_BINARY)
        # UNetInference saves masks with 0 = foreground, 255 = background
        # (same convention process_single_frame un-inverts via bitwise_not
        # before building cotyl_mask/hypo_mask) -- invert here too so
        # MaskEditor's internal representation is 255 = foreground,
        # matching what _render_current_frame expects from result["cotyl_mask"]
        # /result["hypo_mask"] and what "Add"/"Erase" mean to the user.
        binary = cv2.bitwise_not(binary)
        self.mask_editor = MaskEditor(binary, brush_radius=self._get_brush_radius())

    def _persist_edited_mask(self):
        if self.mask_editor is None:
            return
        file_name = self.cropped_filenames[self.current_frame]
        label = LABEL_BY_TARGET[self.edit_target.get()]
        os.makedirs("data/postprocess", exist_ok=True)
        out_path = os.path.join("data/postprocess", f"{file_name[:-4]}-{label}.png")
        # Invert back to the on-disk 0 = foreground convention before saving,
        # so this file stays interchangeable with a raw data/predict/ file
        # for every other reader (resolve_mask_path/process_single_frame).
        cv2.imwrite(out_path, cv2.bitwise_not(self.mask_editor.get_mask()))

    def _set_brush_mode(self, mode):
        self.brush_mode = mode
        self.brush_mode_label.configure(text=f"Brush: {mode.upper()} (A=add, E=erase, Z=undo)")

    def _on_canvas_press(self, event):
        self.canvas.focus_set()
        if self._manual_mode:
            self._place_manual_angle_point(event)
            return
        if self.mask_editor is None:
            return
        self._painting = True
        self.mask_editor.begin_stroke()
        self._paint_at(event.x, event.y)

    def _on_canvas_drag(self, event):
        if not self._painting or self.mask_editor is None:
            return
        self._paint_at(event.x, event.y)

    def _on_canvas_release(self, event):
        if not self._painting:
            return
        self._painting = False
        if self.mask_editor is not None:
            self.mask_editor.end_stroke()
            self._persist_edited_mask()

    def _paint_at(self, canvas_x, canvas_y):
        img_x, img_y = self.canvas.canvas_to_image(canvas_x, canvas_y)
        self.mask_editor.paint(img_x, img_y, add=(self.brush_mode == "add"))
        self._render_current_frame()

    def _undo_stroke(self, event=None):
        if self.mask_editor is None:
            return
        if self.mask_editor.undo():
            self._persist_edited_mask()
            self._render_current_frame()

    def _recompute_angle(self):
        if not self.frame_results:
            return
        self._persist_edited_mask()
        file_name = self.cropped_filenames[self.current_frame]
        result = self.gui.process_single_frame(file_name, self.crop_id, frame_index=self.current_frame)
        self.frame_results[self.current_frame] = result
        self._render_current_frame()

    # --- Manual angle override (ported from the removed legacy window) ----

    def _reset_manual_angle(self):
        self._manual_angle_points = []
        self._manual_angle_value = None

    def _toggle_manual_angle_mode(self):
        self._manual_mode = not self._manual_mode
        self.place_angle_button.configure(bg="#d78a5e" if self._manual_mode else "white")
        if not self._manual_mode:
            self._reset_manual_angle()
            self._render_current_frame()

    def _place_manual_angle_point(self, event):
        img_x, img_y = self.canvas.canvas_to_image(event.x, event.y)
        if len(self._manual_angle_points) >= 3:
            self._reset_manual_angle()
        self._manual_angle_points.append((img_x, img_y))

        if len(self._manual_angle_points) == 3:
            a, b, c = (np.array(p) for p in self._manual_angle_points)
            ba, bc = a - b, c - b
            cosine_ang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = round(np.degrees(np.arccos(np.clip(cosine_ang, -1.0, 1.0))))
            if self.overhook_var.get():
                # Same 0-360 (180=closed) convention as AngleCalculator's
                # automated Overhooked branch (Phase 8) -- not a negation.
                angle = 360 - angle
            self._manual_angle_value = angle

        self._render_current_frame()

    def _current_seed_id(self):
        return self.gui.crop_points_numberID[self.crop_id][0]

    def _sync_angle_list(self, result):
        seed_ids = result["seed_ids"]
        angle_list = []
        for sid in seed_ids:
            a = result["angle_dict"].get(sid, '-')
            angle_list.append(round(a) if isinstance(a, (int, float)) and not np.isnan(a) else '-')
        result["angle_list"] = angle_list
        result["state_list"] = [result.get("state_dict", {}).get(sid, '-') for sid in seed_ids]

    def _replace_angle(self):
        if not self.frame_results or self._manual_angle_value is None:
            return
        result = self.frame_results[self.current_frame]
        if result is None:
            return
        seed_id = self._current_seed_id()
        result["angle_dict"][seed_id] = self._manual_angle_value
        result.setdefault("state_dict", {})[seed_id] = "Manual"
        self._sync_angle_list(result)
        self._reset_manual_angle()
        self._render_current_frame()

    def _blank_angle(self):
        if not self.frame_results:
            return
        result = self.frame_results[self.current_frame]
        if result is None:
            return
        seed_id = self._current_seed_id()
        result["angle_dict"][seed_id] = np.nan
        result.setdefault("state_dict", {})[seed_id] = "Manual"
        self._sync_angle_list(result)
        self._render_current_frame()

    # --- Rendering --------------------------------------------------------

    def _render_current_frame(self):
        if not self.frame_results:
            return

        file_name = self.cropped_filenames[self.current_frame]
        result = self.frame_results[self.current_frame]
        self.frame_label.configure(text=f"{file_name} ({self.current_frame + 1}/{len(self.frame_results)})")

        if result is None:
            self.angle_label.configure(text="No mask/segmentation data for this frame.")
            return

        image = cv2.imread(os.path.join("data/images", file_name), cv2.IMREAD_COLOR)
        overlay = image.copy()

        target = self.edit_target.get()
        if self.show_cotyl.get():
            mask = self.mask_editor.get_mask() if (self.mask_editor is not None and target == "cotyl") else result["cotyl_mask"]
            if mask is not None:
                overlay[mask > 0] = COTYL_COLOR
        if self.show_hypo.get():
            mask = self.mask_editor.get_mask() if (self.mask_editor is not None and target == "hypo") else result["hypo_mask"]
            if mask is not None:
                overlay[mask > 0] = HYPO_COLOR

        # No BGR->RGB conversion, matching the rest of the app's existing
        # (uncorrected) display convention -- consistent look, not a fix here.
        blended = cv2.addWeighted(image, 1.0, overlay, 0.5, 0)

        # Manual angle placement markers are burned into the image itself
        # (rather than drawn as separate Tk canvas items) since
        # ZoomableImageCanvas fully redraws from the image on every zoom/pan.
        marker_color = (135, 124, 184)
        for px, py in self._manual_angle_points:
            cv2.circle(blended, (int(round(px)), int(round(py))), 4, marker_color, -1)
        for (x0, y0), (x1, y1) in zip(self._manual_angle_points, self._manual_angle_points[1:]):
            cv2.line(blended, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), marker_color, 2)

        self.canvas.set_image(blended)

        angle_dict = result["angle_dict"]
        if angle_dict:
            angle_text = ", ".join(
                f"seedling {sid}: {round(a) if isinstance(a, (int, float)) and not np.isnan(a) else '-'}"
                for sid, a in angle_dict.items()
            )
        else:
            angle_text = "no angle computed for this frame"
        if self._manual_angle_value is not None:
            angle_text += f"  |  manual: {self._manual_angle_value}° (Replace angle to commit)"
        self.angle_label.configure(text=angle_text)
