import os
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
import cv2

from utils.gui_thread_safety import ProgressReporter
from utils.mask_editor import MaskEditor
from utils.angle_timeseries import sync_angle_and_state_lists
from utils.germination_detector import GerminationDetector
from ui.zoomable_canvas import ZoomableImageCanvas

# BGR colors, matching ApicalVisualizer's existing green-for-cotyledon convention.
COTYL_COLOR = (0, 255, 0)
HYPO_COLOR = (0, 128, 255)
GERM_COLOR = (255, 0, 255)

# Germ is viewable but not brush-editable: it feeds GerminationDetector rather
# than the angle geometry, and being able to SEE it is what makes a wrong
# germination time-zero diagnosable (an empty or whole-crop germ mask is obvious
# on screen and invisible in the numbers).
LABEL_BY_TARGET = {"cotyl": "1", "hypo": "2"}

CANVAS_SIZE = 600
DEFAULT_BRUSH_RADIUS = 3

# Manual angle markers, drawn as canvas vector items -- RGB hex for Tk, the same
# purple the burned-in cv2 markers showed on screen.
ANGLE_MARKER_COLOR = "#877cb8"


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
        # 1-based in the UI, matching the kinematics legend, the angle readout
        # below and the export's seedling_id; crop_id itself stays 0-based.
        self.title(f"Seedling {crop_id + 1} - analysis")

        self.cropped_filenames = []
        self.frame_results = []
        self.current_frame = 0

        self.show_cotyl = tk.BooleanVar(value=True)
        self.show_hypo = tk.BooleanVar(value=True)
        self.show_germ = tk.BooleanVar(value=False)
        self.edit_target = tk.StringVar(value="cotyl")
        self.brush_radius_var = tk.StringVar(value=str(DEFAULT_BRUSH_RADIUS))
        self.brush_mode = "add"
        self.mask_editor = None
        self._painting = False
        self._last_canvas_xy = None

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
        self.canvas.bind("<Motion>", lambda event: self._update_brush_preview_at(event.x, event.y))
        self.canvas.bind("<Leave>", lambda event: self.canvas.clear_brush_preview())
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
        tk.Checkbutton(overlay_frame, text="Germination", variable=self.show_germ,
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

        germ_frame = tk.Frame(self)
        germ_frame.grid(row=7, column=0, columnspan=4, pady=(0, 8))
        tk.Label(germ_frame, text="Germination:").pack(side=tk.LEFT)
        self.set_germ_button = tk.Button(germ_frame, text="Set to this frame", width=16,
                                         command=self._set_germination_frame, state=tk.DISABLED)
        self.set_germ_button.pack(side=tk.LEFT, padx=(8, 0))
        self.germ_all_button = tk.Button(germ_frame, text="Apply to all seedlings", width=20,
                                        command=self._apply_germination_to_all, state=tk.DISABLED)
        self.germ_all_button.pack(side=tk.LEFT, padx=(4, 0))
        self.clear_germ_button = tk.Button(germ_frame, text="Clear", width=6,
                                          command=self._clear_germination_frame, state=tk.DISABLED)
        self.clear_germ_button.pack(side=tk.LEFT, padx=(4, 0))
        self.germ_label = tk.Label(germ_frame, text="")
        self.germ_label.pack(side=tk.LEFT, padx=(8, 0))

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

        # Store by reference (not a copy) in the Gui's shared per-crop results,
        # so later in-place edits here (Recompute angle / Replace angle / Blank
        # this frame) are visible to Export Results even after this window closes.
        self.gui.frame_results_by_crop[self.crop_id] = results
        self.gui.cropped_filenames_by_crop[self.crop_id] = cropped_filenames

        # Reconciles the Phase 8 "known gap": germination detection used to
        # only ever run inside the full "Start Analysis" batch, so a
        # preview-only seedling never got a germination time-zero at all.
        self.gui._ensure_germination_detected(self.crop_id)
        # Same gap for the angle-timeseries reconstruction (Phase 9): a
        # preview-only seedling never had the temporal pass applied either.
        self.gui._reconstruct_series_for_crop(self.crop_id)

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
        germ_state = tk.NORMAL if has_frames else tk.DISABLED
        self.set_germ_button["state"] = germ_state
        self.germ_all_button["state"] = germ_state
        self.clear_germ_button["state"] = germ_state

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
        if self._last_canvas_xy is not None:
            self._update_brush_preview_at(*self._last_canvas_xy)

    def _get_brush_radius(self):
        try:
            return max(1, int(self.brush_radius_var.get()))
        except ValueError:
            return DEFAULT_BRUSH_RADIUS

    def _on_brush_radius_change(self, event=None):
        if self.mask_editor is not None:
            self.mask_editor.set_brush_radius(self._get_brush_radius())
        if self._last_canvas_xy is not None:
            self._update_brush_preview_at(*self._last_canvas_xy)

    def _load_mask_editor_for_current(self):
        """(Re)loads the brush-editable mask for the current frame + edit
        target from the shared MaskStore (a previous brush edit if one exists,
        else the raw prediction) -- called whenever the frame or the edit
        target changes, never mid-stroke."""
        if not self.cropped_filenames:
            self.mask_editor = None
            return

        file_name = self.cropped_filenames[self.current_frame]
        label = LABEL_BY_TARGET[self.edit_target.get()]
        mask = self.gui.mask_store.get(file_name, label)
        if mask is None:
            self.mask_editor = None
            return

        # The store already holds masks as 255 = foreground -- the same
        # convention MaskEditor works in and _render_current_frame expects from
        # result["cotyl_mask"]/["hypo_mask"] -- so no inversion is needed here.
        # (The old code had to bitwise_not every mask it read, because the
        # on-disk format was inverted; getting that wrong was the cause of the
        # "overlay looks inverted" bug.) Copy so edits don't mutate the stored
        # raw prediction in place before the user commits a stroke.
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        self.mask_editor = MaskEditor(binary, brush_radius=self._get_brush_radius())

    def _persist_edited_mask(self):
        if self.mask_editor is None:
            return
        file_name = self.cropped_filenames[self.current_frame]
        label = LABEL_BY_TARGET[self.edit_target.get()]
        # A dict assignment into the session-wide store on Gui, replacing the
        # full PNG encode+write this used to do on every mouse-up. Kept on Gui
        # (not this Toplevel) so the edit survives closing/reopening the window,
        # which is what the data/postprocess/ file used to provide.
        self.gui.mask_store.put_edited(file_name, label, self.mask_editor.get_mask().copy())

    def _set_brush_mode(self, mode):
        self.brush_mode = mode
        self.brush_mode_label.configure(text=f"Brush: {mode.upper()} (A=add, E=erase, Z=undo)")
        if self._last_canvas_xy is not None:
            self._update_brush_preview_at(*self._last_canvas_xy)

    def _update_brush_preview_at(self, canvas_x, canvas_y):
        """Moves/shows the brush-size preview circle under the cursor, in
        the edit target's mask color -- hidden outside brush-editing (no
        mask loaded, or mid manual-angle placement)."""
        self._last_canvas_xy = (canvas_x, canvas_y)
        if self.mask_editor is None or self._manual_mode:
            self.canvas.clear_brush_preview()
            return
        img_x, img_y = self.canvas.canvas_to_image(canvas_x, canvas_y)
        target_color = COTYL_COLOR if self.edit_target.get() == "cotyl" else HYPO_COLOR
        outline = "#{:02x}{:02x}{:02x}".format(*target_color[::-1])  # BGR -> RGB hex
        self.canvas.set_brush_preview(img_x, img_y, self._get_brush_radius(), outline=outline)

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
        self._update_brush_preview_at(event.x, event.y)
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
        self.gui._reconstruct_series_for_crop(self.crop_id)
        self._render_current_frame()

    # --- Manual angle override (ported from the removed legacy window) ----

    def _reset_manual_angle(self):
        self._manual_angle_points = []
        self._manual_angle_value = None
        self.canvas.clear_angle_markers()

    def _toggle_manual_angle_mode(self):
        self._manual_mode = not self._manual_mode
        self.place_angle_button.configure(bg="#d78a5e" if self._manual_mode else "white")
        if self._manual_mode:
            self.canvas.clear_brush_preview()
        else:
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
            # Kept as a float: the clicked points are float image coordinates
            # (canvas_to_image doesn't quantize), so rounding to whole degrees
            # here would throw away the precision zooming in is meant to buy.
            # Rounded only for display, like every other angle in this window.
            angle = float(np.degrees(np.arccos(np.clip(cosine_ang, -1.0, 1.0))))
            # Store in the "bio" convention used everywhere downstream
            # (180 = closed, decreasing as the hook opens, >180 = overhooked;
            # see utils/angle_timeseries.py and Gui._reconstruct_series_for_crop).
            # The clicked geometric angle is ~0 when the hook is closed, so a
            # closed hook maps to ~180 via 180 - angle; an overhooked hook has
            # folded PAST closed, so it reads just above 180 instead.
            if self.overhook_var.get():
                self._manual_angle_value = 180 + angle
            else:
                self._manual_angle_value = 180 - angle

        self._render_current_frame()

    def _current_seed_id(self):
        return self.gui.crop_points_numberID[self.crop_id][0]

    def _sync_angle_list(self, result):
        sync_angle_and_state_lists(result)

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
        # A manual override is a fixed anchor for the temporal reconstruction,
        # so re-run it over this seedling's whole series: the corrected frame
        # pulls the surrounding frames onto the right branch (see
        # Gui._reconstruct_series_for_crop / angle_timeseries._align_branches).
        # Same contract as _recompute_angle above; it reconstructs from each
        # frame's untouched raw_angle_dict, so repeated overrides compose
        # instead of re-smoothing an already-reconstructed value.
        self.gui._reconstruct_series_for_crop(self.crop_id)
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
        # Same re-anchoring as _replace_angle: a blanked frame stays NaN
        # (_align_branches skips missing frames, and the reconstruction never
        # writes back to a "Manual" one), but dropping a bad reading out of the
        # branch-alignment input lets its neighbours re-resolve.
        self.gui._reconstruct_series_for_crop(self.crop_id)
        self._render_current_frame()

    # --- Germination time-zero -------------------------------------------

    def _germination_detector(self):
        """The Gui's shared GerminationDetector, created on demand -- the same
        lazy init Gui._ensure_germination_detected does, so setting a
        germination frame works even for a seedling whose automatic detection
        hasn't run yet."""
        if self.gui.germination_detector is None:
            self.gui.germination_detector = GerminationDetector()
        return self.gui.germination_detector

    def _set_germination_frame(self):
        if not self.frame_results:
            return
        self._germination_detector().set_override(self.crop_id, self.current_frame)
        self._update_germination_label()

    def _apply_germination_to_all(self):
        """Same germination frame for every seedling on the plate -- the usual
        case, since one plate is imaged as one time series."""
        if not self.frame_results:
            return
        detector = self._germination_detector()
        for crop_id in range(len(self.gui.crop_boxes)):
            detector.set_override(crop_id, self.current_frame)
        self._update_germination_label()

    def _clear_germination_frame(self):
        """Drop the manual override, falling back to automatic detection."""
        self._germination_detector().clear_override(self.crop_id)
        self._update_germination_label()

    def _update_germination_label(self):
        detector = self.gui.germination_detector
        time_zero = detector.get_time_zero(self.crop_id) if detector is not None else None
        if time_zero is None:
            self.germ_label.configure(text="not set")
            return
        # Distinguishing manual from detected matters: it's the difference
        # between "the model found this" and "I told it this".
        source = "manual" if detector.has_override(self.crop_id) else "detected"
        self.germ_label.configure(text=f"{source}: frame {time_zero + 1}")

    # --- Rendering --------------------------------------------------------

    def _render_current_frame(self):
        if not self.frame_results:
            return

        file_name = self.cropped_filenames[self.current_frame]
        result = self.frame_results[self.current_frame]
        self.frame_label.configure(text=f"{file_name} ({self.current_frame + 1}/{len(self.frame_results)})")
        self._update_germination_label()

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
        if self.show_germ.get() and result.get("germ_mask") is not None:
            overlay[result["germ_mask"] > 0] = GERM_COLOR

        # No BGR->RGB conversion, matching the rest of the app's existing
        # (uncorrected) display convention -- consistent look, not a fix here.
        # Weights must sum to 1.0: where overlay == image (no mask), this
        # reproduces the original pixel untouched (background "transparent");
        # only true mask pixels (overlay != image) get the 50/50 tint. Weights
        # summing to >1 (the old 1.0/0.5 split) blew out every background
        # pixel, making it look like the crop sat "under" solid mask colors.
        blended = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)

        self.canvas.set_image(blended)
        # Markers go on as canvas vector items, at a fixed screen size, keeping
        # them crisp at any zoom (they used to be burned into the crop at native
        # resolution and then magnified along with it).
        self.canvas.set_angle_markers(self._manual_angle_points, ANGLE_MARKER_COLOR)

        angle_dict = result["angle_dict"]
        if angle_dict:
            angle_text = ", ".join(
                f"seedling {sid}: {round(a) if isinstance(a, (int, float)) and not np.isnan(a) else '-'}"
                for sid, a in angle_dict.items()
            )
        else:
            angle_text = "no angle computed for this frame"
        if self._manual_angle_value is not None:
            angle_text += f"  |  manual: {self._manual_angle_value:.1f}° (Replace angle to commit)"
        self.angle_label.configure(text=angle_text)
