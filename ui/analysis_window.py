import os
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import cv2

from utils.gui_thread_safety import ProgressReporter
from utils.angle_timeseries import sync_angle_and_state_lists
from utils.germination_detector import GerminationDetector
from utils.hypocotyl_traits import hypocotyl_midline, hypocotyl_traits, to_mm
from utils.landmark_edit import (FLAG_COLOR, LandmarkClickSession, MUTED_COLOR, landmark_shapes,
                                 previous_collar, session_from_readout)
from utils.landmark_temporal import describe_flag
from utils.landmark_edits import edit_record, series_name_from_path, set_record_overhook, write_edit
from utils.angle_annotation import parse_crop_filename
from utils.paths import crop_path
from utils.review_queue import next_in_queue, queue_position, review_queue
from utils.hook_kinetics import MODELS as KINETICS_MODELS, curve
from ui.zoomable_canvas import ZoomableImageCanvas

# Overlay colors (hex). The raw crop is shown unmodified; the segmentation is only drawn as thin lines:
# an alpha-blended contour for the cotyledon / germination masks, a solid polyline for the hypocotyl
# midline (yellow: distinct from the landmark colors of utils/landmark_edit.py).
COTYL_CONTOUR_COLOR = "#00ff00"
GERM_CONTOUR_COLOR = "#ff00ff"
MIDLINE_COLOR = "#ffe600"
CONTOUR_ALPHA = 0.5
CONTOUR_EPSILON_PX = 0.7

CANVAS_SIZE = 600


def mask_contour_shapes(mask, color, alpha=CONTOUR_ALPHA, epsilon=CONTOUR_EPSILON_PX):
    """Canvas "contour" shapes (outer contours, simplified) of a 255 = foreground mask; [] when empty.
    +0.5 moves from pixel indices to pixel centres, the frame the canvas draws the image in."""
    if mask is None:
        return []
    found, _ = cv2.findContours((np.asarray(mask) > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    shapes = []
    for contour in found:
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) >= 2:
            shapes.append({"kind": "contour", "pts": (approx.astype(np.float64) + 0.5).tolist(),
                           "color": color, "alpha": alpha})
    return shapes


class SeedlingAnalysisWindow(tk.Toplevel):
    """
    Per-seedling preview/edit window. Lets the user run segmentation for just
    this one seedling's cropped time series, step through frames over the raw
    image (thin overlays only: cotyledon contour, hypocotyl midline, landmarks),
    zoom in for a closer look and correct the landmarks by hand -- all without
    waiting on the full multi-seedling "Start Analysis" batch.
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
        self.show_landmarks = tk.BooleanVar(value=True)
        self.show_collar = tk.BooleanVar(value=True)
        # Kept on the Gui, so the choice survives switching to another seedling's window.
        self.review_unreadable_var = tk.BooleanVar(value=bool(getattr(gui, "review_include_unreadable", False)))
        self.save_edits_var = tk.BooleanVar(value=False)
        # frame index -> (junction/collar signature, midline or None); see _midline_for
        self._midline_cache = {}

        # Manual landmark edit: 5 clicks (J, H1, H2, C1 base, C2 tip) plus an optional collar and
        # root point (utils/landmark_edit.LandmarkClickSession), scoped to this seedling/current
        # frame. The Overhook box is ticked from the ray-crossing test once C2 is placed.
        self.overhook_var = tk.IntVar()
        self._manual_mode = False
        self._session = LandmarkClickSession()
        self._germ_check_shown = True

        self.progress_reporter = ProgressReporter()

        self._build_widgets()

        # If this seedling was already segmented (a previous preview session,
        # or Export Results lazily segmenting it), pick up the shared results
        # instead of asking the user to launch segmentation again.
        if crop_id in gui.frame_results_by_crop:
            self.cropped_filenames = gui.cropped_filenames_by_crop[crop_id]
            self.frame_results = gui.frame_results_by_crop[crop_id]
            self._update_nav_state()
            self._render_current_frame()

        self.after(100, self._pump_progress)

    def _build_widgets(self):
        self.canvas = ZoomableImageCanvas(self, width=CANVAS_SIZE, height=CANVAS_SIZE)
        self.canvas.grid(row=0, column=0, columnspan=4, padx=8, pady=8)
        self.canvas.bind("<Enter>", lambda event: self.canvas.focus_set())
        self.canvas.bind("<Button-1>", self._on_canvas_press)
        # Landmark click mode
        self.canvas.bind("<KeyPress-BackSpace>", self._landmark_undo)
        self.canvas.bind("<KeyPress-Escape>", self._landmark_reset)
        # Window-wide keys (any focused widget in this window): arrows browse, Enter commits.
        self.bind("<Left>", lambda event: self._show_previous_frame())
        self.bind("<Right>", lambda event: self._show_next_frame())
        self.bind("<Up>", lambda event: self._switch_seedling(-1))
        self.bind("<Down>", lambda event: self._switch_seedling(+1))
        self.bind("<Return>", self._on_enter)
        self.bind("<KP_Enter>", self._on_enter)
        self.bind("<Delete>", lambda event: self._blank_angle())   # same as "Blank this frame"
        for key in ("a", "A"):
            self.bind(f"<KeyPress-{key}>", lambda event: self._accept_reading())     # "Accept reading"
        for key in ("c", "C"):
            self.bind(f"<KeyPress-{key}>", lambda event: self._use_previous_collar())  # "Previous collar"
        for key in ("o", "O"):
            self.bind(f"<KeyPress-{key}>", lambda event: self._on_overhook_key())      # apply / remove the overhook
        for key in ("n", "N"):
            self.bind(f"<KeyPress-{key}>", lambda event: self._review_step(+1))       # next frame to review
        for key in ("p", "P"):
            self.bind(f"<KeyPress-{key}>", lambda event: self._review_step(-1))

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
        tk.Checkbutton(overlay_frame, text="Landmarks", variable=self.show_landmarks,
                        command=self._render_current_frame).pack(side=tk.LEFT)
        tk.Checkbutton(overlay_frame, text="Collar", variable=self.show_collar,
                        command=self._render_current_frame).pack(side=tk.LEFT)
        # Only meaningful when a "4" (radicle) mask exists: hidden otherwise (see _render_current_frame)
        self.germ_check = tk.Checkbutton(overlay_frame, text="Germination", variable=self.show_germ,
                                         command=self._render_current_frame)
        self.germ_check.pack(side=tk.LEFT)

        nav_frame = tk.Frame(self)
        nav_frame.grid(row=3, column=0, columnspan=4, pady=8)
        self.prev_button = tk.Button(nav_frame, text="<--", width=4, command=self._show_previous_frame, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=4)
        self.frame_label = tk.Label(nav_frame, text="No frames yet -- launch segmentation")
        self.frame_label.pack(side=tk.LEFT, padx=8)
        self.next_button = tk.Button(nav_frame, text="-->", width=4, command=self._show_next_frame, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=4)
        n_seedlings = len(getattr(self.gui, "crop_boxes", None) or [])
        self.prev_seedling_button = tk.Button(nav_frame, text="▲ Seedling", width=10,
                                              command=lambda: self._switch_seedling(-1),
                                              state=tk.NORMAL if self.crop_id > 0 else tk.DISABLED)
        self.prev_seedling_button.pack(side=tk.LEFT, padx=(24, 4))
        self.next_seedling_button = tk.Button(nav_frame, text="Seedling ▼", width=10,
                                              command=lambda: self._switch_seedling(+1),
                                              state=tk.NORMAL if self.crop_id < n_seedlings - 1 else tk.DISABLED)
        self.next_seedling_button.pack(side=tk.LEFT, padx=4)
        tk.Label(nav_frame, text="(← → frames, ↑ ↓ seedlings, O overhook, N / P next / previous to review, Del blanks the frame)", fg="#555555").pack(side=tk.LEFT, padx=(8, 0))

        self.angle_label = tk.Label(self, text="")
        self.angle_label.grid(row=4, column=0, columnspan=4, pady=4)

        manual_frame = tk.Frame(self)
        manual_frame.grid(row=5, column=0, columnspan=4, pady=(0, 8))
        tk.Label(manual_frame, text="Manual landmarks:").pack(side=tk.LEFT)
        tk.Checkbutton(manual_frame, text="Overhook", variable=self.overhook_var,
                       command=self._on_overhook_toggled).pack(side=tk.LEFT)
        self.place_angle_button = tk.Button(manual_frame, text="Place landmarks", width=14,
                                             command=self._toggle_manual_angle_mode)
        self.place_angle_button.pack(side=tk.LEFT, padx=(8, 0))
        self.replace_angle_button = tk.Button(manual_frame, text="Apply landmarks", width=14,
                                               command=self._apply_landmarks)
        self.replace_angle_button.pack(side=tk.LEFT, padx=(4, 0))
        self.blank_angle_button = tk.Button(manual_frame, text="Blank this frame", width=14,
                                             command=self._blank_angle)
        self.blank_angle_button.pack(side=tk.LEFT, padx=(4, 0))
        tk.Button(manual_frame, text="Accept reading", width=14,
                  command=self._accept_reading).pack(side=tk.LEFT, padx=(4, 0))
        tk.Button(manual_frame, text="Previous collar", width=14,
                  command=self._use_previous_collar).pack(side=tk.LEFT, padx=(4, 0))
        tk.Checkbutton(manual_frame, text="Save edits for retraining", variable=self.save_edits_var,
                       command=self._on_save_edits_toggle).pack(side=tk.LEFT, padx=(8, 0))

        germ_frame = tk.Frame(self)
        germ_frame.grid(row=6, column=0, columnspan=4, pady=(0, 8))
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

        # Review queue (utils/review_queue.py): spikes and flagged frames of every analysed seedling.
        review_frame = tk.Frame(self)
        review_frame.grid(row=7, column=0, columnspan=4, pady=(0, 8))
        tk.Label(review_frame, text="To review:").pack(side=tk.LEFT)
        tk.Button(review_frame, text="◀ Previous", width=10,
                  command=lambda: self._review_step(-1)).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(review_frame, text="Next ▶", width=10,
                  command=lambda: self._review_step(+1)).pack(side=tk.LEFT, padx=(4, 0))
        tk.Checkbutton(review_frame, text="Include unreadable", variable=self.review_unreadable_var,
                       command=self._on_review_unreadable_toggle).pack(side=tk.LEFT, padx=(8, 0))
        self.review_label = tk.Label(review_frame, text="", fg="#555555")
        self.review_label.pack(side=tk.LEFT, padx=(8, 0))

        # Hook kinetics (utils/hook_kinetics.py): the seedling's fitted curve; "auto" = chosen by BIC.
        kinetics_frame = tk.Frame(self)
        kinetics_frame.grid(row=8, column=0, columnspan=4, pady=(0, 8))
        tk.Label(kinetics_frame, text="Kinetics fit:").pack(side=tk.LEFT)
        forced = (getattr(self.gui, "kinetics_models", None) or {}).get(self.crop_id)
        self.kinetics_var = tk.StringVar(value=forced or "auto")
        self.kinetics_box = ttk.Combobox(kinetics_frame, textvariable=self.kinetics_var, width=9, state="readonly",
                                         values=("auto",) + KINETICS_MODELS)
        self.kinetics_box.pack(side=tk.LEFT, padx=(8, 0))
        self.kinetics_box.bind("<<ComboboxSelected>>", lambda event: self._on_kinetics_model())
        self.kinetics_label = tk.Label(kinetics_frame, text="", fg="#555555")
        self.kinetics_label.pack(side=tk.LEFT, padx=(8, 0))

    # --- Segmentation (background thread) -----------------------------

    def _manual_frame_count(self):
        """Frames of this seedling holding manual work (applied landmarks or a blanked angle, both
        stored as state "Manual") that a re-segmentation would discard."""
        seed_id = self._current_seed_id()
        results = self.gui.frame_results_by_crop.get(self.crop_id) or []
        return sum(1 for r in results if r is not None and r.get("state_dict", {}).get(seed_id) == "Manual")

    def _start_segmentation(self):
        if getattr(self.gui, "_busy", False):
            messagebox.showinfo("Busy", "Another analysis or export is running; try again when it is done.",
                                parent=self)
            return
        n_manual = self._manual_frame_count()
        if n_manual and not messagebox.askyesno(
                "Discard manual edits?",
                f"Re-segmenting seedling {self._current_seed_id()} discards its {n_manual} manual "
                f"edit(s) (applied landmarks and blanked frames). Germination overrides are kept.\n\n"
                "Continue?", parent=self):
            return
        self.launch_button["state"] = tk.DISABLED
        self._set_gui_busy(True)
        self.gui._snapshot_tk_state()
        threading.Thread(target=self._run_segmentation).start()

    def _set_gui_busy(self, busy):
        """Mark the main window busy so Start Analysis / Export cannot race this worker on the
        shared result store."""
        self.gui._busy = busy
        update = getattr(self.gui, "_update_button_states", None)
        if update is not None:
            update()

    def _run_segmentation(self):
        """Runs on a background thread -- must not touch Tk widgets directly. Always reports done,
        with the error if it failed, so the Launch button is never left disabled."""
        try:
            self._segment_and_process()
        except Exception as exc:
            traceback.print_exc()
            self.progress_reporter.report(done=True, error=f"Segmentation failed: {exc}")

    def _segment_and_process(self):
        self.progress_reporter.report(value=0)
        cropped_filenames = self.gui.segment_single_seedling(self.crop_id, progress_reporter=self.progress_reporter)

        results = []
        total = len(cropped_filenames)
        for idx, file_name in enumerate(cropped_filenames):
            result = self.gui.process_single_frame(file_name, self.crop_id)
            results.append(result)
            self.progress_reporter.report(value=int(((idx + 1) / total) * 100))

        # Store by reference (not a copy) in the Gui's shared per-crop results,
        # so later in-place edits here (Apply landmarks / Blank
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
        try:
            for event in self.progress_reporter.drain():
                self._handle_progress_event(event)
        finally:
            # Re-armed even if an event handler raised, so one failure never stops the pump.
            self.after(100, self._pump_progress)

    def _handle_progress_event(self, event):
        if "value" in event:
            self.progress["value"] = event["value"]
        if "message" in event:
            self.status_label.configure(text=event["message"])
        if not event.get("done"):
            return
        self.launch_button["state"] = tk.NORMAL
        self._set_gui_busy(False)
        if event.get("error"):
            self.status_label.configure(text="Segmentation failed")
            messagebox.showerror("DLhook", event["error"], parent=self)
            return
        self.cropped_filenames = event["cropped_filenames"]
        self.frame_results = event["frame_results"]
        self.current_frame = 0
        self.status_label.configure(text="Done")
        self._update_nav_state()
        self._midline_cache.clear()
        self._render_current_frame()
        self._results_changed()

    # --- Frame navigation ------------------------------------------------

    def _update_nav_state(self):
        has_frames = len(self.frame_results) > 0
        self.prev_button["state"] = tk.NORMAL if has_frames and self.current_frame > 0 else tk.DISABLED
        self.next_button["state"] = tk.NORMAL if has_frames and self.current_frame < len(self.frame_results) - 1 else tk.DISABLED
        germ_state = tk.NORMAL if has_frames else tk.DISABLED
        self.set_germ_button["state"] = germ_state
        self.germ_all_button["state"] = germ_state
        self.clear_germ_button["state"] = germ_state

    def show_frame(self, index):
        """Jump to frame `index` (clamped to the seedling's frames); a no-op before segmentation."""
        if not self.frame_results:
            return
        self.current_frame = max(0, min(int(index), len(self.frame_results) - 1))
        self._reset_manual_angle()
        self._update_nav_state()
        self._render_current_frame()

    def _show_previous_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self._reset_manual_angle()
            self._update_nav_state()
            self._render_current_frame()

    def _show_next_frame(self):
        if self.current_frame < len(self.frame_results) - 1:
            self.current_frame += 1
            self._reset_manual_angle()
            self._update_nav_state()
            self._render_current_frame()

    # --- Manual landmark edit ---------------------------------------------

    def _on_canvas_press(self, event):
        self.canvas.focus_set()
        if self._manual_mode:
            self._place_landmark_point(event)

    def _reset_manual_angle(self):
        self._session.reset()

    def _toggle_manual_angle_mode(self):
        self._manual_mode = not self._manual_mode
        self.place_angle_button.configure(bg="#d78a5e" if self._manual_mode else "white")
        if self._manual_mode:
            self.canvas.focus_set()
        else:
            self._reset_manual_angle()
        self._render_current_frame()

    def _place_landmark_point(self, event):
        image = getattr(self.canvas, "_image", None)
        if image is None:
            return
        x, y = self.canvas.canvas_to_image(event.x, event.y)
        h, w = image.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        was_angle_complete = self._session.angle_complete
        self._session.add(x, y)
        if self._session.angle_complete and not was_angle_complete:
            # C2 just placed: tick Overhook from the ray-crossing test (the user can still change it)
            try:
                self.overhook_var.set(int(self._session.result()["overhook_auto"]))
            except ValueError:
                pass
        self._render_current_frame()

    def _landmark_undo(self, event=None):
        if self._manual_mode:
            self._session.undo()
            self._render_current_frame()

    def _landmark_reset(self, event=None):
        if self._manual_mode:
            self._session.reset()
            self._render_current_frame()

    def _landmark_skip_optional(self, event=None):
        if self._manual_mode:
            self._session.skip_optional()
            self._render_current_frame()

    def _on_enter(self, event=None):
        """Enter commits the clicked landmarks (the 5 angle points are enough: collar and root stay
        optional), exactly like the Apply landmarks button."""
        if self._manual_mode and self._session.angle_complete:
            self._apply_landmarks()
        return "break"

    def _switch_seedling(self, delta):
        """Open the neighbouring seedling (delta = -1 / +1) at the same frame, in this window's place.
        Results live on the Gui, so closing this window loses nothing but unapplied clicks."""
        n_seedlings = len(getattr(self.gui, "crop_boxes", None) or [])
        target = self.crop_id + delta
        if not 0 <= target < n_seedlings:
            return
        self._replace_with(target, self.current_frame)

    def _replace_with(self, crop_id, frame):
        """Open seedling `crop_id` at `frame` in this window's place (same geometry), closing this one."""
        geometry = self.geometry()
        other = self.gui.open_seedling_frame(crop_id, frame)
        other.geometry(geometry)
        self.destroy()

    # --- Hook kinetics ----------------------------------------------------

    def _on_kinetics_model(self):
        """Force a kinetics model for this seedling ("auto" = back to the BIC choice); the plot redraws."""
        models = self.gui.__dict__.setdefault("kinetics_models", {})
        choice = self.kinetics_var.get()
        if choice == "auto":
            models.pop(self.crop_id, None)
        else:
            models[self.crop_id] = choice
        self._render_current_frame()
        self._results_changed()

    def _update_kinetics_label(self):
        """The fitted model, its phase times and the fitted angle at this frame (descriptive only)."""
        kinetics = getattr(self.gui, "kinetics_for_crop", None)
        if kinetics is None:
            self.kinetics_label.configure(text="")
            return
        fit, axis, unit, _ = kinetics(self.crop_id)
        if fit is None:
            self.kinetics_label.configure(text="not enough readable frames")
            return
        tr = fit["traits"]
        u = "h" if unit == "h" else " frames"
        parts = [f"{fit['model']}{' (forced)' if fit['forced'] else ''}"]
        if "closing_mid" in tr:
            parts.append(f"closing {tr['closing_start']:.0f}-{tr['closing_end']:.0f}{u}")
        if "maintenance_duration" in tr:
            parts.append(f"maintenance {tr['maintenance_duration']:.0f}{u}")
        if "opening_mid" in tr:
            parts.append(f"opening {tr['opening_start']:.0f}-{tr['opening_end']:.0f}{u}")
        i = self.current_frame
        if i < len(axis) and np.isfinite(axis[i]) and fit["t_first"] <= axis[i] <= fit["t_last"]:
            value = float(curve(fit["model"], fit["params"], [axis[i]])[0])
            parts.append(f"fitted {value:.0f}° here")
        if tr.get("extrapolated"):
            parts.append(f"beyond the series: {tr['extrapolated'].replace(',', ', ')}")
        self.kinetics_label.configure(text="  |  ".join(parts))

    # --- Review queue ------------------------------------------------------

    def _review_queue(self):
        return review_queue(self.gui.frame_results_by_crop,
                            include_unreadable=bool(self.review_unreadable_var.get()))

    def _on_review_unreadable_toggle(self):
        self.gui.review_include_unreadable = bool(self.review_unreadable_var.get())
        self._update_review_label()

    def _review_step(self, step):
        """Go to the next (step=+1) or previous (-1) frame to review, in this seedling or another one."""
        if self._manual_mode:
            return
        item = next_in_queue(self._review_queue(), self.crop_id, self.current_frame, step)
        if item is None:
            self.status_label.configure(text="No more frames to review" if step > 0 else
                                        "No earlier frame to review")
            return
        crop_id, frame, _ = item
        if crop_id == self.crop_id:
            self.show_frame(frame)
        else:
            self._replace_with(crop_id, frame)

    def _update_review_label(self):
        queue = self._review_queue()
        k = queue_position(queue, self.crop_id, self.current_frame)
        if not queue:
            text = "nothing left to review"
        elif k is None:
            text = f"{len(queue)} frame(s) left  (N = next, P = previous)"
        else:
            text = f"{k + 1} of {len(queue)}: {queue[k][2]}  (N = next, P = previous)"
        self.review_label.configure(text=text)

    def _session_result(self):
        """(result dict or None, error text or None) of the current clicks with the Overhook box."""
        if not self._session.angle_complete:
            return None, None
        try:
            return self._session.result(overhook=bool(self.overhook_var.get())), None
        except ValueError as exc:
            return None, str(exc)

    def _current_seed_id(self):
        return self.crop_id + 1       # seed_id = crop_id + 1 everywhere

    def _sync_angle_list(self, result):
        sync_angle_and_state_lists(result)

    def _source_frame_for(self, file_name):
        """(series, source_frame "<stem><ext>") of a crop file, from the Gui's image folder / list."""
        series = series_name_from_path(getattr(self.gui, "path", None))
        parsed = parse_crop_filename(file_name)
        stem = parsed[1] if parsed else os.path.splitext(file_name)[0]
        for name in getattr(self.gui, "file_list", None) or []:
            if os.path.splitext(name)[0] == stem:
                return series, name
        return series, stem

    def _edits_directory(self):
        """Folder for retraining records: Gui.landmark_edits_dir, else asked once (next to the last
        export when there is one). None (and the box is unticked) when the user cancels."""
        directory = getattr(self.gui, "landmark_edits_dir", None)
        if directory:
            return directory
        last = getattr(self.gui, "last_export_path", None)
        chosen = filedialog.askdirectory(parent=self, title="Folder for the landmark edits (retraining records)",
                                         initialdir=os.path.dirname(last) if last else None)
        if not chosen:
            self.save_edits_var.set(False)
            return None
        self.gui.landmark_edits_dir = chosen
        return chosen

    def _on_save_edits_toggle(self):
        if self.save_edits_var.get():
            self._edits_directory()

    def _save_edit_record(self, file_name, seed_id, manual, model_readout, overhook):
        directory = self._edits_directory()
        if directory is None:
            return
        series, source_frame = self._source_frame_for(file_name)
        boxes = getattr(self.gui, "crop_boxes", None) or []
        box = boxes[self.crop_id] if self.crop_id < len(boxes) else None
        record = edit_record(series, source_frame, file_name, box, seed_id, self.crop_id,
                             manual, model_readout, overhook)
        write_edit(directory, record, crop_path(file_name))

    def _apply_landmarks(self):
        """Commit the clicked landmarks as this frame's Manual reading (needs the 5 angle points)."""
        if not self.frame_results or not self._session.angle_complete:
            return
        result = self.frame_results[self.current_frame]
        if result is None:
            return
        try:
            manual = self._session.result(overhook=bool(self.overhook_var.get()))
        except ValueError as exc:
            messagebox.showwarning("Cannot compute angle", str(exc), parent=self)
            return
        self._commit_manual(result, manual)
        # Done with this frame: leave click mode, unless edits are being collected for retraining
        # (then the user is likely to correct frame after frame).
        if self._manual_mode and not self.save_edits_var.get():
            self._toggle_manual_angle_mode()

    # --- Overhook on a reading ---------------------------------------------

    @staticmethod
    def _overhook_applied(result, seed_id):
        """True when this frame's angle is 180 + theta: a Manual reading committed with the overhook.
        The model's own overhook flag (`readout["overhook"]`) is only a suggestion until it is applied."""
        readout = (result.get("landmarks") or {}).get(seed_id)
        return bool(readout and (result.get("state_dict") or {}).get(seed_id) == "Manual"
                    and readout.get("edit") == "manual" and readout.get("overhook"))

    def _on_overhook_key(self):
        """O: while placing landmarks, tick / untick Overhook; otherwise apply the overhook to this
        frame's reading (angle = 180 + theta), or remove it when it is applied."""
        if self._manual_mode:
            self.overhook_var.set(0 if self.overhook_var.get() else 1)
            self._render_current_frame()
            return
        if not self.frame_results or self.frame_results[self.current_frame] is None:
            return
        result = self.frame_results[self.current_frame]
        self._set_overhook(not self._overhook_applied(result, self._current_seed_id()))
        self._render_current_frame()       # also puts the Overhook box back in line with the frame

    def _on_overhook_toggled(self):
        """The Overhook box was clicked: while placing landmarks it only re-draws the prompt; otherwise
        it applies / removes the overhook on this frame's reading."""
        if not self._manual_mode:
            self._set_overhook(bool(self.overhook_var.get()))
        self._render_current_frame()

    def _set_overhook(self, flag):
        """Keep this frame's landmark reading and set whether the overhook is applied to its angle
        (180 + theta when `flag`, else 180 - theta). A model reading becomes a Manual one, like
        `_accept_reading`; an already Manual reading is changed in place. False when nothing was done."""
        result = self.frame_results[self.current_frame] if self.frame_results else None
        if result is None or self._manual_mode:
            return False
        seed_id = self._current_seed_id()
        readout = (result.get("landmarks") or {}).get(seed_id)
        if not readout or readout.get("junction") is None:
            self.status_label.configure(text="No reading on this frame -- place the landmarks")
            return False
        if (result.get("state_dict") or {}).get(seed_id) != "Manual":
            if not flag:
                return True                  # the model reading already has no overhook applied
            return self._accept_reading(overhook=True)
        if readout.get("edit") != "manual":
            self.status_label.configure(text="Frame blanked -- place the landmarks to give it an angle")
            return False
        if bool(readout.get("overhook")) == bool(flag):
            return True
        theta = float(readout["theta"])
        readout["overhook"] = bool(flag)
        readout["bio"] = 180.0 + theta if flag else 180.0 - theta
        result["angle_dict"][seed_id] = readout["bio"]
        self._sync_angle_list(result)
        if self.save_edits_var.get():
            directory = getattr(self.gui, "landmark_edits_dir", None)
            if directory:
                series, _ = self._source_frame_for(self.cropped_filenames[self.current_frame])
                set_record_overhook(directory, series, self.cropped_filenames[self.current_frame], flag)
        self.gui._reconstruct_series_for_crop(self.crop_id)
        self._results_changed()
        self.status_label.configure(text="Overhook applied" if flag else "Overhook removed")
        return True

    def _accept_reading(self, overhook=False):
        """Keep the model's reading of this frame as shown -- also a grey (gated) or amber (flagged)
        one -- as a Manual frame, with the model's angle (180 - theta; 180 + theta with `overhook`).
        Same commit as Apply landmarks. True when the reading was committed."""
        if not self.frame_results or self._manual_mode:
            return False
        result = self.frame_results[self.current_frame]
        if result is None:
            return False
        seed_id = self._current_seed_id()
        if (result.get("state_dict") or {}).get(seed_id) == "Manual":
            self.status_label.configure(text="Already manual")
            return False
        readout = (result.get("landmarks") or {}).get(seed_id)
        session = session_from_readout(readout)
        if session is None:
            self.status_label.configure(text="No model reading to accept -- place the landmarks")
            return False
        try:
            manual = session.result(overhook=bool(overhook))
        except ValueError as exc:
            messagebox.showwarning("Cannot accept reading", str(exc), parent=self)
            return False
        # Keep exactly the theta the window showed (a gated frame keeps it as gated_theta).
        shown = readout.get("gated_theta", readout.get("theta")) if readout.get("gated") else readout.get("theta")
        if shown is not None and np.isfinite(shown):
            manual["theta"] = float(shown)
            manual["bio"] = 180.0 + float(shown) if overhook else 180.0 - float(shown)
        manual["accepted"] = True          # a model reading the user confirmed, not clicked
        self._commit_manual(result, manual)
        self.status_label.configure(text="Reading accepted with the overhook" if overhook else "Reading accepted")
        return True

    def _use_previous_collar(self):
        """Give this frame the collar of the nearest earlier frame whose collar is trusted (Manual, or
        readable and not collar-flagged). The angle is unchanged; the traits and the time-series flags
        follow the new collar."""
        if not self.frame_results or self._manual_mode:
            return
        result = self.frame_results[self.current_frame]
        if result is None:
            return
        seed_id = self._current_seed_id()
        readout = (result.get("landmarks") or {}).get(seed_id)
        if not readout or readout.get("junction") is None:
            self.status_label.configure(text="No reading on this frame -- place the landmarks")
            return
        found = previous_collar(self.frame_results, seed_id, self.current_frame)
        if found is None:
            self.status_label.configure(text="No earlier frame with a trusted collar")
            return
        k, collar, root_dir = found
        readout["collar"] = collar
        readout["root_dir"] = root_dir
        readout["collar_edit"] = "manual"
        file_name = self.cropped_filenames[self.current_frame]
        result.setdefault("traits", {})[seed_id] = hypocotyl_traits(
            self.gui._trait_mask(file_name, result.get("hypo_mask")), readout["junction"], collar)
        self._midline_cache.pop(self.current_frame, None)
        self.gui._reconstruct_series_for_crop(self.crop_id)      # re-derives the time-series flags
        self._render_current_frame()
        self._results_changed()
        self.status_label.configure(text=f"Collar from frame {k + 1}")

    def _commit_manual(self, result, manual):
        """Store `manual` (a LandmarkClickSession.result) as this frame's Manual reading."""
        file_name = self.cropped_filenames[self.current_frame]
        seed_id = self._current_seed_id()
        model_readout = (result.get("landmarks") or {}).get(seed_id)

        if self.save_edits_var.get():
            # built from the clicks alone, before the model's collar is borrowed below
            self._save_edit_record(file_name, seed_id, manual, model_readout, manual["overhook"])

        # No collar clicked: keep the model's, so length and the collar columns stay consistent.
        if manual["collar"] is None and model_readout and model_readout.get("collar") is not None:
            manual["collar"] = model_readout["collar"]
            manual["root_dir"] = model_readout.get("root_dir")
        traits = hypocotyl_traits(self.gui._trait_mask(file_name, result.get("hypo_mask")),
                                  manual["junction"], manual["collar"])

        # Manual angles are stored already in the bio convention (180 = closed, > 180 overhooked).
        result["angle_dict"][seed_id] = manual["bio"]
        result.setdefault("state_dict", {})[seed_id] = "Manual"
        result.setdefault("landmarks", {})[seed_id] = manual
        result.setdefault("traits", {})[seed_id] = traits
        self._sync_angle_list(result)
        # A manual override is a fixed anchor for the temporal reconstruction,
        # so re-run it over this seedling's whole series: the corrected frame
        # pulls the surrounding frames onto the right branch (see
        # Gui._reconstruct_series_for_crop). Same contract as _blank_angle
        # below; it reconstructs from each frame's untouched raw reading, so
        # repeated overrides compose instead of re-smoothing a reconstructed value.
        self._midline_cache.pop(self.current_frame, None)
        self.gui._reconstruct_series_for_crop(self.crop_id)
        self._reset_manual_angle()
        self._render_current_frame()
        self._results_changed()

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
        self._midline_cache.pop(self.current_frame, None)
        # Same re-anchoring as _apply_landmarks: a blanked frame stays NaN
        # (_align_branches skips missing frames, and the reconstruction never
        # writes back to a "Manual" one), but dropping a bad reading out of the
        # branch-alignment input lets its neighbours re-resolve.
        self.gui._reconstruct_series_for_crop(self.crop_id)
        self._render_current_frame()
        self._results_changed()

    def _results_changed(self):
        """Tell the Gui a result changed, so an open kinematics plot redraws (main thread)."""
        notify = getattr(self.gui, "_notify_results_changed", None)
        if notify is not None:
            notify()

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
        self._results_changed()

    def _apply_germination_to_all(self):
        """Same germination frame for every seedling on the plate -- the usual
        case, since one plate is imaged as one time series."""
        if not self.frame_results:
            return
        detector = self._germination_detector()
        for crop_id in range(len(self.gui.crop_boxes)):
            detector.set_override(crop_id, self.current_frame)
        self._update_germination_label()
        self._results_changed()

    def _clear_germination_frame(self):
        """Drop the manual override, falling back to automatic detection."""
        self._germination_detector().clear_override(self.crop_id)
        self._update_germination_label()
        self._results_changed()

    def _update_germination_label(self):
        detector = self.gui.germination_detector
        time_zero = detector.get_time_zero(self.crop_id) if detector is not None else None
        if time_zero is None:
            self.germ_label.configure(text="not set")
            return
        # Distinguishing manual from detected matters: it's the difference
        # between "the model found this" and "I told it this".
        method = detector.method(self.crop_id)
        source = {"manual": "manual", "learned": "detected (learned)"}.get(method, "detected (area rule)")
        self.germ_label.configure(text=f"{source}: frame {time_zero + 1}")

    # --- Rendering --------------------------------------------------------

    def _render_current_frame(self):
        if not self.frame_results:
            return

        file_name = self.cropped_filenames[self.current_frame]
        result = self.frame_results[self.current_frame]
        self.frame_label.configure(text=f"{file_name} ({self.current_frame + 1}/{len(self.frame_results)})")
        self._update_germination_label()
        self._update_review_label()
        self._update_kinetics_label()

        if result is None:
            self.angle_label.configure(text="No mask/segmentation data for this frame.")
            return

        # The Germination overlay needs a "4" mask, which the default multiclass backend never writes.
        has_germ = result.get("germ_mask") is not None
        if has_germ and not self._germ_check_shown:
            self.germ_check.pack(side=tk.LEFT)
            self._germ_check_shown = True
        elif not has_germ and self._germ_check_shown:
            self.germ_check.pack_forget()
            self._germ_check_shown = False

        # The raw crop is shown unmodified; every overlay is a thin line (see ZoomableImageCanvas).
        image = cv2.imread(crop_path(file_name), cv2.IMREAD_COLOR)
        seed_id = self._current_seed_id()
        readout = (result.get("landmarks") or {}).get(seed_id)
        if not self._manual_mode:       # outside click mode the box shows whether the overhook is applied here
            self.overhook_var.set(int(self._overhook_applied(result, seed_id)))

        shapes = []
        if self.show_cotyl.get():
            shapes += mask_contour_shapes(result.get("cotyl_mask"), COTYL_CONTOUR_COLOR)
        if has_germ and self.show_germ.get():
            shapes += mask_contour_shapes(result["germ_mask"], GERM_CONTOUR_COLOR)
        if self.show_hypo.get():
            midline = self._midline_for(file_name, result, readout)
            if midline is not None:
                shapes.append({"kind": "polyline", "pts": (midline + 0.5).tolist(), "color": MIDLINE_COLOR})

        # Landmarks go on as canvas vector items, at a fixed screen size, keeping them crisp at any
        # zoom. While placing, the clicked points (in progress) are drawn even with the box off.
        if self._manual_mode and self._session.points:
            shapes += landmark_shapes(points=self._session.named_points())
        elif self.show_landmarks.get() and not self._manual_mode:
            manual = result.get("state_dict", {}).get(seed_id) == "Manual"
            gated = bool(readout and readout.get("gated") and not manual)
            flagged = bool(readout and readout.get("temporal_flag") and not manual and not gated)
            muted = gated or self._is_blanked(result, seed_id)      # no angle: the reading is not used
            show_collar = self.show_collar.get()
            shapes += landmark_shapes(readout=readout, collar=show_collar,
                                      color=MUTED_COLOR if muted else FLAG_COLOR if flagged else None)
            collar_ref = readout.get("collar_ref") if readout else None
            if flagged and show_collar and collar_ref is not None:     # where the seedling's collar usually is
                shapes.append({"kind": "point", "xy": tuple(collar_ref), "color": FLAG_COLOR, "hollow": True})
        self.canvas.set_image(image, overlay=shapes)

        self.angle_label.configure(text=self._angle_text(result, seed_id, readout))

    @staticmethod
    def _is_blanked(result, seed_id):
        """A frame blanked by hand: Manual with no angle (applied landmarks always give one)."""
        if (result.get("state_dict") or {}).get(seed_id) != "Manual":
            return False
        angle = (result.get("angle_dict") or {}).get(seed_id)
        return angle is None or not np.isfinite(angle)

    def _midline_for(self, file_name, result, readout):
        """Smoothed hypocotyl midline ((N, 2) x, y) of the current frame, on the mask the traits use
        and the frame's current junction/collar; None when there is no junction or no usable mask.
        Cached per frame index and recomputed when the junction/collar change (Apply landmarks)."""
        junction = readout.get("junction") if readout else None
        collar = readout.get("collar") if readout else None
        signature = (None if junction is None else tuple(float(v) for v in junction),
                     None if collar is None else tuple(float(v) for v in collar))
        cached = self._midline_cache.get(self.current_frame)
        if cached is not None and cached[0] == signature:
            return cached[1]
        midline = None
        if junction is not None:
            mask = self.gui._trait_mask(file_name, fallback=result.get("hypo_mask"))
            if mask is not None:
                midline = hypocotyl_midline(mask, junction, collar)
        self._midline_cache[self.current_frame] = (signature, midline)
        return midline

    def _overhook_text(self, result, seed_id, readout):
        """One line saying whether the overhook is part of this frame's angle (180 + theta) or not
        (180 - theta), and what O would do. Empty when the frame has no angle to apply it to."""
        if self._manual_mode or not readout or readout.get("junction") is None:
            return ""
        theta = readout.get("theta")
        manual = (result.get("state_dict") or {}).get(seed_id) == "Manual"
        if theta is None or not np.isfinite(theta) or self._is_blanked(result, seed_id):
            return ""
        if readout.get("gated") and not manual:
            return ""
        closed, over = 180.0 - theta, 180.0 + theta
        if self._overhook_applied(result, seed_id):
            return f"\noverhook applied: angle = 180 + θ = {over:.0f}°  (O removes it: {closed:.0f}°)"
        suggested = bool(readout.get("overhook")) and not manual
        lead = "the model suggests an overhook, not applied" if suggested else "overhook not applied"
        return f"\n{lead}: angle = 180 − θ = {closed:.0f}°  (O applies it: {over:.0f}°)"

    def _angle_text(self, result, seed_id, readout):
        angle_dict = result["angle_dict"]
        if angle_dict:
            text = ", ".join(
                f"seedling {sid}: {round(a) if isinstance(a, (int, float)) and not np.isnan(a) else '-'}"
                for sid, a in angle_dict.items()
            )
        else:
            text = "no angle computed for this frame"

        if result.get("state_dict", {}).get(seed_id) == "Manual":
            source = "manual"
        else:
            source = result.get("angle_source", "ellipse")
        text += f"  |  source: {source}"
        if source == "manual" and self._is_blanked(result, seed_id):
            text += "\nblanked by hand -- place the landmarks to give it an angle"
        if readout and readout.get("peak") is not None and source == "landmarks":
            text += f" (peak {readout['peak']:.2f})"
        if source == "landmarks" and readout and readout.get("gated"):
            gt = readout.get("gated_theta")
            p_unreadable = readout.get("unreadable_p")
            p_text = "-" if p_unreadable is None else f"{p_unreadable:.2f}"
            theta_text = "-" if gt is None else f"{gt:.0f}°"
            text += (f"\nno reading (unreadable p={p_text}; model θ={theta_text})"
                     " -- A accepts it as shown, or place the landmarks")
        elif readout and readout.get("series_edit") == "rejected" and source != "manual":
            theta = readout.get("theta")
            theta_text = "-" if theta is None else f"{180 - theta:.0f}°"
            text += (f"\nreading {theta_text} rejected by the series filter (far from its neighbours)"
                     " -- A accepts it as shown, or place the landmarks")
        elif readout and readout.get("temporal_flag") and source != "manual":
            text += ("\n⚠ " + describe_flag(readout["temporal_flag"], readout.get("temporal_collar_dev"))
                     + " -- A accepts, C takes the previous collar, Del blanks")

        text += self._overhook_text(result, seed_id, readout)

        traits = (result.get("traits") or {}).get(seed_id)
        if traits and isinstance(traits.get("length_px"), (int, float)) and not np.isnan(traits["length_px"]):
            mm = to_mm(traits, getattr(self.gui, "scale_um_per_px", None))
            length = f"{traits['length_px']:.0f} px" + (f" ({mm['length_mm']:.2f} mm)" if mm else "")
            text += f"\nhypocotyl length {length}"
            if traits["length_truncated"]:
                text += " (truncated)"
            diameter = traits.get("diameter_px")
            if isinstance(diameter, (int, float)) and not np.isnan(diameter):
                text += f", diameter {diameter:.1f} px" + (f" ({mm['diameter_mm']:.3f} mm)" if mm else "")

        if self._manual_mode:
            manual, error = self._session_result()
            text += "\nlandmarks: " + self._session.prompt()
            if manual is not None:
                text += (f"   |   theta {manual['theta']:.1f}\u00b0, bio {manual['bio']:.1f}\u00b0"
                         f"{'  (overhook applied: 180 + theta)' if manual['overhook'] else '  (no overhook: 180 - theta)'}"
                         "  (Apply landmarks to commit)")
            elif error:
                text += f"   |   {error}"
            text += "\n[Backspace = undo point, Esc = reset, Enter = apply (collar/root optional)]"
        return text
