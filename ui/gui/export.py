"""Export results (background worker building the long-format CSV) and the kinematics window.

A mixin of seedling_measurement.Gui: the methods run on the Gui instance and use its state."""
import threading
import traceback
import tkinter as tk
from tkinter import messagebox
from tkinter.filedialog import asksaveasfilename

import pandas as pd

from ui.kinematics_window import KinematicsWindow
from utils.export_rows import build_export_row, export_columns


class ExportMixin:
    def _export_results(self):
        """Replaces the removed legacy window's "Save data": consolidates
        every seedling's current angle data -- including manual landmark edits
        and blanked frames made in SeedlingAnalysisWindow -- into one
        long-format CSV (one row per seedling per frame). Runs on a background thread since
        it may need to segment any seedling never previewed."""
        self._exporting = True
        self._busy = True                   # excludes Start Analysis / re-segmentation meanwhile
        self._update_button_states()
        self._snapshot_tk_state()
        threading.Thread(target=self._run_export_results).start()

    def _run_export_results(self):
        """Worker-thread entry point: always reports export_done, with the error if it failed."""
        export_df, error = None, None
        try:
            export_df = self._build_export_df()
        except Exception as exc:
            traceback.print_exc()
            error = f"Export failed: {exc}"
        finally:
            self.progress_reporter.report(value=100, message="Export ready" if error is None else "Export failed",
                                          export_df=export_df, export_done=True, error=error)

    def _build_export_df(self):
        n_crops = len(self.crop_boxes)
        for crop_id in range(n_crops):
            if crop_id in self.frame_results_by_crop:
                continue
            self.progress_reporter.report(message=f"Segmenting seedling {crop_id + 1}...",
                                           value=int((crop_id / n_crops) * 80))
            cropped_filenames = self.segment_single_seedling(crop_id, progress_reporter=self.progress_reporter)
            results = [self.process_single_frame(file_name, crop_id) for file_name in cropped_filenames]
            self.frame_results_by_crop[crop_id] = results
            self.cropped_filenames_by_crop[crop_id] = cropped_filenames

        self.progress_reporter.report(message="Detecting germination...", value=85)
        for crop_id in range(n_crops):
            self._ensure_germination_detected(crop_id)

        self.progress_reporter.report(message="Reconstructing angle-through-time...", value=87)
        for crop_id in range(n_crops):
            self._reconstruct_series_for_crop(crop_id)

        self.progress_reporter.report(message="Building CSV...", value=90)
        um_per_px = getattr(self, "scale_um_per_px", None)
        columns = export_columns(um_per_px)
        rows = []
        for frame_idx, filename in enumerate(self.file_list):
            elapsed = self.time_deltas[frame_idx].get("elapsed_minutes") if frame_idx < len(self.time_deltas) else None
            for crop_id in range(n_crops):
                seed_id = crop_id + 1
                frame_results = self.frame_results_by_crop.get(crop_id, [])
                result = frame_results[frame_idx] if frame_idx < len(frame_results) else None
                germ_frame = self.germination_detector.get_time_zero(crop_id) if self.germination_detector else None
                rows.append(build_export_row(filename, seed_id, result, elapsed, germ_frame, um_per_px))
        return pd.DataFrame(rows, columns=columns)

    def _save_export(self, export_df):
        file_formats = [('CSV-file', '*.csv')]
        save_file_path = asksaveasfilename(filetypes=file_formats, defaultextension=file_formats)
        if not save_file_path:
            return
        try:
            export_df.to_csv(save_file_path, index=False)
        except OSError as exc:
            messagebox.showerror("Export failed", f"Could not write {save_file_path}:\n{exc}", parent=self.root)
            return
        self.last_export_path = save_file_path

    def _open_kinematics_window(self):
        """Phase 8: opens the aggregate kinematics graph, reading whatever is
        currently in frame_results_by_crop -- works from Start Analysis,
        Preview seedling, or Export results alike. An open window is raised and
        redrawn rather than duplicated, so there is one plot to keep up to date."""
        window = self._live_kinematics_window()
        if window is not None:
            window.lift()
            window.refresh()
            return
        self.kinematics_window = KinematicsWindow(self)

    def _live_kinematics_window(self):
        """The open KinematicsWindow, or None (never opened, or closed since)."""
        window = getattr(self, "kinematics_window", None)
        if window is None:
            return None
        try:
            alive = bool(window.winfo_exists())
        except tk.TclError:
            alive = False
        if not alive:
            self.kinematics_window = None
            return None
        return window

    def _notify_results_changed(self):
        """Main thread only. Called after an edit in a seedling's analysis window (landmarks applied,
        frame blanked, germination set, seedling re-segmented): the open kinematics plot redraws."""
        window = self._live_kinematics_window()
        if window is not None:
            window.schedule_refresh()
