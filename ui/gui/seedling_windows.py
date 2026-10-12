"""Opening per-seedling analysis windows (picker, or a given seedling and frame).

A mixin of seedling_measurement.Gui: the methods run on the Gui instance and use its state."""
import tkinter as tk

from ui.analysis_window import SeedlingAnalysisWindow


class SeedlingWindowsMixin:
    def open_seedling_frame(self, crop_id, frame_index=None):
        """Open (or bring forward) the analysis window of seedling `crop_id` (0-based), showing
        `frame_index` when given -- used by the seedling picker and by clicking a point in the
        kinematics window. An already-open window for that seedling is reused, not duplicated."""
        window = None
        for existing in list(self._analysis_windows):
            try:
                alive = bool(existing.winfo_exists())
            except tk.TclError:
                alive = False
            if not alive:
                self._analysis_windows.remove(existing)
            elif existing.crop_id == crop_id:
                window = existing
        if window is None:
            window = SeedlingAnalysisWindow(self, crop_id)
            self._analysis_windows.append(window)
        if frame_index is not None:
            window.show_frame(frame_index)
        window.deiconify()
        window.lift()
        window.focus_force()
        return window

    def open_seedling_picker(self):
        """Phase 5: lightweight Toplevel listing each seedling; opens a
        SeedlingAnalysisWindow for whichever one the user picks, on demand,
        rather than spawning a window per seedling all at once."""
        if not self.crop_boxes:
            return

        picker = tk.Toplevel(self.root)
        picker.title("Preview a seedling")

        listbox = tk.Listbox(picker, width=20, height=min(10, len(self.crop_boxes)))
        for i in range(len(self.crop_boxes)):
            # Labelled 1-based to match the kinematics legend and the export's
            # seedling_id; the row index handed to SeedlingAnalysisWindow below
            # is still the 0-based crop_id.
            listbox.insert(tk.END, f"Seedling {i + 1}")
        listbox.pack(padx=8, pady=8)

        def _open_selected():
            selection = listbox.curselection()
            if not selection:
                return
            self.open_seedling_frame(selection[0])

        listbox.bind("<Double-Button-1>", lambda event: _open_selected())
        tk.Button(picker, text="Open", command=_open_selected).pack(pady=(0, 8))
