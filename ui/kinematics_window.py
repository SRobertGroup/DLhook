import tkinter as tk

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

STATE_COLORS = {
    "Closed": "tab:blue",
    "Opening": "tab:orange",
    "Manual": "tab:purple",
}
UNKNOWN_STATE_COLOR = "0.5"


class KinematicsWindow(tk.Toplevel):
    """
    Phase 8: aggregate view of every seedling's angle over time, points
    color-coded by the reconstructed bio_state (Closed/Opening/Manual, see
    utils/angle_timeseries.py).

    Manual-refresh only (confirmed with user) -- reads directly from
    Gui.frame_results_by_crop/time_deltas/germination_detector on each
    "Refresh" click, so it works with whatever data currently exists
    (Start Analysis, Preview seedling, or Export results all populate the
    same shared store), with no live-updating plumbing required.

    Each seedling's x-axis is zeroed at its own detected germination frame
    (Phase 4's GerminationDetector). A seedling with no detected germination
    yet is plotted on raw elapsed time instead, drawn dashed and labeled
    "(ungerminated)" so it's visually distinct rather than silently
    misleading (confirmed with user).
    """

    def __init__(self, gui):
        super().__init__(gui.root)
        self.gui = gui
        self.title("Kinematics")

        self.figure = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        tk.Button(self, text="Refresh", command=self.refresh).pack(pady=6)

        self.refresh()

    def refresh(self):
        self.ax.clear()

        n_frames = len(self.gui.file_list) if getattr(self.gui, "file_list", None) else 0
        use_time_deltas = n_frames > 0 and len(self.gui.time_deltas) == n_frames
        if use_time_deltas:
            x_axis_all = np.array(
                [np.nan if (d is None or d.get("elapsed_minutes") is None) else d["elapsed_minutes"]
                 for d in self.gui.time_deltas],
                dtype=float,
            )
        else:
            x_axis_all = np.arange(n_frames, dtype=float)

        detector = self.gui.germination_detector
        any_plotted = False

        for crop_id, frame_results in sorted(self.gui.frame_results_by_crop.items()):
            seed_id = crop_id + 1
            time_zero = detector.get_time_zero(crop_id) if detector is not None else None
            unaligned = time_zero is None or time_zero >= len(x_axis_all)

            xs, ys, colors = [], [], []
            for frame_idx, result in enumerate(frame_results):
                if result is None or frame_idx >= len(x_axis_all):
                    continue
                angle = result["angle_dict"].get(seed_id)
                if not isinstance(angle, (int, float)) or np.isnan(angle):
                    continue
                x = x_axis_all[frame_idx]
                if np.isnan(x):
                    continue
                if not unaligned:
                    x = x - x_axis_all[time_zero]
                xs.append(x)
                ys.append(angle)
                state = result.get("state_dict", {}).get(seed_id)
                colors.append(STATE_COLORS.get(state, UNKNOWN_STATE_COLOR))

            if not xs:
                continue
            any_plotted = True
            order = np.argsort(xs)
            xs_sorted = np.array(xs)[order]
            ys_sorted = np.array(ys)[order]
            colors_sorted = [colors[i] for i in order]

            style = "--" if unaligned else "-"
            label = f"Seedling {seed_id}" + (" (ungerminated)" if unaligned else "")
            self.ax.plot(xs_sorted, ys_sorted, style, color="0.7", linewidth=1, zorder=1, label=label)
            self.ax.scatter(xs_sorted, ys_sorted, c=colors_sorted, zorder=2, s=18)

        if not any_plotted:
            self.ax.text(0.5, 0.5, "No results yet -- run Start Analysis or preview a seedling",
                         ha="center", va="center", transform=self.ax.transAxes)
        else:
            x_label = "Time since germination (min)" if use_time_deltas else "Frame index since germination"
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel("Angle (deg)")
            self.ax.axhline(180, color="0.85", linewidth=1, zorder=0)
            self.ax.legend(fontsize=8, loc="best")

        self.canvas.draw_idle()
