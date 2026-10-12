import tkinter as tk

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

STATE_COLORS = {
    "Closed": "tab:blue",
    "Opening": "tab:orange",
    "Manual": "tab:purple",
}
UNKNOWN_STATE_COLOR = "0.5"
# Transparency of the per-seedling curves and their points (overlapping seedlings stay readable).
CURVE_ALPHA = 0.35
POINT_ALPHA = 0.6
UNKNOWN_STATE_LABEL = "Unclassified"
# Points whose landmarks broke a time-series rule (utils/landmark_temporal.py) get a ring.
FLAG_EDGE_COLOR = "black"
FLAG_RING_WIDTH = 1.2
FLAG_LEGEND_LABEL = "Temporal flag"

# Line color per seedling, so a seedling stays identifiable even where its
# points all share one state color. Deliberately disjoint from STATE_COLORS'
# blue/orange/purple, which mean something different on the same axes.
SEEDLING_LINE_COLORS = [
    "tab:green", "tab:red", "tab:brown", "tab:pink",
    "tab:olive", "tab:cyan", "tab:gray", "black",
]


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

    Each seedling's x-axis is zeroed at its own germination frame -- either
    detected by GerminationDetector or set by hand in SeedlingAnalysisWindow,
    which takes precedence. A seedling with no germination frame at all is
    plotted on raw elapsed time instead, drawn dashed and labeled
    "(ungerminated)" so it's visually distinct rather than silently
    misleading (confirmed with user).

    Points are colored by state (see the legend), lines by seedling; a black ring marks a frame whose
    landmarks broke a time-series rule (utils/landmark_temporal.py).
    """

    def __init__(self, gui):
        super().__init__(gui.root)
        self.gui = gui
        self.title("Kinematics")

        self.figure = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Clicking a point opens that seedling's analysis window at that frame.
        self._pick_targets = {}      # scatter artist -> (crop_id, frame indices in plotted order)
        self.canvas.mpl_connect("pick_event", self._on_pick)

        bottom = tk.Frame(self)
        bottom.pack(pady=6)
        tk.Button(bottom, text="Refresh", command=self.refresh).pack(side=tk.LEFT)
        tk.Label(bottom, text="Click a point to open that seedling at that frame", fg="#555555").pack(
            side=tk.LEFT, padx=(12, 0))

        self.refresh()

    def _on_pick(self, event):
        target = self._pick_targets.get(event.artist)
        if target is None or not len(event.ind):
            return
        crop_id, frame_indices = target
        # Several overlapping points can be hit at once: take the one nearest the click.
        if event.mouseevent is not None and event.mouseevent.xdata is not None and len(event.ind) > 1:
            offsets = event.artist.get_offsets()
            px, py = self.ax.transData.transform((event.mouseevent.xdata, event.mouseevent.ydata))
            pts = self.ax.transData.transform(offsets[event.ind])
            best = event.ind[int(np.argmin(np.hypot(pts[:, 0] - px, pts[:, 1] - py)))]
        else:
            best = event.ind[0]
        self.gui.open_seedling_frame(crop_id, frame_indices[best])

    def refresh(self):
        self.ax.clear()
        self._pick_targets = {}

        n_frames = len(self.gui.file_list) if getattr(self.gui, "file_list", None) else 0
        use_time_deltas = n_frames > 0 and len(self.gui.time_deltas) == n_frames
        if use_time_deltas:
            x_axis_all = np.array(
                [np.nan if (d is None or d.get("elapsed_minutes") is None) else d["elapsed_minutes"]
                 for d in self.gui.time_deltas],
                dtype=float,
            )
            # All-NaN happens when the images carried no usable timestamps and
            # the user cancelled the fallback-interval prompt. Without this the
            # window silently reported "No results yet" for a fully analysed run,
            # since every point was dropped for having a NaN x.
            if not np.any(np.isfinite(x_axis_all)):
                use_time_deltas = False
        if not use_time_deltas:
            x_axis_all = np.arange(n_frames, dtype=float)

        detector = self.gui.germination_detector
        any_plotted = False
        # Which encodings actually occur, so each legend lists only what's on
        # screen rather than every state the reconstruction can produce.
        states_seen = []
        any_flagged = False
        seedling_handles = []

        for crop_id, frame_results in sorted(self.gui.frame_results_by_crop.items()):
            seed_id = crop_id + 1
            time_zero = detector.get_time_zero(crop_id) if detector is not None else None
            unaligned = time_zero is None or time_zero >= len(x_axis_all)

            xs, ys, colors, frames, flagged = [], [], [], [], []
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
                frames.append(frame_idx)
                state = result.get("state_dict", {}).get(seed_id)
                colors.append(STATE_COLORS.get(state, UNKNOWN_STATE_COLOR))
                readout = (result.get("landmarks") or {}).get(seed_id) or {}
                flagged.append(bool(readout.get("temporal_flag")) and state != "Manual")
                state_label = state if state in STATE_COLORS else UNKNOWN_STATE_LABEL
                if state_label not in states_seen:
                    states_seen.append(state_label)

            if not xs:
                continue
            any_plotted = True
            order = np.argsort(xs)
            xs_sorted = np.array(xs)[order]
            ys_sorted = np.array(ys)[order]
            colors_sorted = [colors[i] for i in order]
            # per-point ring width (not edge colour: the scatter's alpha would make "none" edges show)
            ring = [FLAG_RING_WIDTH if flagged[i] else 0.0 for i in order]
            any_flagged = any_flagged or any(flagged)

            style = "--" if unaligned else "-"
            line_color = SEEDLING_LINE_COLORS[crop_id % len(SEEDLING_LINE_COLORS)]
            # minutes -> hours on a time axis; a frame-index axis is plotted as is
            xs_plot = xs_sorted / 60 if use_time_deltas else xs_sorted
            # Semi-transparent, so overlapping seedlings stay readable and outliers stand out.
            self.ax.plot(xs_plot, ys_sorted, style, color=line_color, linewidth=1, alpha=CURVE_ALPHA, zorder=1)
            points = self.ax.scatter(xs_plot, ys_sorted, c=colors_sorted, zorder=2, s=18, alpha=POINT_ALPHA,
                                     edgecolors=FLAG_EDGE_COLOR, linewidths=ring, picker=5)
            self._pick_targets[points] = (crop_id, [frames[i] for i in order])

        if not any_plotted:
            self.ax.text(0.5, 0.5, "No results yet -- run Start Analysis or preview a seedling",
                         ha="center", va="center", transform=self.ax.transAxes)
        else:
            x_label = "Time since germination (h)" if use_time_deltas else "Frame index since germination"
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel("Angle (deg)")
            self.ax.axhline(180, color="0.85", linewidth=1, zorder=0)

            # Two legends, because the plot carries two independent encodings:
            # marker color = per-point state, line color = which seedling. The
            # single legend this used to draw labelled only the (then uniformly
            # grey) lines, so nothing on screen explained the point colors.
            state_handles = [
                Line2D([], [], marker="o", linestyle="none",
                       color=STATE_COLORS.get(s, UNKNOWN_STATE_COLOR), label=s)
                for s in states_seen
            ]
            if any_flagged:
                state_handles.append(Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                                            markeredgecolor=FLAG_EDGE_COLOR, label=FLAG_LEGEND_LABEL))
            if state_handles:
                self.ax.legend(handles=state_handles, title="State",
                                fontsize=8, title_fontsize=8, loc="upper right")

        self.canvas.draw_idle()
