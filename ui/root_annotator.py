#!/usr/bin/env python
"""Collar and root-direction annotation tool.

Re-opens the frames you already measured with ui/angle_annotator.py and asks for
the other end of the seedling: is a radicle visible, and if so where is the plant
COLLAR (the root-hypocotyl transition) and which way does the ROOT run. The angle
clicks you made are drawn faintly for context. Frames come in TIME ORDER per
seedling, so you can see the first frame where a radicle shows up (that is the
germination time).

Run:
    python -m ui.root_annotator --folder cropped_training_set --angles angle_landmarks_train.csv
    python -m ui.root_annotator --folder cropped_training_set --angles angle_ground_truth.csv --out root_ground_truth.csv

Per frame:
    radicle visible -> click the COLLAR, then one point on the ROOT axis away from the
                       collar, then Enter
    no radicle yet  -> press N
    cannot tell     -> S (skip)
The CSV (default: <angles file stem>_root.csv next to the angles file) is saved after every
frame and the tool resumes where you stopped. See utils/root_annotation.py.
"""
import argparse
import os
import sys
import tkinter as tk
from tkinter import messagebox

import cv2

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ui.zoomable_canvas import ZoomableImageCanvas  # noqa: E402
from utils.root_annotation import (  # noqa: E402
    RootClickSession, RootStore, angle_points, measured_work_items,
)

COLLAR_COLOR = "#7b2cbf"
ROOT_COLOR = "#e63946"
CONTEXT_COLOR = "#9aa0a6"
HELP_TEXT = (
    "Mouse wheel: zoom at cursor\n"
    "Enter / Space: save radicle & next\n"
    "N: no radicle visible, next\n"
    "Backspace / Z: undo last point\n"
    "R: reset points      S: skip\n"
    "Left: previous\n"
    "Right: next -- a frame you did not\n"
    "annotate is recorded as skipped"
)


def default_out_path(angles_path):
    stem, _ = os.path.splitext(os.path.abspath(angles_path))
    return stem + "_root.csv"


class RootAnnotatorApp(tk.Tk):
    CANVAS_W, CANVAS_H = 560, 720

    def __init__(self, folder, angles_path, out_path):
        super().__init__()
        self.title("DLhook - collar and root direction")
        self.folder = folder
        self.queue = measured_work_items(angles_path)
        if not self.queue:
            messagebox.showerror("Nothing to annotate", f"No measured frames in:\n{angles_path}")
            self.destroy()
            raise SystemExit(1)
        self.context = angle_points(angles_path)
        self.store = RootStore(out_path)
        self.index = 0
        self._shown_seedling = None
        self.session = RootClickSession()
        self._build_widgets()
        self._bind_keys()
        self.index = next((i for i, item in enumerate(self.queue) if self.store.get(item) is None), 0)
        self._load_current()

    # --- widgets ----------------------------------------------------------

    def _build_widgets(self):
        self.canvas = ZoomableImageCanvas(self, width=self.CANVAS_W, height=self.CANVAS_H)
        self.canvas.grid(row=0, column=0, rowspan=2, padx=8, pady=8)
        self.canvas.bind("<Button-1>", self._on_click)

        panel = tk.Frame(self)
        panel.grid(row=0, column=1, sticky="n", padx=(0, 12), pady=8)
        self.info_label = tk.Label(panel, text="", justify=tk.LEFT, font=("Helvetica", 12, "bold"))
        self.info_label.pack(anchor="w")
        self.file_label = tk.Label(panel, text="", justify=tk.LEFT, fg="#666")
        self.file_label.pack(anchor="w", pady=(0, 6))
        self.context_label = tk.Label(panel, text="", justify=tk.LEFT, fg="#444", wraplength=260)
        self.context_label.pack(anchor="w", pady=(0, 8))
        self.progress_label = tk.Label(panel, text="", justify=tk.LEFT)
        self.progress_label.pack(anchor="w", pady=(0, 10))
        self.prompt_label = tk.Label(panel, text="", justify=tk.LEFT, wraplength=260,
                                     font=("Helvetica", 11), fg="#b4531f")
        self.prompt_label.pack(anchor="w", pady=(0, 10))

        legend = tk.Frame(panel)
        legend.pack(anchor="w", pady=(0, 10))
        for color, text in ((COLLAR_COLOR, "collar"), (ROOT_COLOR, "root axis"), (CONTEXT_COLOR, "your angle clicks")):
            tk.Label(legend, text="●", fg=color).pack(side=tk.LEFT)
            tk.Label(legend, text=text).pack(side=tk.LEFT, padx=(0, 10))

        buttons = tk.Frame(panel)
        buttons.pack(anchor="w", pady=4)
        tk.Button(buttons, text="Save radicle & next", width=18, command=self._confirm).grid(row=0, column=0, columnspan=2, pady=2)
        tk.Button(buttons, text="No radicle (N)", width=12, command=self._no_radicle).grid(row=1, column=0, padx=(0, 6), pady=2)
        tk.Button(buttons, text="Skip (S)", width=12, command=self._skip).grid(row=1, column=1, pady=2)
        tk.Button(buttons, text="Undo point", width=12, command=self._undo).grid(row=2, column=0, padx=(0, 6), pady=2)
        tk.Button(buttons, text="Reset", width=12, command=self._reset).grid(row=2, column=1, pady=2)
        tk.Button(buttons, text="◀ Previous", width=12, command=lambda: self._move(-1)).grid(row=3, column=0, padx=(0, 6), pady=2)
        tk.Button(buttons, text="Next ▶", width=12, command=lambda: self._move(1)).grid(row=3, column=1, pady=2)

        tk.Label(panel, text=HELP_TEXT, justify=tk.LEFT, fg="#666").pack(anchor="w", pady=(14, 0))

    def _bind_keys(self):
        for key in ("<Return>", "<space>"):
            self.bind(key, lambda e: self._confirm())
        for key in ("n", "N"):
            self.bind(key, lambda e: self._no_radicle())
        for key in ("<BackSpace>", "z", "Z"):
            self.bind(key, lambda e: self._undo())
        for key in ("r", "R"):
            self.bind(key, lambda e: self._reset())
        for key in ("s", "S"):
            self.bind(key, lambda e: self._skip())
        self.bind("<Left>", lambda e: self._move(-1))
        self.bind("<Right>", lambda e: self._move(1))

    # --- navigation -------------------------------------------------------

    @property
    def item(self):
        return self.queue[self.index]

    def _seedling_frames(self):
        key = self.item.seedling_key
        return [i for i in self.queue if i.seedling_key == key]

    def _load_current(self):
        item = self.item
        img = cv2.imread(os.path.join(self.folder, item.filename))
        if img is None:
            messagebox.showerror("Cannot open image", item.filename)
            return
        self.canvas.set_image(img, reset_view=item.seedling_key != self._shown_seedling)
        self._shown_seedling = item.seedling_key
        self.session = RootClickSession(self.store.points_for(item))
        self._refresh()

    def _move(self, step):
        if step > 0:
            if self.store.get(self.item) is None:
                if self.session.complete:
                    self.bell()
                    self.prompt_label.configure(text="Both points placed but not saved -- Enter to save, or R to reset")
                    return
                self.store.save_skipped(self.item)
            self._advance()
            return
        if self.index > 0:
            self.index -= 1
            self._load_current()

    def _advance(self):
        if self.index + 1 < len(self.queue):
            self.index += 1
            self._load_current()
        else:
            self._refresh()
            messagebox.showinfo("Done", f"All {len(self.queue)} frames reviewed.\n\nSaved to:\n{self.store.path}")

    # --- actions ----------------------------------------------------------

    def _on_click(self, event):
        h, w = self.canvas._image.shape[:2]
        x, y = self.canvas.canvas_to_image(event.x, event.y)
        if 0 <= x < w and 0 <= y < h:
            self.session.add(x, y)
            self._refresh()

    def _undo(self):
        self.session.undo()
        self._refresh()

    def _reset(self):
        self.session.reset()
        self._refresh()

    def _confirm(self):
        try:
            self.store.save_radicle(self.item, self.session)
        except ValueError as exc:
            self.bell()
            self.prompt_label.configure(text=str(exc))
            return
        self._advance()

    def _no_radicle(self):
        self.store.save_no_radicle(self.item)
        self._advance()

    def _skip(self):
        self.store.save_skipped(self.item)
        self._advance()

    # --- display ----------------------------------------------------------

    def _overlay_shapes(self):
        shapes = []
        ctx = self.context.get((self.item.series, self.item.crop_id, self.item.frame))
        if ctx:
            j = ctx["junction"]
            shapes.append({"kind": "point", "xy": j, "color": CONTEXT_COLOR, "label": "J"})
            for a, b in (("hypo1", "hypo2"), ("cotyl1", "cotyl2")):
                shapes.append({"kind": "line", "from": ctx[a], "to": ctx[b], "color": CONTEXT_COLOR, "dash": True})
        pts = self.session.named_points()
        if "collar" in pts:
            shapes.append({"kind": "point", "xy": pts["collar"], "color": COLLAR_COLOR, "label": "collar"})
        if "root" in pts:
            shapes.append({"kind": "point", "xy": pts["root"], "color": ROOT_COLOR, "label": "root"})
            shapes.append({"kind": "line", "from": pts["collar"], "to": pts["root"], "color": ROOT_COLOR, "arrow": True})
        return shapes

    def _context_text(self):
        frames = self._seedling_frames()
        pos = frames.index(self.item)
        earlier = [self.store.get(i) for i in frames[:pos]]
        seen = next((k for k, r in enumerate(earlier) if r and r["status"] == "annotated" and r["radicle_visible"] == "1"), None)
        text = f"frame {pos + 1} of {len(frames)} measured for this seedling"
        if seen is not None:
            text += f"\nradicle first marked at its frame {seen + 1}"
        elif pos and all(r and r["status"] == "annotated" for r in earlier):
            text += "\nno radicle marked in earlier frames"
        return text

    def _refresh(self):
        item = self.item
        row = self.store.get(item)
        status = ""
        if row:
            status = "  [skipped]" if row["status"] == "skipped" else (
                "  [radicle]" if row["radicle_visible"] == "1" else "  [no radicle]")
        with_r, without_r, skipped = self.store.counts(self.queue)
        self.info_label.configure(text=f"{item.seedling_label}  —  {self.index + 1} of {len(self.queue)}{status}")
        self.file_label.configure(text=item.filename)
        self.context_label.configure(text=self._context_text())
        remaining = len(self.queue) - with_r - without_r - skipped
        self.progress_label.configure(
            text=f"Radicle: {with_r}    No radicle: {without_r}    Skipped: {skipped}    Remaining: {remaining}")
        self.prompt_label.configure(text=self.session.prompt())
        self.canvas.set_overlay(self._overlay_shapes())


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", required=True, help="Folder of the crops (e.g. cropped_training_set)")
    parser.add_argument("--angles", required=True,
                        help="Angle CSV whose measured frames are re-opened (angle_landmarks_train.csv or "
                             "angle_ground_truth.csv)")
    parser.add_argument("--out", default=None, help="Collar/root CSV (default: <angles file stem>_root.csv)")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    app = RootAnnotatorApp(args.folder, args.angles, args.out or default_out_path(args.angles))
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
