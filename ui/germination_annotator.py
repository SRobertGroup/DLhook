#!/usr/bin/env python
"""Germination-onset annotation tool: mark the first frame with a visible radicle.

The angle annotator only shows frames where the hook can be measured, which is after
germination. This tool shows each seedling's WHOLE timeline instead. Scrub through the
frames and mark the first one in which a radicle is visible; that single decision labels
every frame of the seedling (before = no radicle, from the onset on = radicle).

Run:
    python -m ui.germination_annotator --folder cropped_training_set --subset train --out germination_train.csv
    python -m ui.germination_annotator --folder cropped_training_set --subset test  --out germination_test.csv
    python -m ui.germination_annotator --folder cropped_training_set --subset train --out germination_train.csv --export-frame-labels frames.csv

--subset holds out whole SEEDLINGS (about 20%, deterministic) so the test set never shares a seedling
with training. Do not use the image-level multiclass split here: it leaves a seedling with a few
scattered frames, which cannot show when a radicle first appears.

Keys:
    Left / Right        previous / next frame        Home / End   first / last frame
    Up / Down           previous / next seedling     click the timeline strip to jump
    G                   the radicle is FIRST visible in THIS frame (saves)
    F                   already visible in the first frame (onset before the series)
    X                   no radicle in any frame
    S                   cannot tell, skip this seedling
Optional, on the onset frame: click the COLLAR then one point on the ROOT axis, then press G
again to save them with the onset. Mouse wheel zooms.

The CSV (default <repo>/germination_ground_truth.csv) is saved after every decision and the
tool resumes at the first seedling without one. See utils/germination_annotation.py.
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
from utils.angle_annotation import default_manifest  # noqa: E402
from utils.germination_annotation import GerminationStore, load_seedlings, write_frame_labels  # noqa: E402
from utils.root_annotation import RootClickSession  # noqa: E402

COLLAR_COLOR = "#7b2cbf"
ROOT_COLOR = "#e63946"
STRIP_W, STRIP_H = 300, 26
HELP_TEXT = (
    "Left/Right: frame     Home/End: first/last\n"
    "Up/Down: seedling     click strip: jump\n"
    "G: radicle first visible HERE\n"
    "F: visible already in frame 1\n"
    "X: no radicle in any frame\n"
    "S: skip seedling\n"
    "Optional on the onset frame: click collar,\n"
    "then a root point, then G again\n"
    "Backspace/Z: undo point   R: reset points"
)


class GerminationAnnotatorApp(tk.Tk):
    CANVAS_W, CANVAS_H = 560, 720

    def __init__(self, folder, out_path, manifest_path=None, subset=None, test_fraction=0.2):
        super().__init__()
        self.title("DLhook - germination onset")
        self.folder = folder
        self.seedlings = load_seedlings(folder, manifest_path, subset, test_fraction)
        if not self.seedlings:
            messagebox.showerror("No seedlings found", f"No seedling with at least 2 frames in:\n{folder}")
            self.destroy()
            raise SystemExit(1)
        self.store = GerminationStore(out_path)
        self.s_index = next((i for i, s in enumerate(self.seedlings) if self.store.get(s) is None), 0)
        self.frame_index = 0
        self.session = RootClickSession()
        self._shown_seedling = None
        self._build_widgets()
        self._bind_keys()
        self._load_seedling()

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
        self.file_label.pack(anchor="w")
        self.strip = tk.Canvas(panel, width=STRIP_W, height=STRIP_H, bg="#f1f3f5", highlightthickness=0)
        self.strip.pack(anchor="w", pady=8)
        self.strip.bind("<Button-1>", self._on_strip_click)
        self.status_label = tk.Label(panel, text="", justify=tk.LEFT, wraplength=300, font=("Helvetica", 11))
        self.status_label.pack(anchor="w", pady=(0, 8))
        self.progress_label = tk.Label(panel, text="", justify=tk.LEFT)
        self.progress_label.pack(anchor="w", pady=(0, 8))
        self.prompt_label = tk.Label(panel, text="", justify=tk.LEFT, wraplength=300, fg="#b4531f")
        self.prompt_label.pack(anchor="w", pady=(0, 8))

        buttons = tk.Frame(panel)
        buttons.pack(anchor="w", pady=4)
        tk.Button(buttons, text="Radicle first visible here (G)", width=28, command=self._mark_onset).grid(
            row=0, column=0, columnspan=2, pady=2)
        tk.Button(buttons, text="Already visible, frame 1 (F)", width=28, command=lambda: self._mark("before_start")).grid(
            row=1, column=0, columnspan=2, pady=2)
        tk.Button(buttons, text="No radicle (X)", width=13, command=lambda: self._mark("none")).grid(row=2, column=0, pady=2)
        tk.Button(buttons, text="Skip (S)", width=13, command=lambda: self._mark("skipped")).grid(row=2, column=1, pady=2)
        tk.Button(buttons, text="◀ Seedling", width=13, command=lambda: self._seedling_step(-1)).grid(row=3, column=0, pady=2)
        tk.Button(buttons, text="Seedling ▶", width=13, command=lambda: self._seedling_step(1)).grid(row=3, column=1, pady=2)

        tk.Label(panel, text=HELP_TEXT, justify=tk.LEFT, fg="#666").pack(anchor="w", pady=(12, 0))

    def _bind_keys(self):
        self.bind("<Left>", lambda e: self._frame_step(-1))
        self.bind("<Right>", lambda e: self._frame_step(1))
        self.bind("<Home>", lambda e: self._go_frame(0))
        self.bind("<End>", lambda e: self._go_frame(len(self.seedling) - 1))
        self.bind("<Up>", lambda e: self._seedling_step(-1))
        self.bind("<Down>", lambda e: self._seedling_step(1))
        for key in ("g", "G", "<Return>"):
            self.bind(key, lambda e: self._mark_onset())
        for key in ("f", "F"):
            self.bind(key, lambda e: self._mark("before_start"))
        for key in ("x", "X"):
            self.bind(key, lambda e: self._mark("none"))
        for key in ("s", "S"):
            self.bind(key, lambda e: self._mark("skipped"))
        for key in ("<BackSpace>", "z", "Z"):
            self.bind(key, lambda e: self._undo())
        for key in ("r", "R"):
            self.bind(key, lambda e: self._reset())

    # --- navigation -------------------------------------------------------

    @property
    def seedling(self):
        return self.seedlings[self.s_index]

    def _load_seedling(self):
        saved = self.store.onset_index(self.seedling)
        self.frame_index = saved if saved is not None else 0
        self.session = RootClickSession(self.store.points_for(self.seedling))
        self._show_frame()

    def _show_frame(self):
        s = self.seedling
        _, filename, _ = s.frames[self.frame_index]
        img = cv2.imread(os.path.join(self.folder, filename))
        if img is None:
            messagebox.showerror("Cannot open image", filename)
            return
        self.canvas.set_image(img, reset_view=s.key != self._shown_seedling)
        self._shown_seedling = s.key
        self._refresh()

    def _go_frame(self, index):
        index = max(0, min(len(self.seedling) - 1, index))
        if index != self.frame_index:
            self.frame_index = index
            self._show_frame()

    def _frame_step(self, step):
        self._go_frame(self.frame_index + step)

    def _seedling_step(self, step):
        new = self.s_index + step
        if 0 <= new < len(self.seedlings):
            self.s_index = new
            self._load_seedling()

    def _on_strip_click(self, event):
        n = len(self.seedling)
        self._go_frame(min(n - 1, int(event.x / STRIP_W * n)))

    # --- annotation -------------------------------------------------------

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

    def _mark_onset(self):
        try:
            self.store.save_found(self.seedling, self.frame_index, self.session)
        except ValueError as exc:
            self.bell()
            self.prompt_label.configure(text=str(exc))
            return
        self._next_seedling()

    def _mark(self, status):
        self.store.save_status(self.seedling, status)
        self._next_seedling()

    def _next_seedling(self):
        if self.s_index + 1 < len(self.seedlings):
            self.s_index += 1
            self._load_seedling()
        else:
            self._refresh()
            messagebox.showinfo("Done", f"All {len(self.seedlings)} seedlings reviewed.\n\nSaved to:\n{self.store.path}")

    # --- display ----------------------------------------------------------

    def _draw_strip(self):
        self.strip.delete("all")
        n = len(self.seedling)
        onset = self.store.onset_index(self.seedling)
        step = STRIP_W / n
        for i in range(n):
            color = "#e9ecef"
            if onset is not None:
                color = "#74c69d" if i >= onset else "#ced4da"
            self.strip.create_rectangle(i * step + 1, 4, (i + 1) * step - 1, STRIP_H - 4, fill=color, outline="")
        cx = (self.frame_index + 0.5) * step
        self.strip.create_line(cx, 0, cx, STRIP_H, fill="#1864ab", width=3)

    def _status_text(self):
        row = self.store.get(self.seedling)
        if not row:
            return "not annotated yet"
        status = row["status"]
        if status == "found":
            return f"onset: frame {int(row['onset_index']) + 1} ({row['onset_frame']})" + (
                "  + collar/root saved" if row["collar_x"] != "" else "")
        return {"before_start": "radicle already visible in frame 1", "none": "no radicle in any frame",
                "skipped": "skipped"}[status]

    def _refresh(self):
        s = self.seedling
        _, filename, _ = s.frames[self.frame_index]
        counts = self.store.counts(self.seedlings)
        done = sum(counts.values())
        self.info_label.configure(text=f"{s.label}  —  seedling {self.s_index + 1} of {len(self.seedlings)}")
        self.file_label.configure(text=f"frame {self.frame_index + 1} of {len(s)}   {filename}")
        self.status_label.configure(text=self._status_text())
        self.progress_label.configure(
            text=f"Found: {counts['found']}   Before start: {counts['before_start']}   None: {counts['none']}   "
                 f"Skipped: {counts['skipped']}   Remaining: {len(self.seedlings) - done}")
        self.prompt_label.configure(text=self.session.prompt() if self.session.points or self.frame_index
                                    else "Scrub to the first frame with a visible radicle, then press G")
        self._draw_strip()
        shapes = []
        pts = self.session.named_points()
        if "collar" in pts:
            shapes.append({"kind": "point", "xy": pts["collar"], "color": COLLAR_COLOR, "label": "collar"})
        if "root" in pts:
            shapes.append({"kind": "point", "xy": pts["root"], "color": ROOT_COLOR, "label": "root"})
            shapes.append({"kind": "line", "from": pts["collar"], "to": pts["root"], "color": ROOT_COLOR, "arrow": True})
        self.canvas.set_overlay(shapes)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", required=True, help="Folder of the crops (e.g. cropped_training_set)")
    parser.add_argument("--manifest", default=None, help="recrop_plates manifest.csv (default: <folder>/manifest.csv)")
    parser.add_argument("--subset", choices=("train", "test"), default=None,
                        help="Seedling-level hold-out: 'test' is a fixed ~20%% of the seedlings, 'train' the rest")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Share of seedlings in 'test' (default 0.2)")
    parser.add_argument("--out", default=None, help="CSV (default: <repo>/germination_ground_truth.csv)")
    parser.add_argument("--export-frame-labels", default=None, metavar="CSV",
                        help="Do not open the window: write one row per labelled frame (radicle visible 0/1 and the "
                             "offset from the onset) from the annotations in --out, then exit")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_path = args.out or os.path.join(_REPO_ROOT, "germination_ground_truth.csv")
    manifest = args.manifest or default_manifest(args.folder)
    if args.export_frame_labels:
        seedlings = load_seedlings(args.folder, manifest, args.subset, args.test_fraction)
        n = write_frame_labels(args.export_frame_labels, seedlings, GerminationStore(out_path))
        print(f"wrote {n} frame label(s) to {args.export_frame_labels}")
        return 0
    app = GerminationAnnotatorApp(args.folder, out_path, manifest, args.subset, args.test_fraction)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
