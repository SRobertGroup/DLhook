#!/usr/bin/env python
"""Review earlier angle annotations: fix mistakes, mark frames where no angle can be measured.

    # the measured frames the model disagrees with most come first (errors from validate_landmarks)
    python -m ui.angle_reviewer --folder data/crops/cropped_training_set --csv data/annotations/angles/angle_landmarks_train.csv --errors errors.csv

    # frames you skipped after the seedling germinated: mark the ones where the cotyledons rotate
    python -m ui.angle_reviewer --folder data/crops/cropped_training_set --csv data/annotations/angles/angle_landmarks_train.csv \
        --skipped --germination data/annotations/germination/germination_train.csv

Every frame opens with the five clicks you made (a skipped frame opens empty). The cotyledon axis is
read as base -> tip: its first click is the BASE (at the junction) and its last click the TIP (the
midpoint between the two tips once the cotyledons have separated).

    Enter / Space   keep the clicks as they are (or save the five you re-clicked)
    click           after R/X resets them, place five new points (junction, hypocotyl x2, cotyledon x2)
    R               rotation: the cotyledons pass in front of or behind the hypocotyl -- no angle
    N               no angle for another reason (out of the crop, hidden, ...)
    X               reset the clicks      Backspace/Z  undo the last click
    S / Right       next without recording anything        Left   previous

Nothing in the original CSV changes: decisions go to `<csv>.review.csv`, which the trainers and the
scorers apply on top of it (utils/angle_review.py). The tool resumes where you stopped.
"""
import argparse
import os
import sys
import tkinter as tk
from tkinter import messagebox

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ui.angle_annotator import AngleAnnotatorApp  # noqa: E402
from utils.angle_annotation import default_manifest  # noqa: E402
from utils.angle_review import (  # noqa: E402
    ReviewStore, disagreement_queue, load_errors, skipped_after_onset_queue,
)

HELP_TEXT = (
    "Enter: keep (or save re-clicked points)\n"
    "R: rotation -- no angle\n"
    "N: no angle, other reason\n"
    "X: reset clicks    Z: undo a click\n"
    "S / Right: next, nothing recorded\n"
    "Left: previous\n"
    "Cotyledon: first click = base,\n"
    "last click = tip (or between tips)"
)


def same_points(a, b, tol=0.01):
    return a is not None and b is not None and len(a) == len(b) and all(
        abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol for p, q in zip(a, b))


class AngleReviewerApp(AngleAnnotatorApp):
    def __init__(self, folder, csv_path, queue, errors=None):
        self.errors = errors or {}
        super().__init__(folder, csv_path, queue=queue, store=ReviewStore(csv_path))
        self.title("DLhook - angle review")

    def _build_widgets(self):
        super()._build_widgets()
        row = tk.Frame(self.panel)
        row.pack(anchor="w", pady=4)
        tk.Button(row, text="Rotation (R)", width=12, command=self._rotation).grid(row=0, column=0, padx=(0, 6))
        tk.Button(row, text="No angle (N)", width=12, command=self._unmeasurable).grid(row=0, column=1)

    def _bind_keys(self):
        super()._bind_keys()
        for key in ("r", "R"):
            self.bind(key, lambda e: self._rotation())
        for key in ("n", "N"):
            self.bind(key, lambda e: self._unmeasurable())
        for key in ("x", "X"):
            self.bind(key, lambda e: self._reset())

    def _load_current(self):
        super()._load_current()
        original = self.store.original_row(self.item)
        self.overhook.set(False)
        self._original_points = self.store.original.points_for(self.item) if original else None

    def _move(self, step):
        new_index = self.index + step
        if 0 <= new_index < len(self.queue):
            self.index = new_index
            self._load_current()
        elif step > 0:
            self._advance()

    def _confirm(self):
        if not self.session.complete:
            self.bell()
            return
        if same_points(self.session.points, self._original_points):
            self.store.save_confirmed(self.item)
        else:
            try:
                self.store.save_measured(self.item, self.session, False)
            except ValueError as exc:
                messagebox.showwarning("Cannot compute angle", str(exc))
                return
        self._advance()

    def _skip(self):
        self._move(1)

    def _rotation(self):
        self.store.save_rotation(self.item)
        self._advance()

    def _unmeasurable(self):
        self.store.save_unmeasurable(self.item)
        self._advance()

    def _refresh(self):
        super()._refresh()
        item = self.item
        row = self.store.get(item)
        status = {"measured": "  [corrected]", "confirmed": "  [kept]", "rotation": "  [rotation]",
                  "unmeasurable": "  [no angle]"}.get(row["status"], "") if row else ""
        err = self.errors.get(item.filename)
        extra = f"   model vs you: {err:.0f}°" if err is not None else ""
        self.info_label.configure(
            text=f"{item.seedling_label}  —  frame {self.index + 1} of {len(self.queue)}{status}{extra}")
        counts = self.store.counts_by_status(self.queue)
        reviewed = sum(counts.values())
        self.progress_label.configure(
            text=f"Reviewed: {reviewed} of {len(self.queue)}    kept {counts['confirmed']}    corrected "
                 f"{counts['measured']}    rotation {counts['rotation']}    no angle {counts['unmeasurable']}")


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", required=True, help="the crop folder the CSV was annotated on")
    p.add_argument("--csv", required=True, help="the annotation CSV to review (never modified)")
    p.add_argument("--manifest", default=None, help="manifest.csv (default: <folder>/manifest.csv)")
    p.add_argument("--errors", default=None,
                   help="per-crop errors CSV (crop_file, err) from multi.validate_landmarks --per-crop-out: "
                        "biggest model-vs-you disagreement first")
    p.add_argument("--only-errors", action="store_true",
                   help="review only the frames listed in --errors (e.g. a list of suspect labels)")
    p.add_argument("--skipped", action="store_true",
                   help="review the frames skipped AFTER germination instead of the measured ones")
    p.add_argument("--germination", default=None, help="germination CSV, required with --skipped")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = args.manifest or default_manifest(args.folder)
    errors = load_errors(args.errors) if args.errors else {}
    if args.skipped:
        if not args.germination:
            print("--skipped needs --germination (the seedling onsets), to leave out the frames before germination")
            return 2
        queue = skipped_after_onset_queue(args.csv, args.germination, args.folder, manifest)
        print(f"{len(queue)} frames skipped after germination, of {len({i.seedling_key for i in queue})} seedlings")
    else:
        if args.only_errors and not errors:
            print("--only-errors needs --errors")
            return 2
        queue = disagreement_queue(args.csv, args.folder, errors, only_errors=args.only_errors)
        print(f"{len(queue)} measured frames" + (", biggest model disagreement first" if errors
                                                 else ", open hooks first (no --errors given)"))
    if not queue:
        print("nothing to review")
        return 1
    app = AngleReviewerApp(args.folder, args.csv, queue, errors)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
