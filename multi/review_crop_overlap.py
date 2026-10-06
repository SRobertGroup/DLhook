#!/usr/bin/env python
"""Review crop boxes of a new boxes file against an old one, and keep the ones that are new.

Opens the last frame of each series (the frame multi.draw_crops draws on) with the OLD boxes
in blue and the NEW boxes in orange. A new box is classified (multi/check_crop_overlap.py):

    free        touches no old box                              -> kept by default
    partial     overlaps an old box by less than half of the     -> YOU decide (red outline)
                smaller box (touching neighbours)
    duplicate   overlaps by half or more: the same seedling      -> dropped by default
                drawn again

The overlapping area is shaded. The list on the right shows the new boxes of the series
(partial first); pick one, look at the zoomed pair, and press K to keep or D to drop it. The
filtered boxes file (kept boxes only) is rewritten after every decision, and an undecided
partial box is NOT kept.

You can also select NEW seedlings here: drag on the plate to outline a seedling (its extent, as in
multi.draw_crops -- the padded box is shown while you drag). The drawn box is checked against the
old boxes like any other (free / partial / duplicate) and against the other new boxes, joins the
list as "drawn", and is saved in <out>.added.json. Delete removes the selected drawn box. A plain
click (no drag) selects the box under the cursor.

Run:
    python -m multi.review_crop_overlap --old crop_boxes.json --new crop_boxes_new.json
    python -m multi.review_crop_overlap --out crop_boxes_new_filtered.json --example-data example_data

Keys:  K keep   D drop   U undecided (partial boxes)   Delete remove the selected drawn box
Up/Down or N/P next/previous box in the list   Left/Right previous/next series
A show all boxes / only those needing a decision   Z zoom to the selected box and its partners
R reset zoom   mouse wheel zoom.
Decisions and drawn boxes are saved next to the output (<out>.decisions.json, <out>.added.json), so
the review resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import tkinter as tk
from pathlib import Path

import cv2

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from multi.check_crop_overlap import box_key, build_filtered, classify_new_boxes, default_decision  # noqa: E402
from multi.draw_crops import discover_series_last_frames  # noqa: E402
from multi.check_crop_overlap import overlap_fraction  # noqa: E402
from multi.src.recrop_geometry import compute_crop_box  # noqa: E402
from ui.zoomable_canvas import MAX_ZOOM_MULTIPLIER, ZoomableImageCanvas  # noqa: E402
from utils.crop_layout import box_to_rect  # noqa: E402

OLD_COLOR = "#2f80ed"
NEW_COLOR = "#f2994a"
PARTIAL_COLOR = "#eb2f2f"
DROP_COLOR = "#9aa0a6"
SELECT_COLOR = "#00c853"
STATUS_TEXT = {"free": "free", "partial": "PARTIAL", "duplicate": "duplicate"}
PREVIEW_COLOR = "#00c8ff"
MIN_DRAG_PX = 4


def round_box(box):
    return {k: int(round(v)) for k, v in box.items()}


def new_box_clashes(entries, decisions_of):
    """{entry index: [(other index, shared fraction of the smaller box), ...]} for pairs of NEW
    boxes of one series that overlap, ignoring dropped ones -- two new seedlings must not share
    pixels any more than a new one may share them with an old one."""
    live = [e for e in entries if decisions_of(e) != "drop"]
    out = {}
    for i, a in enumerate(live):
        for b in live[i + 1:]:
            frac = overlap_fraction(box_to_rect(a["box"]), box_to_rect(b["box"]))
            if frac > 0:
                out.setdefault(a["index"], []).append((b["index"], frac))
                out.setdefault(b["index"], []).append((a["index"], frac))
    return out


def write_json_atomic(path, data):
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".crop_review_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


class OverlapCanvas(ZoomableImageCanvas):
    """The plate with old and new boxes drawn over it (image space -> canvas on every redraw)."""

    def __init__(self, parent, on_pick=None, on_draw=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.old_boxes, self.entries = [], []
        self.decisions = {}                 # box_key -> "keep" | "drop"
        self.selected = None                # index into self.entries
        self.on_pick = on_pick
        self.on_draw = on_draw              # called with the padded box of a finished drag
        self._press = None                  # (canvas x, canvas y, image x, image y)
        self._preview = None
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    def set_data(self, old_boxes, entries, decisions, selected=None):
        self.old_boxes, self.entries, self.decisions, self.selected = old_boxes, entries, decisions, selected
        if self._image is not None:
            self._redraw()

    def _to_canvas(self, x, y):
        x1, y1, _, _ = self._visible_bounds()
        return (x - x1) * self.zoom, (y - y1) * self.zoom

    def _rect(self, box):
        x1, y1, x2, y2 = box_to_rect(box)
        (cx1, cy1), (cx2, cy2) = self._to_canvas(x1, y1), self._to_canvas(x2, y2)
        return cx1, cy1, cx2, cy2

    def decision_of(self, entry):
        return self.decisions.get(box_key(entry["box"])) or default_decision(entry["status"])

    def _redraw(self):
        super()._redraw()
        if self._image is None:
            return
        for i, box in enumerate(self.old_boxes):
            x1, y1, x2, y2 = self._rect(box)
            self.create_rectangle(x1, y1, x2, y2, outline=OLD_COLOR, width=2, tags="review")
            self.create_text(x1 + 3, y1 + 2, text=f"old {i + 1}", anchor=tk.NW, fill=OLD_COLOR,
                             font=("Helvetica", 9, "bold"), tags="review")
        for k, e in enumerate(self.entries):
            decision = self.decision_of(e)
            selected = k == self.selected
            if selected:
                color, width = SELECT_COLOR, 4
            elif decision == "drop":
                color, width = DROP_COLOR, 2
            elif e["status"] == "partial" and decision is None:
                color, width = PARTIAL_COLOR, 3
            else:
                color, width = NEW_COLOR, 2
            x1, y1, x2, y2 = self._rect(e["box"])
            self.create_rectangle(x1, y1, x2, y2, outline=color, width=width, tags="review",
                                  dash=(5, 3) if decision == "drop" else None)
            self.create_text(x1 + 3, y2 - 2, text=f"new {e['index'] + 1}" + (" (drawn)" if e.get("drawn") else ""),
                             anchor=tk.SW, fill=color,
                             font=("Helvetica", 9, "bold"), tags="review")
            for partner in e["partners"]:
                a, b = box_to_rect(e["box"]), box_to_rect(self.old_boxes[partner])
                ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
                if ix2 > ix1 and iy2 > iy1:
                    (cx1, cy1), (cx2, cy2) = self._to_canvas(ix1, iy1), self._to_canvas(ix2, iy2)
                    self.create_rectangle(cx1, cy1, cx2, cy2, fill=PARTIAL_COLOR, stipple="gray50",
                                          outline="", tags="review")

    def _draw_preview(self):
        self.delete("preview")
        if self._preview is not None:
            x1, y1, x2, y2 = self._rect(self._preview)
            self.create_rectangle(x1, y1, x2, y2, outline=PREVIEW_COLOR, width=2, dash=(4, 2), tags="preview")

    def _on_press(self, event):
        self.focus_set()
        if self._image is not None:
            ix, iy = self.canvas_to_image(event.x, event.y)
            self._press = (event.x, event.y, ix, iy)

    def _on_drag(self, event):
        if self._press is None or self._image is None:
            return
        ix, iy = self.canvas_to_image(event.x, event.y)
        ax, ay = self._press[2], self._press[3]
        # the padded box is what gets cut, so it is what the preview shows (same as multi.draw_crops)
        self._preview = round_box(compute_crop_box(min(ax, ix), max(ax, ix), min(ay, iy), max(ay, iy)))
        self._draw_preview()

    def _on_release(self, event):
        press, preview = self._press, self._preview
        self._press, self._preview = None, None
        self.delete("preview")
        if press is None:
            return
        if abs(event.x - press[0]) >= MIN_DRAG_PX or abs(event.y - press[1]) >= MIN_DRAG_PX:
            if preview is not None and self.on_draw:
                self.on_draw(preview)
            return
        x, y = press[2], press[3]                       # a click: select the topmost new box under it
        for k in range(len(self.entries) - 1, -1, -1):
            x1, y1, x2, y2 = box_to_rect(self.entries[k]["box"])
            if x1 <= x < x2 and y1 <= y < y2:
                if self.on_pick:
                    self.on_pick(k)
                return

    def focus_on(self, rects, margin=0.6):
        """Zoom so the union of image-space `rects` fills the view (clamped to the zoom limits)."""
        if self._image is None or not rects:
            return
        x1, y1 = min(r[0] for r in rects), min(r[1] for r in rects)
        x2, y2 = max(r[2] for r in rects), max(r[3] for r in rects)
        w, h = (x2 - x1) * (1 + margin), (y2 - y1) * (1 + margin)
        zoom = min(self.display_width / max(w, 1), self.display_height / max(h, 1))
        self.zoom = min(max(zoom, self._base_zoom), self._base_zoom * MAX_ZOOM_MULTIPLIER)
        self._center_x, self._center_y = (x1 + x2) / 2, (y1 + y2) / 2
        self._clamp_center()
        self._redraw()


class ReviewApp(tk.Tk):
    def __init__(self, old, new, series_list, out_path, partial_below):
        super().__init__()
        self.title("DLhook - crop box overlap review")
        self.old, self.new = old, new
        self.out_path = out_path
        self.partial_below = partial_below
        self.decisions_path = out_path + ".decisions.json"
        self.added_path = out_path + ".added.json"
        self.series_list = list(series_list)
        self.decisions, self.added = {}, {}
        if os.path.exists(self.decisions_path):
            with open(self.decisions_path, encoding="utf-8") as fh:
                self.decisions = json.load(fh)
        if os.path.exists(self.added_path):
            with open(self.added_path, encoding="utf-8") as fh:
                self.added = json.load(fh)
        self.classified = {}
        self._reclassify()
        self.index = 0
        self.only_open = tk.BooleanVar(value=False)
        self.visible = []                    # entry positions shown in the listbox
        self._build()
        self._bind_keys()
        self._save()
        self._load_series(0)

    # --- layout ------------------------------------------------------------

    def _build(self):
        self.status = tk.StringVar()
        tk.Label(self, textvariable=self.status, anchor="w", font=("Segoe UI", 11)).grid(
            row=0, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        self.canvas = OverlapCanvas(self, on_pick=self._pick, on_draw=self._on_draw, width=980, height=720)
        self.canvas.grid(row=1, column=0, padx=8, pady=4)
        side = tk.Frame(self)
        side.grid(row=1, column=1, sticky="ns", padx=(0, 8))
        tk.Checkbutton(side, text="only boxes needing a decision (A)", variable=self.only_open,
                       command=self._fill_list).pack(anchor="w")
        self.listbox = tk.Listbox(side, width=44, height=26, exportselection=False, font=("Consolas", 10))
        self.listbox.pack(fill=tk.Y, expand=True, pady=4)
        self.listbox.bind("<<ListboxSelect>>", lambda e: self._on_list_select())
        row = tk.Frame(side)
        row.pack(anchor="w")
        tk.Button(row, text="Keep (K)", width=9, command=lambda: self._decide("keep")).pack(side=tk.LEFT)
        tk.Button(row, text="Drop (D)", width=9, command=lambda: self._decide("drop")).pack(side=tk.LEFT, padx=4)
        tk.Button(row, text="Undo (U)", width=9, command=lambda: self._decide(None)).pack(side=tk.LEFT)
        row2 = tk.Frame(side)
        row2.pack(anchor="w", pady=4)
        tk.Button(row2, text="<< Series", width=9, command=lambda: self._step_series(-1)).pack(side=tk.LEFT)
        tk.Button(row2, text="Series >>", width=9, command=lambda: self._step_series(1)).pack(side=tk.LEFT, padx=4)
        tk.Button(row2, text="Zoom (Z)", width=9, command=self._zoom_selected).pack(side=tk.LEFT)
        self.summary = tk.StringVar()
        tk.Label(side, textvariable=self.summary, justify=tk.LEFT, fg="#444").pack(anchor="w")
        tk.Label(side, text="Drag on the plate to draw a NEW seedling (* = drawn).\nDelete removes a drawn box.",
                 justify=tk.LEFT, fg="#555").pack(anchor="w", pady=(6, 0))
        legend = ("blue = old box   orange = new box   red = partial overlap (needs a decision)\n"
                  "grey dashed = dropped   green = selected   shaded = overlapping area")
        tk.Label(self, text=legend, anchor="w", fg="#555").grid(row=2, column=0, columnspan=2, sticky="w", padx=8)

    def _bind_keys(self):
        self.bind("<Left>", lambda e: self._step_series(-1))
        self.bind("<Right>", lambda e: self._step_series(1))
        for key in ("k", "K"):
            self.bind(key, lambda e: self._decide("keep"))
        for key in ("d", "D"):
            self.bind(key, lambda e: self._decide("drop"))
        for key in ("u", "U"):
            self.bind(key, lambda e: self._decide(None))
        for key in ("a", "A"):
            self.bind(key, lambda e: (self.only_open.set(not self.only_open.get()), self._fill_list()))
        for key in ("z", "Z"):
            self.bind(key, lambda e: self._zoom_selected())
        self.bind("<Delete>", lambda e: self._delete_drawn())
        for key in ("r", "R"):
            self.bind(key, lambda e: self.canvas.reset_zoom())
        for key in ("<Down>", "n", "N"):
            self.bind(key, lambda e: self._step_list(1))
        for key in ("<Up>", "p", "P"):
            self.bind(key, lambda e: self._step_list(-1))

    # --- data --------------------------------------------------------------

    def _reclassify(self):
        """Classify the file's new boxes followed by the drawn ones, for every series."""
        names = [s[0] for s in self.series_list]
        combined = {n: list(self.new.get(n, [])) + list(self.added.get(n, [])) for n in names}
        self.classified = classify_new_boxes(self.old, combined, self.partial_below)
        for name, entries in self.classified.items():
            n_file = len(self.new.get(name, []))
            for e in entries:
                e["drawn"] = e["index"] >= n_file

    @property
    def series_name(self):
        return self.series_list[self.index][0]

    @property
    def entries(self):
        return self.classified[self.series_name]

    def _series_decisions(self):
        return self.decisions.setdefault(self.series_name, {})

    def _decision(self, entry):
        return self._series_decisions().get(box_key(entry["box"])) or default_decision(entry["status"])

    def _load_series(self, index):
        self.index = index
        name, dir_path, frame = self.series_list[index]
        self.status.set(f"series {index + 1}/{len(self.series_list)}  ({name})  loading {frame} ...")
        self.update_idletasks()
        image = cv2.imread(os.path.join(dir_path, frame))
        if image is None:
            self.status.set(f"series {name}: cannot read {frame}")
            return
        self.canvas.set_image(image)
        self.canvas.reset_zoom()
        self._fill_list()
        self._pick_first_open()

    def _fill_list(self, keep_selection=None):
        entries = self.entries
        self.clashes = new_box_clashes(entries, self._decision)
        self.visible = [k for k, e in enumerate(entries)
                        if not self.only_open.get() or (e["status"] == "partial" and self._decision(e) is None)]
        self.visible.sort(key=lambda k: (entries[k]["status"] != "partial", k))
        self.listbox.delete(0, tk.END)
        for k in self.visible:
            e = entries[k]
            decision = self._decision(e)
            mark = {"keep": "KEEP", "drop": "drop", None: "  ? "}[decision]
            share = f"{e['fraction']:4.0%}" if e["partners"] else "    "
            partners = ",".join(str(i + 1) for i in e["partners"])
            clash = self.clashes.get(e["index"])
            note = f"  x new {','.join(str(i + 1) for i, _ in clash)}" if clash else ""
            tag = "*" if e.get("drawn") else " "
            self.listbox.insert(tk.END, f"new {e['index'] + 1:3d}{tag} {STATUS_TEXT[e['status']]:9s} {share} {mark}  "
                                        f"old:{partners}{note}")
        self._refresh(keep_selection)

    def _refresh(self, selected=None):
        sel = selected if selected is not None else self.canvas.selected
        self.canvas.set_data(self.old.get(self.series_name, []), self.entries, self._series_decisions(), sel)
        counts = {"free": 0, "partial": 0, "duplicate": 0}
        undecided = 0
        for e in self.entries:
            counts[e["status"]] += 1
            undecided += e["status"] == "partial" and self._decision(e) is None
        kept = sum(1 for e in self.entries if self._decision(e) == "keep")
        self.status.set(f"series {self.index + 1}/{len(self.series_list)}  ({self.series_name})   new boxes "
                        f"{len(self.entries)}: {counts['free']} free, {counts['partial']} partial, "
                        f"{counts['duplicate']} duplicate   |   kept {kept}, undecided partial {undecided}")
        total_open = sum(1 for name, es in self.classified.items() for e in es
                         if e["status"] == "partial" and self.decisions.get(name, {}).get(box_key(e["box"])) is None)
        total_kept = sum(len(v) for v in build_filtered(self.classified, self.decisions).values())
        self.summary.set(f"all series: {total_kept} boxes kept,\n{total_open} partial box(es) still undecided\n"
                         f"-> {self.out_path}")

    # --- interaction -------------------------------------------------------

    def _pick_first_open(self):
        for pos, k in enumerate(self.visible):
            e = self.entries[k]
            if e["status"] == "partial" and self._decision(e) is None:
                self._select_position(pos)
                return
        self.canvas.selected = None
        self._refresh(None)

    def _select_position(self, pos):
        if not 0 <= pos < len(self.visible):
            return
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(pos)
        self.listbox.see(pos)
        k = self.visible[pos]
        self._refresh(k)
        self._zoom_selected()

    def _on_list_select(self):
        sel = self.listbox.curselection()
        if sel:
            self._select_position(sel[0])

    def _step_list(self, step):
        sel = self.listbox.curselection()
        self._select_position((sel[0] if sel else -1) + step)

    def _pick(self, k):
        if k in self.visible:
            self._select_position(self.visible.index(k))

    def _zoom_selected(self):
        k = self.canvas.selected
        if k is None:
            return
        e = self.entries[k]
        rects = [box_to_rect(e["box"])] + [box_to_rect(self.old[self.series_name][i]) for i in e["partners"]]
        self.canvas.focus_on(rects)

    def _step_series(self, step):
        if 0 <= self.index + step < len(self.series_list):
            self._load_series(self.index + step)

    def _decide(self, decision):
        k = self.canvas.selected
        if k is None:
            return
        e = self.entries[k]
        decisions = self._series_decisions()
        if decision is None:
            decisions.pop(box_key(e["box"]), None)
        else:
            decisions[box_key(e["box"])] = decision
        self._save()
        pos = self.visible.index(k) if k in self.visible else 0
        self._fill_list(keep_selection=k)
        if self.only_open.get():
            self._pick_first_open()
        else:
            self._select_position(min(pos + 1, len(self.visible) - 1) if decision else pos)

    def _on_draw(self, box):
        """A finished drag: add the padded box as a new seedling of this series and select it."""
        name = self.series_name
        self.added.setdefault(name, []).append(box)
        self._reclassify()
        self._save()
        self._fill_list()
        k = len(self.entries) - 1
        e = self.entries[k]
        self.only_open.set(False)
        self._fill_list()
        self._select_position(self.visible.index(k))
        clash = self.clashes.get(e["index"])
        msg = f"drawn new {e['index'] + 1}: {STATUS_TEXT[e['status']]}"
        if e["partners"]:
            msg += f" ({e['fraction']:.0%} of old {e['partners'][0] + 1})"
        if clash:
            msg += f"; overlaps new {', '.join(str(i + 1) for i, _ in clash)}"
        self.status.set(self.status.get() + "   |   " + msg)

    def _delete_drawn(self):
        k = self.canvas.selected
        if k is None or not self.entries[k].get("drawn"):
            return
        name = self.series_name
        box = self.entries[k]["box"]
        self._series_decisions().pop(box_key(box), None)
        self.added[name].remove(box)
        self.canvas.selected = None
        self._reclassify()
        self._save()
        self._fill_list()

    def _save(self):
        write_json_atomic(self.decisions_path, self.decisions)
        write_json_atomic(self.added_path, self.added)
        kept = build_filtered(self.classified, self.decisions)
        write_json_atomic(self.out_path, {name: boxes for name, boxes in kept.items() if boxes})


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old", default=str(_REPO_ROOT / "crop_boxes.json"))
    p.add_argument("--new", default=str(_REPO_ROOT / "crop_boxes_new.json"))
    p.add_argument("--out", default=str(_REPO_ROOT / "crop_boxes_new_filtered.json"),
                   help="filtered boxes file, rewritten after every decision")
    p.add_argument("--example-data", default=str(_REPO_ROOT / "example_data"))
    p.add_argument("--partial-below", type=float, default=0.5,
                   help="a shared area below this fraction of the smaller box is 'partial' (default 0.5)")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    old = json.loads(Path(args.old).read_text(encoding="utf-8"))
    new = json.loads(Path(args.new).read_text(encoding="utf-8"))
    series_list = discover_series_last_frames(args.example_data)
    app = ReviewApp(old, new, series_list, args.out, args.partial_below)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
