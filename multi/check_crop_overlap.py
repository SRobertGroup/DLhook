#!/usr/bin/env python
"""Do the crop boxes of one boxes file intercept those of another?

Per series, every box of --new is tested against every box of --old with the
same geometry as the GUI (utils/crop_layout.py: half-open pixel rects, touching
is not overlapping). Boxes of two files overlapping means the same seedling is
(partly) in both crop sets -- a leak if one set is for training and the other
for testing, or a duplicate seedling.

    python -m multi.check_crop_overlap [--old crop_boxes.json] [--new crop_boxes_new.json]

Prints each overlapping pair with the shared area as a fraction of the smaller
box (IoU-free: 1.0 = one box lies fully inside the other). Exit status 1 when
anything overlaps, so it can gate a script. Boxes and image coordinates are the
{"cx", "cy", "half_w", "half_h"} dicts draw_crops.py writes; crop ids shown are
0-based, matching the crop file names.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.crop_layout import box_to_rect, rects_overlap  # noqa: E402


def overlap_fraction(a, b):
    """Shared area / area of the smaller rect (0 when disjoint)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return (w * h) / smaller if smaller > 0 else 0.0


def find_cross_overlaps(old, new):
    """[(series, old_id, new_id, fraction)] for boxes of `new` intercepting boxes of `old`
    in the same series. Both are {series: [box dict, ...]}; ids are list positions."""
    hits = []
    for series, new_boxes in new.items():
        old_boxes = old.get(series, [])
        for j, nb in enumerate(new_boxes):
            n_rect = box_to_rect(nb)
            for i, ob in enumerate(old_boxes):
                o_rect = box_to_rect(ob)
                if rects_overlap(n_rect, o_rect):
                    hits.append((series, i, j, overlap_fraction(n_rect, o_rect)))
    return hits


PARTIAL_BELOW = 0.5


def classify_new_boxes(old, new, partial_below=PARTIAL_BELOW):
    """{series: [entry, ...]}, one entry per new box in file order:
    {"index", "box", "status", "fraction", "partners"} where status is
      "free"       touches no old box,
      "partial"    its largest shared area with an old box is below `partial_below` of the smaller
                   box (neighbours that touch -- needs a human look),
      "duplicate"  shares at least that much (the same seedling drawn again);
    `partners` are the old box indices it overlaps, biggest overlap first."""
    out = {}
    for series, new_boxes in new.items():
        old_boxes = old.get(series, [])
        entries = []
        for j, nb in enumerate(new_boxes):
            n_rect = box_to_rect(nb)
            shared = sorted(((overlap_fraction(n_rect, box_to_rect(ob)), i) for i, ob in enumerate(old_boxes)
                             if rects_overlap(n_rect, box_to_rect(ob))), reverse=True)
            fraction = shared[0][0] if shared else 0.0
            status = "free" if not shared else ("duplicate" if fraction >= partial_below else "partial")
            entries.append({"index": j, "box": nb, "status": status, "fraction": fraction,
                            "partners": [i for _, i in shared]})
        out[series] = entries
    return out


def box_key(box):
    return f"{box['cx']},{box['cy']},{box['half_w']},{box['half_h']}"


def default_decision(status):
    """Free boxes are kept, duplicates dropped, partial ones wait for a human (None)."""
    return {"free": "keep", "duplicate": "drop"}.get(status)


def build_filtered(classified, decisions):
    """{series: [kept boxes]} from classify_new_boxes() output and the user's decisions
    ({series: {box_key: "keep" | "drop"}}). A box with no explicit decision follows
    default_decision; an undecided partial box is NOT kept (leakage is the safer error)."""
    kept = {}
    for series, entries in classified.items():
        boxes = []
        for e in entries:
            decision = decisions.get(series, {}).get(box_key(e["box"])) or default_decision(e["status"])
            if decision == "keep":
                boxes.append(dict(e["box"]))
        kept[series] = boxes
    return kept


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old", default=str(_REPO_ROOT / "crop_boxes.json"))
    p.add_argument("--new", default=str(_REPO_ROOT / "crop_boxes_new.json"))
    args = p.parse_args(argv)
    old = json.loads(Path(args.old).read_text(encoding="utf-8"))
    new = json.loads(Path(args.new).read_text(encoding="utf-8"))

    n_new = sum(len(v) for v in new.values())
    print(f"{args.new}: {n_new} boxes in {len(new)} series; {args.old}: {sum(len(v) for v in old.values())} boxes "
          f"in {len(old)} series")
    only_new = [s for s in new if s not in old]
    if only_new:
        print(f"series with no old boxes (cannot overlap): {', '.join(only_new)}")
    hits = find_cross_overlaps(old, new)
    if not hits:
        print("no new box intercepts an old one")
        return 0
    print(f"\n{len(hits)} overlapping pair(s):")
    for series, i, j, frac in sorted(hits, key=lambda h: (h[0], -h[3])):
        print(f"  {series}: new box {j} (seedling {j + 1}) x old box {i} (seedling {i + 1}) "
              f"- shared area {frac:.0%} of the smaller box")
    return 1


if __name__ == "__main__":
    sys.exit(main())
