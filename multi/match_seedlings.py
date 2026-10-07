"""Which seedlings of a new crop set are the SAME physical seedlings as ones in earlier crop sets?

Two crop folders cut from the same plates can box the same seedling (different frames, a taller
box, a different crop_id). For landmark training that matters twice: the held-out seedlings that
pick the checkpoint must not share a plant with the training ones, and a validation set must not
contain a plant that was trained on. This matches seedlings by their crop boxes, read from the
crop manifests (what was actually cut, not what a boxes file says), as the shared area over the
smaller box in the same series.

    python -m multi.match_seedlings --new cropped_open_set/manifest.csv \
        --ref cropped_training_set/manifest.csv --ref new=cropped_new_set/manifest.csv \
        --out open_seedling_groups.json

`--ref [TAG=]MANIFEST` is repeatable; TAG is the `tag:` the reference gets under
`landmarks.extra_sources` (empty for the main training set). The output maps each matched new
seedling "series:crop_id" to the reference seedling it duplicates ("TAG/series:crop_id"); give it
to the trainer as `seedling_groups:` on the new source so the held-out split treats them as one.
Matches need >= --min-overlap (default 0.5); the best reference wins.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi.check_crop_overlap import overlap_fraction  # noqa: E402


def manifest_boxes(path):
    """{(series, crop_id): (x1, y1, x2, y2)} -- one box per seedling (the geometry is the same in every frame)."""
    boxes = OrderedDict()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            boxes.setdefault((row["series"], int(row["crop_id"])),
                             tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2")))
    return boxes


def match_seedlings(new, refs, min_overlap=0.5):
    """{(series, crop_id): (tag, series, crop_id, overlap)} for the seedlings of `new` that match a
    reference. `new` is a manifest_boxes dict, `refs` a list of (tag, manifest_boxes dict)."""
    out = {}
    for (series, cid), rect in new.items():
        best = None
        for tag, ref in refs:
            for (r_series, r_cid), r_rect in ref.items():
                if r_series != series:
                    continue
                frac = overlap_fraction(rect, r_rect)
                if frac >= min_overlap and (best is None or frac > best[3]):
                    best = (tag, r_series, r_cid, frac)
        if best:
            out[(series, cid)] = best
    return out


def groups_json(matches):
    """The trainer's `seedling_groups` file: {"series:crop_id": "[tag/]series:crop_id"}."""
    return {f"{s}:{c}": (f"{tag}/" if tag else "") + f"{rs}:{rc}" for (s, c), (tag, rs, rc, _) in matches.items()}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new", required=True, help="manifest.csv of the new crop folder")
    p.add_argument("--ref", action="append", required=True, metavar="[TAG=]MANIFEST")
    p.add_argument("--min-overlap", type=float, default=0.5)
    p.add_argument("--out", default=None, help="write the seedling_groups JSON here")
    args = p.parse_args(argv)
    new = manifest_boxes(args.new)
    refs = []
    for item in args.ref:
        tag, sep, path = item.rpartition("=")
        refs.append((tag, manifest_boxes(path)))
    matches = match_seedlings(new, refs, args.min_overlap)
    by_ref = {}
    for tag, *_ in matches.values():
        by_ref[tag or "(training set)"] = by_ref.get(tag or "(training set)", 0) + 1
    print(f"{len(new)} seedlings in {args.new}: {len(matches)} are already in a reference set "
          f"({', '.join(f'{n} in {t}' for t, n in sorted(by_ref.items())) or 'none'}), "
          f"{len(new) - len(matches)} are new plants")
    if args.out:
        Path(args.out).write_text(json.dumps(groups_json(matches), indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
