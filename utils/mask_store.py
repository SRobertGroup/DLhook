"""
In-memory store for segmentation masks -- replaces the data/predict/ +
data/postprocess/ PNG round-trip.

WHY THIS EXISTS. UNetInference already holds each mask as a numpy array; the
old pipeline encoded it to PNG, wrote it to data/predict/, then immediately
read and decoded it back in process_single_frame. Those directories were pure
scratch -- RemoveData (utils/clean_on_exit.py) wipes them both at startup and
at exit, and nothing ever resumed from them -- so the whole round-trip bought
nothing. Note the PNG format itself was never the problem: a real 98x302 mask
is ~718 bytes on disk (deflate compresses these smooth blobs ~41x), which beats
any bit-packed representation. What cost time was the encode/write/read/decode
per mask, not the bytes.

PIXEL CONVENTION -- read this before touching anything.
This store holds **255 = foreground, 0 = background**, the same convention
process_single_frame's cotyl_mask/hypo_mask, MaskEditor and the analysis
window's overlay rendering all use.

The *on-disk* files written by the old pipeline used the OPPOSITE convention
(0 = foreground), which forced a cv2.bitwise_not in every single consumer and
was the direct cause of the Phase 7 "mask overlay looks inverted" bug. Holding
one consistent convention in memory removes all of those inversions. The only
place the legacy convention still appears is dump() below, which re-inverts on
the way out so dumped files stay byte-identical to what the pipeline used to
write (the ground-truth validation workflow depends on that).

EDIT PRECEDENCE. get() returns a brush-edited mask if one exists, else the raw
model output. This reproduces the old resolve_mask_path(), where the *existence
of a file* in data/postprocess/ was the only record that a mask had been
edited. has_edit() makes that predicate explicit instead of implicit.
"""

import os
import cv2


class MaskStore:
    """
    Session-scoped mask storage keyed by (file_name, label).

    `file_name` is the cropped frame's filename ("{crop_id}-crop-{name}.png",
    the same key the old on-disk paths were derived from) and `label` is the
    model label string ("1" = cotyledon, "2" = hypocotyl, "4" = germination).

    Must be held on Gui, not on a preview Toplevel: SeedlingAnalysisWindow
    rebuilds its MaskEditor from this store every time the frame or edit target
    changes, and again when a window is reopened for the same seedling, so the
    store has to outlive any individual window for brush edits to survive.
    """

    def __init__(self):
        self._raw = {}      # (file_name, label) -> uint8 mask, 255 = foreground
        self._edited = {}   # (file_name, label) -> uint8 mask, 255 = foreground

    # --- writes ---------------------------------------------------------

    def put_raw(self, file_name, label, mask):
        """Store a model-produced mask (255 = foreground)."""
        self._raw[(file_name, label)] = mask

    def put_raw_bulk(self, masks_by_filename, label):
        """Store a whole {file_name: mask} batch for one label."""
        for file_name, mask in masks_by_filename.items():
            self._raw[(file_name, label)] = mask

    def put_edited(self, file_name, label, mask):
        """Store a brush-edited mask (255 = foreground). Takes precedence over
        the raw prediction for the same key, permanently -- edited masks are
        never dropped by discard_raw()."""
        self._edited[(file_name, label)] = mask

    # --- reads ----------------------------------------------------------

    def get(self, file_name, label):
        """Edited mask if one exists, else the raw prediction, else None.
        Same precedence the old resolve_mask_path() implemented via
        os.path.exists on data/postprocess/."""
        key = (file_name, label)
        if key in self._edited:
            return self._edited[key]
        return self._raw.get(key)

    def has_edit(self, file_name, label):
        return (file_name, label) in self._edited

    def __contains__(self, key):
        return key in self._edited or key in self._raw

    # --- lifecycle ------------------------------------------------------

    def discard_raw(self, file_names, labels=("1", "2", "4")):
        """Drop raw masks for these frames once their contours have been
        extracted -- bounds peak memory to one processing chunk instead of the
        whole run. Edited masks are deliberately kept: the user's brush work
        must survive, and there are only ever a handful of them."""
        for file_name in file_names:
            for label in labels:
                self._raw.pop((file_name, label), None)

    def clear_crop(self, crop_id, labels=("1", "2", "4")):
        """Forget everything for one seedling (its crop_id prefix), so
        re-segmenting it doesn't leave stale masks behind."""
        prefix = f"{crop_id}-crop-"
        for store in (self._raw, self._edited):
            for key in [k for k in store if k[0].startswith(prefix) and k[1] in labels]:
                del store[key]

    def clear(self):
        self._raw.clear()
        self._edited.clear()

    # --- opt-in disk dump ------------------------------------------------

    def dump(self, out_dir, include_raw=True, include_edited=True):
        """
        Write masks out as PNGs in the LEGACY on-disk convention
        (0 = foreground), so dumped files are byte-identical to what the old
        pipeline wrote and remain directly comparable against previously
        exported ground-truth masks (e.g. example_data/*/segmented_s*/).

        Off by default in the app; enabled via DLHOOK_DUMP_MASKS. Edited masks
        are written last so they overwrite the raw file for the same key,
        matching the old data/postprocess/-wins-over-data/predict/ precedence.
        """
        os.makedirs(out_dir, exist_ok=True)
        count = 0
        sources = []
        if include_raw:
            sources.append(self._raw)
        if include_edited:
            sources.append(self._edited)

        for store in sources:
            for (file_name, label), mask in store.items():
                out_name = f"{os.path.splitext(file_name)[0]}-{label}.png"
                cv2.imwrite(os.path.join(out_dir, out_name), cv2.bitwise_not(mask))
                count += 1

        print(f"[INFO] Dumped {count} masks to {out_dir}")
        return count
