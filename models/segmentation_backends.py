"""Which segmentation models actually run underneath the GUI pipeline.

Today (and until a later, separately-gated step flips it) that is always
three independent binary RootPainter checkpoints -- cotyledon_v5.pkl,
hypocot_v5.pkl, germ_v1.pkl, feeding labels "1", "2", "4" respectively. That
exact three-model sequence used to be duplicated almost verbatim at two call
sites in seedling_measurment.py (run_apical_pipeline and
segment_single_seedling) and had already drifted apart once. This module
gives it one place to live, `BinaryBackend.predict_into`, so both call sites
can eventually delegate to the same code instead of maintaining their own
copies.

Both GUI call sites now route here through `Gui._segment_paths`
(seedling_measurment.py), so this module is the only place a weight file is
named. `MulticlassBackend` (one 4-class checkpoint standing in for all three
binary models) is also implemented here, but only reachable by explicitly
setting DLHOOK_SEG_BACKEND=multiclass -- flipping DEFAULT_BACKEND to it is a
later, separately-gated step. Until then this resolves to BinaryBackend and
runtime behaviour is exactly what it was before the routing change.
"""
from __future__ import annotations

import os

import numpy as np

from models.UNetInference import MARGIN, get_predictor

# DLHOOK_SEG_BACKEND env var name, and the backend names it (and get_backend)
# accept. Keep DEFAULT_BACKEND "binary" until the multiclass backend is
# implemented and proven out -- flipping this is a later, separately-gated
# step, not this one.
ENV_VAR = "DLHOOK_SEG_BACKEND"
DEFAULT_BACKEND = "binary"
VALID_BACKENDS = ("binary", "multiclass")

# Default 4-class checkpoint MulticlassBackend loads when the caller does not
# name one explicitly. A groupnorm-head UNetGNRes trained at
# data.patch_size: 252 (see MulticlassBackend's docstring for why that
# geometry, not the GUI's default 572, is what this checkpoint needs).
DEFAULT_MULTICLASS_CHECKPOINT = "weights/multiclass/dlhook_4class_v1.pt"

# Class index -> mask store label. Class 0 (background) is intentionally
# absent: MaskStore only ever holds foreground labels, and nothing downstream
# reads a "background" label.
_CLASS_TO_LABEL = {1: "1", 2: "2", 3: "4"}


def resolve_backend_name() -> str:
    """The backend name to use: the `DLHOOK_SEG_BACKEND` env var if set
    (must be one of VALID_BACKENDS), else DEFAULT_BACKEND. Raises a clear
    ValueError on an unrecognised env var value rather than silently falling
    back, so a typo'd env var fails loudly instead of quietly running the
    wrong model."""
    name = os.environ.get(ENV_VAR, DEFAULT_BACKEND)
    if name not in VALID_BACKENDS:
        raise ValueError(
            f"{ENV_VAR}={name!r} is not a valid segmentation backend "
            f"(expected one of {VALID_BACKENDS})"
        )
    return name


class BinaryBackend:
    """Reproduces exactly today's three-binary-model pipeline: cotyledon_v5
    (label "1"), hypocot_v5 (label "2"), germ_v1 (label "4"), run in that
    order via the process-wide `get_predictor` cache. Until the routing
    change, run_apical_pipeline and segment_single_seedling each built this
    identical (label, weight_path) list and looped over it independently;
    both now call `Gui._segment_paths`, which lands here. cotyledon_v3
    (label "3") is deliberately absent, same as those call sites were:
    process_single_frame never reads it.

    `get_predictor` is called per invocation rather than once up front. That
    is not a reload: it is a process-wide cache keyed on the weight path, so
    every chunk of a batch run shares one loaded model, exactly as the old
    hoisted list did.

    `predict_into` takes the already-decided list of image paths to segment
    (a batch-pipeline chunk, or one seedling's full frame list on the
    on-demand path) -- IMAGE_CHUNK-sized batching over a larger file list, if
    any, is the caller's concern (it already differs between the two
    existing call sites: chunked in run_apical_pipeline, unchunked in
    segment_single_seedling), not this method's.
    """

    # The exact (label, weight_path) sequence both existing call sites use,
    # in this order. Pinned by tests/test_segmentation_backends.py.
    WEIGHTS = (
        ("1", "weights/RootPainter_weights/cotyledon_v5.pkl"),
        ("2", "weights/RootPainter_weights/hypocot_v5.pkl"),
        ("4", "weights/RootPainter_weights/germ_v1.pkl"),
    )

    def predict_into(self, mask_store, image_paths):
        """Segment `image_paths` with each of the three binary models in
        turn and store every label's masks into `mask_store` -- one
        `predict_files` call and one `put_raw_bulk` call per label, in
        WEIGHTS order. No progress/reporter argument is threaded through
        here: neither existing call site passes one into predict_files or
        put_raw_bulk either."""
        for label, weight_path in self.WEIGHTS:
            predictor = get_predictor(weight_path)
            masks_by_filename = predictor.predict_files(image_paths, label=label)
            mask_store.put_raw_bulk(masks_by_filename, label)


class MulticlassBackend:
    """One 4-class checkpoint whose argmax output populates the same three
    labels ("1"/"2"/"4") that BinaryBackend gets from three separate models.
    DEFAULT_BACKEND stays "binary" until a later, separately-gated step
    flips it -- this backend is reachable only via
    DLHOOK_SEG_BACKEND=multiclass until then.

    Geometry is pinned to the checkpoint's own training patch size, NOT the
    GUI's default 572/560/6 (UNetInference.IN_SIZE/OUT_SIZE/MARGIN): the
    checkpoint was trained at training_config.yaml's `data.patch_size: 252`,
    chosen because these crops are narrow (median 74x246px). Run at 572, the
    default geometry reflect-pads ~94% of a typical crop with synthetic
    context that patch_size=252 avoids -- previously producing meaningless
    output. So out_size is 252, in_size is 252 + 2*MARGIN (MARGIN imported
    from models.UNetInference, same fixed context margin UNetInference
    itself uses, just not its IN_SIZE/OUT_SIZE), and margin is MARGIN.

    Every constructor arg may be None ("use the default"), matching how
    get_backend already calls this class -- resolved here rather than via
    mutable default arguments so the resolved values are also what's cached
    into get_backend's _BACKEND_CACHE key.
    """

    def __init__(self, checkpoint=None, *, in_size=None, out_size=None, num_classes=None):
        self.checkpoint = checkpoint or DEFAULT_MULTICLASS_CHECKPOINT
        self.out_size = out_size if out_size is not None else 252
        self.in_size = in_size if in_size is not None else self.out_size + 2 * MARGIN
        self.margin = MARGIN
        self.num_classes = num_classes if num_classes is not None else 4
        # Constructed lazily, on first predict_into call -- see that
        # method's docstring for why. get_backend() must be free to hand
        # back a MulticlassBackend without paying for a checkpoint load.
        self._predictor = None

    def _get_predictor(self):
        """Load (and cache on self) the MulticlassInference for this
        backend's checkpoint/geometry, only on first use. Deliberately not
        done in __init__: get_backend() is called from Gui._segment_paths
        just to resolve which backend is in play, and must not load a
        multi-megabyte checkpoint onto the GPU merely because a backend
        object was requested rather than actually run."""
        if self._predictor is None:
            # Imported here, not at module scope, so importing
            # segmentation_backends never imports torch/MulticlassInference
            # unless the multiclass backend is actually used.
            from models.multiclass_inference import MulticlassInference

            self._predictor = MulticlassInference(
                self.checkpoint,
                num_classes=self.num_classes,
                in_size=self.in_size,
                out_size=self.out_size,
                margin=self.margin,
            )
        return self._predictor

    def predict_into(self, mask_store, image_paths):
        """Segment `image_paths` with the one 4-class checkpoint and store
        the result into `mask_store` under the same three labels
        BinaryBackend uses.

        Why binary-split rather than storing the raw 0..3 label map
        directly: MaskStore keys on (file_name, label) and MaskStore.dump()
        applies cv2.bitwise_not() to whatever it holds, on the assumption
        that a stored mask is binary (0/255) -- see MaskStore's own
        docstring and models/UNetInference.py's _to_binary_mask. A raw
        label map handed to dump() unmodified would invert to 255/254/253/
        252 instead of a sensible mask, and every downstream MaskStore.get()
        call site would need to know to un-argmax it first. Splitting each
        label map into three independent binary masks here -- at the one
        point where this backend's output enters the shared store -- keeps
        every consumer (MaskStore, the GUI, export) exactly as it already
        is for BinaryBackend's masks.

        Class 0 (background) is never split out or stored -- nothing reads
        a "background" label. Class 3 maps to label "4" (not "3"): "4" is
        the GUI's radicle/germination label; label "3" exists on disk for
        legacy reasons but is deliberately never populated (see CLAUDE.md).
        """
        predictor = self._get_predictor()
        label_maps = predictor.predict_files_labelmaps(image_paths)

        for class_index, label in _CLASS_TO_LABEL.items():
            masks_by_filename = {
                file_name: (label_map == class_index).astype(np.uint8) * 255
                for file_name, label_map in label_maps.items()
            }
            mask_store.put_raw_bulk(masks_by_filename, label)


# Process-wide cache of backend instances. Deliberately its OWN cache, not a
# reuse of models.UNetInference.get_predictor's cache: that cache ignores
# **kwargs on a cache hit (models/UNetInference.py:89, "later calls with the
# same path return the existing instance and ignore them"), so a second
# get_backend() call with a different checkpoint/geometry would silently get
# back the first call's (wrongly configured) instance instead of a fresh one.
# Keying on the full (name, checkpoint, in_size, out_size, num_classes) tuple
# means a geometry mismatch produces a distinct cache entry instead of a
# stale hit.
_BACKEND_CACHE = {}


def get_backend(name=None, *, checkpoint=None, in_size=None, out_size=None, num_classes=None):
    """Return a process-cached backend instance for `name` (defaulting to
    `resolve_backend_name()`), constructing it only on first request for
    this exact `(name, checkpoint, in_size, out_size, num_classes)` key.
    `checkpoint`/`in_size`/`out_size`/`num_classes` are unused by
    BinaryBackend (it has no checkpoint or configurable geometry) but are
    accepted and folded into the cache key regardless, so the identical call
    signature also serves MulticlassBackend, whose checkpoint/geometry
    default to None meaning "use MulticlassBackend's own defaults" -- see its
    __init__."""
    if name is None:
        name = resolve_backend_name()
    elif name not in VALID_BACKENDS:
        raise ValueError(f"Unknown segmentation backend {name!r} (expected one of {VALID_BACKENDS})")

    key = (name, checkpoint, in_size, out_size, num_classes)
    backend = _BACKEND_CACHE.get(key)
    if backend is not None:
        return backend

    if name == "binary":
        backend = BinaryBackend()
    else:
        backend = MulticlassBackend(
            checkpoint, in_size=in_size, out_size=out_size, num_classes=num_classes
        )

    _BACKEND_CACHE[key] = backend
    return backend
