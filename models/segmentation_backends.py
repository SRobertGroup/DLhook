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
named. Adding the multiclass backend, and flipping DEFAULT_BACKEND to it,
are LATER separately-gated steps -- until then this resolves to
BinaryBackend and runtime behaviour is exactly what it was before the
routing change.
"""
from __future__ import annotations

import os

from models.UNetInference import get_predictor

# DLHOOK_SEG_BACKEND env var name, and the backend names it (and get_backend)
# accept. Keep DEFAULT_BACKEND "binary" until the multiclass backend is
# implemented and proven out -- flipping this is a later, separately-gated
# step, not this one.
ENV_VAR = "DLHOOK_SEG_BACKEND"
DEFAULT_BACKEND = "binary"
VALID_BACKENDS = ("binary", "multiclass")


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
    """Placeholder -- NOT implemented in this step. A later, separately-gated
    step adds this: one 4-class checkpoint whose argmax output populates the
    same three labels ("1"/"2"/"4") that BinaryBackend gets from three
    separate models, after which DEFAULT_BACKEND may be flipped to
    "multiclass". Do not implement its body here."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "MulticlassBackend is not implemented yet -- DEFAULT_BACKEND stays "
            "'binary' until a later step adds it."
        )


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
    BinaryBackend today (it has no checkpoint or configurable geometry) but
    are accepted and folded into the cache key regardless, so the same call
    signature already works for the future MulticlassBackend without another
    cache-invalidation bug."""
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
