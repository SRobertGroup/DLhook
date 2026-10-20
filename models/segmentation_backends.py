"""Which segmentation models actually run underneath the GUI pipeline.

The default is now the 4-class MulticlassBackend (labels "1", "2"; v2 checkpoint,
v1 when v2 is missing; its "2" is also the hypocotyl trait mask); the legacy path
(DLHOOK_SEG_BACKEND=binary) is three independent binary RootPainter
checkpoints -- cotyledon_v5.pkl,
hypocot_v5.pkl, germ_v1.pkl, feeding labels "1", "2", "4" respectively. That
exact three-model sequence used to be duplicated almost verbatim at two call
sites in seedling_measurement.py (run_apical_pipeline and
segment_single_seedling) and had already drifted apart once. This module
gives it one place to live, `BinaryBackend.predict_into`, so both call sites
can eventually delegate to the same code instead of maintaining their own
copies.

Both GUI call sites now route here through `Gui._segment_paths`
(seedling_measurement.py), so this module is the only place a weight file is
named. `MulticlassBackend` (one 4-class checkpoint standing in for all three
binary models) is also implemented here, but only reachable by explicitly
setting DLHOOK_SEG_BACKEND=multiclass -- flipping DEFAULT_BACKEND to it is a
later, separately-gated step. Until then this resolves to BinaryBackend and
runtime behaviour is exactly what it was before the routing change.
"""
from __future__ import annotations

import logging
import os

import numpy as np

from models.UNetInference import MARGIN, get_predictor
from utils.paths import repo_path

log = logging.getLogger(__name__)

# DLHOOK_SEG_BACKEND env var name, and the backend names it (and get_backend)
# accept. "multiclass" is the default; "binary" is the legacy three-model path
# (DLHOOK_SEG_BACKEND=binary), also used automatically when the default
# multiclass checkpoint is not on disk (see resolve_backend_name).
ENV_VAR = "DLHOOK_SEG_BACKEND"
DEFAULT_BACKEND = "multiclass"
VALID_BACKENDS = ("binary", "multiclass")

# Default 4-class checkpoint MulticlassBackend loads when the caller does not
# name one explicitly. A groupnorm-head UNetGNRes trained at
# data.patch_size: 252 (see MulticlassBackend's docstring for why that
# geometry, not the GUI's default 572, is what this checkpoint needs).
# v2 (multi/configs/training_config_v2.yaml): v1 warm start retrained on every crop folder with
# human strokes on the hook region (ui/seg_annotator.py); its hypocotyl reaches the junction, so it
# also feeds the traits. v1 stays on disk: the learned germination detector was fitted on it.
DEFAULT_MULTICLASS_CHECKPOINT = "weights/multiclass/dlhook_4class_v2.pt"
PREVIOUS_MULTICLASS_CHECKPOINT = "weights/multiclass/dlhook_4class_v1.pt"

# Class index -> mask store label. Class 0 (background) is intentionally
# absent: MaskStore only ever holds foreground labels, and nothing downstream
# reads a "background" label.
_CLASS_TO_LABEL = {1: "1", 2: "2", 3: "4"}

# Mask-store label for the binary hypocot_v5 mask MulticlassBackend writes beside its own "2"
# only with hypocotyl_binary=True (off by default since v2: the binary mask misses the hook region,
# so the traits read "2"; Gui._trait_mask still prefers this label when present).
HYPOCOTYL_BINARY_LABEL = "hypocotyl_binary"


def default_multiclass_checkpoint() -> str:
    """DEFAULT_MULTICLASS_CHECKPOINT when it is on disk, else PREVIOUS_MULTICLASS_CHECKPOINT (with a
    warning) when that one is; else DEFAULT_MULTICLASS_CHECKPOINT (missing: the caller decides)."""
    if os.path.exists(repo_path(DEFAULT_MULTICLASS_CHECKPOINT)):
        return DEFAULT_MULTICLASS_CHECKPOINT
    if os.path.exists(repo_path(PREVIOUS_MULTICLASS_CHECKPOINT)):
        _warn_once("seg-checkpoint", f"{DEFAULT_MULTICLASS_CHECKPOINT} not found; using "
                                     f"{PREVIOUS_MULTICLASS_CHECKPOINT}.")
        return PREVIOUS_MULTICLASS_CHECKPOINT
    return DEFAULT_MULTICLASS_CHECKPOINT


def resolve_backend_name() -> str:
    """The backend name to use: the `DLHOOK_SEG_BACKEND` env var if set
    (must be one of VALID_BACKENDS), else DEFAULT_BACKEND. Raises a clear
    ValueError on an unrecognised env var value rather than silently falling
    back, so a typo'd env var fails loudly instead of quietly running the
    wrong model.

    When the env var is NOT set and the default (multiclass) checkpoint is
    missing on disk, falls back to "binary" with a [WARNING] so a fresh
    checkout still runs. An explicit DLHOOK_SEG_BACKEND=multiclass is never
    downgraded: it errors at load, as before."""
    explicit = os.environ.get(ENV_VAR)
    name = explicit if explicit is not None else DEFAULT_BACKEND
    if name not in VALID_BACKENDS:
        raise ValueError(
            f"{ENV_VAR}={name!r} is not a valid segmentation backend "
            f"(expected one of {VALID_BACKENDS})"
        )
    if explicit is None and name == "multiclass" and not os.path.exists(repo_path(default_multiclass_checkpoint())):
        _warn_once(
            "seg-backend",
            f"Multiclass checkpoints {DEFAULT_MULTICLASS_CHECKPOINT} / {PREVIOUS_MULTICLASS_CHECKPOINT} "
            f"not found; falling back to the legacy binary segmentation models.",
        )
        return "binary"
    return name


_WARNED = set()


def _warn_once(key, message):
    """Print `message` the first time `key` is seen in this process."""
    if key not in _WARNED:
        _WARNED.add(key)
        log.warning(message)


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
    This is the default backend (DEFAULT_BACKEND). It writes "1"/"2" from the
    argmax; label "4" depends on `radicle_source` ("none" by default: germination
    comes from the learned detector, so no radicle mask is stored; "multiclass"
    stores argmax class 3 under "4"). With `hypocotyl_binary` (default False since v2)
    it also runs the binary hypocot_v5 model into HYPOCOTYL_BINARY_LABEL.

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

    RADICLE_SOURCES = ("none", "multiclass")

    def __init__(self, checkpoint=None, *, in_size=None, out_size=None, num_classes=None,
                 radicle_source="none", hypocotyl_binary=False):
        if radicle_source not in self.RADICLE_SOURCES:
            raise ValueError(
                f"radicle_source={radicle_source!r} is not valid (expected one of {self.RADICLE_SOURCES})"
            )
        self.radicle_source = radicle_source
        self.hypocotyl_binary = bool(hypocotyl_binary)
        self.checkpoint = checkpoint or default_multiclass_checkpoint()
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
            if label == "4" and self.radicle_source != "multiclass":
                continue            # no radicle mask: written nowhere, by design
            masks_by_filename = {
                file_name: (label_map == class_index).astype(np.uint8) * 255
                for file_name, label_map in label_maps.items()
            }
            mask_store.put_raw_bulk(masks_by_filename, label)

        if self.hypocotyl_binary:
            # Same weights BinaryBackend uses for label "2" (reused, not re-named here).
            hypocotyl_weights = dict(BinaryBackend.WEIGHTS)["2"]
            predictor = get_predictor(hypocotyl_weights)
            masks_by_filename = predictor.predict_files(image_paths, label="2")
            mask_store.put_raw_bulk(masks_by_filename, HYPOCOTYL_BINARY_LABEL)


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


def get_backend(name=None, *, checkpoint=None, in_size=None, out_size=None, num_classes=None,
                radicle_source="none", hypocotyl_binary=False):
    """Return a process-cached backend instance for `name` (defaulting to
    `resolve_backend_name()`), constructing it only on first request for
    this exact `(name, checkpoint, in_size, out_size, num_classes,
    radicle_source, hypocotyl_binary)` key.
    These arguments are unused by BinaryBackend (it has no checkpoint,
    configurable geometry or options) but are accepted and folded into the
    cache key regardless, so the identical call signature also serves
    MulticlassBackend, whose checkpoint/geometry default to None meaning "use
    MulticlassBackend's own defaults" -- see its __init__."""
    if name is None:
        name = resolve_backend_name()
    elif name not in VALID_BACKENDS:
        raise ValueError(f"Unknown segmentation backend {name!r} (expected one of {VALID_BACKENDS})")

    key = (name, checkpoint, in_size, out_size, num_classes, radicle_source, bool(hypocotyl_binary))
    backend = _BACKEND_CACHE.get(key)
    if backend is not None:
        return backend

    if name == "binary":
        backend = BinaryBackend()
    else:
        backend = MulticlassBackend(
            checkpoint, in_size=in_size, out_size=out_size, num_classes=num_classes,
            radicle_source=radicle_source, hypocotyl_binary=hypocotyl_binary,
        )

    _BACKEND_CACHE[key] = backend
    return backend


# --- germination time-zero ----------------------------------------------------
#
# Germination sets kinematic time-zero. Two methods:
#   "learned" (default) -- utils/germination_learned.py: a per-frame classifier on the
#       four-class model's softmax maps plus one onset per seedling. Needs no seed point.
#       Scored on 75 unseen seedlings: 75/75 found, 71% exact, 91% within one frame.
#   "rule" -- utils/germination_detector.py's radicle area near the operator's seed point
#       (the previous behaviour).
# DLHOOK_GERMINATION=rule restores the rule; the rule is also used automatically when the
# weights file or the checkpoint it names is missing, or when the learned run fails.
GERMINATION_ENV_VAR = "DLHOOK_GERMINATION"
DEFAULT_GERMINATION = "learned"
VALID_GERMINATION = ("learned", "rule")
DEFAULT_GERMINATION_WEIGHTS = "weights/germination_detector.json"


def resolve_germination_method(weights_path=DEFAULT_GERMINATION_WEIGHTS) -> str:
    """'learned' or 'rule': the DLHOOK_GERMINATION env var if set (must be valid), else the
    default -- downgraded to 'rule' when the learned detector's files are not on disk."""
    name = os.environ.get(GERMINATION_ENV_VAR, DEFAULT_GERMINATION)
    if name not in VALID_GERMINATION:
        raise ValueError(f"{GERMINATION_ENV_VAR}={name!r} is not valid (expected one of {VALID_GERMINATION})")
    if name == "learned":
        if not os.path.exists(repo_path(weights_path)):
            return "rule"
        try:
            from utils.germination_learned import GerminationModel
            checkpoint = GerminationModel.load(weights_path).checkpoint or PREVIOUS_MULTICLASS_CHECKPOINT
        except (OSError, ValueError, KeyError):
            return "rule"
        if not os.path.exists(repo_path(checkpoint)):
            return "rule"
    return name


class LearnedGerminationRunner:
    """Runs the four-class model at its training geometry over one seedling's crops and returns
    the learned onset. The softmax statistics it needs are recomputed here rather than taken from
    the segmentation pass, so it works whichever backend segmented the crops."""

    def __init__(self, weights_path=DEFAULT_GERMINATION_WEIGHTS):
        from models.multiclass_inference import MulticlassInference
        from utils.germination_learned import GerminationModel

        self.weights_path = weights_path
        self.model = GerminationModel.load(weights_path)
        checkpoint = self.model.checkpoint or PREVIOUS_MULTICLASS_CHECKPOINT   # fitted on v1
        out_size = 252                       # the training geometry, as MulticlassBackend
        self.inference = MulticlassInference(checkpoint, num_classes=4, in_size=out_size + 2 * MARGIN,
                                             out_size=out_size, margin=MARGIN)

    def visible_probs(self, image_paths):
        """Per-frame probability that a radicle is visible; NaN for an unreadable crop."""
        import cv2
        from models.UNetInference import IMAGE_CHUNK
        from utils.germination_learned import frame_features

        probs = []
        for start in range(0, len(image_paths), IMAGE_CHUNK):
            images, ok = [], []
            for path in image_paths[start:start + IMAGE_CHUNK]:
                image = cv2.imread(path)
                ok.append(image is not None)
                if image is not None:
                    images.append(image)
            maps = iter(self.inference.segment_many_argmax(images, return_probs=True) if images else [])
            for readable in ok:
                if readable:
                    probs.append(float(self.model.visible_probs([frame_features(next(maps))])[0]))
                else:
                    probs.append(float("nan"))
        return probs

    def onset(self, image_paths):
        """(onset frame index or None, per-frame probabilities) for one seedling's crops in time order.
        Unreadable crops are treated as 'unknown' (probability 0.5) so they never decide the onset."""
        from utils.germination_learned import onset_or_none

        probs = self.visible_probs(image_paths)
        filled = [0.5 if p != p else p for p in probs]
        return onset_or_none(filled), probs


_GERMINATION_CACHE = {}


def get_germination_runner(weights_path=DEFAULT_GERMINATION_WEIGHTS):
    """Process-cached LearnedGerminationRunner (loads the four-class model once)."""
    runner = _GERMINATION_CACHE.get(weights_path)
    if runner is None:
        runner = _GERMINATION_CACHE[weights_path] = LearnedGerminationRunner(weights_path)
    return runner


# --- angle source -------------------------------------------------------------
#
# Where the per-frame hook angle comes from:
#   "landmarks" (default) -- a landmark-head ensemble (junction + direction fields), read at the
#       training geometry; validated far closer to hand measurements than the ellipse fit.
#   "ellipse" -- the legacy cotyledon/hypocotyl ellipse geometry (utils/apicalhook_angle.py).
# DLHOOK_ANGLE_SOURCE=ellipse restores the legacy path; it is also used automatically (with a
# warning) when any landmark checkpoint is missing.
ANGLE_SOURCE_ENV = "DLHOOK_ANGLE_SOURCE"
DEFAULT_ANGLE_SOURCE = "landmarks"
VALID_ANGLE_SOURCES = ("landmarks", "ellipse")
# "+"-joined spec, the syntax LandmarkEnsemble.from_checkpoints takes. The default members carry the
# learned "measurable" head (audit M-10; multi/configs/training_landmarks_all_{a,b,c}_mt.yaml): on
# BerryBox v2 they gate 76% of the skipped frames at 99% of the measured kept, and the angle error is
# 13.5 deg against 13.9 for the previous members (multi/results/measurable_validation_mt.json,
# angle_validation_bbval_*). A machine holding only the previous members uses those, with the
# logistic gate (default_landmark_checkpoints).
DEFAULT_LANDMARK_CHECKPOINTS = (
    "weights/landmarks/dlhook_landmarks_m_a.pt"
    "+weights/landmarks/dlhook_landmarks_m_b.pt"
    "+weights/landmarks/dlhook_landmarks_m_c.pt"
)
PREVIOUS_LANDMARK_CHECKPOINTS = (
    "weights/landmarks/dlhook_landmarks_a.pt"
    "+weights/landmarks/dlhook_landmarks_b.pt"
    "+weights/landmarks/dlhook_landmarks_c.pt"
)


def _checkpoint_paths(spec):
    return [p.strip() for p in spec.split("+") if p.strip()]


def _all_exist(spec):
    return all(os.path.exists(repo_path(p)) for p in _checkpoint_paths(spec))


def default_landmark_checkpoints():
    """DEFAULT_LANDMARK_CHECKPOINTS when all its files exist, else PREVIOUS_LANDMARK_CHECKPOINTS when
    all of those do (with a warning), else DEFAULT (resolve_angle_source then falls back to ellipse)."""
    if not _all_exist(DEFAULT_LANDMARK_CHECKPOINTS) and _all_exist(PREVIOUS_LANDMARK_CHECKPOINTS):
        _warn_once(
            "landmark-previous",
            "Landmark checkpoints with the measurable head not found "
            f"({DEFAULT_LANDMARK_CHECKPOINTS}); using the previous members with the logistic gate.",
        )
        return PREVIOUS_LANDMARK_CHECKPOINTS
    return DEFAULT_LANDMARK_CHECKPOINTS


def resolve_angle_source(checkpoints=None) -> str:
    """'landmarks' or 'ellipse': the DLHOOK_ANGLE_SOURCE env var if set (must be valid), else the
    default -- downgraded to 'ellipse' (with a warning) when any checkpoint of the spec is missing.
    `checkpoints` None: default_landmark_checkpoints()."""
    if checkpoints is None:
        checkpoints = default_landmark_checkpoints()
    name = os.environ.get(ANGLE_SOURCE_ENV, DEFAULT_ANGLE_SOURCE)
    if name not in VALID_ANGLE_SOURCES:
        raise ValueError(f"{ANGLE_SOURCE_ENV}={name!r} is not valid (expected one of {VALID_ANGLE_SOURCES})")
    if name == "landmarks":
        missing = [p for p in _checkpoint_paths(checkpoints) if not os.path.exists(repo_path(p))]
        if missing:
            _warn_once(
                "angle-source",
                f"Landmark checkpoint(s) not found ({', '.join(missing)}); "
                f"falling back to the legacy ellipse angle source.",
            )
            return "ellipse"
    return name


class LandmarkRunner:
    """Reads landmarks (junction, directions, hook angle) from crop files with the landmark
    ensemble at the training geometry (out_size=252, in_size=252+2*MARGIN, as MulticlassBackend),
    not the 572 default. The model loads on first use, not in __init__, so resolving a runner
    never costs a checkpoint load. Flip TTA stays at its default (on)."""

    def __init__(self, checkpoints=DEFAULT_LANDMARK_CHECKPOINTS):
        self.checkpoints = checkpoints
        self.out_size = 252
        self.in_size = self.out_size + 2 * MARGIN
        self.margin = MARGIN
        self.num_classes = 4
        self._model = None

    def _get_model(self):
        if self._model is None:
            # Imported here so importing this module never pulls in torch.
            from models.multiclass_inference import LandmarkEnsemble

            # A single path gives a plain MulticlassInference; both have predict_files_landmarks.
            self._model = LandmarkEnsemble.from_checkpoints(
                self.checkpoints,
                num_classes=self.num_classes,
                in_size=self.in_size,
                out_size=self.out_size,
                margin=self.margin,
            )
        return self._model

    def predict_paths(self, image_paths):
        """{basename: readout dict or None}; unreadable files are absent."""
        return self._get_model().predict_files_landmarks(list(image_paths))


_LANDMARK_CACHE = {}


def get_landmark_runner(checkpoints=None):
    """Process-cached LandmarkRunner (the ensemble loads once, on first predict_paths).
    `checkpoints` None: default_landmark_checkpoints()."""
    if checkpoints is None:
        checkpoints = default_landmark_checkpoints()
    runner = _LANDMARK_CACHE.get(checkpoints)
    if runner is None:
        runner = _LANDMARK_CACHE[checkpoints] = LandmarkRunner(checkpoints)
    return runner
