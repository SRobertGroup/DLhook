"""Tests for models/segmentation_backends.py.

This module is additive-only (see its docstring): nothing in
seedling_measurment.py imports it yet, so these tests are the only thing
pinning its behaviour for now -- most importantly the exact three-model
call sequence BinaryBackend.predict_into reproduces, so a later step that
wires the GUI through it (or adds MulticlassBackend) can be proven not to
have changed that sequence.

House style: synthesise everything in-test (fake predictor/mask store, no
real weights or torch models needed), plain asserts, no GPU."""
from __future__ import annotations

import numpy as np
import pytest

from models import segmentation_backends
from models.segmentation_backends import (
    DEFAULT_BACKEND,
    ENV_VAR,
    BinaryBackend,
    MulticlassBackend,
    get_backend,
    resolve_backend_name,
)
from models.UNetInference import MARGIN


class _RecordingPredictor:
    """Stands in for a real UNetInference: just records the exact
    `predict_files` call it received and returns a mask keyed by label so
    the caller's put_raw_bulk call can be checked too."""

    def __init__(self, weight_path, calls):
        self.weight_path = weight_path
        self._calls = calls

    def predict_files(self, image_paths, label="0"):
        self._calls.append((self.weight_path, tuple(image_paths), label))
        return {"frame.png": f"mask-for-{label}"}


class _RecordingMaskStore:
    def __init__(self):
        self.put_raw_bulk_calls = []

    def put_raw_bulk(self, masks_by_filename, label):
        self.put_raw_bulk_calls.append((masks_by_filename, label))


class _FakeMulticlassInference:
    """Test double for models.multiclass_inference.MulticlassInference:
    records the constructor kwargs MulticlassBackend._get_predictor passed
    in (so the geometry pin can be checked without a real checkpoint), and
    returns a fixed, test-supplied label map from predict_files_labelmaps
    instead of running one."""

    #: set by each test before exercising the backend.
    label_maps_to_return = {}

    def __init__(self, checkpoint, num_classes=None, in_size=None, out_size=None, margin=None):
        self.checkpoint = checkpoint
        self.num_classes = num_classes
        self.in_size = in_size
        self.out_size = out_size
        self.margin = margin
        self.predict_files_labelmaps_calls = []

    def predict_files_labelmaps(self, image_paths):
        self.predict_files_labelmaps_calls.append(tuple(image_paths))
        return type(self).label_maps_to_return


# --- resolve_backend_name ---------------------------------------------------

def test_resolve_backend_name_defaults_to_binary(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert DEFAULT_BACKEND == "binary"
    assert resolve_backend_name() == "binary"


def test_resolve_backend_name_env_var_overrides(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "multiclass")
    assert resolve_backend_name() == "multiclass"


def test_resolve_backend_name_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "some_typo")
    with pytest.raises(ValueError):
        resolve_backend_name()


# --- BinaryBackend.predict_into: the fallback pin ---------------------------

def test_binary_backend_predict_into_pins_todays_call_sequence(monkeypatch):
    get_predictor_calls = []
    predict_files_calls = []

    def fake_get_predictor(weight_path, **kwargs):
        get_predictor_calls.append(weight_path)
        return _RecordingPredictor(weight_path, predict_files_calls)

    monkeypatch.setattr(segmentation_backends, "get_predictor", fake_get_predictor)

    mask_store = _RecordingMaskStore()
    backend = BinaryBackend()
    backend.predict_into(mask_store, ["0-crop-a.png", "0-crop-b.png"])

    # Exactly cotyledon_v5, hypocot_v5, germ_v1, in that order.
    assert get_predictor_calls == [
        "weights/RootPainter_weights/cotyledon_v5.pkl",
        "weights/RootPainter_weights/hypocot_v5.pkl",
        "weights/RootPainter_weights/germ_v1.pkl",
    ]

    # One predict_files call per label, in order "1", "2", "4", each against
    # the exact same image_paths list that was handed to predict_into.
    assert [call[2] for call in predict_files_calls] == ["1", "2", "4"]
    assert all(call[1] == ("0-crop-a.png", "0-crop-b.png") for call in predict_files_calls)
    assert [call[0] for call in predict_files_calls] == [
        "weights/RootPainter_weights/cotyledon_v5.pkl",
        "weights/RootPainter_weights/hypocot_v5.pkl",
        "weights/RootPainter_weights/germ_v1.pkl",
    ]

    # One put_raw_bulk call per label, same order, carrying that label's own
    # predict_files result through unmodified.
    assert [call[1] for call in mask_store.put_raw_bulk_calls] == ["1", "2", "4"]
    assert mask_store.put_raw_bulk_calls[0][0] == {"frame.png": "mask-for-1"}
    assert mask_store.put_raw_bulk_calls[1][0] == {"frame.png": "mask-for-2"}
    assert mask_store.put_raw_bulk_calls[2][0] == {"frame.png": "mask-for-4"}


# --- get_backend cache -------------------------------------------------------

def test_get_backend_cache_returns_identical_object_for_identical_keys():
    segmentation_backends._BACKEND_CACHE.clear()
    a = get_backend("binary")
    b = get_backend("binary")
    assert a is b
    assert isinstance(a, BinaryBackend)


def test_get_backend_cache_returns_distinct_objects_for_differing_keys():
    segmentation_backends._BACKEND_CACHE.clear()
    a = get_backend("binary")
    b = get_backend("binary", checkpoint="some/other/checkpoint.pt")
    c = get_backend("binary", in_size=264, out_size=252, num_classes=4)
    assert a is not b
    assert a is not c
    assert b is not c


def test_get_backend_rejects_unknown_name():
    with pytest.raises(ValueError):
        get_backend("not_a_real_backend")


def test_get_backend_does_not_reuse_get_predictors_cache():
    """get_backend must key its OWN cache rather than piggyback on
    models.UNetInference.get_predictor's -- that cache ignores **kwargs on a
    cache hit (models/UNetInference.py:89), so reusing it here would make a
    later geometry change silently sticky. Distinct dicts is enough to prove
    they're not the same cache."""
    from models.UNetInference import _PREDICTOR_CACHE

    assert segmentation_backends._BACKEND_CACHE is not _PREDICTOR_CACHE


# --- MulticlassBackend -------------------------------------------------------

def test_multiclass_backend_pins_training_geometry(monkeypatch):
    """The single most valuable test here: MulticlassBackend must construct
    MulticlassInference at the checkpoint's own training geometry
    (out_size=252, in_size=252+2*MARGIN, num_classes=4), never the GUI's
    default 572/560/6 -- that default reflect-pads ~94% of a typical crop
    with synthetic context and previously produced meaningless output. This
    guards against regressing back to that bug."""
    monkeypatch.setattr("models.multiclass_inference.MulticlassInference", _FakeMulticlassInference)

    backend = MulticlassBackend()
    predictor = backend._get_predictor()

    assert predictor.out_size == 252
    assert predictor.in_size == 252 + 2 * MARGIN
    assert predictor.num_classes == 4
    assert predictor.margin == MARGIN


def test_multiclass_backend_predict_into_splits_labelmap_correctly(monkeypatch):
    """A synthetic label map containing all four class values, split into
    three binary masks: each must be exactly (label_map == k) * 255, uint8,
    pairwise disjoint, and their union must equal the non-background
    region. Also pins the label order/mapping: put_raw_bulk is called
    exactly three times, labels "1", "2", "4" in that order, with class
    index 3 landing under "4" (the GUI's radicle label, not "3")."""
    monkeypatch.setattr("models.multiclass_inference.MulticlassInference", _FakeMulticlassInference)

    label_map = np.array([
        [0, 1, 2, 3],
        [3, 2, 1, 0],
        [1, 0, 3, 2],
    ], dtype=np.uint8)
    _FakeMulticlassInference.label_maps_to_return = {"frame.png": label_map}

    mask_store = _RecordingMaskStore()
    backend = MulticlassBackend()
    backend.predict_into(mask_store, ["frame.png"])

    assert [call[1] for call in mask_store.put_raw_bulk_calls] == ["1", "2", "4"]

    masks_by_label = {label: masks["frame.png"] for masks, label in mask_store.put_raw_bulk_calls}
    mask_1, mask_2, mask_4 = masks_by_label["1"], masks_by_label["2"], masks_by_label["4"]

    for mask, class_index in ((mask_1, 1), (mask_2, 2), (mask_4, 3)):
        assert mask.dtype == np.uint8
        np.testing.assert_array_equal(mask, (label_map == class_index).astype(np.uint8) * 255)

    assert not np.any((mask_1 > 0) & (mask_2 > 0))
    assert not np.any((mask_1 > 0) & (mask_4 > 0))
    assert not np.any((mask_2 > 0) & (mask_4 > 0))

    union = (mask_1 > 0) | (mask_2 > 0) | (mask_4 > 0)
    np.testing.assert_array_equal(union, label_map != 0)


def test_multiclass_backend_constructs_inference_lazily(monkeypatch):
    """get_backend("multiclass") alone must not load the checkpoint --
    only the first predict_into call may construct MulticlassInference,
    and a second predict_into call must reuse that cached instance."""
    construct_calls = []

    class _CountingFake(_FakeMulticlassInference):
        def __init__(self, *args, **kwargs):
            construct_calls.append((args, kwargs))
            super().__init__(*args, **kwargs)

    _CountingFake.label_maps_to_return = {"frame.png": np.zeros((2, 2), dtype=np.uint8)}
    monkeypatch.setattr("models.multiclass_inference.MulticlassInference", _CountingFake)

    segmentation_backends._BACKEND_CACHE.clear()
    backend = get_backend("multiclass")
    assert construct_calls == []

    mask_store = _RecordingMaskStore()
    backend.predict_into(mask_store, ["frame.png"])
    assert len(construct_calls) == 1

    backend.predict_into(mask_store, ["frame.png"])
    assert len(construct_calls) == 1


def test_get_backend_multiclass_reachable_only_via_env_var(monkeypatch):
    """DEFAULT_BACKEND must stay "binary" with no env var set, and the
    multiclass backend must only become reachable by explicitly setting
    DLHOOK_SEG_BACKEND=multiclass."""
    segmentation_backends._BACKEND_CACHE.clear()
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert resolve_backend_name() == "binary"
    assert isinstance(get_backend(), BinaryBackend)

    segmentation_backends._BACKEND_CACHE.clear()
    monkeypatch.setenv(ENV_VAR, "multiclass")
    assert resolve_backend_name() == "multiclass"
    assert isinstance(get_backend(), MulticlassBackend)


# --- moved symbols: importable from both old and new locations -------------

def test_moved_symbols_are_the_same_object_at_old_and_new_locations():
    import models.multiclass_inference as models_multiclass_inference
    import models.normalization as models_normalization
    import models.unet as models_unet
    import multi.src.loss_functions as multi_loss_functions
    import multi.src.model as multi_model
    import multi.src.multiclass_inference as multi_multiclass_inference
    import multi.src.normalization as multi_normalization

    assert models_normalization.zscore_normalize is multi_normalization.zscore_normalize
    assert models_multiclass_inference.zscore_normalize is models_normalization.zscore_normalize
    assert multi_multiclass_inference.zscore_normalize is models_normalization.zscore_normalize

    assert models_multiclass_inference.MulticlassInference is (
        multi_multiclass_inference.MulticlassInference
    )

    assert models_unet.head_from_state_dict is multi_model.head_from_state_dict
    assert models_unet.align_output_to_target is multi_loss_functions.align_output_to_target
