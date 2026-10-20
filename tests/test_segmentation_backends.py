"""Tests for models/segmentation_backends.py.

This module is additive-only (see its docstring): nothing in
seedling_measurement.py imports it yet, so these tests are the only thing
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

def test_resolve_backend_name_defaults_to_multiclass_when_checkpoint_exists(monkeypatch, tmp_path):
    ckpt = tmp_path / "dlhook_4class_v1.pt"
    ckpt.write_bytes(b"x")
    monkeypatch.setattr(segmentation_backends, "DEFAULT_MULTICLASS_CHECKPOINT", str(ckpt))
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert DEFAULT_BACKEND == "multiclass"
    assert resolve_backend_name() == "multiclass"


def test_resolve_backend_name_default_falls_back_to_binary_with_warning(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(segmentation_backends, "DEFAULT_MULTICLASS_CHECKPOINT", str(tmp_path / "missing.pt"))
    monkeypatch.setattr(segmentation_backends, "PREVIOUS_MULTICLASS_CHECKPOINT", str(tmp_path / "missing_v1.pt"))
    monkeypatch.setattr(segmentation_backends, "_WARNED", set())
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert resolve_backend_name() == "binary"
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_default_multiclass_checkpoint_falls_back_to_v1(monkeypatch, tmp_path, caplog):
    v1 = tmp_path / "dlhook_4class_v1.pt"
    v1.write_bytes(b"x")
    monkeypatch.setattr(segmentation_backends, "DEFAULT_MULTICLASS_CHECKPOINT", str(tmp_path / "missing_v2.pt"))
    monkeypatch.setattr(segmentation_backends, "PREVIOUS_MULTICLASS_CHECKPOINT", str(v1))
    monkeypatch.setattr(segmentation_backends, "_WARNED", set())
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert segmentation_backends.default_multiclass_checkpoint() == str(v1)
    assert resolve_backend_name() == "multiclass"
    assert MulticlassBackend().checkpoint == str(v1)
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_default_multiclass_checkpoint_is_v2_and_germination_keeps_v1():
    assert segmentation_backends.DEFAULT_MULTICLASS_CHECKPOINT.endswith("dlhook_4class_v2.pt")
    assert segmentation_backends.PREVIOUS_MULTICLASS_CHECKPOINT.endswith("dlhook_4class_v1.pt")
    import json
    from utils.paths import repo_path
    with open(repo_path(segmentation_backends.DEFAULT_GERMINATION_WEIGHTS), encoding="utf-8") as fh:
        assert json.load(fh)["checkpoint"] == segmentation_backends.PREVIOUS_MULTICLASS_CHECKPOINT


def test_resolve_backend_name_explicit_multiclass_not_downgraded(monkeypatch, tmp_path):
    monkeypatch.setattr(segmentation_backends, "DEFAULT_MULTICLASS_CHECKPOINT", str(tmp_path / "missing.pt"))
    monkeypatch.setenv(ENV_VAR, "multiclass")
    assert resolve_backend_name() == "multiclass"


def test_resolve_backend_name_explicit_binary(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "binary")
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
    backend = MulticlassBackend(radicle_source="multiclass", hypocotyl_binary=False)
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
    backend = get_backend("multiclass", hypocotyl_binary=False)
    assert construct_calls == []

    mask_store = _RecordingMaskStore()
    backend.predict_into(mask_store, ["frame.png"])
    assert len(construct_calls) == 1

    backend.predict_into(mask_store, ["frame.png"])
    assert len(construct_calls) == 1


def test_get_backend_default_and_env_selection(monkeypatch, tmp_path):
    """No env var -> multiclass (checkpoint present); DLHOOK_SEG_BACKEND=binary -> legacy."""
    ckpt = tmp_path / "dlhook_4class_v1.pt"
    ckpt.write_bytes(b"x")
    monkeypatch.setattr(segmentation_backends, "DEFAULT_MULTICLASS_CHECKPOINT", str(ckpt))
    segmentation_backends._BACKEND_CACHE.clear()
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert isinstance(get_backend(), MulticlassBackend)

    segmentation_backends._BACKEND_CACHE.clear()
    monkeypatch.setenv(ENV_VAR, "binary")
    assert isinstance(get_backend(), BinaryBackend)


def test_get_backend_cache_key_includes_new_options():
    segmentation_backends._BACKEND_CACHE.clear()
    a = get_backend("multiclass")
    b = get_backend("multiclass", radicle_source="multiclass")
    c = get_backend("multiclass", hypocotyl_binary=True)
    assert a is get_backend("multiclass")
    assert len({id(a), id(b), id(c)}) == 3


def test_multiclass_hypocotyl_binary_on_writes_1_2_hypocotyl_binary_and_no_4(monkeypatch):
    monkeypatch.setattr("models.multiclass_inference.MulticlassInference", _FakeMulticlassInference)
    label_map = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.uint8)
    _FakeMulticlassInference.label_maps_to_return = {"frame.png": label_map}

    get_predictor_calls, predict_files_calls = [], []

    def fake_get_predictor(weight_path, **kwargs):
        get_predictor_calls.append(weight_path)
        return _RecordingPredictor(weight_path, predict_files_calls)

    monkeypatch.setattr(segmentation_backends, "get_predictor", fake_get_predictor)

    mask_store = _RecordingMaskStore()
    MulticlassBackend(hypocotyl_binary=True).predict_into(mask_store, ["frame.png"])

    hypocot = dict(BinaryBackend.WEIGHTS)["2"]
    assert [call[1] for call in mask_store.put_raw_bulk_calls] == ["1", "2", "hypocotyl_binary"]
    assert get_predictor_calls == [hypocot]
    assert predict_files_calls == [(hypocot, ("frame.png",), "2")]
    assert mask_store.put_raw_bulk_calls[2][0] == {"frame.png": "mask-for-2"}


def test_multiclass_default_skips_binary_model(monkeypatch):
    monkeypatch.setattr("models.multiclass_inference.MulticlassInference", _FakeMulticlassInference)
    _FakeMulticlassInference.label_maps_to_return = {"frame.png": np.zeros((2, 2), dtype=np.uint8)}

    def boom(*a, **k):
        raise AssertionError("get_predictor must not be called")

    monkeypatch.setattr(segmentation_backends, "get_predictor", boom)
    mask_store = _RecordingMaskStore()
    MulticlassBackend().predict_into(mask_store, ["frame.png"])
    assert [call[1] for call in mask_store.put_raw_bulk_calls] == ["1", "2"]


def test_multiclass_rejects_unknown_radicle_source():
    with pytest.raises(ValueError):
        MulticlassBackend(radicle_source="germ_v9")


# --- angle source ------------------------------------------------------------

def _make_ckpts(tmp_path, n=3):
    paths = []
    for i in range(n):
        p = tmp_path / f"m{i}.pt"
        p.write_bytes(b"x")
        paths.append(str(p))
    return "+".join(paths)


def test_angle_source_constants():
    assert segmentation_backends.ANGLE_SOURCE_ENV == "DLHOOK_ANGLE_SOURCE"
    assert segmentation_backends.DEFAULT_ANGLE_SOURCE == "landmarks"
    assert segmentation_backends.VALID_ANGLE_SOURCES == ("landmarks", "ellipse")
    assert segmentation_backends.DEFAULT_LANDMARK_CHECKPOINTS.split("+") == [
        f"weights/landmarks/dlhook_landmarks_m_{c}.pt" for c in "abc"
    ]
    assert segmentation_backends.PREVIOUS_LANDMARK_CHECKPOINTS.split("+") == [
        f"weights/landmarks/dlhook_landmarks_{c}.pt" for c in "abc"
    ]


def test_default_checkpoints_fall_back_to_the_previous_members(monkeypatch):
    sb = segmentation_backends
    have = set()
    monkeypatch.setattr(sb, "_all_exist", lambda spec: spec in have)
    monkeypatch.setattr(sb, "_warn_once", lambda *a: None)
    assert sb.default_landmark_checkpoints() == sb.DEFAULT_LANDMARK_CHECKPOINTS      # nothing: ellipse later
    have.add(sb.PREVIOUS_LANDMARK_CHECKPOINTS)
    assert sb.default_landmark_checkpoints() == sb.PREVIOUS_LANDMARK_CHECKPOINTS     # only the old members
    have.add(sb.DEFAULT_LANDMARK_CHECKPOINTS)
    assert sb.default_landmark_checkpoints() == sb.DEFAULT_LANDMARK_CHECKPOINTS      # both: the new ones


def test_resolve_angle_source_default_is_landmarks(monkeypatch, tmp_path):
    monkeypatch.delenv("DLHOOK_ANGLE_SOURCE", raising=False)
    assert segmentation_backends.resolve_angle_source(_make_ckpts(tmp_path)) == "landmarks"


def test_resolve_angle_source_env_ellipse(monkeypatch, tmp_path):
    monkeypatch.setenv("DLHOOK_ANGLE_SOURCE", "ellipse")
    assert segmentation_backends.resolve_angle_source(_make_ckpts(tmp_path)) == "ellipse"


def test_resolve_angle_source_invalid_raises(monkeypatch):
    monkeypatch.setenv("DLHOOK_ANGLE_SOURCE", "magic")
    with pytest.raises(ValueError):
        segmentation_backends.resolve_angle_source()


def test_resolve_angle_source_missing_checkpoint_falls_back(monkeypatch, tmp_path, caplog):
    monkeypatch.delenv("DLHOOK_ANGLE_SOURCE", raising=False)
    monkeypatch.setattr(segmentation_backends, "_WARNED", set())
    spec = _make_ckpts(tmp_path, 2) + "+" + str(tmp_path / "gone.pt")
    assert segmentation_backends.resolve_angle_source(spec) == "ellipse"
    assert any(r.levelname == "WARNING" for r in caplog.records)


# --- LandmarkRunner ----------------------------------------------------------

def test_landmark_runner_is_lazy_and_pins_training_geometry(monkeypatch):
    calls = []

    class _FakeEnsemble:
        def predict_files_landmarks(self, paths):
            return {"a.png": {"theta": 1.0}}

    def fake_from_checkpoints(spec, **kwargs):
        calls.append((spec, kwargs))
        return _FakeEnsemble()

    monkeypatch.setattr("models.multiclass_inference.LandmarkEnsemble.from_checkpoints", fake_from_checkpoints)

    runner = segmentation_backends.LandmarkRunner("a.pt+b.pt")
    assert calls == []                                   # lazy

    assert runner.predict_paths(["a.png"]) == {"a.png": {"theta": 1.0}}
    runner.predict_paths(["a.png"])
    assert len(calls) == 1                               # loaded once
    spec, kwargs = calls[0]
    assert spec == "a.pt+b.pt"
    assert kwargs == dict(num_classes=4, in_size=252 + 2 * MARGIN, out_size=252, margin=MARGIN)


def test_get_landmark_runner_caches():
    segmentation_backends._LANDMARK_CACHE.clear()
    a = segmentation_backends.get_landmark_runner("x.pt")
    assert a is segmentation_backends.get_landmark_runner("x.pt")
    assert a is not segmentation_backends.get_landmark_runner("y.pt")
    segmentation_backends._LANDMARK_CACHE.clear()


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
