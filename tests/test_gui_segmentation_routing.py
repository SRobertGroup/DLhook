"""Recorder + no-drift tests for Gui._segment_paths (Step 3 of the
multiclass-segmentation rollout, see CLAUDE.md).

Before this step, Gui.run_apical_pipeline and Gui.segment_single_seedling
each built their own `predictors = [(label, get_predictor(weight_path)), ...]`
list and looped over it. Both now delegate to the single
`Gui._segment_paths` helper, which calls
`get_backend(resolve_backend_name()).predict_into(self.mask_store, paths)`
(models/segmentation_backends.py). These tests exist to prove that routing
change didn't alter the observable segmentation call sequence at either call
site.

House style: no real Tk, no real weights/torch models. Gui is built with
`Gui.__new__(Gui)` (bypassing `__init__`, which builds the whole Tk window)
and given only the attributes each method under test actually touches --
the narrowest seam that still exercises the real, unmodified method bodies
(including the real _segment_paths -> get_backend -> BinaryBackend chain),
rather than a full end-to-end batch/preview run. Both tests stop the method
under test right after the segmentation call they're pinning, via a
deliberate sentinel exception raised from the next thing each method calls
(_ensure_germination_detected / the return statement) -- everything past
segmentation (angle reconstruction, CSV export, Tk widgets) is unrelated to
this step and none of it is driven here.
"""
from __future__ import annotations

import inspect
import os

import pytest

from models import segmentation_backends
from seedling_measurment import Gui

# The exact (weight_path, label) sequence pinned by
# tests/test_segmentation_backends.py::test_binary_backend_predict_into_pins_todays_call_sequence.
# Derived from BinaryBackend.WEIGHTS itself (the single source of truth for
# "which models run") rather than re-hardcoded here a third time -- this
# test's job is to pin that the two GUI call sites *reach* that sequence via
# _segment_paths, not to re-pin what the sequence is.
_EXPECTED_WEIGHT_PATHS = [w for _, w in segmentation_backends.BinaryBackend.WEIGHTS]
_EXPECTED_LABELS = [label for label, _ in segmentation_backends.BinaryBackend.WEIGHTS]


class _StopAfterSegmentation(Exception):
    """Raised deliberately once the recorder has captured what it needs, to
    short-circuit the rest of the method under test (which needs a lot more
    Gui state -- germination_detector, mask_store.dump, Tk's debug_var,
    save_data -- that has nothing to do with this step)."""


class _RecordingPredictor:
    """Stands in for a real UNetInference predictor (mirrors
    tests/test_segmentation_backends.py's _RecordingPredictor)."""

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

    def discard_raw(self, *args, **kwargs):
        pass


class _FakeReporter:
    def report(self, **kwargs):
        pass


@pytest.fixture(autouse=True)
def _clean_backend_state(monkeypatch):
    # Default backend ("binary"), regardless of the running shell's env.
    monkeypatch.delenv(segmentation_backends.ENV_VAR, raising=False)
    segmentation_backends._BACKEND_CACHE.clear()
    yield
    segmentation_backends._BACKEND_CACHE.clear()


def _patch_get_predictor(monkeypatch):
    get_predictor_calls = []
    predict_files_calls = []

    def fake_get_predictor(weight_path, **kwargs):
        get_predictor_calls.append(weight_path)
        return _RecordingPredictor(weight_path, predict_files_calls)

    monkeypatch.setattr(segmentation_backends, "get_predictor", fake_get_predictor)
    return get_predictor_calls, predict_files_calls


def test_run_apical_pipeline_segments_via_segment_paths(monkeypatch):
    get_predictor_calls, predict_files_calls = _patch_get_predictor(monkeypatch)

    fake = Gui.__new__(Gui)
    fake.transformed_mid_points = [0]  # one seedling
    fake.file_list = ["frame1.png"]  # one frame -> one chunk
    fake.reset_frame_accumulators = lambda crop_id=None: None
    fake.progress_reporter = _FakeReporter()
    fake.mask_store = _RecordingMaskStore()
    fake.dump_masks = False
    # None short-circuits the per-frame body via "if result is None: continue"
    # -- process_single_frame's own behaviour is out of scope for this test.
    fake.process_single_frame = lambda file_name, crop_id: None

    def _stop(crop_id):
        raise _StopAfterSegmentation()

    fake._ensure_germination_detected = _stop

    with pytest.raises(_StopAfterSegmentation):
        Gui.run_apical_pipeline(fake)

    assert get_predictor_calls == _EXPECTED_WEIGHT_PATHS
    assert [call[2] for call in predict_files_calls] == _EXPECTED_LABELS

    expected_paths = (os.path.join("data/images", "0-crop-frame1.png"),)
    assert all(call[1] == expected_paths for call in predict_files_calls)

    assert [call[1] for call in fake.mask_store.put_raw_bulk_calls] == _EXPECTED_LABELS


def test_segment_single_seedling_segments_via_segment_paths(monkeypatch):
    get_predictor_calls, predict_files_calls = _patch_get_predictor(monkeypatch)

    fake = Gui.__new__(Gui)
    fake.progress_reporter = _FakeReporter()
    fake.reset_frame_accumulators = lambda crop_id=None: None
    fake.mask_store = _RecordingMaskStore()
    cropped = ["3-crop-frameA.png", "3-crop-frameB.png"]
    fake.crop_single_seedling = lambda crop_id: cropped

    result = Gui.segment_single_seedling(fake, 3)

    assert result == cropped
    assert get_predictor_calls == _EXPECTED_WEIGHT_PATHS
    assert [call[2] for call in predict_files_calls] == _EXPECTED_LABELS

    expected_paths = tuple(os.path.join("data/images", f) for f in cropped)
    assert all(call[1] == expected_paths for call in predict_files_calls)

    assert [call[1] for call in fake.mask_store.put_raw_bulk_calls] == _EXPECTED_LABELS


# --- no-drift: the GUI file must no longer name a model or weight itself ---

def test_gui_segmentation_methods_have_no_get_predictor_or_pkl_literal():
    for method in (Gui.run_apical_pipeline, Gui.segment_single_seedling):
        source = inspect.getsource(method)
        assert "get_predictor" not in source, method.__name__
        assert ".pkl" not in source, method.__name__


def test_segment_paths_helper_delegates_to_backend():
    source = inspect.getsource(Gui._segment_paths)
    assert "get_backend" in source
    assert "resolve_backend_name" in source
    assert "get_predictor" not in source
    assert ".pkl" not in source
