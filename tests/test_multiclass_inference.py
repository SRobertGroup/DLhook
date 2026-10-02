"""Regression tests for the multiclass train/inference preprocessing mismatch:
training (multi/src/data_loader.py) used to z-score its patches while
inference (multi/src/multiclass_inference.py) divided by 255 -- two
completely different input distributions fed to the same trained weights.
Also covers the companion fix: MulticlassInference is now runnable at the
model's actual training tile geometry (264/252/6) instead of only the live
GUI's 572/560/6, with the architecture's +4px output remainder cropped off
before stitching.

House style: synthesise everything in-test, plain asserts, no fixtures
beyond tmp_path, no GPU required, fast."""
import csv

import numpy as np
import torch
from PIL import Image

from models.UNetInference import IN_SIZE, MARGIN, OUT_SIZE
from models.unet import UNetGNRes
from multi.src import data_loader, multiclass_inference, normalization
from multi.src.data_loader import PatchDataset
from multi.src.multiclass_inference import MulticlassInference
from multi.src.normalization import zscore_normalize

CSV_FIELDNAMES = ["filename", "x", "y", "patch_size", "foreground_fraction"]


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _make_dataset(root, patch_size, seed=0):
    """A one-row PatchDataset backed by a single synthetic RGB image sized
    exactly `patch_size` x `patch_size` -- no `pad_to_min` padding kicks in,
    so `PatchDataset._load_patch`'s only geometry operation is the
    MARGIN reflect-pad, keeping the comparison against the inference path
    (below) exact."""
    raw_dir = root / "raw"
    masks_dir = root / "masks"
    raw_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    rng = np.random.RandomState(seed)
    raw = rng.randint(0, 256, size=(patch_size, patch_size, 3), dtype=np.uint8)
    mask = np.zeros((patch_size, patch_size), dtype=np.uint8)

    Image.fromarray(raw).save(raw_dir / "img0.png")
    Image.fromarray(mask).save(masks_dir / "img0.png")

    csv_path = root / "patches.csv"
    _write_csv(csv_path, [{
        "filename": "img0.png", "x": 0, "y": 0, "patch_size": patch_size,
        "foreground_fraction": 0.0,
    }])

    return PatchDataset(csv_path, raw_dir, masks_dir, augment=False)


class _RecordingModel(torch.nn.Module):
    """Stands in for UNetGNRes to capture exactly the tensor
    MulticlassInference._run_batch feeds the network, without needing a real
    (or even randomly initialised) checkpoint. Returns zeros already shaped
    like the real network's post-crop output so align_output_to_target is a
    no-op and the rest of _run_batch runs unmodified."""

    def __init__(self, num_classes, out_size):
        super().__init__()
        self.num_classes = num_classes
        self.out_size = out_size
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        return torch.zeros(x.shape[0], self.num_classes, self.out_size, self.out_size)


def _make_inference_with_recording_model(in_size, out_size, margin, num_classes=4):
    """MulticlassInference instance built via __new__ (bypassing __init__'s
    checkpoint load -- same isolation trick test_multi_data_loader.py uses
    for PatchDataset._augment) with a _RecordingModel standing in for the
    real network, so _run_batch's normalisation can be tested without a
    trained checkpoint."""
    mi = MulticlassInference.__new__(MulticlassInference)
    mi.device = torch.device("cpu")
    mi.num_classes = num_classes
    mi.in_size = in_size
    mi.out_size = out_size
    mi.margin = margin
    mi._out_shape_ref = torch.empty(out_size, out_size)
    mi.model = _RecordingModel(num_classes, out_size)
    mi.batch_size = 4
    return mi


def test_data_loader_and_multiclass_inference_share_the_same_normalization_function():
    """Guards against the two paths quietly re-diverging: both modules must
    import the identical function object from normalization.py, not two
    separately (mis)implemented copies of "z-score"."""
    assert data_loader.zscore_normalize is normalization.zscore_normalize
    assert multiclass_inference.zscore_normalize is normalization.zscore_normalize


def test_training_and_inference_paths_normalize_identically(tmp_path):
    """The regression test that was missing. Before this fix, training fed
    the model mean 0 / std 1 (z-scored) patches while inference fed mean
    ~0.6 / std ~0.06 (/255) patches -- entirely different input
    distributions for the same trained weights. This exercises the REAL
    production normalisation code on each side (PatchDataset.__getitem__ for
    training; MulticlassInference._run_batch for inference) on the exact
    same source pixels, and asserts the resulting network-input tensors are
    numerically identical.
    """
    patch_size = 20
    dataset = _make_dataset(tmp_path, patch_size, seed=3)

    # Training side: PatchDataset's real per-item normalisation.
    training_input, _ = dataset[0]
    training_input = training_input.numpy()

    # The raw (pre-normalisation) patch training fed the model -- already
    # includes the +2*MARGIN context, i.e. it is exactly the shape and
    # content of one inference tile.
    raw_patch, _ = dataset._load_patch(dataset.rows[0])
    tile_size = raw_patch.shape[0]
    assert tile_size == patch_size + 2 * MARGIN

    # Inference side: feed that SAME array of pixels (cast to uint8, as a
    # real cv2-loaded tile would be -- the PNG round-trip above is lossless,
    # so the values are identical) through the real _run_batch, standing in
    # only the network forward pass since no checkpoint is needed to test
    # preprocessing.
    mi = _make_inference_with_recording_model(
        in_size=tile_size, out_size=patch_size, margin=MARGIN
    )
    mi._run_batch([raw_patch.astype(np.uint8)])
    inference_input = mi.model.last_input[0].numpy()

    assert training_input.shape == inference_input.shape
    np.testing.assert_allclose(training_input, inference_input, atol=1e-5)

    # And a sanity check on what the old, broken inference normalisation
    # would have produced -- it must NOT match, or this test would not
    # actually be exercising the fix.
    broken_input = (raw_patch.astype(np.float32) / 255.0).transpose(2, 0, 1)
    assert not np.allclose(training_input, broken_input, atol=1e-2)


def test_default_constructor_geometry_still_matches_old_572_560_6(tmp_path):
    """MulticlassInference's in_size/out_size/margin constructor params must
    default to the pre-existing UNetInference constants, so every caller
    that does not opt into the training geometry keeps the old behaviour."""
    assert IN_SIZE == 572 and OUT_SIZE == 560 and MARGIN == 6

    model = UNetGNRes(n_classes=4)
    checkpoint_path = tmp_path / "random.pt"
    torch.save(model.state_dict(), checkpoint_path)

    mi = MulticlassInference(str(checkpoint_path), device=torch.device("cpu"))
    assert (mi.in_size, mi.out_size, mi.margin) == (IN_SIZE, OUT_SIZE, MARGIN)

    # And it still actually runs end-to-end at that geometry.
    image = np.zeros((60, 98, 3), dtype=np.uint8)
    labels = mi.segment_many_argmax([image])
    assert labels[0].shape == (60, 98)


def test_stitched_output_matches_input_crop_size_at_training_geometry(tmp_path):
    """Shape regression test for the training-geometry fix (264/252/6):
    the stitched output must equal the input crop's H x W for crops smaller
    than one tile and larger than one tile in each dimension -- the real
    crops range 60-232px wide, 98-1038px tall (cropped_training_set/
    manifest.csv). This exercises the actual +4px output-remainder crop
    (align_output_to_target) needed at this geometry but not at the default
    572/560/6 (see MulticlassInference.__init__'s comment)."""
    training_patch_size = 252
    in_size, out_size, margin = training_patch_size + 2 * MARGIN, training_patch_size, MARGIN

    model = UNetGNRes(n_classes=4)
    checkpoint_path = tmp_path / "random.pt"
    torch.save(model.state_dict(), checkpoint_path)

    mi = MulticlassInference(
        str(checkpoint_path), in_size=in_size, out_size=out_size, margin=margin,
        device=torch.device("cpu"),
    )
    assert (mi.in_size, mi.out_size, mi.margin) == (264, 252, 6)

    crop_sizes = [
        (60, 98),      # smaller than one tile in both dims
        (232, 1038),   # smaller in width, much larger in height
        (300, 300),    # larger than one tile in both dims
    ]
    for h, w in crop_sizes:
        image = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
        labels = mi.segment_many_argmax([image])
        assert labels[0].shape == (h, w), (h, w, labels[0].shape)

        probs = mi.segment_many_argmax([image], return_probs=True)
        assert probs[0].shape == (4, h, w), (h, w, probs[0].shape)
        # A valid stitched softmax: every pixel's 4 class probabilities
        # sum to ~1.
        np.testing.assert_allclose(probs[0].sum(axis=0), 1.0, atol=1e-4)


def test_checkpoints_of_either_head_round_trip_through_the_same_load_path(tmp_path):
    """The plumbing constraint behind the plain-head experiment: the two
    conv_out heads have different state_dict keys (the groupnorm head adds
    conv_out.2.{weight,bias}), so a fixed `UNetGNRes(n_classes=...)` would
    raise on one of them. MulticlassInference must infer the head from the
    checkpoint's own keys, so evaluate_multiclass.py scores either checkpoint
    with the identical command line."""
    for head, expected_conv_out_len in (("groupnorm", 3), ("plain", 1)):
        checkpoint_path = tmp_path / f"{head}.pt"
        torch.save(UNetGNRes(n_classes=4, head=head).state_dict(), checkpoint_path)

        mi = MulticlassInference(str(checkpoint_path), device=torch.device("cpu"))
        assert mi.head == head
        assert len(mi.model.conv_out) == expected_conv_out_len

        labels = mi.segment_many_argmax([np.zeros((60, 98, 3), dtype=np.uint8)])
        assert labels[0].shape == (60, 98)


def test_plain_head_checkpoint_weights_actually_load_not_just_construct(tmp_path):
    """Guards against the head being inferred correctly but the weights being
    dropped: the loaded conv_out kernel must equal the saved one."""
    source = UNetGNRes(n_classes=4, head="plain")
    checkpoint_path = tmp_path / "plain.pt"
    torch.save(source.state_dict(), checkpoint_path)

    mi = MulticlassInference(str(checkpoint_path), device=torch.device("cpu"))
    torch.testing.assert_close(mi.model.conv_out[0].weight, source.conv_out[0].weight)


def test_segment_many_argmax_return_probs_matches_its_own_argmax():
    """return_probs=True must expose the exact same per-pixel softmax the
    default argmax readout is computed from -- not a separately recomputed
    or differently-shaped stack."""
    mi = _make_inference_with_recording_model(in_size=32, out_size=20, margin=MARGIN)

    # Force a fixed, non-uniform softmax to make the argmax unambiguous:
    # class channel 2 wins at every pixel. `_plan_tiles` always emits at
    # least a 2x2 (4-tile) grid here (in_size=32 > out_size=20, same as the
    # real geometries -- see multiclass_inference.py's `_run_batch`
    # docstring), so the fake forward must honour the actual batch size
    # rather than a fixed batch of 1.
    def _fake_forward(x):
        out = torch.zeros(x.shape[0], 4, 20, 20)
        out[:, 2] = 10.0
        return out

    mi.model.forward = _fake_forward

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    labels = mi.segment_many_argmax([image])
    probs = mi.segment_many_argmax([image], return_probs=True)

    assert labels[0].shape == (20, 20)
    assert probs[0].shape == (4, 20, 20)
    np.testing.assert_array_equal(labels[0], np.argmax(probs[0], axis=0).astype(np.uint8))
    assert np.all(labels[0] == 2)


def test_predict_files_labelmaps_skips_unreadable_paths_like_predict_files(tmp_path, capsys):
    """MulticlassBackend (models/segmentation_backends.py) routes an
    unreadable image through predict_files_labelmaps exactly the same way
    BinaryBackend routes it through UNetInference.predict_files -- same
    `[WARNING] Could not load: ...` line, same "just absent from the
    returned dict" outcome -- so the two backends cannot diverge on how a
    missing/corrupt frame is handled."""
    mi = _make_inference_with_recording_model(in_size=32, out_size=20, margin=MARGIN)

    def _fake_forward(x):
        return torch.zeros(x.shape[0], 4, 20, 20)

    mi.model.forward = _fake_forward

    good_path = tmp_path / "0-crop-good.png"
    Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(good_path)
    bad_path = tmp_path / "0-crop-missing.png"  # never written -- cv2.imread returns None

    label_maps = mi.predict_files_labelmaps([str(good_path), str(bad_path)])

    assert set(label_maps.keys()) == {"0-crop-good.png"}
    assert label_maps["0-crop-good.png"].shape == (20, 20)

    captured = capsys.readouterr()
    assert f"[WARNING] Could not load: {bad_path}" in captured.out
