"""Regression tests for the PatchDataset image/label geometry bug: UNetGNRes
shrinks its output by 2*MARGIN relative to its input (12px total, confirmed
empirically against several models/unet.py:get_valid_patch_sizes() values),
so PatchDataset must hand back an image patch that is `patch_size + 2*MARGIN`
and a label patch that is exactly `patch_size` -- not the same size, which is
what F.nll_loss (multi/src/loss_functions.py) crashed on."""
import csv
import random

import numpy as np
import torch
from PIL import Image

from models.UNetInference import MARGIN
from models.unet import UNetGNRes, get_valid_patch_sizes
from multi.src.data_loader import PatchDataset
from multi.src.loss_functions import align_output_to_target, build_loss
from multi.src.patch_index import IGNORE_VALUE

CSV_FIELDNAMES = ["filename", "x", "y", "patch_size", "foreground_fraction"]


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _make_dataset(root, size, image_size=None, x=0, y=0, seed=0):
    """A one-row PatchDataset backed by a single synthetic RGB image/mask
    pair, both `image_size` x `image_size` (defaults to `size`, i.e. no
    pad_to_min padding is exercised)."""
    image_size = image_size or size
    raw_dir = root / "raw"
    masks_dir = root / "masks"
    raw_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    rng = np.random.RandomState(seed)
    raw = (rng.rand(image_size, image_size, 3) * 255).astype(np.uint8)
    mask = np.zeros((image_size, image_size), dtype=np.uint8)

    Image.fromarray(raw).save(raw_dir / "img0.png")
    Image.fromarray(mask).save(masks_dir / "img0.png")

    csv_path = root / "patches.csv"
    _write_csv(csv_path, [{
        "filename": "img0.png", "x": x, "y": y, "patch_size": size,
        "foreground_fraction": 0.0,
    }])

    return PatchDataset(csv_path, raw_dir, masks_dir, augment=False)


def test_dataset_image_patch_is_label_patch_plus_two_margins(tmp_path):
    """Contract check independent of a forward pass: the image PatchDataset
    returns must be exactly patch_size + 2*MARGIN, the label exactly
    patch_size, for several valid patch sizes."""
    for size in (92, 108, 252):
        dataset = _make_dataset(tmp_path / f"contract_{size}", size)
        image, label = dataset[0]
        assert label.shape[-2:] == (size, size)
        assert image.shape[-2:] == (size + 2 * MARGIN, size + 2 * MARGIN)


def test_model_output_matches_label_shape_for_several_patch_sizes(tmp_path):
    """The exact bug this suite must catch: whatever PatchDataset returns
    must let a real UNetGNRes forward pass feed loss_fn without the 'batch
    or spatial sizes don't match' RuntimeError (multi/src/loss_functions.py:
    38, hit from multi/train_unet_multiclass.py:65-66). A genuine CPU
    forward pass through the REAL production path -- model, then
    align_output_to_target (as run_epoch does), then the actual loss_fn --
    not just the dataset's own contract.

    UNetGNRes's output is always a multiple of 16 (four MaxPool2d(2) stages),
    so on the patch_size + 2*MARGIN input PatchDataset feeds it, the model's
    raw output is NOT exactly patch_size for any get_valid_patch_sizes()
    value (they are all ==12 mod 16, and adding 2*MARGIN==12 shifts the
    residue): it comes back exactly patch_size + 4, a constant, deterministic
    remainder of the architecture's granularity. This is exactly why
    train_unet_multiclass.py aligns the output before computing the loss --
    this test proves that alignment step actually closes the gap, for
    several distinct patch sizes, via a real forward pass and a real loss
    computation (not a shape assertion in isolation).
    """
    model = UNetGNRes(n_classes=4)
    model.eval()
    loss_fn = build_loss({"type": "cross_entropy", "ignore_index": IGNORE_VALUE})

    valid_sizes = get_valid_patch_sizes()
    # Smallest two valid sizes -- keeps the forward pass fast while still
    # proving the remainder is constant (+4) rather than a function of size.
    for size in sorted(valid_sizes)[:2]:
        dataset = _make_dataset(tmp_path / f"forward_{size}", size)
        image, label = dataset[0]
        with torch.no_grad():
            output = model(image.unsqueeze(0))

        # Document the exact subtlety this test exists to catch: the raw
        # model output does NOT already match the label -- it is larger by
        # the architecture's fixed +4px remainder, for every patch_size in
        # this family.
        assert output.shape[-2:] == (size + 4, size + 4), (
            f"patch_size={size}: expected the architecture's constant +4px "
            f"remainder, got model output {tuple(output.shape[-2:])} vs "
            f"label {tuple(label.shape)}"
        )

        aligned = align_output_to_target(output, label.unsqueeze(0))
        assert aligned.shape[-2:] == label.shape[-2:]

        # The real production call, exactly as multi/train_unet_multiclass.py's
        # run_epoch makes it -- must not raise.
        loss = loss_fn(aligned, label.unsqueeze(0))
        assert torch.isfinite(loss)


def test_label_patch_aligns_with_image_patch_no_offset_shift(tmp_path):
    """A marker placed at one known pixel of the source image must land at
    the same relative offset (shifted by exactly MARGIN) in the returned
    image patch, and at the un-shifted offset in the returned label patch --
    catches an off-by-one/off-by-margin error that would shift the label
    relative to the image it is supposed to describe."""
    image_size = 60
    patch_size = 20
    tile_x, tile_y = 15, 10
    marker_row_in_patch, marker_col_in_patch = 7, 3
    marker_value = 200

    marker_abs_row = tile_y + marker_row_in_patch
    marker_abs_col = tile_x + marker_col_in_patch

    raw = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    raw[marker_abs_row, marker_abs_col] = marker_value
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    mask[marker_abs_row, marker_abs_col] = 3  # radicle class, an arbitrary distinct value

    raw_dir = tmp_path / "raw"
    masks_dir = tmp_path / "masks"
    raw_dir.mkdir()
    masks_dir.mkdir()
    Image.fromarray(raw).save(raw_dir / "img0.png")
    Image.fromarray(mask).save(masks_dir / "img0.png")

    csv_path = tmp_path / "patches.csv"
    _write_csv(csv_path, [{
        "filename": "img0.png", "x": tile_x, "y": tile_y,
        "patch_size": patch_size, "foreground_fraction": 0.01,
    }])

    dataset = PatchDataset(csv_path, raw_dir, masks_dir, augment=False)
    raw_patch, mask_patch = dataset._load_patch(dataset.rows[0])

    # Label patch: same coordinate frame as before -- unshifted.
    assert mask_patch[marker_row_in_patch, marker_col_in_patch] == 3
    assert np.count_nonzero(mask_patch == 3) == 1

    # Image patch: shifted by exactly MARGIN in both axes because it carries
    # MARGIN pixels of extra context on every side of the labelled region.
    assert raw_patch[marker_row_in_patch + MARGIN, marker_col_in_patch + MARGIN, 0] == marker_value
    assert np.count_nonzero(raw_patch[..., 0] == marker_value) == 1


def test_rotation_augmentation_never_interpolates_ignore_into_a_class(tmp_path):
    """The label patch must be rotated with nearest-neighbour interpolation:
    IGNORE_VALUE (255) pixels can never blend with a real class index (or
    vice versa) the way a linear/cubic interpolation would. Constructed
    directly against _augment (bypassing __init__'s CSV read) since this is
    purely a geometry/interpolation property of the augmentation step."""
    size = 40
    mask_patch = np.zeros((size, size), dtype=np.uint8)
    mask_patch[:, size // 2:] = 3       # right half is class 3 (radicle)
    mask_patch[:5, :] = IGNORE_VALUE    # a strip of unlabelled/no-consensus pixels
    raw_patch = np.zeros((size + 2 * MARGIN, size + 2 * MARGIN, 3), dtype=np.float32)

    dataset = PatchDataset.__new__(PatchDataset)
    dataset.horizontal_flip = False
    dataset.rotation_degrees = 30.0
    dataset.brightness_jitter = 0.0
    dataset.contrast_jitter = 0.0
    dataset.rng = random.Random(0)

    input_values = set(np.unique(mask_patch).tolist())
    _, augmented_mask = dataset._augment(raw_patch, mask_patch)

    # warpAffine's border fill also uses IGNORE_VALUE, so the allowed set is
    # exactly the input's own class values plus the ignore value -- nothing
    # else (no blended intermediate) may appear.
    allowed_values = input_values | {IGNORE_VALUE}
    assert set(np.unique(augmented_mask).tolist()) <= allowed_values
