import csv

import numpy as np
from PIL import Image

from multi.src.patch_index import build_patch_index, discover_annotation_split, pad_to_min

CLASS_NAMES = ["background", "cotyledon", "hypocotyl", "radicle"]


def _write_mask(path, size, value=0):
    Image.fromarray(np.full((size, size), value, dtype=np.uint8)).save(path)


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_no_image_level_leakage_between_train_and_val(tmp_path):
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    out_dir = tmp_path / "patch_index"

    # 20 unannotated images, big enough to need no padding for a small patch.
    for i in range(20):
        _write_mask(masks_dir / f"img_{i}.png", size=64, value=1)

    n_train, n_val = build_patch_index(
        masks_dir=masks_dir,
        out_dir=out_dir,
        patch_size=32,
        train_stride=32,
        val_fraction=0.3,
        num_classes=4,
        class_names=CLASS_NAMES,
        split_seed=42,
        min_foreground_fraction=0.0,
        background_keep_ratio=1.0,
    )

    assert n_train > 0 and n_val > 0

    train_rows = _read_csv(out_dir / "train_patches.csv")
    val_rows = _read_csv(out_dir / "val_patches.csv")
    train_files = {r["filename"] for r in train_rows}
    val_files = {r["filename"] for r in val_rows}

    assert train_files.isdisjoint(val_files)
    assert len(train_files | val_files) == 20


def test_annotated_images_keep_their_rootpainter_split(tmp_path):
    """An image with a RootPainter val/ annotation must land in the val
    patch manifest even if the unannotated-remainder shuffle would have
    put it in train."""
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    for i in range(10):
        _write_mask(masks_dir / f"img_{i}.png", size=64, value=1)

    ann_dir = tmp_path / "annotations_cotyledon"
    (ann_dir / "val").mkdir(parents=True)
    (ann_dir / "val" / "img_0.png").write_bytes(b"stroke-bytes")

    n_train, n_val = build_patch_index(
        masks_dir=masks_dir,
        out_dir=tmp_path / "patch_index",
        patch_size=32,
        train_stride=32,
        val_fraction=0.0,  # every unannotated image would go to train
        num_classes=4,
        class_names=CLASS_NAMES,
        split_seed=1,
        min_foreground_fraction=0.0,
        background_keep_ratio=1.0,
        annotations_dirs={"cotyledon": ann_dir},
    )

    val_rows = _read_csv(tmp_path / "patch_index" / "val_patches.csv")
    assert {r["filename"] for r in val_rows} == {"img_0.png"}
    assert n_val > 0


def test_conflicting_split_across_classes_keeps_first_and_warns():
    import warnings
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cot_dir = tmp / "cot"
        hyp_dir = tmp / "hyp"
        (cot_dir / "train").mkdir(parents=True)
        (hyp_dir / "val").mkdir(parents=True)
        (cot_dir / "train" / "shared.png").write_bytes(b"a")
        (hyp_dir / "val" / "shared.png").write_bytes(b"b")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assignment = discover_annotation_split({"cotyledon": cot_dir, "hypocotyl": hyp_dir})

    assert assignment["shared.png"] == "train"
    assert any("shared.png" in str(w.message) for w in caught)


def test_small_images_are_reflect_padded_up_to_patch_size():
    small = np.arange(9, dtype=np.uint8).reshape(3, 3)
    padded = pad_to_min(small, 7)
    assert padded.shape == (7, 7)


def test_min_foreground_fraction_and_background_keep_ratio_are_applied(tmp_path):
    """A patch below min_foreground_fraction with background_keep_ratio=0 is
    dropped; one at or above the threshold is always kept."""
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    size = 32
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[:2, :2] = 1  # a tiny foreground speck, well under 1% of the image
    Image.fromarray(mask).save(masks_dir / "img.png")

    n_train, n_val = build_patch_index(
        masks_dir=masks_dir,
        out_dir=tmp_path / "patch_index",
        patch_size=size,
        train_stride=size,
        val_fraction=0.0,
        num_classes=4,
        class_names=CLASS_NAMES,
        split_seed=0,
        min_foreground_fraction=0.5,  # this patch's fraction is far below this
        background_keep_ratio=0.0,    # ...and background_keep_ratio=0 drops it
    )

    assert n_train == 0
    assert n_val == 0
