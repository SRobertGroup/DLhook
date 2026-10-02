import os

import numpy as np
import torch

from multi.src.loss_functions import FocalLoss, align_output_to_target, build_loss
from multi.src.model import build_model, head_from_state_dict, warm_start_from_checkpoint

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPED_CHECKPOINT = os.path.join(REPO_ROOT, "weights", "RootPainter_weights", "cotyledon_v5.pkl")


def test_build_model_defaults_to_two_classes_matching_shipped_checkpoints():
    model = build_model({})
    assert model.conv_out[0].out_channels == 2


def test_default_unetgnres_still_loads_a_shipped_checkpoint():
    """The load-bearing compatibility check: models/UNetInference.py always
    constructs UNetGNRes() with no args, so the n_classes parameterisation
    must not change that default, and the shipped weights (saved from a
    DataParallel wrapper, hence the 'module.' prefix) must still load."""
    from models.unet import UNetGNRes

    model = UNetGNRes()
    assert model.conv_out[0].out_channels == 2

    state_dict = torch.load(SHIPPED_CHECKPOINT, map_location="cpu")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)


def test_build_model_with_four_classes_has_the_right_head_shape():
    model = build_model({"num_classes": 4, "in_channels": 3})
    assert model.conv_out[0].out_channels == 4

    x = torch.zeros(1, 3, 572, 572)
    with torch.no_grad():
        out = model(x)
    assert out.shape[1] == 4


def test_warm_start_loads_everything_except_conv_out():
    model = build_model({"num_classes": 4, "in_channels": 3})
    before = model.conv_out[0].weight.clone()

    loaded = warm_start_from_checkpoint(model, SHIPPED_CHECKPOINT)

    assert loaded, "expected at least some layers to load"
    assert not any(key.startswith("conv_out") for key in loaded)
    # conv_out is 4-class here vs. 2-class in the checkpoint: shape mismatch
    # means it must have been skipped, not just excluded by name.
    assert torch.equal(model.conv_out[0].weight, before)


DEFAULT_CONV_OUT_KEYS = [
    "conv_out.0.weight", "conv_out.0.bias",  # the 1x1 Conv2d
    "conv_out.2.weight", "conv_out.2.bias",  # GroupNorm(n_classes, n_classes) affine
]


def test_default_head_state_dict_keys_and_module_tree_are_unchanged():
    """The hard backward-compatibility constraint on the `head` parameter:
    UNetGNRes() with no arguments must still be byte-for-byte the
    architecture every shipped RootPainter checkpoint was trained with, since
    models/UNetInference.py constructs it that way for the live GUI. Pins the
    exact key list (96 tensors, conv_out being Conv2d/ReLU/GroupNorm) and the
    module repr, so any future head refactor that perturbs either fails here
    rather than at GUI startup."""
    from models.unet import UNetGNRes

    model = UNetGNRes()
    keys = list(model.state_dict().keys())
    assert len(keys) == 96
    assert [k for k in keys if k.startswith("conv_out")] == DEFAULT_CONV_OUT_KEYS

    children = list(model.conv_out.children())
    assert isinstance(children[0], torch.nn.Conv2d)
    assert isinstance(children[1], torch.nn.ReLU)
    assert isinstance(children[2], torch.nn.GroupNorm)
    assert (children[2].num_groups, children[2].num_channels) == (2, 2)
    assert repr(model.conv_out) == (
        "Sequential(\n"
        "  (0): Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1))\n"
        "  (1): ReLU()\n"
        "  (2): GroupNorm(2, 2, eps=1e-05, affine=True)\n"
        ")"
    )


def test_plain_head_drops_relu_and_groupnorm_and_their_keys():
    from models.unet import UNetGNRes

    model = UNetGNRes(n_classes=4, head="plain")
    children = list(model.conv_out.children())
    assert len(children) == 1
    assert isinstance(children[0], torch.nn.Conv2d)
    assert children[0].out_channels == 4
    assert not any(isinstance(m, (torch.nn.ReLU, torch.nn.GroupNorm))
                   for m in model.conv_out.modules())

    conv_out_keys = [k for k in model.state_dict() if k.startswith("conv_out")]
    assert conv_out_keys == ["conv_out.0.weight", "conv_out.0.bias"]

    # Everything OUTSIDE the head is untouched: the plain head's keys are a
    # strict subset of the groupnorm head's, differing only by the GroupNorm
    # affine pair.
    groupnorm_keys = set(UNetGNRes(n_classes=4).state_dict())
    assert set(model.state_dict()) == groupnorm_keys - {"conv_out.2.weight", "conv_out.2.bias"}


def test_plain_head_forward_can_emit_negative_logits():
    """The point of the plain head: the groupnorm head's ReLU clamps every
    negative logit to 0 and the per-channel GroupNorm then re-centres each
    class to zero mean per image, so no class can be uniformly low. The plain
    head must be able to produce genuinely negative logits."""
    from models.unet import UNetGNRes

    torch.manual_seed(0)
    model = UNetGNRes(n_classes=4, head="plain").eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 268, 268))
    assert out.shape[1] == 4
    assert (out < 0).any()


def test_unknown_head_name_is_rejected():
    from models.unet import UNetGNRes

    try:
        UNetGNRes(n_classes=4, head="instancenorm")
        assert False, "expected a ValueError for an unknown head name"
    except ValueError:
        pass


def test_build_model_head_defaults_to_groupnorm_and_is_config_selectable():
    """A config without the `head` key (every config written before it
    existed) must build today's architecture."""
    assert build_model({"num_classes": 4}).conv_out[-1].__class__.__name__ == "GroupNorm"
    assert build_model({"num_classes": 4, "head": None}).conv_out[-1].__class__.__name__ == "GroupNorm"
    assert build_model({"num_classes": 4, "head": "groupnorm"}).conv_out[-1].__class__.__name__ == "GroupNorm"
    assert len(build_model({"num_classes": 4, "head": "plain"}).conv_out) == 1


def test_head_from_state_dict_infers_the_head_from_the_keys_alone():
    """Checkpoints here are bare state_dicts with no architecture metadata,
    so the GroupNorm affine keys are the only signal -- including through the
    'module.' prefix a DataParallel-saved checkpoint carries."""
    from models.unet import UNetGNRes

    groupnorm_sd = UNetGNRes(n_classes=4).state_dict()
    plain_sd = UNetGNRes(n_classes=4, head="plain").state_dict()

    assert head_from_state_dict(groupnorm_sd) == "groupnorm"
    assert head_from_state_dict(plain_sd) == "plain"

    prefixed = {f"module.{k}": v for k, v in plain_sd.items()}
    assert head_from_state_dict(prefixed) == "plain"
    prefixed_gn = {f"module.{k}": v for k, v in groupnorm_sd.items()}
    assert head_from_state_dict(prefixed_gn) == "groupnorm"

    # And the real shipped binary checkpoints, which predate the parameter.
    assert head_from_state_dict(torch.load(SHIPPED_CHECKPOINT, map_location="cpu")) == "groupnorm"


def test_warm_start_skips_conv_out_for_the_plain_head_too():
    """A groupnorm-head binary checkpoint warm-starting a PLAIN-head model:
    conv_out.2.* has no counterpart at all in the target, so the skip must be
    by key prefix (it is) rather than relying on a shape mismatch."""
    model = build_model({"num_classes": 4, "in_channels": 3, "head": "plain"})
    before = model.conv_out[0].weight.clone()

    loaded = warm_start_from_checkpoint(model, SHIPPED_CHECKPOINT)

    assert loaded
    assert not any(key.startswith("conv_out") for key in loaded)
    assert torch.equal(model.conv_out[0].weight, before)
    assert len(model.conv_out) == 1  # head not silently rebuilt by the load


def test_build_loss_cross_entropy_ignores_255():
    loss_fn = build_loss({"type": "cross_entropy", "ignore_index": 255})
    logits = torch.zeros(1, 4, 2, 2)
    target = torch.full((1, 2, 2), 255, dtype=torch.long)

    loss = loss_fn(logits, target)
    assert torch.isnan(loss) or loss.item() == 0.0


def test_focal_loss_ignores_255_and_matches_class_count_check():
    loss_fn = FocalLoss(gamma=2.0, alpha=[0.1, 0.3, 0.3, 0.3], ignore_index=255)
    logits = torch.zeros(1, 4, 2, 2)
    target = torch.tensor([[[0, 255], [1, 255]]], dtype=torch.long)

    loss = loss_fn(logits, target)
    assert loss.item() > 0  # the two non-ignored pixels still contribute

    bad_alpha_loss_fn = FocalLoss(gamma=2.0, alpha=[0.1, 0.9], ignore_index=255)
    try:
        bad_alpha_loss_fn(logits, target)
        assert False, "expected a ValueError for mismatched alpha length"
    except ValueError:
        pass


def test_align_output_to_target_is_a_noop_when_shapes_already_match():
    output = torch.zeros(2, 4, 240, 240)
    target = torch.zeros(2, 240, 240)
    aligned = align_output_to_target(output, target)
    assert aligned.shape == output.shape


def test_align_output_to_target_center_crops_the_constant_plus_four_remainder():
    """The exact scenario PatchDataset + UNetGNRes produce: model output
    4px larger (2px/side) than the label, for any patch_size drawn from
    models/unet.py:get_valid_patch_sizes(). Also checks the crop is
    CENTERED (a marker placed at the true center survives at the target's
    own center after cropping), not shifted to one side."""
    output = torch.zeros(1, 4, 244, 244)
    output[:, :, 122, 122] = 99.0  # true center of the 244x244 output
    target = torch.zeros(1, 240, 240)

    aligned = align_output_to_target(output, target)
    assert aligned.shape[-2:] == (240, 240)
    assert aligned[0, :, 120, 120].eq(99.0).all()  # true center of the 240x240 crop


def test_align_output_to_target_raises_if_output_is_smaller_than_target():
    output = torch.zeros(1, 4, 238, 238)
    target = torch.zeros(1, 240, 240)
    try:
        align_output_to_target(output, target)
        assert False, "expected a RuntimeError when output is smaller than target"
    except RuntimeError:
        pass
