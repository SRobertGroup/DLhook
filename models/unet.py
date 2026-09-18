# pylint: disable=C0111, W0221, R0902
"""
U-Net architecture based on:
https://arxiv.org/abs/1505.04597
And modified to use Group Normalization
https://arxiv.org/abs/1803.08494
And then modified to use residual style connections.
With other alterations.

Copyright (C) 2019, 2020 Abraham George Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import torch
import torch.nn as nn
from PIL import Image


def get_valid_patch_sizes():
    return list((572 - (16*i) for i in range(31)))

class DownBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels*2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels*2,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels*2)
        )
        self.conv1x1 = nn.Sequential(
            # down sample channels again.
            nn.Conv2d(in_channels*2, in_channels,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        out1 = self.pool(x)
        out2 = self.conv1(out1)
        out3 = self.conv2(out2)
        out4 = self.conv1x1(out3)
        return out4 + out1


def crop_tensor(tensor, target):
    """ Crop tensor to target size """
    _, _, tensor_height, tensor_width = tensor.size()
    _, _, crop_height, crop_width = target.size()
    left = (tensor_width - crop_height) // 2
    top = (tensor_height - crop_width) // 2
    right = left + crop_width
    bottom = top + crop_height
    cropped_tensor = tensor[:, :, top: bottom, left: right]
    return cropped_tensor


class UpBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels,
                               kernel_size=2, stride=2, padding=0), # padding 0
            nn.ReLU(),
            nn.GroupNorm(32, in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, padding=1), # padding 0
            nn.ReLU(),
            nn.GroupNorm(32, in_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, padding=1), # padding 0
            nn.ReLU(),
            nn.GroupNorm(32, in_channels)
        )

    def forward(self, x, down_out):
        out = self.conv1(x)
        cropped = crop_tensor(down_out, out)
        out = cropped + out # residual
        out = self.conv2(out)
        out = self.conv3(out)
        return out


HEAD_GROUPNORM = 'groupnorm'
HEAD_PLAIN = 'plain'
VALID_HEADS = (HEAD_GROUPNORM, HEAD_PLAIN)


def build_conv_out(n_classes, head=HEAD_GROUPNORM):
    """Build the classification head.

    'groupnorm' (the default, and the ONLY behaviour before this parameter
    existed) is Conv2d -> ReLU -> GroupNorm(n_classes, n_classes). Every
    shipped RootPainter checkpoint under weights/ was trained with it, and
    models/UNetInference.py (the live GUI path) constructs UNetGNRes() with
    no arguments, so this default must never change: it produces
    conv_out.0.{weight,bias} plus conv_out.2.{weight,bias}.

    'plain' is a bare 1x1 Conv2d -- no ReLU, no GroupNorm -- and produces
    only conv_out.0.{weight,bias}. One group per channel makes the GroupNorm
    an *instance* norm over each class channel independently, per image, so
    every class is rescaled to zero mean / unit variance in every image and
    the network structurally cannot output "this class is absent here"; the
    preceding ReLU additionally clamps every negative logit to 0, erasing
    the "confidently not this class" signal. For classes covering ~0.1-0.2%
    of pixels that drives over-prediction (a precision-side failure). The
    plain head is the opt-in alternative under test -- see multi/Readme.md.

    Both variants are wrapped in nn.Sequential so the conv is `conv_out.0`
    either way, i.e. the plain head's keys are a strict subset of the
    groupnorm head's.
    """
    if head == HEAD_GROUPNORM:
        return nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.GroupNorm(n_classes, n_classes)
        )
    if head == HEAD_PLAIN:
        return nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1, padding=0)
        )
    raise ValueError(f"head must be one of {VALID_HEADS}, got {head!r}")


def head_from_state_dict(state_dict: dict) -> str:
    """Infer which `conv_out` head a bare state_dict was saved from.

    Checkpoints in the multi/ pipeline are bare state_dicts (`best.pt`/`last.pt`,
    and every shipped RootPainter `.pkl`) with no metadata recording the
    architecture, so the keys themselves are the only evidence available --
    and they are unambiguous: the groupnorm head's GroupNorm carries affine
    parameters at `conv_out.2.{weight,bias}`, which the plain head simply
    does not have. Inferring from the keys means every already-trained
    checkpoint keeps loading with no format change and no caller has to
    remember which head a given file was trained with.

    Tolerates the 'module.' prefix DataParallel-saved checkpoints carry (see
    multi/src/model.py's `_strip_module_prefix`), so it can be called before
    stripping.
    """
    for key in state_dict:
        name = key[len("module."):] if key.startswith("module.") else key
        if name.startswith("conv_out.2."):
            return HEAD_GROUPNORM
    return HEAD_PLAIN


class UNetGNRes(nn.Module):
    def __init__(self, im_channels=3, n_classes=2, head=HEAD_GROUPNORM):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(im_channels, 64, kernel_size=3, padding=1),  # padding 0
            nn.ReLU(),
            nn.GroupNorm(32, 64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),# padding 0
            nn.ReLU(),
            nn.GroupNorm(32, 64)
            # now at 568 x 568, 64 channels
        )
        self.down1 = DownBlock(64)
        self.down2 = DownBlock(64)
        self.down3 = DownBlock(64)
        self.down4 = DownBlock(64)
        self.up1 = UpBlock(64)
        self.up2 = UpBlock(64)
        self.up3 = UpBlock(64)
        self.up4 = UpBlock(64)
        # n_classes defaults to 2 and head defaults to 'groupnorm' so every
        # existing binary checkpoint (and the GUI, via
        # models/UNetInference.py) keeps loading unmodified; multi/'s 4-class
        # training passes n_classes=4, and optionally head='plain'.
        self.head = head
        self.conv_out = build_conv_out(n_classes, head)

    def forward(self, x):
        out1 = self.conv_in(x)
        out2 = self.down1(out1)
        out3 = self.down2(out2)
        out4 = self.down3(out3)
        out5 = self.down4(out4)
        out = self.up1(out5, out4)
        out = self.up2(out, out3)
        out = self.up3(out, out2)
        out = self.up4(out, out1)
        out = self.conv_out(out)
        return out


def align_output_to_target(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Center-crop a model's spatial output down to `target`'s spatial size,
    if they differ.

    UNetGNRes's output is always a multiple of 16 (four MaxPool2d(2) stages
    -- see above), so it is only ever exactly `input - 2*MARGIN` when `input`
    is itself in get_valid_patch_sizes()'s family (input % 16 == 12, same
    residue as the historical IN_SIZE=572). PatchDataset feeds the model
    `patch_size + 2*MARGIN` (see multi/src/data_loader.py), which shifts
    that residue, so for every patch_size in that family the model's real
    output comes back exactly 4px larger (2px/side) than `patch_size` -- a
    constant, deterministic remainder of the architecture's granularity, not
    a sizing bug in the dataset. Crop that remainder off the model's own
    freshly computed output here, right before the loss (the original crash
    site, multi/train_unet_multiclass.py:65-66 -> this function) -- this
    never touches or discards any of the label's own hand/pseudo-labelled
    pixels.

    Raises if `output` is smaller than `target` in either spatial dimension:
    that would mean PatchDataset's margin was too small for this patch_size,
    which should never happen with MARGIN imported from
    models/UNetInference.py, but must fail loudly rather than silently
    misalign predictions against labels if it ever does.
    """
    oh, ow = output.shape[-2:]
    th, tw = target.shape[-2:]
    if (oh, ow) == (th, tw):
        return output
    if oh < th or ow < tw:
        raise RuntimeError(
            f"Model output {oh}x{ow} is smaller than the label {th}x{tw} -- "
            "PatchDataset's context margin is too small for this patch_size "
            "(see align_output_to_target's docstring)."
        )
    top, left = (oh - th) // 2, (ow - tw) // 2
    return output[..., top:top + th, left:left + tw]


if __name__ == '__main__':
    import torch
    from torch.nn.functional import softmax
    from skimage.io import imsave
    import numpy as np
    unet = UNetGNRes()
    unet.eval()
    # test_input = np.random.rand(1, 3, 572, 572)
    test_input = np.zeros((1, 3, 572, 572))
    test_input = torch.from_numpy(test_input)
    if torch.cuda.is_available():
        test_input.cuda()
    test_input = test_input.float()
    output = unet(test_input)
    output = output.detach()
    print('output.shape', output.shape)
    softmaxed = softmax(output, 1)[:, 1, :] # just fg probability
    softmaxed = softmaxed[0] # single image.
    print('softmaxed shape = ', softmaxed.shape)

    im = Image.fromarray(np.array(softmaxed) * 256)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save('out.png')

       