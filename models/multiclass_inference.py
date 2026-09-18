"""Multiclass counterpart to `models.UNetInference.UNetInference`, used by
`multi/evaluate_multiclass.py` and friends to run a trained 4-class
checkpoint over full-size crops.

`models/UNetInference.py` is off-limits (it drives the live GUI pipeline --
see CLAUDE.md's "Do NOT modify the live GUI pipeline" list), and it is
hardcoded to a 2-class softmax readout (`softmax(out, dim=1)[:, 1]`), so it
cannot serve a 4-class checkpoint as-is. Rather than fork its logic, this
module imports its tiling/stitching *helpers* (the pad/tile-coordinate
functions, plus the IN_SIZE/OUT_SIZE/MARGIN constants as this class's
defaults) unchanged, but:

- runs them at a caller-chosen geometry (`in_size`/`out_size`/`margin`
  constructor args, still defaulting to 572/560/6) rather than the hardcoded
  constants, so a checkpoint can be evaluated at the tile size it was
  actually trained at (see `__init__`'s docstring);
- normalises each input tile with the same per-array z-score training uses
  (`normalization.zscore_normalize`) instead of UNetInference's `/255.0` --
  see `_run_batch`'s docstring for why per-tile is the scope that matches
  training;
- replaces the readout with a per-class softmax, stitched one channel at a
  time, reduced with argmax after the tiles are assembled back into the
  image (or left as raw per-class probabilities -- see
  `segment_many_argmax`'s `return_probs`);
- infers the `conv_out` head variant from the checkpoint's own keys
  (`models/unet.py`'s 'groupnorm' vs 'plain'), so the same call site loads
  either without the caller tracking which is which -- see `__init__`.

Note on RGB vs BGR: `multi/src/data_loader.py` (training) reads training
crops via PIL (`.convert("RGB")`) while this module (like UNetInference)
reads crops via cv2 (BGR). This is *not* a bug worth fixing: every crop in
`cropped_training_set/` is greyscale (PIL mode "L" upstream, R==B in every
pixel once expanded to 3 channels), so channel order carries no information
here and swapping it would be a no-op change dressed up as a fix. Left
alone deliberately.

Lives under models/ (rather than multi/src/) because models/ is the shared
home for inference-time code that both multi/ and the GUI depend on -- see
CLAUDE.md's multi/ boundary note. multi/src/multiclass_inference.py
re-exports this module unchanged so existing imports keep working.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn.functional import softmax

from models.UNetInference import (
    IN_SIZE,
    MARGIN,
    OUT_SIZE,
    _crop_from_pad,
    _get_tile_coords,
    _pad_reflect,
    _pad_to_min,
)
from models.normalization import zscore_normalize
from models.unet import UNetGNRes, align_output_to_target, head_from_state_dict

CPU_BATCH_SIZE = 4


class MulticlassInference:
    """Runs a 4-class UNetGNRes checkpoint over full-size BGR crops using
    the same tiling *mechanism* as the binary UNetInference (reflect-pad,
    tile, batch, stitch), but at whatever tile geometry the caller passes --
    see `in_size`/`out_size`/`margin` below. Unlike UNetInference, this class
    is not process-cached -- evaluate_multiclass.py constructs one instance
    for the one checkpoint under test."""

    def __init__(self, checkpoint_path, num_classes: int = 4, batch_size: int | None = None,
                 device: torch.device | None = None,
                 in_size: int = IN_SIZE, out_size: int = OUT_SIZE, margin: int = MARGIN):
        """`in_size`/`out_size`/`margin` default to the module-level
        IN_SIZE/OUT_SIZE/MARGIN (572/560/6) so existing callers are
        unaffected. Pass the *training* patch geometry instead
        (in_size=264, out_size=252, margin=MARGIN -- see
        multi/evaluate_multiclass.py) to evaluate a checkpoint the way it was
        actually trained: training_config.yaml's `data.patch_size: 252` (264
        = 252 + 2*MARGIN input) was chosen because the real crops are narrow
        (median 74x246px) and 572 reflect-pads ~94% of each tile with
        synthetic context that patch_size=252 avoids.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.in_size = in_size
        self.out_size = out_size
        self.margin = margin
        # UNetGNRes's output is always 16*floor(input/16) (four MaxPool2d(2)
        # stages -- see models/unet.py). At the default IN_SIZE=572 that is
        # exactly OUT_SIZE=560, but at the training geometry (in_size=264)
        # it comes back as 256, not out_size=252 -- the same +4px constant
        # remainder models/unet.py's align_output_to_target already crops
        # off in training (see that function's docstring). Reused here
        # (in _run_batch) rather than duplicated so both paths agree on
        # exactly how the remainder is removed (a centered crop). This
        # tensor only ever has its .shape read, never its values.
        self._out_shape_ref = torch.empty(out_size, out_size)
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        # The two conv_out heads have DIFFERENT state_dict keys (the
        # groupnorm head adds conv_out.2.{weight,bias}), and these
        # checkpoints are bare state_dicts with no architecture metadata, so
        # the head is inferred from the checkpoint's own keys rather than
        # passed in -- evaluate_multiclass.py can then score a groupnorm-head
        # and a plain-head checkpoint with the identical command line. See
        # models/unet.py:head_from_state_dict.
        self.head = head_from_state_dict(state_dict)
        self.model = UNetGNRes(n_classes=num_classes, head=self.head)
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            # Shipped/checkpointed state_dicts saved from a DataParallel-wrapped
            # model have a "module." prefix on every key (see
            # multi/src/model.py's _strip_module_prefix) -- fall back to
            # wrapping instead of stripping, mirroring UNetInference._load_model
            # exactly.
            self.model = torch.nn.DataParallel(self.model)
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size or (CPU_BATCH_SIZE if self.device.type != "cuda" else CPU_BATCH_SIZE)

    def _plan_tiles(self, image):
        base_image, base_pad = _pad_to_min(image, self.in_size)
        padded = _pad_reflect(base_image, self.margin)
        base_h, base_w = base_image.shape[:2]
        tile_coords = _get_tile_coords(
            base_h, base_w, padded.shape[0], padded.shape[1], self.out_size, self.in_size
        )
        return padded, base_pad, base_h, base_w, tile_coords

    def _run_batch(self, tiles):
        """Same batching as UNetInference._run_batch, except normalisation
        uses the shared training z-score (normalization.zscore_normalize)
        instead of UNetInference's `/255.0`, and the full
        (num_classes, out_size, out_size) softmax is kept instead of only
        channel 1.

        Normalisation scope -- per TILE, not per whole source image:
        PatchDataset (multi/src/data_loader.py) z-scores each training patch
        using only that patch's own pixels (mean/std pooled over the exact
        HxWxC array fed to the network, margin included) -- i.e. training
        always normalises exactly the one array about to enter the network,
        nothing more. `_plan_tiles` (via `_pad_to_min`/`_get_tile_coords`,
        both unchanged from UNetInference) reflect-pads the base image up to
        at least `in_size` *before* tiling, and `in_size` is by construction
        larger than `out_size`, so in practice this NEVER reduces to a
        single tile -- even a source crop far smaller than one tile still
        gets tiled into a minimum 2x2 (4-tile) grid of heavily overlapping
        `in_size` windows (verified empirically: a 60x98 crop at the
        training geometry still produces 4 tiles). There is consequently no
        well-defined "whole image" canvas here smaller than the padded base
        that would correspond to one network call anyway. Given that,
        per-tile is still the choice that matches training's actual
        semantics -- normalise exactly what is about to be forward-passed,
        using only that array's own pixels -- applied consistently to
        whichever tile a given network call happens to be. Pooling
        statistics across tiles (a form of "per-image" normalisation)
        would instead let one tile's z-score be influenced by pixels no
        single training patch was ever normalised against.
        """
        batch = np.stack([zscore_normalize(t) for t in tiles])  # N, H, W, C
        batch = batch.transpose(0, 3, 1, 2)  # N, C, H, W
        tensor = torch.from_numpy(batch).to(self.device)
        with torch.inference_mode():
            out = self.model(tensor)
            # See __init__'s comment on _out_shape_ref: crop the model's raw
            # (16-multiple) output down to out_size before softmax/stitching.
            out = align_output_to_target(out, self._out_shape_ref)
            probs = softmax(out, dim=1).float().cpu().numpy()  # N, num_classes, out_size, out_size
        return [probs[i] for i in range(probs.shape[0])]

    def segment_many_argmax(self, images, return_probs: bool = False):
        """Segment a list of native BGR crops. Returns one array per image,
        each exactly the input image's H x W (no resize):

        - `return_probs=False` (default): uint8 (H, W) label map -- argmax
          class index (0..num_classes-1) after stitching. This is the
          readout evaluate_multiclass.py uses for a fair, threshold-free
          comparison against the binary baselines (see that module's
          docstring).
        - `return_probs=True`: float32 (num_classes, H, W) softmax
          probability stack after stitching, before the argmax reduction --
          for callers that need actual per-class probabilities rather than
          the network's discrete top-1 call.
        """
        plans = []          # per image: (orig_h, orig_w, base_pad)
        outputs = []        # per image: (num_classes, base_h, base_w) prob accumulator
        work = []           # flat list of (image_index, x, y, tile_uint8)

        for idx, image in enumerate(images):
            orig_h, orig_w = image.shape[:2]
            padded, base_pad, base_h, base_w, tile_coords = self._plan_tiles(image)
            outputs.append(np.zeros((self.num_classes, base_h, base_w), dtype=np.float32))
            plans.append((orig_h, orig_w, base_pad))
            for (x, y) in tile_coords:
                work.append((idx, x, y, padded[y:y + self.in_size, x:x + self.in_size]))

        for start in range(0, len(work), self.batch_size):
            chunk = work[start:start + self.batch_size]
            probs = self._run_batch([item[3] for item in chunk])
            for (idx, x, y, _), prob in zip(chunk, probs):
                outputs[idx][:, y:y + self.out_size, x:x + self.out_size] = prob

        results = []
        for out, (orig_h, orig_w, base_pad) in zip(outputs, plans):
            cropped = np.stack([_crop_from_pad(out[c], base_pad) for c in range(self.num_classes)])
            assert cropped.shape[1:] == (orig_h, orig_w)
            if return_probs:
                results.append(cropped)
            else:
                results.append(np.argmax(cropped, axis=0).astype(np.uint8))
        return results
