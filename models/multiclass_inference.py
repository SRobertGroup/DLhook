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

import os

import cv2
import numpy as np
import torch
from torch.nn.functional import softmax

from models.UNetInference import (
    IMAGE_CHUNK,
    IN_SIZE,
    MARGIN,
    OUT_SIZE,
    _crop_from_pad,
    _get_tile_coords,
    _pad_reflect,
    _pad_to_min,
)
from models.normalization import zscore_normalize
from models.unet import UNetGNRes, align_output_to_target, has_landmark_head, head_from_state_dict, landmark_hidden_from_state_dict, landmark_root_from_state_dict
from utils.landmark_readout import READOUT_RADIUS, readout_from_fields

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
        # A checkpoint trained with the optional landmark head carries conv_kp /
        # cls_overhook keys; build the matching architecture so the strict load
        # works. segment_many_argmax and the rest of the segmentation path
        # ignore the extra head entirely (UNetGNRes.forward is unchanged).
        self.has_landmarks = has_landmark_head(state_dict)
        self.model = UNetGNRes(n_classes=num_classes, head=self.head, landmark_head=self.has_landmarks,
                               landmark_hidden=landmark_hidden_from_state_dict(state_dict),
                               landmark_root=landmark_root_from_state_dict(state_dict))
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

    @classmethod
    def from_model(cls, model, num_classes: int = 4, batch_size: int | None = None,
                   device: torch.device | None = None,
                   in_size: int = IN_SIZE, out_size: int = OUT_SIZE, margin: int = MARGIN):
        """Wrap a live UNetGNRes (e.g. the model being trained) instead of loading a
        checkpoint, so the training loop can validate through exactly the tiling and
        readout used at inference. The caller owns the model's train/eval mode."""
        self = cls.__new__(cls)
        self.device = device or next(model.parameters()).device
        self.num_classes = num_classes
        self.in_size, self.out_size, self.margin = in_size, out_size, margin
        self._out_shape_ref = torch.empty(out_size, out_size)
        self.model = model
        core = self._core()
        self.head = core.head
        self.has_landmarks = bool(getattr(core, "landmark_head", False))
        self.batch_size = batch_size or CPU_BATCH_SIZE
        return self

    def _core(self):
        """The underlying UNetGNRes, whether or not DataParallel wraps it."""
        return self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

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

    def predict_files_labelmaps(self, image_paths):
        """Multiclass counterpart to `UNetInference.predict_files`
        (models/UNetInference.py:222) -- same contract, but returns label
        maps instead of a single-label binary mask, since one checkpoint
        here stands in for all of BinaryBackend's per-label models at once.

        Returns {basename: uint8 (H, W) label map, values 0..num_classes-1}
        -- no disk writes; the caller (MulticlassBackend.predict_into)
        decides how each class value becomes a stored mask.

        Images are read and chunked exactly like predict_files: IMAGE_CHUNK
        images decoded at a time via cv2.imread to bound host memory, an
        unreadable path prints the identical `[WARNING] Could not load: ...`
        line and is skipped (and so is simply absent from the returned
        dict), and the per-image argmax stitching itself is delegated to
        `segment_many_argmax` rather than reimplemented here.
        """
        label_maps = {}

        for start in range(0, len(image_paths), IMAGE_CHUNK):
            chunk_paths = image_paths[start:start + IMAGE_CHUNK]
            valid_paths, images = [], []
            for img_path in chunk_paths:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"[WARNING] Could not load: {img_path}")
                    continue
                valid_paths.append(img_path)
                images.append(image)

            if not images:
                continue

            label_map_batch = self.segment_many_argmax(images)
            for img_path, label_map in zip(valid_paths, label_map_batch):
                label_maps[os.path.basename(img_path)] = label_map

        return label_maps

    # --- landmark head -------------------------------------------------------

    def _run_batch_landmarks(self, tiles):
        """Per tile: (junction probability map, 4 direction-field maps, overhook
        probability, extra), the maps cropped to out_size like the segmentation
        output; extra is None, or for a collar/root head {"collar_hm", "root_vec"}.
        Same per-tile z-score as _run_batch."""
        batch = np.stack([zscore_normalize(t) for t in tiles]).transpose(0, 3, 1, 2)
        tensor = torch.from_numpy(batch).to(self.device)
        with torch.inference_mode():
            _, kp, overhook = self._core().forward_with_landmarks(tensor)
            kp = align_output_to_target(kp, self._out_shape_ref).float()
            hm = torch.sigmoid(kp[:, 0]).cpu().numpy()
            vec = kp[:, 1:5].cpu().numpy()
            overhook = torch.sigmoid(overhook.float()).cpu().numpy()
            extra = None
            if kp.shape[1] >= 8:
                collar = torch.sigmoid(kp[:, 5]).cpu().numpy()
                root = kp[:, 6:8].cpu().numpy()
                extra = [{"collar_hm": collar[i], "root_vec": root[i]} for i in range(hm.shape[0])]
        return [(hm[i], vec[i], float(overhook[i]), extra[i] if extra else None) for i in range(hm.shape[0])]

    def predict_landmarks(self, images, readout_radius: float = READOUT_RADIUS):
        """Hook angle from the landmark head for a list of native BGR crops.

        Returns one entry per image: the dict from
        utils.landmark_readout.readout_from_fields (junction, theta, bio,
        overhook, directions, peak), or None when no junction is found."""
        return [readout_from_fields(hm, vec, prob, radius=readout_radius, **(extra or {}))
                for hm, vec, prob, extra in self.predict_landmark_fields(images)]

    def predict_landmark_fields(self, images):
        """Raw landmark maps for a list of native BGR crops: one (hm (H, W),
        vec (4, H, W), overhook_prob, extra) per image, at the crop's own size (extra is
        None, or {"collar_hm" (H, W), "root_vec" (2, H, W)} for a collar/root head), so a
        caller can read the angle at a junction other than the heatmap peak
        (e.g. one tracked across frames).
        Tiles are stitched like the segmentation output (overwrite). The
        overhook probability is image-level, so it is taken from the tile with
        the strongest junction peak -- the tile that actually contains the hook.
        Requires a checkpoint trained with the landmark head."""
        if not self.has_landmarks:
            raise RuntimeError("this checkpoint has no landmark head (no conv_kp / cls_overhook keys)")

        plans, hm_out, vec_out, best = [], [], [], []
        collar_out, root_out = [], []
        work = []
        for idx, image in enumerate(images):
            orig_h, orig_w = image.shape[:2]
            padded, base_pad, base_h, base_w, tile_coords = self._plan_tiles(image)
            hm_out.append(np.zeros((base_h, base_w), dtype=np.float32))
            vec_out.append(np.zeros((4, base_h, base_w), dtype=np.float32))
            collar_out.append(np.zeros((base_h, base_w), dtype=np.float32))
            root_out.append(np.zeros((2, base_h, base_w), dtype=np.float32))
            best.append((-1.0, 0.0))                       # (strongest tile peak, its overhook prob)
            plans.append((orig_h, orig_w, base_pad))
            for (x, y) in tile_coords:
                work.append((idx, x, y, padded[y:y + self.in_size, x:x + self.in_size]))

        for start in range(0, len(work), self.batch_size):
            chunk = work[start:start + self.batch_size]
            for (idx, x, y, _), (hm, vec, overhook, extra) in zip(chunk, self._run_batch_landmarks([c[3] for c in chunk])):
                hm_out[idx][y:y + self.out_size, x:x + self.out_size] = hm
                vec_out[idx][:, y:y + self.out_size, x:x + self.out_size] = vec
                if extra is not None:
                    collar_out[idx][y:y + self.out_size, x:x + self.out_size] = extra["collar_hm"]
                    root_out[idx][:, y:y + self.out_size, x:x + self.out_size] = extra["root_vec"]
                if float(hm.max()) > best[idx][0]:
                    best[idx] = (float(hm.max()), overhook)

        results = []
        for idx, (orig_h, orig_w, base_pad) in enumerate(plans):
            hm = _crop_from_pad(hm_out[idx], base_pad)
            vec = np.stack([_crop_from_pad(vec_out[idx][c], base_pad) for c in range(4)])
            assert hm.shape == (orig_h, orig_w)
            extra = None
            if self._core().landmark_root:
                extra = {"collar_hm": _crop_from_pad(collar_out[idx], base_pad),
                         "root_vec": np.stack([_crop_from_pad(root_out[idx][c], base_pad) for c in range(2)])}
            results.append((hm, vec, best[idx][1], extra))
        return results

    def predict_files_landmarks(self, image_paths, fields=False, readout_radius: float = READOUT_RADIUS):
        """{basename: readout dict or None} for image paths, chunked like
        predict_files_labelmaps (unreadable paths are skipped with the same
        warning and absent from the result). With fields=True the value is the
        raw (hm, vec, overhook_prob, extra) from predict_landmark_fields instead."""
        out = {}
        for start in range(0, len(image_paths), IMAGE_CHUNK):
            paths, images = [], []
            for img_path in image_paths[start:start + IMAGE_CHUNK]:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"[WARNING] Could not load: {img_path}")
                    continue
                paths.append(img_path)
                images.append(image)
            if images:
                results = self.predict_landmark_fields(images)
                for img_path, (hm, vec, prob, extra) in zip(paths, results):
                    out[os.path.basename(img_path)] = (
                        (hm, vec, prob, extra) if fields
                        else readout_from_fields(hm, vec, prob, radius=readout_radius, **(extra or {})))
        return out
