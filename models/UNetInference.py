import os
from math import ceil
import cv2
import torch
import numpy as np
from torch.nn.functional import softmax
from models.unet import UNetGNRes

# DLhook's UNetGNRes uses same-padding (3x3, padding=1) convs, but its
# MaxPool2d stages still floor-divide non-16-divisible sizes, so a 572x572
# input does NOT come back out at 572x572 -- it comes back at 560x560
# (572 -> 286 -> 143 -> 71 -> 35 -> 70 -> 140 -> 280 -> 560, verified against
# the actual model). So, same as RootPainter/RootSAS, tiles need a fixed
# context margin around each output region rather than in_size == out_size.
IN_SIZE = 572
OUT_SIZE = 560
MARGIN = (IN_SIZE - OUT_SIZE) // 2

# Every tile shares the fixed IN_SIZE input, so cuDNN's autotuned algorithm
# stays valid across the whole run -- turning benchmark on lets it pick the
# fastest kernel once and reuse it. No-op on CPU.
torch.backends.cudnn.benchmark = True

# Tile-batching bounds (see UNetInference._auto_batch_size). MAX_BATCH caps the
# per-forward-pass tile count even when plenty of VRAM is free; CPU_BATCH_SIZE
# is the fixed fallback when there's no CUDA device to size against.
MAX_BATCH = 32
CPU_BATCH_SIZE = 4
_VRAM_SAFETY_FRACTION = 0.7
# Conservative peak activation estimate for one 572x572x3 tile through
# UNetGNRes in fp32 (inference, no autograd graph). Deliberately generous so
# the auto-sizer errs toward fewer tiles rather than an out-of-memory crash.
_BYTES_PER_TILE = 500 * 1024 * 1024

# How many source images to decode/hold in memory at once inside predict_files.
# Tiles are still batched across all images in a chunk, so this only bounds peak
# host memory for the decoded crops themselves, not the batching benefit.
IMAGE_CHUNK = 64


def _pad_reflect(image, margin):
    return np.pad(image, [(margin, margin), (margin, margin), (0, 0)], mode='reflect')


def _pad_to_min(image, min_size):
    """Reflect-pad image (H,W,C) up to at least min_size in both dimensions."""
    h, w = image.shape[:2]
    h_pad = max(0, min_size - h)
    w_pad = max(0, min_size - w)
    h_before, h_after = h_pad // 2, h_pad - h_pad // 2
    w_before, w_after = w_pad // 2, w_pad - w_pad // 2
    if h_pad or w_pad:
        image = np.pad(image, [(h_before, h_after), (w_before, w_after), (0, 0)], mode='reflect')
    return image, (h_before, h_after, w_before, w_after)


def _crop_from_pad(image, pad_settings):
    h_before, h_after, w_before, w_after = pad_settings
    h, w = image.shape[:2]
    return image[h_before:h - h_after, w_before:w - w_after]


def _get_tile_coords(base_height, base_width, padded_height, padded_width, out_size, in_size):
    """
    Coordinates (into the margin-padded image) of IN_SIZE input tiles spaced
    OUT_SIZE apart, covering (base_height, base_width). The last row/column
    is shifted inward (rather than resized) so every tile stays at the
    network's native input size -- same approach RootSAS/RootPainter use.
    """
    horizontal_count = ceil(base_width / out_size)
    vertical_count = ceil(base_height / out_size)

    x_coords = [i * out_size for i in range(horizontal_count - 1)]
    y_coords = [i * out_size for i in range(vertical_count - 1)]
    x_coords.append(padded_width - in_size)
    y_coords.append(padded_height - in_size)

    return [(x, y) for x in x_coords for y in y_coords]


# Process-wide cache of loaded predictors, keyed by weight path. Each RootPainter
# .pkl is loaded from disk and moved to the device exactly once per process;
# both the batch pipeline (run_apical_pipeline) and the per-seedling preview
# (segment_single_seedling) then share the same resident model instead of
# re-loading weights on every call.
_PREDICTOR_CACHE = {}


def get_predictor(model_path, **kwargs):
    """Return a process-cached UNetInference for `model_path`, constructing it
    (and loading its weights) only on first request. `kwargs` (batch_size,
    half) are applied only when the instance is first created; later calls with
    the same path return the existing instance and ignore them."""
    predictor = _PREDICTOR_CACHE.get(model_path)
    if predictor is None:
        predictor = UNetInference(model_path, **kwargs)
        _PREDICTOR_CACHE[model_path] = predictor
    return predictor


class UNetInference:
    def __init__(self, model_path, batch_size=None, half=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # FP16 only ever makes sense on CUDA; ignore the flag on CPU.
        self.half = bool(half) and self.device.type == "cuda"
        self.model = UNetGNRes()
        self._load_model(model_path)
        self.model.eval()
        self.batch_size = batch_size if batch_size else self._auto_batch_size()
        print(f"[INFO] UNetInference {os.path.basename(model_path)}: "
              f"device={self.device.type}, batch_size={self.batch_size}, half={self.half}")

    def _auto_batch_size(self):
        """Largest tile batch that fits under a safety fraction of free VRAM,
        clamped to [1, MAX_BATCH]. On CPU (or if the CUDA query fails) fall back
        to a fixed safe default."""
        if self.device.type != "cuda":
            return CPU_BATCH_SIZE
        try:
            free_bytes, _total = torch.cuda.mem_get_info(self.device)
        except Exception as e:
            print(f"[WARNING] Could not query CUDA memory ({e}); using batch size 1")
            return 1
        budget = int(free_bytes * _VRAM_SAFETY_FRACTION)
        # A single tile's activations use less memory in fp16.
        per_tile = _BYTES_PER_TILE // 2 if self.half else _BYTES_PER_TILE
        return int(max(1, min(MAX_BATCH, budget // per_tile)))

    def _load_model(self, model_path):
        state_dict = torch.load(model_path, map_location=self.device)
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            self.model = torch.nn.DataParallel(self.model)
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def _plan_tiles(self, image):
        """
        Geometry for one native H x W x 3 BGR crop: reflect-pad up to at least
        one input tile, add the fixed context margin, and enumerate the input
        tile coordinates -- identical to the original per-image tiling, just
        separated from the forward pass so tiles from many images can be
        batched together. Returns the padded image, the pad settings needed to
        crop the stitched output back to native size, the base (pre-margin)
        H/W, and the list of (x, y) tile origins.
        """
        base_image, base_pad = _pad_to_min(image, IN_SIZE)
        padded = _pad_reflect(base_image, MARGIN)
        base_h, base_w = base_image.shape[:2]
        tile_coords = _get_tile_coords(base_h, base_w, padded.shape[0], padded.shape[1], OUT_SIZE, IN_SIZE)
        return padded, base_pad, base_h, base_w, tile_coords

    def _run_batch(self, tiles):
        """
        Forward-pass a list of IN_SIZE x IN_SIZE x 3 BGR uint8 tiles in one go.
        Returns a list of OUT_SIZE x OUT_SIZE float32 foreground-probability
        maps, one per input tile. Numerics match the old per-tile path exactly
        in fp32: the same /255 scaling, BGR order, and softmax channel-1
        readout -- and because UNetGNRes uses GroupNorm (per-sample, not
        BatchNorm), batching several tiles together does not change any single
        tile's output.
        """
        batch = np.stack([t.astype(np.float32) / 255.0 for t in tiles])  # N, H, W, C
        batch = batch.transpose(0, 3, 1, 2)  # N, C, H, W
        tensor = torch.from_numpy(batch).to(self.device)
        with torch.inference_mode():
            if self.half:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = self.model(tensor)
            else:
                out = self.model(tensor)
            probs = softmax(out, dim=1)[:, 1].float().cpu().numpy()  # N, OUT_SIZE, OUT_SIZE
        return [probs[i] for i in range(probs.shape[0])]

    def _segment_many(self, images):
        """
        Segment a list of native BGR crops, batching input tiles across ALL of
        them into fixed-size forward passes (self.batch_size tiles each).
        Returns one foreground-probability map per image, each exactly the same
        H x W as its input -- no resize, so no aspect-ratio distortion.
        """
        plans = []          # per image: (orig_h, orig_w, base_pad)
        outputs = []        # per image: base_h x base_w prob accumulator
        work = []           # flat list of (image_index, x, y, tile_uint8)

        for idx, image in enumerate(images):
            orig_h, orig_w = image.shape[:2]
            padded, base_pad, base_h, base_w, tile_coords = self._plan_tiles(image)
            outputs.append(np.zeros((base_h, base_w), dtype=np.float32))
            plans.append((orig_h, orig_w, base_pad))
            for (x, y) in tile_coords:
                work.append((idx, x, y, padded[y:y + IN_SIZE, x:x + IN_SIZE]))

        for start in range(0, len(work), self.batch_size):
            chunk = work[start:start + self.batch_size]
            probs = self._run_batch([item[3] for item in chunk])
            for (idx, x, y, _), prob in zip(chunk, probs):
                outputs[idx][y:y + OUT_SIZE, x:x + OUT_SIZE] = prob

        results = []
        for out, (orig_h, orig_w, base_pad) in zip(outputs, plans):
            out = _crop_from_pad(out, base_pad)
            assert out.shape == (orig_h, orig_w)
            results.append(out)
        return results

    @staticmethod
    def _to_binary_mask(foreground_prob):
        """
        Threshold a foreground-probability map into a binary uint8 mask using
        the in-memory convention: **255 = foreground, 0 = background**.

        Note this is the opposite of what the old _write_mask() put on disk
        (0 = foreground), which forced every consumer to cv2.bitwise_not() the
        mask back before use. MaskStore now holds the un-inverted form and
        re-inverts only in MaskStore.dump(), so dumped files stay byte-
        compatible with the legacy on-disk convention.
        """
        return (foreground_prob > 0.5).astype(np.uint8) * 255

    def predict_files(self, image_paths, label="0"):
        """
        Segment an explicit list of image paths and RETURN the masks as
        {basename: uint8 mask (255 = foreground)} -- no disk writes. The caller
        (normally MaskStore.put_raw_bulk) decides where they live.

        Images are processed in IMAGE_CHUNK-sized windows to bound host memory,
        and within each window every tile from every image is batched together
        for the forward pass, so many small single-tile crops keep the device
        busy instead of running one tile at a time.

        `label` is accepted for logging/symmetry with the store's keying; it no
        longer affects any filename since nothing is written here.
        """
        masks = {}

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

            prob_maps = self._segment_many(images)
            for img_path, prob in zip(valid_paths, prob_maps):
                masks[os.path.basename(img_path)] = self._to_binary_mask(prob)

        print(f"[INFO] Segmented {len(masks)} frames (label {label})")
        return masks
