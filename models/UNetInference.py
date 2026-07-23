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


class UNetInference:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNetGNRes()
        self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path):
        state_dict = torch.load(model_path, map_location=self.device)
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            self.model = torch.nn.DataParallel(self.model)
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def _segment_native(self, image):
        """
        Run the model over `image` (native H x W x 3 BGR crop), tiled at the
        network's native resolution and stitched back together. The returned
        foreground-probability map is always exactly the same H x W as the
        input -- no resize, so no aspect-ratio distortion regardless of the
        crop's own shape/size.
        """
        orig_h, orig_w = image.shape[:2]

        # Ensure at least one full input tile's worth of real image before
        # the smaller context-margin pad below (mirrors RootPainter/RootSAS).
        base_image, base_pad = _pad_to_min(image, IN_SIZE)
        padded = _pad_reflect(base_image, MARGIN)
        base_h, base_w = base_image.shape[:2]

        output = np.zeros((base_h, base_w), dtype=np.float32)
        tile_coords = _get_tile_coords(base_h, base_w, padded.shape[0], padded.shape[1], OUT_SIZE, IN_SIZE)
        for x, y in tile_coords:
            tile = padded[y:y + IN_SIZE, x:x + IN_SIZE]
            tile = tile.astype(np.float32) / 255.0
            tile = tile.transpose(2, 0, 1)  # HWC to CHW
            tensor = torch.tensor(tile).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(tensor)
            probs = softmax(out, dim=1)[0, 1].cpu().numpy()  # OUT_SIZE x OUT_SIZE
            output[y:y + OUT_SIZE, x:x + OUT_SIZE] = probs

        output = _crop_from_pad(output, base_pad)
        assert output.shape == (orig_h, orig_w)
        return output

    def _predict_one(self, img_path, output_dir, label):
        filename = os.path.basename(img_path)
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARNING] Could not load: {img_path}")
            return

        foreground_prob = self._segment_native(image)

        # Inverse the mask (same convention as before: 0 = foreground)
        mask = cv2.bitwise_not((foreground_prob > 0.5).astype(np.uint8) * 255)

        # Ensure binary mask
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        out_name = os.path.splitext(filename)[0] + f"-{label}.png"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, binary_mask)

    def predict_folder(self, image_dir, output_dir, label="0"):
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(image_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            self._predict_one(os.path.join(image_dir, filename), output_dir, label)
        print(f"[INFO] Saved segmentation files")

    def predict_files(self, image_paths, output_dir, label="0"):
        """Same as predict_folder, but scoped to an explicit list of image
        paths instead of an entire directory -- lets a caller segment just
        one seedling's cropped frames without touching every other seedling's
        files that also live in the same data/images/ directory."""
        os.makedirs(output_dir, exist_ok=True)

        for img_path in image_paths:
            self._predict_one(img_path, output_dir, label)
        print(f"[INFO] Saved segmentation files")
