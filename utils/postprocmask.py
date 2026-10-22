import numpy as np
import cv2


def zoom_out_mask(mask: np.ndarray, scale: float = 0.98) -> np.ndarray:
    """Shrink `mask` about its centre by `scale` (nearest neighbour), keeping its size: trims a
    thin border so a mask touching the crop edge no longer does."""
    h, w = mask.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize the mask to new dimensions
    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Create a black canvas of original size
    zoomed_mask = np.zeros_like(mask)

    # Compute top-left corner to paste resized mask
    x_start = (w - new_w) // 2
    y_start = (h - new_h) // 2

    # Paste resized mask into the center
    zoomed_mask[y_start:y_start+new_h, x_start:x_start+new_w] = resized

    return zoomed_mask
