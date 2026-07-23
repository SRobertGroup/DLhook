import os
import numpy as np
import cv2

"""
This class takes lists of tuples as input containing the seedling starting points (x,y) and returns a
a list contating in the same structure as the input which gives each point a specific nunber/id
"""
class point_num():
    def __init__(self,list1):
        self.list1=list1
    def point_numbering(self):
        numb=[]
        
        n=1
        for crop in self.list1:
            
            n_x=len(crop)
            n_x=n+n_x
            list_x=[n for n in range(n,n_x)]
            n = n_x
            numb.append(list_x)
        return numb
                    

"""
This class postprocess the predicted masks
1: Takes away the lower cotyledon - remove cotyl under the set starting point by the user.
2: Removes hypocotyl which are located above the highest cotyledon in the image
"""
class PostprocessMasks:
    # TODO(reforge): revived by mask-editing feature
    def __init__(self, seed_coat_points, image_shape):
        for idx, points in enumerate(seed_coat_points):
            if not points:
                continue
            img_paths = self._get_image_paths(idx)
            self._process_mask_pair(img_paths, points, image_shape)

    def zoom_out_mask(mask: np.ndarray, scale: float = 0.98) -> np.ndarray:
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

    def _get_image_paths(self, id_n):
        path = 'data/predict/'
        cotyl_paths = [os.path.join(path, f) for f in os.listdir(path)
                       if f.startswith(f"{id_n}-crop") and f.endswith("1.png")]
        hypo_paths = [os.path.join(path, f) for f in os.listdir(path)
                      if f.startswith(f"{id_n}-crop") and f.endswith("2.png")]

        print(f"[DEBUG] For crop ID {id_n}, found:")

        return cotyl_paths, hypo_paths

    def _create_mask_region(self, seed_points, image_shape):
        """Extend the seed point polygon to cover the lower image region."""
        height, width = image_shape
        mask = seed_points[:]
        mask.append([width - 1, seed_points[-1][1]])  # lower-right
        mask.append([width - 1, height - 1])          # bottom-right
        mask.append([0, height - 1])                  # bottom-left
        mask.append([0, seed_points[0][1]])           # upper-left
        return np.array(mask, dtype=np.int32)

    def _process_mask_pair(self, image_paths, seed_points):
        cotyl_paths, hypo_paths = image_paths
        output_dir = 'data/postprocess/'
        os.makedirs(output_dir, exist_ok=True)

        for cotyl_path, hypo_path in zip(cotyl_paths, hypo_paths):
            img_hypo = cv2.imread(hypo_path, cv2.IMREAD_GRAYSCALE)
            img_cotyl = cv2.imread(cotyl_path, cv2.IMREAD_GRAYSCALE)
            if img_hypo is None or img_cotyl is None:
                print(f"[WARNING] Could not read one of: {cotyl_path}, {hypo_path}")
                continue

            mask_region = self._create_mask_region(seed_points.copy(), img_hypo.shape)

            hypo_processed = self._postprocess_hypocotyl(img_hypo.copy(), mask_region)
            cotyl_processed = self._postprocess_cotyledon(img_cotyl.copy(), mask_region)

            hypo_name = os.path.basename(hypo_path)
            cotyl_name = os.path.basename(cotyl_path)
            cv2.imwrite(os.path.join(output_dir, hypo_name), hypo_processed)
            cv2.imwrite(os.path.join(output_dir, cotyl_name), cotyl_processed)

    def _postprocess_cotyledon(self, mask, mask_region):
        mask_filled = cv2.fillConvexPoly(mask, mask_region, 255)
        mask_filled = cv2.bitwise_not(mask_filled)

        kernel = np.ones((3, 3), np.uint8)
        result = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, kernel, iterations=3)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=3)
        result = cv2.erode(result, kernel, iterations=2)
        result = cv2.dilate(result, kernel, iterations=2)
        return cv2.bitwise_not(result)

    def _postprocess_hypocotyl(self, mask, mask_region):
        height = mask.shape[0]
        mask_filled = cv2.fillConvexPoly(mask, mask_region, 255)

        # Locate highest cotyledon contour to crop above
        _, binary = cv2.threshold(mask_filled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        y_min = height
        for c in contours:
            if len(c) >= 5:
                try:
                    ellipse = cv2.fitEllipse(c)
                    y_min = min(y_min, int(ellipse[0][1]))
                except cv2.error:
                    continue

        # Clear region above cotyledon
        mask_filled[0:y_min, :] = 255
        mask_filled = cv2.fillConvexPoly(mask_filled, mask_region, 255)
        mask_filled = cv2.bitwise_not(mask_filled)

        kernel = np.ones((3, 3), np.uint8)
        mask_filled = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, kernel, iterations=3)
        mask_filled = cv2.morphologyEx(mask_filled, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_filled = cv2.erode(mask_filled, kernel, iterations=2)
        mask_filled = cv2.dilate(mask_filled, kernel, iterations=2)
        return cv2.bitwise_not(mask_filled)
