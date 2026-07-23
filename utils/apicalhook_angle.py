
import os
import cv2
import ast
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

class ApicalHook:
    def __init__(self, img_name, time_series_contours, germ_time, hypo_mask, seedling_points, point_ids, image):
        self.img_name = img_name
        self.image = image
        self.point_ids = point_ids
        self.seedling_points = seedling_points
        self.cotyl_contours = time_series_contours
        self.germ_time_series = germ_time
        self.crop_size = image.shape[1]
        self.search_radius = round(self.crop_size*0.016)
        self.matcher = RegionMatcher(time_series_contours, hypo_mask, seedling_points, point_ids, search_radius=self.search_radius)
        self.angle_calc = AngleCalculator()
        self.visualizer = ApicalVisualizer(image)

        self.matches = []
        self.angles = {}

    def process(self):
       
        self.matches = self.matcher.match()

        print(f"Matched {len(self.matches)} cotyledon-hypocotyl pairs")

        for match in self.matches:
            try:
                angle_diff_2, bio_state = self.angle_calc.compute_biological_angle(match.cotyl_ellipse, match.stem_ellipse)
            except (ValueError, TypeError) as e:
                print(f"[WARNING] Skipping match due to ellipse error: {e}")
                continue
            print(f"Angle = {angle_diff_2:.2f}°, Type = {bio_state}")
            self.angles[match.stem_id] = angle_diff_2
            self.visualizer.draw_match(match, angle_diff_2)
            # Draw all cotyls and seedling points
            self.visualizer.draw_all_cotyls_and_seed_ids(
                cotyl_contours=self.cotyl_contours[-1],
                seedling_points=self.seedling_points,
                seedling_ids=self.point_ids
            )

    def save(self, out_path):
        self.visualizer.save(out_path)

    def get_angles(self):
        return self.angles

@dataclass
class EllipseMatch:
    stem_id: int
    cotyl_id: int
    cotyl_center: Tuple[int, int]
    cotyl_ellipse: Tuple
    stem_ellipse: Tuple


class RegionMatcher:
    def __init__(self, time_series_cotyl_contours: list,
                 hypo_mask: np.ndarray,
                 seedling_points: list[Tuple[int, int]],
                 point_ids: list[int],
                 search_radius: int = 10):
        self.cotyl_contours = time_series_cotyl_contours[-1]
        self.hypo_mask = hypo_mask
        self.seedling_points = seedling_points
        self.point_ids = point_ids
        self.radius = search_radius

    def _rectangle(self, cx, cy, r):
        x1, y1 = max(cx - r, 0), max(cy - 3*r, 0)
        x2, y2 = min(cx + r, self.hypo_mask.shape[1]), min(cy + 3*r, self.hypo_mask.shape[0])
        return x1, y1, x2, y2

    @staticmethod
    def _assign_ids_by_order(seed_points, seed_ids, cotyl_centers):
        # Match closest by x-position (not assuming exact count match)
        sorted_seeds = sorted(zip(seed_points, seed_ids), key=lambda t: t[0][0])  # seed x
        sorted_cotyls = sorted(enumerate(cotyl_centers), key=lambda t: t[1][0])   # cotyl x

        assigned = {}
        for (cotyl_idx, _), (_, seed_id) in zip(sorted_cotyls, sorted_seeds):
            assigned[cotyl_idx] = seed_id

        return assigned
    
    @staticmethod
    def _assign_ids_by_closest_x(seed_points, seed_ids, cotyl_centers):

        # Sort seed IDs and cotyls by x
        sorted_seeds = sorted(zip(seed_points, seed_ids), key=lambda t: t[0][0])
        sorted_cotyls = sorted(enumerate(cotyl_centers), key=lambda t: t[1][0])  # (index, (x, y))

        assignments = {}
        for (cotyl_idx, _), (_, seed_id) in zip(sorted_cotyls, sorted_seeds):
            assignments[cotyl_idx] = seed_id

        return assignments
    
    def match(self) -> list[EllipseMatch]:
        valid_cotyls = []
        valid_matches = []

        for cotyl_id, contour in enumerate(self.cotyl_contours):
            if len(contour) < 5:
                continue

            try:
                cotyl_ellipse = cv2.fitEllipse(contour)
            except cv2.error:
                continue

            cx, cy = map(int, cotyl_ellipse[0])
            x1, y1, x2, y2 = self._rectangle(cx, cy, self.radius)

            hypo_crop = self.hypo_mask[y1:y2, x1:x2]
            contours, _ = cv2.findContours(hypo_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = [cnt for cnt in contours if len(cnt) >= 5]

            if not valid:
                continue

            best_cnt = max(valid, key=cv2.contourArea)

            try:
                local_ellipse = cv2.fitEllipse(best_cnt)
            except cv2.error:
                continue

            (xc, yc), (MA, ma), angle = local_ellipse
            global_center = (xc + x1, yc + y1)
            stem_ellipse = (global_center, (MA, ma), angle)

            valid_cotyls.append((cx, cy))
            valid_matches.append({
                "cotyl_id": cotyl_id,
                "cotyl_center": (cx, cy),
                "cotyl_ellipse": cotyl_ellipse,
                "stem_ellipse": stem_ellipse
            })

        # Assign seed IDs based on closest X
        assignments_1 = self._assign_ids_by_order(self.seedling_points, self.point_ids, valid_cotyls)
        assignments_2 = self._assign_ids_by_closest_x(self.seedling_points, self.point_ids, valid_cotyls)

        # Step 2: Compare exact equality
        if assignments_1 == assignments_2:
            best_assignment = assignments_2
        else:
            # Step 3: Score each
            def score(assignment):
                matched_ids = set(assignment.values())
                return len(assignment) + len(matched_ids)

            score_1 = score(assignments_1)
            score_2 = score(assignments_2)

            if score_1 >= score_2:
                print(f"[INFO] Using closest-X assignment: score {score_2} >= {score_1}")
                best_assignment = assignments_2
            else:
                print(f"[INFO] Using order assignment: score {score_1} > {score_2}")
                best_assignment = assignments_1

        # Build final list
        final_matches = []
        for i, m in enumerate(valid_matches):
            seed_id = best_assignment.get(i, None)
            if seed_id is not None:
                m["stem_id"] = seed_id
                final_matches.append(EllipseMatch(**m))

        return final_matches

    
class AngleCalculator:
    @staticmethod
    def major_axis_points(center, length, angle_deg):
        if any(map(lambda x: np.isnan(x), [*center, length, angle_deg])):
            raise ValueError(f"Invalid ellipse parameters: center={center}, length={length}, angle={angle_deg}")
    
        theta = np.deg2rad(angle_deg-90)
        dx = (length / 2) * np.cos(theta)
        dy = (length / 2) * np.sin(theta)
        pt1 = (center[0] - dx, center[1] - dy)
        pt2 = (center[0] + dx, center[1] + dy)
        return pt1, pt2, np.array(pt2) - np.array(pt1)
    
    @staticmethod
    def angle_between_vectors(vec1, vec2, in_degrees=True):
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)

        # Clip the cosine to the valid domain [-1, 1] to avoid numerical errors
        cos_theta = np.clip(dot_product / norm_product, -1.0, 1.0)

        angle_rad = np.arccos(cos_theta)

        if in_degrees:
            return np.degrees(angle_rad)
        else:
            return angle_rad
    
    def compute_biological_angle(self, cotyl_ellipse, stem_ellipse):

        (xc, yc), (Lc, _), angle_c = cotyl_ellipse
        (xs, ys), (Ls, _), angle_s = stem_ellipse

        # 1. Major axis points
        c_pt1, c_pt2, _ = self.major_axis_points((xc, yc), Lc, angle_c)
        s_pt1, s_pt2, _ = self.major_axis_points((xs, ys), Ls, angle_s)
        insert_y = self.intersection_point(s_pt1, s_pt2, c_pt1, c_pt2)

        # 2. Determine upper stem point
        s_pt1 = np.array(s_pt1)
        s_pt2 = np.array(s_pt2)
        c_pt1 = np.array(c_pt1)
        c_pt2 = np.array(c_pt2)

        stem_vec = s_pt2 - s_pt1
        upper_stem = s_pt2 if stem_vec[1] >= 0 else s_pt1
        lower_stem = s_pt1 if np.array_equal(upper_stem, s_pt2) else s_pt2
        stem_vec = upper_stem - lower_stem  # Ensure "upward"

        # 3. Closest cotyl point to upper stem
        d1 = np.linalg.norm(np.array(c_pt1) - upper_stem)
        d2 = np.linalg.norm(np.array(c_pt2) - upper_stem)
        cotyl_tip = c_pt1 if d1 >= d2 else c_pt2
        cotyl_base = c_pt1 if d1 < d2 else c_pt2
        # 4. Insertion vector (cotyl → upper stem)
        cotyl_vec = np.array(cotyl_tip) - np.array(cotyl_base)

        # 5. Reference angle
        angle = abs(self.angle_between_vectors(stem_vec, cotyl_vec))

        if insert_y[1] < min(yc, ys):
            bio_state = "Closed"
            angle = 180-angle
        elif insert_y[1] > max(yc, ys):
            bio_state = "Overhooked"
            angle -= 180
        else:
            bio_state = "Open"

        return angle, bio_state

    @staticmethod
    def intersection_point(pt1, pt2, pt3, pt4):
        def line(p1, p2):
            A = p2[1] - p1[1]
            B = p1[0] - p2[0]
            C = A * p1[0] + B * p1[1]
            return A, B, C
        
        A1, B1, C1 = line(pt1, pt2)
        A2, B2, C2 = line(pt3, pt4)
        det = A1 * B2 - A2 * B1
        if det == 0:
            return ((pt1[0] + pt3[0]) / 2, (pt1[1] + pt3[1]) / 2)
        x = (B2 * C1 - B1 * C2) / det
        y = (A1 * C2 - A2 * C1) / det
        return (x, y)

class AngleDictHandler:
    def __init__(self, csv_path="img_angle_data.csv"):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

        # Set index to filename (strip whitespace just in case)
        self.df.set_index('filename', inplace=True)
        self.df.index = self.df.index.map(lambda x: str(x).strip())

        # Parse stringified lists
        self.df['angles'] = self.df['angles'].apply(self._parse_angle_list)
        self.df['tot_numb'] = self.df['tot_numb'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    def _parse_angle_list(self, val):
        try:
            result = ast.literal_eval(val) if isinstance(val, str) else val
            return [float(v) if isinstance(v, (int, float, np.integer, np.floating)) else '-' for v in result]
        except Exception as e:
            print(f"[WARNING] Failed to parse angle list: {val} ({e})")
            return []

    def get_angles_for_image(self, img_name, seedling_ids):
        """Return list of (seedling_id, angle) for given image."""
        img_name = str(img_name).strip()
        if img_name not in self.df.index:
            print(f"[WARNING] Image '{img_name}' not found in angle data.")
            return [(sid, '-') for sid in seedling_ids]

        row = self.df.loc[img_name]
        angle_list = row['angles']
        tot_numb = row['tot_numb']

        angle_map = dict(zip(tot_numb, angle_list))
        result = []
        for sid in seedling_ids:
            angle = angle_map.get(sid, '-')
            if isinstance(angle, float) and np.isnan(angle):
                result.append((sid, '-'))
            else:
                result.append((sid, round(angle) if isinstance(angle, (float, int)) else '-'))
        return result

    def set_angle(self, img_name, seedling_id, new_angle):
        """Set new angle for seedling ID in image (by list position)."""
        img_name = str(img_name).strip()
        if img_name not in self.df.index:
            raise ValueError(f"Image '{img_name}' not in data.")

        row = self.df.loc[img_name]
        angles = row['angles']
        tot_numb = row['tot_numb']

        try:
            idx = tot_numb.index(seedling_id)
        except ValueError:
            raise ValueError(f"Seedling ID {seedling_id} not found in tot_numb for image '{img_name}'")

        angles[idx] = float(new_angle)
        self.df.at[img_name, 'angles'] = angles

    def blank_angle(self, img_name, seedling_id):
        """Set angle for seedling ID to NaN."""
        self.set_angle(img_name, seedling_id, np.nan)

    def save(self):
        """Save the DataFrame back to CSV with stringified angle lists."""
        df_copy = self.df.copy()
        df_copy['angles'] = df_copy['angles'].apply(str)
        df_copy['tot_numb'] = df_copy['tot_numb'].apply(str)
        df_copy.reset_index().to_csv(self.csv_path, index=False)
    
    
def mask_below_seed_line(image_shape: tuple, seed_points: list[tuple[int, int]]) -> np.ndarray:
    """
    Returns a binary mask that masks out all areas *below* the line defined by seed points.
    """
    # Fit a smooth line through seed points
    seed_points = np.array(seed_points, dtype=np.int32)
    seed_curve = cv2.fitLine(seed_points, cv2.DIST_L2, 0, 0.01, 0.01)
    
    # Get points along that line to define the upper boundary
    vx, vy, x0, y0 = seed_curve.flatten()
    height, width = image_shape

    # Create line endpoints across the width of the image
    x_vals = np.linspace(0, width - 1, num=width)
    y_vals = (vy / vx) * (x_vals - x0) + y0
    line_pts = np.stack([x_vals, y_vals], axis=-1).astype(np.int32)

    # Construct polygon below line (line + bottom of image)
    bottom = np.array([[width - 1, height - 1], [0, height - 1]], dtype=np.int32)
    full_poly = np.concatenate([line_pts, bottom])

    # Fill the mask
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [full_poly], 255)

    return mask


class ApicalVisualizer:
    def __init__(self, image: np.ndarray):
        self.image = image.copy()

    def draw_cotyl_history(self, time_series_contours: list, current_frame_idx: int):
        if len(self.image.shape) == 2 or self.image.shape[2] == 1:
            canvas = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        else:
            canvas = self.image

        n = min(current_frame_idx + 1, len(time_series_contours))

        for t in range(n):
            contours = time_series_contours[t]
            if not contours:
                continue

        # Fade older frames (lighter green)
            alpha = (t + 1) / n  # 0.1 to 1.0
            color = (0, int(255 * alpha), 0)  # from (0,25,0) to (0,255,0)

            cv2.drawContours(canvas, contours, -1, color, 1)

        self.image = canvas

    def draw_all_cotyls_and_seed_ids(self,
                                     cotyl_contours: list[np.ndarray],
                                     seedling_points: list[Tuple[int, int]],
                                     seedling_ids: list[int]):
        """
        Draw all cotyledon contours and seedling points with their IDs.
        """
        canvas = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR) if len(self.image.shape) == 2 else self.image

        # --- Draw all cotyledon contours in green ---
        cv2.drawContours(canvas, cotyl_contours, -1, (0, 255, 0), 2)

        # --- Draw seedling points and label IDs ---
        for (x, y), sid in zip(seedling_points, seedling_ids):
            cv2.circle(canvas, (x, y), 5, (255, 0, 0), -1)  # Blue dot
            cv2.putText(canvas, f"ID: {sid}", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self.image = canvas

    def draw_match(self, match: EllipseMatch, angle: Optional[float] = None):
        """
        Draw cotyledon and hypocotyl ellipses' major axes, seedling ID, and angle (if provided).
        """
        # Ensure color image
        if len(self.image.shape) == 2 or self.image.shape[2] == 1:
            canvas = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        else:
            canvas = self.image

        # ---- Draw cotyledon major axis (light orange) ----
        (xc, yc), (L1_c, _), angle_c = match.cotyl_ellipse
        theta_c = np.deg2rad(angle_c-90)
        dx_c = (L1_c) * np.cos(theta_c)
        dy_c = (L1_c) * np.sin(theta_c)
        pt1_c = (int(xc - dx_c), int(yc - dy_c))
        pt2_c = (int(xc + dx_c), int(yc + dy_c))
        cv2.line(canvas, pt1_c, pt2_c, (215, 238, 94), 2)

        intersection_point = None

        # ---- Draw hypocotyl major axis (dark orange) ----
        if match.stem_ellipse:
            (xs, ys), (L1_s, _), angle_s = match.stem_ellipse
            theta_s = np.deg2rad(angle_s-90)
            dx_s = (L1_s) * np.cos(theta_s)
            dy_s = (L1_s) * np.sin(theta_s)
            pt1_s = (int(xs - dx_s), int(ys - dy_s))
            pt2_s = (int(xs + dx_s), int(ys + dy_s))
            cv2.line(canvas, pt1_s, pt2_s, (206, 113, 106), 2)

            # ---- Compute intersection point ----
            intersection_point = self._compute_intersection(pt1_c, pt2_c, pt1_s, pt2_s)
            if intersection_point:
                cv2.circle(canvas, intersection_point, 5, (201, 30, 76), -1)

        # ---- Draw seedling ID ----
        cv2.putText(canvas, f"ID: {match.stem_id}", (int(xc), int(yc - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (113, 131, 168), 1)

        # ---- Draw angle if provided ----
        if angle is not None:
            cv2.putText(canvas, f"{angle:.1f}", (int(xc+10), int(yc + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        self.image = canvas  # Store updated canvas

    def _compute_intersection(self, a1: Tuple[int, int], a2: Tuple[int, int],
                               b1: Tuple[int, int], b2: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Return intersection point of lines (a1, a2) and (b1, b2) if they intersect.
        """
        def to_float(p): return np.array(p, dtype=np.float32)
        
        a1, a2, b1, b2 = map(to_float, [a1, a2, b1, b2])

        da = a2 - a1
        db = b2 - b1
        dp = a1 - b1

        dap = np.array([-da[1], da[0]])
        denom = np.dot(dap, db)
        if denom == 0:
            return None  # Parallel lines

        num = np.dot(dap, dp)
        intersection = (num / denom) * db + b1
        return tuple(np.round(intersection).astype(int))
        

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, self.image)
