"""Row building for the long-format results CSV (one row per seedling per frame) -- pure, no Tk.

The first seven columns are the original schema and keep their order and formatting; the landmark /
hypocotyl-trait columns (incl. unreadable_p / unreadable, the landmark readability gate) are appended after them, and the millimetre columns only when a scale is set.
Missing values are empty cells.
"""
from __future__ import annotations

import math

import numpy as np

from utils.hypocotyl_traits import to_mm

BASE_COLUMNS = ['img_name', 'seedling_id', 'raw_angle', 'state', 'bio_angle',
                'time_elapsed_min', 'germination_frame']
LANDMARK_COLUMNS = ['angle_source', 'landmark_edit', 'landmark_peak', 'overhook_flag',
                    'unreadable_p', 'unreadable',
                    'junction_x', 'junction_y', 'hypo_dir_x', 'hypo_dir_y', 'cotyl_dir_x', 'cotyl_dir_y',
                    'cotyl_base_x', 'cotyl_base_y', 'cotyl_tip_x', 'cotyl_tip_y', 'collar_x', 'collar_y',
                    'root_dir_x', 'root_dir_y']
TRAIT_COLUMNS = ['hypocotyl_length_px', 'hypocotyl_length_truncated', 'hypocotyl_curvature_mean',
                 'hypocotyl_curvature_max', 'hypocotyl_diameter_px']
MM_COLUMNS = ['hypocotyl_length_mm', 'hypocotyl_diameter_mm',
              'hypocotyl_curvature_mean_per_mm', 'hypocotyl_curvature_max_per_mm']

# readout key -> the two column names its (x, y) pair fills
_POINT_COLUMNS = [('junction', 'junction_x', 'junction_y'), ('hypo_dir', 'hypo_dir_x', 'hypo_dir_y'),
                  ('cotyl_dir', 'cotyl_dir_x', 'cotyl_dir_y'), ('base', 'cotyl_base_x', 'cotyl_base_y'),
                  ('tip', 'cotyl_tip_x', 'cotyl_tip_y'), ('collar', 'collar_x', 'collar_y'),
                  ('root_dir', 'root_dir_x', 'root_dir_y')]


def export_columns(um_per_px=None):
    """Column order of the export: original columns first, then landmarks and traits, then mm."""
    cols = BASE_COLUMNS + LANDMARK_COLUMNS + TRAIT_COLUMNS
    return cols + MM_COLUMNS if um_per_px else cols


def _is_num(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and not math.isnan(v)


def _cell(v, ndigits=4):
    return round(float(v), ndigits) if _is_num(v) else ''


def build_export_row(filename, seed_id, result, elapsed, germ_frame, um_per_px=None):
    """One CSV row (dict keyed by export_columns(um_per_px)) for a seedling in one frame.
    `result` is a process_single_frame result dict, or None for a frame with no result."""
    angle = result["angle_dict"].get(seed_id) if result else None
    raw = result.get("raw_angle_dict", {}).get(seed_id, angle) if result else None
    row = {
        'img_name': filename,
        'seedling_id': seed_id,
        'raw_angle': round(raw) if _is_num(raw) else '',
        'state': result.get("state_dict", {}).get(seed_id, '') if result else '',
        'bio_angle': round(angle) if _is_num(angle) else '',
        'time_elapsed_min': round(elapsed, 2) if _is_num(elapsed) else '',
        'germination_frame': germ_frame if germ_frame is not None else '',
    }
    row.update({c: '' for c in LANDMARK_COLUMNS + TRAIT_COLUMNS})
    if um_per_px:
        row.update({c: '' for c in MM_COLUMNS})
    if not result:
        return row

    row['angle_source'] = result.get("angle_source", '')
    readout = (result.get("landmarks") or {}).get(seed_id)
    if readout:
        row['landmark_edit'] = readout.get("edit", '')
        row['landmark_peak'] = _cell(readout.get("peak"))
        flag = readout.get("overhook")
        row['overhook_flag'] = '' if flag is None else int(bool(flag))
        row['unreadable_p'] = _cell(readout.get("unreadable_p"))
        gate = readout.get("gated", readout.get("unreadable"))     # "gated": the gate was on and fired
        row['unreadable'] = '' if gate is None else int(bool(gate))
        for key, cx, cy in _POINT_COLUMNS:
            pt = readout.get(key)
            if pt is not None and len(pt) >= 2:
                row[cx], row[cy] = _cell(pt[0]), _cell(pt[1])

    traits = (result.get("traits") or {}).get(seed_id)
    if traits:
        length = traits.get("length_px")
        row['hypocotyl_length_px'] = _cell(length, 2)
        row['hypocotyl_length_truncated'] = bool(traits.get("length_truncated")) if _is_num(length) else ''
        row['hypocotyl_curvature_mean'] = _cell(traits.get("curvature_mean"), 5)
        row['hypocotyl_curvature_max'] = _cell(traits.get("curvature_max"), 5)
        row['hypocotyl_diameter_px'] = _cell(traits.get("diameter_px"), 2)
        if um_per_px:
            mm = to_mm(traits, um_per_px)
            row['hypocotyl_length_mm'] = _cell(mm.get("length_mm"), 4)
            row['hypocotyl_diameter_mm'] = _cell(mm.get("diameter_mm"), 4)
            row['hypocotyl_curvature_mean_per_mm'] = _cell(mm.get("curvature_mean_per_mm"), 4)
            row['hypocotyl_curvature_max_per_mm'] = _cell(mm.get("curvature_max_per_mm"), 4)
    return row
