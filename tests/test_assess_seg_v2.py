"""multi/assess_seg_v2.py metrics on synthetic masks (no models)."""
import math

import numpy as np

from multi.assess_seg_v2 import dice, junction_gap, n_parts, segment_coverage, tricky_score


def _hypocotyl(top=20):
    m = np.zeros((100, 40), np.uint8)
    m[top:90, 18:22] = 255
    return m


def test_junction_gap_and_coverage():
    m = _hypocotyl(top=20)
    assert junction_gap(m, (20, 25)) == 0.0
    assert abs(junction_gap(m, (20, 10)) - 10) < 0.5               # the mask starts 10 px below the junction
    assert math.isinf(junction_gap(np.zeros_like(m), (5, 5)))
    cov = segment_coverage(m, (20, 10), (20, 50))
    assert 0.65 < cov < 0.85                                        # the top quarter is missing


def test_parts_dice_and_score():
    m = _hypocotyl()
    split = m.copy()
    split[50:54] = 0
    assert n_parts(m) == 1 and n_parts(split) == 2
    assert dice(m, m) == 1.0 and math.isnan(dice(np.zeros(3), np.zeros(3)))
    score, reason = tricky_score({"gap_student": 12.0, "gap_teacher": 2.0, "parts_student": 2,
                                  "dice_hypo": 0.5, "dice_cot": 0.9})
    assert score > 0 and "junction gap 12 px" in reason and "2 pieces" in reason and "hypocotyl disagrees" in reason
    assert tricky_score({"gap_student": 1.0, "parts_student": 1, "dice_hypo": 0.95, "dice_cot": 0.9}) == (0.0, "")


def test_stroke_confusion_counts_painted_pixels_only():
    from multi.assess_seg_v2 import stroke_confusion
    from utils.seg_strokes import UNLABELLED

    strokes = np.full((10, 10), UNLABELLED, np.uint8)
    strokes[0:5, 0] = 2                    # 5 hypocotyl stroke px
    strokes[0:5, 1] = 0                    # 5 background stroke px
    pred = np.zeros((10, 10), np.uint8)
    pred[0:3, 0] = 2                       # 3 of the hypocotyl found
    pred[0:2, 1] = 2                       # 2 background called hypocotyl
    pred[8:, 8:] = 2                       # unpainted: ignored
    assert stroke_confusion(pred, strokes)["hypocotyl"] == (3, 2, 2)
    assert stroke_confusion(pred, strokes)["cotyledon"] == (0, 0, 0)
