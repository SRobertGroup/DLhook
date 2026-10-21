"""multi/build_seg_v2_dataset.py: the ignore band between a clicked junction and the teacher hypocotyl."""
import numpy as np

from multi.build_seg_v2_dataset import IGNORE, ignore_band


def test_ignore_band_covers_the_hook_gap_only():
    label = np.zeros((100, 40), np.uint8)
    label[40:90, 18:22] = 2                     # teacher hypocotyl starts 25 px below the junction
    label[10:16, 24:30] = 1                     # a cotyledon next to the band
    before = label.copy()
    n = ignore_band(label, (20, 15))
    assert n > 0 and (label[18:38, 20] == IGNORE).all()            # the gap is ignored
    assert (label[before == 2] == 2).all() and (label[before == 1] == 1).all()   # tissue untouched
    assert (label[60:, :10] == 0).all()                            # far background unchanged


def test_ignore_band_skips_small_gaps_and_empty_masks():
    label = np.zeros((50, 30), np.uint8)
    assert ignore_band(label, (10, 10)) == 0
    label[11:40, 9:12] = 2
    assert ignore_band(label, (10, 10)) == 0                       # junction touches the mask
