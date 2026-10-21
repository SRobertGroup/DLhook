import math

import numpy as np
import pytest

from utils.hypocotyl_traits import DIAMETER_ERODE_PX, hypocotyl_midline, hypocotyl_traits, to_mm


def _bar(h=120, w=60, x0=25, x1=35, y0=10, y1=100):
    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = 255
    return m


def test_straight_bar_diameter_and_length():
    t = hypocotyl_traits(_bar(), junction=(30, 10))
    assert t["diameter_px"] == pytest.approx(10 - 2 * DIAMETER_ERODE_PX, abs=1)   # the mask overreach correction
    assert t["length_px"] == pytest.approx(90, abs=3)
    assert t["n_transects"] >= 8
    assert t["curvature_mean"] < 0.01


def test_midline_oriented_from_junction():
    mid = hypocotyl_midline(_bar(), junction=(30, 100))
    assert mid[0][1] > mid[-1][1]
    mid = hypocotyl_midline(_bar(), junction=(30, 10))
    assert mid[0][1] < mid[-1][1]


def test_arc_curvature():
    R, th = 40.0, 10.0
    size = 130
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(xx - 20, yy - 20)
    ang = np.degrees(np.arctan2(yy - 20, xx - 20))
    m = (((r >= R - th / 2) & (r <= R + th / 2)) & (ang >= 5) & (ang <= 85)).astype(np.uint8) * 255
    j = (20 + R * math.cos(math.radians(5)), 20 + R * math.sin(math.radians(5)))
    t = hypocotyl_traits(m, junction=j)
    assert t["curvature_mean"] == pytest.approx(1 / R, rel=0.2)
    assert t["diameter_px"] == pytest.approx(th - 2 * DIAMETER_ERODE_PX, abs=1.5)
    assert t["length_px"] == pytest.approx(R * math.radians(80), rel=0.15)


def test_collar_cut_shortens_length():
    full = hypocotyl_traits(_bar(), junction=(30, 10))
    cut = hypocotyl_traits(_bar(), junction=(30, 10), collar=(30, 50))
    assert cut["length_px"] == pytest.approx(40, abs=3)
    assert cut["length_px"] < full["length_px"] - 30


def test_truncated_flag():
    touching = _bar(y1=120)
    assert hypocotyl_traits(touching, junction=(30, 10))["length_truncated"] is True
    assert hypocotyl_traits(touching, junction=(30, 10), collar=(30, 60))["length_truncated"] is False
    assert hypocotyl_traits(_bar(), junction=(30, 10))["length_truncated"] is False


def test_empty_and_missing_inputs_are_nan():
    for t in (hypocotyl_traits(np.zeros((50, 50), np.uint8), (5, 5)),
              hypocotyl_traits(None, (5, 5)),
              hypocotyl_traits(_bar(), None)):
        assert math.isnan(t["length_px"]) and math.isnan(t["diameter_px"])
        assert math.isnan(t["curvature_mean"]) and t["length_truncated"] is False and t["n_transects"] == 0
    assert hypocotyl_midline(np.zeros((9, 9), np.uint8), (1, 1)) is None


def test_to_mm():
    t = {"length_px": 100.0, "diameter_px": 10.0, "curvature_mean": 0.02, "curvature_max": 0.05}
    mm = to_mm(t, 10.0)   # 10 um/px = 0.01 mm/px
    assert mm["length_mm"] == pytest.approx(1.0)
    assert mm["diameter_mm"] == pytest.approx(0.1)
    assert mm["curvature_mean_per_mm"] == pytest.approx(2.0)
    assert mm["curvature_max_per_mm"] == pytest.approx(5.0)
    assert to_mm(t, None) == {}
    nan_mm = to_mm({"length_px": float("nan")}, 10.0)
    assert math.isnan(nan_mm["length_mm"]) and math.isnan(nan_mm["curvature_mean_per_mm"])


def test_split_hypocotyl_is_bridged_from_the_junction():
    whole = _bar()
    split = whole.copy()
    split[55:61, :] = 0                                     # a 6 px gap across the stem
    a = hypocotyl_traits(whole, (30, 10))["length_px"]
    b = hypocotyl_traits(split, (30, 10))["length_px"]
    assert b == pytest.approx(a, abs=4)                     # measured through the gap, not half the stem


def test_far_split_piece_is_not_joined():
    m = _bar()
    m[40:75, :] = 0                                         # 35 px gap: not one hypocotyl
    top = hypocotyl_traits(m, (30, 10))["length_px"]
    assert top < 40


def test_larger_neighbour_seedling_is_ignored():
    m = np.zeros((120, 120), np.uint8)
    m[10:60, 25:35] = 255                                   # this plant, near the junction
    m[5:115, 80:95] = 255                                   # a bigger neighbour further away
    t = hypocotyl_traits(m, (30, 10))
    mid = hypocotyl_midline(m, (30, 10))
    assert mid[:, 0].max() < 40
    assert t["length_px"] < 60


def test_without_junction_the_largest_component_is_used():
    from utils.hypocotyl_traits import _oriented_path
    m = np.zeros((120, 120), np.uint8)
    m[10:60, 25:35] = 255
    m[5:115, 80:95] = 255
    path, comp = _oriented_path(m, None)
    assert comp[50, 85] == 1 and comp[30, 30] == 0


def test_diameter_runs_on_the_eroded_mask_only(monkeypatch):
    import utils.hypocotyl_traits as ht
    eroded = hypocotyl_traits(_bar(), junction=(30, 10))
    monkeypatch.setattr(ht, "DIAMETER_ERODE_PX", 0)
    monkeypatch.setattr(ht._diameter_mask, "__defaults__", (0,))
    raw = hypocotyl_traits(_bar(), junction=(30, 10))
    assert raw["diameter_px"] - eroded["diameter_px"] == pytest.approx(2 * DIAMETER_ERODE_PX, abs=0.6)
    assert raw["length_px"] == eroded["length_px"]                      # the midline is unchanged
