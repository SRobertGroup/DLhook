import math

import numpy as np
import pytest

from utils.hook_kinetics import TRAIT_COLUMNS, curve, fit_kinetics, seedling_points, trait_row

T = np.arange(8.0, 109.0, 1.0)            # hours after germination, one frame per hour (the user's typical curves)
DOUBLE = {"a": 100.0, "p": 180.0, "b": 2.0, "t1": 15.0, "t2": 62.0, "k1": 0.25, "k2": 0.12}


def _noisy(model, params, seed=0, sd=4.0, t=T):
    rng = np.random.default_rng(seed)
    return curve(model, params, t) + rng.normal(0, sd, len(t))


def test_double_sigmoid_is_chosen_and_recovers_the_phases():
    fit = fit_kinetics(T, _noisy("double", DOUBLE))
    assert fit["model"] == "double" and fit["n"] == len(T)
    tr = fit["traits"]
    assert tr["closing_mid"] == pytest.approx(15, abs=2) and tr["opening_mid"] == pytest.approx(62, abs=2)
    assert tr["opening_duration"] == pytest.approx(2 * math.log(9) / 0.12, rel=0.15)
    assert tr["maintenance_duration"] > 10 and tr["peak_angle"] > 170
    assert tr["extrapolated"] == "closing_start" and fit["rmse"] < 6     # closing began before the first frame (8 h)


def test_overlapping_phases_give_a_negative_maintenance_and_a_peak_below_the_plateau():
    """The early-opening line of the user's figure: peaks near 155 deg and opens at once."""
    params = {"a": 110.0, "p": 185.0, "b": 3.0, "t1": 18.0, "t2": 34.0, "k1": 0.15, "k2": 0.1}
    fit = fit_kinetics(T, _noisy("double", params, seed=1, sd=3.0))
    assert fit["model"] == "double"
    tr = fit["traits"]
    assert tr["maintenance_duration"] < 0
    assert tr["peak_angle"] < fit["params"]["p"] - 5


def test_an_unfinished_opening_is_flagged_as_extrapolated():
    """The black line of the user's figure: still opening at the last frame."""
    params = dict(DOUBLE, t2=95.0, k2=0.08, b=20.0)
    fit = fit_kinetics(T, _noisy("double", params, seed=2))
    assert fit["model"] == "double" and "opening_end" in fit["traits"]["extrapolated"]


def test_first_seen_closed_picks_opening_only_and_no_transition_picks_flat():
    t = np.arange(30.0, 109.0, 1.0)
    fit = fit_kinetics(t, _noisy("opening", {"p": 178.0, "b": 5.0, "t2": 65.0, "k2": 0.12}, seed=3, t=t))
    assert fit["model"] == "opening" and "closing_mid" not in fit["traits"]
    flat = fit_kinetics(T, _noisy("flat", {"p": 176.0}, seed=4))
    assert flat["model"] == "flat" and flat["params"]["p"] == pytest.approx(176, abs=2)


def test_closing_only_is_chosen_for_a_hook_that_never_opens():
    fit = fit_kinetics(T, _noisy("closing", {"a": 95.0, "p": 179.0, "t1": 20.0, "k1": 0.2}, seed=5))
    assert fit["model"] == "closing" and "opening_mid" not in fit["traits"]


def test_a_mirrored_run_barely_moves_the_robust_fit():
    y = _noisy("double", DOUBLE, seed=6)
    clean = fit_kinetics(T, y)["traits"]["opening_mid"]
    y[20:24] = 180 - y[20:24]                                  # four mirrored readings of a closed hook
    assert fit_kinetics(T, y)["traits"]["opening_mid"] == pytest.approx(clean, abs=2)


def test_forced_model_and_too_few_frames():
    y = _noisy("double", DOUBLE, seed=7)
    assert fit_kinetics(T, y, model="opening")["model"] == "opening"
    assert fit_kinetics(T, y, model="opening")["forced"] is True
    assert fit_kinetics([1.0], [170.0]) is None
    short = fit_kinetics([0.0, 1.0, 2.0, 3.0], [170, 172, 171, 170])
    assert short["model"] == "flat" and set(short["bic"]) == {"flat"}            # only flat has enough frames


def test_levels_stay_within_bounds():
    fit = fit_kinetics(T, _noisy("double", dict(DOUBLE, b=-20.0), seed=8))
    assert all(0 <= fit["params"][k] <= 270 for k in ("a", "p", "b") if k in fit["params"])


def test_seedling_points_start_at_germination_and_weight_manual_frames():
    results = [{"angle_dict": {1: 100.0 + i}, "state_dict": {1: "Manual" if i == 4 else "Closed"}} for i in range(6)]
    results[3]["angle_dict"][1] = float("nan")
    results[5] = None
    minutes = [0, 60, 120, 180, 240, 300]
    t, y, w, idx = seedling_points(results, 1, minutes, time_zero=2)
    assert list(idx) == [2, 4] and list(t) == [0.0, 2.0] and list(w) == [1.0, 2.0]
    t, _, _, idx = seedling_points(results, 1, None, time_zero=None)        # no time axis: frames from the first
    assert list(t) == [0.0, 1.0, 2.0, 4.0] and idx == [0, 1, 2, 4]


def test_trait_row_has_every_column():
    fit = fit_kinetics(T, _noisy("double", DOUBLE, seed=9))
    row = trait_row(3, fit, "h", 12)
    assert set(row) == set(TRAIT_COLUMNS) and row["model"] == "double" and row["time_zero_frame"] == 12
    assert trait_row(4, None, "h", None)["model"] == "none"


def test_gui_fit_is_cached_follows_a_forced_model_and_feeds_the_export():
    from types import SimpleNamespace
    from ui.gui.export import ExportMixin

    class FakeGui(ExportMixin):
        pass

    n, germ = 110, 8
    t = np.arange(n, dtype=float) - germ                       # one frame per hour, germination at frame 8
    y = curve("double", dict(DOUBLE, t1=10.0, t2=55.0), t) + np.random.default_rng(10).normal(0, 3, n)
    results = [{"angle_dict": {1: float(v)}, "state_dict": {1: "Closed"}} for v in y]
    g = FakeGui()
    g.frame_results_by_crop = {0: results}
    g.file_list = [f"f{i}.jpg" for i in range(n)]
    g.time_deltas = [{"elapsed_minutes": 60.0 * i} for i in range(n)]
    g.germination_detector = SimpleNamespace(get_time_zero=lambda crop_id: germ)

    fit, axis, unit, time_zero = g.kinetics_for_crop(0)
    assert fit["model"] == "double" and unit == "h" and time_zero == germ and axis[germ] == 0
    assert fit["n"] == n - germ                                 # nothing before germination
    assert g.kinetics_for_crop(0)[0] is fit                     # cached
    angles = g._fit_angles(0)
    assert min(angles) == germ and max(angles) == n - 1
    assert angles[60] == pytest.approx(curve("double", fit["params"], [60 - germ])[0])

    g.kinetics_models = {0: "flat"}                             # forced in the analysis window
    assert g.kinetics_for_crop(0)[0]["model"] == "flat"
    table = g._kinetics_table()
    assert list(table["model"]) == ["flat"] and table["forced"][0] == 1 and table["time_unit"][0] == "h"

    g.time_deltas = []                                          # no time axis: frames since germination
    assert g.kinetics_for_crop(0)[2] == "frame"
