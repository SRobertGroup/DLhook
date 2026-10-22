import pytest

from utils.apicalhook_angle import AngleCalculator

calc = AngleCalculator()


# Case 1: stem axis vertical (x=100, y from 225 to 375), cotyledon axis
# horizontal (y=220, x from 90 to 170), crossing at a right angle away from
# the seed/apex ends -- a plain, unambiguous "Open" configuration with hand-
# checkable numbers (see PR description / commit for the by-hand derivation):
# insert_y=(100, 220), seed=(100, 375), apex=(170, 220), angle=90, state=Open.
STEM_OPEN = ((100.0, 300.0), (150.0, 20.0), 0.0)
COTYL_OPEN = ((130.0, 220.0), (80.0, 15.0), 90.0)

# Case 2: identical stem, cotyledon shifted down slightly (center y 220->235)
# so its farther (apex) endpoint dips below the stem's near endpoint --
# triggers the Overhooked branch and the "360 - angle" formula.
STEM_OVERHOOK = STEM_OPEN
COTYL_OVERHOOK = ((130.0, 235.0), (80.0, 15.0), 90.0)


def test_compute_biological_angle_open_case_matches_hand_derivation():
    angle, state = calc.compute_biological_angle(COTYL_OPEN, STEM_OPEN)

    assert state == "Open"
    assert angle == pytest.approx(90.0, abs=1e-6)


def test_compute_biological_angle_overhooked_case_matches_hand_derivation():
    angle, state = calc.compute_biological_angle(COTYL_OVERHOOK, STEM_OVERHOOK)

    assert state == "Overhooked"
    # 360 - angle_between(...) with the vectors still at 90 degrees apart.
    assert angle == pytest.approx(270.0, abs=1e-6)


def test_major_axis_points_raises_value_error_on_nan_ellipse_params():
    """major_axis_points rejects NaN ellipse parameters (callers such as
    ApicalHook.process/compute_biological_angle rely on catching
    ValueError/TypeError around this code for degenerate contours)."""
    with pytest.raises(ValueError):
        AngleCalculator.major_axis_points((float("nan"), 0.0), 10.0, 0.0)
