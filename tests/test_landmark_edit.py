"""utils/landmark_edit.py: click session, readout-shaped result, overlay shapes (no Tk)."""
from __future__ import annotations

import math

import pytest

from utils.angle_annotation import AngleClickSession, hook_angle, orient_axis
from utils.landmark_edit import (
    COTYL_COLOR, HYPO_COLOR, JUNCTION_COLOR, LandmarkClickSession, landmark_shapes,
)

J, H1, H2 = (30.0, 20.0), (30.0, 40.0), (30.0, 60.0)
OPEN = [(35.0, 18.0), (50.0, 10.0)]          # base, tip: theta ~118, rays diverge
CLOSED = [(34.0, 22.0), (34.0, 50.0)]        # folded back along the hypocotyl: parallel rays
CROSSING = [(34.0, 22.0), (28.0, 40.0)]      # tip swung past the stem: rays meet


def _session(cotyl, extra=()):
    s = LandmarkClickSession()
    for p in (J, H1, H2, *cotyl, *extra):
        s.add(*p)
    return s


def test_steps_and_prompts_follow_the_order():
    s = LandmarkClickSession()
    seen = []
    for p in (J, H1, H2, *OPEN, (31.0, 70.0), (33.0, 80.0)):
        seen.append(s.step)
        assert s.prompt()
        s.add(*p)
    assert seen == ["junction", "hypo_1", "hypo_2", "cotyl_base", "cotyl_tip", "collar", "root"]
    assert s.complete and s.step is None
    assert "junction" in LandmarkClickSession().prompt().lower()
    s = LandmarkClickSession()
    for p in (J, H1, H2):
        s.add(*p)
    assert "BASE" in s.prompt()
    s.add(*OPEN[0])
    assert "TIP" in s.prompt()
    s.add(*OPEN[1])
    assert "optional" in s.prompt()


def test_extra_clicks_are_ignored_once_complete():
    s = _session(OPEN, extra=[(1.0, 1.0), (2.0, 2.0)])
    assert len(s.points) == 7
    s.add(9.0, 9.0)
    assert len(s.points) == 7


def test_angle_complete_after_five_and_skip_optional_completes():
    s = _session(OPEN)
    assert s.angle_complete and not s.complete and s.step == "collar"
    s.skip_optional()
    assert s.complete and s.step is None
    s.add(5.0, 5.0)                      # ignored: skipped
    assert len(s.points) == 5


def test_skip_optional_needs_the_angle_points():
    s = LandmarkClickSession()
    s.add(*J)
    s.skip_optional()
    assert not s.complete and s.step == "hypo_1"


def test_skip_after_collar_keeps_the_collar_without_root():
    s = _session(OPEN, extra=[(31.0, 70.0)])
    s.skip_optional()
    assert s.complete
    r = s.result()
    assert r["collar"] == (31.0, 70.0) and r["root_dir"] is None and r["root_point"] is None


def test_undo_reset():
    s = _session(OPEN)
    s.undo()
    assert len(s.points) == 4 and not s.angle_complete
    s.skip_optional()                    # no effect below 5 points
    assert not s.skipped
    s = _session(OPEN)
    s.skip_optional()
    s.undo()
    assert not s.skipped and len(s.points) == 4
    s.reset()
    assert s.points == [] and s.step == "junction"
    LandmarkClickSession().undo()        # nothing to undo is fine


def test_result_needs_five_points():
    s = LandmarkClickSession()
    for p in (J, H1, H2, OPEN[0]):
        s.add(*p)
    with pytest.raises(ValueError):
        s.result()


def test_theta_equals_hook_angle_for_the_same_clicks():
    s = _session(OPEN)
    r = s.result(overhook=False)
    theta, bio = hook_angle(J, (H1, H2), tuple(OPEN), overhook=False)
    assert r["theta"] == pytest.approx(theta, abs=1e-6)
    assert r["bio"] == pytest.approx(bio, abs=1e-6)
    assert r["edit"] == "manual" and r["peak"] == 1.0
    assert r["junction"] == J
    assert r["hypo_dir"] == pytest.approx(tuple(orient_axis(H1, H2, J)))
    assert r["cotyl_dir"] == pytest.approx(tuple(orient_axis(OPEN[0], OPEN[1], J)))
    assert (r["base"], r["tip"]) == (OPEN[0], OPEN[1])
    assert (r["hypo1"], r["hypo2"], r["cotyl1"], r["cotyl2"]) == (H1, H2, OPEN[0], OPEN[1])


def test_overhook_auto_for_crossing_closed_and_open():
    assert _session(CROSSING).result()["overhook_auto"] is True
    assert _session(CLOSED).result()["overhook_auto"] is False
    assert _session(OPEN).result()["overhook_auto"] is False


def test_bio_follows_the_override_and_defaults_to_auto():
    s = _session(CROSSING)
    theta = s.result()["theta"]
    assert s.result()["overhook"] is True and s.result()["bio"] == pytest.approx(180 + theta)
    assert s.result(overhook=False)["bio"] == pytest.approx(180 - theta)
    s = _session(OPEN)
    theta = s.result()["theta"]
    assert s.result()["bio"] == pytest.approx(180 - theta)
    assert s.result(overhook=True)["bio"] == pytest.approx(180 + theta)
    assert s.result(overhook=True)["overhook_auto"] is False


def test_root_dir_is_a_unit_vector_from_collar_to_root():
    s = _session(OPEN, extra=[(30.0, 70.0), (30.0, 90.0)])
    r = s.result()
    assert r["collar"] == (30.0, 70.0)
    assert r["root_dir"] == pytest.approx((0.0, 1.0))
    assert math.hypot(*r["root_dir"]) == pytest.approx(1.0)
    assert r["root_point"] == (30.0, 90.0)


def test_degenerate_clicks_raise():
    s = LandmarkClickSession()
    for p in (J, H1, H1, *OPEN):         # H1 == H2
        s.add(*p)
    with pytest.raises(ValueError):
        s.result()


# ---- shapes ------------------------------------------------------------------------------------

def _old_annotator_shapes(pts):
    """The annotator's _overlay_shapes as it was before the logic moved into landmark_edit."""
    shapes = []
    colors = {"junction": JUNCTION_COLOR, "hypo_1": HYPO_COLOR, "hypo_2": HYPO_COLOR,
              "cotyl_1": COTYL_COLOR, "cotyl_2": COTYL_COLOR}
    labels = {"junction": "J", "hypo_1": "H1", "hypo_2": "H2", "cotyl_1": "C1", "cotyl_2": "C2"}
    for name, xy in pts.items():
        shapes.append({"kind": "point", "xy": xy, "color": colors[name], "label": labels[name]})
    if "junction" in pts:
        j = pts["junction"]
        for a, b, color in (("hypo_1", "hypo_2", HYPO_COLOR), ("cotyl_1", "cotyl_2", COTYL_COLOR)):
            if a in pts and b in pts:
                shapes.append({"kind": "line", "from": pts[a], "to": pts[b], "color": color, "dash": True})
                try:
                    ux, uy = orient_axis(pts[a], pts[b], j)
                except ValueError:
                    continue
                length = max(40.0, 1.2 * max(abs(pts[a][0] - j[0]) + abs(pts[a][1] - j[1]),
                                              abs(pts[b][0] - j[0]) + abs(pts[b][1] - j[1])))
                shapes.append({"kind": "line", "from": j, "to": (j[0] + ux * length, j[1] + uy * length),
                               "color": color, "arrow": True})
    return shapes


@pytest.mark.parametrize("n", [0, 1, 3, 4, 5])
def test_shapes_from_points_match_the_annotators_output(n):
    session = AngleClickSession()
    for p in (J, H1, H2, *OPEN)[:n]:
        session.add(*p)
    assert landmark_shapes(session.named_points()) == _old_annotator_shapes(session.named_points())


def test_shapes_from_points_match_the_annotators_output_degenerate_axis():
    session = AngleClickSession()
    for p in (J, H1, H1, *OPEN):
        session.add(*p)
    assert landmark_shapes(session.named_points()) == _old_annotator_shapes(session.named_points())


def test_frozen_expected_shapes_for_five_clicks():
    session = AngleClickSession()
    for p in (J, H1, H2, *OPEN):
        session.add(*p)
    shapes = landmark_shapes(session.named_points())
    assert [s["kind"] for s in shapes] == ["point"] * 5 + ["line"] * 4
    assert [s.get("label") for s in shapes[:5]] == ["J", "H1", "H2", "C1", "C2"]
    assert shapes[5]["dash"] and shapes[6]["arrow"] and shapes[7]["dash"] and shapes[8]["arrow"]
    assert shapes[6]["from"] == J and shapes[6]["to"][0] == pytest.approx(30.0)
    assert shapes[6]["to"][1] == pytest.approx(20.0 + 1.2 * 40.0)      # 1.2 * (|0|+|40|) = 48 > 40


def test_session_points_add_collar_and_root():
    s = _session(OPEN, extra=[(30.0, 70.0), (30.0, 90.0)])
    shapes = landmark_shapes(s.named_points())
    labels = [x.get("label") for x in shapes if x["kind"] == "point"]
    assert labels == ["J", "H1", "H2", "C1", "C2", "collar", "root"]
    root_arrow = shapes[-1]
    assert root_arrow["kind"] == "line" and root_arrow["arrow"] and root_arrow["from"] == (30.0, 70.0)
    # collar alone: a point, no arrow
    s = _session(OPEN, extra=[(30.0, 70.0)])
    assert landmark_shapes(s.named_points())[-1]["label"] == "collar"


def test_shapes_from_a_readout():
    readout = {"junction": (30.0, 20.0), "hypo_dir": (0.0, 1.0), "cotyl_dir": (1.0, 0.0),
               "base": (32.0, 22.0), "tip": (50.0, 22.0), "collar": (30.0, 80.0), "root_dir": (0.0, 1.0)}
    shapes = landmark_shapes(readout=readout)
    labels = [s.get("label") for s in shapes if s["kind"] == "point"]
    assert labels == ["J", "C1", "C2", "collar"]
    arrows = [s for s in shapes if s["kind"] == "line"]
    assert len(arrows) == 3 and all(a["arrow"] for a in arrows)
    assert arrows[0]["to"] == pytest.approx((30.0, 60.0))                # hypocotyl, arrow_min = 40
    assert arrows[1]["to"] == pytest.approx((70.0, 20.0))
    assert arrows[2]["from"] == (30.0, 80.0)
    assert landmark_shapes(readout=readout, arrow_min=10)[1]["to"] == pytest.approx((30.0, 30.0))


def test_shapes_from_none_or_partial_readouts():
    assert landmark_shapes() == []
    assert landmark_shapes(readout=None) == []
    assert landmark_shapes(points={}) == []
    assert landmark_shapes(readout={"junction": None, "theta": None}) == []
    partial = landmark_shapes(readout={"junction": (10.0, 10.0), "hypo_dir": (0.0, 1.0)})
    assert [s["kind"] for s in partial] == ["point", "line"]
