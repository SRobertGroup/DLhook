import json

from multi import trim_to_references as tr
from utils.crop_layout import box_to_rect, rects_overlap


def box(x1, y1, x2, y2):
    return {"cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2, "half_w": (x2 - x1) // 2, "half_h": (y2 - y1) // 2}


def test_a_strip_overlap_is_cut_off_the_cheapest_side():
    ref = box(100, 0, 200, 300)
    mine = box(40, 20, 110, 280)                           # 10 px strip on its right
    new, status = tr.trim_one(mine, [ref])
    assert status == "trimmed" and box_to_rect(new)[2] <= 100 and box_to_rect(new)[0] == 40
    assert not rects_overlap(box_to_rect(new), box_to_rect(ref))
    assert tr.trim_one(box(0, 0, 60, 200), [ref]) == (box(0, 0, 60, 200), "clear")


def test_corner_overlap_picks_the_side_that_removes_less():
    ref = box(100, 200, 300, 400)
    mine = box(40, 0, 130, 210)                            # overlaps by 30 wide x 10 high: cut the bottom, not the right
    new, status = tr.trim_one(mine, [ref])
    x1, y1, x2, y2 = box_to_rect(new)
    assert status == "trimmed" and y2 <= 200 and x2 >= 128


def test_too_costly_or_too_small_is_left_for_manual():
    ref = box(50, 0, 200, 300)
    deep = box(0, 0, 100, 300)                              # half of it lies under the reference
    assert tr.trim_one(deep, [ref], max_trim=0.4) == (deep, "manual")
    narrow = box(0, 0, 60, 300)
    assert tr.trim_one(narrow, [box(40, 0, 200, 300)], min_half=25)[1] == "manual"   # would leave 40 px
    assert tr.trim_one(narrow, [box(40, 0, 200, 300)], min_half=15)[1] == "trimmed"


def test_two_references_and_the_cli(tmp_path):
    boxes = {"S": [box(40, 20, 110, 280), box(500, 0, 560, 100)], "T": [box(0, 0, 50, 50)]}
    refs = [("a", {"S": [box(100, 0, 200, 300)]}), ("b", {"S": [box(20, 270, 80, 400)]})]
    out, report = tr.trim_set(boxes, refs)
    assert [r[2] for r in report] == ["trimmed", "clear", "clear"]
    assert all(not rects_overlap(box_to_rect(b), box_to_rect(r)) for b in out["S"] for _, per in refs for r in per["S"])
    (tmp_path / "b.json").write_text(json.dumps(boxes), encoding="utf-8")
    (tmp_path / "ref.json").write_text(json.dumps(refs[0][1]), encoding="utf-8")
    assert tr.main(["--boxes", str(tmp_path / "b.json"), "--out", str(tmp_path / "o.json"),
                    "--existing", f"a={tmp_path / 'ref.json'}"]) == 0
    assert json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))["S"][1] == boxes["S"][1]
