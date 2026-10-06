import csv

import pytest

from utils.germination_annotation import (
    GerminationStore, Seedling, frame_labels, in_test_subset, load_seedlings, write_frame_labels,
)
from utils.root_annotation import RootClickSession


def seedling(n=5, crop_id=0, series="S"):
    return Seedling(series, crop_id, [(f"f{i}", f"{crop_id}-crop-f{i}.png", f"f{i}.tif") for i in range(n)])


def test_found_onset_labels_frames_before_and_after(tmp_path):
    s, store = seedling(), GerminationStore(tmp_path / "g.csv")
    store.save_found(s, 2)
    rows = frame_labels([s], store)
    assert [r["radicle_visible"] for r in rows] == [0, 0, 1, 1, 1]
    assert [r["offset_from_onset"] for r in rows] == [-2, -1, 0, 1, 2]


def test_censored_and_skipped_seedlings(tmp_path):
    a, b, c = seedling(3, 0), seedling(3, 1), seedling(3, 2)
    store = GerminationStore(tmp_path / "g.csv")
    store.save_status(a, "before_start")
    store.save_status(b, "none")
    store.save_status(c, "skipped")
    rows = frame_labels([a, b, c], store)
    assert [r["radicle_visible"] for r in rows if r["crop_id"] == 0] == [1, 1, 1]
    assert [r["radicle_visible"] for r in rows if r["crop_id"] == 1] == [0, 0, 0]
    assert not [r for r in rows if r["crop_id"] == 2]                       # skipped is not exported
    assert all(r["offset_from_onset"] == "" for r in rows)
    assert store.counts([a, b, c]) == {"found": 0, "before_start": 1, "none": 1, "skipped": 1}


def test_store_resumes_and_keeps_collar_and_root(tmp_path):
    path = tmp_path / "g.csv"
    s = seedling()
    GerminationStore(path).save_found(s, 3, RootClickSession([(10, 10), (10, 40)]))
    again = GerminationStore(path)
    assert again.onset_index(s) == 3 and again.points_for(s) == [(10.0, 10.0), (10.0, 40.0)]
    assert again.get(s)["onset_frame"] == "f3" and again.get(s)["n_frames"] == "5"


def test_invalid_input_is_rejected_without_writing(tmp_path):
    path = tmp_path / "g.csv"
    s, store = seedling(), GerminationStore(path)
    with pytest.raises(ValueError):
        store.save_found(s, 9)
    with pytest.raises(ValueError):
        store.save_found(s, 1, RootClickSession([(1, 1)]))                  # one of the two clicks
    with pytest.raises(ValueError):
        store.save_status(s, "found")
    assert not path.exists()


def test_export_writes_a_csv(tmp_path):
    s, store = seedling(), GerminationStore(tmp_path / "g.csv")
    store.save_found(s, 1)
    out = tmp_path / "labels.csv"
    assert write_frame_labels(out, [s], store) == 5
    with open(out, newline="", encoding="utf-8") as fh:
        assert [r["radicle_visible"] for r in csv.DictReader(fh)] == ["0", "1", "1", "1", "1"]


def test_load_seedlings_orders_frames_and_drops_single_frame_timelines(tmp_path):
    for name in ("0-crop-S_10.png", "0-crop-S_9.png", "1-crop-S_1.png"):
        (tmp_path / name).write_bytes(b"x")
    got = load_seedlings(str(tmp_path))
    assert len(got) == 1 and [f[0] for f in got[0].frames] == ["S_9", "S_10"]


def test_subsets_partition_the_seedlings_deterministically(tmp_path):
    for crop_id in range(40):
        for frame in ("S_1", "S_2"):
            (tmp_path / f"{crop_id}-crop-{frame}.png").write_bytes(b"x")
    train = {s.crop_id for s in load_seedlings(str(tmp_path), subset="train")}
    test = {s.crop_id for s in load_seedlings(str(tmp_path), subset="test")}
    assert train | test == set(range(40)) and not train & test
    assert 2 <= len(test) <= 16                                              # about 20% of 40
    assert test == {s.crop_id for s in load_seedlings(str(tmp_path), subset="test")}
    assert in_test_subset("S", 3, 0.0) is False and in_test_subset("S", 3, 1.0) is True
