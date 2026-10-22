"""multi/run_series.py: building the headless Gui from a series + boxes file, and the per-seedling
tracks and summary it derives from the export rows (no models are run here)."""
import json

import numpy as np
import pandas as pd
from PIL import Image

from utils.export_rows import export_columns


def test_build_gui_orders_frames_and_normalises_boxes(tmp_path):
    from multi.run_series import build_gui

    series = tmp_path / "Plate_X"
    series.mkdir()
    for i in (3, 1, 2):
        Image.fromarray(np.full((200, 300, 3), 90, np.uint8)).save(series / f"Plate_X_{i:03d}.png")
    boxes = tmp_path / "boxes.json"
    boxes.write_text(json.dumps({"Plate_X": [
        {"cx": 250, "cy": 150, "half_w": 20, "half_h": 40},
        {"cx": 50, "cy": 60, "half_w": 20, "half_h": 40, "collar": [55, 90]},
    ]}), encoding="utf-8")
    gui = build_gui(series, boxes)
    assert gui.file_list == ["Plate_X_001.png", "Plate_X_002.png", "Plate_X_003.png"]
    assert [b["cx"] for b in gui.crop_boxes] == [50, 250]                # spatial order, as the GUI numbers them
    assert gui.collar_overrides[0] == (55, 90) and gui.collar_overrides[1] is None
    assert len(gui.time_deltas) == 3 and gui.model_superres is None


def _rows():
    cols = export_columns(None)
    rows = []
    for f in range(4):
        r = {c: "" for c in cols}
        r.update(img_name=f"f{f}.png", seedling_id=1, bio_angle=170 - f, state="Closed",
                 unreadable=1 if f == 0 else 0, temporal_flag="junction" if f == 2 else "",
                 junction_x=40.0, junction_y=200.0 - 10 * f, collar_x=40.0, collar_y=300.0,
                 collar_ref_x=40.0, collar_ref_y=300.0, germination_frame=1)
        rows.append(r)
    return pd.DataFrame(rows, columns=cols)


def test_tracks_and_summary():
    from multi.run_series import seedling_tracks, summarise

    t = seedling_tracks(_rows(), 1)
    assert list(t["jc_dist"]) == [100.0, 110.0, 120.0, 130.0]
    assert list(t["unreadable"]) == [True, False, False, False] and t["flag"][2] == "junction"
    text = summarise(_rows())
    assert "1 unreadable, 1 with a temporal flag" in text
