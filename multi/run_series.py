"""Run the GUI's own batch analysis headless on one series and a saved boxes file.

    python -m multi.run_series --images example_data/Plate_1_Old \\
        --boxes example_data/Plate_1_Old/crop_boxes.json --out multi/results/series_Plate_1_Old

It is the code path behind *Start Analysis* (Gui._run_analysis: crop, segment, landmarks,
readability gate, reconstruction, temporal flags, germination), on a bare Gui with no Tk window,
so what it writes is what the GUI would show and export:

- series.csv          one row per seedling per frame, the GUI export (utils/export_rows.py);
- seedling_NN.png     per seedling: bio angle, junction-collar distance and collar deviation
                      from the seedling's stable collar, with unreadable and flagged frames marked;
- summary.txt         per seedling: frames read, unreadable, temporal flags;
- seedlings.csv       GUI seedling_id -> box index in the boxes file (the GUI numbers boxes spatially).

The crops go to a scratch data folder (--data-dir, default <out>/data), never the repo's data/.
`--checkpoints a.pt+b.pt+c.pt` swaps the landmark ensemble (e.g. to compare a retrained one).
"""
import argparse
import math
import os
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--images", required=True, help="Folder of the series' frames")
    p.add_argument("--boxes", required=True, help="Crop boxes JSON saved by the GUI (Save boxes...)")
    p.add_argument("--series", default=None, help="Series name in the boxes file (default: the folder name)")
    p.add_argument("--out", required=True, help="Output folder")
    p.add_argument("--data-dir", default=None, help="Scratch folder for the crops (default: <out>/data)")
    p.add_argument("--checkpoints", default=None, help="Landmark checkpoints joined by '+' (default: the GUI's)")
    p.add_argument("--no-figures", action="store_true", help="Write the CSV and summary only")
    return p.parse_args(argv)


def build_gui(images, boxes_file, series=None):
    """A bare Gui (no Tk) holding the ordered series and its boxes, ready for _run_analysis()."""
    import cv2
    from seedling_measurement import Gui, _list_image_files
    from utils.crop_boxes_io import load_boxes_json
    from utils.gui_thread_safety import ProgressReporter
    from utils.preprocess_model_input import compute_time_deltas, metadata_has_gaps, order_series

    gui = Gui.__new__(Gui)
    gui._init_state()
    gui.model_superres = None
    gui.progress_reporter = ProgressReporter()
    gui._debug_snapshot = False
    gui.path = str(images)
    gui.series_name = series or os.path.basename(os.path.normpath(images))
    gui.file_list, gui.capture_times = order_series(gui.path, _list_image_files(gui.path))
    if not gui.file_list:
        raise SystemExit(f"no images in {images}")
    first = cv2.imread(os.path.join(gui.path, gui.file_list[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise SystemExit(f"cannot read {gui.file_list[0]}")
    gui.height1, gui.width1 = first.shape[:2]

    loaded = load_boxes_json(boxes_file, gui.series_name)
    _, boxes, collars = loaded
    raw = [dict(box, collar=list(c)) if c is not None else dict(box) for box, c in zip(boxes, collars)]
    boxes, collars, order = gui._normalise_boxes(raw)
    gui.box_file_order = order          # GUI seedling k+1 is box order[k] of the boxes file
    gui.crop_boxes, gui.collar_overrides = boxes, collars
    gui.transformed_mid_points = [(b["cx"], b["cy"]) for b in boxes]

    gui.metadata_list = [{"filename": f, "creation_time": gui.capture_times.get(f)} for f in gui.file_list]
    trusted = not metadata_has_gaps(gui.metadata_list)
    gui.time_deltas = compute_time_deltas(gui.metadata_list, None, trust_timestamps=trusted)
    return gui


def write_seedling_map(gui, path):
    """seedlings.csv: GUI seedling_id -> index of its box in the boxes file (the GUI renumbers boxes in
    spatial order; crops cut by recrop_plates and their annotations keep the file order)."""
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["seedling_id", "box_index", "cx", "cy"])
        for k, (box, i) in enumerate(zip(gui.crop_boxes, getattr(gui, "box_file_order", range(len(gui.crop_boxes))))):
            w.writerow([k + 1, i, box["cx"], box["cy"]])


def export_rows(gui):
    """The GUI export rows (one per seedling per frame) as a DataFrame."""
    import pandas as pd
    from utils.export_rows import build_export_row, export_columns

    rows = []
    for frame_idx, filename in enumerate(gui.file_list):
        delta = gui.time_deltas[frame_idx] if frame_idx < len(gui.time_deltas) else None
        elapsed = delta.get("elapsed_minutes") if delta else None
        for crop_id in range(len(gui.crop_boxes)):
            results = gui.frame_results_by_crop.get(crop_id, [])
            result = results[frame_idx] if frame_idx < len(results) else None
            germ = gui.germination_detector.get_time_zero(crop_id) if gui.germination_detector else None
            rows.append(build_export_row(filename, crop_id + 1, result, elapsed, germ, None))
    return pd.DataFrame(rows, columns=export_columns(None))


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f


def seedling_tracks(df, seed_id):
    """Per-frame series of one seedling for the figure: frame index, bio angle, distance, collar deviation."""
    import numpy as np

    sub = df[df["seedling_id"] == seed_id].reset_index(drop=True)
    jx, jy = sub["junction_x"].map(_num), sub["junction_y"].map(_num)
    rx, ry = sub["collar_ref_x"].map(_num), sub["collar_ref_y"].map(_num)
    cx, cy = sub["collar_x"].map(_num), sub["collar_y"].map(_num)
    return {
        "frame": np.arange(len(sub)),
        "bio": sub["bio_angle"].map(_num).to_numpy(),
        "unreadable": sub["unreadable"].map(lambda v: str(v) == "1").to_numpy(),
        "flag": sub["temporal_flag"].fillna("").astype(str).to_numpy(),
        "state": sub["state"].fillna("").astype(str).to_numpy(),
        "jc_dist": np.hypot(jx - rx, jy - ry).to_numpy(),
        "collar_dev": np.hypot(cx - rx, cy - ry).to_numpy(),
    }


def plot_seedling(tracks, seed_id, out_path, series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from utils.landmark_temporal import COLLAR_JUMP_PX

    t = tracks
    flagged = np.array([bool(f) for f in t["flag"]])
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    ax = axes[0]
    ax.plot(t["frame"], t["bio"], "-", color="0.6", lw=1)
    ok = ~t["unreadable"] & ~flagged
    ax.scatter(t["frame"][ok], t["bio"][ok], s=14, color="tab:blue", label="read")
    if flagged.any():
        ax.scatter(t["frame"][flagged], t["bio"][flagged], s=30, facecolors="none", edgecolors="tab:orange",
                   linewidths=1.5, label="temporal flag")
    for f in t["frame"][t["unreadable"]]:
        ax.axvline(f, color="0.85", lw=3, zorder=0)
    ax.axhline(180, color="0.85", lw=1)
    ax.set_ylabel("bio angle (deg)")
    ax.set_title(f"{series} - seedling {seed_id}  (grey bands: unreadable)")
    ax.legend(fontsize=8, loc="lower left")

    axes[1].plot(t["frame"], t["jc_dist"], ".-", color="tab:green", lw=1)
    jflag = np.array(["junction" in f for f in t["flag"]])
    axes[1].scatter(t["frame"][jflag], t["jc_dist"][jflag], s=40, facecolors="none", edgecolors="tab:orange", linewidths=1.5)
    axes[1].set_ylabel("junction-collar\ndistance (px)")

    axes[2].plot(t["frame"], t["collar_dev"], ".-", color="tab:purple", lw=1)
    axes[2].axhline(COLLAR_JUMP_PX, color="tab:orange", lw=1, ls="--")
    cflag = np.array(["collar" in f for f in t["flag"]])
    axes[2].scatter(t["frame"][cflag], t["collar_dev"][cflag], s=40, facecolors="none", edgecolors="tab:orange", linewidths=1.5)
    axes[2].set_ylabel("collar deviation\nfrom stable (px)")
    axes[2].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def summarise(df):
    lines = [f"{'seedling':>8} {'frames':>6} {'read':>5} {'unread':>6} {'collar':>6} {'junct':>6} {'germ':>5}"]
    for seed_id, sub in df.groupby("seedling_id"):
        flags = Counter()
        for f in sub["temporal_flag"].fillna(""):
            for part in str(f).split("+"):
                if part:
                    flags[part] += 1
        read = int(sub["bio_angle"].map(lambda v: v != "" and not math.isnan(_num(v))).sum())
        unread = int(sub["unreadable"].map(lambda v: str(v) == "1").sum())
        germ = sub["germination_frame"].iloc[0]
        lines.append(f"{seed_id:>8} {len(sub):>6} {read:>5} {unread:>6} {flags['collar']:>6} {flags['junction']:>6} {str(germ):>5}")
    total_flags = int((df["temporal_flag"].fillna("") != "").sum())
    total_unread = int(df["unreadable"].map(lambda v: str(v) == "1").sum())
    lines.append(f"\n{len(df)} seedling-frames, {total_unread} unreadable, {total_flags} with a temporal flag")
    return "\n".join(lines)


def main(argv=None):
    args = _parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.checkpoints:
        os.environ["DLHOOK_ANGLE_SOURCE"] = "landmarks"

    from utils import paths
    paths.DATA_ROOT = Path(args.data_dir) if args.data_dir else out / "data"
    images = Path(paths.data_path("images", make=True))
    for old in images.glob("*.png"):          # a previous run's crops
        old.unlink()

    if args.checkpoints:
        import seedling_measurement as sm
        from models import segmentation_backends as sb
        sm.get_landmark_runner = lambda: sb.get_landmark_runner(args.checkpoints)
        sm.resolve_angle_source = lambda: sb.resolve_angle_source(args.checkpoints)

    gui = build_gui(args.images, args.boxes, args.series)
    print(f"{gui.series_name}: {len(gui.file_list)} frames x {len(gui.crop_boxes)} seedlings")
    gui._run_analysis()
    df = export_rows(gui)
    df.to_csv(out / "series.csv", index=False)
    write_seedling_map(gui, out / "seedlings.csv")
    summary = summarise(df)
    (out / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    if not args.no_figures:
        for seed_id in sorted(df["seedling_id"].unique()):
            plot_seedling(seedling_tracks(df, seed_id), seed_id, out / f"seedling_{seed_id:02d}.png", gui.series_name)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
