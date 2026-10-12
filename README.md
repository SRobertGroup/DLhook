# DLhook

> _High-throughput apical hook phenotyping of dark-grown Arabidopsis thaliana seedlings_

`v0.3.0` · pre-release · GPL-3.0

**Lead Development:** David Radianu, Adrien Heymans

**Contributors:** Sara Raggi, Siamsa Doyle, Hemamshu Ratnakaram, Sandeep Yadav

**Coordination:** Stéphanie Robert, Adrien Heymans


![DLhook](docs/img/header.png)

---

## What DLhook does

`DLhook` measures the apical hook angle of dark-grown *Arabidopsis thaliana* seedlings across an image time series, replacing the tedious work of measuring each hook by hand.

The pipeline has four stages:

1. **Segmentation.** A U-Net with group normalisation and residual connections (`UNetGNRes`, PyTorch) segments each cropped seedling into **cotyledon** and **hypocotyl** (thin overlays to judge the reading against, plus the hypocotyl mask that length, curvature and diameter are measured on). A separate learned detector finds **germination** (the emerging radicle), which fixes time zero for each seedling independently; the radicle itself is no longer segmented. Earlier versions ran three separate two-class models and fitted ellipses; both paths are retained and can still be selected — see [Which segmentation models run](#which-segmentation-models-run) and [Which angle method runs](#which-angle-method-runs).
2. **Geometry.** For every frame, an ensemble of three landmark networks reads the cotyledon/hypocotyl **junction** and the direction of each axis directly from the crop, then computes the angle between them (`raw_angle`). On the hypocotyl mask DLhook also measures its length, curvature and diameter.
3. **Temporal reconstruction.** Each seedling's frames are read as one series: frames with no usable information are rejected, short gaps are interpolated, and every frame gets a biological angle (`bio_angle`) and a state (`Closed` / `Opening` / `Unreadable`). Frames whose collar or junction breaks the way a seedling grows are flagged for review.
4. **Review and export.** Every seedling can be inspected frame by frame and its landmarks placed by hand before the results are written to a `.CSV`.

The network architecture and the trained weights derive from [RootPainter](https://github.com/Abe404/root_painter). Inference runs on [PyTorch](https://pytorch.org/).

```mermaid
flowchart LR
    A["Image time series\n(one directory)"] --> B["Per-seedling crops\n(drawn boxes)"]
    B --> C1["Cotyledon + hypocotyl\nsegmentation"]
    B --> C2["Landmark ensemble\n(junction + directions)"]
    B --> C3["Germination\ndetection (learned)"]
    C2 --> D["Per-frame geometry\n(raw_angle, traits)"]
    C1 --> D
    C3 --> E["Time zero\nper seedling"]
    D --> F["Temporal reconstruction\n(bio_angle, state, flags)"]
    E --> F
    F --> G["CSV export"]
    F --> H["Kinematics plot"]
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SRobertGroup/DLhook/
cd DLhook
```

### 2. Create the environment

> [!NOTE]
> We recommend [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) for creating the virtual environment ([Anaconda](https://www.anaconda.com/download) works too).
>
> For more on setting up conda, see the [conda user guide](https://conda.io/projects/conda/en/latest/user-guide/install).

The maintained path is the environment file, which pins Python 3.10 and installs the pinned packages of `requirements.txt` with a CUDA 12.8 build of PyTorch. Run it from the repository root:

```bash
mamba env create -f env.yaml
mamba activate dlhook_env
```

Every package has an exact version. `requirements-lock.txt` pins the full environment, transitive packages included (`pip install -r requirements-lock.txt` reproduces it), and `requirements-dev.txt` adds `pytest`.

<details>
<summary>Alternative: build the environment by hand</summary>

```bash
mamba create -n dlhook_env python=3.10 -y
mamba activate dlhook_env
```

For **GPU** (CUDA), take PyTorch from the PyTorch index. The pins match the `2.8.0+cu128` build; use `cu126` for an older driver:

```bash
pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
```

For **CPU only**:

```bash
pip3 install --no-cache-dir -r requirements.txt
```

</details>

Dependencies, all pinned (see [`requirements.txt`](requirements.txt) and [`env.yaml`](env.yaml)): `torch`, `torchvision`, `numpy`, `pandas`, `pillow`, `matplotlib`, `scikit-image`, `opencv-python`, `PyYAML`, `tqdm`, `basicsr` and `realesrgan`.

> [!TIP]
> A GPU is optional but strongly recommended — segmentation is roughly **8× faster** on CUDA. Match the CUDA version to your driver; `env.yaml` installs the CUDA 12.8 build, which recent GPU generations (Blackwell) need; change `cu128` to `cu126` for an older driver. DLhook checks that CUDA actually works (a tiny test operation) before using the GPU and falls back to the CPU with a warning when it does not; `DLHOOK_DEVICE=cpu` forces the CPU.

### 3. Start DLhook

```bash
python main.py
```

> [!NOTE]
> You have to activate the environment (`mamba activate dlhook_env`) each time before launching the application.

### Model weights

The default pipeline needs these files. **None of them is in git** (they are untracked, like `weights/multiclass/`): they must be obtained from the maintainers or copied from the machine that trained them, and put at exactly these paths.

| File | Provides | Used for |
|---|---|---|
| `weights/landmarks/dlhook_landmarks_{a,b,c}.pt` | landmark ensemble (three members) | the hook angle, junction and axis directions |
| `weights/multiclass/dlhook_4class_v1.pt` | cotyledon + hypocotyl | the overlays |
| [`weights/RootPainter_weights/hypocot_v5.pkl`](weights/RootPainter_weights/) | hypocotyl (binary) | the mask the length, curvature and diameter are measured on (the four-class model cuts the hypocotyl in half) |
| `weights/germination_detector.json` | learned germination detector | per-seedling time zero (it reads the four-class checkpoint above) |

**When a file is missing DLhook falls back, with a warning in the console, instead of failing:**

- no `weights/landmarks/*.pt`: the angle comes from the legacy ellipse fit (see below);
- no `weights/multiclass/dlhook_4class_v1.pt`: the legacy three binary models run for segmentation;
- no `weights/germination_detector.json` (or no four-class checkpoint): germination is **unknown**. `germination_frame` is an empty cell and time zero stays unset. Set the frame by hand in the analysis window if you need it.

Results therefore differ between machines that do not hold the same files: check the console for `WARNING` lines before comparing runs.

The legacy paths use the three two-class RootPainter models (`cotyledon_v5.pkl`, `hypocot_v5.pkl`, `germ_v1.pkl`, all in [`weights/RootPainter_weights/`](weights/RootPainter_weights/)).

A further model, Real-ESRGAN (`models/superres/RealESRGAN_x4plus.pth`), can super-resolve crops on CUDA, but it is **off by default** (`DLHOOK_SUPERRES=1` turns it on): the landmark models were built on plain crops, and super-resolved crops raised their validation error from 20.6 to 34.3 degrees. The 64 MB file is not in the repository: it is downloaded from the official Real-ESRGAN release on first use and checked against a pinned SHA-256 (`models/superres/weights_info.py`); a file already at that path is used as is.

### Which angle method runs

Set `DLHOOK_ANGLE_SOURCE` before launching to choose:

| Value | Behaviour |
|---|---|
| `landmarks` *(default)* | Three landmark networks (`dlhook_landmarks_{a,b,c}.pt`, flip-averaged) read the junction and the hypocotyl and cotyledon directions on the plain crop. |
| `ellipse` | The legacy method: ellipses fitted to the cotyledon and hypocotyl masks. |

```bash
DLHOOK_ANGLE_SOURCE=ellipse python main.py     # legacy ellipse method
DLHOOK_LANDMARK_GATE=0 python main.py          # keep every landmark reading (no readability gate)
```

On the final BerryBox v2 validation set (287 hand-measured frames, plants no model had seen) the landmark ensemble has a mean error of **13.9°** (median **7.2°**), against **26.8°** for the ellipse pipeline; repeating your own measurement by hand differs by **4.2°** on average (see [`docs/AUDIT.md`](docs/AUDIT.md), §8). The landmark angle is a folded angle, `180 - theta`; the automatic overhook flag is exported separately and does not change `bio_angle`. Landmarks are read from the plain crop, so they need no segmentation mask: an angle exists even where a mask is empty.

**Readability gate.** The ensemble reads an angle on every frame, including frames with no usable information (the ones an annotator would skip). A gate rejects such frames:

- a rejected frame gets no angle (`bio_angle` empty, state `Unreadable`) and is never interpolated across;
- the analysis window still shows the rejected reading, in grey, and placing the landmarks by hand overrides the gate;
- `DLHOOK_LANDMARK_GATE=0` disables it (every frame keeps its reading; `unreadable_p` is still exported).

The gate is a logistic rule on the junction peak, the cotyledon-tip peak and the disagreement between the three networks. On BerryBox v2 it rejects 68% of the skipped frames while keeping 99% of the measured ones. Landmark models that carry the learned **measurable** output (an image-level head trained on the frames annotators skipped) are gated on that output instead, and `measurable_p` is exported.

**Time-series checks.** Two facts of a growing seedling are checked across each seedling's frames:

- the collar barely moves (median 2.6 px between annotated frames);
- the junction moves away from the collar as the hypocotyl elongates (the distance grew in 99.7% of 737 annotated frame pairs).

A frame is **flagged, not changed**, when its collar jumps away from the seedling's stable collar (by more than 15 px, or more than a quarter of the junction–collar distance), or when its junction moves back toward the collar by more than 5 px. A flagged frame is drawn in amber in the analysis window, with the seedling's usual collar as a hollow ring and a one-line explanation; its point gets a black ring in the kinematics plot; and the export names the check in `temporal_flag`. A manual edit is trusted and clears the flag.

### Which segmentation models run

Set `DLHOOK_SEG_BACKEND` before launching to choose:

| Value | Behaviour |
|---|---|
| `multiclass` *(default)* | Four-class model for cotyledon and hypocotyl (the overlays), plus `hypocot_v5.pkl` for the mask the hypocotyl traits use. No radicle mask: germination comes from the learned detector. Falls back to `binary` when `weights/multiclass/dlhook_4class_v1.pt` is missing. |
| `binary` *(legacy)* | Three two-class models: `cotyledon_v5.pkl`, `hypocot_v5.pkl`, `germ_v1.pkl`, unchanged from earlier versions. |

```bash
DLHOOK_SEG_BACKEND=binary python main.py     # legacy three-model path
```

With no radicle mask the area-rule germination fallback is unavailable: when the learned detector cannot run, germination is reported as **unknown** (empty cell, with a warning) rather than guessed. The value is read once when the application starts, so it cannot be changed part-way through a session: one set of results can never mix output from two different models. An unrecognised value stops the program with an error rather than silently falling back.

![Example of a good hypocotyl segmentation](docs/img/Good_segmentation_RootPainter_hypocotyl.png)

---

## Input data

The input is a single directory holding one time series:

```
images_folder
   |-- Col0_001.tif
   |-- Col0_002.tif
   |-- ...
```

**Accepted extensions:** `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, `.bmp` (any case).

**Frame ordering.** DLhook orders the series by the **capture time embedded in each file** (EXIF sub-IFD `DateTimeOriginal` / `DateTimeDigitized`, or TIFF tag 306), taking the earliest timestamp present. It falls back to the first integer found in the filename only when the embedded times are missing or unusable. Filesystem creation and modification times are deliberately **not** used — they are frequently identical across a whole series and therefore meaningless as an ordering key.

If the timestamps cannot be trusted — any frame missing one, or the series not strictly increasing — DLhook asks once for the **interval between images in minutes** (dialog: *No usable capture times*) and builds the time axis from that. ImageJ-exported TIFFs typically carry no timestamp at all and will always prompt.

---

## Workflow

Load a directory, draw one box per seedling, and run the analysis. The sidebar is grouped as **Images**, **Seedlings**, **Analysis** and **Settings**; the window and the canvas can be resized.

![DLhook main window](docs/img/gui-overview.png)

### Loading the series

Click **`Open image directory`** and select the folder containing one time series. The frames appear in the list, in acquisition order, and the last frame is shown at its native resolution. Select any other frame in the list to display it; the boxes and the zoom stay where they are. Opening another folder starts a new series (you are asked first if that would discard boxes or results).

### 1. Draw the crop boxes

Work on the **last frame** of the series, where the seedlings are most developed. The canvas works like `python -m multi.draw_crops`:

| Action | Result |
|---|---|
| **Drag** on empty space | Draw a seedling: drag over its visible extent (seed coat to hook tip). The padded crop box is shown live and is the box that is stored. |
| **Drag** a box | Move it. |
| **Click** a box | Select it. Drag a **corner handle** of the selected box to resize it. |
| **`Delete`** / **`Backspace`** (or `Delete selected`) | Delete the selected box. |
| **`Esc`** | Deselect. |
| **Mouse wheel** | Zoom at the cursor. **Right-drag** (or middle-drag) pans, **`R`** (or `Reset zoom`) fits the whole image. |

- **Boxes are highlighted, not trimmed, when they overlap.** Overlapping boxes are outlined in **magenta** and the status line below the canvas names them (for example `OVERLAP: 2 & 3`). Boxes are clamped to the image.
- **Each box shows its seedling ID** in its top-left corner. IDs are numbered across the plate **top-to-bottom, then left-to-right**, and renumbered whenever a box is added, moved or removed. This number is the `seedling_id` in the exported CSV and the number in *Preview seedling*. If results already exist, changing a box asks for confirmation and then clears them, because the results are tied to the box numbers.
- The status line also shows the number of seedlings, the number of collar overrides and the current frame.

### 2. Optional collar, save and load boxes

**Collar (optional).** Everything below the seedling's collar (the seed coat or root side) can be cut off from the masks. By default DLhook uses the collar that the landmark model finds in each frame, and masks nothing when it finds none. To fix the collar of one seedling yourself, **Shift+click inside its box**: a purple dot marks it (Shift+drag moves it; the dot travels with the box). **Shift+right-click** inside the box, or `Clear collar`, removes the override. Collar overrides apply to every frame of that seedling.

**Save boxes… / Load boxes….** The boxes are saved as JSON in the format of `multi/draw_crops.py` and `multi/recrop_plates.py`, keyed by the image folder name, with the optional collar as `"collar": [x, y]` in image pixels:

```json
{"plate_one": [{"cx": 428, "cy": 783, "half_w": 120, "half_h": 90, "collar": [430, 850]}]}
```

Saving to an existing file replaces only this folder's entry. When loading, the series named like the open folder is used; a file with a single series is used as is; otherwise you are asked which series to load.

### 3. `Start Analysis`

Runs segmentation and the landmark ensemble across every frame and every crop, then computes the per-frame angles and hypocotyl traits. Progress is shown in the bar at the bottom of the window: cropping occupies the first 30%, segmentation and angle computation the remainder.

When it finishes, the status reads *"Analysis complete -- use Preview seedling to review, or Export results when done"*.

![Step 3 — running the analysis](docs/img/5-repo.png)

### 4. `Preview seedling`

Opens a picker; choose a seedling to open its **`Seedling N - analysis`** window. This is where all reviewing and correcting happens.

**Viewing**
- `Launch segmentation`: only needed if this seedling has not been segmented yet. Re-segmenting replaces the seedling's results, so you are asked first when it holds manual frames.
- `Show:` `Cotyledon` (green alpha contour) · `Hypocotyl` (yellow smoothed midline) · `Landmarks` (junction, axes and arrows of the current reading, model or manual) · `Germination` (magenta contour, off by default; only shown with the legacy `binary` backend, which still segments the radicle).
- `<--` / `-->` step through frames.
- `Zoom +` / `Zoom -` / `Reset zoom`, or the mouse wheel.

The raw image is shown at full intensity; the overlays are thin lines only, so you can judge the angle against the pixels. There is no mask editing. A reading drawn in **grey** was rejected by the readability gate; one drawn in **amber** broke a time-series check (see [Which angle method runs](#which-angle-method-runs)) and the line under the angle says which.

**Overriding the angle by hand (landmarks)**

`Place landmarks`, then click on the image, in this order:

1. **J**: the junction where the cotyledons meet the hypocotyl,
2. **H1**, **H2**: two points on the hypocotyl midline (the axis is oriented away from the junction, so their order does not matter),
3. **C1**: the cotyledon **base**, at the junction,
4. **C2**: the cotyledon **tip** (once the two cotyledons have separated, the midpoint between the two tips),
5. optional **collar**: the hypocotyl/root junction; it ends the length measurement there,
6. optional **root** point: a second point that gives the root direction.

The prompt in the status line names the next click. The angle is complete after C2; `Enter` applies the landmarks (the two optional points can be left out), `BackSpace` undoes the last point and `Escape` resets. Anywhere in the window, `←` / `→` step through the frames and `↑` / `↓` (or the `▲ Seedling` / `Seedling ▼` buttons) open the previous / next seedling at the same frame, in the same place on screen. Once C2 is placed, the `Overhook` box is **ticked automatically** when the cotyledon ray crosses the hypocotyl ray (the same test as for the model's reading); you can still tick or untick it. `Apply landmarks` commits the frame as **Manual** (an overhooked hook is stored as `180 + theta`, otherwise `180 - theta`), recomputes the hypocotyl traits with your collar (or the model's) and re-reconstructs the series around your frame. `Blank this frame` discards the frame's measurement entirely — use it when the seedling has rotated so the cotyledon sits in front of or behind the hypocotyl, since a 2D image cannot yield a correct angle in that case.

**Saving corrections for retraining.** Tick `Save edits for retraining` (you are asked for a folder once). Each `Apply landmarks` then writes a JSON record (the clicked points, the overhook tick, the model reading it replaced) and a copy of the crop to that folder. Convert the records into annotation CSVs and a crop folder that the landmark training code reads with

```bash
python -m multi.edits_to_csv path/to/landmark_edits --out-prefix angle_edits
```

and add the result to `landmarks.extra_sources` in the training config (see [Training a landmark head](#training-a-landmark-head-from-your-annotations)). A frame edited twice keeps its latest record.

**Germination**
- The detected frame is shown; `Set to this frame` overrides it, `Apply to all seedlings` propagates it, `Clear` removes it.

![Step 4 — the per-seedling analysis window](docs/img/Seedling_analysis_windows.png)


### 5. `Show kinematics`

Plots angle against time for all seedlings at once. Time is measured **since germination**, in hours when timestamps are available, otherwise in frames.

The legend gives the **State** (marker colour: Closed, Opening, Manual, Unclassified); a black ring marks a **temporal flag**. `Unreadable` frames have no angle and are not plotted. Each seedling has its own line colour; clicking a point opens that seedling at that frame. A **dashed** line means no germination frame was found for that seedling, so its time axis is not aligned to the others. The grey horizontal line marks 180° — a fully closed hook.

The plot does not auto-update; click `Refresh` after making corrections.


![Step 5 — kinematics plot](docs/img/9-repo.png)

### 6. `Export results`

Choose a destination and DLhook writes the `.CSV`. Any seedling never opened in step 4 is segmented now, so exporting is safe even if you only reviewed a subset. Germination detection and temporal reconstruction run as part of the export.

**Physical scale (optional).** Before exporting, type the image scale in **`Scale (µm/px)`** in the sidebar, or click **`Calibrate scale…`**, click the two ends of an object of known length on the first frame (a ruler, a grid square) and enter its length in mm. The scale is kept for the session. Without a scale the CSV holds pixel values only; with one it gains the `*_mm` columns below.

---

## Output

The exported CSV is in **long format** — one row per seedling per frame. The first seven columns are unchanged from earlier versions and keep their order; the landmark and hypocotyl columns follow them.

| Column | Description |
|---|---|
| `img_name` | source image filename |
| `seedling_id` | seedling number, 1-based |
| `raw_angle` | the unmodified per-frame geometric reading, for auditing |
| `state` | `Closed`, `Opening`, `Manual` (a hand-placed angle), or `Unreadable` (landmark frame rejected by the readability gate: no angle) |
| `bio_angle` | the reconstructed biological angle — **use this one** |
| `time_elapsed_min` | time (min) |
| `germination_frame` | frame at which germination of the seedling was detected (empty when unknown) |

**Landmark columns.** All positions are in **crop pixels**, in the pixel frame of that seedling's crop (origin top-left, x to the right, y down). Directions are **unit vectors** in the same image coordinates (y down). Cells are empty when the value does not exist for that frame (for example `collar_*` when no collar was found or clicked).

| Column | Description |
|---|---|
| `angle_source` | `landmarks` or `ellipse`: the method that produced the reading |
| `landmark_edit` | `model` (the network's reading) or `manual` (placed in the analysis window) |
| `landmark_peak` | strength of the junction heatmap peak |
| `overhook_flag` | 1 when the cotyledon ray crosses the hypocotyl ray (overhooked), else 0. It is informational: **it does not change `bio_angle`**, which stays `180 - theta` (a manual edit with `Overhook` ticked stores `180 + theta`) |
| `unreadable_p` | probability that the frame carries no usable information (readability gate; empty for manual frames) |
| `unreadable` | 1 when the gate rejected the frame (then `bio_angle` is empty and `state` is `Unreadable`), else 0 |
| `measurable_p` | the learned head's probability that an annotator could measure this frame (empty with models that lack the head) |
| `temporal_flag` | `collar`, `junction` or `collar+junction` when the frame broke a time-series check, else empty. Flag only: the angle is kept |
| `junction_x`, `junction_y` | the cotyledon/hypocotyl junction |
| `hypo_dir_x`, `hypo_dir_y` | direction of the hypocotyl axis, away from the junction |
| `cotyl_dir_x`, `cotyl_dir_y` | direction of the cotyledon axis, base to tip |
| `cotyl_base_x`, `cotyl_base_y`, `cotyl_tip_x`, `cotyl_tip_y` | cotyledon base and tip |
| `collar_x`, `collar_y` | hypocotyl/root junction (collar), when available |
| `collar_ref_x`, `collar_ref_y` | the seedling's stable collar around this frame (rolling median), the reference of the collar check |
| `root_dir_x`, `root_dir_y` | root direction, when available |

**Hypocotyl traits** (measured on the binary `hypocot_v5` hypocotyl mask; empty when there is no mask or no junction):

| Column | Description |
|---|---|
| `hypocotyl_length_px` | length in pixels, from the junction to the collar along the midline (skeleton path) of the hypocotyl mask. With no collar it runs to the end of the mask |
| `hypocotyl_length_truncated` | `True` when there is no collar and the midline touches the crop border, so the length is a lower bound |
| `hypocotyl_curvature_mean`, `hypocotyl_curvature_max` | mean and maximum of the absolute curvature \|κ\| of the smoothed midline, in 1/px (the first and last 5 px are dropped) |
| `hypocotyl_diameter_px` | diameter in pixels: the median width of transects perpendicular to the midline, taken every 5 px along its inner 80% |

**Millimetre columns**, written only when a scale is set: `hypocotyl_length_mm`, `hypocotyl_diameter_mm`, `hypocotyl_curvature_mean_per_mm`, `hypocotyl_curvature_max_per_mm`. The pixel columns are always written.

> [!NOTE]
> The hypocotyl traits are **not yet validated** against hand measurements. Treat absolute values with care until they have been compared with manual length and diameter readings.

```csv
img_name,seedling_id,raw_angle,state,bio_angle,time_elapsed_min,germination_frame,angle_source,landmark_edit,landmark_peak,overhook_flag,unreadable_p,unreadable,measurable_p,temporal_flag,junction_x,junction_y,...
Col0_001.tif,1,2.1,Closed,177.9,0,,landmarks,model,0.93,0,0.04,0,,,36.2,41.5,...
Col0_001.tif,2,4.8,Closed,175.2,0,,landmarks,model,0.91,0,0.07,0,,,30.8,38.0,...
Col0_002.tif,1,5.6,Closed,174.4,60,,landmarks,model,0.95,0,0.03,0,,junction,36.4,41.1,...
```

Missing or unmeasurable values are written as empty strings.

### Angle convention

`bio_angle` is a continuous scale on which **180° means a fully closed hook**, decreasing toward 0° as the cotyledon opens. An **overhooked** seedling — folded back past closed — reads **above** 180°. Reconstructed values are constrained to the biologically admissible band **0°–220°** (up to 40° of overhook past closed).

State is assigned with hysteresis to stop the label flickering at the transition: above **160°** the hook is treated as closed, below **150°** as opening, and the 150–160° band is a deadband. A seedling must stay below 150° for **5 consecutive frames** before the state commits to `Opening`.

---

## Ground-truth angle annotation

To check the automatic angles against your own measurements, measure a random sample of frames by hand with the annotation tool. It shows each seedling crop zoomed in and records the angle you measure.

```bash
python -m ui.angle_annotator --folder path/to/crops     # crops named {id}-crop-{frame}.png
python -m ui.angle_annotator --folder path/to/crops --per-seedling 15 --seed 1
```

DLhook writes the plain (not super-resolved) crops to `data/images/` during *Start Analysis*, but wipes `data/` at startup and exit, so copy them to a folder of your own first if you want to annotate across sessions.

You can also annotate the crops used to train the four-class model (`data/crops/cropped_training_set/`, made by `multi/recrop_plates.py`). Its `manifest.csv` is picked up automatically, which keeps each series' seedlings apart (`crop_id` repeats across series) and records the source frame as `img_name`. Add `--split val` to keep only images the four-class model never trained on, which is the unbiased set to validate it with:

```bash
python -m ui.angle_annotator --folder data/crops/cropped_training_set --split val
```

For each frame, click five points (the mouse wheel zooms at the cursor, and the zoom is kept between frames of the same seedling):

1. the **junction** where the cotyledons meet the hypocotyl,
2. two points on the **hypocotyl** axis,
3. two points on the **cotyledon** axis.

Each axis is oriented automatically away from the junction, so the order of the two points on an axis does not matter. Tick **Overhook** if the hook has folded back past closed. Press `Enter` to save and move on, `Backspace` to undo a point and `S` (or `Right` / *Next*) to skip a frame you cannot measure. Moving on from a frame you did not measure records it as **skipped**, so the CSV tells "looked at and rejected" apart from "not reached yet"; `Left` goes back without recording anything. If a CSV predates this behaviour, `python -m ui.angle_annotator --folder data/crops/cropped_training_set --split val --mark-unrecorded-skipped` records every frame of the same sample that has no row as skipped (use the same `--folder`, `--split`, `--per-seedling` and `--seed` as the session) and exits without opening the window.

To compare your measurements with the pipeline, run `python -m multi.validate_angles`. It runs the real pipeline with both backends on every crop of the seedlings you measured and prints the error per backend and per series (written to `multi/results/angle_validation/`).

Frames are a reproducible random sample (`--per-seedling`, `--seed`), grouped by seedling and shuffled within it so you cannot be guided by time order. Results go to `data/annotations/angles/angle_ground_truth.csv` in the repository (change with `--out`). The file is saved after every frame, and the tool resumes where you stopped. `bio_angle` uses the same convention as the exported CSV (180 = closed, decreasing as the hook opens, above 180 = overhooked), and `seedling_id` and `frame` match the crop file names (plus `series` and `img_name` for the training crops), so the two can be joined directly.

## Reviewing earlier annotations

`python -m ui.angle_reviewer` re-opens frames you already measured, to fix a mistake or to mark a frame where no angle can be measured, without touching the original CSV. Decisions go to `<csv>.review.csv`, and the training and scoring code applies it on top of the original (`utils/angle_review.py`).

```bash
# measured frames, the ones the model disagrees with most first
python -m multi.validate_landmarks --truth data/annotations/angles/angle_landmarks_train.csv --folder data/crops/cropped_training_set --checkpoints weights/landmarks/dlhook_landmarks_a.pt+weights/landmarks/dlhook_landmarks_b.pt+weights/landmarks/dlhook_landmarks_c.pt --per-crop-out multi/results/review_errors/train
python -m ui.angle_reviewer --folder data/crops/cropped_training_set --csv data/annotations/angles/angle_landmarks_train.csv --errors multi/results/review_errors/train/<per-crop CSV written above>

# frames you skipped after the seedling germinated (before germination there is nothing to measure)
python -m ui.angle_reviewer --folder data/crops/cropped_training_set --csv data/annotations/angles/angle_landmarks_train.csv --skipped --germination data/annotations/germination/germination_train.csv
```

Each frame opens with your five clicks. `Enter` keeps them (or saves five you re-clicked after `X`), `R` marks **rotation** (the cotyledons pass in front of or behind the hypocotyl, so no angle is measured), `N` marks **no angle** for another reason, `S` / `Right` moves on without recording anything. The tool resumes where you stopped.

## Training a landmark head from your annotations

The same clicks train the landmark head: it finds the cotyledon/hypocotyl **junction** and the two axis directions directly, which removes the direction ambiguity of a fitted ellipse. The landmark angle is always `180 - theta`. The network has no overhook output, because overhooked frames sit only a few degrees past 180° (median 6.5°), below the repeatability of the clicks themselves (your own repeat measurements differ by about 4° at the median, and the overhook call by 8%); the exported `overhook_flag` is a geometric test on the reading and does not change the angle.

**1. Annotate training frames** (never reuse your test set):

```bash
python -m ui.angle_annotator --folder data/crops/cropped_training_set --split train --out data/annotations/angles/angle_landmarks_train.csv --per-seedling 10 --seed 1
```

Aim for at least 400 measured frames spread over the series. Click the same way every time, because the model learns exactly what you click:

- **junction**: the joint where the cotyledons meet the hypocotyl;
- **hypocotyl** points: on the midline of the stem just below the junction;
- **cotyledon** points: on the midline of the cotyledon. The **first** click is its **base**, at the junction, and the **last** click is its **tip** (once the two cotyledons have separated, the midpoint between the two tips). The model reads the cotyledon direction as base to tip, which is what tells an open hook from a closed one;
- the **Overhook** tick is still recorded in the CSV but ignored by landmark training and by `multi/validate_angles.py` (which scores against `180 - theta`; `--keep-overhook` restores the raw angle).

`--split train` keeps your frames out of the validation split, and the training code drops any validation-split crop it finds in the CSV anyway, so `data/annotations/angles/angle_ground_truth.csv` stays a clean test set.

**2. Fine-tune** from the shipped four-class checkpoint (adds a landmark head, keeps the segmentation):

```bash
python multi/train_unet_multiclass.py --config multi/configs/training_landmarks.yaml
```

The first epochs train only the new head, then the whole network (`landmarks.head_only_epochs`). The best epoch is chosen by the angle error on held-out annotated seedlings, among epochs whose segmentation score stays above `landmarks.min_mean_fg_dice`. Checkpoints go to `multi/results/models_landmarks/`.

The shipped ensemble was trained with `multi/configs/training_landmarks_all_{a,b,c}.yaml`: every annotated set, the landmark head only (the segmentation stays the four-class model's), three seeds. The `_m` variants of those configs add the measurable output on top of a shipped member (`python -m multi.validate_measurable` scores it).

**3. Score it on your held-out ground truth**, next to the current pipeline:

```bash
python -m multi.validate_angles --backends multiclass --landmark-checkpoint multi/results/models_landmarks/best.pt
```

The landmark readings appear as `landmark` (per frame) and `landmark_recon` (after reconstruction). Several checkpoints joined by `+` are scored as an ensemble.

## Advanced

**Debug Mode** (checkbox, off by default) logs what the germination detector measured for each seedling.

**Console output.** DLhook logs through Python's `logging` module. `DLHOOK_LOG_LEVEL=DEBUG python main.py` also shows the per-frame details (contours, angles); `DLHOOK_LOG_LEVEL=WARNING` keeps only problems.

**Where files go.** Weights are found relative to the repository, so `python <path-to-DLhook>/main.py` works from any directory. The session's scratch files (crops, mask dumps) go to `data/` inside the repository, or to `DLHOOK_DATA_DIR` when that is set; they are wiped at start and exit.

**Running a series without the GUI.** With a boxes file saved from the GUI, `multi/run_series.py` runs the same code as *Start Analysis* and writes the export CSV, a summary, and one figure per seedling (angle, junction–collar distance and collar deviation, with unreadable and flagged frames marked):

```bash
python -m multi.run_series --images example_data/Plate_1_Old --boxes example_data/Plate_1_Old/crop_boxes.json --out multi/results/series_Plate_1_Old
```

**Dumping masks.** Masks are kept in memory. To also write them to `data/predict/` as PNGs:

```bash
DLHOOK_DUMP_MASKS=1 python main.py
```

**Tests.** About 570 tests cover the angle reconstruction, readability gate and time-series checks, capture-time parsing, germination detection, crop-box layout, segmentation backends and the `multi/` training pipeline. No GPU or image data is needed:

```bash
pytest
```

> [!CAUTION]
> The `torchvision.transforms.functional_tensor` shim at the top of `seedling_measurement.py` is load-bearing. It keeps `basicsr` / `realesrgan` importable on current torchvision versions — removing it breaks startup.

---

## Status & known limitations

DLhook v0.3.0 is a **pre-release**. It is usable and in active development, but the following are known and unresolved:

- **Germination timing is not validated.** Detection thresholds were tuned only on a few seedlings. The detected germination *frame* has not passed a proper quality check. Verify time zero manually for your own material.
- **Super-resolution is opt-in and adds a tiny fraction of detail** (`DLHOOK_SUPERRES=1`, CUDA only). The 4× output is immediately resampled back to native size, so there is nothing extra to zoom into.
- **Hypocotyl length, curvature and diameter are not validated.** They have not been compared with hand measurements yet.
- **The angle is slow on CPU.** Each frame costs three landmark networks with flip averaging (six forward passes).
- **The time-series thresholds were set on annotated series** (15 px collar jump, 5 px junction drop). Very different magnifications may need other values (`utils/landmark_temporal.py`).
- **GPU and CPU results are not bit-identical** — roughly 1 pixel in 86,000 (0.001%) can differ.

---

## Licence

DLhook is released under the **GNU General Public License v3.0** — see [`LICENSE.txt`](LICENSE.txt).

The segmentation network architecture and the trained weights derive from [RootPainter](https://github.com/Abe404/root_painter) (Copyright © 2019, 2020 Abraham George Smith), which is GPLv3; DLhook is therefore GPLv3 as well.

---

## Citation

Please cite this repository while we work on the manuscript describing the method (see also [`CITATION.cff`](CITATION.cff)):

> David Radianu, Sara Raggi, Siamsa Doyle, Hemamshu Ratnakaram, Sandeep Yadav, Stéphanie Robert\*, Adrien Heymans\* (2025). DLhook: High-throughput apical hook phenotyping of dark-grown Arabidopsis thaliana seedlings. v0.3.0.
