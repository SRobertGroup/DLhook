# Multiclass U-Net training pipeline

Trains one 4-class segmentation model (`0 background / 1 cotyledon / 2 hypocotyl / 3 radicle`)
to replace the three binary RootPainter models the GUI runs today. Standalone and config-driven
— the GUI does not import anything in this folder.

Training labels are produced by running the three shipped binary models over hand-cropped
seedlings and merging their probability maps, so this model **distils three teachers into one**.
Read its evaluation as "does it match the teachers", not "does it beat them". The one route to
actually beating them is fine-tuning on human strokes (not yet done).

## Environment

Everything runs in `dlhook_env`.

```bash
mamba activate dlhook_env
```

If activating isn't possible, call the interpreter with the DLL directories on PATH — a bare
absolute `python.exe` misses `Library/bin` and crashes on the numpy/OpenCV/torch native loads:

```bash
E="C:/Users/heymansad/AppData/Local/miniconda3/envs/dlhook_env"
export PATH="$E:$E/Library/bin:$E/Library/mingw-w64/bin:$E/Library/usr/bin:$E/Scripts:$PATH"
"$E/python.exe" multi/train_unet_multiclass.py --config multi/configs/training_config.yaml
```

All commands below are run **from the repo root**, not from `multi/`.

## Full pipeline, in order

### 1. Draw crop boxes (interactive, once per series)

```bash
python -m multi.draw_crops --example-data example_data --out data/crop_boxes/crop_boxes.json
```

Tk tool that loads the last frame of each `example_data/<series>/` folder so you can draw one
box per seedling. Resumable — it reloads `data/crop_boxes/crop_boxes.json` on startup and saves atomically on
every edit, so you can close it and come back.

Controls: drag empty space = new box · drag box body = move · drag a corner = resize · click =
select · `Delete` = remove selected · `←`/`→` = previous/next series · `r` = reset zoom ·
wheel = zoom.

### 2. Cut the crops out of every Nth frame

```bash
# always dry-run first: reports per-series counts and writes nothing
python multi/recrop_plates.py --boxes-file data/crop_boxes/crop_boxes.json --output-dir data/crops/cropped_training_set --every-n 5 --dry-run
python multi/recrop_plates.py --boxes-file data/crop_boxes/crop_boxes.json --output-dir data/crops/cropped_training_set --every-n 5
```

Produces `data/crops/cropped_training_set/` (flat, one PNG per crop) plus `manifest.csv`. Consecutive
time-series frames are near-duplicates, so only every 5th frame is cropped — plus the last
frame of each series always, since that is the frame the boxes were drawn on.

Frames are sampled in **time order** (EXIF where usable, natural sort otherwise), not
lexicographic order — `example_data/MB/` filenames aren't zero-padded, so plain sorting would
pick the wrong frames.

Add `--series NAME1,NAME2` to restrict to specific series (see "Adding one new series" below).
Refuses to overwrite existing crop files unless `--overwrite` is passed.

### 3. Run the three binary teachers over every crop

```bash
python multi/segment_crops.py --crops data/crops/cropped_training_set --out data/crops/cropped_training_set_seg
```

Writes `{crop_stem}-{cotyledon,hypocotyl,radicle}.png` uint8 probability maps, **255 = HIGH
probability**. This is the opposite of the repo's legacy on-disk mask convention
(`MaskStore.dump()` writes 0 = foreground) — do not invert these.

Resumable: a crop whose outputs all exist is skipped, so re-running only processes new crops.
Add `--cpu` to force CPU inference, `--classes radicle` to run a single class (see below).

Weight files are hardcoded in `segment_crops.py`'s `CLASS_WEIGHTS`, **not** read from the
config's `weights_*` keys. Changing a teacher means editing both.

### 4. Merge the probability maps into multiclass labels

```bash
python multi/merge_masks.py --config multi/configs/training_config.yaml --from-probs data/crops/cropped_training_set_seg --preview
```

Produces `data/crops/cropped_training_set_labels/` (values `0,1,2,3`, plus `255` = ignore), colorized
previews, and `multi/processed/mask_merge_report.csv`. `--from-probs` is the disk-fed path: pure
numpy, no torch, no GPU.

Each class has `(t_lo, t_hi)` thresholds in the config. Above `t_hi` → that class; below `t_lo`
→ background; in between → `ignore` (255), which the loss skips. Where two classes both claim a
pixel, the higher probability wins unless they are within `conflict_margin`, in which case the
pixel becomes `ignore`.

Skips already-merged labels; pass `--force` to redo them (needed after regenerating any
probability map). Note the report CSV is **not cumulative** — it only covers images processed in
that invocation, so it can't be diffed against a whole-corpus baseline after a partial re-run.

### 5. Build the train/val patch index

```bash
python multi/build_patch_index.py --config multi/configs/training_config.yaml
```

Writes `multi/processed/patch_index/{train,val}_patches.csv`. The split is **per image**
(`split_level: image`), so patches from one crop never straddle train and val.

### 6. Train

```bash
python multi/train_unet_multiclass.py --config multi/configs/training_config.yaml
python multi/train_unet_multiclass.py --config multi/configs/training_config.yaml --resume   # from last.pt
```

Writes `multi/results/models/{best,last}.pt` and a per-epoch CSV to `multi/results/logs/`, with
per-class IoU and Dice columns.

Best-checkpoint selection and early stopping both use `training.selection_metric`, default
`mean_fg_dice` (mean Dice over classes 1–3, excluding background). Do **not** select on
`val_loss` here: background is ~96% of pixels, so loss is dominated by it — on a measured run
val_loss improved 44% between two epochs while cotyledon Dice got *worse*.

**Use `--resume` only to continue an interrupted run of the same data.** If the labels changed
(new teacher weights, new series, different merge thresholds), start fresh instead — resuming
carries over optimizer and LR-schedule state fitted to the old label set.

### 7. Evaluate against human annotations

```bash
python multi/evaluate_multiclass.py --config multi/configs/training_config.yaml --model multi/results/models/best.pt
python multi/evaluate_multiclass.py --config multi/configs/training_config.yaml --baseline-only
```

Scores against the held-out RootPainter annotations in `Root_Painter_Sync/` (paths under the
config's `evaluation:` block, absolute and machine-specific). `--baseline-only` scores just the
three binary teachers, and works before any multiclass checkpoint exists — run both so the
student and its teachers are compared on identical pixels.

Only **defined** pixels count: RootPainter marks foreground and background explicitly and leaves
the rest undefined, and undefined pixels are excluded from every count. Reports micro *and*
macro precision/recall plus a count of images below 0.05 recall — micro alone hides whole-image
misses, which is exactly how the radicle model fails.

## Common partial re-runs

**Swapping one teacher's weights** (e.g. adopting a retrained radicle model):

```bash
# 1. copy the new round in, keep the old file for reproducibility
cp Root_Painter_Sync/dlhook_germ/models/000022_1789547787.pkl weights/RootPainter_weights/germ_v2.pkl
# 2. point BOTH the config's weights_radicle key AND segment_crops.py's CLASS_WEIGHTS at it
# 3. delete just that class's probability maps, then regenerate only that class
rm data/crops/cropped_training_set_seg/*-radicle.png
python multi/segment_crops.py --crops data/crops/cropped_training_set --out data/crops/cropped_training_set_seg --classes radicle
# 4. re-merge (--force, since the labels already exist), re-index, retrain from scratch
python multi/merge_masks.py --config multi/configs/training_config.yaml --from-probs data/crops/cropped_training_set_seg --preview --force
python multi/build_patch_index.py --config multi/configs/training_config.yaml
python multi/train_unet_multiclass.py --config multi/configs/training_config.yaml
```

`--classes` matters here: plain `--force` reruns **all three** models for any crop missing an
output, rewriting the other two classes' PNGs (and their mtimes) for no reason.

**Adding one new series** (boxes already drawn for it in `data/crop_boxes/crop_boxes.json`):

```bash
python multi/recrop_plates.py --boxes-file data/crop_boxes/crop_boxes.json --series 75mM --output-dir data/crops/cropped_training_set --every-n 5 --dry-run
python multi/recrop_plates.py --boxes-file data/crop_boxes/crop_boxes.json --series 75mM --output-dir data/crops/cropped_training_set --every-n 5
python multi/segment_crops.py --crops data/crops/cropped_training_set --out data/crops/cropped_training_set_seg   # skips the old crops
python multi/merge_masks.py --config multi/configs/training_config.yaml --from-probs data/crops/cropped_training_set_seg --preview
python multi/build_patch_index.py --config multi/configs/training_config.yaml
python multi/train_unet_multiclass.py --config multi/configs/training_config.yaml
```

Steps 3–5 need no `--force`: each skips what already exists and only processes the new crops. A
`--series`-restricted re-crop preserves the other series' manifest rows rather than replacing the
whole file.

## Config knobs that actually matter

`multi/configs/training_config.yaml`:

- `data.patch_size: 252` — sized to the **data**, not the inference tile. The crops are narrow
  strips (median 74×246 px); at the old 572 every crop was narrower than the patch, so a median
  patch was ~94% reflect-padding. Must be a valid `UNetGNRes` size (`models/unet.py:
  get_valid_patch_sizes`, i.e. `572 - 16*i`).
- `loss.type: focal` with `focal_alpha` — the merged labels are ~96% background, so plain
  cross-entropy scores 96% by predicting background everywhere and collapses the rare classes.
- `training.selection_metric: mean_fg_dice` — see step 6.
- `model.warm_start_from` — optionally start from a shipped binary encoder instead of random
  init; every layer except the `n_classes`-shaped `conv_out` is loaded (true for either head —
  the skip is by key prefix).
- `model.head` — `groupnorm` (default, and what every shipped RootPainter checkpoint was trained
  with) or `plain`. `groupnorm` is `Conv2d → ReLU → GroupNorm(n_classes, n_classes)`: one group
  per channel, i.e. each class's logits are instance-normalised per image, so the network cannot
  express "this class is absent here" and the ReLU clamps away "confidently not this class".
  `plain` is a bare 1×1 `Conv2d`. The two heads have **different state_dict keys** (`groupnorm`
  adds `conv_out.2.{weight,bias}`) and are not checkpoint-compatible; since checkpoints here are
  bare state_dicts with no metadata, the head is inferred from the keys at load time
  (`multi/src/model.py:head_from_state_dict`), so `evaluate_multiclass.py --model <any best.pt>`
  works for either without extra flags.
- `data.class_thresholds` / `conflict_margin` — the merge bands from step 4.

Note `loss.class_weights` is only honoured on the `cross_entropy` path; under `type: focal` it is
silently ignored.

## Where things stand

See [session_recap.md](session_recap.md) for the current results table and the open problems.
Short version as of 2026-09-17: the model matches its pseudo-label teachers well
(`mean_fg_dice` 0.868) but against real human annotations its **precision** is far below theirs
(0.156 hypocotyl, 0.379 cotyledon vs 0.90+), i.e. it over-predicts foreground. The leading
structural suspect is `conv_out`'s `GroupNorm(n_classes, n_classes)` in `models/unet.py`, which
instance-normalises each class channel per image and so cannot express "this class is absent
here".

## Tests

```bash
pytest tests/ -q
```

The trainer's own tests run against tiny synthetic fixtures — none of the commands above are
exercised end-to-end by the suite, since they need the real data.
