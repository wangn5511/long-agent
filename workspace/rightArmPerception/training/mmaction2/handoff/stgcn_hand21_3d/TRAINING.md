# STGCN Hand21 3D Training Guide

This document summarizes the training workflow for the `handoff/stgcn_hand21_3d`
bundle and organizes the notes from `3d_stgcn.txt` into a repository-friendly
reference.

## Scope

This guide covers the following training variants:

- baseline 3D skeleton model
- core-aware sampling model
- global-motion model
- global-motion + SupCon model
- RGB fusion model
- RGB fusion + SupCon model

Unless otherwise stated, commands below are run from:

```bash
/home/liutianyi/rightArmPerception/training/mmaction2
```

## Environment

### Docker image

From:

```bash
/home/liutianyi/rightArmPerception
```

Build the training image:

```bash
DOCKER_BUILDKIT=1 docker build \
  -f Dockerfile.stgcn_openmmlab \
  -t stgcn_openmmlab:v1 \
  .
```

Create the container:

```bash
docker run -itd \
  --name stgcn_3d \
  --restart unless-stopped \
  --gpus all \
  --shm-size=100g \
  -v /home/liutianyi/rightArmPerception:/home/liutianyi/rightArmPerception:rw \
  -v /home/results/6t1:/home/results/6t1:rw \
  -v /home/datasets:/home/datasets:rw \
  stgcn_openmmlab:v1 \
  tail -f /dev/null
```

Enter the container:

```bash
docker exec -it stgcn_3d bash
source /opt/miniconda3/etc/profile.d/conda.sh
```

Conda environments:

- data preparation: `conda activate mp_py310`
- training: `conda activate openmmlab`

## Model Family

Config inheritance for the main 3D training line:

```text
stgcn_hand21-joint-u16-30e_class9-keypoint-3d.py
  -> stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware.py
    -> stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion.py
      -> stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion_supcon.py
      -> stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion_rgb.py
        -> stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion_rgb_supcon.py
```

The custom modules used by these configs live under:

- `code/backbones/`
- `code/heads/`
- `code/recognizers/`
- `code/transforms/`
- `code/utils/`

## Data Preparation

### Dataset assumptions

Current data preparation is based on these assumptions:

1. Reuse already-cut clip mp4 files.
2. Do not recut clips from long raw videos.
3. Re-run MediaPipe 3D extraction on the existing clips.
4. Write 3D annotations, prepared json files, and pkl files under
   `/home/results/6t1/gesture_class9_3d`.
5. The first 3D baseline reuses the clip set already filtered in the previous
   2D stage.

### Per-frame fields

Each prepared clip is expected to preserve:

1. `track_id`
2. `handedness_label`
3. `handedness_score`
4. `world_landmarks`
5. `world_landmarks_smooth`
6. `kpts_xy`
7. `kpts_xy_smooth`
8. `score`
9. `bbox`
10. `bbox_smooth`
11. `frame_idx`
12. detection marker

### Video-level train/val split

```bash
python data/hand21_3d/generate_hand21_video_split.py \
  --video-root /home/results/6t1/annotation \
  --annotation-file /home/results/6t1/annotation/annotations.csv \
  --output-json /home/results/6t1/video_split/video_split.json \
  --annotation-format compact_4col \
  --frame-base 1 \
  --val-ratio 0.2 \
  --seed 42 \
  --extra-video-root /home/results/6t1/dynamic_gestures_260413_NONE
```

### Annotation coverage check

```bash
python data/hand21_3d/report_annotation_video_name_coverage.py \
  --annotation-file /home/results/6t1/annotation/annotations.csv \
  --video-root /home/results/6t1/annotation \
  --output-csv /home/results/6t1/annotation/annotation_video_name_coverage_report.csv \
  --summary-txt /home/results/6t1/annotation/annotation_video_name_coverage_summary.txt \
  --annotated-missing-txt /home/results/6t1/annotation/annotated_but_missing_on_disk.txt \
  --unannotated-videos-txt /home/results/6t1/annotation/unannotated_video_names.txt
```

### Reused clip roots

Recommended clip roots to reuse directly or via symlink:

- `/home/results/6t1/gesture_class9/gesture_class9_prepared_train_segments_videos`
- `/home/results/6t1/gesture_class9/gesture_class9_prepared_val_segments_videos`
- `/home/results/6t1/gesture_class9/none_prepared_train_segments_videos`
- `/home/results/6t1/gesture_class9/none_prepared_val_segments_videos`

### Prepare 3D clip json files

Gesture train:

```bash
python data/hand21_3d/prepare_hand21_from_existing_clips_3d.py \
  --prepared-root /home/results/6t1/gesture_class9/gesture_class9_prepared_train \
  --clip-root /home/results/6t1/gesture_class9/gesture_class9_prepared_train_segments_videos \
  --output-root /home/results/6t1/gesture_class9_3d/gesture_class9_prepared_train \
  --save-world-landmarks \
  --save-handedness \
  --default-feature-source world_landmarks_smooth \
  --jobs 4
```

Gesture val:

```bash
python data/hand21_3d/prepare_hand21_from_existing_clips_3d.py \
  --prepared-root /home/results/6t1/gesture_class9/gesture_class9_prepared_val \
  --clip-root /home/results/6t1/gesture_class9/gesture_class9_prepared_val_segments_videos \
  --output-root /home/results/6t1/gesture_class9_3d/gesture_class9_prepared_val \
  --save-world-landmarks \
  --save-handedness \
  --default-feature-source world_landmarks_smooth \
  --jobs 4
```

NONE train:

```bash
python data/hand21_3d/prepare_hand21_from_existing_clips_3d.py \
  --prepared-root /home/results/6t1/gesture_class9/none_prepared_train \
  --clip-root /home/results/6t1/gesture_class9/none_prepared_train_segments_videos \
  --output-root /home/results/6t1/gesture_class9_3d/none_prepared_train \
  --save-world-landmarks \
  --save-handedness \
  --default-feature-source world_landmarks_smooth \
  --jobs 4
```

NONE val:

```bash
python data/hand21_3d/prepare_hand21_from_existing_clips_3d.py \
  --prepared-root /home/results/6t1/gesture_class9/none_prepared_val \
  --clip-root /home/results/6t1/gesture_class9/none_prepared_val_segments_videos \
  --output-root /home/results/6t1/gesture_class9_3d/none_prepared_val \
  --save-world-landmarks \
  --save-handedness \
  --default-feature-source world_landmarks_smooth \
  --jobs 4
```

Filter label whitelist:

```bash
python data/hand21_3d/filter_prepared_hand21_by_label_whitelist.py \
  --prepared-roots \
    /home/results/6t1/gesture_class9_3d/gesture_class9_prepared_train \
    /home/results/6t1/gesture_class9_3d/gesture_class9_prepared_val \
  --allowed-labels \
    WAVE_IN FIVE CLICK ROTATE SLIDE_DOWN SLIDE_UP OK CURSOR NONE \
  --cleanup-files
```

### Convert prepared json to MMAction2 pkl

The training pkl keeps raw world landmarks and defers normalization to the
training and inference pipelines. It may also include `video_path` and
`bbox_per_frame` for RGB models.

```bash
python data/hand21_3d/convert_hand21_json_to_pkl_3d.py \
  --train-root /home/results/6t1/gesture_class9_3d/gesture_class9_prepared_train \
  --val-root /home/results/6t1/gesture_class9_3d/gesture_class9_prepared_val \
  --train-extra-roots /home/results/6t1/gesture_class9_3d/none_prepared_train \
  --val-extra-roots /home/results/6t1/gesture_class9_3d/none_prepared_val \
  --feature-source world_landmarks_smooth \
  --include-handedness \
  --output /home/results/6t1/gesture_class9_3d/gesture_class9_hand21_3d.pkl
```

## Feature and Sampling Conventions

### Skeleton branch

The 3D skeleton branch uses `world_landmarks_smooth` with shape `(21, 3)`.

Normalization convention:

1. wrist-relative: subtract joint `0` from every joint
2. scale normalize: divide by the distance from wrist to joint `9`
3. no global rotation normalization
4. training and inference must use the same math definition

### Global-motion branch

The global-motion branch uses a 2D palm-center trajectory:

1. `palm_center_xy[t] = mean(kpts_xy_smooth[t][0, 5, 9, 13, 17])`
2. `global_traj[t] = palm_center_xy[t] - palm_center_xy[first_valid]`
3. no hand-scale normalization is applied here; coordinates are already image-normalized

### Core-aware CLICK sampling

For CLICK, training uses online onset anchoring instead of offline core labels.

Key idea:

- energy is computed from wrist-relative displacement
- a 3-frame smoothing window is applied
- onset is the first frame satisfying
  `E_smooth[t] >= onset_threshold * max(E_smooth)`
- the clip window starts at `onset - pre_context + jitter`

Important parameters used in the notes:

- `core_classes=(2,)`
- `peak_smooth_window=3`
- `onset_threshold=0.5`
- `abs_energy_floor=0.003`
- `pre_context=4`
- `jitter_range=2`

The current checked-in configs may tune some of these values differently, so
prefer the config file as the final source of truth for an actual run.

## Training Commands

### Baseline

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 \
bash tools/dist_train.sh \
  handoff/stgcn_hand21_3d/configs/stgcn_hand21-joint-u16-30e_class9-keypoint-3d.py \
  4 \
  --work-dir /home/results/6t1/stgcn9_hand21_u16_3d_v2
```

### Core-aware

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 \
bash tools/dist_train.sh \
  handoff/stgcn_hand21_3d/configs/stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware.py \
  4 \
  --work-dir /home/results/6t1/stgcn9_hand21_u16_3d_core
```

### Global-motion

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 \
bash tools/dist_train.sh \
  handoff/stgcn_hand21_3d/configs/stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion.py \
  4 \
  --work-dir /home/results/6t1/stgcn9_hand21_u16_3d_motion
```

### Global-motion + SupCon

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 \
bash tools/dist_train.sh \
  handoff/stgcn_hand21_3d/configs/stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion_supcon.py \
  4 \
  --work-dir /home/results/6t1/stgcn9_hand21_u16_3d_motion_supcon
```

### RGB fusion

Before the first RGB run, rebuild the dataset so that `clip_video_path`,
`video_path`, and `bbox_per_frame` are present.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 \
bash tools/dist_train.sh \
  handoff/stgcn_hand21_3d/configs/stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion_rgb.py \
  4 \
  --work-dir /home/results/6t1/stgcn9_hand21_u16_3d_motion_rgb
```

### RGB fusion + SupCon

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 \
bash tools/dist_train.sh \
  handoff/stgcn_hand21_3d/configs/stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion_rgb_supcon.py \
  4 \
  --work-dir /home/results/6t1/stgcn9_hand21_u16_3d_motion_rgb_supcon
```

## Evaluation Commands

### Global-motion

```bash
python tools/test.py \
  handoff/stgcn_hand21_3d/configs/stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion.py \
  /home/results/6t1/stgcn9_hand21_u16_3d_motion/best_acc_top1_epoch_*.pth \
  --work-dir /home/results/6t1/stgcn9_hand21_u16_3d_motion
```

### Global-motion + SupCon

```bash
python tools/test.py \
  handoff/stgcn_hand21_3d/configs/stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion_supcon.py \
  /home/results/6t1/stgcn9_hand21_u16_3d_motion_supcon/best_acc_top1_epoch_*.pth \
  --work-dir /home/results/6t1/stgcn9_hand21_u16_3d_motion_supcon
```

### RGB fusion

```bash
python tools/test.py \
  handoff/stgcn_hand21_3d/configs/stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion_rgb.py \
  /home/results/6t1/stgcn9_hand21_u16_3d_motion_rgb/best_acc_top1_epoch_*.pth \
  --work-dir /home/results/6t1/stgcn9_hand21_u16_3d_motion_rgb
```

### RGB fusion + SupCon

```bash
python tools/test.py \
  handoff/stgcn_hand21_3d/configs/stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion_rgb_supcon.py \
  /home/results/6t1/stgcn9_hand21_u16_3d_motion_rgb_supcon/best_acc_top1_epoch_*.pth \
  --work-dir /home/results/6t1/stgcn9_hand21_u16_3d_motion_rgb_supcon
```

### Inspect misclassified samples

```bash
python handoff/stgcn_hand21_3d/tools/collect_misclassified_samples.py \
  --work-dir /home/results/6t1/stgcn9_hand21_u16_3d_motion \
  --save-overlay-video \
  --save-sampled-overlay-video \
  --skip-missing \
  --skip-none-misclassified
```

## RGB Dataset Rebuild

RGB models require `clip_video_path` in prepared json files and
`video_path` plus `bbox_per_frame` in the exported pkl.

### Step 1: rerun prepare

Use the data environment:

```bash
conda activate mp_py310
```

Gesture train:

```bash
python data/hand21_3d/prepare_hand21_from_existing_clips_3d.py \
  --prepared-root /home/results/6t1/gesture_class9/gesture_class9_prepared_train \
  --clip-root /home/results/6t1/gesture_class9/gesture_class9_prepared_train_segments_videos \
  --output-root /home/results/6t1/gesture_class9_3d/gesture_class9_prepared_train \
  --save-world-landmarks \
  --save-handedness \
  --default-feature-source world_landmarks_smooth \
  --jobs 4
```

Gesture val:

```bash
python data/hand21_3d/prepare_hand21_from_existing_clips_3d.py \
  --prepared-root /home/results/6t1/gesture_class9/gesture_class9_prepared_val \
  --clip-root /home/results/6t1/gesture_class9/gesture_class9_prepared_val_segments_videos \
  --output-root /home/results/6t1/gesture_class9_3d/gesture_class9_prepared_val \
  --save-world-landmarks \
  --save-handedness \
  --default-feature-source world_landmarks_smooth \
  --jobs 8
```

NONE train:

```bash
python data/hand21_3d/prepare_hand21_from_existing_clips_3d.py \
  --prepared-root /home/results/6t1/gesture_class9/none_prepared_train \
  --clip-root /home/results/6t1/gesture_class9/none_prepared_train_segments_videos \
  --output-root /home/results/6t1/gesture_class9_3d/none_prepared_train \
  --save-world-landmarks \
  --save-handedness \
  --default-feature-source world_landmarks_smooth \
  --jobs 8
```

NONE val:

```bash
python data/hand21_3d/prepare_hand21_from_existing_clips_3d.py \
  --prepared-root /home/results/6t1/gesture_class9/none_prepared_val \
  --clip-root /home/results/6t1/gesture_class9/none_prepared_val_segments_videos \
  --output-root /home/results/6t1/gesture_class9_3d/none_prepared_val \
  --save-world-landmarks \
  --save-handedness \
  --default-feature-source world_landmarks_smooth \
  --jobs 8
```

### Step 2: regenerate pkl

```bash
python data/hand21_3d/convert_hand21_json_to_pkl_3d.py \
  --train-root /home/results/6t1/gesture_class9_3d/gesture_class9_prepared_train \
  --val-root /home/results/6t1/gesture_class9_3d/gesture_class9_prepared_val \
  --train-extra-roots /home/results/6t1/gesture_class9_3d/none_prepared_train \
  --val-extra-roots /home/results/6t1/gesture_class9_3d/none_prepared_val \
  --feature-source world_landmarks_smooth \
  --include-handedness \
  --output /home/results/6t1/gesture_class9_3d/gesture_class9_hand21_3d.pkl
```

### Step 3: train RGB models

Enter the training container and environment:

```bash
docker exec -it stgcn_3d bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab
cd /home/liutianyi/rightArmPerception/training/mmaction2
```

RGB:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 \
bash tools/dist_train.sh \
  handoff/stgcn_hand21_3d/configs/stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion_rgb.py \
  4 \
  --work-dir /home/results/6t1/stgcn9_hand21_u16_3d_motion_rgb
```

RGB + SupCon:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 \
bash tools/dist_train.sh \
  handoff/stgcn_hand21_3d/configs/stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion_rgb_supcon.py \
  4 \
  --work-dir /home/results/6t1/stgcn9_hand21_u16_3d_motion_rgb_supcon
```

## ONNX Export

Example baseline export command:

```bash
python handoff/stgcn_hand21_3d/tools/export_onnx_gcn_3d.py \
  handoff/stgcn_hand21_3d/configs/stgcn_hand21-joint-u16-30e_class9-keypoint-3d.py \
  /home/results/6t1/stgcn9_hand21_u16_3d/best_acc_top1_epoch_*.pth \
  --num_frames 16 \
  --num_joints 21 \
  --num_channels 3 \
  --device cpu \
  --output_file /home/results/6t1/stgcn9_hand21_u16_3d/model.onnx
```

For the deployment-specific handoff of the global-motion + SupCon model, also
see `GLOBAL_MOTION_SUPCON_HANDOFF.md`.

## Notes

- For actual parameter values, prefer the checked-in config files over any older
  planning notes.
- `SupCon` affects training only; deployment inference still uses the normal
  classification output.
- RGB models depend on extra fields in the prepared json and pkl files. If those
  fields are missing, rebuild the dataset before training.
