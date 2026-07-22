# Global-Motion SupCon Handoff

This document defines the minimal complete handoff bundle for the
`stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion_supcon.py`
model so that downstream engineers can:

1. Load the custom model structure from `.pth`
2. Re-export ONNX if needed
3. Convert ONNX to RKNN
4. Verify input/output conventions without relying on hidden repo context

## Target model

- Config:
  `configs/stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion_supcon.py`
- Inference branches:
  - skeleton STGCN branch
  - global-motion branch
  - handedness metadata input
- Note:
  `SupCon` only affects training. Deployment inference does not consume the
  contrastive loss or projection head outputs directly, but the head source is
  still needed if `.pth` must be reloaded and re-exported.

## Required bundle

### 1. Weights and config

- `weights/best_acc_top1_epoch_XX.pth`
- `configs/stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion_supcon.py`
- `configs/stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion.py`
- `configs/stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware.py`
- `configs/stgcn_hand21-joint-u16-30e_class9-keypoint-3d.py`

Reason:
- The supcon config inherits from the global-motion config
- The global-motion config inherits from the core-aware config
- The core-aware config inherits from the base 3D class9 config

### 2. Custom model source code

- `code/backbones/stgcn.py`
- `code/backbones/global_motion_branch.py`
- `code/heads/gcn_motion_head.py`
- `code/recognizers/recognizer_gcn_motion.py`
- `code/utils/graph.py`

Also include package init files so custom imports resolve cleanly:
- `__init__.py`
- `code/__init__.py`
- `code/backbones/__init__.py`
- `code/heads/__init__.py`
- `code/recognizers/__init__.py`
- `code/utils/__init__.py`

Reason:
- These files define all custom modules referenced by the config and ONNX export
  path for this model

### 3. ONNX export tool

- `tools/export_onnx_gcn_3d.py`

Reason:
- This script already knows how to export models with:
  - `handedness`
  - `handedness_score`
  - `global_traj`

### 4. Label and input metadata

- `meta/hand21_labels.txt`
- `meta/hand21_label_mapping.json`
- `meta/model_input_requirements.json`

Important:
- The existing `model_input_requirements.json` may need to be accompanied by a
  short override note if deployment should follow the latest "no interpolation"
  policy.

## Strongly recommended extras

### 5. Exported ONNX

- `model.onnx`

Reason:
- If ONNX is already verified, downstream RKNN conversion can proceed without
  rebuilding the PyTorch graph first

### 6. Calibration samples for RKNN

Provide 100-500 representative samples covering:
- `CLICK`
- `CURSOR`
- `NONE`
- common confusion pairs

Recommended format per sample:
- `input_tensor.npy`
- `global_traj.npy`
- `handedness.npy`
- `handedness_score.npy`
- `expected_top1.txt`

### 7. Smoke test samples

Provide 3-10 tiny regression samples for conversion validation.

Suggested per sample:
- `input_tensor.npy`
- `global_traj.npy`
- `handedness.npy`
- `handedness_score.npy`
- `expected_logits.npy` or `expected_top1.txt`

## Input contract

The deployed model expects 4 inputs:

### 1. `input_tensor`

- Shape: `(1, 16, 21, 3)`
- Meaning:
  3D hand keypoints from `world_landmarks_smooth`
- Channel order:
  `(x, y, z)`

### 2. `global_traj`

- Shape: `(1, 16, 2)`
- Meaning:
  palm-center 2D trajectory sequence
- Definition:
  `global_traj[t] = palm_center_xy[t] - palm_center_xy[first_valid]`

### 3. `handedness`

- Shape: `(1,)`
- Type: integer
- Mapping:
  - `0 = Left`
  - `1 = Right`
  - `2 = Unknown`

### 4. `handedness_score`

- Shape: `(1,)`
- Type: float

## Output contract

- Output name:
  `cls_score`
- Shape:
  `(1, 9)`
- Postprocess:
  `argmax(cls_score, dim=-1)`

## ONNX export command

Run from:
`/home/liutianyi/rightArmPerception/training/mmaction2`

```bash
python handoff/stgcn_hand21_3d/tools/export_onnx_gcn_3d.py \
  handoff/stgcn_hand21_3d/configs/stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion_supcon.py \
  /home/results/6t1/stgcn9_hand21_u16_3d_motion_supcon/best_acc_top1_epoch_*.pth \
  --num_frames 16 \
  --num_joints 21 \
  --num_channels 3 \
  --device cpu \
  --output_file /home/results/6t1/stgcn9_hand21_u16_3d_motion_supcon/model.onnx
```

## Recommended delivered directory tree

```text
global_motion_supcon_handoff/
  README.md
  model.onnx
  weights/
    best_acc_top1_epoch_XX.pth
  configs/
    stgcn_hand21-joint-u16-30e_class9-keypoint-3d.py
    stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware.py
    stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion.py
    stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion_supcon.py
  code/
    __init__.py
    backbones/
      __init__.py
      stgcn.py
      global_motion_branch.py
    heads/
      __init__.py
      gcn_motion_head.py
    recognizers/
      __init__.py
      recognizer_gcn_motion.py
    utils/
      __init__.py
      graph.py
  tools/
    export_onnx_gcn_3d.py
  meta/
    hand21_labels.txt
    hand21_label_mapping.json
    model_input_requirements.json
  calib_samples/
  smoke_tests/
```

## What can be omitted

If the downstream team only needs `ONNX -> RKNN` and will never reload `.pth`,
the following are optional:

- training-only experiment docs
- misclassification analysis tools
- RGB-branch code
- dataset preparation scripts
- `gcn_motion_rgb_head.py`
- `recognizer_gcn_motion_rgb.py`

## Final recommendation

For deployment collaboration, deliver both:

1. a minimal inference artifact set:
   - `model.onnx`
   - labels
   - input spec
   - calibration samples

2. a reproducibility artifact set:
   - config inheritance chain
   - custom source modules
   - checkpoint
   - export script

This avoids getting blocked later if ONNX must be regenerated or RKNN
conversion exposes a graph issue.
