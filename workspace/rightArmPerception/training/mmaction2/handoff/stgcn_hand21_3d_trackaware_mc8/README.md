# STGCN Hand21 3D Track-Aware MC8 Handoff

This bundle is a self-contained handoff package for the
`stgcn_3d_supcon_label_trackaware_mc8` experiment.

It includes:

- verified `model.onnx`
- the exact checkpoint used for this export
- a standalone flattened config with no `_base_` inheritance
- custom model, transform, and track-policy source files required by the config
- label and input metadata for deployment

## Source experiment

- Result directory:
  `/home/results/6t1/stgcn_3d_supcon_label_trackaware_mc8`
- Checkpoint:
  `weights/best_acc_top1_epoch_21.pth`
- ONNX:
  `model.onnx`

## Model interface

The exported ONNX model consumes three inputs:

1. `input_tensor`
   shape `(1, 16, 21, 3)`, float32
2. `handedness`
   shape `(1,)`, int64
3. `global_traj`
   shape `(1, 16, 3)`, float32

This bundle does not use `handedness_score` at inference time because the
current head sets `use_handedness_score=False`.

The output is:

- `cls_score`
  shape `(1, 9)`

## Bundle layout

```text
stgcn_hand21_3d_trackaware_mc8/
  README.md
  model.onnx
  weights/
  configs/
  code/
  tools/
  meta/
```

## Configs

- `configs/stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion_supcon_track-aware_mc8.py`
  is the standalone bundle config to use for export or reload.
- `configs/frozen_run_config.py`
  is the same flattened definition kept as a frozen snapshot artifact.

Both configs import custom modules from this handoff package, so they do not
depend on `handoff/stgcn_hand21_3d/`.

## Re-export ONNX

Run from:
`/home/liutianyi/rightArmPerception/training/mmaction2`

```bash
python handoff/stgcn_hand21_3d_trackaware_mc8/tools/export_onnx_gcn_3d.py \
  handoff/stgcn_hand21_3d_trackaware_mc8/configs/stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion_supcon_track-aware_mc8.py \
  handoff/stgcn_hand21_3d_trackaware_mc8/weights/best_acc_top1_epoch_21.pth \
  --num_frames 16 \
  --num_joints 21 \
  --num_channels 3 \
  --device cpu \
  --output_file handoff/stgcn_hand21_3d_trackaware_mc8/model.onnx
```

## Notes

- Skeleton input channels are `(x, y, z)`.
- The skeleton branch uses 21 joints and clip length 16.
- `global_traj` is required by the global-motion branch and uses 3 channels.
- Track-aware sampling is part of the training and evaluation pipeline, not the exported ONNX signature.
- Label order must match `meta/hand21_labels.txt` exactly.
