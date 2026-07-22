# STGCN Hand21 3D RGB WAVE_OUT Handoff

This bundle is a self-contained handoff package for the
`stgcn_3d_rgb_wave_out` experiment.

It includes:

- verified `model.onnx`
- the exact checkpoint used for this export
- a standalone flattened config with no `_base_` inheritance
- custom model, transform, preprocessor, and track-policy source files
- label and input metadata for deployment

## Source experiment

- Result directory:
  `/home/results/6t1/stgcn_3d_rgb_wave_out`
- Checkpoint:
  `weights/best_acc_top1_epoch_22.pth`
- ONNX:
  `model.onnx`

## Model interface

The exported ONNX model consumes four inputs:

1. `keypoint`
   shape `(1, 16, 21, 3)`, float32
2. `imgs`
   shape `(1, 16, 3, 96, 96)`, float32
3. `handedness`
   shape `(1,)`, int64
4. `global_traj`
   shape `(1, 16, 3)`, float32

This bundle does not use `handedness_score` at inference time because the
current head sets `use_handedness_score=False`.

The output is:

- `cls_score`
  shape `(1, 10)`

## Bundle layout

```text
stgcn_hand21_3d_rgb_wave_out/
  README.md
  model.onnx
  weights/
  configs/
  code/
  tools/
  meta/
```

## Configs

- `configs/stgcn_hand21-rgb-rgb48.py`
  is the standalone bundle config to use for export or reload.
- `configs/frozen_run_config.py`
  is the same flattened definition kept as a frozen snapshot artifact.

Both configs import custom modules from this handoff package, so they do not
depend on `handoff/stgcn_hand21_3d/`.

## Re-export ONNX

Run from:
`/home/liutianyi/rightArmPerception/training/mmaction2`

```bash
python handoff/stgcn_hand21_3d_rgb_wave_out/tools/export_onnx_gcn_3d_rgb.py \
  handoff/stgcn_hand21_3d_rgb_wave_out/configs/stgcn_hand21-rgb-rgb48.py \
  handoff/stgcn_hand21_3d_rgb_wave_out/weights/best_acc_top1_epoch_22.pth \
  --device cpu \
  --output-file handoff/stgcn_hand21_3d_rgb_wave_out/model.onnx
```

## Notes

- Skeleton input channels are `(x, y, z)`.
- The skeleton branch uses 21 joints and clip length 16.
- `imgs` expects hand ROI RGB clips before ImageNet mean/std normalization.
- `global_traj` is required by the global-motion branch and uses 3 channels.
- Track-aware sampling and ROI extraction are part of the training and
  evaluation pipeline, not the exported ONNX signature.
- Label order must match `meta/hand21_labels.txt` exactly.
