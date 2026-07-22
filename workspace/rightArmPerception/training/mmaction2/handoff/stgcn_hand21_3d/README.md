STGCN hand21 deploy bundle

Contents
- `configs/stgcn_hand21-joint-u36-30e_custom-keypoint-2d.py`: training and inference config for the hand21 STGCN model.
- `weights/best_acc_top1_epoch_12.pth`: best checkpoint from the current training run.
- `code/backbones/stgcn.py`: STGCN backbone implementation.
- `code/utils/graph.py`: graph layout implementation used by STGCN.
- `code/recognizers/recognizer_gcn.py`: GCN recognizer wrapper.
- `code/heads/gcn_head.py`: classification head.
- `tools/export_onnx_gcn.py`: ONNX export script for skeleton GCN models.
- `meta/hand21_labels.txt`: class labels.

Recommended weight
- Use `weights/best_acc_top1_epoch_12.pth` for deployment.

Model input convention
- Input shape: `(batch, num_frames, num_joints, channels)`
- Current deployment setting: `(1, 36, 21, 3)`
- Channels are `(x, y, score)`.
