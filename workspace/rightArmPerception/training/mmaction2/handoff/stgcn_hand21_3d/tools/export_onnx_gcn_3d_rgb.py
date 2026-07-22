#!/usr/bin/env python3
"""Export hand21 3D STGCN RGB fusion models to ONNX.

Supports models that take:
  - skeleton keypoints
  - RGB hand ROI frames
  - global motion trajectory
  - handedness metadata
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from mmengine import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.structures import LabelData
from mmengine.utils import import_modules_from_strings

from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample

try:
    import onnxruntime
except ImportError:
    onnxruntime = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export 3D hand21 STGCN RGB fusion model to ONNX."
    )
    parser.add_argument("config", help="config file path")
    parser.add_argument("checkpoint", help="checkpoint file path")
    parser.add_argument("--num-frames", type=int, default=0)
    parser.add_argument("--num-joints", type=int, default=0)
    parser.add_argument("--num-channels", type=int, default=0)
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="RGB ROI input size; default infers DecodeVideoROI.target_size.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--opset-version", type=int, default=12)
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip ONNXRuntime verification after export.",
    )
    return parser.parse_args()


class AvgPool2d(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(-1, -2), keepdim=True)


class GCNRGBMotionNet(nn.Module):
    """ONNX wrapper for ``RecognizerGCNGlobalMotionHandednessRGB`` and its
    gated variant.

    Key adjustments vs. the runtime model:
      * ImageNet mean / std normalisation is baked into the graph so callers
        feed raw [0, 255] ``imgs``.
      * ``AdaptiveAvgPool2d`` is replaced with a static ``mean(dim=(-1,-2))``
        for cleaner ONNX export.
      * When the head is the **gated** variant (``GCNHead3DGlobalMotionHandednessRGBGated``),
        we:
          1. Force ``use_gate_at_inference = False`` so the hard NONE mask
             with +/- 1e9 logits never enters the graph (this would otherwise
             destroy INT8 quantisation scale).
          2. Compute the fused hidden vector once, return ``cls_score`` (clean
             10-class logits) **and** ``gate_logit`` (raw scalar) as two
             separate ONNX outputs.  The deployment side applies its own
             sigmoid threshold to force NONE - this keeps the threshold
             tunable post-export.
    """

    def __init__(self, base_model) -> None:
        super().__init__()
        self.backbone = base_model.backbone
        self.motion_branch = base_model.motion_branch
        self.rgb_branch = base_model.rgb_branch
        self.cls_head = base_model.cls_head

        if hasattr(self.rgb_branch, "gap") and isinstance(self.rgb_branch.gap, nn.AdaptiveAvgPool2d):
            self.rgb_branch.gap = AvgPool2d()
        if hasattr(self.cls_head, "pool") and isinstance(self.cls_head.pool, nn.AdaptiveAvgPool2d):
            self.cls_head.pool = AvgPool2d()

        preprocessor = getattr(base_model, "data_preprocessor", None)
        if preprocessor is not None and hasattr(preprocessor, "img_mean"):
            img_mean = preprocessor.img_mean.detach().clone()
            img_std = preprocessor.img_std.detach().clone()
        else:
            img_mean = torch.tensor((123.675, 116.28, 103.53), dtype=torch.float32).view(
                1, 1, 3, 1, 1
            )
            img_std = torch.tensor((58.395, 57.12, 57.375), dtype=torch.float32).view(
                1, 1, 3, 1, 1
            )
        self.register_buffer("img_mean", img_mean, persistent=False)
        self.register_buffer("img_std", img_std, persistent=False)
        self.use_handedness_score = getattr(self.cls_head, "score_proj", None) is not None

        # Detect the gated head and disable the hard NONE mask inside it.
        self.has_gate = hasattr(self.cls_head, "gate_head")
        if self.has_gate:
            # Forcing this off ensures even direct cls_head(...) calls inside
            # this wrapper would return clean logits without +/-1e9 masks.
            self.cls_head.use_gate_at_inference = False

    def forward(
        self,
        keypoint: torch.Tensor,
        imgs: torch.Tensor,
        handedness: torch.Tensor,
        global_traj: torch.Tensor,
        handedness_score: Optional[torch.Tensor] = None,
    ):
        imgs = (imgs.float() - self.img_mean) / self.img_std
        skel_feat = self.backbone(keypoint)
        motion_feat = self.motion_branch(global_traj)
        rgb_feat = self.rgb_branch(imgs)
        feats = (skel_feat, motion_feat, rgb_feat)

        if self.has_gate:
            # Compute hidden once, derive both outputs from it.  Bypasses the
            # head's forward() entirely so we are guaranteed not to bake any
            # threshold / masking logic into the graph.
            hidden = self.cls_head._compute_fused_hidden(
                feats,
                handedness=handedness,
                handedness_score=handedness_score if self.use_handedness_score else None,
            )
            cls_score = self.cls_head.fc2(self.cls_head.dropout(hidden))
            gate_logit = self.cls_head.gate_head(hidden)  # (N, 1)
            return cls_score, gate_logit

        return self.cls_head(
            feats,
            handedness=handedness,
            handedness_score=handedness_score if self.use_handedness_score else None,
        )


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x / x.sum(axis=-1, keepdims=True)


def find_first_transform_value(config: Config, key: str, transform_type: Optional[str] = None):
    for pipeline_name in ("test_pipeline", "val_pipeline", "train_pipeline"):
        pipeline = config.get(pipeline_name, None)
        if pipeline is None:
            continue
        for transform in pipeline:
            if transform_type is not None and transform.get("type") != transform_type:
                continue
            if key in transform:
                return transform[key]
    return None


def infer_num_frames(config: Config) -> int:
    clip_len = find_first_transform_value(config, key="clip_len")
    if clip_len is None:
        return 16
    if isinstance(clip_len, int):
        return int(clip_len)
    raise TypeError(f"Unsupported clip_len value: {clip_len}")


def infer_num_joints(config: Config) -> int:
    graph_cfg = config.model.backbone.graph_cfg
    layout = graph_cfg.layout
    if isinstance(layout, dict) and "num_node" in layout:
        return int(layout["num_node"])
    if isinstance(layout, str):
        lookup = {"openpose": 18, "nturgb+d": 25, "coco": 17}
        if layout not in lookup:
            raise KeyError(f'Unsupported layout "{layout}", please specify --num-joints')
        return lookup[layout]
    raise KeyError("Unable to infer joint count, please specify --num-joints")


def infer_num_channels(config: Config) -> int:
    return int(config.model.backbone.get("in_channels", 3))


def infer_img_size(config: Config) -> Tuple[int, int]:
    target_size = find_first_transform_value(
        config, key="target_size", transform_type="DecodeVideoROI"
    )
    if target_size is None:
        return (96, 96)
    if isinstance(target_size, (list, tuple)) and len(target_size) == 2:
        return (int(target_size[0]), int(target_size[1]))
    raise TypeError(f"Unsupported target_size value: {target_size}")


@contextmanager
def disable_torchvision_pretrained_download():
    try:
        import torchvision.models as tvm
    except ImportError:
        yield
        return

    original = tvm.mobilenet_v3_small

    def wrapped_mobilenet_v3_small(*args, **kwargs):
        kwargs.pop("weights", None)
        kwargs.pop("pretrained", None)
        try:
            return original(*args, weights=None, **kwargs)
        except TypeError:
            return original(*args, pretrained=False, **kwargs)

    tvm.mobilenet_v3_small = wrapped_mobilenet_v3_small
    try:
        yield
    finally:
        tvm.mobilenet_v3_small = original


def build_dummy_inputs(args: argparse.Namespace, config: Config, base_model):
    num_frames = args.num_frames or infer_num_frames(config)
    num_joints = args.num_joints or infer_num_joints(config)
    num_channels = args.num_channels or infer_num_channels(config)
    img_h, img_w = tuple(args.img_size or infer_img_size(config))
    motion_channels = int(base_model.motion_branch.in_channels)
    use_handedness_score = getattr(base_model.cls_head, "score_proj", None) is not None

    keypoint = torch.randn(1, num_frames, num_joints, num_channels).clamp(-3, 3)
    imgs = torch.rand(1, num_frames, 3, img_h, img_w) * 255.0
    global_traj = torch.zeros(1, num_frames, motion_channels, dtype=torch.float32)
    handedness = torch.tensor([1], dtype=torch.long)
    handedness_score = torch.tensor([1.0], dtype=torch.float32) if use_handedness_score else None
    return keypoint, imgs, global_traj, handedness, handedness_score


def build_data_sample(
    global_traj: torch.Tensor,
    handedness: torch.Tensor,
    handedness_score: Optional[torch.Tensor],
) -> ActionDataSample:
    meta = {"handedness_label_id": int(handedness.item())}
    if handedness_score is not None:
        meta["handedness_score"] = float(handedness_score.item())

    data_sample = ActionDataSample()
    data_sample.pred_scores = LabelData()
    data_sample.pred_labels = LabelData()
    data_sample.set_metainfo(meta)
    data_sample.set_field(global_traj.squeeze(0).cpu().numpy(), "global_traj")
    return data_sample


def verify_export(
    base_model,
    onnx_path: str,
    keypoint: torch.Tensor,
    imgs: torch.Tensor,
    global_traj: torch.Tensor,
    handedness: torch.Tensor,
    handedness_score: Optional[torch.Tensor],
    device: str,
    has_gate: bool = False,
) -> None:
    if onnxruntime is None:
        print("onnxruntime is not installed, skip verification.")
        return

    # NOTE: when the head is gated, base_model still has
    # use_gate_at_inference=True by default - so its predict() path may apply
    # the +/-1e9 NONE mask.  To compare apples-to-apples with the exported
    # ONNX (which we forced to be gate-free), temporarily disable the runtime
    # gate masking here too.
    saved_flag = None
    if has_gate and hasattr(base_model.cls_head, "use_gate_at_inference"):
        saved_flag = base_model.cls_head.use_gate_at_inference
        base_model.cls_head.use_gate_at_inference = False

    try:
        normalized_imgs = (
            imgs.to(device).float() - base_model.data_preprocessor.img_mean
        ) / base_model.data_preprocessor.img_std
        data_sample = build_data_sample(global_traj, handedness, handedness_score)

        with torch.no_grad():
            base_output = base_model(
                dict(
                    keypoint=keypoint.to(device).unsqueeze(1),
                    imgs=normalized_imgs,
                ),
                data_samples=[data_sample],
                mode="predict",
            )[0]
        base_output = base_output.pred_score.detach().cpu().numpy()
    finally:
        if saved_flag is not None:
            base_model.cls_head.use_gate_at_inference = saved_flag

    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_feed = {
        "keypoint": keypoint.cpu().numpy(),
        "imgs": imgs.cpu().numpy(),
        "handedness": handedness.cpu().numpy(),
        "global_traj": global_traj.cpu().numpy(),
    }
    if handedness_score is not None:
        input_feed["handedness_score"] = handedness_score.cpu().numpy()
    out_names = ["cls_score", "gate_logit"] if has_gate else ["cls_score"]
    outputs = session.run(out_names, input_feed)
    onnx_output = softmax(outputs[0])

    diff = np.abs(base_output - onnx_output[0]).max()
    print(f"Max probability diff between PyTorch and ONNX: {diff:.8f}")
    if has_gate:
        import scipy.special as _sp  # local import to keep deps optional
        try:
            gate_prob = float(_sp.expit(outputs[1].reshape(-1)[0]))
        except Exception:
            gate_prob = float(1.0 / (1.0 + np.exp(-outputs[1].reshape(-1)[0])))
        print(f"Gate output (sigmoid of gate_logit) on the dummy sample: {gate_prob:.4f}")


def main() -> None:
    args = parse_args()
    config = Config.fromfile(args.config)
    if "custom_imports" in config:
        import_modules_from_strings(**config["custom_imports"])
    init_default_scope(config.get("default_scope", "mmaction"))

    with disable_torchvision_pretrained_download():
        base_model = MODELS.build(config.model)
    load_checkpoint(base_model, args.checkpoint, map_location="cpu")
    base_model.to(args.device)
    base_model.eval()

    if config.model.type != "RecognizerGCNGlobalMotionHandednessRGB":
        print(
            "Warning: this script is tailored for "
            "RecognizerGCNGlobalMotionHandednessRGB models."
        )

    keypoint, imgs, global_traj, handedness, handedness_score = build_dummy_inputs(
        args, config, base_model
    )
    keypoint = keypoint.to(args.device)
    imgs = imgs.to(args.device)
    global_traj = global_traj.to(args.device)
    handedness = handedness.to(args.device)
    if handedness_score is not None:
        handedness_score = handedness_score.to(args.device)

    model = GCNRGBMotionNet(base_model).to(args.device)
    model.eval()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_inputs: Sequence[torch.Tensor]
    input_names = ["keypoint", "imgs", "handedness", "global_traj"]
    dynamic_axes = {
        "keypoint": {0: "batch_size", 1: "num_frames"},
        "imgs": {0: "batch_size", 1: "num_frames"},
        "handedness": {0: "batch_size"},
        "global_traj": {0: "batch_size", 1: "num_frames"},
        "cls_score": {0: "batch_size"},
    }

    output_names = ["cls_score"]
    if model.has_gate:
        output_names.append("gate_logit")
        dynamic_axes["gate_logit"] = {0: "batch_size"}
        print("Detected gated head; exporting `gate_logit` as a second output "
              "(deploy side applies its own sigmoid threshold).")

    if handedness_score is not None:
        export_inputs = (keypoint, imgs, handedness, global_traj, handedness_score)
        input_names.append("handedness_score")
        dynamic_axes["handedness_score"] = {0: "batch_size"}
    else:
        export_inputs = (keypoint, imgs, handedness, global_traj)

    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(export_inputs),
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            export_params=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=args.opset_version,
            dynamic_axes=dynamic_axes,
        )

    print(f"Successfully exported the ONNX file to {output_path}")

    if not args.skip_verify:
        verify_export(
            base_model,
            str(output_path),
            keypoint,
            imgs,
            global_traj,
            handedness,
            handedness_score,
            args.device,
            has_gate=model.has_gate,
        )


if __name__ == "__main__":
    main()
