#!/usr/bin/env python3
"""Export the 3D hand21 STGCN model with handedness inputs to ONNX."""

import argparse

import numpy as np
import onnxruntime
import torch
import torch.nn as nn
from mmengine import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.structures import LabelData
from mmengine.utils import import_modules_from_strings

from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample


def parse_args():
    parser = argparse.ArgumentParser(description="Export 3D hand21 STGCN to ONNX.")
    parser.add_argument("config", help="config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--num_joints", type=int, default=21)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_file", type=str, default="stgcn_hand21_3d.onnx")
    return parser.parse_args()


class AvgPool2d(nn.Module):
    def forward(self, x):
        return x.mean(dim=(-1, -2), keepdims=True)


class _GCNBase(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model.backbone
        self.head = base_model.cls_head
        self.motion_branch = getattr(base_model, "motion_branch", None)
        if hasattr(self.head, "pool") and isinstance(self.head.pool, nn.AdaptiveAvgPool2d):
            self.head.pool = AvgPool2d()


class _GCNScoreMotion(_GCNBase):
    """has_score=True, has_motion=True."""

    def forward(self, input_tensor, handedness, handedness_score, global_traj):
        feat = self.backbone(input_tensor)
        feat = (feat, self.motion_branch(global_traj))
        return self.head(feat, handedness=handedness, handedness_score=handedness_score)


class _GCNScore(_GCNBase):
    """has_score=True, has_motion=False."""

    def forward(self, input_tensor, handedness, handedness_score):
        feat = self.backbone(input_tensor)
        return self.head(feat, handedness=handedness, handedness_score=handedness_score)


class _GCNMotion(_GCNBase):
    """has_score=False, has_motion=True."""

    def forward(self, input_tensor, handedness, global_traj):
        feat = self.backbone(input_tensor)
        feat = (feat, self.motion_branch(global_traj))
        return self.head(feat, handedness=handedness, handedness_score=None)


class _GCNNeither(_GCNBase):
    """has_score=False, has_motion=False."""

    def forward(self, input_tensor, handedness):
        feat = self.backbone(input_tensor)
        return self.head(feat, handedness=handedness, handedness_score=None)


def _build_wrapper(base_model, has_score, has_motion):
    if has_score and has_motion:
        return _GCNScoreMotion(base_model)
    elif has_score:
        return _GCNScore(base_model)
    elif has_motion:
        return _GCNMotion(base_model)
    else:
        return _GCNNeither(base_model)


def softmax(x):
    x = np.exp(x - x.max())
    return x / x.sum()


def main():
    args = parse_args()
    config = Config.fromfile(args.config)
    if "custom_imports" in config:
        import_modules_from_strings(**config["custom_imports"])
    init_default_scope(config.get("default_scope", "mmaction"))

    base_model = MODELS.build(config.model)
    load_checkpoint(base_model, args.checkpoint, map_location="cpu")
    base_model.to(args.device)
    base_model.eval()

    has_score = getattr(base_model.cls_head, "score_proj", None) is not None
    has_motion = getattr(base_model, "motion_branch", None) is not None

    input_tensor = torch.randn(1, args.num_frames, args.num_joints, args.num_channels).clamp(-3, 3).to(args.device)
    handedness = torch.tensor([1], dtype=torch.long, device=args.device)
    global_traj_channels = base_model.motion_branch.in_channels if has_motion else 2
    global_traj = torch.zeros(1, args.num_frames, global_traj_channels, dtype=torch.float32, device=args.device)
    handedness_score = torch.tensor([1.0], dtype=torch.float32, device=args.device)

    meta = {"handedness_label_id": int(handedness.item())}
    if has_score:
        meta["handedness_score"] = float(handedness_score.item())
    data_sample = ActionDataSample()
    data_sample.pred_scores = LabelData()
    data_sample.pred_labels = LabelData()
    data_sample.set_metainfo(meta)
    if has_motion:
        data_sample.set_field(global_traj.squeeze(0).cpu().numpy(), "global_traj")

    base_output = base_model(
        input_tensor.unsqueeze(0), data_samples=[data_sample], mode="predict"
    )[0]
    base_output = base_output.pred_score.detach().cpu().numpy()

    model = _build_wrapper(base_model, has_score, has_motion)
    model.to(args.device)
    model.eval()

    export_inputs = [input_tensor, handedness]
    input_names = ["input_tensor", "handedness"]
    dynamic_axes = {
        "input_tensor": {0: "batch_size", 1: "num_frames"},
        "handedness": {0: "batch_size"},
        "cls_score": {0: "batch_size"},
    }
    input_feed = {
        "input_tensor": input_tensor.cpu().numpy(),
        "handedness": handedness.cpu().numpy(),
    }
    if has_score:
        export_inputs.append(handedness_score)
        input_names.append("handedness_score")
        dynamic_axes["handedness_score"] = {0: "batch_size"}
        input_feed["handedness_score"] = handedness_score.cpu().numpy()
    if has_motion:
        export_inputs.append(global_traj)
        input_names.append("global_traj")
        dynamic_axes["global_traj"] = {0: "batch_size", 1: "num_frames"}
        input_feed["global_traj"] = global_traj.cpu().numpy()

    torch.onnx.export(
        model,
        tuple(export_inputs),
        args.output_file,
        input_names=input_names,
        output_names=["cls_score"],
        export_params=True,
        do_constant_folding=True,
        verbose=False,
        opset_version=12,
        dynamic_axes=dynamic_axes,
    )

    session = onnxruntime.InferenceSession(args.output_file)
    outputs = session.run(["cls_score"], input_feed=input_feed)
    output = softmax(outputs[0][0])
    diff = abs(base_output - output).max()
    print(f"Successfully exported ONNX to {args.output_file}")
    print(f"Max output diff: {diff:.8f}")


if __name__ == "__main__":
    main()
