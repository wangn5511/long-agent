"""
ONNX export script for MobileNetV2 (and other models in this repo).

本脚本参考 `onnx_export_v4.py`，用于将训练好的 PyTorch 模型导出为 ONNX。
默认支持本仓库中的 `mobilenetv2`，也可以导出 torchvision 的其它模型。
"""

import argparse
from collections import OrderedDict
from typing import Optional, Tuple, List

import numpy as np
import torch
import onnx
import torchvision.models as tv_models

import models.imagenet as customized_models


# 将自定义模型注册到 torchvision.models 字典中，方式与 `imagenet.py` 一致
default_model_names = sorted(
    name for name in tv_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(tv_models.__dict__[name])
)

customized_models_names = sorted(
    name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name])
)

for name in customized_models.__dict__:
    if (
        name.islower()
        and not name.startswith("__")
        and callable(customized_models.__dict__[name])
    ):
        tv_models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names


parser = argparse.ArgumentParser(description="PyTorch ONNX Deployment (MobileNetV2)")

parser.add_argument(
    "--output",
    metavar="ONNX_FILE",
    default=None,
    type=str,
    help="output model filename",
)

# Model & datasets params
parser.add_argument(
    "--model",
    default="mobilenetv2",
    type=str,
    metavar="MODEL",
    choices=model_names,
    help="Name of model to export",
)
parser.add_argument(
    "--checkpoint",
    default="./checkpoints_mbv2_truth/model_best.pth.tar",
    type=str,
    metavar="PATH",
    help="path to checkpoint",
)
parser.add_argument(
    "--batch-size",
    default=1,
    type=int,
    metavar="N",
    help="mini-batch size (default: 1)",
)
parser.add_argument(
    "--img-size",
    default=224,
    type=int,
    metavar="N",
    help="input image size (short side)",
)
parser.add_argument(
    "--nb-classes",
    type=int,
    default=8,
    help="number of classes in dataset",
)
parser.add_argument(
    "--width-mult",
    type=float,
    default=1.0,
    help="MobileNet width multiplier (only对支持该参数的模型生效)",
)

parser.add_argument(
    "--opset",
    type=int,
    default=10,
    help="ONNX opset to use (default: 10)",
)
parser.add_argument(
    "--keep-init",
    action="store_true",
    default=False,
    help="Keep initializers as input.",
)
parser.add_argument(
    "--aten-fallback",
    action="store_true",
    default=False,
    help="Fallback to ATEN ops.",
)
parser.add_argument(
    "--dynamic-size",
    action="store_true",
    default=False,
    help="Export model with dynamic height/width.",
)
parser.add_argument(
    "--dynamic-batch",
    action="store_true",
    default=False,
    help="Export model with dynamic batch dimension.",
)
parser.add_argument(
    "--check-forward",
    action="store_true",
    default=False,
    help="Check torch vs onnx forward after export.",
)
parser.add_argument(
    "--training",
    action="store_true",
    default=False,
    help="Export in training mode (default: eval).",
)
parser.add_argument(
    "--verbose",
    action="store_true",
    default=False,
    help="Extra stdout output during export.",
)


def onnx_forward(onnx_file, example_input):
    import onnxruntime

    sess_options = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(
        onnx_file, sess_options, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output = session.run([], {input_name: example_input.numpy()})
    output = output[0]
    return output


def onnx_export(
    model: torch.nn.Module,
    output_file: str,
    example_input: Optional[torch.Tensor] = None,
    training: bool = False,
    verbose: bool = False,
    check: bool = True,
    check_forward: bool = False,
    batch_size: int = 1,
    input_size: Tuple[int, int, int] = None,
    opset: Optional[int] = None,
    dynamic_size: bool = False,
    dynamic_batch: bool = False,
    aten_fallback: bool = False,
    keep_initializers: Optional[bool] = None,
    input_names: List[str] = None,
    output_names: List[str] = None,
):
    if training:
        training_mode = torch.onnx.TrainingMode.TRAINING
        model.train()
    else:
        training_mode = torch.onnx.TrainingMode.EVAL
        model.eval()

    if example_input is None:
        assert input_size is not None, "input_size must be provided"
        example_input = torch.randn((batch_size,) + input_size, requires_grad=training)

    # 先跑一遍前向，确保 padding 等状态固定
    with torch.no_grad():
        original_out = model(example_input)

    print("==> Exporting model to ONNX format at '{}'".format(output_file))

    input_names = input_names or ["input0"]
    output_names = output_names or ["output0"]

    dynamic_axes = {}
    if dynamic_batch:
        dynamic_axes = {"input0": {0: "batch"}, "output0": {0: "batch"}}
    if dynamic_size:
        if "input0" not in dynamic_axes:
            dynamic_axes["input0"] = {}
        dynamic_axes["input0"][2] = "height"
        dynamic_axes["input0"][3] = "width"

    if aten_fallback:
        export_type = torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    else:
        export_type = torch.onnx.OperatorExportTypes.ONNX

    torch_out = torch.onnx._export(
        model,
        example_input,
        output_file,
        training=training_mode,
        export_params=True,
        verbose=verbose,
        input_names=input_names,
        output_names=output_names,
        keep_initializers_as_inputs=keep_initializers,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        operator_export_type=export_type,
    )

    if check:
        print("==> Loading and checking exported model from '{}'".format(output_file))
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model, full_check=True)
        if check_forward and not training:
            onnx_out = onnx_forward(output_file, example_input)
            if torch_out is not None:
                np.testing.assert_almost_equal(
                    torch_out.numpy(), onnx_out, decimal=3
                )
                np.testing.assert_almost_equal(
                    original_out.numpy(), torch_out.numpy(), decimal=5
                )
            else:
                np.testing.assert_almost_equal(
                    original_out.numpy(), onnx_out, decimal=3
                )


def load_model_from_checkpoint(args):
    print("==> Creating PyTorch {} model".format(args.model))

    # 目前主要针对 mobilenetv2，兼容其它模型
    if args.model == "mobilenetv2":
        model = tv_models.__dict__[args.model](
            width_mult=args.width_mult, num_classes=args.nb_classes
        )
    else:
        # 其它模型一般只接受 num_classes
        model = tv_models.__dict__[args.model](num_classes=args.nb_classes)

    print("==> Loading checkpoint '{}'".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # 兼容几种常见的保存格式：直接 state_dict / {'state_dict': ...} / {'model': ...}
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            # 可能本身就是一个 state_dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def main():
    args = parser.parse_args()

    if args.output is None:
        args.output = f"./{args.model}.onnx"

    model = load_model_from_checkpoint(args)

    onnx_export(
        model=model,
        output_file=args.output,
        opset=args.opset,
        dynamic_size=args.dynamic_size,
        dynamic_batch=args.dynamic_batch,
        aten_fallback=args.aten_fallback,
        keep_initializers=args.keep_init,
        check_forward=args.check_forward,
        training=args.training,
        verbose=args.verbose,
        input_size=(3, args.img_size, args.img_size),
        batch_size=args.batch_size,
    )

    print("==> ONNX export finished: '{}'".format(args.output))


if __name__ == "__main__":
    main()


