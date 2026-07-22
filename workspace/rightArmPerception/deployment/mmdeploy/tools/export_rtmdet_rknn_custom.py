import argparse
from pathlib import Path
from typing import Optional

from rknn.api import RKNN


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export ONNX model to RKNN format.')
    parser.add_argument('onnx_path', type=str, help='Path to input ONNX model')
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Path to output RKNN model. If not specified, will use '
             'the same directory and name as input with .rknn extension.')
    parser.add_argument(
        '--target',
        type=str,
        default='rk3588',
        help='Target platform (default: rk3588)')
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Enable INT8 quantization')
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Dataset file for INT8 quantization (required if --quantize)')
    parser.add_argument(
        '--optimization-level',
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help='Optimization level (default: 3)')
    return parser.parse_args()


def main():
    args = parse_args()

    onnx_path = Path(args.onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    # Determine output path
    if args.output:
        rknn_path = Path(args.output)
    else:
        rknn_path = onnx_path.with_suffix('.rknn')

    # Create output directory if needed
    rknn_path.parent.mkdir(parents=True, exist_ok=True)

    target = args.target
    quantize = args.quantize
    dataset_file: Optional[Path] = Path(args.dataset) if args.dataset else None

    if quantize and dataset_file is None:
        raise ValueError("--dataset is required when --quantize is enabled")

    print(
        f"[INFO] Building RKNN from {onnx_path.name} -> {rknn_path.name} "
        f"(target={target}, quantize={quantize})"
    )

    rknn = RKNN(verbose=True)

    cfg = dict(
        target_platform=target,
        sparse_infer=False,
        quantized_algorithm='mmse',
        quantized_method='channel',
        optimization_level=args.optimization_level,
        float_dtype='float16',           # 浮点部分使用 FP16
        single_core_mode=False,          # 启用多核 NPU
        remove_reshape=True,             # 合并冗余 Reshape 提高 NPU 覆盖率
    )
    ret = rknn.config(**cfg)
    if ret != 0:
        raise RuntimeError(f"RKNN config failed for {onnx_path}")

    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise RuntimeError(f"Failed to load ONNX: {onnx_path}")

    build_kwargs = dict(do_quantization=quantize)
    if quantize and dataset_file:
        build_kwargs['dataset'] = str(dataset_file)

    ret = rknn.build(**build_kwargs)
    if ret != 0:
        raise RuntimeError("Build model failed!")

    ret = rknn.export_rknn(rknn_path.as_posix())
    if ret != 0:
        raise RuntimeError("Export rknn model failed!")

    print(f"[INFO] RKNN model exported to {rknn_path}")


if __name__ == '__main__':
    main()
