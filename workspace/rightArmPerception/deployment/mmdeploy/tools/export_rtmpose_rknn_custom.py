import argparse
from pathlib import Path
from typing import List, Optional

from rknn.api import RKNN


# --------- ONNX patch helpers ----------
def _fix_conv_2d_attr_list(model_path: Path, save_path: Path) -> Path:
    """Patch some ONNX Conv attributes for older ORT shape-inference used inside RKNN.

    Error seen in rknn-toolkit2 build (fold_constant via onnxruntime):
      Op (Conv) [ShapeInferenceError] Attribute dilations has incorrect size

    Some exported models may contain `dilations` / `strides` with length 1 for a 2D Conv.
    ORT expects length == spatial_rank (2). This function normalizes them to length 2.
    """
    import onnx

    model = onnx.load(model_path.as_posix())
    changed = False
    bad_conv_attrs = []

    def _try_fix_2d_list(name: str, vals: List[int]) -> Optional[List[int]]:
        """Return fixed 2-length list if confidently reducible, else None."""
        if len(vals) == 2:
            return vals
        if len(vals) == 0:
            return [1, 1]
        if len(vals) == 1:
            v = int(vals[0])
            return [v, v]
        # some exporters mistakenly include extra dims, e.g. [1,1,1,1]
        if len(vals) == 4:
            # [h,w,h,w] -> [h,w]
            if vals[0] == vals[2] and vals[1] == vals[3]:
                return [int(vals[0]), int(vals[1])]
            # all equal -> [v,v]
            if vals[0] == vals[1] == vals[2] == vals[3]:
                v = int(vals[0])
                return [v, v]
        # if all values equal, reduce safely
        if len(vals) > 2 and all(v == vals[0] for v in vals):
            v = int(vals[0])
            return [v, v]
        return None

    for node in model.graph.node:
        if node.op_type != 'Conv':
            continue
        for attr in node.attribute:
            if attr.name not in ('dilations', 'strides'):
                continue
            vals = list(attr.ints)
            fixed = _try_fix_2d_list(attr.name, vals)
            if fixed is None:
                if len(vals) != 2:
                    bad_conv_attrs.append((node.name, attr.name, vals))
                continue
            if vals != fixed:
                attr.ints[:] = fixed
                changed = True

    # Always save a patched copy if any change is made, OR if we detected a suspicious attr
    # (so users can inspect and we can diff easily).
    if changed or bad_conv_attrs:
        if bad_conv_attrs:
            print("[WARN] Found Conv attrs with non-2 length that could not be safely fixed:")
            for n, a, v in bad_conv_attrs[:20]:
                print(f"  - node={n or '<unnamed>'} attr={a} vals={v}")
            if len(bad_conv_attrs) > 20:
                print(f"  ... ({len(bad_conv_attrs) - 20} more)")
        onnx.save(model, save_path.as_posix())
        print(f"[INFO] Patched ONNX saved to: {save_path}")
        return save_path

    # no changes
    return model_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export RTMPose ONNX model to RKNN format.')
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
        '--input-name',
        type=str,
        default='input',
        help='Input tensor name (default: input)')
    parser.add_argument(
        '--input-size',
        type=int,
        nargs=3,
        default=[3, 256, 256],
        metavar=('C', 'H', 'W'),
        help='Input size as C H W (default: 3 256 256)')
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
    parser.add_argument(
        '--skip-patch',
        action='store_true',
        help='Skip ONNX Conv attribute patching')
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

    # Patch ONNX for RKNN's internal ORT fold_constant if needed.
    if not args.skip_patch:
        patched_onnx_path = onnx_path.with_name(onnx_path.stem + "_patched.onnx")
        onnx_to_load = _fix_conv_2d_attr_list(onnx_path, patched_onnx_path)
    else:
        onnx_to_load = onnx_path

    # The exported ONNX often keeps a dynamic batch dim (e.g. ['batch', 3, 256, 256]).
    # rknn-toolkit2 requires a concrete input_size_list for static build.
    ret = rknn.load_onnx(
        model=str(onnx_to_load),
        inputs=[args.input_name],
        input_size_list=[args.input_size],
    )
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
