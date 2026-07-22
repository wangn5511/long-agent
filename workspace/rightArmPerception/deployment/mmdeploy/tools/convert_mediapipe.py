import argparse
import re
from typing import List

from rknn.api import RKNN


def _infer_format(path: str) -> str:
    p = path.lower()
    if p.endswith(".onnx"):
        return "onnx"
    if p.endswith(".tflite"):
        return "tflite"
    raise ValueError(f"Cannot infer format from file extension: {path}")


def _get_onnx_input_names(model_path: str) -> List[str]:
    """Get ONNX graph input names excluding initializers (weights)."""
    import onnx

    m = onnx.load(model_path)
    initializer_names = {init.name for init in m.graph.initializer}
    names = []
    for inp in m.graph.input:
        if inp.name not in initializer_names:
            names.append(inp.name)
    return names


def _patch_onnx_conv2d_attrs(model_path: str, save_path: str) -> str:
    """Patch some ONNX Conv attributes for older ORT shape-inference used inside RKNN.

    Error seen in rknn-toolkit2 build (fold_constant via onnxruntime):
      Op (Conv) [ShapeInferenceError] Attribute dilations has incorrect size

    Some exporters produce Conv `dilations`/`strides` with wrong length for 2D conv
    (e.g. length 1 or 4). ORT expects length==2 for 2D Conv.
    """
    import onnx

    m = onnx.load(model_path)
    changed = False

    def _fix(vals: List[int]) -> List[int]:
        if len(vals) == 2:
            return vals
        if len(vals) == 0:
            return [1, 1]
        if len(vals) == 1:
            v = int(vals[0])
            return [v, v]
        if len(vals) == 4:
            # [h,w,h,w] -> [h,w]
            if vals[0] == vals[2] and vals[1] == vals[3]:
                return [int(vals[0]), int(vals[1])]
            # all equal -> [v,v]
            if vals[0] == vals[1] == vals[2] == vals[3]:
                v = int(vals[0])
                return [v, v]
        # if all equal, reduce safely
        if len(vals) > 2 and all(v == vals[0] for v in vals):
            v = int(vals[0])
            return [v, v]
        # fallback: keep original (may still fail)
        return vals

    for node in m.graph.node:
        if node.op_type != "Conv":
            continue
        for attr in node.attribute:
            if attr.name not in ("dilations", "strides"):
                continue
            old = list(attr.ints)
            new = _fix(old)
            if new != old:
                attr.ints[:] = new
                changed = True

    if changed:
        onnx.save(m, save_path)
        print(f"[WARN] Patched ONNX saved to: {save_path}")
        return save_path
    return model_path


def _parse_triplet(s: str):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated numbers, got: {s}")
    return [float(parts[0]), float(parts[1]), float(parts[2])]


def convert_model(
    model_path: str,
    rknn_path: str,
    fmt: str = "auto",
    target: str = "rk3588",
    quantize: bool = False,
    dataset: str = "",
    is_image_model: bool = True,
    mean_values=None,
    std_values=None,
    inputs=None,
    input_size_list=None,
):
    """Convert ONNX/TFLite to RKNN (rk3588 by default).

    Notes:
    - rknn-toolkit2 may require `inputs`/`input_size_list` for ONNX models with
      dynamic dims (e.g. batch='batch').
    - If `quantize=True`, `dataset` is required by rknn-toolkit2.
    """
    if fmt == "auto":
        fmt = _infer_format(model_path)
    fmt = fmt.lower()
    if fmt not in ("onnx", "tflite"):
        raise ValueError(f"Unsupported format: {fmt}")

    # Retry loop: if RKNN build fails due to ORT Conv attr issue, patch ONNX and retry once.
    model_path_to_use = model_path
    for attempt in range(2):
        rknn = RKNN(verbose=True)
        try:
            cfg = dict(target_platform=target)
            # Use image-style normalization (0..255 -> 0..1) unless overridden.
            if is_image_model:
                cfg["mean_values"] = mean_values if mean_values is not None else [[0, 0, 0]]
                cfg["std_values"] = std_values if std_values is not None else [[255, 255, 255]]

            cfg["quantized_dtype"] = "w8a8"

            ret = rknn.config(**cfg)
            if ret != 0:
                raise RuntimeError(f"rknn.config failed: ret={ret}")

            # 2) Load model
            if fmt == "onnx":
                load_kwargs = {"model": model_path_to_use}
                if input_size_list is not None:
                    load_kwargs["input_size_list"] = input_size_list
                if inputs is not None:
                    load_kwargs["inputs"] = inputs
                elif input_size_list is not None:
                    names = _get_onnx_input_names(model_path_to_use)
                    if names:
                        load_kwargs["inputs"] = [names[0]]
                        print(f"[INFO] Auto-detected ONNX input: {names[0]}")
                try:
                    ret = rknn.load_onnx(**load_kwargs)
                except ValueError as e:
                    msg = str(e)
                    m = re.search(
                        r"The input shape \['batch',\s*(\d+),\s*(\d+),\s*(\d+)\] of '([^']+)' is not support",
                        msg)
                    if m:
                        c, h, w, in_name = int(m.group(1)), int(m.group(2)), int(
                            m.group(3)), m.group(4)
                        if "inputs" not in load_kwargs:
                            load_kwargs["inputs"] = [in_name]
                        if "input_size_list" not in load_kwargs:
                            load_kwargs["input_size_list"] = [[c, h, w]]
                        print(
                            f"[WARN] Dynamic batch dim detected. Auto-set inputs={load_kwargs['inputs']} "
                            f"input_size_list={load_kwargs['input_size_list']}")
                        ret = rknn.load_onnx(**load_kwargs)
                    elif "in inputs=" in msg and "is invalid" in msg and input_size_list is not None:
                        names = _get_onnx_input_names(model_path_to_use)
                        if not names:
                            raise
                        load_kwargs["inputs"] = [names[0]]
                        print(f"[WARN] Invalid --input_name. Auto-detected ONNX input: {names[0]}")
                        ret = rknn.load_onnx(**load_kwargs)
                    else:
                        raise
            else:
                load_kwargs = {"model": model_path_to_use}
                if inputs is not None:
                    load_kwargs["inputs"] = inputs
                if input_size_list is not None:
                    load_kwargs["input_size_list"] = input_size_list
                ret = rknn.load_tflite(**load_kwargs)

            if ret != 0:
                raise RuntimeError(f"Failed to load {fmt} model: {model_path_to_use} (ret={ret})")

            # 3) Build
            try:
                if quantize:
                    if not dataset:
                        raise ValueError("quantize=True requires --dataset /path/to/dataset.txt")
                    ret = rknn.build(do_quantization=True, dataset=dataset)
                else:
                    ret = rknn.build(do_quantization=False)
            except Exception as e:
                msg = str(e)
                if fmt == "onnx" and attempt == 0 and "dilations has incorrect size" in msg:
                    patched = model_path.rsplit(".", 1)[0] + "_patched.onnx"
                    model_path_to_use = _patch_onnx_conv2d_attrs(model_path, patched)
                    print("[WARN] RKNN build failed due to Conv attr issue; retrying with patched ONNX...")
                    continue
                raise

            if ret != 0:
                raise RuntimeError(f"Build model failed! (ret={ret})")

            # 4) Export
            ret = rknn.export_rknn(rknn_path)
            if ret != 0:
                raise RuntimeError(f"Export rknn model failed! (ret={ret})")
            return
        finally:
            try:
                rknn.release()
            except Exception:
                pass

    raise RuntimeError("Conversion failed after retrying with patched ONNX.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Input .onnx or .tflite")
    parser.add_argument("--format", type=str, default="auto", choices=["auto", "onnx", "tflite"])
    parser.add_argument("--out", type=str, required=True, help="Output .rknn path")

    parser.add_argument("--target", type=str, default="rk3588")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--dataset", type=str, default="", help="Required when --quantize (path to dataset.txt)")

    parser.add_argument("--is_image_model", action="store_true",
                        help="If set, configure mean/std (default 0/255) during conversion")
    parser.add_argument("--mean", type=str, default="", help="Override mean, format: r,g,b (e.g. 0,0,0)")
    parser.add_argument("--std", type=str, default="", help="Override std, format: r,g,b (e.g. 255,255,255)")

    # For dynamic-shape ONNX, you often must specify these:
    parser.add_argument("--input_name", type=str, default="", help="Input name (default: input)")
    parser.add_argument(
        "--input_hw",
        type=str,
        default="",
        help=("Input H,W (e.g. 256,256). Used to build input_size_list. "
              "If --input_layout=chw -> [3,H,W]; if --input_layout=hwc -> [H,W,3]."))
    parser.add_argument("--input_layout", type=str, default="chw", choices=["chw", "hwc"])
    args = parser.parse_args()

    mean_values = None
    std_values = None
    if args.mean:
        mean_values = [_parse_triplet(args.mean)]
    if args.std:
        std_values = [_parse_triplet(args.std)]

    inputs = None
    input_size_list = None
    if args.input_hw:
        h, w = [int(x.strip()) for x in args.input_hw.split(",")]
        # If user doesn't specify --input_name, we will auto-detect from ONNX.
        inputs = [args.input_name] if args.input_name else None
        if args.input_layout == "chw":
            input_size_list = [[3, h, w]]
        else:
            input_size_list = [[h, w, 3]]

    convert_model(
        model_path=args.model,
        rknn_path=args.out,
        fmt=args.format,
        target=args.target,
        quantize=args.quantize,
        dataset=args.dataset,
        is_image_model=args.is_image_model,
        mean_values=mean_values,
        std_values=std_values,
        inputs=inputs,
        input_size_list=input_size_list,
    )
