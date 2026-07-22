#!/usr/bin/env python3
"""ONNX vs RKNN-simulator consistency check for the 4-input gated RGB+skeleton
fusion model.

Iterates through the calibration manifest, runs each sample through
onnxruntime and the RKNN simulator, and reports:
  * per-sample top1 match
  * top3 score diffs
  * gate_logit (sigmoid) agreement
  * overall top1 match rate

Defaults to the layout produced by
``build_stgcn_3d_rgb_gated_calibration.py`` (4 inputs, 2 outputs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from rknn.api import RKNN


LABELS = [
    "WAVE_IN", "WAVE_OUT", "FIVE", "CLICK", "ROTATE",
    "SLIDE_DOWN", "SLIDE_UP", "OK", "CURSOR", "NONE",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", required=True, help="Path to model.onnx")
    p.add_argument(
        "--dataset",
        required=True,
        help="Path to calibration manifest (4 input paths per row). "
             "Used both as calibration source AND verification samples.",
    )
    p.add_argument(
        "--quantized-method",
        default="channel",
        help="passed to rknn.config (must match RKNN YAML).",
    )
    p.add_argument(
        "--auto-hybrid",
        action="store_true",
        help="Enable auto_hybrid in rknn.build (demotes sensitive layers "
             "back to FP16).  Match the YAML's build.auto_hybrid setting.",
    )
    p.add_argument(
        "--quantized-algorithm",
        default="normal",
        choices=["normal", "mmse", "kl_divergence"],
        help="Calibration algorithm passed to rknn.config (match YAML).",
    )
    p.add_argument("--summary-json", default=None,
                   help="Optional path to summary.json for per-sample ground-truth labels.")
    p.add_argument("--max-samples", type=int, default=0,
                   help="Cap the number of samples (0 = all in the manifest).")
    p.add_argument("--target", default="rk3588")
    p.add_argument(
        "--output",
        default=None,
        help="Optional path to write a per-sample JSON report.",
    )
    return p.parse_args()


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    e = np.exp(x - x.max())
    return e / e.sum()


def topk(probs: np.ndarray, k: int = 3):
    ids = np.argsort(probs)[::-1][:k]
    return [(int(i), LABELS[int(i)], float(probs[int(i)])) for i in ids]


def main() -> None:
    args = parse_args()

    # ---- load manifest + (optional) summary labels ----
    rows = [
        line.strip().split()
        for line in Path(args.dataset).read_text().splitlines()
        if line.strip()
    ]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError("manifest has no rows")
    if len(rows[0]) != 4:
        raise RuntimeError(
            f"expected 4 columns per row (keypoint imgs handedness global_traj), "
            f"got {len(rows[0])}"
        )

    gt_by_idx: dict[int, int] = {}
    if args.summary_json:
        smry = json.loads(Path(args.summary_json).read_text())
        for s in smry.get("samples", []):
            gt_by_idx[int(s["dataset_idx"])] = int(s["label"])

    # ---- ONNX session ----
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    onnx_input_names = [i.name for i in sess.get_inputs()]
    onnx_output_names = [o.name for o in sess.get_outputs()]
    print(f"[onnx] inputs:  {onnx_input_names}")
    print(f"[onnx] outputs: {onnx_output_names}")
    if onnx_input_names[:4] != ["keypoint", "imgs", "handedness", "global_traj"]:
        raise RuntimeError(f"unexpected ONNX input order: {onnx_input_names}")

    # ---- RKNN simulator ----
    # NB: the simulator only works when the model is built in-memory via
    # load_onnx + build (load_rknn -> init_runtime requires a real device).
    # So we re-run the quantisation here using the same calibration manifest
    # and YAML settings - which yields an equivalent INT8 graph to the saved
    # .rknn but is queryable through the simulator.
    rknn = RKNN(verbose=False)
    assert rknn.config(
        target_platform=args.target,
        quantized_dtype="w8a8",
        quantized_algorithm=args.quantized_algorithm,
        quantized_method=args.quantized_method,
        optimization_level=3,
        float_dtype="float16",
        quant_img_RGB2BGR=False,
    ) == 0
    assert rknn.load_onnx(
        model=args.onnx,
        inputs=["keypoint", "imgs", "handedness", "global_traj"],
        input_size_list=[[1, 16, 21, 4], [1, 16, 3, 96, 96], [1], [1, 16, 5]],
        outputs=["cls_score", "gate_logit"],
    ) == 0
    print(f"[rknn] building (do_quantization=True, auto_hybrid={args.auto_hybrid}, "
          f"dataset={args.dataset}) ...")
    assert rknn.build(
        do_quantization=True,
        dataset=args.dataset,
        auto_hybrid=args.auto_hybrid,
    ) == 0
    assert rknn.init_runtime() == 0
    print(f"[rknn] simulator ready")

    # ---- iterate ----
    n_top1_match = 0
    n_top3_overlap = 0
    cls_max_abs_diff = []
    gate_abs_diff = []
    per_sample = []

    for row_i, row in enumerate(rows):
        kp_p, imgs_p, hand_p, traj_p = row
        kp = np.load(kp_p).astype(np.float32)
        imgs = np.load(imgs_p).astype(np.float32)
        hand = np.load(hand_p).astype(np.int64)
        traj = np.load(traj_p).astype(np.float32)

        # ONNX
        onnx_feed = {
            "keypoint": kp, "imgs": imgs, "handedness": hand, "global_traj": traj,
        }
        onnx_outs = sess.run(onnx_output_names, onnx_feed)
        cls_onnx = onnx_outs[0].reshape(-1)
        gate_onnx = onnx_outs[1].reshape(-1)[0] if "gate_logit" in onnx_output_names else None

        # RKNN simulator (note: RKNN expects a list of inputs in model.inputs order).
        # data_format='nchw' applied to all inputs prevents RKNN from
        # mis-interpreting the 4-D `keypoint` tensor as an NHWC image and
        # silently transposing the V/C axes.
        rknn_outs = rknn.inference(
            inputs=[kp, imgs, hand, traj],
            data_format=["nchw", "nchw", "nchw", "nchw"],
        )
        cls_rknn = np.asarray(rknn_outs[0]).reshape(-1)
        gate_rknn = np.asarray(rknn_outs[1]).reshape(-1)[0] if len(rknn_outs) > 1 else None

        # metrics
        p_onnx = softmax(cls_onnx)
        p_rknn = softmax(cls_rknn)
        top_onnx = topk(p_onnx, 3)
        top_rknn = topk(p_rknn, 3)
        match1 = int(top_onnx[0][0] == top_rknn[0][0])
        n_top1_match += match1
        overlap3 = len(set(t[0] for t in top_onnx) & set(t[0] for t in top_rknn))
        n_top3_overlap += int(overlap3 >= 2)
        cls_max_abs_diff.append(float(np.abs(p_onnx - p_rknn).max()))
        if gate_onnx is not None and gate_rknn is not None:
            sigm = lambda x: 1.0 / (1.0 + np.exp(-x))
            gate_abs_diff.append(abs(float(sigm(gate_onnx) - sigm(gate_rknn))))

        gt_label = None
        idx_in_name = int(Path(kp_p).stem.split("__")[-1]) if "__" in Path(kp_p).stem else None
        if idx_in_name is not None and idx_in_name in gt_by_idx:
            gt_label = gt_by_idx[idx_in_name]

        per_sample.append({
            "row": row_i,
            "file": Path(kp_p).stem,
            "gt_label": gt_label,
            "gt_name": LABELS[gt_label] if gt_label is not None else None,
            "onnx_top1": top_onnx[0][0], "onnx_top1_name": top_onnx[0][1], "onnx_top1_prob": top_onnx[0][2],
            "rknn_top1": top_rknn[0][0], "rknn_top1_name": top_rknn[0][1], "rknn_top1_prob": top_rknn[0][2],
            "top1_match": bool(match1),
            "max_prob_diff": cls_max_abs_diff[-1],
            "gate_onnx": float(gate_onnx) if gate_onnx is not None else None,
            "gate_rknn": float(gate_rknn) if gate_rknn is not None else None,
        })

        if row_i < 20 or row_i % 10 == 0:
            print(f"[{row_i:3d}] gt={LABELS[gt_label] if gt_label is not None else '?':>10s} "
                  f"onnx={top_onnx[0][1]:>10s}({top_onnx[0][2]:.3f}) "
                  f"rknn={top_rknn[0][1]:>10s}({top_rknn[0][2]:.3f}) "
                  f"match1={match1} diff={cls_max_abs_diff[-1]:.4f}")

    rknn.release()

    # summary
    n = len(rows)
    n_onnx_match_gt = sum(1 for r in per_sample if r["gt_label"] is not None and r["onnx_top1"] == r["gt_label"])
    n_rknn_match_gt = sum(1 for r in per_sample if r["gt_label"] is not None and r["rknn_top1"] == r["gt_label"])
    n_with_gt = sum(1 for r in per_sample if r["gt_label"] is not None)

    print()
    print("=" * 60)
    print(f"Total samples           : {n}")
    print(f"ONNX vs RKNN top1 match : {n_top1_match}/{n}  ({n_top1_match/n*100:.1f}%)")
    print(f"top3 overlap (>=2)      : {n_top3_overlap}/{n}  ({n_top3_overlap/n*100:.1f}%)")
    if cls_max_abs_diff:
        print(f"max |Δprob| / sample    : avg={np.mean(cls_max_abs_diff):.4f} max={np.max(cls_max_abs_diff):.4f}")
    if gate_abs_diff:
        print(f"max |Δgate_sigmoid|     : avg={np.mean(gate_abs_diff):.4f} max={np.max(gate_abs_diff):.4f}")
    if n_with_gt:
        print(f"ONNX top1 vs GT label   : {n_onnx_match_gt}/{n_with_gt}  ({n_onnx_match_gt/n_with_gt*100:.1f}%)")
        print(f"RKNN top1 vs GT label   : {n_rknn_match_gt}/{n_with_gt}  ({n_rknn_match_gt/n_with_gt*100:.1f}%)")

    if args.output:
        Path(args.output).write_text(json.dumps({
            "summary": {
                "n_total": n,
                "n_top1_match": n_top1_match,
                "n_top3_overlap": n_top3_overlap,
                "n_with_gt": n_with_gt,
                "n_onnx_match_gt": n_onnx_match_gt,
                "n_rknn_match_gt": n_rknn_match_gt,
            },
            "per_sample": per_sample,
        }, indent=2, ensure_ascii=False))
        print(f"per-sample report: {args.output}")


if __name__ == "__main__":
    main()
