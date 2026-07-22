import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional, Literal, Dict, Any

import cv2
import numpy as np
import onnxruntime as ort


# =========================
# Utilities: preprocess
# =========================

@dataclass
class LetterboxMeta:
    orig_hw: Tuple[int, int]     # (H, W)
    input_hw: Tuple[int, int]    # (H, W) model input
    scale: float
    pad_left: int
    pad_top: int


def letterbox_keep_ratio_pad(
    img_bgr: np.ndarray,
    dst_hw: Tuple[int, int] = (320, 320),
    pad_val: int = 114
) -> Tuple[np.ndarray, LetterboxMeta]:
    """
    Match mmdet test_pipeline:
      Resize(scale=(320,320), keep_ratio=True)
      Pad(size=(320,320), pad_val=114)
    """
    oh, ow = img_bgr.shape[:2]
    dh, dw = dst_hw

    scale = min(dw / ow, dh / oh)
    nh, nw = int(round(oh * scale)), int(round(ow * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    out = np.full((dh, dw, 3), pad_val, dtype=np.uint8)
    pad_left = (dw - nw) // 2
    pad_top = (dh - nh) // 2
    out[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized

    meta = LetterboxMeta(
        orig_hw=(oh, ow),
        input_hw=(dh, dw),
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top
    )
    return out, meta


def normalize_bgr_to_nchw(
    img_bgr: np.ndarray,
    mean=(103.53, 116.28, 123.675),
    std=(57.375, 57.12, 58.395),
) -> np.ndarray:
    """
    Match DetDataPreprocessor with bgr_to_rgb=False:
    - input is BGR
    - (img - mean) / std
    - NCHW float32, add batch dim
    """
    x = img_bgr.astype(np.float32)
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))  # CHW
    x = x[None, ...]                # 1CHW
    return x


def unletterbox_boxes_to_original(
    boxes_xyxy: np.ndarray,
    meta: LetterboxMeta
) -> np.ndarray:
    """
    Map boxes from 320x320 padded input coords back to original coords.
    """
    boxes = boxes_xyxy.astype(np.float32).copy()
    boxes[:, [0, 2]] -= meta.pad_left
    boxes[:, [1, 3]] -= meta.pad_top
    boxes[:, :4] /= max(meta.scale, 1e-12)

    oh, ow = meta.orig_hw
    boxes[:, 0] = np.clip(boxes[:, 0], 0, ow - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, oh - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, ow - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, oh - 1)
    return boxes


# =========================
# Utilities: postprocess
# =========================

def nms_numpy_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    if boxes.size == 0:
        return []

    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)

        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep


def draw_boxes(
    img_bgr: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: Optional[np.ndarray] = None,
    score_thr: float = 0.05
) -> np.ndarray:
    vis = img_bgr.copy()
    for i in range(len(boxes)):
        if scores[i] < score_thr:
            continue
        x1, y1, x2, y2 = boxes[i].astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if labels is None:
            text = f"{scores[i]:.2f}"
        else:
            text = f"{int(labels[i])}:{scores[i]:.2f}"
        cv2.putText(vis, text, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


# =========================
# ORT wrapper + output parsing
# =========================

class ORTSession:
    def __init__(self, onnx_path: str, providers=None):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.outs = self.sess.get_outputs()

    def run(self, inp: np.ndarray) -> List[np.ndarray]:
        return self.sess.run(None, {self.in_name: inp})

    def describe_io(self):
        print("== ORT IO ==")
        print("Input:", self.in_name, self.sess.get_inputs()[0].shape, self.sess.get_inputs()[0].type)
        for o in self.outs:
            print("Output:", o.name, o.shape, o.type)


def _squeeze_batch(x: np.ndarray) -> np.ndarray:
    if x.ndim >= 1 and x.shape[0] == 1:
        return x[0]
    return x


def try_parse_outputs(outputs: List[np.ndarray]) -> Optional[Dict[str, Any]]:
    """
    Support two common mmdeploy exports:

    1) end2end (embedded NMS):
      - dets:  (1, N, 5) or (N, 5)   [x1,y1,x2,y2,score]
      - labels:(1, N) or (N,)        int64

    2) pre-nms (no NMS embedded):
      - boxes:  (1, N, 4) or (N, 4)  [x1,y1,x2,y2] in input coords
      - scores: (1, N, C) or (N, C)  float scores (sigmoid/softmax)
        NOTE: some configs keep output names as dets/labels, but semantics are
        boxes/scores when NMS is disabled/removed.
    """
    if len(outputs) < 1:
        return None

    out0 = outputs[0]
    out1 = outputs[1] if len(outputs) >= 2 else None

    # ---- case 1) end2end dets/labels
    dets = out0
    if dets.ndim in (2, 3) and dets.shape[-1] == 5:
        dets = _squeeze_batch(dets)
        boxes = dets[:, :4].astype(np.float32)
        scores = dets[:, 4].astype(np.float32)
        labels = None
        if out1 is not None:
            lab = _squeeze_batch(out1)
            # typical is (N,) or (N,1)
            if lab.ndim == 2 and lab.shape[-1] == 1:
                lab = lab[:, 0]
            labels = lab.astype(np.int64)
        return dict(kind='end2end', boxes=boxes, scores=scores, labels=labels)

    # ---- case 2) pre-nms boxes/scores (no embedded NMS)
    if out1 is None:
        return None

    boxes = out0
    scores = out1
    if boxes.ndim in (2, 3) and boxes.shape[-1] == 4:
        boxes = _squeeze_batch(boxes).astype(np.float32)
        scores = _squeeze_batch(scores).astype(np.float32)
        # normalize scores shape to (N, C) or (N,)
        if scores.ndim == 2 and scores.shape[0] == boxes.shape[0]:
            # (N, C)
            pass
        elif scores.ndim == 1 and scores.shape[0] == boxes.shape[0]:
            # (N,)
            pass
        elif scores.ndim == 2 and scores.shape[-1] == 1 and scores.shape[0] == boxes.shape[0]:
            scores = scores[:, 0]
        else:
            # unexpected
            return None
        return dict(kind='pre_nms', boxes=boxes, scores=scores, labels=None)

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--out", default="vis.jpg")
    ap.add_argument("--manual_nms", action="store_true")

    # 对齐你的 test_cfg
    ap.add_argument("--score_thr", type=float, default=0.3)
    ap.add_argument("--iou_thr", type=float, default=0.6)
    ap.add_argument("--max_per_img", type=int, default=100)

    # 对齐你的 pipeline / preprocessor
    ap.add_argument("--input_size", type=int, default=320)
    ap.add_argument("--pad_val", type=int, default=114)
    args = ap.parse_args()

    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(args.img)

    # preprocess (exactly matching your config)
    img_lb, meta = letterbox_keep_ratio_pad(img, (args.input_size, args.input_size), pad_val=args.pad_val)
    inp = normalize_bgr_to_nchw(img_lb,
                               mean=(103.53, 116.28, 123.675),
                               std=(57.375, 57.12, 58.395))

    sess = ORTSession(args.onnx)
    sess.describe_io()

    outputs = sess.run(inp)

    parsed = try_parse_outputs(outputs)
    if parsed is None:
        print("\n[!] This ONNX does NOT look like end2end (dets/labels).")
        print("    Outputs were:")
        for i, o in enumerate(outputs):
            print(f"    out[{i}] shape={o.shape} dtype={o.dtype}")
        print("\nNext step:")
        print("  - If you want: I can implement raw-head decode for RTMDetSepBNHead "
              "(MlvlPointGenerator strides [8,16,32] + DistancePointBBoxCoder) once you paste output names/shapes.")
        return

    boxes_in = parsed["boxes"]
    scores = parsed["scores"]
    labels = parsed.get("labels", None)
    embedded_nms = (parsed["kind"] == "end2end")

    """
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,  # for YOLOv3
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1,
    )
    """

    # If the model does NOT embed NMS, default to manual NMS unless user wants raw.
    do_manual_nms = (args.manual_nms or (not embedded_nms))

    # If scores are per-class (N, C), expand to per-detection candidates.
    # Special-case C==1: treat it as a plain score vector (single-class).
    if not embedded_nms and isinstance(scores, np.ndarray) and scores.ndim == 2:
        N, C = scores.shape
        if C == 1:
            scores = scores[:, 0]
            labels = None
        else:
            # filter low scores early (align with post_processing.score_threshold=0.05)
            score_thr0 = 0.05
            box_ids, cls_ids = np.where(scores >= score_thr0)
            if box_ids.size == 0:
                print("\nNo candidates above score_thr0=0.05")
                cv2.imwrite(args.out, img)
                return
            cand_boxes = boxes_in[box_ids]
            cand_scores = scores[box_ids, cls_ids]
            cand_labels = cls_ids.astype(np.int64)
            boxes_in, scores, labels = cand_boxes, cand_scores.astype(
                np.float32), cand_labels

    if do_manual_nms and not embedded_nms:
        # ============= (manual NMS)
        # filter by score_thr and nms_pre (your test_cfg.nms_pre=1000)
        score_mask = scores >= 0.05
        boxes_in = boxes_in[score_mask]
        scores = scores[score_mask]
        if labels is not None:
            labels = labels[score_mask]

        # map back to original image
        boxes = unletterbox_boxes_to_original(boxes_in, meta)

        # NMS (class-wise if labels available)
        if labels is None:
            keep = nms_numpy_xyxy(boxes, scores, args.iou_thr)
        else:
            # offset trick to avoid suppressing between classes
            max_wh = float(max(meta.orig_hw) + 1)
            boxes_for_nms = boxes.copy()
            boxes_for_nms[:, [0, 2]] += labels.astype(np.float32) * max_wh
            keep = nms_numpy_xyxy(boxes_for_nms, scores, args.iou_thr)
        boxes = boxes[keep]
        scores = scores[keep]
        if labels is not None:
            labels = labels[keep]

        # max_per_img=100
        if len(scores) > args.max_per_img:
            order = scores.argsort()[::-1][:args.max_per_img]
            boxes, scores = boxes[order], scores[order]
            if labels is not None:
                labels = labels[order]
    else:
        # ============= (onnx embed NMS) OR (no-NMS but user requested raw)
        # map boxes back to original image coords
        boxes = unletterbox_boxes_to_original(boxes_in, meta)

    # score filter only (since ONNX already includes NMS)
    keep = scores >= args.score_thr
    boxes = boxes[keep]
    scores = scores[keep]
    if labels is not None:
        labels = labels[keep]

    vis = draw_boxes(img, boxes, scores, labels=labels, score_thr=args.score_thr)
    cv2.imwrite(args.out, vis)
    print(f"\nSaved: {args.out}")
    print(f"Detections: {len(scores)}")


if __name__ == "__main__":
    main()
