import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional

import cv2
import numpy as np
import onnxruntime as ort


# ============================================================
# Common: ORT wrapper
# ============================================================

class ORTSession:
    def __init__(self, onnx_path: str, providers: Optional[List[str]] = None):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.inp = self.sess.get_inputs()[0]
        self.input_name = self.inp.name
        self.outputs = self.sess.get_outputs()

    def describe_io(self, tag: str = ""):
        if tag:
            print(f"== ORT IO ({tag}) ==")
        else:
            print("== ORT IO ==")
        print("Input:", self.input_name, self.inp.shape, self.inp.type)
        for o in self.outputs:
            print("Output:", o.name, o.shape, o.type)

    def run(self, x: np.ndarray) -> List[np.ndarray]:
        return self.sess.run(None, {self.input_name: x})


# ============================================================
# Part A: RTMDet preprocess + parse (end2end with NMS)
# ============================================================

@dataclass
class LetterboxMeta:
    orig_hw: Tuple[int, int]
    input_hw: Tuple[int, int]
    scale: float
    pad_left: int
    pad_top: int


def letterbox_keep_ratio_pad(
    img_bgr: np.ndarray,
    dst_hw: Tuple[int, int] = (320, 320),
    pad_val: int = 114
) -> Tuple[np.ndarray, LetterboxMeta]:
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
    x = img_bgr.astype(np.float32)
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))  # CHW
    return x[None, ...]            # 1CHW


def unletterbox_boxes_to_original(
    boxes_xyxy: np.ndarray,
    meta: LetterboxMeta
) -> np.ndarray:
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


def parse_rtmdet_end2end(outputs: List[np.ndarray]):
    """
    Expect:
      dets: (1,N,5) or (N,5) => [x1,y1,x2,y2,score] in model-input coords
      labels: optional
    """
    dets = outputs[0]
    if dets.ndim == 3:
        dets = dets[0]
    if dets.ndim != 2 or dets.shape[1] < 5:
        raise RuntimeError(f"Unexpected RTMDet dets shape: {outputs[0].shape}")

    boxes = dets[:, :4].astype(np.float32)
    scores = dets[:, 4].astype(np.float32)

    labels = None
    if len(outputs) >= 2:
        lab = outputs[1]
        if lab.ndim == 2:
            lab = lab[0]
        labels = lab.astype(np.int64)

    return boxes, scores, labels


# ============================================================
# Part B: RTMPose SimCC preprocess + decode
# ============================================================

def get_center_scale_from_bbox_xyxy(
    bbox_xyxy: np.ndarray,
    input_size: Tuple[int, int] = (256, 256),
    padding: float = 1.25,
) -> Tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = bbox_xyxy.astype(np.float32)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    center = np.array([x1 + w * 0.5, y1 + h * 0.5], dtype=np.float32)

    aspect = input_size[0] / input_size[1]  # w/h
    if w > aspect * h:
        h = w / aspect
    elif w < aspect * h:
        w = h * aspect

    scale = np.array([w * padding, h * padding], dtype=np.float32)
    return center, scale


def get_affine_transform(center: np.ndarray, scale: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    out_w, out_h = output_size
    src_w, src_h = scale[0], scale[1]

    src_pt1 = center
    src_pt2 = center + np.array([0, -0.5 * src_h], dtype=np.float32)
    src_pt3 = center + np.array([0.5 * src_w, 0], dtype=np.float32)

    dst_center = np.array([out_w * 0.5, out_h * 0.5], dtype=np.float32)
    dst_pt1 = dst_center
    dst_pt2 = dst_center + np.array([0, -0.5 * out_h], dtype=np.float32)
    dst_pt3 = dst_center + np.array([0.5 * out_w, 0], dtype=np.float32)

    src = np.stack([src_pt1, src_pt2, src_pt3], axis=0).astype(np.float32)
    dst = np.stack([dst_pt1, dst_pt2, dst_pt3], axis=0).astype(np.float32)
    return cv2.getAffineTransform(src, dst)


def warp_affine(img_bgr: np.ndarray, M: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    out_w, out_h = output_size
    return cv2.warpAffine(
        img_bgr, M, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def preprocess_pose_bgr2rgb_norm_nchw(
    img_bgr: np.ndarray,
    mean=(123.675, 116.28, 103.53),
    std=(58.395, 57.12, 57.375),
) -> np.ndarray:
    # config: bgr_to_rgb=True
    x = img_bgr.astype(np.float32)[..., ::-1]  # BGR->RGB
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    return x[None, ...]


def affine_inverse(M: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(M)


def transform_points(points_xy: np.ndarray, M: np.ndarray) -> np.ndarray:
    pts = points_xy.astype(np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)
    return pts_h @ M.T


def decode_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray, split_ratio: float = 2.0):
    if simcc_x.ndim == 3:
        simcc_x = simcc_x[0]
    if simcc_y.ndim == 3:
        simcc_y = simcc_y[0]

    ix = np.argmax(simcc_x, axis=1).astype(np.float32)
    iy = np.argmax(simcc_y, axis=1).astype(np.float32)

    sx = np.max(simcc_x, axis=1).astype(np.float32)
    sy = np.max(simcc_y, axis=1).astype(np.float32)
    scores = np.minimum(sx, sy)

    x = ix / split_ratio
    y = iy / split_ratio
    kpts = np.stack([x, y], axis=1).astype(np.float32)
    return kpts, scores


# ============================================================
# Visualization
# ============================================================

HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]


def draw_bbox(img: np.ndarray, bbox_xyxy: np.ndarray, score: float):
    x1, y1, x2, y2 = bbox_xyxy.astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{score:.2f}", (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


def draw_hand_kpts(img: np.ndarray, kpts_xy: np.ndarray, kpt_scores: Optional[np.ndarray] = None,
                  kpt_thr: float = 0.0, draw_skeleton: bool = True):
    if draw_skeleton and len(kpts_xy) >= 21:
        for a, b in HAND_SKELETON:
            if kpt_scores is not None and (kpt_scores[a] < kpt_thr or kpt_scores[b] < kpt_thr):
                continue
            pa = tuple(np.round(kpts_xy[a]).astype(int))
            pb = tuple(np.round(kpts_xy[b]).astype(int))
            cv2.line(img, pa, pb, (0, 255, 0), 2, cv2.LINE_AA)

    for i, p in enumerate(kpts_xy):
        if kpt_scores is not None and kpt_scores[i] < kpt_thr:
            continue
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(img, str(i), (x + 3, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


# ============================================================
# Pipeline
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--det_onnx", required=True, help="RTMDet end2end ONNX (with NMS)")
    ap.add_argument("--pose_onnx", required=True, help="RTMPose SimCC ONNX")
    ap.add_argument("--out", default="vis_pipeline.jpg")

    # detector config-aligned defaults
    ap.add_argument("--det_input", type=int, default=320)
    ap.add_argument("--det_pad_val", type=int, default=114)
    ap.add_argument("--det_score_thr", type=float, default=0.3)

    # pose config-aligned defaults
    ap.add_argument("--pose_w", type=int, default=256)
    ap.add_argument("--pose_h", type=int, default=256)
    ap.add_argument("--padding", type=float, default=1.25)
    ap.add_argument("--split_ratio", type=float, default=2.0)
    ap.add_argument("--kpt_thr", type=float, default=0.0)

    # pipeline
    ap.add_argument("--topk", type=int, default=2, help="run pose for top-k hands by det score")
    ap.add_argument("--no_skeleton", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(args.img)

    # Load models
    det_sess = ORTSession(args.det_onnx)
    pose_sess = ORTSession(args.pose_onnx)
    det_sess.describe_io("RTMDet")
    pose_sess.describe_io("RTMPose")

    # -------------------------
    # 1) Detection
    # -------------------------
    det_in_img, meta = letterbox_keep_ratio_pad(img, (args.det_input, args.det_input), pad_val=args.det_pad_val)
    det_inp = normalize_bgr_to_nchw(det_in_img,
                                   mean=(103.53, 116.28, 123.675),
                                   std=(57.375, 57.12, 58.395))

    det_outs = det_sess.run(det_inp)
    boxes_in, scores, _labels = parse_rtmdet_end2end(det_outs)

    # map back + filter
    boxes = unletterbox_boxes_to_original(boxes_in, meta)
    keep = scores >= args.det_score_thr
    boxes = boxes[keep]
    scores = scores[keep]

    if len(scores) == 0:
        cv2.imwrite(args.out, img)
        print(f"[!] No detections >= {args.det_score_thr}. Saved: {args.out}")
        return

    # sort by score desc and keep topk
    order = np.argsort(scores)[::-1]
    if args.topk > 0:
        order = order[:args.topk]
    boxes = boxes[order]
    scores = scores[order]

    vis = img.copy()

    # -------------------------
    # 2) Pose per box
    # -------------------------
    for i in range(len(boxes)):
        bbox = boxes[i]
        sc = float(scores[i])
        draw_bbox(vis, bbox, sc)

        center, scale = get_center_scale_from_bbox_xyxy(
            bbox, input_size=(args.pose_w, args.pose_h), padding=args.padding
        )
        M = get_affine_transform(center, scale, (args.pose_w, args.pose_h))
        crop = warp_affine(img, M, (args.pose_w, args.pose_h))

        pose_inp = preprocess_pose_bgr2rgb_norm_nchw(
            crop,
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        )
        pose_outs = pose_sess.run(pose_inp)

        if len(pose_outs) < 2:
            raise RuntimeError(f"Expected 2 outputs (simcc_x, simcc_y), got {len(pose_outs)}")

        simcc_x, simcc_y = pose_outs[0], pose_outs[1]
        kpts_in, kpt_scores = decode_simcc(simcc_x, simcc_y, split_ratio=args.split_ratio)

        Minv = affine_inverse(M)
        kpts_orig = transform_points(kpts_in, Minv)

        draw_hand_kpts(
            vis,
            kpts_orig,
            kpt_scores=kpt_scores,
            kpt_thr=args.kpt_thr,
            draw_skeleton=(not args.no_skeleton)
        )

    cv2.imwrite(args.out, vis)
    print(f"Saved: {args.out}  (hands={len(boxes)})")


if __name__ == "__main__":
    main()
