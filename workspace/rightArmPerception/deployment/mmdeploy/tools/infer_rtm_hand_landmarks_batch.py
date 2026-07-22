from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import onnxruntime as ort


# ============================================================
# Common ORT session
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
        self.outs = self.sess.get_outputs()

    def run(self, x: np.ndarray) -> List[np.ndarray]:
        return self.sess.run(None, {self.input_name: x})

    def describe(self, tag: str = ""):
        if tag:
            print(f"== ORT IO ({tag}) ==")
        else:
            print("== ORT IO ==")
        print("Input:", self.input_name, self.inp.shape, self.inp.type)
        for o in self.outs:
            print("Output:", o.name, o.shape, o.type)


# ============================================================
# RTMDet: letterbox + normalize + parse end2end
# ============================================================

@dataclass
class LetterboxMeta:
    orig_hw: Tuple[int, int]
    input_hw: Tuple[int, int]
    scale: float
    pad_left: int
    pad_top: int


def letterbox_keep_ratio_pad(img_bgr: np.ndarray, dst_hw=(320, 320), pad_val=114):
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


def normalize_bgr_to_nchw(img_bgr: np.ndarray,
                          mean=(103.53, 116.28, 123.675),
                          std=(57.375, 57.12, 58.395)):
    x = img_bgr.astype(np.float32)
    mean = np.array(mean, np.float32).reshape(1, 1, 3)
    std = np.array(std, np.float32).reshape(1, 1, 3)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    return x[None, ...].astype(np.float32)


def unletterbox_boxes_to_original(boxes_xyxy: np.ndarray, meta: LetterboxMeta):
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
    # dets: [1,N,5] or [N,5]
    dets = outputs[0]
    if dets.ndim == 3:
        dets = dets[0]
    if dets.ndim != 2 or dets.shape[1] < 5:
        raise RuntimeError(f"Unexpected RTMDet dets shape: {outputs[0].shape}")

    boxes = dets[:, :4].astype(np.float32)
    scores = dets[:, 4].astype(np.float32)

    # labels sometimes float in mmdeploy ORT graph
    labels = None
    if len(outputs) >= 2:
        lab = outputs[1]
        if lab.ndim == 2:
            lab = lab[0]
        labels = lab.astype(np.float32)

    return boxes, scores, labels


# ============================================================
# Top-down pose: fast ROI crop+resize (replace warpAffine)
# ============================================================

@dataclass
class CropMap:
    # map from crop coord -> original coord:
    # x_orig = x0 + x * sx
    # y_orig = y0 + y * sy
    x0: float
    y0: float
    sx: float
    sy: float


def get_center_scale_from_bbox_xyxy(
    bbox_xyxy: np.ndarray,
    input_size: Tuple[int, int] = (256, 256),
    padding: float = 1.25,
) -> Tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = bbox_xyxy.astype(np.float32)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    center = np.array([x1 + 0.5 * w, y1 + 0.5 * h], dtype=np.float32)

    aspect = input_size[0] / input_size[1]
    if w > aspect * h:
        h = w / aspect
    elif w < aspect * h:
        w = h * aspect

    scale = np.array([w * padding, h * padding], dtype=np.float32)
    return center, scale


def fast_crop_resize(
    img_bgr: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    out_size: Tuple[int, int],
) -> Tuple[np.ndarray, CropMap]:
    """
    Faster alternative to warpAffine for rotation=0 case.
    We take axis-aligned rectangle:
      [cx - sw/2, cy - sh/2, cx + sw/2, cy + sh/2]
    then pad if out-of-bounds, then resize to out_size.

    Returns:
      crop_bgr: (H,W,3)
      crop_map: mapping from crop coords to original coords
    """
    out_w, out_h = out_size
    cx, cy = float(center[0]), float(center[1])
    sw, sh = float(scale[0]), float(scale[1])

    x1 = cx - sw * 0.5
    y1 = cy - sh * 0.5
    x2 = cx + sw * 0.5
    y2 = cy + sh * 0.5

    H, W = img_bgr.shape[:2]

    # integer ROI for slicing
    ix1 = int(np.floor(x1))
    iy1 = int(np.floor(y1))
    ix2 = int(np.ceil(x2))
    iy2 = int(np.ceil(y2))

    # padding amounts
    pad_left = max(0, -ix1)
    pad_top = max(0, -iy1)
    pad_right = max(0, ix2 - W)
    pad_bottom = max(0, iy2 - H)

    # clip to valid
    sx1 = max(0, ix1)
    sy1 = max(0, iy1)
    sx2 = min(W, ix2)
    sy2 = min(H, iy2)

    roi = img_bgr[sy1:sy2, sx1:sx2]
    if pad_left or pad_top or pad_right or pad_bottom:
        roi = cv2.copyMakeBorder(
            roi,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

    # Now roi corresponds to [ix1:ix2, iy1:iy2] in original float space.
    crop = cv2.resize(roi, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    # mapping crop -> original:
    src_w = (ix2 - ix1)
    src_h = (iy2 - iy1)
    sx_map = src_w / float(out_w)
    sy_map = src_h / float(out_h)
    x0_map = float(ix1)
    y0_map = float(iy1)

    return crop, CropMap(x0=x0_map, y0=y0_map, sx=sx_map, sy=sy_map)


# ============================================================
# RTMPose SimCC: preprocess + batch decode
# ============================================================

def preprocess_pose_batch_bgr2rgb_norm_nchw(
    crops_bgr: List[np.ndarray],
    mean=(123.675, 116.28, 103.53),
    std=(58.395, 57.12, 57.375),
) -> np.ndarray:
    """
    crops_bgr list -> (B,3,H,W) float32
    (config: bgr_to_rgb=True and mean/std are in RGB order)
    """
    x = np.stack(crops_bgr, axis=0).astype(np.float32)  # (B,H,W,3) BGR
    x = x[..., ::-1]  # BGR->RGB
    mean = np.array(mean, np.float32).reshape(1, 1, 1, 3)
    std = np.array(std, np.float32).reshape(1, 1, 1, 3)
    x = (x - mean) / std
    x = np.transpose(x, (0, 3, 1, 2))  # (B,3,H,W)
    return x.astype(np.float32)


def decode_simcc_batch(simcc_x: np.ndarray, simcc_y: np.ndarray, split_ratio: float = 2.0):
    """
    simcc_x/y: (B,K,L) -> kpts (B,K,2), scores (B,K)
    """
    if simcc_x.ndim == 2:
        simcc_x = simcc_x[None, ...]
    if simcc_y.ndim == 2:
        simcc_y = simcc_y[None, ...]

    ix = np.argmax(simcc_x, axis=2).astype(np.float32)
    iy = np.argmax(simcc_y, axis=2).astype(np.float32)

    sx = np.max(simcc_x, axis=2).astype(np.float32)
    sy = np.max(simcc_y, axis=2).astype(np.float32)
    scores = np.minimum(sx, sy)

    x = ix / split_ratio
    y = iy / split_ratio
    kpts = np.stack([x, y], axis=2).astype(np.float32)
    return kpts, scores


def map_kpts_crop_to_orig(kpts_xy: np.ndarray, crop_map: CropMap) -> np.ndarray:
    """
    kpts_xy: (K,2) in crop coords (0..out_w/out_h)
    """
    out = kpts_xy.astype(np.float32).copy()
    out[:, 0] = crop_map.x0 + out[:, 0] * crop_map.sx
    out[:, 1] = crop_map.y0 + out[:, 1] * crop_map.sy
    return out


# ============================================================
# Visualization helpers
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


# ============================================================
# Classes
# ============================================================

class HandDetectorORT:
    """
    RTMDet end2end ONNX (with NMS in graph).
    """
    def __init__(self,
                 onnx_path: str,
                 input_size: int = 320,
                 pad_val: int = 114,
                 score_thr: float = 0.3,
                 providers: Optional[List[str]] = None):
        self.sess = ORTSession(onnx_path, providers=providers)
        self.input_size = int(input_size)
        self.pad_val = int(pad_val)
        self.score_thr = float(score_thr)

    def detect(self, img_bgr: np.ndarray, topk: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        det_in_img, meta = letterbox_keep_ratio_pad(
            img_bgr, (self.input_size, self.input_size), pad_val=self.pad_val
        )
        det_inp = normalize_bgr_to_nchw(det_in_img)
        outs = self.sess.run(det_inp)
        boxes_in, scores, _ = parse_rtmdet_end2end(outs)

        boxes = unletterbox_boxes_to_original(boxes_in, meta)
        keep = scores >= self.score_thr
        boxes = boxes[keep]
        scores = scores[keep]

        if boxes.shape[0] == 0:
            return boxes, scores

        order = np.argsort(scores)[::-1]
        if topk and topk > 0:
            order = order[:topk]
        return boxes[order], scores[order]


class HandPoseSimCCORT:
    """
    RTMPose SimCC ONNX with dynamic batch.
    Fast crop: ROI crop+resize instead of warpAffine.
    """
    def __init__(self,
                 onnx_path: str,
                 input_size: Tuple[int, int] = (256, 256),
                 padding: float = 1.25,
                 split_ratio: float = 2.0,
                 kpt_thr: float = 0.0,
                 providers: Optional[List[str]] = None):
        self.sess = ORTSession(onnx_path, providers=providers)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.padding = float(padding)
        self.split_ratio = float(split_ratio)
        self.kpt_thr = float(kpt_thr)

        # from your config
        self.mean = (123.675, 116.28, 103.53)
        self.std = (58.395, 57.12, 57.375)

    def infer_batch(self, img_bgr: np.ndarray, bboxes_xyxy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
          bboxes_xyxy: (B,4) in original image coords
        Returns:
          kpts_orig: (B,21,2)
          kpt_scores: (B,21)
        """
        B = int(bboxes_xyxy.shape[0])
        if B == 0:
            return np.zeros((0, 21, 2), np.float32), np.zeros((0, 21), np.float32)

        crops: List[np.ndarray] = []
        maps: List[CropMap] = []

        out_w, out_h = self.input_size

        # fast crop per hand (still loop, but much cheaper than warpAffine)
        for i in range(B):
            center, scale = get_center_scale_from_bbox_xyxy(
                bboxes_xyxy[i], input_size=self.input_size, padding=self.padding
            )
            crop, cmap = fast_crop_resize(img_bgr, center, scale, (out_w, out_h))
            crops.append(crop)
            maps.append(cmap)

        inp = preprocess_pose_batch_bgr2rgb_norm_nchw(crops, mean=self.mean, std=self.std)  # (B,3,H,W)
        outs = self.sess.run(inp)
        if len(outs) < 2:
            raise RuntimeError(f"Expected 2 outputs (simcc_x, simcc_y), got {len(outs)}")
        simcc_x, simcc_y = outs[0], outs[1]

        kpts_in, kpt_scores = decode_simcc_batch(simcc_x, simcc_y, split_ratio=self.split_ratio)  # (B,K,2), (B,K)

        # map back to original coords
        kpts_orig = np.zeros_like(kpts_in, dtype=np.float32)
        for i in range(B):
            kpts_orig[i] = map_kpts_crop_to_orig(kpts_in[i], maps[i])

        return kpts_orig, kpt_scores


class HandLandmarkPipeline:
    """
    Detector -> Pose pipeline.
    """
    def __init__(self,
                 det: HandDetectorORT,
                 pose: HandPoseSimCCORT,
                 topk: int = 2,
                 draw_skeleton: bool = True):
        self.det = det
        self.pose = pose
        self.topk = int(topk)
        self.draw_skeleton = bool(draw_skeleton)

    def infer(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        boxes, scores = self.det.detect(img_bgr, topk=self.topk)
        kpts, kpt_scores = self.pose.infer_batch(img_bgr, boxes)

        return {
            "bboxes_xyxy": boxes,            # (B,4)
            "bbox_scores": scores,           # (B,)
            "kpts_xy": kpts,                 # (B,21,2)
            "kpt_scores": kpt_scores,        # (B,21)
        }

    def visualize(self, img_bgr: np.ndarray, result: Dict[str, Any],
                  kpt_thr: Optional[float] = None) -> np.ndarray:
        vis = img_bgr.copy()
        boxes = result["bboxes_xyxy"]
        scores = result["bbox_scores"]
        kpts = result["kpts_xy"]
        kpt_scores = result["kpt_scores"]

        thr = self.pose.kpt_thr if kpt_thr is None else float(kpt_thr)

        for i in range(len(boxes)):
            draw_bbox(vis, boxes[i], float(scores[i]))
            draw_hand_kpts(
                vis,
                kpts[i],
                kpt_scores=kpt_scores[i],
                kpt_thr=thr,
                draw_skeleton=self.draw_skeleton
            )
        return vis


# ============================================================
# Quick CLI demo (optional)
# ============================================================

def _cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--det_onnx", required=True)
    ap.add_argument("--pose_onnx", required=True)
    ap.add_argument("--out", default="vis_pipeline.jpg")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--det_score_thr", type=float, default=0.3)
    ap.add_argument("--padding", type=float, default=1.25)
    ap.add_argument("--kpt_thr", type=float, default=0.0)
    ap.add_argument("--no_skeleton", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(args.img)

    det = HandDetectorORT(args.det_onnx, score_thr=args.det_score_thr)
    pose = HandPoseSimCCORT(args.pose_onnx, padding=args.padding, kpt_thr=args.kpt_thr)
    pipe = HandLandmarkPipeline(det, pose, topk=args.topk, draw_skeleton=(not args.no_skeleton))

    # optional: print io once
    det.sess.describe("RTMDet")
    pose.sess.describe("RTMPose")

    res = pipe.infer(img)
    vis = pipe.visualize(img, res, kpt_thr=args.kpt_thr)
    cv2.imwrite(args.out, vis)
    print(f"Saved: {args.out} (hands={len(res['bboxes_xyxy'])})")


if __name__ == "__main__":
    _cli()
