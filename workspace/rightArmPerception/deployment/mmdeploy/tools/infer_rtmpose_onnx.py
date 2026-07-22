import argparse
from dataclasses import dataclass
from typing import Tuple, Optional, List

import cv2
import numpy as np
import onnxruntime as ort


# =========================
# Utilities: Top-down bbox -> center/scale
# =========================

def get_center_scale_from_bbox_xyxy(
    bbox_xyxy: np.ndarray,
    input_size: Tuple[int, int] = (256, 256),
    padding: float = 1.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximation of MMPose GetBBoxCenterScale (topdown) for a single bbox.
    - bbox_xyxy: [x1,y1,x2,y2] in original image coords
    - returns:
        center: (2,) float32
        scale: (2,) float32  (w, h) in pixels after padding and aspect fix

    Notes:
    - MMPose uses center/scale in pixel space, then TopdownAffine warps to input_size.
    - We enforce aspect ratio = input_w / input_h by expanding w or h.
    """
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


# =========================
# Utilities: TopdownAffine (cv2.warpAffine)
# =========================

def get_affine_transform(
    center: np.ndarray,
    scale: np.ndarray,
    output_size: Tuple[int, int],
) -> np.ndarray:
    """
    Build affine transform that maps a scaled bbox region (center, scale) to output_size.
    Equivalent spirit to MMPose top-down affine.
    """
    out_w, out_h = output_size
    src_w = scale[0]
    src_h = scale[1]

    # Three points in source
    src_pt1 = center
    src_pt2 = center + np.array([0, -0.5 * src_h], dtype=np.float32)   # up direction
    src_pt3 = center + np.array([0.5 * src_w, 0], dtype=np.float32)    # right direction

    # Three points in destination
    dst_center = np.array([out_w * 0.5, out_h * 0.5], dtype=np.float32)
    dst_pt1 = dst_center
    dst_pt2 = dst_center + np.array([0, -0.5 * out_h], dtype=np.float32)
    dst_pt3 = dst_center + np.array([0.5 * out_w, 0], dtype=np.float32)

    src = np.stack([src_pt1, src_pt2, src_pt3], axis=0).astype(np.float32)
    dst = np.stack([dst_pt1, dst_pt2, dst_pt3], axis=0).astype(np.float32)

    M = cv2.getAffineTransform(src, dst)  # 2x3
    return M


def warp_affine(
    img_bgr: np.ndarray,
    M: np.ndarray,
    output_size: Tuple[int, int],
) -> np.ndarray:
    out_w, out_h = output_size
    return cv2.warpAffine(
        img_bgr,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def affine_inverse(M: np.ndarray) -> np.ndarray:
    """2x3 affine inverse."""
    Minv = cv2.invertAffineTransform(M)
    return Minv


def transform_points(points_xy: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Apply 2x3 affine to Nx2 points.
    """
    pts = points_xy.astype(np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)  # Nx3
    out = pts_h @ M.T  # Nx2
    return out


# =========================
# Utilities: Preprocess (match PoseDataPreprocessor)
# =========================

def preprocess_bgr_to_rgb_norm_nchw(
    img_bgr: np.ndarray,
    mean=(123.675, 116.28, 103.53),
    std=(58.395, 57.12, 57.375),
    bgr_to_rgb=True,
) -> np.ndarray:
    """
    Match PoseDataPreprocessor:
      - if bgr_to_rgb: BGR -> RGB
      - (x - mean) / std
      - NCHW float32
    """
    x = img_bgr.astype(np.float32)
    if bgr_to_rgb:
        x = x[..., ::-1]  # BGR->RGB

    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))  # CHW
    return x[None, ...]             # 1CHW


# =========================
# Utilities: SimCC decode
# =========================

def decode_simcc(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    split_ratio: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode SimCC to keypoints in input image coordinate (e.g. 256x256).
    Expected shapes:
      simcc_x: (K, Lx) or (1, K, Lx)
      simcc_y: (K, Ly) or (1, K, Ly)
    Returns:
      keypoints_xy: (K,2) float32 in [0..W/H)
      scores: (K,) float32 confidence proxy (max logit)
    """
    if simcc_x.ndim == 3:
        simcc_x = simcc_x[0]
    if simcc_y.ndim == 3:
        simcc_y = simcc_y[0]

    # argmax indices
    ix = np.argmax(simcc_x, axis=1).astype(np.float32)
    iy = np.argmax(simcc_y, axis=1).astype(np.float32)

    # confidence proxy: max value (logit/prob)
    sx = np.max(simcc_x, axis=1).astype(np.float32)
    sy = np.max(simcc_y, axis=1).astype(np.float32)
    scores = np.minimum(sx, sy)

    x = ix / split_ratio
    y = iy / split_ratio
    kpts = np.stack([x, y], axis=1).astype(np.float32)
    return kpts, scores


# =========================
# ORT wrapper
# =========================

class ORTModel:
    def __init__(self, onnx_path: str, providers: Optional[List[str]] = None):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.inp = self.sess.get_inputs()[0]
        self.in_name = self.inp.name
        self.outs = self.sess.get_outputs()

    def describe(self):
        print("== ORT IO ==")
        print("Input:", self.in_name, self.inp.shape, self.inp.type)
        for o in self.outs:
            print("Output:", o.name, o.shape, o.type)

    def run(self, x: np.ndarray) -> List[np.ndarray]:
        return self.sess.run(None, {self.in_name: x})


# =========================
# Visualization
# =========================

HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),# ring
    (0, 17), (17, 18), (18, 19), (19, 20) # pinky
]

def draw_hand_kpts(
    img_bgr: np.ndarray,
    kpts_xy: np.ndarray,
    kpt_scores: Optional[np.ndarray] = None,
    kpt_thr: float = 0.0,
    draw_skeleton: bool = True
) -> np.ndarray:
    vis = img_bgr.copy()

    # skeleton
    if draw_skeleton and kpts_xy.shape[0] >= 21:
        for a, b in HAND_SKELETON:
            if a >= len(kpts_xy) or b >= len(kpts_xy):
                continue
            if kpt_scores is not None:
                if kpt_scores[a] < kpt_thr or kpt_scores[b] < kpt_thr:
                    continue
            pa = tuple(np.round(kpts_xy[a]).astype(int))
            pb = tuple(np.round(kpts_xy[b]).astype(int))
            cv2.line(vis, pa, pb, (0, 255, 0), 2, cv2.LINE_AA)

    # points
    for i, p in enumerate(kpts_xy):
        if kpt_scores is not None and kpt_scores[i] < kpt_thr:
            continue
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.circle(vis, (x, y), 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(vis, str(i), (x + 3, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="RTMPose SimCC ONNX path")
    ap.add_argument("--img", required=True, help="input image path")
    ap.add_argument("--out", default="vis_pose.jpg")

    # bbox: x1 y1 x2 y2 in original image coords
    ap.add_argument("--bbox", type=float, nargs=4, default=None,
                    help="hand bbox xyxy in original image coords. If not set, use full image.")

    # config-aligned defaults
    ap.add_argument("--input_w", type=int, default=256)
    ap.add_argument("--input_h", type=int, default=256)
    ap.add_argument("--padding", type=float, default=1.25)  # typical topdown padding
    ap.add_argument("--split_ratio", type=float, default=2.0)

    ap.add_argument("--kpt_thr", type=float, default=0.0)
    ap.add_argument("--no_skeleton", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(args.img)

    oh, ow = img.shape[:2]
    if args.bbox is None:
        bbox = np.array([0, 0, ow - 1, oh - 1], dtype=np.float32)
    else:
        bbox = np.array(args.bbox, dtype=np.float32)

    # 1) bbox -> center/scale
    center, scale = get_center_scale_from_bbox_xyxy(
        bbox, input_size=(args.input_w, args.input_h), padding=args.padding
    )

    # 2) topdown affine crop/warp
    M = get_affine_transform(center, scale, (args.input_w, args.input_h))
    crop = warp_affine(img, M, (args.input_w, args.input_h))

    # 3) preprocess (PoseDataPreprocessor: bgr_to_rgb=True)
    x = preprocess_bgr_to_rgb_norm_nchw(
        crop,
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        bgr_to_rgb=True
    )

    # 4) ORT inference
    model = ORTModel(args.onnx)
    model.describe()
    outs = model.run(x)

    # 5) parse outputs (expect SimCC x/y)
    # Most mmdeploy exports produce 2 outputs: simcc_x, simcc_y
    if len(outs) < 2:
        raise RuntimeError(f"Expected 2 outputs (simcc_x, simcc_y), got {len(outs)}")

    simcc_x, simcc_y = outs[0], outs[1]
    kpts_in, kpt_scores = decode_simcc(simcc_x, simcc_y, split_ratio=args.split_ratio)

    # 6) map keypoints back to original image via inverse affine
    Minv = affine_inverse(M)
    kpts_orig = transform_points(kpts_in, Minv)

    # 7) draw
    vis = draw_hand_kpts(
        img,
        kpts_orig,
        kpt_scores,
        kpt_thr=args.kpt_thr,
        draw_skeleton=(not args.no_skeleton)
    )
    cv2.imwrite(args.out, vis)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
