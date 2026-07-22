import argparse
import time
from dataclasses import dataclass
from typing import Tuple, Optional, List

import cv2
import numpy as np

# Reuse RTMDet RKNN detector implementation from sibling script.
import infer_rtmdet_rknn as det_rknn


# =========================
# Pose utilities (Top-down affine + SimCC decode)
# =========================

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


def get_affine_transform(center: np.ndarray, scale: np.ndarray,
                         output_size: Tuple[int, int]) -> np.ndarray:
    out_w, out_h = output_size
    src_w = scale[0]
    src_h = scale[1]

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


def warp_affine(img_bgr: np.ndarray, M: np.ndarray,
                output_size: Tuple[int, int]) -> np.ndarray:
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
    return cv2.invertAffineTransform(M)


def transform_points(points_xy: np.ndarray, M: np.ndarray) -> np.ndarray:
    pts = points_xy.astype(np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)  # Nx3
    out = pts_h @ M.T  # Nx2
    return out


def preprocess_pose_bgr_to_rgb_norm(
    img_bgr: np.ndarray,
    mean=(123.675, 116.28, 103.53),
    std=(58.395, 57.12, 57.375),
    data_format: str = 'nhwc',
) -> np.ndarray:
    """PoseDataPreprocessor style:
    - BGR->RGB
    - (x-mean)/std using RGB mean/std
    - return float32 tensor with batch dim
    """
    x = img_bgr.astype(np.float32)
    x = x[..., ::-1]  # BGR->RGB
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    x = (x - mean) / std
    if data_format == 'nchw':
        x = np.transpose(x, (2, 0, 1))  # CHW
        x = x[None, ...]  # 1CHW
    else:
        x = x[None, ...]  # 1HWC
    return x


def _normalize_simcc_to_k_l(simcc: np.ndarray, expected_k: int = 21) -> np.ndarray:
    """Normalize SimCC output tensor to shape (K, L).

    RKNN/RKNNLite may return SimCC tensors with different singleton dims or axis
    orders (e.g. (1,1,K,L), (1,L,K), (L,K), (K,L,1), ...).
    We squeeze singleton dims then, if needed, transpose to make K first.
    """
    x = np.asarray(simcc)
    x = np.squeeze(x)  # remove ALL singleton dims

    if x.ndim != 2:
        # last-resort: try to flatten extra dims into (K, L) if one dim matches expected_k
        # e.g. (K, A, B) -> (K, A*B) or (A, B, K) -> (K, A*B)
        if x.ndim == 3:
            if x.shape[0] == expected_k:
                return x.reshape(expected_k, -1)
            if x.shape[-1] == expected_k:
                return np.transpose(x, (2, 0, 1)).reshape(expected_k, -1)
        raise ValueError(f"SimCC unexpected ndim={x.ndim}, shape={x.shape}")

    # Prefer matching expected K=21 (hand)
    if x.shape[0] == expected_k:
        return x
    if x.shape[1] == expected_k:
        return x.T

    # Fallback: no clear K axis. Keep as-is (will decode K=x.shape[0]).
    return x


def decode_simcc(simcc_x: np.ndarray,
                simcc_y: np.ndarray,
                split_ratio: float = 2.0,
                expected_k: int = 21) -> Tuple[np.ndarray, np.ndarray]:
    # Normalize to (K, L)
    simcc_x = _normalize_simcc_to_k_l(simcc_x, expected_k=expected_k)
    simcc_y = _normalize_simcc_to_k_l(simcc_y, expected_k=expected_k)

    ix = np.argmax(simcc_x, axis=1).astype(np.float32)
    iy = np.argmax(simcc_y, axis=1).astype(np.float32)
    sx = np.max(simcc_x, axis=1).astype(np.float32)
    sy = np.max(simcc_y, axis=1).astype(np.float32)
    scores = np.minimum(sx, sy)

    x = ix / split_ratio
    y = iy / split_ratio
    kpts = np.stack([x, y], axis=1).astype(np.float32)
    return kpts, scores


HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]


def draw_hand_kpts(img_bgr: np.ndarray,
                  kpts_xy: np.ndarray,
                  kpt_scores: Optional[np.ndarray] = None,
                  kpt_thr: float = 0.0,
                  draw_skeleton: bool = True) -> np.ndarray:
    vis = img_bgr.copy()

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

    for i, p in enumerate(kpts_xy):
        if kpt_scores is not None and kpt_scores[i] < kpt_thr:
            continue
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.circle(vis, (x, y), 3, (0, 0, 255), -1, cv2.LINE_AA)
    return vis


# =========================
# RKNN runtime helper for pose
# =========================

def _rknn_infer(rknn, inp: np.ndarray, data_format: str):
    try:
        out = rknn.inference(inputs=[inp], data_format=[data_format])
    except TypeError:
        out = rknn.inference(inputs=[inp], data_format=data_format)
    if out is None:
        raise RuntimeError("rknn.inference returned None")
    return out


class RTMPoseRKNN:
    """RTMPose SimCC RKNN runner (single person crop)."""

    def __init__(
        self,
        rknn_path: str,
        target: str = "rk3588",
        device_id: Optional[str] = None,
        core_mask: str = "AUTO",
        data_format: str = "nhwc",
        input_size: Tuple[int, int] = (256, 256),
        split_ratio: float = 2.0,
        warmup: int = 1,
    ):
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.split_ratio = float(split_ratio)
        self.data_format = data_format.lower()

        runtime_kind, RuntimeCls = det_rknn._load_rknn_runtime()
        self.runtime_kind = runtime_kind
        self.rknn = RuntimeCls(verbose=False)

        if self.runtime_kind == 'lite' and self.data_format == 'nchw':
            self.data_format = 'nhwc'

        ret = self.rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: ret={ret}")

        init_kwargs = {}
        if self.runtime_kind == 'api':
            init_kwargs['target'] = target
            if device_id is not None:
                init_kwargs['device_id'] = device_id
        else:
            if hasattr(self.rknn, 'NPU_CORE_0_1_2'):
                core_map = {
                    'AUTO': None,
                    '0': self.rknn.NPU_CORE_0,
                    '1': self.rknn.NPU_CORE_1,
                    '2': self.rknn.NPU_CORE_2,
                    '01': self.rknn.NPU_CORE_0_1,
                    '012': self.rknn.NPU_CORE_0_1_2,
                }
                cm = core_map.get(core_mask.upper(), None)
                if cm is not None:
                    init_kwargs['core_mask'] = cm
            if device_id is not None:
                init_kwargs['device_id'] = device_id

        ret = self.rknn.init_runtime(**init_kwargs)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: ret={ret}, kwargs={init_kwargs}")

        # warmup
        for _ in range(max(0, int(warmup))):
            if self.data_format == 'nhwc':
                dummy = np.zeros((1, self.input_size[1], self.input_size[0], 3), dtype=np.float32)
            else:
                dummy = np.zeros((1, 3, self.input_size[1], self.input_size[0]), dtype=np.float32)
            _ = _rknn_infer(self.rknn, dummy, self.data_format)

    def release(self):
        try:
            self.rknn.release()
        except Exception:
            pass

    def infer_crop(self, crop_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = preprocess_pose_bgr_to_rgb_norm(crop_bgr, data_format=self.data_format)
        outs = _rknn_infer(self.rknn, x, self.data_format)
        if len(outs) < 2:
            raise RuntimeError(f"Expected 2 outputs (simcc_x, simcc_y), got {len(outs)}")
        simcc_x, simcc_y = outs[0], outs[1]
        return simcc_x, simcc_y


@dataclass
class HandPoseResult:
    bbox_xyxy: np.ndarray         # (4,)
    det_score: float
    kpts_xy: np.ndarray           # (K,2) in original image coords
    kpt_scores: np.ndarray        # (K,)


class HandKptsPipelineRKNN:
    """RTMDet(RKNN) -> crop -> RTMPose(RKNN) pipeline."""

    def __init__(
        self,
        det_rknn_path: str,
        pose_rknn_path: str,
        det_target: str = "rk3588",
        pose_target: str = "rk3588",
        device_id: Optional[str] = None,
        det_core_mask: str = "AUTO",
        pose_core_mask: str = "AUTO",
        det_data_format: str = "nhwc",
        pose_data_format: str = "nhwc",
        det_input_size: int = 320,
        pose_input_size: Tuple[int, int] = (256, 256),
        det_score_thr: float = 0.3,
        topk_hands: int = 2,
        padding: float = 1.25,
        split_ratio: float = 2.0,
        kpt_thr: float = 0.0,
        warmup: int = 1,
    ):
        self.det_score_thr = float(det_score_thr)
        self.topk_hands = int(topk_hands)
        self.padding = float(padding)
        self.kpt_thr = float(kpt_thr)
        self.pose_input_size = (int(pose_input_size[0]), int(pose_input_size[1]))

        self.det = det_rknn.RTMDetRKNNDetector(
            rknn_path=det_rknn_path,
            target=det_target,
            device_id=device_id,
            core_mask=det_core_mask,
            data_format=det_data_format,
            input_size=det_input_size,
            warmup=warmup,
        )
        self.pose = RTMPoseRKNN(
            rknn_path=pose_rknn_path,
            target=pose_target,
            device_id=device_id,
            core_mask=pose_core_mask,
            data_format=pose_data_format,
            input_size=self.pose_input_size,
            split_ratio=split_ratio,
            warmup=warmup,
        )

    def release(self):
        self.det.release()
        self.pose.release()

    def infer(self, frame_bgr: np.ndarray) -> List[HandPoseResult]:
        boxes, scores, _ = self.det.infer(frame_bgr, score_thr=self.det_score_thr)
        if boxes.size == 0:
            return []

        order = scores.argsort()[::-1]
        if self.topk_hands > 0:
            order = order[:self.topk_hands]

        results: List[HandPoseResult] = []
        for idx in order:
            bbox = boxes[int(idx)].astype(np.float32)
            det_score = float(scores[int(idx)])

            center, scale = get_center_scale_from_bbox_xyxy(
                bbox, input_size=self.pose_input_size, padding=self.padding)
            M = get_affine_transform(center, scale, self.pose_input_size)
            crop = warp_affine(frame_bgr, M, self.pose_input_size)

            simcc_x, simcc_y = self.pose.infer_crop(crop)
            kpts_in, kpt_scores = decode_simcc(
                simcc_x, simcc_y, split_ratio=self.pose.split_ratio, expected_k=21)
            Minv = affine_inverse(M)
            kpts_orig = transform_points(kpts_in, Minv)

            results.append(
                HandPoseResult(
                    bbox_xyxy=bbox,
                    det_score=det_score,
                    kpts_xy=kpts_orig,
                    kpt_scores=kpt_scores,
                ))
        return results

    def infer_with_timing(self, frame_bgr: np.ndarray):
        """Return (results, timing_ms_dict).

        Timing includes:
        - det_ms: RTMDet end-to-end (preprocess + inference + postprocess)
        - pose_ms: RTMPose total for all hands (crop + inference + decode + inverse affine)
        """
        t_det0 = time.time()
        boxes, scores, _ = self.det.infer(frame_bgr, score_thr=self.det_score_thr)
        det_ms = (time.time() - t_det0) * 1000.0

        if boxes.size == 0:
            return [], {
                "det_ms": det_ms,
                "pose_ms": 0.0,
                "hands": 0,
                "pose_ms_per_hand": 0.0,
            }

        order = scores.argsort()[::-1]
        if self.topk_hands > 0:
            order = order[:self.topk_hands]

        t_pose0 = time.time()
        results: List[HandPoseResult] = []
        for idx in order:
            bbox = boxes[int(idx)].astype(np.float32)
            det_score = float(scores[int(idx)])

            center, scale = get_center_scale_from_bbox_xyxy(
                bbox, input_size=self.pose_input_size, padding=self.padding)
            M = get_affine_transform(center, scale, self.pose_input_size)
            crop = warp_affine(frame_bgr, M, self.pose_input_size)

            simcc_x, simcc_y = self.pose.infer_crop(crop)
            kpts_in, kpt_scores = decode_simcc(
                simcc_x, simcc_y, split_ratio=self.pose.split_ratio, expected_k=21)
            Minv = affine_inverse(M)
            kpts_orig = transform_points(kpts_in, Minv)

            results.append(
                HandPoseResult(
                    bbox_xyxy=bbox,
                    det_score=det_score,
                    kpts_xy=kpts_orig,
                    kpt_scores=kpt_scores,
                ))

        pose_ms = (time.time() - t_pose0) * 1000.0
        hands = len(results)
        return results, {
            "det_ms": det_ms,
            "pose_ms": pose_ms,
            "hands": hands,
            "pose_ms_per_hand": (pose_ms / hands) if hands > 0 else 0.0,
        }

    def draw(self, frame_bgr: np.ndarray, results: List[HandPoseResult]) -> np.ndarray:
        vis = frame_bgr.copy()
        # draw det boxes
        if results:
            b = np.stack([r.bbox_xyxy for r in results], axis=0)
            s = np.array([r.det_score for r in results], dtype=np.float32)
            vis = det_rknn.draw_boxes(vis, b, s, labels=None, score_thr=0.0)
        # draw pose
        for r in results:
            vis = draw_hand_kpts(vis, r.kpts_xy, r.kpt_scores, kpt_thr=self.kpt_thr)
        return vis


def _flip_frame(frame: np.ndarray, temp_flip: Optional[str]) -> np.ndarray:
    if temp_flip == "horizontal":
        return frame[:, ::-1]
    elif temp_flip == "vertical":
        return frame[::-1, :]
    elif temp_flip == "both":
        return frame[::-1, ::-1]
    return frame


def webcam_demo(
    pipe: HandKptsPipelineRKNN,
    cam_id: int = 0,
    window: str = "hand_kpts_rknn",
    show: bool = True,
    save_video: Optional[str] = None,
    print_every: int = 30,
    temp_flip: str = "none",
):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open webcam id={cam_id}")

    writer = None
    if save_video is not None:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_video, fourcc, fps, (w, h))

    frame_idx = 0
    t_start = time.time()
    last_print = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()

        if temp_flip != "none":
            frame = _flip_frame(frame, temp_flip)

        results, tms = pipe.infer_with_timing(frame)
        infer_ms = (time.time() - t0) * 1000.0
        frame_idx += 1
        fps = frame_idx / max(time.time() - t_start, 1e-9)

        if print_every > 0 and (frame_idx - last_print) >= print_every:
            last_print = frame_idx
            print(
                f"[perf] frame={frame_idx}  total={infer_ms:.2f} ms  "
                f"det={tms['det_ms']:.2f} ms  pose={tms['pose_ms']:.2f} ms  "
                f"pose/hand={tms['pose_ms_per_hand']:.2f} ms  "
                f"fps={fps:.2f}  hands={tms['hands']}"
            )

        vis = pipe.draw(frame, results)
        cv2.putText(
            vis,
            f"tot {infer_ms:.1f}  det {tms['det_ms']:.1f}  pose {tms['pose_ms']:.1f}  fps {fps:.1f}",
            (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if writer is not None:
            writer.write(vis)
        if show:
            cv2.imshow(window, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det_rknn", required=True, help="RTMDet .rknn (no-nms) path")
    ap.add_argument("--pose_rknn", required=True, help="RTMPose SimCC .rknn path")

    # modes
    ap.add_argument("--img", default=None)
    ap.add_argument("--out", default="vis_hand_kpts.jpg")
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--cam_id", type=int, default=0)
    ap.add_argument("--save_video", default=None)
    ap.add_argument("--no_show", action="store_true")
    ap.add_argument("--temp_flip", type=str, default="none")

    # det
    ap.add_argument("--det_input_size", type=int, default=320)
    ap.add_argument("--det_score_thr", type=float, default=0.3)
    ap.add_argument("--topk_hands", type=int, default=2)

    # pose
    ap.add_argument("--pose_w", type=int, default=256)
    ap.add_argument("--pose_h", type=int, default=256)
    ap.add_argument("--padding", type=float, default=1.25)
    ap.add_argument("--split_ratio", type=float, default=2.0)
    ap.add_argument("--kpt_thr", type=float, default=0.0)

    # runtime
    ap.add_argument("--target", default="rk3588")
    ap.add_argument("--device_id", default=None)
    # Default: split NPU cores to reduce contention/jitter on RK3588.
    # - det uses core 0+1
    # - pose uses core 2
    ap.add_argument("--det_core_mask", default="01")
    ap.add_argument("--pose_core_mask", default="2")
    ap.add_argument("--det_data_format", default="nhwc", choices=["nchw", "nhwc"])
    ap.add_argument("--pose_data_format", default="nhwc", choices=["nchw", "nhwc"])
    ap.add_argument("--warmup", type=int, default=1)

    # perf
    ap.add_argument("--print_every", type=int, default=30)
    args = ap.parse_args()

    pipe = HandKptsPipelineRKNN(
        det_rknn_path=args.det_rknn,
        pose_rknn_path=args.pose_rknn,
        det_target=args.target,
        pose_target=args.target,
        device_id=args.device_id,
        det_core_mask=args.det_core_mask,
        pose_core_mask=args.pose_core_mask,
        det_data_format=args.det_data_format,
        pose_data_format=args.pose_data_format,
        det_input_size=args.det_input_size,
        pose_input_size=(args.pose_w, args.pose_h),
        det_score_thr=args.det_score_thr,
        topk_hands=args.topk_hands,
        padding=args.padding,
        split_ratio=args.split_ratio,
        kpt_thr=args.kpt_thr,
        warmup=args.warmup,
    )

    try:
        if args.webcam:
            webcam_demo(
                pipe,
                cam_id=args.cam_id,
                show=(not args.no_show),
                save_video=args.save_video,
                print_every=args.print_every,
                temp_flip=args.temp_flip,
            )
            return

        if args.img is None:
            raise ValueError("Provide --img for image mode, or use --webcam")
        img = cv2.imread(args.img)
        if img is None:
            raise FileNotFoundError(args.img)
        results = pipe.infer(img)
        vis = pipe.draw(img, results)
        cv2.imwrite(args.out, vis)
        print(f"Saved: {args.out}  hands={len(results)}")
    finally:
        pipe.release()


if __name__ == "__main__":
    main()


