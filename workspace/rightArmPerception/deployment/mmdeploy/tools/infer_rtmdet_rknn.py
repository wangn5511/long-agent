import argparse
import time
from dataclasses import dataclass
from typing import Tuple, List, Optional, Any, Dict

import cv2
import numpy as np

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
    """Match mmdet test_pipeline:
      Resize(scale=(W,H), keep_ratio=True)
      Pad(size=(W,H), pad_val=114)
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
    """Match DetDataPreprocessor with bgr_to_rgb=False:
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


def normalize_bgr_to_nhwc(
    img_bgr: np.ndarray,
    mean=(103.53, 116.28, 123.675),
    std=(57.375, 57.12, 58.395),
) -> np.ndarray:
    """Same normalization as `normalize_bgr_to_nchw`, but output NHWC (1HWC).

    This avoids an extra transpose on RKNNLite where runtime prefers NHWC.
    """
    x = img_bgr.astype(np.float32)
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    x = (x - mean) / std
    x = x[None, ...]  # 1HWC
    return x


def unletterbox_boxes_to_original(
    boxes_xyxy: np.ndarray,
    meta: LetterboxMeta
) -> np.ndarray:
    """Map boxes from padded input coords back to original coords."""
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
    """Plain NMS on xyxy boxes (single-class)."""
    if boxes.size == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
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
# RKNN runtime wrapper + output parsing + detector class
# =========================

def _load_rknn_runtime():
    """Prefer RKNNLite on device; fall back to RKNN on PC."""
    try:
        from rknnlite.api import RKNNLite  # type: ignore
        return 'lite', RKNNLite
    except Exception:
        from rknn.api import RKNN  # type: ignore
        return 'api', RKNN


def parse_rknn_outputs(outputs: List[np.ndarray]):
    """Parse outputs for a 'no NMS' export.

    Expected:
      - boxes:  (1, N, 4) or (N, 4)
      - scores: (1, N, 1) or (N, 1) or (1, N, C) or (N, C) or (N,)

    NOTE:
      On RKNNLite, some models may return extra singleton dims, e.g.
        boxes:  (N, 1, 4)
        scores: (N, 1, 1) or (N, 1)
    """
    if len(outputs) < 2:
        raise ValueError(f'Expected 2 outputs (boxes, scores), got {len(outputs)}')

    boxes = outputs[0].astype(np.float32)
    scores = outputs[1].astype(np.float32)

    # strip batch dim if present
    if boxes.ndim >= 1 and boxes.shape[0] == 1:
        boxes = boxes[0]
    if scores.ndim >= 1 and scores.shape[0] == 1:
        scores = scores[0]

    # Normalize boxes to (N, 4)
    # e.g. RKNNLite: (N, 1, 4) or (N, 4)
    if boxes.ndim == 3 and boxes.shape[-1] == 4 and boxes.shape[1] == 1:
        boxes = boxes[:, 0, :]  # (N,4)
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f'Unexpected boxes shape: {boxes.shape}')

    # Normalize score shapes
    # Accept common layouts:
    # - (N,) / (N,1) / (N,C)
    # - (C,H,W) or (H,W) where H*W == N (RKNNLite sometimes returns BCHW without B)
    N = boxes.shape[0]

    if scores.ndim == 3:
        # (C,H,W) -> (H*W, C)
        C, H, W = scores.shape
        if H * W != N:
            raise ValueError(
                f'Unexpected scores shape: {scores.shape} (boxes={boxes.shape}), '
                f'expected H*W==N ({H}*{W}!={N})')
        scores = np.transpose(scores, (1, 2, 0)).reshape(-1, C)  # (N,C)
    elif scores.ndim == 2:
        # could be (N,1)/(N,C) or (H,W) with H*W==N
        if scores.shape[0] == N:
            # (N, C) or (N, 1)
            pass
        else:
            H, W = scores.shape
            if H * W != N:
                raise ValueError(
                    f'Unexpected scores shape: {scores.shape} (boxes={boxes.shape}), '
                    f'expected H*W==N ({H}*{W}!={N})')
            scores = scores.reshape(-1, 1)  # (N,1)
    elif scores.ndim == 1:
        if scores.shape[0] != N:
            raise ValueError(f'Unexpected scores shape: {scores.shape} (boxes={boxes.shape})')
    else:
        raise ValueError(f'Unexpected scores shape: {scores.shape} (boxes={boxes.shape})')

    # squeeze single-class (N,1) -> (N,)
    if scores.ndim == 2 and scores.shape[0] == N and scores.shape[1] == 1:
        scores = scores[:, 0]

    return boxes, scores


def _rknn_infer(rknn, inp: np.ndarray, data_format: str) -> List[np.ndarray]:
    """Call rknn.inference with best-effort compatibility across versions."""
    try:
        out = rknn.inference(inputs=[inp], data_format=[data_format])
    except TypeError:
        out = rknn.inference(inputs=[inp], data_format=data_format)
    if out is None:
        raise RuntimeError("rknn.inference returned None")
    return out


class RTMDetRKNNDetector:
    """Minimal RKNN detector for RTMDet (no embedded NMS).

    Call `infer(frame_bgr)` to get boxes/scores/labels on original image coords.
    """

    def __init__(
        self,
        rknn_path: str,
        target: str = "rk3588",
        device_id: Optional[str] = None,
        core_mask: str = "AUTO",
        data_format: str = "nhwc",
        input_size: int = 320,
        pad_val: int = 114,
        mean: Tuple[float, float, float] = (103.53, 116.28, 123.675),
        std: Tuple[float, float, float] = (57.375, 57.12, 58.395),
        score_thr0: float = 0.05,
        iou_thr: float = 0.6,
        max_per_img: int = 100,
        warmup: int = 1,
    ):
        self.target = target
        self.device_id = device_id
        self.core_mask = core_mask
        self.data_format = data_format.lower()
        self.input_size = int(input_size)
        self.pad_val = int(pad_val)
        self.mean = mean
        self.std = std
        self.score_thr0 = float(score_thr0)
        self.iou_thr = float(iou_thr)
        self.max_per_img = int(max_per_img)

        runtime_kind, RuntimeCls = _load_rknn_runtime()
        self.runtime_kind = runtime_kind
        self.rknn = RuntimeCls(verbose=False)

        # On-board RKNNLite commonly prefers NHWC input. If user didn't override,
        # switch to NHWC to avoid implicit reformat warnings/cost.
        if self.runtime_kind == 'lite' and self.data_format == 'nchw':
            self.data_format = 'nhwc'

        ret = self.rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: ret={ret}")

        init_kwargs: Dict[str, Any] = {}
        if self.runtime_kind == 'api':
            init_kwargs['target'] = self.target
            if self.device_id is not None:
                init_kwargs['device_id'] = self.device_id
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
                cm = core_map.get(self.core_mask.upper(), None)
                if cm is not None:
                    init_kwargs['core_mask'] = cm
            if self.device_id is not None:
                init_kwargs['device_id'] = self.device_id

        ret = self.rknn.init_runtime(**init_kwargs)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: ret={ret}, kwargs={init_kwargs}")

        # warmup
        for _ in range(max(0, int(warmup))):
            dummy = np.zeros(
                (1, self.input_size, self.input_size, 3),
                dtype=np.float32) if self.data_format == 'nhwc' else np.zeros(
                (1, 3, self.input_size, self.input_size),
                dtype=np.float32)
            _ = _rknn_infer(self.rknn, dummy, self.data_format)

    def release(self):
        try:
            self.rknn.release()
        except Exception:
            pass

    def preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, LetterboxMeta]:
        img_lb, meta = letterbox_keep_ratio_pad(
            frame_bgr, (self.input_size, self.input_size), pad_val=self.pad_val)
        if self.data_format == 'nhwc':
            inp = normalize_bgr_to_nhwc(img_lb, mean=self.mean, std=self.std)
        else:
            inp = normalize_bgr_to_nchw(img_lb, mean=self.mean, std=self.std)
        return inp, meta

    def infer_raw(self, inp: np.ndarray) -> List[np.ndarray]:
        return _rknn_infer(self.rknn, inp, self.data_format)

    def postprocess(
        self,
        outputs: List[np.ndarray],
        meta: LetterboxMeta,
        score_thr: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        boxes_in, scores = parse_rknn_outputs(outputs)

        labels = None
        # multi-class scores (N,C): expand candidates
        if isinstance(scores, np.ndarray) and scores.ndim == 2 and scores.shape[1] > 1:
            box_ids, cls_ids = np.where(scores >= self.score_thr0)
            boxes_in = boxes_in[box_ids]
            scores = scores[box_ids, cls_ids].astype(np.float32)
            labels = cls_ids.astype(np.int64)

        # early score filter
        score_mask = scores >= self.score_thr0
        boxes_in = boxes_in[score_mask]
        scores = scores[score_mask]
        if labels is not None:
            labels = labels[score_mask]

        boxes = unletterbox_boxes_to_original(boxes_in, meta)

        # NMS (class-wise if labels available)
        if labels is None:
            keep = nms_numpy_xyxy(boxes, scores, self.iou_thr)
        else:
            max_wh = float(max(meta.orig_hw) + 1)
            boxes_for_nms = boxes.copy()
            boxes_for_nms[:, [0, 2]] += labels.astype(np.float32) * max_wh
            keep = nms_numpy_xyxy(boxes_for_nms, scores, self.iou_thr)
        boxes = boxes[keep]
        scores = scores[keep]
        if labels is not None:
            labels = labels[keep]

        # max_per_img
        if len(scores) > self.max_per_img:
            order = scores.argsort()[::-1][:self.max_per_img]
            boxes = boxes[order]
            scores = scores[order]
            if labels is not None:
                labels = labels[order]

        # final threshold
        keep2 = scores >= float(score_thr)
        boxes = boxes[keep2]
        scores = scores[keep2]
        if labels is not None:
            labels = labels[keep2]

        return boxes, scores, labels

    def infer(self, frame_bgr: np.ndarray, score_thr: float = 0.3):
        inp, meta = self.preprocess(frame_bgr)
        outputs = self.infer_raw(inp)
        return self.postprocess(outputs, meta, score_thr=score_thr)


def _flip_frame(frame: np.ndarray, temp_flip: Optional[str]) -> np.ndarray:
    if temp_flip == "horizontal":
        return frame[:, ::-1]
    elif temp_flip == "vertical":
        return frame[::-1, :]
    elif temp_flip == "both":
        return frame[::-1, ::-1]
    return frame


def webcam_demo(
    detector: RTMDetRKNNDetector,
    cam_id: int = 0,
    score_thr: float = 0.3,
    window: str = "rtmdet_rknn",
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

        boxes, scores, labels = detector.infer(frame, score_thr=score_thr)
        infer_ms = (time.time() - t0) * 1000.0
        frame_idx += 1

        # FPS over the whole loop (end-to-end)
        elapsed = max(time.time() - t_start, 1e-9)
        fps = frame_idx / elapsed

        if print_every > 0 and (frame_idx - last_print) >= print_every:
            last_print = frame_idx
            print(f"[perf] frame={frame_idx}  infer={infer_ms:.2f} ms  fps={fps:.2f}")

        vis = draw_boxes(frame, boxes, scores, labels=labels, score_thr=score_thr)
        if writer is not None:
            writer.write(vis)
        if show:
            cv2.putText(vis, f"{infer_ms:.1f} ms  {fps:.1f} FPS", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
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
    ap.add_argument("--rknn", required=True, help="Path to .rknn model")

    # mode
    ap.add_argument("--img", default=None, help="Optional image path (image mode)")
    ap.add_argument("--out", default="vis_rknn.jpg", help="Output image path (image mode)")
    ap.add_argument("--webcam", action="store_true", help="Run webcam demo")
    ap.add_argument("--cam_id", type=int, default=0)
    ap.add_argument("--save_video", default=None, help="Optional mp4 path for webcam mode")
    ap.add_argument("--no_show", action="store_true", help="Do not open an imshow window")
    ap.add_argument("--print_every", type=int, default=30,
                    help="Print perf stats every N frames in webcam mode (0 to disable)")
    ap.add_argument("--temp_flip", type=str, default="none")

    # postprocess
    ap.add_argument("--score_thr", type=float, default=0.3)
    ap.add_argument("--iou_thr", type=float, default=0.6)
    ap.add_argument("--max_per_img", type=int, default=100)

    # preprocess
    ap.add_argument("--input_size", type=int, default=320)
    ap.add_argument("--pad_val", type=int, default=114)
    ap.add_argument("--mean", type=float, nargs=3, default=(103.53, 116.28, 123.675))
    ap.add_argument("--std", type=float, nargs=3, default=(57.375, 57.12, 58.395))

    # runtime
    ap.add_argument("--target", default="rk3588")
    ap.add_argument("--device_id", default=None)
    ap.add_argument("--core_mask", default="AUTO", help="AUTO|0|1|2|01|012")
    ap.add_argument("--data_format", default="nhwc", choices=["nchw", "nhwc"])
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    detector = RTMDetRKNNDetector(
        rknn_path=args.rknn,
        target=args.target,
        device_id=args.device_id,
        core_mask=args.core_mask,
        data_format=args.data_format,
        input_size=args.input_size,
        pad_val=args.pad_val,
        mean=tuple(args.mean),
        std=tuple(args.std),
        iou_thr=args.iou_thr,
        max_per_img=args.max_per_img,
        warmup=args.warmup,
    )

    try:
        if args.webcam:
            webcam_demo(
                detector,
                cam_id=args.cam_id,
                score_thr=args.score_thr,
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

        boxes, scores, labels = detector.infer(img, score_thr=args.score_thr)
        vis = draw_boxes(img, boxes, scores, labels=labels, score_thr=args.score_thr)
        cv2.imwrite(args.out, vis)
        print(f"Saved: {args.out}  dets={len(scores)}")
    finally:
        detector.release()


if __name__ == "__main__":
    main()


