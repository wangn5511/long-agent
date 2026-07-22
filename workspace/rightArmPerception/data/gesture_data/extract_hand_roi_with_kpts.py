"""
Extract hand bounding-box and keypoints from a video recording dataset.

Pipeline:
  1. Read each video frame-by-frame.
  2. Detect hands via MediaPipe HandLandmarker (VIDEO mode) and obtain 21 keypoints.
  3. Track hands across frames (IoU + handedness matching) to assign persistent IDs.
  4. Smooth keypoints with a per-track OneEuro filter to reduce jitter.
  5. Derive bounding boxes from both raw and smoothed keypoints.
  6. Save per-video annotations to JSON; optionally write a visualisation MP4.

Input:
  <src_dir>/[<label>/]<video>.mp4

Output:
  <dst_dir>/[<label>/]<video>/annotations.json
      Per-frame annotation list. Each frame entry contains:
        - frame_idx     : int, 0-based frame number
        - num_hands     : int, number of hands detected in this frame (0, 1, or 2)
        - hands         : list (present only when num_hands > 0), each element:
            - track_id      : int, persistent ID for this hand track
            - handedness    : str, "Left" or "Right"
            - score         : float, handedness classification confidence
            - bbox          : [x1, y1, x2, y2], padded square bbox (raw)
            - kpts_xy       : [[x, y], ...] x21, raw landmark pixel coords
            - bbox_smooth   : [x1, y1, x2, y2], bbox derived from smoothed keypoints
            - kpts_xy_smooth: [[x, y], ...] x21, OneEuro-filtered landmark coords

  <dst_dir>_vis/[<label>/]<video>.mp4   (--save_vis)
      Same resolution & FPS as the source video.
      Each frame overlays tracked hand(s) with:
        - green rectangle for the smoothed bounding box
        - green lines for the 21-point hand skeleton (smoothed)
        - red dots for each smoothed landmark
        - yellow text label showing track_id and handedness
      Frames without detection are written as-is.
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


# ---------------------------------------------------------------------------
# MediaPipe hand detector
# ---------------------------------------------------------------------------

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _ensure_model(model_path: str) -> str:
    """Download ``hand_landmarker.task`` if it does not exist locally."""
    p = Path(model_path)
    if p.exists():
        return model_path
    print(f"[+] Downloading hand_landmarker.task -> {p}")
    import urllib.request
    p.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, str(p))
    print(f"[+] Downloaded ({p.stat().st_size / 1024 / 1024:.1f} MB)")
    return model_path


class MediaPipeHandDetector:
    """Wrapper around ``mp.tasks.vision.HandLandmarker`` (VIDEO mode).

    Uses the new Tasks API with temporal tracking across frames.
    """

    def __init__(
        self,
        model_path: str = "hand_landmarker.task",
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        import mediapipe as mp  # type: ignore

        _ensure_model(model_path)

        self._mp = mp
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
            options
        )

    def detect(
        self, frame_bgr: np.ndarray, timestamp_ms: int
    ) -> List[Tuple[np.ndarray, float, str]]:
        """Detect hands in a video frame.

        Args:
            frame_bgr: BGR image (H, W, 3).
            timestamp_ms: Monotonically increasing timestamp in milliseconds.

        Returns a list of ``(kpts_xy, score, handedness)`` tuples.
          * kpts_xy    : (21, 2) float32, pixel coords
          * score      : handedness classification confidence
          * handedness : "Left" or "Right"
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb,
        )
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        hands: List[Tuple[np.ndarray, float, str]] = []
        if not result.hand_landmarks:
            return hands

        for i, landmarks in enumerate(result.hand_landmarks):
            pts_xy = np.zeros((21, 2), dtype=np.float32)
            for j, lm in enumerate(landmarks):
                pts_xy[j] = [lm.x * w, lm.y * h]

            score = 1.0
            handedness = "Unknown"
            try:
                cat = result.handedness[i][0]
                score = float(cat.score)
                handedness = cat.category_name
            except Exception:
                pass

            hands.append((pts_xy, score, handedness))
        return hands

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# OneEuro filter
# ---------------------------------------------------------------------------

class OneEuroFilter:
    """1D OneEuro filter for real-time signal smoothing.

    Reference: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter
    for Noisy Input in Interactive Systems", CHI 2012.

    Args:
        min_cutoff: Minimum cutoff frequency (Hz). Lower = smoother but more lag.
        beta: Speed coefficient. Higher = less smoothing when signal moves fast.
        d_cutoff: Cutoff frequency for the derivative (Hz).
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    @staticmethod
    def _smoothing_factor(t_e: float, cutoff: float) -> float:
        r = 2.0 * np.pi * cutoff * t_e
        return r / (r + 1.0)

    def __call__(self, t: float, x: float) -> float:
        if self._t_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = t
            return x

        t_e = t - self._t_prev
        if t_e <= 0:
            return self._x_prev  # type: ignore[return-value]

        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self._x_prev) / t_e  # type: ignore[operator]
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self._x_prev  # type: ignore[operator]

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat

    def reset(self):
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


class OneEuroFilter2D:
    """Convenience wrapper: applies OneEuro independently to N 2D points.

    Call with ``(t, points)`` where *points* is ``(N, 2)`` float32.
    """

    def __init__(self, n_points: int, **kwargs):
        self._filters = [
            (OneEuroFilter(**kwargs), OneEuroFilter(**kwargs))
            for _ in range(n_points)
        ]

    def __call__(self, t: float, pts: np.ndarray) -> np.ndarray:
        out = np.empty_like(pts)
        for i, (fx, fy) in enumerate(self._filters):
            out[i, 0] = fx(t, float(pts[i, 0]))
            out[i, 1] = fy(t, float(pts[i, 1]))
        return out

    def reset(self):
        for fx, fy in self._filters:
            fx.reset()
            fy.reset()


# ---------------------------------------------------------------------------
# Hand tracker
# ---------------------------------------------------------------------------

class HandTracker:
    """Simple IoU + handedness-based tracker for ≤ 2 hands.

    Assigns a persistent ``track_id`` to each detected hand and applies
    OneEuro filtering to smooth keypoints across frames.

    A track is considered *lost* after ``max_lost`` consecutive frames
    without a matching detection.
    """

    def __init__(
        self,
        max_lost: int = 10,
        iou_threshold: float = 0.3,
        one_euro_min_cutoff: float = 1.7,
        one_euro_beta: float = 0.3,
    ):
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self._oe_kwargs = dict(min_cutoff=one_euro_min_cutoff, beta=one_euro_beta)
        self._next_id = 0
        # active tracks: {track_id: {state dict}}
        self._tracks: dict = {}

    def _new_track(self, handedness: str) -> int:
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = {
            "handedness": handedness,
            "bbox": None,
            "kpts_filter": OneEuroFilter2D(21, **self._oe_kwargs),
            "lost": 0,
        }
        return tid

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(
        self,
        detections: List[Tuple[np.ndarray, float, str]],
        bboxes: List[Tuple[int, int, int, int]],
        t: float,
    ) -> List[dict]:
        """Match detections to existing tracks; return per-hand results.

        Returns a list of dicts, one per matched/new detection::

            {
                "track_id": int,
                "handedness": str,
                "score": float,
                "kpts_xy": ndarray (21,2),       # raw
                "kpts_xy_smooth": ndarray (21,2), # filtered
                "bbox": (x1,y1,x2,y2),
            }
        """
        matched_tids: set = set()
        results: List[dict] = []
        unmatched_dets: List[int] = list(range(len(detections)))

        # Greedy matching: for each detection, find best overlapping track
        # with the same handedness; fall back to pure IoU if handedness
        # doesn't match any track.
        for d_idx in list(unmatched_dets):
            kpts_xy, score, hand = detections[d_idx]
            det_bbox = bboxes[d_idx]
            best_tid = None
            best_iou = self.iou_threshold

            for tid, trk in self._tracks.items():
                if tid in matched_tids:
                    continue
                if trk["bbox"] is None:
                    continue
                iou_val = self._iou(det_bbox, trk["bbox"])
                # Bonus for matching handedness
                if trk["handedness"] == hand:
                    iou_val += 0.2
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_tid = tid

            if best_tid is not None:
                matched_tids.add(best_tid)
                unmatched_dets.remove(d_idx)
                trk = self._tracks[best_tid]
                trk["handedness"] = hand
                trk["bbox"] = det_bbox
                trk["lost"] = 0
                smooth_kpts = trk["kpts_filter"](t, kpts_xy)
                results.append({
                    "track_id": best_tid,
                    "handedness": hand,
                    "score": score,
                    "kpts_xy": kpts_xy,
                    "kpts_xy_smooth": smooth_kpts,
                    "bbox": det_bbox,
                })

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            kpts_xy, score, hand = detections[d_idx]
            det_bbox = bboxes[d_idx]
            tid = self._new_track(hand)
            trk = self._tracks[tid]
            trk["bbox"] = det_bbox
            smooth_kpts = trk["kpts_filter"](t, kpts_xy)
            results.append({
                "track_id": tid,
                "handedness": hand,
                "score": score,
                "kpts_xy": kpts_xy,
                "kpts_xy_smooth": smooth_kpts,
                "bbox": det_bbox,
            })

        # Age unmatched tracks
        lost_ids = []
        for tid in self._tracks:
            if tid not in matched_tids and tid not in {r["track_id"] for r in results}:
                self._tracks[tid]["lost"] += 1
                if self._tracks[tid]["lost"] > self.max_lost:
                    lost_ids.append(tid)
        for tid in lost_ids:
            del self._tracks[tid]

        return results


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def kpts_to_bbox(
    kpts_xy: np.ndarray,
    img_w: int,
    img_h: int,
    padding_ratio: float = 0.25,
) -> Tuple[int, int, int, int]:
    """Compute a square bounding box from 21 keypoints with padding.

    Returns ``(x1, y1, x2, y2)`` clipped to image bounds.
    """
    x_min, y_min = kpts_xy.min(axis=0)
    x_max, y_max = kpts_xy.max(axis=0)
    bw, bh = x_max - x_min, y_max - y_min
    side = max(bw, bh)
    pad = side * padding_ratio

    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    half = (side + 2 * pad) / 2.0

    x1 = int(max(0, cx - half))
    y1 = int(max(0, cy - half))
    x2 = int(min(img_w, cx + half))
    y2 = int(min(img_h, cy + half))
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_hand_on_frame(
    img: np.ndarray,
    kpts_xy: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    """Draw skeleton + bbox on ``img`` **in-place** and return it."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for a, b in HAND_SKELETON:
        pa = tuple(np.round(kpts_xy[a]).astype(int))
        pb = tuple(np.round(kpts_xy[b]).astype(int))
        cv2.line(img, pa, pb, (0, 255, 0), 2, cv2.LINE_AA)

    for pt in kpts_xy:
        cv2.circle(img, (int(round(pt[0])), int(round(pt[1]))), 3,
                   (0, 0, 255), -1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Per-video extraction
# ---------------------------------------------------------------------------

def process_single_video(
    video_path: Path,
    output_dir: Path,
    detector: MediaPipeHandDetector,
    padding_ratio: float,
    tracker: HandTracker,
    vis_path: Optional[Path] = None,
) -> dict:
    """Process one video and write annotations.json.

    All detected hands (up to ``num_hands``) are saved per frame with
    ``track_id`` for temporal association and ``kpts_xy_smooth`` for
    OneEuro-filtered keypoints.
    If *vis_path* is given, an overlay MP4 is written alongside.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [!] Cannot open video: {video_path}")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir.mkdir(parents=True, exist_ok=True)

    writer = None  # type: Optional[cv2.VideoWriter]
    if vis_path is not None:
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        for codec, suffix in [("mp4v", ".mp4"), ("XVID", ".avi")]:
            out_path = vis_path.with_suffix(suffix)
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                str(out_path), fourcc, fps, (img_w, img_h),
            )
            if writer.isOpened():
                vis_path = out_path
                break
            writer.release()
            writer = None
        if writer is None:
            print(f"  [!] Failed to create VideoWriter for {vis_path}")
        else:
            print(f"  [+] Vis video -> {vis_path}")

    from tqdm import tqdm

    frame_duration_ms = 1000.0 / fps
    annotations: List[dict] = []
    frame_idx = 0
    detected_count = 0

    pbar = tqdm(total=total_frames, desc=video_path.name, unit="frame")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_ms = int(round(frame_idx * frame_duration_ms))
            t_sec = timestamp_ms / 1000.0
            raw_hands = detector.detect(frame, timestamp_ms)

            raw_bboxes = [
                kpts_to_bbox(kpts, img_w, img_h, padding_ratio)
                for kpts, _, _ in raw_hands
            ]

            tracked = tracker.update(raw_hands, raw_bboxes, t_sec)

            entry: dict = {
                "frame_idx": frame_idx,
                "num_hands": len(tracked),
            }

            if tracked:
                hands_list = []
                for trk in tracked:
                    bbox_smooth = kpts_to_bbox(
                        trk["kpts_xy_smooth"], img_w, img_h, padding_ratio,
                    )
                    hands_list.append({
                        "track_id": trk["track_id"],
                        "handedness": trk["handedness"],
                        "score": round(trk["score"], 4),
                        "bbox": list(trk["bbox"]),
                        "kpts_xy": trk["kpts_xy"].tolist(),
                        "bbox_smooth": list(bbox_smooth),
                        "kpts_xy_smooth": trk["kpts_xy_smooth"].tolist(),
                    })
                entry["hands"] = hands_list
                detected_count += 1

            if writer is not None:
                vis = frame.copy()
                if tracked:
                    for trk in tracked:
                        bbox_smooth = kpts_to_bbox(
                            trk["kpts_xy_smooth"], img_w, img_h,
                            padding_ratio,
                        )
                        draw_hand_on_frame(
                            vis, trk["kpts_xy_smooth"], bbox_smooth,
                        )
                        x1, y1 = bbox_smooth[:2]
                        label = (f"id={trk['track_id']} "
                                 f"{trk['handedness']}")
                        cv2.putText(
                            vis, label, (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 0), 1, cv2.LINE_AA,
                        )
                writer.write(vis)

            annotations.append(entry)
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        if writer is not None:
            writer.release()

    meta = {
        "video": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "resolution": [img_w, img_h],
        "detected_frames": detected_count,
        "padding_ratio": padding_ratio,
        "frames": annotations,
    }
    with open(output_dir / "annotations.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def extract_hand_roi_from_videos(
    src_dir: str,
    dst_dir: str,
    model_path: str = "hand_landmarker.task",
    padding_ratio: float = 0.25,
    num_hands: int = 2,
    min_hand_detection_confidence: float = 0.5,
    min_hand_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    save_vis: bool = False,
):
    """Batch-extract hand bbox + keypoints from a video dataset.

    Supports two input layouts (auto-detected):

    With labels::

        <src_dir>/<label>/<video>.mp4
          -> <dst_dir>/<label>/<video>/annotations.json
          -> <dst_dir>_vis/<label>/<video>.mp4        (--save_vis)

    Flat (no label subdirs)::

        <src_dir>/<video>.mp4
          -> <dst_dir>/<video>/annotations.json
          -> <dst_dir>_vis/<video>.mp4                (--save_vis)

    Example (flat)::

        --src_dir /data/recordings
        --dst_dir /data/recordings_kpts

        /data/recordings/
            clip_001.mp4
            clip_002.mp4

        /data/recordings_kpts/            <-- annotations
            clip_001/annotations.json
            clip_002/annotations.json

        /data/recordings_kpts_vis/        <-- visualisation (--save_vis)
            clip_001.mp4
            clip_002.mp4
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    video_paths = sorted(src_dir.rglob("*.mp4"))
    if not video_paths:
        print(f"[!] No .mp4 files found under {src_dir}")
        return

    print("=" * 60)
    print(f"Hand bbox+kpts extraction  ({len(video_paths)} videos)")
    print(f"  src : {src_dir}")
    print(f"  dst : {dst_dir}")
    print(f"  padding_ratio  : {padding_ratio}")
    print(f"  save_vis       : {save_vis}")
    print("=" * 60)

    _ensure_model(model_path)
    detector_kwargs = dict(
        model_path=model_path,
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    vis_root = dst_dir.parent / f"{dst_dir.name}_vis" if save_vis else None
    t_start = time.time()

    for i, video_path in enumerate(video_paths, 1):
        if "debug_20251222_122227.mp4" in video_path.as_posix():
            continue
        rel = video_path.relative_to(src_dir)
        rel_stem = rel.with_suffix("")

        output_subdir = dst_dir / rel_stem
        vis_mp4 = (vis_root / rel) if vis_root else None

        if (output_subdir / "annotations.json").exists():
            print(f"[{i}/{len(video_paths)}] Skip (already done): {rel}")
            continue

        # VIDEO mode requires monotonically increasing timestamps,
        # so we create a fresh detector (and tracker) per video.
        detector = MediaPipeHandDetector(**detector_kwargs)
        tracker = HandTracker()

        print(f"[{i}/{len(video_paths)}] Processing: {rel}")
        t0 = time.time()

        meta = process_single_video(
            video_path=video_path,
            output_dir=output_subdir,
            detector=detector,
            padding_ratio=padding_ratio,
            tracker=tracker,
            vis_path=vis_mp4,
        )

        detector.close()

        elapsed = time.time() - t0
        if meta:
            det = meta["detected_frames"]
            tot = meta["total_frames"]
            rate = det / tot * 100 if tot else 0
            print(f"       {det}/{tot} frames detected ({rate:.1f}%), "
                  f"{elapsed:.1f}s")

    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"Done!  Total time: {total_time:.1f}s")
    print(f"  Output: {dst_dir}")
    if vis_root:
        print(f"  Visualisation: {vis_root}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract hand bounding-box and 21-keypoint annotations "
                    "from a gesture video dataset using MediaPipe HandLandmarker."
    )
    parser.add_argument(
        "--src_dir", type=str, required=True,
        help="Source video directory: <src_dir>/[<label>/]<video>.mp4")
    parser.add_argument(
        "--dst_dir", type=str, required=True,
        help="Output directory: <dst_dir>/[<label>/]<video>/annotations.json")
    parser.add_argument(
        "--model_path", type=str, default="hand_landmarker.task",
        help="Path to hand_landmarker.task (auto-downloaded if missing)")
    parser.add_argument(
        "--padding_ratio", type=float, default=0.25,
        help="Padding ratio around the keypoint bounding box")
    parser.add_argument(
        "--num_hands", type=int, default=2,
        help="Max number of hands to detect per frame")
    parser.add_argument(
        "--min_det_conf", type=float, default=0.5,
        help="Minimum hand detection confidence")
    parser.add_argument(
        "--min_presence_conf", type=float, default=0.5,
        help="Minimum hand presence confidence")
    parser.add_argument(
        "--min_track_conf", type=float, default=0.5,
        help="Minimum tracking confidence")
    parser.add_argument(
        "--save_vis", action="store_true",
        help="Write a visualisation MP4 with keypoints & bbox overlay")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_hand_roi_from_videos(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        model_path=args.model_path,
        padding_ratio=args.padding_ratio,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_det_conf,
        min_hand_presence_confidence=args.min_presence_conf,
        min_tracking_confidence=args.min_track_conf,
        save_vis=args.save_vis,
    )
