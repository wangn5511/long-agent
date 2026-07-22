import argparse
import time
from typing import Optional, List, Tuple

import cv2
import numpy as np


HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]


def draw_hand_kpts(
    img_bgr: np.ndarray,
    kpts_xy: np.ndarray,
    kpt_scores: Optional[np.ndarray] = None,
    kpt_thr: float = 0.0,
    draw_skeleton: bool = True
) -> np.ndarray:
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


def _flip_frame(frame: np.ndarray, temp_flip: str) -> np.ndarray:
    if temp_flip == "horizontal":
        return np.ascontiguousarray(frame[:, ::-1])
    if temp_flip == "vertical":
        return np.ascontiguousarray(frame[::-1, :])
    if temp_flip == "both":
        return np.ascontiguousarray(frame[::-1, ::-1])
    return np.ascontiguousarray(frame)


class MediaPipeHands:
    """MediaPipe Hands runner for webcam.

    Returns 21 keypoints per detected hand in pixel coords.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        try:
            import mediapipe as mp  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "mediapipe is not available in this environment. "
                "Install it on your RK3588 python env, then rerun."
            ) from e

        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def infer(self, frame_bgr: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """Return list of (kpts_xy, hand_score).

        hand_score is best-effort: MediaPipe exposes handedness/classification score,
        not per-kpt scores.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)
        out: List[Tuple[np.ndarray, float]] = []

        if not res.multi_hand_landmarks:
            return out

        for i, lm in enumerate(res.multi_hand_landmarks):
            pts = np.zeros((21, 2), dtype=np.float32)
            for j, p in enumerate(lm.landmark):
                pts[j, 0] = float(p.x) * w
                pts[j, 1] = float(p.y) * h

            score = 1.0
            try:
                if res.multi_handedness and i < len(res.multi_handedness):
                    score = float(res.multi_handedness[i].classification[0].score)
            except Exception:
                pass
            out.append((pts, score))
        return out

    def close(self):
        try:
            self._hands.close()
        except Exception:
            pass


def webcam_demo(
    cam_id: int = 0,
    max_num_hands: int = 2,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    kpt_thr: float = 0.0,
    draw_skeleton: bool = True,
    window: str = "mediapipe_hands",
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

    runner = MediaPipeHands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    frame_idx = 0
    t_start = time.time()
    last_print = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if temp_flip != "none":
                frame = _flip_frame(frame, temp_flip)

            t0 = time.time()
            hands = runner.infer(frame)
            infer_ms = (time.time() - t0) * 1000.0

            frame_idx += 1
            fps = frame_idx / max(time.time() - t_start, 1e-9)
            if print_every > 0 and (frame_idx - last_print) >= print_every:
                last_print = frame_idx
                print(f"[perf] frame={frame_idx}  infer={infer_ms:.2f} ms  fps={fps:.2f}  hands={len(hands)}")

            vis = frame
            for pts, score in hands:
                # no per-kpt score in mediapipe; use None
                _ = score  # keep for potential future use
                vis = draw_hand_kpts(vis, pts, kpt_scores=None, kpt_thr=kpt_thr, draw_skeleton=draw_skeleton)

            # ensure OpenCV-compatible memory layout (esp. after flipping)
            vis = np.ascontiguousarray(vis)
            cv2.putText(vis, f"{infer_ms:.1f} ms  {fps:.1f} FPS", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if writer is not None:
                writer.write(vis)
            if show:
                cv2.imshow(window, vis)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break
    finally:
        runner.close()
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--webcam", action="store_true", help="Run webcam demo (default)")
    ap.add_argument("--cam_id", type=int, default=0)
    ap.add_argument("--save_video", default=None)
    ap.add_argument("--no_show", action="store_true")
    ap.add_argument("--print_every", type=int, default=30)
    ap.add_argument("--temp_flip", type=str, default="none")

    ap.add_argument("--max_num_hands", type=int, default=2)
    ap.add_argument("--model_complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--min_det_conf", type=float, default=0.5)
    ap.add_argument("--min_track_conf", type=float, default=0.5)
    ap.add_argument("--kpt_thr", type=float, default=0.0)
    ap.add_argument("--no_skeleton", action="store_true")
    args = ap.parse_args()

    webcam_demo(
        cam_id=args.cam_id,
        max_num_hands=args.max_num_hands,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_det_conf,
        min_tracking_confidence=args.min_track_conf,
        kpt_thr=args.kpt_thr,
        draw_skeleton=(not args.no_skeleton),
        show=(not args.no_show),
        save_video=args.save_video,
        print_every=args.print_every,
        temp_flip=args.temp_flip,
    )


if __name__ == "__main__":
    main()


