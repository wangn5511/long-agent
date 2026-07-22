# Copyright (c) OpenMMLab. All rights reserved.
"""Local override of ``DecodeVideoROI``: derive the hand ROI bbox from the 21
per-frame keypoints instead of relying on a pre-stored ``bbox_per_frame``
annotation field.

Why this exists
---------------
The upstream ``mmaction.datasets.transforms.video_roi_decode.DecodeVideoROI``
reads ``results['bbox_per_frame']`` directly from the pkl.  The newer RTM-based
extractor (``data/gesture_data/extract_rtm_kpts_from_prepared_clips.py``)
no longer writes ``bbox`` / ``bbox_smooth`` fields into its JSON, so the pkl
converter silently fills ``bbox_per_frame = [None, ..., None]``.  The upstream
transform then falls back to ``_center_crop`` for every frame, meaning the
RGB branch sees a 720x720 center crop of the full frame instead of the hand
ROI.

This local override fixes the regression at the transform layer:

* At this stage of the pipeline (after ``SelectLocalPrimaryTrack`` and before
  ``Hand21WorldNormalize``), ``results['keypoint']`` is the full per-frame
  21-joint sequence **in pixel coordinates**.  We can therefore compute the
  hand bbox directly from the keypoints' min / max with a square padding,
  using the exact same recipe as
  ``data/gesture_data/extract_hand_roi_with_kpts.py:kpts_to_bbox``.
* If keypoints are missing for a sampled frame, we forward / backward fill
  from the nearest valid frame.
* As a last resort (no valid keypoints at all in the clip) we fall back to
  the original square center crop, matching upstream behaviour.
* For backward compatibility with old pkls, we still accept a non-None entry
  in ``bbox_per_frame`` as the highest-priority source.

Pipeline placement is identical to the upstream version: this transform
**must** sit after the frame sampler (which populates ``frame_inds``) and
before ``PoseDecode`` (which consumes / pops ``frame_inds``).
"""
import warnings
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from mmcv.transforms import BaseTransform

from mmaction.registry import TRANSFORMS

# Module-level set of video paths we have already warned about, to keep the
# per-worker log noise bounded.  We do not try to share this across processes;
# each DataLoader worker will warn at most once per missing file it sees.
_WARNED_PATHS: Set[str] = set()


def kpts_to_bbox(
    kpts_xy: np.ndarray,
    img_w: int,
    img_h: int,
    padding_ratio: float = 0.25,
) -> Optional[List[int]]:
    """Compute a square padded bounding box from a set of 2D keypoints.

    Mirrors :func:`data.gesture_data.extract_hand_roi_with_kpts.kpts_to_bbox`.

    Returns ``None`` if the keypoint set is empty or collapses to a zero-area
    box after clamping to image bounds.
    """
    kpts_xy = np.asarray(kpts_xy, dtype=np.float32)
    if kpts_xy.size == 0:
        return None
    x_min, y_min = kpts_xy.min(axis=0)
    x_max, y_max = kpts_xy.max(axis=0)
    bw, bh = float(x_max - x_min), float(y_max - y_min)
    side = max(bw, bh, 1.0)
    pad = side * float(padding_ratio)
    cx = (float(x_min) + float(x_max)) / 2.0
    cy = (float(y_min) + float(y_max)) / 2.0
    half = (side + 2.0 * pad) / 2.0
    x1 = int(max(0, cx - half))
    y1 = int(max(0, cy - half))
    x2 = int(min(int(img_w), cx + half))
    y2 = int(min(int(img_h), cy + half))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


@TRANSFORMS.register_module(force=True)
class DecodeVideoROI(BaseTransform):
    """Decode video frames at the sampled indices and crop the hand ROI.

    Bbox source priority for each sampled frame:
      1. ``results['bbox_per_frame'][frame_idx]`` if present and non-None
         (legacy pkls).
      2. Square padded bbox derived from ``results['keypoint'][0, frame_idx]``
         using :func:`kpts_to_bbox`.
      3. Forward / backward fill from the nearest valid neighbour bbox.
      4. ``_center_crop`` of the raw frame (last-resort fallback).

    Args:
        target_size (tuple[int, int]): ``(H, W)`` of the output ROI clip.
            Defaults to ``(96, 96)``.
        padding_ratio (float): Extra padding around the keypoint bbox, as
            a fraction of the larger side.  Defaults to ``0.25`` to match
            the offline MediaPipe extractor.
        score_threshold (float): Minimum per-joint score for a keypoint to
            count as valid when deriving the bbox.  Defaults to ``0.0``.
        person_index (int): Index of the person dimension to read from
            ``results['keypoint']``.  After ``SelectLocalPrimaryTrack`` the
            keypoint array is single-person, so the default ``0`` is correct.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (96, 96),
        padding_ratio: float = 0.25,
        score_threshold: float = 0.0,
        person_index: int = 0,
    ) -> None:
        self.target_size = tuple(target_size)
        self.padding_ratio = float(padding_ratio)
        self.score_threshold = float(score_threshold)
        self.person_index = int(person_index)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, results: Dict) -> Dict:
        video_path: str = results["video_path"]
        frame_inds = np.asarray(results["frame_inds"]).reshape(-1).astype(np.int64)
        if "img_shape" not in results:
            raise KeyError("Missing `img_shape` in results; cannot derive ROI bbox.")
        img_h, img_w = int(results["img_shape"][0]), int(results["img_shape"][1])

        # 1) Compute a candidate bbox per sampled frame.
        bbox_per_frame_field = results.get("bbox_per_frame")
        kp = results.get("keypoint")
        kp_score = results.get("keypoint_score")

        per_frame_bboxes: List[Optional[List[int]]] = []
        for fi in frame_inds.tolist():
            per_frame_bboxes.append(
                self._compute_bbox(
                    fi=fi,
                    img_w=img_w,
                    img_h=img_h,
                    bbox_per_frame_field=bbox_per_frame_field,
                    kp=kp,
                    kp_score=kp_score,
                )
            )

        # 2) Forward + backward fill so that contiguous invalid frames inherit
        #    the nearest valid bbox instead of falling all the way to centre crop.
        per_frame_bboxes = self._fill_bboxes(per_frame_bboxes)

        # 3) Open video once and seek to each sampled frame index.
        out_h, out_w = self.target_size

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # Fail soft: a tiny fraction of clips in the pkl can reference
            # videos that have been moved / renamed / deleted on disk.  Raising
            # here would crash a multi-GPU training run.  Instead emit black
            # frames for this clip and warn at most once per (worker, path).
            global _WARNED_PATHS
            if video_path not in _WARNED_PATHS:
                _WARNED_PATHS.add(video_path)
                warnings.warn(
                    f"DecodeVideoROI: cannot open video, emitting black frames: "
                    f"{video_path}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            blank = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            blank_chw = self._to_chw_rgb(blank)
            results["imgs"] = np.stack(
                [blank_chw for _ in range(frame_inds.size)], axis=0
            )
            return results

        imgs: List[np.ndarray] = []

        # Augmentation rotation (set by ``JointRandomRotate`` if present) is
        # applied to every decoded frame so it matches the rotated bbox /
        # rotated keypoints.  See
        # ``handoff.stgcn_hand21_3d.code.transforms.geom_aug``.
        aug_k = int(results.get("_aug_rotation_discrete_k", 0))
        aug_angle = float(results.get("_aug_rotation_angle", 0.0))

        try:
            for fi, bbox in zip(frame_inds.tolist(), per_frame_bboxes):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Couldn't decode the frame at all - emit a black ROI.
                    blank = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                    imgs.append(self._to_chw_rgb(blank))
                    continue

                if aug_k != 0:
                    frame = np.ascontiguousarray(np.rot90(frame, k=aug_k))
                elif aug_angle != 0.0:
                    fh, fw = frame.shape[:2]
                    M = cv2.getRotationMatrix2D(
                        (fw / 2.0, fh / 2.0), aug_angle, 1.0
                    )
                    frame = cv2.warpAffine(
                        frame, M, (fw, fh),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )

                if bbox is not None:
                    cropped = self._safe_crop(frame, bbox)
                else:
                    cropped = self._center_crop(frame)

                resized = cv2.resize(cropped, (out_w, out_h))
                imgs.append(self._to_chw_rgb(resized))
        finally:
            cap.release()

        results["imgs"] = np.stack(imgs, axis=0)  # (T, 3, H, W) float32 in [0, 255]
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_bbox(
        self,
        *,
        fi: int,
        img_w: int,
        img_h: int,
        bbox_per_frame_field: Optional[Sequence],
        kp: Optional[np.ndarray],
        kp_score: Optional[np.ndarray],
    ) -> Optional[List[int]]:
        # Priority 1: legacy pre-stored bbox.
        if bbox_per_frame_field is not None and 0 <= fi < len(bbox_per_frame_field):
            stored = bbox_per_frame_field[fi]
            if stored is not None:
                try:
                    x1, y1, x2, y2 = (int(stored[0]), int(stored[1]),
                                       int(stored[2]), int(stored[3]))
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(img_w, x2); y2 = min(img_h, y2)
                    if x2 > x1 and y2 > y1:
                        return [x1, y1, x2, y2]
                except (TypeError, ValueError, IndexError):
                    pass  # fall through to keypoint-based path

        # Priority 2: derive from keypoints.
        if kp is None:
            return None
        kp_arr = np.asarray(kp)
        if kp_arr.ndim != 4:
            return None
        if self.person_index >= kp_arr.shape[0]:
            return None
        if not (0 <= fi < kp_arr.shape[1]):
            return None

        kp_xy = kp_arr[self.person_index, fi, :, :2]
        if kp_score is not None:
            score_arr = np.asarray(kp_score)
            if (score_arr.ndim == 3
                    and self.person_index < score_arr.shape[0]
                    and fi < score_arr.shape[1]):
                valid = score_arr[self.person_index, fi] > self.score_threshold
            else:
                valid = np.any(np.abs(kp_xy) > 0, axis=-1)
        else:
            valid = np.any(np.abs(kp_xy) > 0, axis=-1)

        if not bool(np.any(valid)):
            return None
        return kpts_to_bbox(
            kp_xy[valid], img_w=img_w, img_h=img_h,
            padding_ratio=self.padding_ratio,
        )

    @staticmethod
    def _fill_bboxes(
        bboxes: List[Optional[List[int]]],
    ) -> List[Optional[List[int]]]:
        """Forward-then-backward fill ``None`` entries from neighbour bboxes."""
        filled = list(bboxes)
        last = None
        for i, b in enumerate(filled):
            if b is not None:
                last = b
            elif last is not None:
                filled[i] = list(last)
        last = None
        for i in range(len(filled) - 1, -1, -1):
            if filled[i] is not None:
                last = filled[i]
            elif last is not None:
                filled[i] = list(last)
        return filled

    @staticmethod
    def _safe_crop(frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0])); y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2])); y2 = min(h, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return DecodeVideoROI._center_crop(frame)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return DecodeVideoROI._center_crop(frame)
        return crop

    @staticmethod
    def _center_crop(frame: np.ndarray) -> np.ndarray:
        """Square centre crop, used as the final fallback."""
        h, w = frame.shape[:2]
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        return frame[y0:y0 + s, x0:x0 + s]

    @staticmethod
    def _to_chw_rgb(frame_bgr_hwc: np.ndarray) -> np.ndarray:
        """Convert BGR HWC uint8 to RGB CHW float32 in [0, 255]."""
        return (frame_bgr_hwc[:, :, ::-1]
                .transpose(2, 0, 1)
                .astype(np.float32))

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"target_size={self.target_size}, "
                f"padding_ratio={self.padding_ratio}, "
                f"score_threshold={self.score_threshold})")
