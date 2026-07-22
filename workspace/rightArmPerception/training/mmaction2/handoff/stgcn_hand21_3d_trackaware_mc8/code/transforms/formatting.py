# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import numpy as np
from mmcv.transforms import BaseTransform

from mmaction.registry import TRANSFORMS
from ..utils.window_track_policy import summarize_window_track_stats


@TRANSFORMS.register_module()
class Hand21WorldNormalize(BaseTransform):
    """Normalize world landmarks with wrist-relative translation and scale."""

    def __init__(self, wrist_index: int = 0, scale_index: int = 9, eps: float = 1e-6) -> None:
        self.wrist_index = wrist_index
        self.scale_index = scale_index
        self.eps = eps

    def transform(self, results: Dict) -> Dict:
        keypoint = np.asarray(results["keypoint"], dtype=np.float32).copy()
        keypoint_score = results.get("keypoint_score")
        if keypoint_score is None:
            valid_mask = np.any(np.abs(keypoint) > 0, axis=-1)
        else:
            score = np.asarray(keypoint_score)
            valid_mask = np.any(score > 0, axis=-1)

        for person_idx in range(keypoint.shape[0]):
            for frame_idx in range(keypoint.shape[1]):
                if not bool(valid_mask[person_idx, frame_idx]):
                    keypoint[person_idx, frame_idx] = 0.0
                    continue
                frame = keypoint[person_idx, frame_idx]
                wrist = frame[self.wrist_index].copy()
                frame = frame - wrist[None, :]
                scale = float(np.linalg.norm(frame[self.scale_index]))
                if scale <= self.eps:
                    scale = 1.0
                frame = frame / scale
                keypoint[person_idx, frame_idx] = frame

        results["keypoint"] = keypoint
        return results


@TRANSFORMS.register_module()
class ExtractGlobalCenterTrajectory(BaseTransform):
    """Extract sampled trajectory displacement from a precomputed source.

    Supports both 2-channel (x, y) and 3-channel (x, y, z) sources:
      - 2-channel: 2D palm-center from kpts_xy_smooth (25D pipeline).
      - 3-channel: upperbody_wrist_xyz from RTM upper-body model (3D pipeline).
        x and y are normalized by img_w / img_h; z is kept raw.

    The output trajectory is the per-frame displacement relative to the first
    valid frame.
    """

    def __init__(self, source_key: str = "global_traj_source", person_index: int = 0) -> None:
        self.source_key = source_key
        self.person_index = person_index

    def transform(self, results: Dict) -> Dict:
        if self.source_key not in results:
            raise KeyError(
                f"Missing `{self.source_key}` in sample. Rebuild the pkl with "
                "`data/hand21_3d/convert_hand21_json_to_pkl_3d.py` so it includes "
                "the trajectory source."
            )

        source = np.asarray(results[self.source_key], dtype=np.float32)
        if source.ndim != 2 or source.shape[1] not in (2, 3):
            raise ValueError(
                f"Expected `{self.source_key}` with shape (T, 2) or (T, 3), got {tuple(source.shape)}"
            )

        if "img_shape" not in results:
            raise KeyError("Missing `img_shape` in sample; cannot normalize trajectory.")
        img_h, img_w = results["img_shape"]
        if float(img_w) <= 0 or float(img_h) <= 0:
            raise ValueError(f"Invalid img_shape={results['img_shape']}; width/height must be positive.")

        frame_inds = np.asarray(results.get("frame_inds", np.arange(source.shape[0])), dtype=np.int64).reshape(-1)
        n_ch = source.shape[1]
        scale = np.array([img_w, img_h], dtype=np.float32) if n_ch == 2 else np.array([img_w, img_h, 1.0], dtype=np.float32)
        sampled_source = source[frame_inds] / scale
        traj = np.zeros_like(sampled_source, dtype=np.float32)

        keypoint_score = results.get("keypoint_score")
        if keypoint_score is None:
            valid_mask = np.any(np.abs(sampled_source) > 0, axis=-1)
        else:
            score = np.asarray(keypoint_score)
            if score.ndim != 3 or score.shape[0] <= self.person_index:
                raise ValueError(
                    f"Expected sampled keypoint_score with shape (M, T, V), got {tuple(score.shape)}"
                )
            valid_mask = np.any(score[self.person_index] > 0, axis=-1)

        valid_indices = np.flatnonzero(valid_mask)
        if valid_indices.size > 0:
            anchor = sampled_source[valid_indices[0]]
            for idx in valid_indices.tolist():
                traj[idx] = sampled_source[idx] - anchor

        results["global_traj"] = traj
        return results


@TRANSFORMS.register_module()
class FormatGlobalTrajectory(BaseTransform):
    """Format trajectory input for clip-based training.

    Converts ``(T, C)`` trajectory arrays into ``(num_clips, T/num_clips, C)``.
    """

    def transform(self, results: Dict) -> Dict:
        global_traj = np.asarray(results["global_traj"], dtype=np.float32)
        total_frames = global_traj.shape[0]
        num_clips = int(results.get("num_clips", 1))
        if total_frames % num_clips != 0:
            raise ValueError(
                f"Number of trajectory frames {total_frames} is not divisible by num_clips {num_clips}."
            )
        global_traj = global_traj.reshape(num_clips, total_frames // num_clips, global_traj.shape[-1])
        results["global_traj"] = np.ascontiguousarray(global_traj)
        return results


@TRANSFORMS.register_module()
class FormatGCNInput3D(BaseTransform):
    """Format skeleton input for single-instance 3D STGCN training."""

    def __init__(self, num_person: int = 1, mode: str = "zero", include_score: bool = False) -> None:
        if num_person != 1:
            raise ValueError("FormatGCNInput3D only supports num_person=1.")
        if mode not in ["zero", "loop"]:
            raise ValueError(f"Unsupported mode: {mode}")
        self.num_person = num_person
        self.mode = mode
        self.include_score = include_score

    def transform(self, results: Dict) -> Dict:
        keypoint = results["keypoint"]
        if self.include_score and "keypoint_score" in results:
            keypoint = np.concatenate((keypoint, results["keypoint_score"][..., None]), axis=-1)

        cur_num_person = keypoint.shape[0]
        if cur_num_person < self.num_person:
            pad_dim = self.num_person - cur_num_person
            pad = np.zeros((pad_dim,) + keypoint.shape[1:], dtype=keypoint.dtype)
            keypoint = np.concatenate((keypoint, pad), axis=0)
        elif cur_num_person > self.num_person:
            keypoint = keypoint[:self.num_person]

        _, t, _, _ = keypoint.shape
        num_clips = results.get("num_clips", 1)
        if t % num_clips != 0:
            raise ValueError(f"Number of frames {t} is not divisible by num_clips {num_clips}.")

        keypoint = keypoint.reshape((self.num_person, num_clips, t // num_clips) + keypoint.shape[2:]).transpose(1, 0, 2, 3, 4)
        results["keypoint"] = np.ascontiguousarray(keypoint[:, 0])
        return results


@TRANSFORMS.register_module()
class SelectLocalPrimaryTrack(BaseTransform):
    """Select the window-level primary track and expose it as single-track input.

    This transform expects RTM 3D samples that preserve per-track arrays in the
    pkl. When track-aware metadata is unavailable, it can fall back to the
    original single-track fields so existing behavior is preserved.
    """

    def __init__(
        self,
        *,
        allow_fallback: bool = True,
        auto_select_if_missing: bool = True,
        valid_score_threshold: float = 0.0,
    ) -> None:
        self.allow_fallback = bool(allow_fallback)
        self.auto_select_if_missing = bool(auto_select_if_missing)
        self.valid_score_threshold = float(valid_score_threshold)

    def _select_track_id_if_missing(self, results: Dict) -> Optional[int]:
        track_ids = results.get("track_ids")
        track_frame_score = results.get("track_frame_score")
        if track_ids is None or track_frame_score is None:
            return None
        total_frames = int(np.asarray(track_frame_score).shape[1]) if np.asarray(track_frame_score).ndim == 2 else 0
        frame_inds = np.asarray(results.get("frame_inds", np.arange(total_frames)), dtype=np.int64).reshape(-1)
        stats = summarize_window_track_stats(
            track_ids=track_ids,
            track_frame_score=np.asarray(track_frame_score, dtype=np.float32),
            frame_inds=frame_inds,
            valid_score_threshold=self.valid_score_threshold,
        )
        if not stats:
            return None
        results["selected_window_track_stats"] = {
            "selected_track_id": int(stats[0]["track_id"]),
            "selected_track_stats": {
                key: value
                for key, value in stats[0].items()
                if key not in {"present_mask", "valid_mask"}
            },
            "track_candidates": [
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"present_mask", "valid_mask"}
                }
                for item in stats
            ],
        }
        return int(stats[0]["track_id"])

    def transform(self, results: Dict) -> Dict:
        track_ids = results.get("track_ids")
        track_keypoint = results.get("track_keypoint")
        track_keypoint_score = results.get("track_keypoint_score")
        track_global_traj_source = results.get("track_global_traj_source")

        if track_ids is None or track_keypoint is None or track_keypoint_score is None or track_global_traj_source is None:
            if self.allow_fallback:
                return results
            raise KeyError(
                "Missing track-aware RTM fields. Rebuild the pkl with "
                "`data/hand21_3d/convert_hand21_json_to_pkl_3d.py --include-track-candidates`."
            )

        selected_track_id = results.get("selected_window_track_id")
        if selected_track_id is None and self.auto_select_if_missing:
            selected_track_id = self._select_track_id_if_missing(results)
            if selected_track_id is not None:
                results["selected_window_track_id"] = int(selected_track_id)

        if selected_track_id is None:
            if self.allow_fallback:
                return results
            raise ValueError("No selected_window_track_id was provided for SelectLocalPrimaryTrack.")

        track_ids_arr = np.asarray(track_ids, dtype=np.int64).reshape(-1)
        matches = np.where(track_ids_arr == int(selected_track_id))[0]
        if matches.size <= 0:
            if self.allow_fallback:
                return results
            raise ValueError(f"selected_window_track_id={selected_track_id} not found in track_ids.")
        track_index = int(matches[0])

        results["keypoint"] = np.asarray(track_keypoint[track_index:track_index + 1], dtype=np.float32)
        results["keypoint_score"] = np.asarray(track_keypoint_score[track_index:track_index + 1], dtype=np.float32)
        results["global_traj_source"] = np.asarray(track_global_traj_source[track_index], dtype=np.float32)
        results["track_id"] = int(selected_track_id)

        track_handedness_label_id = results.get("track_handedness_label_id")
        if track_handedness_label_id is not None:
            label_id = int(np.asarray(track_handedness_label_id, dtype=np.int64).reshape(-1)[track_index])
            results["handedness_label_id"] = label_id
            results["handedness_label"] = {0: "Left", 1: "Right", 2: "Unknown"}.get(label_id, "Unknown")

        track_handedness_score = results.get("track_handedness_score")
        if track_handedness_score is not None:
            score = float(np.asarray(track_handedness_score, dtype=np.float32).reshape(-1)[track_index])
            results["handedness_score"] = score

        return results
