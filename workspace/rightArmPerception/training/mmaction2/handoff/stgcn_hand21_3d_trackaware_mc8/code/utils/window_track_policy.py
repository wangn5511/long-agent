from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def _longest_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in mask.astype(bool).tolist():
        if value:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return int(longest)


def _strip_masks(stat: Dict[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in stat.items()
        if key not in {"present_mask", "valid_mask"}
    }


def summarize_window_track_stats(
    track_ids: Sequence[int],
    track_frame_score: np.ndarray,
    frame_inds: Sequence[int],
    *,
    valid_score_threshold: float = 0.0,
) -> List[Dict[str, object]]:
    scores = np.asarray(track_frame_score, dtype=np.float32)
    ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
    inds = np.asarray(frame_inds, dtype=np.int64).reshape(-1)
    if scores.ndim != 2:
        raise ValueError(f"Expected track_frame_score with shape (M, T), got {tuple(scores.shape)}")
    if scores.shape[0] != ids.shape[0]:
        raise ValueError(
            "track_ids and track_frame_score disagree on track count: "
            f"{ids.shape[0]} vs {scores.shape[0]}"
        )
    if scores.shape[1] <= 0 or inds.size <= 0:
        return []

    inds = np.mod(inds, scores.shape[1])
    window_scores = scores[:, inds]
    window_len = int(max(1, inds.size))
    stats: List[Dict[str, object]] = []
    for track_index, track_id in enumerate(ids.tolist()):
        frame_scores = window_scores[track_index]
        present_mask = frame_scores > 0.0
        present_count = int(np.count_nonzero(present_mask))
        if present_count <= 0:
            continue
        valid_mask = frame_scores > float(valid_score_threshold)
        valid_count = int(np.count_nonzero(valid_mask))
        stats.append(
            {
                "track_id": int(track_id),
                "track_index": int(track_index),
                "present_count": present_count,
                "present_ratio": float(present_count / window_len),
                "valid_count": valid_count,
                "valid_ratio": float(valid_count / window_len),
                "longest_run": _longest_true_run(valid_mask),
                "score_sum": float(frame_scores.sum()),
                "present_mask": present_mask.astype(bool, copy=False),
                "valid_mask": valid_mask.astype(bool, copy=False),
            }
        )

    stats.sort(
        key=lambda item: (
            int(item["valid_count"]),
            int(item["longest_run"]),
            int(item["present_count"]),
            float(item["score_sum"]),
            -int(item["track_id"]),
        ),
        reverse=True,
    )
    return stats


def evaluate_window_track_policy(
    track_stats: Sequence[Dict[str, object]],
    *,
    window_len: int,
    primary_valid_ratio_thr: float,
    primary_min_longest_run: int,
    secondary_max_ratio_thr: float,
    secondary_total_ratio_thr: float,
    primary_margin_frames_thr: int,
) -> Dict[str, object]:
    if not track_stats:
        return {
            "selected_track_id": None,
            "selected_track_index": None,
            "selected_track_stats": None,
            "secondary_max_ratio": 0.0,
            "secondary_total_ratio": 0.0,
            "secondary_best_track_id": None,
            "secondary_best_valid_count": 0,
            "primary_margin_frames": -1,
            "passed": False,
            "reasons": ["no_track_candidates"],
            "track_candidates": [],
        }

    primary = dict(track_stats[0])
    secondary = dict(track_stats[1]) if len(track_stats) > 1 else None
    secondary_max_ratio = float(secondary["present_ratio"]) if secondary is not None else 0.0
    if len(track_stats) > 1:
        secondary_union = np.zeros((window_len,), dtype=bool)
        for stat in track_stats[1:]:
            secondary_union |= np.asarray(stat["present_mask"], dtype=bool)
        secondary_total_ratio = float(np.count_nonzero(secondary_union) / max(1, window_len))
    else:
        secondary_total_ratio = 0.0

    primary_margin_frames = int(primary["valid_count"]) - int(secondary["valid_count"]) if secondary is not None else int(primary["valid_count"])
    reasons: List[str] = []
    if float(primary["valid_ratio"]) < float(primary_valid_ratio_thr):
        reasons.append(f"primary_valid_ratio<{primary_valid_ratio_thr:.3f}")
    if int(primary["longest_run"]) < int(primary_min_longest_run):
        reasons.append(f"primary_longest_run<{int(primary_min_longest_run)}")
    if float(secondary_max_ratio) > float(secondary_max_ratio_thr):
        reasons.append(f"secondary_max_ratio>{secondary_max_ratio_thr:.3f}")
    if float(secondary_total_ratio) > float(secondary_total_ratio_thr):
        reasons.append(f"secondary_total_ratio>{secondary_total_ratio_thr:.3f}")
    if int(primary_margin_frames) < int(primary_margin_frames_thr):
        reasons.append(f"primary_margin_frames<{int(primary_margin_frames_thr)}")

    return {
        "selected_track_id": int(primary["track_id"]),
        "selected_track_index": int(primary["track_index"]),
        "selected_track_stats": _strip_masks(primary),
        "secondary_max_ratio": float(secondary_max_ratio),
        "secondary_total_ratio": float(secondary_total_ratio),
        "secondary_best_track_id": int(secondary["track_id"]) if secondary is not None else None,
        "secondary_best_valid_count": int(secondary["valid_count"]) if secondary is not None else 0,
        "primary_margin_frames": int(primary_margin_frames),
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "track_candidates": [_strip_masks(dict(stat)) for stat in track_stats],
    }


def window_policy_rank_key(summary: Dict[str, object]) -> tuple:
    selected = summary.get("selected_track_stats") or {}
    return (
        int(bool(summary.get("passed", False))),
        int(selected.get("valid_count", 0)),
        int(selected.get("longest_run", 0)),
        int(selected.get("present_count", 0)),
        int(summary.get("primary_margin_frames", -1)),
        -float(summary.get("secondary_max_ratio", 1.0)),
        -float(summary.get("secondary_total_ratio", 1.0)),
        float(selected.get("score_sum", 0.0)),
    )
