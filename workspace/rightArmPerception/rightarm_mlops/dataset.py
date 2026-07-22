"""Compile unified RTM tracking JSON into the V3 MMAction PoseDataset."""

from __future__ import annotations

import json
import pickle
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from rightarm_mlops.batch import LABELS
from rightarm_mlops.protocol import atomic_json, deterministic_tar, sha256_file

LABEL_ORDER = [
    "WAVE_IN",
    "WAVE_OUT",
    "FIVE",
    "CLICK",
    "ROTATE",
    "SLIDE_DOWN",
    "SLIDE_UP",
    "OK",
    "CURSOR",
    "GUN",
    "NONE",
]


def compile_rtm_dataset(
    *,
    train_root: Path,
    val_root: Path,
    output_pkl: Path,
    report_path: Path,
    frozen_test_set: Optional[Path] = None,
    min_primary_valid_ratio: float = 0.5,
) -> Dict[str, Any]:
    if LABELS != set(LABEL_ORDER):
        raise AssertionError("label ontology constants disagree")
    split_ids: Dict[str, List[str]] = {"train": [], "val": []}
    annotations = []
    reports = []
    seen_ids = set()
    for split, root in (("train", train_root), ("val", val_root)):
        root = root.resolve(strict=True)
        json_paths = sorted(root.rglob("*.json"))
        if not json_paths:
            raise ValueError(f"no RTM JSON files found under {root}")
        for path in json_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "frames" not in payload or "clip_id" not in payload:
                continue
            sample, quality = _build_sample(
                payload,
                path,
                split=split,
                min_primary_valid_ratio=min_primary_valid_ratio,
            )
            sample_id = sample["frame_dir"]
            if sample_id in seen_ids:
                if any(
                    sample_id in split_ids[name] for name in split_ids if name != split
                ):
                    raise ValueError(
                        f"sample leakage across train/val splits: {sample_id}"
                    )
                sample_id = f"{split}__{path.parent.name}__{sample_id}"
                sample["frame_dir"] = sample_id
            if sample_id in seen_ids:
                raise ValueError(f"duplicate sample id after namespacing: {sample_id}")
            seen_ids.add(sample_id)
            split_ids[split].append(sample_id)
            annotations.append(sample)
            reports.append(quality)
    _assert_split_disjoint(split_ids)
    output = {
        "split": split_ids,
        "annotations": annotations,
        "categories": LABEL_ORDER,
        "label_mapping": {index: label for index, label in enumerate(LABEL_ORDER)},
        "schema_version": "rightarm.gesture.mmaction-pose/v1",
    }
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_pkl.with_suffix(".pkl.tmp")
    with temporary.open("xb") as stream:
        pickle.dump(output, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output_pkl)

    label_counts = Counter(LABEL_ORDER[item["label"]] for item in annotations)
    low_quality = [item for item in reports if not item["quality_gate_passed"]]
    report: Dict[str, Any] = {
        "schema_version": "rightarm.gesture.dataset-report/v1",
        "dataset": str(output_pkl.resolve()),
        "train_samples": len(split_ids["train"]),
        "val_samples": len(split_ids["val"]),
        "label_counts": {label: label_counts.get(label, 0) for label in LABEL_ORDER},
        "low_quality_samples": len(low_quality),
        "quality_gate": {"min_primary_valid_ratio": min_primary_valid_ratio},
        "samples": reports,
        "frozen_test_set": None,
    }
    if frozen_test_set is not None:
        frozen_test_set = frozen_test_set.resolve(strict=True)
        digest, size = sha256_file(frozen_test_set)
        report["frozen_test_set"] = {
            "path": str(frozen_test_set),
            "sha256": digest,
            "size_bytes": size,
            "excluded_from_training_splits": True,
        }
    atomic_json(report_path, report)
    return report


def build_portable_dataset_bundle(dataset_pkl: Path, output_tar: Path) -> None:
    """Bundle the PKL and videos using relative, content-addressed media paths."""

    dataset_pkl = dataset_pkl.resolve(strict=True)
    with dataset_pkl.open("rb") as stream:
        dataset = pickle.load(stream)  # noqa: S301 - produced by this trusted job
    with tempfile.TemporaryDirectory(
        dir=output_tar.parent, prefix=".dataset-bundle-"
    ) as temporary_name:
        root = Path(temporary_name)
        media = root / "media"
        media.mkdir()
        for sample in dataset["annotations"]:
            video = Path(sample["video_path"]).resolve(strict=True)
            digest, _ = sha256_file(video)
            relative = Path("media") / f"{digest}{video.suffix.lower()}"
            target = root / relative
            if not target.exists():
                shutil.copyfile(video, target)
            sample["video_path"] = relative.as_posix()
            sample["clip_video_path"] = relative.as_posix()
        with (root / "rightarm-gesture-11.pkl").open("wb") as stream:
            pickle.dump(dataset, stream, protocol=pickle.HIGHEST_PROTOCOL)
        deterministic_tar(root, output_tar)


def materialize_dataset_bundle(
    dataset_root: Path, dataset_pkl: Path, output_pkl: Path
) -> Path:
    """Rewrite relative bundle media references to verified local absolute paths."""

    with dataset_pkl.open("rb") as stream:
        dataset = pickle.load(stream)  # noqa: S301 - sealed trusted job artifact
    root = dataset_root.resolve(strict=True)
    for sample in dataset["annotations"]:
        for field in ("video_path", "clip_video_path"):
            relative = Path(sample[field])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"dataset bundle contains unsafe {field}: {relative}")
            resolved = (root / relative).resolve(strict=True)
            if not resolved.is_relative_to(root) or not resolved.is_file():
                raise ValueError(f"dataset bundle video is missing: {relative}")
            sample[field] = str(resolved)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("xb") as stream:
        pickle.dump(dataset, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return output_pkl


def _build_sample(
    payload: Dict[str, Any],
    source_path: Path,
    *,
    split: str,
    min_primary_valid_ratio: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    clip_id = str(payload["clip_id"])
    label = _label_from_clip_id(clip_id)
    total_frames = int(payload["total_frames"])
    if total_frames <= 0 or len(payload["frames"]) != total_frames:
        raise ValueError(f"invalid total_frames in {source_path}")
    resolution = payload.get("resolution")
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise ValueError(f"invalid resolution in {source_path}")
    img_w, img_h = (int(value) for value in resolution)
    tracks: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for frame in payload["frames"]:
        frame_index = int(frame["frame_idx"])
        if frame_index < 0 or frame_index >= total_frames:
            raise ValueError(
                f"frame index out of range in {source_path}: {frame_index}"
            )
        for hand in frame.get("hands", []):
            tracks[int(hand["track_id"])][frame_index] = hand
    if not tracks:
        raise ValueError(f"sample contains no tracked hands: {source_path}")
    track_ids = sorted(tracks)
    track_keypoint = np.zeros((len(track_ids), total_frames, 21, 3), dtype=np.float32)
    track_score = np.zeros((len(track_ids), total_frames, 21), dtype=np.float32)
    track_traj = np.zeros((len(track_ids), total_frames, 3), dtype=np.float32)
    track_frame_score = np.zeros((len(track_ids), total_frames), dtype=np.float32)
    handedness_ids = np.full((len(track_ids),), 2, dtype=np.int64)
    handedness_scores = np.zeros((len(track_ids),), dtype=np.float32)
    for track_index, track_id in enumerate(track_ids):
        handedness_votes: Counter[int] = Counter()
        for frame_index, hand in tracks[track_id].items():
            xyz = np.asarray(hand["hand21_kpts_xyz"], dtype=np.float32)
            scores = np.asarray(hand["hand21_kpt_scores"], dtype=np.float32)
            if xyz.shape != (21, 3) or scores.shape != (21,):
                raise ValueError(
                    f"track {track_id} has invalid hand21 tensor shape in {source_path}"
                )
            track_keypoint[track_index, frame_index] = xyz
            track_score[track_index, frame_index] = scores
            wrist = hand.get("upperbody_wrist_xyz")
            if wrist is None or len(wrist) != 3:
                wrist = xyz[0].tolist()
            track_traj[track_index, frame_index] = np.asarray(wrist, dtype=np.float32)
            frame_score = float(hand.get("score", float(scores.mean())))
            track_frame_score[track_index, frame_index] = frame_score
            handedness_votes[int(hand.get("handedness_label", 2))] += 1
        if handedness_votes:
            handedness_id, votes = handedness_votes.most_common(1)[0]
            handedness_ids[track_index] = handedness_id
            handedness_scores[track_index] = votes / sum(handedness_votes.values())
    valid_counts = np.count_nonzero(track_frame_score > 0, axis=1)
    primary_index = int(np.argmax(valid_counts))
    primary_track_id = track_ids[primary_index]
    primary_ratio = float(valid_counts[primary_index] / total_frames)
    video_path = Path(str(payload.get("clip_path") or payload.get("video", "")))
    if not video_path.is_file():
        raise ValueError(f"RTM sample video is missing: {video_path} ({source_path})")
    handedness_id = int(handedness_ids[primary_index])
    sample = {
        "frame_dir": clip_id,
        "label": LABEL_ORDER.index(label),
        "img_shape": (img_h, img_w),
        "original_shape": (img_h, img_w),
        "total_frames": total_frames,
        "keypoint": track_keypoint[primary_index : primary_index + 1],
        "keypoint_score": track_score[primary_index : primary_index + 1],
        "global_traj_source": track_traj[primary_index],
        "video_path": str(video_path.resolve()),
        "clip_video_path": str(video_path.resolve()),
        "bbox_per_frame": [None] * total_frames,
        "video_name": video_path.name,
        "handedness_label_id": handedness_id,
        "handedness_label": {0: "Left", 1: "Right", 2: "Unknown"}.get(
            handedness_id, "Unknown"
        ),
        "handedness_score": float(handedness_scores[primary_index]),
        "track_id": primary_track_id,
        "track_ids": np.asarray(track_ids, dtype=np.int64),
        "track_keypoint": track_keypoint,
        "track_keypoint_score": track_score,
        "track_global_traj_source": track_traj,
        "track_frame_score": track_frame_score,
        "track_handedness_label_id": handedness_ids,
        "track_handedness_score": handedness_scores,
    }
    quality = {
        "sample_id": clip_id,
        "split": split,
        "label": label,
        "total_frames": total_frames,
        "track_count": len(track_ids),
        "primary_track_id": primary_track_id,
        "primary_valid_ratio": primary_ratio,
        "quality_gate_passed": primary_ratio >= min_primary_valid_ratio,
        "source_json": str(source_path.resolve()),
    }
    return sample, quality


def _label_from_clip_id(clip_id: str) -> str:
    matches = [
        label
        for label in LABEL_ORDER
        if clip_id == label or clip_id.startswith(f"{label}_")
    ]
    if not matches:
        raise ValueError(f"cannot resolve an 11-class label from clip id {clip_id!r}")
    return max(matches, key=len)


def _assert_split_disjoint(split_ids: Dict[str, Sequence[str]]) -> None:
    overlap = set(split_ids["train"]) & set(split_ids["val"])
    if overlap:
        raise ValueError(f"train/val sample leakage detected: {sorted(overlap)[:20]}")
