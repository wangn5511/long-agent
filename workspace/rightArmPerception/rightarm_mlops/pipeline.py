"""Deterministic raw-batch to RTM preparation pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from rightarm_mlops.protocol import atomic_json, safe_extract_tar, sha256_file


def prepare_raw_rtm_splits(
    *,
    raw_dataset: Path,
    repo_root: Path,
    output_root: Path,
    det_onnx: Path,
    person_pose_onnx: Path,
    hand_pose_onnx: Path,
    frame_base: int,
    val_ratio: float,
    seed: int,
    device: str,
    workers: int,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """Materialize, split, crop and run the fixed RTM extraction script."""

    if frame_base not in {0, 1}:
        raise ValueError("frame_base must be 0 or 1")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if workers < 1:
        raise ValueError("workers must be positive")
    repo_root = repo_root.resolve(strict=True)
    for model in (det_onnx, person_pose_onnx, hand_pose_onnx):
        model.resolve(strict=True)
    extracted = output_root / "raw-extracted"
    materialized = output_root / "raw-materialized"
    safe_extract_tar(raw_dataset, extracted)
    canonical = _materialize_snapshot(extracted, materialized)
    annotation_path = materialized / "annotations.csv"
    split_path = output_root / "source-split.json"
    split = build_grouped_split(
        canonical,
        annotation_path=annotation_path,
        materialized_root=materialized,
        val_ratio=val_ratio,
        seed=seed,
    )
    atomic_json(split_path, split)

    clips = output_root / "clips"
    segment_log = output_root / "segment-job.log"
    _run(
        [
            sys.executable,
            str(repo_root / "data/gesture_data/cut_segments_from_annotation_csv.py"),
            "--annotation_csv",
            str(annotation_path),
            "--video_root",
            str(materialized / "videos"),
            "--out_dir",
            str(clips),
            "--annotation_frame_base",
            str(frame_base),
        ],
        cwd=repo_root,
        log_path=segment_log,
    )
    segment_summary_path = clips / "segment_summary.json"
    segment_summary = json.loads(segment_summary_path.read_text(encoding="utf-8"))
    missing = int(segment_summary.get("status_counts", {}).get("missing_video", 0))
    invalid = int(segment_summary.get("status_counts", {}).get("invalid_range", 0))
    if missing or invalid:
        raise ValueError(
            f"segment preparation failed hard gates: missing={missing}, invalid={invalid}"
        )

    rtm_all = output_root / "rtm-all"
    rtm_log = output_root / "rtm-job.log"
    _run(
        [
            sys.executable,
            str(
                repo_root / "data/gesture_data/extract_rtm_kpts_from_prepared_clips.py"
            ),
            "--src_dir",
            str(clips),
            "--dst_dir",
            str(rtm_all),
            "--output_layout",
            "mirror",
            "--det_onnx",
            str(det_onnx.resolve()),
            "--person_pose_onnx",
            str(person_pose_onnx.resolve()),
            "--hand_pose_onnx",
            str(hand_pose_onnx.resolve()),
            "--device",
            device,
            "--num_workers",
            str(workers),
        ],
        cwd=repo_root,
        log_path=rtm_log,
    )
    train_root = output_root / "rtm-train"
    val_root = output_root / "rtm-val"
    split_report = _partition_rtm_outputs(
        rtm_all=rtm_all,
        segment_summary=segment_summary,
        split=split,
        train_root=train_root,
        val_root=val_root,
    )
    pipeline_report = {
        "schema_version": "rightarm.gesture.data-pipeline/v1",
        "raw_dataset_sha256": sha256_file(raw_dataset.resolve(strict=True))[0],
        "source_split": split,
        "rtm_partition": split_report,
        "commands": {
            "segment_log": str(segment_log),
            "rtm_log": str(rtm_log),
        },
    }
    atomic_json(output_root / "data-pipeline-report.json", pipeline_report)
    return train_root, val_root, pipeline_report


def build_grouped_split(
    canonical: Dict[str, Any],
    *,
    annotation_path: Path,
    materialized_root: Path,
    val_ratio: float,
    seed: int,
) -> Dict[str, Any]:
    """Assign subject/device/session groups without cross-split leakage."""

    object_by_name: Dict[str, Dict[str, Any]] = {}
    for item in canonical["objects"]:
        relative = Path(item["source_relative_path"])
        if relative.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
            continue
        if relative.name in object_by_name:
            raise ValueError(f"ambiguous video basename in batch: {relative.name}")
        object_by_name[relative.name] = item
    referenced = _annotation_video_names(annotation_path)
    missing = sorted(referenced - set(object_by_name))
    if missing:
        raise ValueError(f"annotations reference missing videos: {missing[:20]}")
    groups: Dict[str, list[str]] = {}
    video_records: Dict[str, Dict[str, Any]] = {}
    for video_name in sorted(referenced):
        item = object_by_name[video_name]
        group_parts = (
            str(item.get("subject_group_id") or "anonymous"),
            str(item.get("device_id") or "unknown-device"),
            str(item.get("session_id") or Path(item["source_relative_path"]).parent),
        )
        group_id = hashlib.sha256("\x00".join(group_parts).encode()).hexdigest()
        source_path = (materialized_root / item["source_relative_path"]).resolve()
        groups.setdefault(group_id, []).append(str(source_path))
        video_records[str(source_path)] = {
            "video_name": video_name,
            "group_id": group_id,
        }
    group_ids = sorted(
        groups,
        key=lambda value: hashlib.sha256(f"{seed}\x00{value}".encode()).hexdigest(),
    )
    if len(group_ids) < 2:
        raise ValueError(
            "at least two independent subject/device/session groups are required"
        )
    val_count = min(max(round(len(group_ids) * val_ratio), 1), len(group_ids) - 1)
    val_groups = set(group_ids[:val_count])
    for source_path, record in video_records.items():
        record["split"] = "val" if record["group_id"] in val_groups else "train"
    return {
        "schema_version": "rightarm.gesture.grouped-split/v1",
        "seed": seed,
        "val_ratio": val_ratio,
        "group_key": ["subject_group_id", "device_id", "session_id"],
        "groups": {
            group_id: {
                "split": "val" if group_id in val_groups else "train",
                "source_videos": sorted(groups[group_id]),
            }
            for group_id in sorted(groups)
        },
        "videos": video_records,
    }


def _materialize_snapshot(extracted: Path, destination: Path) -> Dict[str, Any]:
    manifest_path = extracted / "raw-dataset.manifest.json"
    canonical = json.loads(manifest_path.read_text(encoding="utf-8"))
    if canonical.get("schema_version") != "rightarm.gesture.raw-dataset/v1":
        raise ValueError("raw dataset manifest schema is invalid")
    for item in canonical["objects"]:
        source = (extracted / item["path"]).resolve(strict=True)
        digest, size = sha256_file(source)
        if digest != item["sha256"] or size != item["size_bytes"]:
            raise ValueError(f"raw snapshot object is corrupt: {item['path']}")
        target = destination / item["source_relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    annotation = canonical["annotations"]
    source = (extracted / annotation["path"]).resolve(strict=True)
    digest, size = sha256_file(source)
    if digest != annotation["sha256"] or size != annotation["size_bytes"]:
        raise ValueError("raw snapshot annotations are corrupt")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination / "annotations.csv")
    return canonical


def _annotation_video_names(path: Path) -> set[str]:
    names: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            value = (row.get("视频名") or "").strip()
            if value:
                names.add(Path(value).name)
    return names


def _partition_rtm_outputs(
    *,
    rtm_all: Path,
    segment_summary: Dict[str, Any],
    split: Dict[str, Any],
    train_root: Path,
    val_root: Path,
) -> Dict[str, Any]:
    segment_sources = {
        str(Path(item["output_video"]).resolve()): str(
            Path(item["source_video"]).resolve()
        )
        for item in segment_summary["outputs"]
    }
    counts = {"train": 0, "val": 0}
    seen: set[str] = set()
    for annotation_path in sorted(rtm_all.rglob("annotations.json")):
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        clip_path = str(Path(payload["video"]).resolve())
        source_path = segment_sources.get(clip_path)
        if source_path is None or source_path not in split["videos"]:
            raise ValueError(
                f"RTM output cannot be mapped to a source split: {clip_path}"
            )
        split_name = split["videos"][source_path]["split"]
        destination = train_root if split_name == "train" else val_root
        identity = hashlib.sha256(clip_path.encode()).hexdigest()[:16]
        sample_id = f"{payload['clip_id']}__{identity}"
        if sample_id in seen:
            raise ValueError(f"duplicate RTM sample identity: {sample_id}")
        seen.add(sample_id)
        destination.mkdir(parents=True, exist_ok=True)
        target_video = destination / f"{sample_id}{Path(clip_path).suffix.lower()}"
        shutil.copyfile(clip_path, target_video)
        payload["clip_id"] = sample_id
        payload["video"] = str(target_video.resolve())
        payload["clip_path"] = str(target_video.resolve())
        atomic_json(destination / f"{sample_id}.json", payload)
        counts[split_name] += 1
    if not counts["train"] or not counts["val"]:
        raise ValueError(f"RTM partition produced an empty split: {counts}")
    return {"sample_counts": counts}


def _run(argv: list[str], *, cwd: Path, log_path: Path) -> None:
    with log_path.open("wb") as log:
        subprocess.run(
            argv,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
