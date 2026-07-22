#!/usr/bin/env python3
"""Collect misclassified 3D hand21 samples into one folder.

This script supports two input formats:
1. `misclassified_test.txt` already contains `video_name` in column 1.
2. `misclassified_test.txt` still has `None` in column 1. In this case the
   script replays the val split order against `classified_test.txt` and the pkl
   to recover the corresponding `frame_dir` / `video_name`.

For each misclassified sample it exports:
  - the clip mp4 from the 2D prepared segments directory
  - the per-clip 3D annotations.json
  - the per-video-kpts 3D annotations.json
  - a meta.json sidecar with prediction details

It also writes:
  - `misclassified_summary.csv`
  - `archive_command.sh`
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

HAND21_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
DEFAULT_PKL = Path("/home/results/6t1/gesture_class9_3d/gesture_class9_hand21_3d.pkl")
TRAJECTORY_WINDOW = 12
JOINT_COLORS = (
    (255, 255, 255),
    (90, 180, 255), (90, 180, 255), (90, 180, 255), (90, 180, 255),
    (80, 220, 120), (80, 220, 120), (80, 220, 120), (80, 220, 120),
    (80, 220, 220), (80, 220, 220), (80, 220, 220), (80, 220, 220),
    (220, 180, 80), (220, 180, 80), (220, 180, 80), (220, 180, 80),
    (220, 80, 220), (220, 80, 220), (220, 80, 220), (220, 80, 220),
)
JOINT_COLOR = (80, 220, 255)
EDGE_COLOR = (0, 180, 80)
TEXT_COLOR = (20, 230, 20)
MISSING_COLOR = (0, 0, 255)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect misclassified 3D hand21 samples.")
    parser.add_argument(
        "--pkl",
        type=Path,
        default=DEFAULT_PKL,
        help="PoseDataset pkl used during evaluation.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Experiment work dir that contains misclassified_test.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Export dir. Defaults to <work-dir>/misclassified_samples.",
    )
    parser.add_argument("--bad-file", type=str, default="misclassified_test.txt")
    parser.add_argument("--good-file", type=str, default="classified_test.txt")
    parser.add_argument(
        "--base-2d",
        type=Path,
        default=Path("/home/results/6t1/gesture_class9"),
        help="Base dir that contains *_prepared_*_segments_videos.",
    )
    parser.add_argument(
        "--base-3d",
        type=Path,
        default=Path("/home/results/6t1/gesture_class9_3d"),
        help="Base dir that contains 3D prepared roots.",
    )
    parser.add_argument("--none-class-name", type=str, default="NONE")
    parser.add_argument(
        "--skip-none-misclassified",
        action="store_true",
        help="Skip misclassified samples whose gt or pred matches --none-class-name.",
    )
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--crop-mode", type=str, default="center", choices=["center", "left"])
    parser.add_argument("--sampled-fps", type=int, default=4)
    parser.add_argument("--skip-sampled-media", action="store_true")
    parser.add_argument("--save-overlay-video", action="store_true")
    parser.add_argument("--save-sampled-overlay-video", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return str(name).replace("/", "__").replace("\\", "__")


def is_none_label(label_name: str, none_class_name: str) -> bool:
    return str(label_name).strip().casefold() == str(none_class_name).strip().casefold()


def infer_ann_file_from_config(config_path: Path) -> Optional[Path]:
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        return None

    patterns = (
        r"^\s*ann_file\s*=\s*['\"]([^'\"]+)['\"]",
        r"ann_file\s*=\s*['\"]([^'\"]+)['\"]",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            return Path(match.group(1))
    return None


def resolve_pkl_path(work_dir: Path, requested_pkl: Path) -> Tuple[Path, List[str]]:
    warnings: List[str] = []
    config_candidates = list(work_dir.glob("*.py"))
    config_candidates.extend(sorted(work_dir.glob("*/vis_data/config.py")))
    inferred: Optional[Path] = None
    for config_path in config_candidates:
        ann_file = infer_ann_file_from_config(config_path)
        if ann_file is None:
            continue
        inferred = ann_file
        break

    if inferred is None:
        return requested_pkl, warnings

    if requested_pkl == DEFAULT_PKL and inferred != requested_pkl:
        warnings.append(
            f"Using ann_file from {config_path} instead of default pkl: {inferred}"
        )
        return inferred, warnings

    if inferred != requested_pkl:
        warnings.append(
            f"work-dir config ann_file differs from --pkl: config={inferred}, requested={requested_pkl}"
        )
    return requested_pkl, warnings


def read_prediction_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            parts = raw.rstrip("\n").split(",")
            if len(parts) < 4:
                continue
            rows.append(
                {
                    "name": parts[0].strip(),
                    "pred": parts[1].strip(),
                    "gt": parts[2].strip(),
                    "score": parts[3].strip(),
                }
            )
    return rows


def recover_bad_rows(
    bad_rows: List[dict],
    good_rows: List[dict],
    *,
    val_ids: List[str],
    id_to_ann: Dict[str, dict],
    categories: List[str],
) -> List[dict]:
    if bad_rows and bad_rows[0]["name"] not in {"", "None", "none", "NULL", "null"}:
        by_video_name = {str(ann.get("video_name", "")): frame_dir for frame_dir, ann in id_to_ann.items()}
        recovered: List[dict] = []
        for row in bad_rows:
            video_name = row["name"]
            frame_dir = by_video_name.get(video_name)
            recovered.append(
                {
                    "frame_dir": frame_dir,
                    "video_name": video_name,
                    "gt": row["gt"],
                    "pred": row["pred"],
                    "score": row["score"],
                }
            )
        return recovered

    if len(val_ids) != len(bad_rows) + len(good_rows):
        raise ValueError(
            "Line count mismatch between val split and classified/misclassified files. "
            "Please make sure they come from the same evaluation run."
        )

    bad_ptr = 0
    good_ptr = 0
    recovered_bad: List[dict] = []

    for val_id in val_ids:
        ann = id_to_ann[val_id]
        gt_name = categories[int(ann["label"])]

        next_bad_gt = bad_rows[bad_ptr]["gt"] if bad_ptr < len(bad_rows) else None
        next_good_gt = good_rows[good_ptr]["gt"] if good_ptr < len(good_rows) else None

        bad_match = next_bad_gt == gt_name
        good_match = next_good_gt == gt_name

        if bad_match and not good_match:
            row = bad_rows[bad_ptr]
            bad_ptr += 1
            recovered_bad.append(
                {
                    "frame_dir": val_id,
                    "video_name": str(ann.get("video_name", "")),
                    "gt": row["gt"],
                    "pred": row["pred"],
                    "score": row["score"],
                }
            )
        elif good_match and not bad_match:
            good_ptr += 1
        elif bad_match and good_match:
            bad_pred_is_wrong = bad_rows[bad_ptr]["pred"] != gt_name
            good_pred_is_right = good_rows[good_ptr]["pred"] == gt_name
            if bad_pred_is_wrong and good_pred_is_right:
                row = bad_rows[bad_ptr]
                bad_ptr += 1
                recovered_bad.append(
                    {
                        "frame_dir": val_id,
                        "video_name": str(ann.get("video_name", "")),
                        "gt": row["gt"],
                        "pred": row["pred"],
                        "score": row["score"],
                    }
                )
            else:
                good_ptr += 1
        else:
            raise ValueError(
                f"Cannot align val sample {val_id}: gt={gt_name}, "
                f"next_bad_gt={next_bad_gt}, next_good_gt={next_good_gt}"
            )

    return recovered_bad


def build_sample_paths(
    *,
    base_2d: Path,
    base_3d: Path,
    prefix: str,
    split_name: str,
    rel_video_name: Path,
) -> dict:
    return {
        "clip_mp4": base_2d / f"{prefix}_prepared_{split_name}_segments_videos" / rel_video_name.with_suffix(".mp4"),
        "clip_json_3d": base_3d / f"{prefix}_prepared_{split_name}" / rel_video_name.parent / "clips" / f"{rel_video_name.name}.json",
        "video_kpts_json_3d": base_3d / f"{prefix}_prepared_{split_name}_video_kpts" / rel_video_name / "annotations.json",
    }


def resolve_relative_video_name(
    *,
    base_2d: Path,
    split_name: str,
    prefixes: List[str],
    video_name: str,
    ann: Optional[dict],
) -> Tuple[Optional[str], List[str]]:
    warnings: List[str] = []

    if ann is not None:
        raw_video_path = str(ann.get("video_path", "")).strip()
        if raw_video_path:
            abs_video_path = Path(raw_video_path)
            for prefix in prefixes:
                root = base_2d / f"{prefix}_prepared_{split_name}_segments_videos"
                try:
                    rel = abs_video_path.relative_to(root)
                    return str(rel.with_suffix("")), warnings
                except ValueError:
                    continue

    video_path = Path(video_name)
    if len(video_path.parts) > 1:
        return str(video_path), warnings

    unique_matches: List[Tuple[str, Path]] = []
    segment_stem = video_path.name
    for prefix in prefixes:
        root = base_2d / f"{prefix}_prepared_{split_name}_segments_videos"
        if not root.exists():
            continue
        matches = sorted(root.rglob(f"{segment_stem}.mp4"))
        for match_path in matches:
            unique_matches.append((prefix, match_path.relative_to(root).with_suffix("")))

    if len(unique_matches) == 1:
        _, rel = unique_matches[0]
        return str(rel), warnings

    if len(unique_matches) > 1:
        warnings.append(
            f"Multiple clip matches found for {video_name} under split={split_name}; using first match {unique_matches[0][1]}"
        )
        _, rel = unique_matches[0]
        return str(rel), warnings

    return None, warnings


def infer_sample_paths(
    *,
    base_2d: Path,
    base_3d: Path,
    split_name: str,
    video_name: str,
    label_name: str,
    none_class_name: str,
    ann: Optional[dict] = None,
) -> Tuple[dict, List[str]]:
    prefixes = ["gesture_class9"]
    if label_name == none_class_name:
        prefixes = ["none", "gesture_class9"]

    rel_video_name, warnings = resolve_relative_video_name(
        base_2d=base_2d,
        split_name=split_name,
        prefixes=prefixes,
        video_name=video_name,
        ann=ann,
    )
    if rel_video_name is not None:
        rel_path = Path(rel_video_name)
        for prefix in prefixes:
            paths = build_sample_paths(
                base_2d=base_2d,
                base_3d=base_3d,
                prefix=prefix,
                split_name=split_name,
                rel_video_name=rel_path,
            )
            if any(path.exists() for path in paths.values()):
                return paths, warnings

    fallback = build_sample_paths(
        base_2d=base_2d,
        base_3d=base_3d,
        prefix=prefixes[0],
        split_name=split_name,
        rel_video_name=Path(video_name),
    )
    return fallback, warnings


def copy_if_exists(src: Path, dst: Path, *, skip_missing: bool) -> Optional[str]:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return None
    message = f"missing: {src}"
    if skip_missing:
        return message
    raise FileNotFoundError(message)


def write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def find_ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def import_cv2():
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    return cv2


def log_progress(message: str) -> None:
    print(message, flush=True)


def compute_deterministic_indices(total_frames: int, clip_len: int, crop_mode: str) -> List[int]:
    if total_frames <= 0:
        return []
    if total_frames <= clip_len or crop_mode == "left":
        start = 0
    else:
        start = (total_frames - clip_len) // 2
    return [((start + i) % total_frames) for i in range(clip_len)]


def build_select_expr(frame_indices: List[int]) -> str:
    unique_sorted = sorted(dict.fromkeys(int(i) for i in frame_indices))
    return "+".join(f"eq(n\\,{idx})" for idx in unique_sorted)


def export_sampled_media(
    *,
    ffmpeg_bin: str,
    clip_mp4_path: Path,
    sample_dir: Path,
    frame_indices: List[int],
    sampled_fps: int,
) -> None:
    sampled_dir = sample_dir / "sampled_frames"
    sampled_dir.mkdir(parents=True, exist_ok=True)
    select_expr = build_select_expr(frame_indices)

    frames_cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(clip_mp4_path),
        "-vf",
        f"select={select_expr}",
        "-vsync",
        "0",
        str(sampled_dir / "frame_%02d.jpg"),
    ]
    subprocess.run(frames_cmd, check=True)

    clip_cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(clip_mp4_path),
        "-vf",
        f"select={select_expr},setpts=N/{sampled_fps}/TB",
        "-an",
        "-r",
        str(sampled_fps),
        str(sample_dir / "sampled_clip.mp4"),
    ]
    subprocess.run(clip_cmd, check=True)


def load_clip_annotations(clip_json_path: Path) -> List[dict]:
    with clip_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload.get("annotations", []))


def extract_overlay_points(ann: dict) -> List[tuple[int, int]]:
    raw_points = ann.get("kpts_xy_smooth") or ann.get("kpts_xy") or []
    points: List[tuple[int, int]] = []
    for item in raw_points:
        if not isinstance(item, list) or len(item) < 2:
            points.append((0, 0))
            continue
        points.append((int(round(float(item[0]))), int(round(float(item[1])))))
    return points


def draw_trajectory(cv2, canvas, history_points: List[List[tuple[int, int]]], detected_hist: List[bool]) -> None:
    if len(history_points) < 2:
        return
    num_joints = min(len(history_points[0]), len(JOINT_COLORS))
    for joint_idx in range(num_joints):
        valid_points: List[tuple[int, int]] = []
        for points, detected in zip(history_points, detected_hist):
            if not detected or joint_idx >= len(points):
                continue
            point = points[joint_idx]
            if point[0] <= 0 or point[1] <= 0:
                continue
            valid_points.append(point)
        if len(valid_points) < 2:
            continue
        base_color = JOINT_COLORS[joint_idx]
        for idx in range(1, len(valid_points)):
            alpha_color = tuple(max(40, int(channel * idx / len(valid_points))) for channel in base_color)
            cv2.line(canvas, valid_points[idx - 1], valid_points[idx], alpha_color, 1, cv2.LINE_AA)


def draw_hand_overlay(
    cv2,
    frame,
    ann: dict,
    frame_idx: int,
    history_annotations: Optional[List[dict]] = None,
) -> object:
    canvas = frame.copy()
    detected = bool(ann.get("detected", False))
    points = extract_overlay_points(ann)

    if history_annotations:
        history_points = [extract_overlay_points(item) for item in history_annotations]
        history_detected = [bool(item.get("detected", False)) for item in history_annotations]
        draw_trajectory(cv2, canvas, history_points, history_detected)

    if detected and len(points) >= 21:
        for start, end in HAND21_EDGES:
            pt1 = points[start]
            pt2 = points[end]
            cv2.line(canvas, pt1, pt2, EDGE_COLOR, 2, cv2.LINE_AA)
        for joint_idx, (x, y) in enumerate(points[:21]):
            cv2.circle(canvas, (x, y), 3, JOINT_COLOR, -1, cv2.LINE_AA)
            if joint_idx in {0, 4, 8, 12, 16, 20}:
                cv2.putText(
                    canvas,
                    str(joint_idx),
                    (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    JOINT_COLOR,
                    1,
                    cv2.LINE_AA,
                )
        label = f"frame={frame_idx} detected=1 handed={ann.get('handedness_label', 'Unknown')} score={float(ann.get('handedness_score', 0.0)):.3f}"
        label_color = TEXT_COLOR
    else:
        label = f"frame={frame_idx} detected=0"
        label_color = MISSING_COLOR

    cv2.putText(
        canvas,
        label,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        label_color,
        2,
        cv2.LINE_AA,
    )
    return canvas


def export_overlay_video(
    *,
    cv2,
    clip_mp4_path: Path,
    clip_annotations: List[dict],
    output_path: Path,
    frame_indices: Optional[List[int]] = None,
    output_fps: Optional[int] = None,
) -> None:
    cap = cv2.VideoCapture(str(clip_mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open clip video: {clip_mp4_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not src_fps or src_fps <= 0:
        src_fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(output_fps or src_fps), (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open overlay writer: {output_path}")

    frame_index_set = set(frame_indices) if frame_indices is not None else None
    selected_order = frame_indices or []
    selected_cursor = 0
    current_frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if current_frame_idx >= len(clip_annotations):
                break
            should_write = frame_index_set is None or current_frame_idx in frame_index_set
            if should_write:
                ann = clip_annotations[current_frame_idx]
                hist_start = max(0, current_frame_idx - TRAJECTORY_WINDOW + 1)
                overlaid = draw_hand_overlay(
                    cv2,
                    frame,
                    ann,
                    current_frame_idx,
                    history_annotations=clip_annotations[hist_start : current_frame_idx + 1],
                )
                writer.write(overlaid)
                if frame_indices is not None:
                    if selected_cursor < len(selected_order) and current_frame_idx == selected_order[selected_cursor]:
                        selected_cursor += 1
            current_frame_idx += 1
    finally:
        cap.release()
        writer.release()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.work_dir / "misclassified_samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_path, pkl_warnings = resolve_pkl_path(args.work_dir, args.pkl)
    log_progress(f"[misclf] work_dir={args.work_dir}")
    log_progress(f"[misclf] output_dir={output_dir}")
    log_progress(f"[misclf] loading pkl={pkl_path}")
    with pkl_path.open("rb") as f:
        payload = pickle.load(f)

    categories = [str(x) for x in payload["categories"]]
    annotations = payload["annotations"]
    split_map = payload["split"]
    val_ids = [str(x) for x in split_map["val"]]
    id_to_ann = {str(ann["frame_dir"]): ann for ann in annotations}
    frame_dir_to_split = {
        str(frame_dir): split_name
        for split_name, frame_dirs in split_map.items()
        for frame_dir in frame_dirs
    }

    bad_path = args.work_dir / args.bad_file
    good_path = args.work_dir / args.good_file
    bad_rows = read_prediction_rows(bad_path)
    good_rows = read_prediction_rows(good_path)
    recovered_bad = recover_bad_rows(
        bad_rows,
        good_rows,
        val_ids=val_ids,
        id_to_ann=id_to_ann,
        categories=categories,
    )
    ffmpeg_bin = None if args.skip_sampled_media else find_ffmpeg()
    cv2 = import_cv2() if (args.save_overlay_video or args.save_sampled_overlay_video) else None
    if not args.skip_sampled_media and ffmpeg_bin is None:
        warnings = ["ffmpeg not found; sampled frame images and sampled_clip.mp4 were skipped."]
    else:
        warnings = []
    warnings.extend(pkl_warnings)
    if (args.save_overlay_video or args.save_sampled_overlay_video) and cv2 is None:
        warnings.append("cv2 not found; overlay videos were skipped.")
    for warning in pkl_warnings:
        log_progress(f"[misclf][warn] {warning}")

    log_progress(
        f"[misclf] loaded val_split={len(val_ids)} recovered_misclassified={len(recovered_bad)} "
        f"save_overlay={bool(args.save_overlay_video)} save_sampled_overlay={bool(args.save_sampled_overlay_video)}"
    )

    summary_rows: List[dict] = []
    skipped_none_rows = 0

    total_bad = len(recovered_bad)
    for index, row in enumerate(recovered_bad, start=1):
        frame_dir = row["frame_dir"]
        if not frame_dir:
            raise ValueError(f"Cannot resolve frame_dir for row: {row}")

        ann = id_to_ann.get(frame_dir)
        if ann is None:
            raise KeyError(f"frame_dir not found in pkl annotations: {frame_dir}")

        video_name = str(row["video_name"] or ann.get("video_name") or frame_dir)
        split_name = frame_dir_to_split.get(frame_dir)
        if split_name is None:
            raise KeyError(f"frame_dir not found in pkl split map: {frame_dir}")

        label_idx = int(ann["label"])
        gt_name = row["gt"]
        pred_name = row["pred"]
        if args.skip_none_misclassified and (
            is_none_label(gt_name, args.none_class_name)
            or is_none_label(pred_name, args.none_class_name)
        ):
            skipped_none_rows += 1
            log_progress(
                f"[misclf][{index}/{total_bad}] skip NONE-related sample gt={gt_name} pred={pred_name} video={video_name}"
            )
            continue
        log_progress(
            f"[misclf][{index}/{total_bad}] start gt={gt_name} pred={pred_name} "
            f"score={row['score']} split={split_name} video={video_name}"
        )
        total_frames = int(ann["total_frames"])
        source_paths, path_warnings = infer_sample_paths(
            base_2d=args.base_2d,
            base_3d=args.base_3d,
            split_name=split_name,
            video_name=video_name,
            label_name=categories[label_idx],
            none_class_name=args.none_class_name,
            ann=ann,
        )
        warnings.extend(path_warnings)
        if path_warnings:
            for warning in path_warnings:
                log_progress(f"[misclf][{index}/{total_bad}][warn] {warning}")

        sample_slug = f"{index:03d}__{sanitize_name(gt_name)}__to__{sanitize_name(pred_name)}__{sanitize_name(video_name)}"
        sample_dir = output_dir / "samples" / sample_slug
        sample_dir.mkdir(parents=True, exist_ok=True)
        frame_indices = compute_deterministic_indices(total_frames, args.clip_len, args.crop_mode)
        clip_annotations: Optional[List[dict]] = None
        log_progress(
            f"[misclf][{index}/{total_bad}] resolved clip_mp4={source_paths['clip_mp4']} "
            f"clip_json_3d={source_paths['clip_json_3d']}"
        )

        missing_items: List[str] = []
        for key, src in source_paths.items():
            ext = ".mp4" if src.suffix.lower() == ".mp4" else ".json"
            dst_name = key + ext
            warning = copy_if_exists(src, sample_dir / dst_name, skip_missing=args.skip_missing)
            if warning:
                missing_items.append(key)
                warnings.append(warning)
                log_progress(f"[misclf][{index}/{total_bad}][warn] {warning}")
        if not missing_items:
            log_progress(f"[misclf][{index}/{total_bad}] copied source files")

        overlay_missing = False
        overlay_error = None
        sampled_overlay_missing = False
        sampled_overlay_error = None
        if cv2 is not None and (args.save_overlay_video or args.save_sampled_overlay_video):
            clip_json_path = source_paths["clip_json_3d"]
            clip_mp4_path = source_paths["clip_mp4"]
            if clip_json_path.exists() and clip_mp4_path.exists():
                try:
                    clip_annotations = load_clip_annotations(clip_json_path)
                    if args.save_overlay_video:
                        export_overlay_video(
                            cv2=cv2,
                            clip_mp4_path=clip_mp4_path,
                            clip_annotations=clip_annotations,
                            output_path=sample_dir / "overlay_clip.mp4",
                        )
                        log_progress(f"[misclf][{index}/{total_bad}] wrote overlay_clip.mp4")
                    if args.save_sampled_overlay_video:
                        export_overlay_video(
                            cv2=cv2,
                            clip_mp4_path=clip_mp4_path,
                            clip_annotations=clip_annotations,
                            output_path=sample_dir / "sampled_overlay_clip.mp4",
                            frame_indices=frame_indices,
                            output_fps=args.sampled_fps,
                        )
                        log_progress(f"[misclf][{index}/{total_bad}] wrote sampled_overlay_clip.mp4")
                except Exception as exc:
                    if args.save_overlay_video:
                        overlay_error = f"overlay export failed for {clip_mp4_path}: {exc}"
                        warnings.append(overlay_error)
                        log_progress(f"[misclf][{index}/{total_bad}][warn] {overlay_error}")
                    if args.save_sampled_overlay_video:
                        sampled_overlay_error = f"sampled overlay export failed for {clip_mp4_path}: {exc}"
                        warnings.append(sampled_overlay_error)
                        log_progress(f"[misclf][{index}/{total_bad}][warn] {sampled_overlay_error}")
            else:
                if args.save_overlay_video:
                    overlay_missing = True
                    log_progress(
                        f"[misclf][{index}/{total_bad}][warn] overlay input missing: clip_json_3d_exists={clip_json_path.exists()} clip_mp4_exists={clip_mp4_path.exists()}"
                    )
                if args.save_sampled_overlay_video:
                    sampled_overlay_missing = True

        sampled_media_missing = False
        sampled_media_error = None
        sampled_frame_indices_path = sample_dir / "sampled_frame_indices.json"
        with sampled_frame_indices_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "sampler": "DeterministicContinuousSampleFrames",
                    "clip_len": args.clip_len,
                    "crop_mode": args.crop_mode,
                    "total_frames": total_frames,
                    "frame_indices": frame_indices,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        if not args.skip_sampled_media:
            clip_mp4_path = source_paths["clip_mp4"]
            if clip_mp4_path.exists():
                if ffmpeg_bin is not None:
                    try:
                        export_sampled_media(
                            ffmpeg_bin=ffmpeg_bin,
                            clip_mp4_path=clip_mp4_path,
                            sample_dir=sample_dir,
                            frame_indices=frame_indices,
                            sampled_fps=args.sampled_fps,
                        )
                        log_progress(f"[misclf][{index}/{total_bad}] wrote sampled media")
                    except subprocess.CalledProcessError as exc:
                        sampled_media_error = f"ffmpeg failed for {clip_mp4_path}: {exc}"
                        warnings.append(sampled_media_error)
                        log_progress(f"[misclf][{index}/{total_bad}][warn] {sampled_media_error}")
                else:
                    sampled_media_missing = True
                    log_progress(f"[misclf][{index}/{total_bad}][warn] ffmpeg missing; sampled media skipped")
            else:
                sampled_media_missing = True
                log_progress(f"[misclf][{index}/{total_bad}][warn] sampled media source missing: {clip_mp4_path}")

        meta = {
            "index": index,
            "frame_dir": frame_dir,
            "video_name": video_name,
            "split": split_name,
            "label_index": label_idx,
            "label_name_from_pkl": categories[label_idx],
            "gt": gt_name,
            "pred": pred_name,
            "score": row["score"],
            "total_frames": total_frames,
            "sampled_clip_len": args.clip_len,
            "sampled_crop_mode": args.crop_mode,
            "sampled_frame_indices": frame_indices,
            "paths": {k: str(v) for k, v in source_paths.items()},
            "pkl_path": str(pkl_path),
            "missing_items": missing_items,
            "path_resolution_warnings": path_warnings,
            "sampled_media_missing": sampled_media_missing,
            "sampled_media_error": sampled_media_error,
            "overlay_missing": overlay_missing,
            "overlay_error": overlay_error,
            "sampled_overlay_missing": sampled_overlay_missing,
            "sampled_overlay_error": sampled_overlay_error,
        }
        with (sample_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        summary_rows.append(
            {
                "index": index,
                "frame_dir": frame_dir,
                "video_name": video_name,
                "split": split_name,
                "gt": gt_name,
                "pred": pred_name,
                "score": row["score"],
                "total_frames": total_frames,
                "sampled_frame_indices_json": str(sampled_frame_indices_path),
                "sampled_clip_mp4": str(sample_dir / "sampled_clip.mp4"),
                "sampled_frames_dir": str(sample_dir / "sampled_frames"),
                "overlay_clip_mp4": str(sample_dir / "overlay_clip.mp4"),
                "sampled_overlay_clip_mp4": str(sample_dir / "sampled_overlay_clip.mp4"),
                "sample_dir": str(sample_dir),
                "clip_mp4": str(source_paths["clip_mp4"]),
                "clip_json_3d": str(source_paths["clip_json_3d"]),
                "video_kpts_json_3d": str(source_paths["video_kpts_json_3d"]),
                "pkl_path": str(pkl_path),
                "path_resolution_warnings": ";".join(path_warnings),
                "missing_items": ";".join(missing_items),
                "sampled_media_missing": str(sampled_media_missing),
                "sampled_media_error": sampled_media_error or "",
                "overlay_missing": str(overlay_missing),
                "overlay_error": overlay_error or "",
                "sampled_overlay_missing": str(sampled_overlay_missing),
                "sampled_overlay_error": sampled_overlay_error or "",
            }
        )
        log_progress(
            f"[misclf][{index}/{total_bad}] done sample_dir={sample_dir} "
            f"missing_items={len(missing_items)} overlay_missing={overlay_missing} "
            f"sampled_overlay_missing={sampled_overlay_missing} sampled_media_missing={sampled_media_missing}"
        )

    summary_csv = output_dir / "misclassified_summary.csv"
    write_csv(
        summary_csv,
        summary_rows,
        fieldnames=[
            "index",
            "frame_dir",
            "video_name",
            "split",
            "gt",
            "pred",
            "score",
            "total_frames",
            "sampled_frame_indices_json",
            "sampled_clip_mp4",
            "sampled_frames_dir",
            "overlay_clip_mp4",
            "sampled_overlay_clip_mp4",
            "sample_dir",
            "clip_mp4",
            "clip_json_3d",
            "video_kpts_json_3d",
            "pkl_path",
            "path_resolution_warnings",
            "missing_items",
            "sampled_media_missing",
            "sampled_media_error",
            "overlay_missing",
            "overlay_error",
            "sampled_overlay_missing",
            "sampled_overlay_error",
        ],
    )

    pair_counter = Counter((row["gt"], row["pred"]) for row in summary_rows)
    stats_path = output_dir / "stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_misclassified": len(summary_rows),
                "num_skipped_none_misclassified": skipped_none_rows,
                "confusion_pairs": [
                    {"gt": gt, "pred": pred, "count": count}
                    for (gt, pred), count in sorted(pair_counter.items(), key=lambda item: (-item[1], item[0]))
                ],
                "warnings": warnings,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    tar_name = output_dir.name.rstrip("/") + ".tar.gz"
    archive_cmd = f'tar -czf "{output_dir.parent / tar_name}" -C "{output_dir.parent}" "{output_dir.name}"'
    archive_cmd_path = output_dir / "archive_command.sh"
    with archive_cmd_path.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n")
        f.write(archive_cmd + "\n")

    print(f"Collected {len(summary_rows)} misclassified samples into {output_dir}")
    if args.skip_none_misclassified:
        print(f"Skipped {skipped_none_rows} NONE-related misclassified samples")
    print(f"Summary CSV: {summary_csv}")
    print(f"Archive command: {archive_cmd}")
    if warnings:
        print(f"Warnings: {len(warnings)} missing files were recorded in {stats_path}")


if __name__ == "__main__":
    main()
