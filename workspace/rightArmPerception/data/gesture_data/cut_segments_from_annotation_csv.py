#!/usr/bin/env python3
"""Cut annotated gesture segments from long videos.

The annotation CSV may use the compact layout used by the gesture data:

    video_name,start_frame,end_frame,label
    debug_20260331_180243,113,177,SLIDE_DOWN
    ,263,313,CLICK

Rows with an empty video name inherit the previous non-empty video name. Videos
that have no annotation rows are ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


@dataclass(frozen=True)
class Segment:
    row_idx: int
    video_name: str
    start_frame: int
    end_frame: int
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cut gesture clip videos from annotation CSV.")
    parser.add_argument("--annotation_csv", required=True, help="Annotation CSV path.")
    parser.add_argument(
        "--video_root",
        action="append",
        required=True,
        help="Root that contains long videos. Can be specified multiple times.",
    )
    parser.add_argument("--out_dir", required=True, help="Output root for segment mp4 files.")
    parser.add_argument(
        "--annotation_frame_base",
        type=int,
        default=1,
        choices=(0, 1),
        help="Frame index base used by annotation start/end columns. Default: 1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing segment files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N annotation rows after filtering. Default: all.",
    )
    parser.add_argument(
        "--include_labels",
        nargs="*",
        default=None,
        help="Only cut labels in this list.",
    )
    parser.add_argument(
        "--exclude_labels",
        nargs="*",
        default=None,
        help="Skip labels in this list.",
    )
    return parser.parse_args()


def _is_header(row: Sequence[str]) -> bool:
    lowered = [cell.strip().lower() for cell in row]
    header_words = {"video", "video_name", "start", "start_frame", "end", "end_frame", "label"}
    return any(cell in header_words for cell in lowered)


def _clean_label(label: str) -> str:
    label = label.strip()
    label = re.sub(r"\s+", "_", label)
    return label.upper()


def read_segments(csv_path: Path, frame_base: int) -> list[Segment]:
    segments: list[Segment] = []
    current_video = ""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row_idx, raw_row in enumerate(reader, start=1):
            if not raw_row or all(not cell.strip() for cell in raw_row):
                continue
            row = [cell.strip() for cell in raw_row]
            if row_idx == 1 and _is_header(row):
                continue
            if len(row) < 4:
                continue

            if row[0]:
                current_video = Path(row[0]).stem
            if not current_video:
                continue

            try:
                start_raw = int(float(row[1]))
                end_raw = int(float(row[2]))
            except ValueError:
                continue
            label = _clean_label(row[3])
            if not label:
                continue

            start = start_raw - frame_base
            end = end_raw - frame_base
            if start < 0 or end < start:
                continue

            segments.append(
                Segment(
                    row_idx=row_idx,
                    video_name=current_video,
                    start_frame=start,
                    end_frame=end,
                    label=label,
                )
            )
    return segments


def filter_segments(
    segments: list[Segment],
    include_labels: Sequence[str] | None,
    exclude_labels: Sequence[str] | None,
) -> list[Segment]:
    include = {_clean_label(label) for label in include_labels or [] if label.strip()}
    exclude = {_clean_label(label) for label in exclude_labels or [] if label.strip()}
    filtered = []
    for seg in segments:
        if include and seg.label not in include:
            continue
        if exclude and seg.label in exclude:
            continue
        filtered.append(seg)
    return filtered


def iter_video_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            yield path


def build_video_index(roots: Sequence[Path]) -> dict[str, list[tuple[Path, Path]]]:
    index: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for root in roots:
        for path in iter_video_files(root):
            index[path.stem].append((root, path))
    return dict(index)


def output_parent(out_dir: Path, root: Path, video_path: Path) -> Path:
    rel_video = video_path.relative_to(root)
    base = out_dir / root.name
    if video_path.parent.name == video_path.stem:
        return base / rel_video.parent
    return base / rel_video.parent / video_path.stem


def cut_segment(video_path: Path, out_path: Path, start_frame: int, end_frame: int, overwrite: bool) -> str:
    if out_path.exists() and not overwrite:
        return "skip_exists"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "failed_open"
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0 or start_frame >= total_frames:
            return "invalid_range"
        end_frame = min(end_frame, total_frames - 1)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )
        if not writer.isOpened():
            return "failed_writer"
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _frame_idx in range(start_frame, end_frame + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
        finally:
            writer.release()
    finally:
        cap.release()
    return "written"


def main() -> None:
    args = parse_args()
    annotation_csv = Path(args.annotation_csv).expanduser().resolve()
    video_roots = [Path(root).expanduser().resolve() for root in args.video_root]
    out_dir = Path(args.out_dir).expanduser().resolve()

    all_segments = read_segments(annotation_csv, args.annotation_frame_base)
    segments = filter_segments(all_segments, args.include_labels, args.exclude_labels)
    if args.limit > 0:
        segments = segments[: args.limit]
    if not segments:
        raise FileNotFoundError(f"No valid annotation rows found in {annotation_csv}")

    video_index = build_video_index(video_roots)
    summary = {
        "annotation_csv": str(annotation_csv),
        "video_roots": [str(root) for root in video_roots],
        "out_dir": str(out_dir),
        "annotation_frame_base": args.annotation_frame_base,
        "include_labels": args.include_labels,
        "exclude_labels": args.exclude_labels,
        "segments_before_filter": len(all_segments),
        "segments": len(segments),
        "status_counts": Counter(),
        "label_counts": Counter(),
        "missing_video_names": Counter(),
        "outputs": [],
    }

    for seg in segments:
        matches = video_index.get(seg.video_name, [])
        if not matches:
            summary["status_counts"]["missing_video"] += 1
            summary["missing_video_names"][seg.video_name] += 1
            continue

        clip_name = f"{seg.label}_f{seg.start_frame:05d}_{seg.end_frame:05d}.mp4"
        for root, video_path in matches:
            out_path = output_parent(out_dir, root, video_path) / clip_name
            status = cut_segment(video_path, out_path, seg.start_frame, seg.end_frame, args.overwrite)
            summary["status_counts"][status] += 1
            if status in {"written", "skip_exists"}:
                summary["label_counts"][seg.label] += 1
                summary["outputs"].append(
                    {
                        "row_idx": seg.row_idx,
                        "video_name": seg.video_name,
                        "source_video": str(video_path),
                        "output_video": str(out_path),
                        "label": seg.label,
                        "start_frame": seg.start_frame,
                        "end_frame": seg.end_frame,
                        "status": status,
                    }
                )

    summary["status_counts"] = dict(summary["status_counts"])
    summary["label_counts"] = dict(summary["label_counts"])
    summary["missing_video_names"] = dict(summary["missing_video_names"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "segment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({k: v for k, v in summary.items() if k != "outputs"}, ensure_ascii=False, indent=2))
    print(f"[summary] wrote {out_dir / 'segment_summary.json'}")


if __name__ == "__main__":
    main()
