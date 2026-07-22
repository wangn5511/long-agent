#!/usr/bin/env python3
"""Generate a canonical source-video split manifest for hand21 data prep."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


BASE_SCRIPT_PATH = Path(__file__).with_name("prepare_hand21_from_excel_segments.py")


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_hand21_from_excel_segments_base",
        BASE_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base script from {BASE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a canonical train/val(/test) split at the source-video level "
            "using valid annotation rows only."
        )
    )
    parser.add_argument("--video-root", type=Path, required=True, help="Root directory containing source mp4 files.")
    parser.add_argument("--annotation-file", type=Path, required=True, help="Excel/CSV file with segment annotations.")
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output manifest path, e.g. /path/to/video_split.json",
    )
    parser.add_argument("--sheet-name", type=str, default=None, help="Excel sheet name. Uses the first sheet by default.")
    parser.add_argument("--video-col", type=str, default="video_name", help="Column name for video path/name.")
    parser.add_argument("--label-col", type=str, default="label", help="Column name for gesture label.")
    parser.add_argument("--start-col", type=str, default="start_frame", help="Column name for start frame.")
    parser.add_argument("--end-col", type=str, default="end_frame", help="Column name for end frame.")
    parser.add_argument(
        "--annotation-format",
        choices=["auto", "table", "compact_4col"],
        default="auto",
        help="Annotation layout. Matches prepare_hand21_from_excel_segments.py.",
    )
    parser.add_argument(
        "--split-col",
        type=str,
        default=None,
        help="Optional existing split column. Used only when --split-source=metadata or auto.",
    )
    parser.add_argument(
        "--split-source",
        choices=["auto", "metadata", "grouped_video"],
        default="grouped_video",
        help=(
            "'grouped_video' randomly assigns valid source videos; "
            "'metadata' uses --split-col; 'auto' uses metadata only when every valid "
            "source video has a consistent split."
        ),
    )
    parser.add_argument(
        "--frame-base",
        type=int,
        choices=[0, 1],
        default=0,
        help="Whether start/end frame in the annotation file are 0-based or 1-based.",
    )
    parser.add_argument("--pre-pad", type=int, default=0, help="Frames to extend before the annotated segment.")
    parser.add_argument("--post-pad", type=int, default=0, help="Frames to extend after the annotated segment.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio at source-video level.")
    parser.add_argument("--test-ratio", type=float, default=0.0, help="Optional test ratio at source-video level.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for grouped_video splitting.")
    parser.add_argument(
        "--extra-video-root",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Optional extra roots of unannotated videos to include in the source-video split. "
            "Useful for long-video NONE pools that should be assigned to train/val "
            "at the source-video level."
        ),
    )
    parser.add_argument(
        "--train-video-list",
        type=Path,
        default=None,
        help="Optional output txt path for train source-video names. Defaults to <output-json dir>/train_video_names.txt",
    )
    parser.add_argument(
        "--val-video-list",
        type=Path,
        default=None,
        help="Optional output txt path for val source-video names. Defaults to <output-json dir>/val_video_names.txt",
    )
    parser.add_argument(
        "--test-video-list",
        type=Path,
        default=None,
        help="Optional output txt path for test source-video names. Defaults to <output-json dir>/test_video_names.txt",
    )
    return parser.parse_args()


def default_list_paths(output_json: Path) -> Tuple[Path, Path, Path]:
    parent = output_json.parent
    return (
        parent / "train_video_names.txt",
        parent / "val_video_names.txt",
        parent / "test_video_names.txt",
    )


def write_text_list(path: Path, values: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for value in values:
            f.write(f"{value}\n")


def collect_extra_videos(extra_roots: Sequence[Path]) -> Dict[str, dict]:
    extra_videos: Dict[str, dict] = {}
    for root in extra_roots:
        if not root.exists():
            raise FileNotFoundError(f"Extra video root does not exist: {root}")
        for video_path in sorted(root.rglob("*.mp4")):
            rel_stem = (Path(root.name) / video_path.relative_to(root).with_suffix("")).as_posix()
            if rel_stem in extra_videos:
                raise ValueError(
                    f"Duplicate extra video key {rel_stem!r} found under multiple --extra-video-root paths."
                )
            extra_videos[rel_stem] = {
                "source_video_path": str(video_path),
                "source_kind": "extra_video_pool",
                "source_group": root.name,
            }
    return extra_videos


def metadata_video_split(base_module, segments: Sequence[Dict[str, object]]) -> Dict[str, str]:
    video_map: Dict[str, str] = {}
    for segment in segments:
        source_video = str(segment["video_rel_stem"])
        split_value = segment.get("split")
        if split_value in (None, ""):
            raise ValueError(f"Valid segment from {source_video!r} is missing split metadata.")
        normalized = base_module.normalize_split_name(split_value)
        existing = video_map.get(source_video)
        if existing is None:
            video_map[source_video] = normalized
            continue
        if existing != normalized:
            raise ValueError(
                f"Source video {source_video!r} has inconsistent split metadata: "
                f"{existing!r} vs {normalized!r}"
            )
    return video_map


def build_video_split_map(
    base_module,
    args: argparse.Namespace,
    segments: Sequence[Dict[str, object]],
    extra_video_keys: Sequence[str],
) -> Tuple[Dict[str, str], str]:
    metadata_available = True
    try:
        metadata_map = metadata_video_split(base_module, segments)
    except ValueError:
        metadata_available = False
        metadata_map = {}

    split_source = args.split_source
    if split_source == "auto":
        split_source = "metadata" if metadata_available else "grouped_video"

    annotated_videos = sorted({str(segment["video_rel_stem"]) for segment in segments})
    extra_videos = sorted(extra_video_keys)

    if split_source == "metadata" and not metadata_available:
        raise ValueError(
            "Requested metadata split generation but split metadata is missing or inconsistent. "
            "Provide --split-col or use --split-source grouped_video."
        )

    if split_source == "metadata":
        annotated_split_map = metadata_map
    else:
        annotated_split_map = base_module.grouped_random_split(
            videos=annotated_videos,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    extra_split_map: Dict[str, str] = {}
    if extra_videos:
        # Keep NONE/extra-video pools independent so adding them does not reshuffle
        # the train/val assignment of annotated videos from --video-root.
        extra_split_map = base_module.grouped_random_split(
            videos=extra_videos,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    combined_map = dict(annotated_split_map)
    combined_map.update(extra_split_map)

    split_source_used = split_source
    if extra_videos:
        split_source_used = f"{split_source}+extra_grouped_video"
    return combined_map, split_source_used


def build_payload(
    args: argparse.Namespace,
    *,
    split_source_used: str,
    categories: Sequence[str],
    segments: Sequence[Dict[str, object]],
    missing_video_rows: Sequence[Dict[str, object]],
    invalid_frame_rows: Sequence[Dict[str, object]],
    video_split_map: Dict[str, str],
    extra_videos: Dict[str, dict],
) -> dict:
    video_entries: Dict[str, dict] = {}
    for segment in segments:
        source_video = str(segment["video_rel_stem"])
        entry = video_entries.setdefault(
            source_video,
            {
                "split": video_split_map[source_video],
                "source_video_path": str(segment["video_path"]),
                "num_segments": 0,
                "labels": Counter(),
                "row_indices": [],
                "source_kind": "annotated_segments",
                "source_group": None,
            },
        )
        entry["num_segments"] += 1
        entry["labels"][str(segment["label"])] += 1
        entry["row_indices"].append(int(segment["row_idx"]))

    for source_video, meta in extra_videos.items():
        video_entries.setdefault(
            source_video,
            {
                "split": video_split_map[source_video],
                "source_video_path": str(meta["source_video_path"]),
                "num_segments": 0,
                "labels": Counter(),
                "row_indices": [],
                "source_kind": str(meta["source_kind"]),
                "source_group": meta.get("source_group"),
            },
        )

    videos_payload: Dict[str, dict] = {}
    split_video_counts: Counter[str] = Counter()
    split_segment_counts: Counter[str] = Counter()
    split_label_counts: Dict[str, Counter[str]] = {"train": Counter(), "val": Counter(), "test": Counter()}
    for source_video in sorted(video_entries):
        entry = video_entries[source_video]
        split_name = str(entry["split"])
        split_video_counts[split_name] += 1
        split_segment_counts[split_name] += int(entry["num_segments"])
        split_label_counts.setdefault(split_name, Counter()).update(entry["labels"])
        videos_payload[source_video] = {
            "split": split_name,
            "source_video_path": entry["source_video_path"],
            "num_segments": int(entry["num_segments"]),
            "labels": dict(sorted(entry["labels"].items())),
            "row_indices": sorted(entry["row_indices"]),
            "source_kind": entry.get("source_kind", "annotated_segments"),
            "source_group": entry.get("source_group"),
        }

    return {
        "video_root": str(args.video_root),
        "annotation_file": str(args.annotation_file),
        "extra_video_roots": [str(path) for path in (args.extra_video_root or [])],
        "annotation_format": args.annotation_format,
        "split_source": split_source_used,
        "split_col": args.split_col,
        "frame_base": int(args.frame_base),
        "pre_pad": int(args.pre_pad),
        "post_pad": int(args.post_pad),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "seed": int(args.seed),
        "categories": list(categories),
        "num_valid_segments": int(len(segments)),
        "num_valid_source_videos": int(len(videos_payload)),
        "num_annotation_rows_missing_video": int(len(missing_video_rows)),
        "num_annotation_rows_invalid_frame": int(len(invalid_frame_rows)),
        "split_video_counts": dict(sorted(split_video_counts.items())),
        "split_segment_counts": dict(sorted(split_segment_counts.items())),
        "split_label_counts": {
            split_name: dict(sorted(counter.items()))
            for split_name, counter in sorted(split_label_counts.items())
            if counter
        },
        "missing_video_rows": list(missing_video_rows),
        "invalid_frame_rows": list(invalid_frame_rows),
        "videos": videos_payload,
    }


def main() -> None:
    args = parse_args()
    base = load_base_module()
    extra_video_roots = list(args.extra_video_root or [])

    base.log_status(
        f"[start] video_root={args.video_root} | annotation_file={args.annotation_file} | "
        f"output_json={args.output_json} | split_source={args.split_source} | "
        f"extra_video_roots={extra_video_roots or 'NONE'}"
    )
    video_index = base.build_video_index(args.video_root)
    base.log_status(f"[scan] Indexed {len(video_index)} video lookup keys under {args.video_root}")

    if args.annotation_format == "compact_4col":
        rows = base.load_compact_4col_rows(args.annotation_file, args.sheet_name)
    else:
        rows = base.load_tabular_rows(args.annotation_file, args.sheet_name)
        if args.annotation_format == "auto":
            required = {args.video_col, args.label_col, args.start_col, args.end_col}
            if args.split_col:
                required.add(args.split_col)
            header = set(rows[0].keys()) if rows else set()
            if not required.issubset(header):
                rows = base.load_compact_4col_rows(args.annotation_file, args.sheet_name)
                args.annotation_format = "compact_4col"
            else:
                args.annotation_format = "table"
    base.log_status(f"[load] Loaded {len(rows)} annotation rows with format={args.annotation_format}")

    # collect_segments expects args.categories (optional override); not used by this script
    if not hasattr(args, "categories"):
        args.categories = None
    segments, categories, missing_video_rows, invalid_frame_rows = base.collect_segments(args, rows, video_index)
    if not segments:
        raise FileNotFoundError("No valid segments remained after missing-video and invalid-frame filtering.")

    extra_videos = collect_extra_videos(extra_video_roots)
    if extra_videos:
        base.log_status(f"[extra] Loaded {len(extra_videos)} extra source videos from {extra_video_roots}")

    video_split_map, split_source_used = build_video_split_map(
        base,
        args,
        segments,
        sorted(extra_videos),
    )
    payload = build_payload(
        args,
        split_source_used=split_source_used,
        categories=categories,
        segments=segments,
        missing_video_rows=missing_video_rows,
        invalid_frame_rows=invalid_frame_rows,
        video_split_map=video_split_map,
        extra_videos=extra_videos,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    train_txt_default, val_txt_default, test_txt_default = default_list_paths(args.output_json)
    train_txt = args.train_video_list or train_txt_default
    val_txt = args.val_video_list or val_txt_default
    test_txt = args.test_video_list or test_txt_default
    train_videos = sorted(video for video, meta in payload["videos"].items() if meta["split"] == "train")
    val_videos = sorted(video for video, meta in payload["videos"].items() if meta["split"] == "val")
    test_videos = sorted(video for video, meta in payload["videos"].items() if meta["split"] == "test")
    write_text_list(train_txt, train_videos)
    write_text_list(val_txt, val_videos)
    write_text_list(test_txt, test_videos)

    base.log_status(f"[done] Saved video split manifest: {args.output_json}")
    base.log_status(
        f"[done] train_videos={len(train_videos)} val_videos={len(val_videos)} test_videos={len(test_videos)}"
    )
    base.log_status(f"[done] Saved train video list: {train_txt}")
    base.log_status(f"[done] Saved val video list: {val_txt}")
    if test_videos:
        base.log_status(f"[done] Saved test video list: {test_txt}")


if __name__ == "__main__":
    main()
