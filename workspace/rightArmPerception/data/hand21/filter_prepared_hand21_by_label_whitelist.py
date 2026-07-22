#!/usr/bin/env python3
"""Filter prepared hand21 roots by a label whitelist.

This tool operates directly on prepared dataset roots such as:
- ``gesture_class9_prepared_train``
- ``gesture_class9_prepared_val``

For each prepared root it will:
1. Load ``category_annotation.json``.
2. Remove samples whose ``label`` is not in the whitelist.
3. Refresh per-video ``video_manifest.json`` files.
4. Refresh the root ``category_annotation.json``.
5. Optionally delete removed clip JSON / crop / kpts outputs.

It is intended for post-processing prepared outputs before running
``convert_hand21_json_to_pkl.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter prepared hand21 roots by a label whitelist."
    )
    parser.add_argument(
        "--prepared-roots",
        type=Path,
        nargs="+",
        required=True,
        help="Prepared dataset roots to filter.",
    )
    parser.add_argument(
        "--allowed-labels",
        nargs="+",
        required=True,
        help="Only samples with these labels will be kept.",
    )
    parser.add_argument(
        "--category-file",
        type=str,
        default="category_annotation.json",
        help="Root category annotation file name.",
    )
    parser.add_argument(
        "--cleanup-files",
        action="store_true",
        help="Delete removed sample JSON / crop / kpts outputs in addition to refreshing metadata.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned changes without writing files.",
    )
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp = Path(tmp_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def remove_path_if_exists(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def infer_kpts_root(prepared_root: Path) -> Path:
    return prepared_root.parent / f"{prepared_root.name}_video_kpts"


def infer_crop_root(prepared_root: Path) -> Path:
    return prepared_root.parent / f"{prepared_root.name}_segments_videos"


def segment_kpts_dir(kpts_root: Path, source_video: str, segment_stem: str) -> Path:
    return kpts_root / source_video / segment_stem


def segment_crop_path(crop_root: Path, source_video: str, segment_stem: str) -> Path:
    return crop_root / source_video / f"{segment_stem}.mp4"


def group_annotations_by_source_video(
    annotations: Dict[str, dict],
) -> "OrderedDict[str, OrderedDict[str, dict]]":
    grouped: "OrderedDict[str, OrderedDict[str, dict]]" = OrderedDict()
    for sample_name, meta in annotations.items():
        source_video = str(meta.get("source_video", "")).strip()
        grouped.setdefault(source_video, OrderedDict())[str(sample_name)] = dict(meta)
    return grouped


def kept_categories_in_order(
    original_categories: Iterable[str],
    kept_annotations: Dict[str, dict],
    allowed_labels: set[str],
) -> List[str]:
    present_labels = {
        str(meta.get("label", "")).strip()
        for meta in kept_annotations.values()
        if str(meta.get("label", "")).strip() in allowed_labels
    }
    ordered = [str(label) for label in original_categories if str(label) in present_labels]
    missing_from_original = sorted(present_labels - set(ordered))
    ordered.extend(missing_from_original)
    return ordered


def cleanup_removed_samples(
    *,
    prepared_root: Path,
    removed_annotations: Dict[str, dict],
) -> Dict[str, int]:
    kpts_root = infer_kpts_root(prepared_root)
    crop_root = infer_crop_root(prepared_root)
    removed_json = 0
    removed_kpts = 0
    removed_crop = 0
    for meta in removed_annotations.values():
        rel_json = str(meta.get("relative_json_path", "")).strip()
        source_video = str(meta.get("source_video", "")).strip()
        segment_stem = str(meta.get("segment_stem", "")).strip()
        if rel_json and remove_path_if_exists(prepared_root / rel_json):
            removed_json += 1
        if source_video and segment_stem and remove_path_if_exists(segment_kpts_dir(kpts_root, source_video, segment_stem)):
            removed_kpts += 1
        if source_video and segment_stem and remove_path_if_exists(segment_crop_path(crop_root, source_video, segment_stem)):
            removed_crop += 1
    return {
        "removed_clip_json": removed_json,
        "removed_kpts_dirs": removed_kpts,
        "removed_crop_videos": removed_crop,
    }


def cleanup_empty_source_dirs(prepared_root: Path, source_videos: Iterable[str]) -> Dict[str, int]:
    kpts_root = infer_kpts_root(prepared_root)
    crop_root = infer_crop_root(prepared_root)
    removed_prepared_dirs = 0
    removed_kpts_dirs = 0
    removed_crop_dirs = 0
    for source_video in source_videos:
        prepared_dir = prepared_root / source_video
        kpts_dir = kpts_root / source_video
        crop_dir = crop_root / source_video
        if prepared_dir.exists() and not any(prepared_dir.iterdir()):
            prepared_dir.rmdir()
            removed_prepared_dirs += 1
        if kpts_dir.exists() and not any(kpts_dir.iterdir()):
            kpts_dir.rmdir()
            removed_kpts_dirs += 1
        if crop_dir.exists() and not any(crop_dir.iterdir()):
            crop_dir.rmdir()
            removed_crop_dirs += 1
    return {
        "removed_empty_prepared_dirs": removed_prepared_dirs,
        "removed_empty_kpts_dirs": removed_kpts_dirs,
        "removed_empty_crop_dirs": removed_crop_dirs,
    }


def refresh_video_manifests(
    *,
    prepared_root: Path,
    kept_by_video: "OrderedDict[str, OrderedDict[str, dict]]",
    dry_run: bool,
) -> Tuple[int, int]:
    refreshed = 0
    removed = 0
    all_manifest_paths = sorted(prepared_root.rglob("video_manifest.json"))
    kept_video_set = set(kept_by_video.keys())
    for manifest_path in all_manifest_paths:
        rel_parent = manifest_path.parent.relative_to(prepared_root)
        source_video = rel_parent.as_posix()
        if source_video not in kept_video_set:
            if not dry_run:
                remove_path_if_exists(manifest_path)
            removed += 1
            continue
        manifest = load_json(manifest_path)
        clips = kept_by_video[source_video]
        manifest["clips"] = clips
        manifest["num_samples_written"] = len(clips)
        manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if not dry_run:
            atomic_write_json(manifest_path, manifest)
        refreshed += 1
    return refreshed, removed


def process_prepared_root(
    prepared_root: Path,
    *,
    category_file: str,
    allowed_labels: set[str],
    cleanup_files: bool,
    dry_run: bool,
) -> dict:
    category_path = prepared_root / category_file
    category_payload = load_json(category_path)
    original_categories = [str(item) for item in category_payload.get("categories", [])]
    annotations_raw = category_payload.get("annotations", {})
    annotations: "OrderedDict[str, dict]" = OrderedDict(
        (str(sample_name), dict(meta)) for sample_name, meta in annotations_raw.items()
    )

    kept_annotations: "OrderedDict[str, dict]" = OrderedDict()
    removed_annotations: "OrderedDict[str, dict]" = OrderedDict()
    removed_label_counts: Counter[str] = Counter()
    kept_label_counts: Counter[str] = Counter()

    for sample_name, meta in annotations.items():
        label = str(meta.get("label", "")).strip()
        if label in allowed_labels:
            kept_annotations[sample_name] = meta
            kept_label_counts[label] += 1
        else:
            removed_annotations[sample_name] = meta
            removed_label_counts[label] += 1

    kept_categories = kept_categories_in_order(
        original_categories,
        kept_annotations,
        allowed_labels,
    )
    kept_by_video = group_annotations_by_source_video(kept_annotations)

    removed_file_counts = {
        "removed_clip_json": 0,
        "removed_kpts_dirs": 0,
        "removed_crop_videos": 0,
    }
    removed_empty_dirs = {
        "removed_empty_prepared_dirs": 0,
        "removed_empty_kpts_dirs": 0,
        "removed_empty_crop_dirs": 0,
    }
    if cleanup_files and not dry_run:
        removed_file_counts = cleanup_removed_samples(
            prepared_root=prepared_root,
            removed_annotations=removed_annotations,
        )
        removed_empty_dirs = cleanup_empty_source_dirs(
            prepared_root,
            {str(meta.get("source_video", "")).strip() for meta in removed_annotations.values()},
        )

    manifests_refreshed, manifests_removed = refresh_video_manifests(
        prepared_root=prepared_root,
        kept_by_video=kept_by_video,
        dry_run=dry_run,
    )

    new_category_payload = {
        "categories": kept_categories,
        "annotations": kept_annotations,
    }
    if not dry_run:
        atomic_write_json(category_path, new_category_payload)

        summary_path = prepared_root / "prepare_summary.json"
        if summary_path.exists():
            summary = load_json(summary_path)
            summary["filter_label_whitelist"] = {
                "allowed_labels": sorted(allowed_labels),
                "removed_samples": len(removed_annotations),
                "removed_label_counts": dict(sorted(removed_label_counts.items())),
            }
            summary["num_existing_video_manifests_after_run"] = len(kept_by_video)
            summary["num_existing_samples_after_run"] = len(kept_annotations)
            summary["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            atomic_write_json(summary_path, summary)

    return {
        "prepared_root": str(prepared_root),
        "samples_before": len(annotations),
        "samples_kept": len(kept_annotations),
        "samples_removed": len(removed_annotations),
        "categories_before": original_categories,
        "categories_after": kept_categories,
        "kept_label_counts": dict(sorted(kept_label_counts.items())),
        "removed_label_counts": dict(sorted(removed_label_counts.items())),
        "manifests_refreshed": manifests_refreshed,
        "manifests_removed": manifests_removed,
        **removed_file_counts,
        **removed_empty_dirs,
    }


def main() -> None:
    args = parse_args()
    allowed_labels = {str(label).strip() for label in args.allowed_labels if str(label).strip()}
    if not allowed_labels:
        raise ValueError("No valid allowed labels were provided.")

    print(f"Allowed labels: {sorted(allowed_labels)}")
    for prepared_root in args.prepared_roots:
        report = process_prepared_root(
            prepared_root,
            category_file=args.category_file,
            allowed_labels=allowed_labels,
            cleanup_files=args.cleanup_files,
            dry_run=args.dry_run,
        )
        print("")
        print(f"Prepared root: {report['prepared_root']}")
        print(f"  samples_before: {report['samples_before']}")
        print(f"  samples_kept: {report['samples_kept']}")
        print(f"  samples_removed: {report['samples_removed']}")
        print(f"  categories_after: {report['categories_after']}")
        print(f"  removed_label_counts: {report['removed_label_counts']}")
        print(f"  manifests_refreshed: {report['manifests_refreshed']}")
        print(f"  manifests_removed: {report['manifests_removed']}")
        if args.cleanup_files:
            print(f"  removed_clip_json: {report['removed_clip_json']}")
            print(f"  removed_kpts_dirs: {report['removed_kpts_dirs']}")
            print(f"  removed_crop_videos: {report['removed_crop_videos']}")
            print(f"  removed_empty_prepared_dirs: {report['removed_empty_prepared_dirs']}")
            print(f"  removed_empty_kpts_dirs: {report['removed_empty_kpts_dirs']}")
            print(f"  removed_empty_crop_dirs: {report['removed_empty_crop_dirs']}")
    if args.dry_run:
        print("")
        print("Dry run only: no files were changed.")


if __name__ == "__main__":
    main()
