#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count original JSON samples per class in train/val splits."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/datasets/cybopal_dataset/final_gesture_stgcn_v1"),
        help="Dataset root containing train/ val directories and category_annotation.json",
    )
    return parser.parse_args()


def load_category_mapping(dataset_root):
    annotation_path = dataset_root / "category_annotation.json"
    with open(annotation_path, encoding="utf-8") as f:
        payload = json.load(f)

    categories = payload.get("categories", [])
    annotations = payload.get("annotations", {})
    return categories, annotations


def iter_json_files(split_dir):
    return sorted(path for path in split_dir.glob("*.json") if path.is_file())


def resolve_category_id(json_path, sample, fallback_annotations):
    if "category_id" in sample:
        return sample["category_id"]

    info = sample.get("info", {})
    video_name = info.get("video_name")
    if video_name and video_name in fallback_annotations:
        return fallback_annotations[video_name]["category_id"]

    json_stem = json_path.stem
    candidate_mp4 = f"{json_stem}.mp4"
    if candidate_mp4 in fallback_annotations:
        return fallback_annotations[candidate_mp4]["category_id"]

    raise KeyError(f"Unable to resolve category_id for {json_path}")


def count_split(split_dir, fallback_annotations):
    counter = Counter()
    total = 0

    for json_path in iter_json_files(split_dir):
        with open(json_path, encoding="utf-8") as f:
            sample = json.load(f)

        category_id = resolve_category_id(json_path, sample, fallback_annotations)
        counter[category_id] += 1
        total += 1

    return counter, total


def print_stats(split_name, counter, total, categories):
    print(f"[{split_name}] total json samples: {total}")
    for class_id in sorted(counter):
        label = categories[class_id] if 0 <= class_id < len(categories) else f"class_{class_id}"
        print(f"  {class_id}: {label} -> {counter[class_id]}")


def main():
    args = parse_args()
    categories, fallback_annotations = load_category_mapping(args.dataset_root)

    for split_name in ("train", "val"):
        split_dir = args.dataset_root / split_name
        if not split_dir.is_dir():
            continue

        counter, total = count_split(split_dir, fallback_annotations)
        print_stats(split_name, counter, total, categories)
        print()


if __name__ == "__main__":
    main()
