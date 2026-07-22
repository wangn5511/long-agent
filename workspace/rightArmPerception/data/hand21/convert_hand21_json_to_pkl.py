#!/usr/bin/env python3
import argparse
import json
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert hand21 json annotations to an MMAction2 PoseDataset pkl."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Legacy split dataset root containing flattened train/ and val/ clip json files. "
            "Use either this option or both --train-root and --val-root."
        ),
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=None,
        help="Prepared dataset root for the training split.",
    )
    parser.add_argument(
        "--val-root",
        type=Path,
        default=None,
        help="Prepared dataset root for the validation split.",
    )
    parser.add_argument(
        "--train-extra-roots",
        type=Path,
        nargs="*",
        default=None,
        help="Optional extra prepared roots to merge into the training split (for example mined NONE roots).",
    )
    parser.add_argument(
        "--val-extra-roots",
        type=Path,
        nargs="*",
        default=None,
        help="Optional extra prepared roots to merge into the validation split (for example mined NONE roots).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output pkl path.",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="train",
        help="Legacy subdirectory name for training samples under --dataset-root.",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="val",
        help="Legacy subdirectory name for validation samples under --dataset-root.",
    )
    parser.add_argument(
        "--category-file",
        type=str,
        default="category_annotation.json",
        help="Category annotation file name under each dataset root.",
    )
    args = parser.parse_args()
    if args.dataset_root is None and (args.train_root is None or args.val_root is None):
        parser.error("Provide either --dataset-root or both --train-root and --val-root.")
    return args


def load_category_payload(category_path: Path) -> Tuple[List[str], Dict[str, dict]]:
    with category_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    categories = [str(item) for item in data.get("categories", [])]
    annotations = data.get("annotations", {})
    return categories, {str(key): dict(value) for key, value in annotations.items()}


def resolve_label_name(meta: dict, categories: Sequence[str]) -> str:
    label = str(meta.get("label", "")).strip()
    if label:
        return label
    category_id = meta.get("category_id")
    if category_id is None:
        raise KeyError("Cannot resolve label name because both label and category_id are missing.")
    category_id = int(category_id)
    if category_id < 0 or category_id >= len(categories):
        raise IndexError(f"category_id={category_id} out of range for categories={categories}")
    return str(categories[category_id])


def build_sample(
    json_path: Path,
    *,
    sample_id: str,
    category_id: Optional[int],
) -> Dict:
    with json_path.open("r", encoding="utf-8") as f:
        sample = json.load(f)

    info = sample["info"]
    ann = sample["annotations"]

    video_name = info["video_name"]
    width, height = info["resolution"]
    total_frames = int(info["num_frame"])
    num_keypoints = int(info["num_keypoints"])

    if not ann:
        raise ValueError(f"No frame annotations found in {json_path}")

    # When merging multiple prepared roots, the caller may have already
    # remapped labels into a shared global category space. In that case we
    # must prefer the caller-provided category_id over the clip-local id
    # stored inside the json payload.
    sample_category_id = category_id if category_id is not None else sample.get("category_id")
    if sample_category_id is None:
        raise KeyError(f"Cannot resolve category_id for {sample_id} from {json_path}")

    keypoint = np.zeros((1, total_frames, num_keypoints, 2), dtype=np.float32)
    keypoint_score = np.zeros((1, total_frames, num_keypoints), dtype=np.float32)

    for frame_ann in ann:
        frame_index = int(frame_ann["frame_index"])
        if frame_index < 0 or frame_index >= total_frames:
            raise IndexError(f"Frame index {frame_index} out of range for {json_path}")

        points = frame_ann["keypoints"]
        if len(points) != num_keypoints:
            raise ValueError(
                f"Unexpected keypoint count in {json_path}: {len(points)} vs {num_keypoints}"
            )

        coords = np.asarray(points, dtype=np.float32)
        keypoint[0, frame_index] = coords[:, :2]
        keypoint_score[0, frame_index] = coords[:, 2]

    return dict(
        frame_dir=sample_id,
        label=int(sample_category_id),
        img_shape=(int(height), int(width)),
        original_shape=(int(height), int(width)),
        total_frames=total_frames,
        keypoint=keypoint,
        keypoint_score=keypoint_score,
        video_name=video_name,
    )


def collect_prepared_roots(prepared_roots: Iterable[Path], split_name: str, category_file: str) -> Tuple[List[dict], List[str]]:
    sample_specs: List[dict] = []
    categories_merged: List[str] = []
    seen_ids: set[str] = set()
    missing_json_paths: List[str] = []

    for prepared_root in prepared_roots:
        category_path = prepared_root / category_file
        categories, annotations = load_category_payload(category_path)
        for label in categories:
            if label not in categories_merged:
                categories_merged.append(label)

        root_collected = 0
        for sample_name, meta in annotations.items():
            rel_json = str(meta.get("relative_json_path", "")).strip()
            if not rel_json:
                raise ValueError(f"Sample {sample_name!r} is missing relative_json_path in {category_path}")
            json_path = prepared_root / rel_json
            if not json_path.exists():
                missing_json_paths.append(str(json_path))
                continue
            base_id = str(sample_name).replace("/", "__")
            sample_id = base_id
            if sample_id in seen_ids:
                sample_id = f"{prepared_root.name}__{base_id}"
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample_id remained after root prefixing: {sample_id}")
            seen_ids.add(sample_id)
            sample_specs.append(
                {
                    "sample_id": sample_id,
                    "json_path": json_path,
                    "label_name": resolve_label_name(meta, categories),
                }
            )
            root_collected += 1
        print(f'Collected {root_collected} samples for split "{split_name}" from prepared root {prepared_root}')

    if missing_json_paths:
        preview = ", ".join(missing_json_paths[:5])
        suffix = ", ..." if len(missing_json_paths) > 5 else ""
        raise FileNotFoundError(
            f"Missing clip json files under prepared roots: {preview}{suffix}"
        )

    return sample_specs, categories_merged


def collect_flat_split(split_dir: Path, split_name: str, category_file_path: Path) -> Tuple[List[str], List[Dict], List[str]]:
    categories, annotations = load_category_payload(category_file_path)
    label_map = {str(key): int(meta["category_id"]) for key, meta in annotations.items()}
    ids: List[str] = []
    samples: List[Dict] = []
    for json_path in sorted(split_dir.glob("*.json")):
        sample_id = json_path.stem
        category_id = label_map.get(sample_id)
        sample = build_sample(json_path, sample_id=sample_id, category_id=category_id)
        ids.append(sample["frame_dir"])
        samples.append(sample)
    print(f'Collected {len(ids)} samples for split "{split_name}" from {split_dir}')
    return ids, samples, categories


def ensure_same_categories(train_categories: Iterable[str], val_categories: Iterable[str]) -> List[str]:
    train_list = [str(item) for item in train_categories]
    val_list = [str(item) for item in val_categories]
    if train_list != val_list:
        raise ValueError(
            "Train/val category order differs. "
            f"train={train_list}, val={val_list}"
        )
    return train_list


def merge_category_order(train_categories: Iterable[str], val_categories: Iterable[str]) -> List[str]:
    merged: List[str] = []
    for label in list(train_categories) + list(val_categories):
        label = str(label)
        if label not in merged:
            merged.append(label)
    return merged


def main():
    args = parse_args()

    if args.dataset_root is not None:
        dataset_root = args.dataset_root
        train_dir = dataset_root / args.train_dir
        val_dir = dataset_root / args.val_dir
        category_path = dataset_root / args.category_file
        train_ids, train_ann, train_categories = collect_flat_split(train_dir, "train", category_path)
        val_ids, val_ann, val_categories = collect_flat_split(val_dir, "val", category_path)
        categories = ensure_same_categories(train_categories, val_categories)
    else:
        train_roots = [args.train_root] + list(args.train_extra_roots or [])
        val_roots = [args.val_root] + list(args.val_extra_roots or [])
        train_specs, train_categories = collect_prepared_roots(train_roots, "train", args.category_file)
        val_specs, val_categories = collect_prepared_roots(val_roots, "val", args.category_file)
        categories = merge_category_order(train_categories, val_categories)
        category_to_id = {label: idx for idx, label in enumerate(categories)}
        train_ids = [spec["sample_id"] for spec in train_specs]
        val_ids = [spec["sample_id"] for spec in val_specs]
        train_ann = [
            build_sample(
                spec["json_path"],
                sample_id=spec["sample_id"],
                category_id=category_to_id[spec["label_name"]],
            )
            for spec in train_specs
        ]
        val_ann = [
            build_sample(
                spec["json_path"],
                sample_id=spec["sample_id"],
                category_id=category_to_id[spec["label_name"]],
            )
            for spec in val_specs
        ]

    output = dict(
        split=dict(train=train_ids, val=val_ids),
        annotations=train_ann + val_ann,
        categories=categories,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved annotation pkl to {args.output}")
    print(f"Categories: {categories}")


if __name__ == "__main__":
    main()
