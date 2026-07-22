#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


DEFAULT_TIME_STEPS = 16
NUM_KEYPOINTS = 21
DIMENSION = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert JSON hand-keypoint dataset to fixed-length skeleton CSV files."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root containing train/ val directories and category_annotation.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model/skeleton_classifier"),
        help="Directory to write converted CSV files.",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=DEFAULT_TIME_STEPS,
        help="Number of frames kept for each sample after resampling.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum keypoint score required to keep a coordinate.",
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


def extract_category_id(json_path, sample, fallback_annotations):
    if "category_id" in sample:
        return sample["category_id"]

    info = sample.get("info", {})
    video_name = info.get("video_name")
    if video_name and video_name in fallback_annotations:
        return fallback_annotations[video_name]["category_id"]

    candidate_mp4 = f"{json_path.stem}.mp4"
    if candidate_mp4 in fallback_annotations:
        return fallback_annotations[candidate_mp4]["category_id"]

    raise KeyError(f"Unable to resolve category_id for {json_path}")


def extract_resolution(sample):
    info = sample.get("info", {})
    resolution = info.get("resolution", [1, 1])
    if len(resolution) != 2:
        return 1.0, 1.0
    return max(float(resolution[0]), 1.0), max(float(resolution[1]), 1.0)


def extract_frame_keypoints(annotation, min_score):
    keypoints = annotation.get("keypoints", [])
    frame_points = []

    for keypoint_index in range(NUM_KEYPOINTS):
        if keypoint_index >= len(keypoints):
            frame_points.append([0.0, 0.0])
            continue

        point = keypoints[keypoint_index]
        if len(point) < 2:
            frame_points.append([0.0, 0.0])
            continue

        score = point[2] if len(point) > 2 else 1.0
        if score < min_score:
            frame_points.append([0.0, 0.0])
            continue

        frame_points.append([float(point[0]), float(point[1])])

    return frame_points


def normalize_sequence_length(frames, target_length):
    if target_length <= 0:
        return []
    if not frames:
        return [[[0.0, 0.0] for _ in range(NUM_KEYPOINTS)] for _ in range(target_length)]
    if len(frames) == target_length:
        return [[point[:] for point in frame] for frame in frames]
    if target_length == 1:
        return [[point[:] for point in frames[-1]]]

    normalized_frames = []
    source_last_index = len(frames) - 1
    target_last_index = target_length - 1

    for target_index in range(target_length):
        source_index = round(target_index * source_last_index / target_last_index)
        normalized_frames.append([point[:] for point in frames[source_index]])

    return normalized_frames


def pre_process_skeleton(frames, image_width, image_height):
    temp_frames = [[point[:] for point in frame] for frame in frames]

    base_x, base_y = temp_frames[0][0]
    features = []

    for frame in temp_frames:
        for point in frame:
            point[0] = (point[0] - base_x) / image_width
            point[1] = (point[1] - base_y) / image_height
            features.extend(point[:DIMENSION])

    return features


def sample_to_row(json_path, sample, time_steps, min_score, fallback_annotations):
    category_id = extract_category_id(json_path, sample, fallback_annotations)
    image_width, image_height = extract_resolution(sample)
    annotations = sorted(sample.get("annotations", []), key=lambda item: item.get("frame_index", 0))
    frames = [extract_frame_keypoints(annotation, min_score) for annotation in annotations]
    normalized_frames = normalize_sequence_length(frames, time_steps)
    feature = pre_process_skeleton(normalized_frames, image_width, image_height)

    if not any(value != 0 for value in feature):
        return None

    return [category_id, *feature]


def write_split_csv(split_dir, output_path, time_steps, min_score, fallback_annotations):
    total_rows = 0
    total_files = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for json_path in iter_json_files(split_dir):
            with open(json_path, encoding="utf-8") as sample_file:
                sample = json.load(sample_file)

            row = sample_to_row(
                json_path=json_path,
                sample=sample,
                time_steps=time_steps,
                min_score=min_score,
                fallback_annotations=fallback_annotations,
            )
            if row is not None:
                writer.writerow(row)
                total_rows += 1
            total_files += 1

    return total_files, total_rows


def write_label_csv(categories, output_dir):
    label_path = output_dir / "skeleton_classifier_label.csv"
    with open(label_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for category in categories:
            writer.writerow([category])


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    categories, fallback_annotations = load_category_mapping(args.dataset_root)
    write_label_csv(categories, args.output_dir)

    for split_name in ("train", "val"):
        split_dir = args.dataset_root / split_name
        if not split_dir.is_dir():
            continue

        output_path = args.output_dir / f"skeleton_{split_name}.csv"
        total_files, total_rows = write_split_csv(
            split_dir=split_dir,
            output_path=output_path,
            time_steps=args.time_steps,
            min_score=args.min_score,
            fallback_annotations=fallback_annotations,
        )
        print(f"{split_name}: converted {total_files} json files into {total_rows} rows -> {output_path}")


if __name__ == "__main__":
    main()
