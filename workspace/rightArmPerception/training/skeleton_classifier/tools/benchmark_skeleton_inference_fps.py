#!/usr/bin/env python3
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf


TIME_STEPS = 16
NUM_KEYPOINTS = 21
DIMENSION = 2
FRAME_DIMENSION = NUM_KEYPOINTS * DIMENSION
INPUT_DIMENSION = TIME_STEPS * FRAME_DIMENSION


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark skeleton gesture model inference FPS using a JSON sample."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to a .keras/.h5 or .tflite model file.",
    )
    parser.add_argument(
        "--model-type",
        choices=("keras", "tflite"),
        required=True,
        help="Model runtime to benchmark.",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Input JSON sample path.",
    )
    parser.add_argument(
        "--label-csv",
        type=Path,
        default=Path("model/skeleton_classifier/skeleton_classifier_label.csv"),
        help="Optional label CSV path.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=200,
        help="Number of benchmark runs.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup runs before timing.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum keypoint score required to keep a coordinate.",
    )
    return parser.parse_args()


def load_labels(label_csv):
    if not label_csv.exists():
        return []
    with open(label_csv, encoding="utf-8-sig") as f:
        return [row[0] for row in csv.reader(f) if row]


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

    return np.asarray([features], dtype=np.float32)


def prepare_input(input_json, min_score):
    with open(input_json, encoding="utf-8") as f:
        sample = json.load(f)

    image_width, image_height = extract_resolution(sample)
    annotations = sorted(sample.get("annotations", []), key=lambda item: item.get("frame_index", 0))
    frames = [extract_frame_keypoints(annotation, min_score) for annotation in annotations]
    normalized_frames = normalize_sequence_length(frames, TIME_STEPS)
    features = pre_process_skeleton(normalized_frames, image_width, image_height)

    if features.shape != (1, INPUT_DIMENSION):
        raise ValueError(f"Unexpected input feature shape: {features.shape}")
    return features


def load_keras_model(model_path):
    model = tf.keras.models.load_model(str(model_path), compile=False)
    return model


def benchmark_keras(model, inputs, warmup, runs):
    for _ in range(warmup):
        _ = model(inputs, training=False).numpy()

    start = time.perf_counter()
    for _ in range(runs):
        output = model(inputs, training=False).numpy()
    elapsed = time.perf_counter() - start
    return output, elapsed


def load_tflite_interpreter(model_path):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def benchmark_tflite(interpreter, inputs, warmup, runs):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    for _ in range(warmup):
        interpreter.set_tensor(input_index, inputs)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_index)

    start = time.perf_counter()
    for _ in range(runs):
        interpreter.set_tensor(input_index, inputs)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)
    elapsed = time.perf_counter() - start
    return output, elapsed


def benchmark_end_to_end(args):
    if args.model_type == "keras":
        model = load_keras_model(args.model_path)
        start = time.perf_counter()
        for _ in range(args.runs):
            inputs = prepare_input(args.input_json, args.min_score)
            output = model(inputs, training=False).numpy()
        elapsed = time.perf_counter() - start
        return output, elapsed

    interpreter = load_tflite_interpreter(args.model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    start = time.perf_counter()
    for _ in range(args.runs):
        inputs = prepare_input(args.input_json, args.min_score)
        interpreter.set_tensor(input_index, inputs)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)
    elapsed = time.perf_counter() - start
    return output, elapsed


def decode_prediction(output, labels):
    probabilities = np.squeeze(output)
    class_id = int(np.argmax(probabilities))
    class_name = labels[class_id] if 0 <= class_id < len(labels) else str(class_id)
    return class_id, class_name, probabilities


def main():
    args = parse_args()
    labels = load_labels(args.label_csv)
    inputs = prepare_input(args.input_json, args.min_score)

    if args.model_type == "keras":
        model = load_keras_model(args.model_path)
        output, model_elapsed = benchmark_keras(model, inputs, args.warmup, args.runs)
    else:
        interpreter = load_tflite_interpreter(args.model_path)
        output, model_elapsed = benchmark_tflite(interpreter, inputs, args.warmup, args.runs)

    output_e2e, end_to_end_elapsed = benchmark_end_to_end(args)
    class_id, class_name, probabilities = decode_prediction(output_e2e, labels)

    avg_model_time_ms = (model_elapsed / args.runs) * 1000.0
    avg_e2e_time_ms = (end_to_end_elapsed / args.runs) * 1000.0
    model_fps = args.runs / model_elapsed if model_elapsed > 0 else float("inf")
    end_to_end_fps = args.runs / end_to_end_elapsed if end_to_end_elapsed > 0 else float("inf")

    print(f"model type: {args.model_type}")
    print(f"model path: {args.model_path}")
    print(f"input json: {args.input_json}")
    print(f"runs: {args.runs}")
    print(f"warmup: {args.warmup}")
    print(f"input shape: {inputs.shape}")
    print(f"predicted class id: {class_id}")
    print(f"predicted class name: {class_name}")
    print(f"probabilities: {probabilities}")
    print(f"avg model-only time: {avg_model_time_ms:.3f} ms")
    print(f"model-only fps: {model_fps:.3f}")
    print(f"avg end-to-end time: {avg_e2e_time_ms:.3f} ms")
    print(f"end-to-end fps: {end_to_end_fps:.3f}")


if __name__ == "__main__":
    main()
