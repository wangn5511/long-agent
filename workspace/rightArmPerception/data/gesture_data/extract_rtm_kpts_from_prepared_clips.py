#!/usr/bin/env python3
"""Extract RTM hand and upperbody keypoints for prepared gesture clips.

Supports two input layouts:
1. A directory tree of clip videos: `<clips_root>/**/*.mp4`
2. A prepared dataset tree that contains clip metadata jsons, e.g.
   `/home/results/.../gesture_class8_prepared/**/clips/*.json`

For metadata json input, the script resolves the source video path from the
json payload when possible, otherwise falls back to:
  <recording_dir>/<clip_parent_name>.mp4

Output format follows the existing hand `annotations.json` schema used in this
repo so downstream code can keep working unchanged.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
try:
    import onnxruntime as ort
except ImportError:
    ort = None


HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
ARM_ONLY_IDXS = [5, 6, 7, 8, 9, 10]
ARM_ONLY_SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
]
ARM_HAND27_SKELETON = HAND_SKELETON + [
    (u + 21, v + 21) for u, v in ARM_ONLY_SKELETON
]
LEFT_HAND_LABEL = 0
RIGHT_HAND_LABEL = 1
UNKNOWN_HAND_LABEL = 2
LEFT_SHOULDER_ARM_IDX = 0
RIGHT_SHOULDER_ARM_IDX = 1
LEFT_ELBOW_ARM_IDX = 2
RIGHT_ELBOW_ARM_IDX = 3
LEFT_WRIST_ARM_IDX = 4
RIGHT_WRIST_ARM_IDX = 5
CLIP_RANGE_RE = re.compile(r"_f(\d+)_(\d+)$")
DEBUG_DIR_RE = re.compile(r"debug_(\d{8})_(\d{6})$")
DATASET_DATE_RE = re.compile(r"dynamic_gestures_(\d{6})$")

MODEL_ROOT = Path(__file__).resolve().parent / "models"
DEFAULT_DET_ONNX = str(MODEL_ROOT / "rtmdet_personhand_320_192_v1.onnx")
DEFAULT_PERSON_POSE_ONNX = str(MODEL_ROOT / "rtmpose3d-s-upperbody-face.onnx")
DEFAULT_HAND_POSE_ONNX = str(MODEL_ROOT / "rtmpose3d-s-hand-v2.onnx")

DET_CLASS_PERSON = 0
DET_CLASS_HAND = 1
DET_NUM_CLASSES = 2
DET_STRIDES = (8, 16, 32)
DET_PAD_VALUE = 114
DET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
DET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

POSE_INPUT_SIZE = (256, 256)
POSE_SIMCC_SPLIT_RATIO = 2.0
POSE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
POSE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
POSE_PADDING = 1.5
UPPERBODY_CROP_REGION = (0.0, 0.0, 1.0, 0.75)
HAND_INPUT_SIZE_Z = 256
HAND_Z_RANGE = 1
HAND_ROOT_Z = 0

HAND_PALETTE = np.array([
    [255, 255, 255],
    [0, 0, 255],
    [0, 128, 255],
    [0, 255, 255],
    [0, 255, 0],
    [255, 0, 0],
], dtype=np.uint8)
HAND_LINK_COLORS = HAND_PALETTE[np.array([
    1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    4, 4, 4, 4,
    5, 5, 5, 5,
], dtype=np.int32)]
HAND_POINT_COLORS = HAND_PALETTE[np.array([
    0,
    1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    4, 4, 4, 4,
    5, 5, 5, 5,
], dtype=np.int32)]
DEMO_HAND_SKELETON = tuple(HAND_SKELETON)

ARGS = None
WORKER_ARGS = None
WORKER_DET_SESS = None
WORKER_PERSON_SPEC = None
WORKER_HAND_SPEC = None


@dataclass
class PosePreprocessConfig:
    input_size: Tuple[int, int]
    mean: np.ndarray
    std: np.ndarray
    bgr_to_rgb: bool = True
    crop_region: Optional[Tuple[float, float, float, float]] = None
    padding: float = POSE_PADDING


@dataclass
class PoseDrawConfig:
    skeleton: Tuple[Tuple[int, int], ...]
    link_colors: np.ndarray
    point_colors: np.ndarray


@dataclass
class PoseResult:
    keypoints: np.ndarray
    keypoint_scores: np.ndarray
    keypoints_2d: np.ndarray
    keypoints_bbox_norm: np.ndarray
    bboxes: np.ndarray
    input_bboxes: np.ndarray
    input_bbox_scores: np.ndarray


@dataclass
class PoseModelSpec:
    session: "ort.InferenceSession"
    preprocess_cfg: PosePreprocessConfig
    draw_cfg: PoseDrawConfig
    num_keypoints: int
    input_size_z: int
    z_range: float
    root_z: float
    z_output_mode: str


@dataclass
class FrameResult:
    det_boxes: np.ndarray
    det_scores: np.ndarray
    det_labels: np.ndarray
    person_result: PoseResult
    hand_result: PoseResult


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def build_session(onnx_path: str, device: str = "cpu") -> "ort.InferenceSession":
    if ort is None:
        raise ImportError("onnxruntime is required. Install onnxruntime or onnxruntime-gpu first.")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device != "cpu" else ["CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, providers=providers)


def _rotate_point(point: np.ndarray, angle_rad: float) -> np.ndarray:
    sin_v, cos_v = np.sin(angle_rad), np.cos(angle_rad)
    return np.array(
        [point[0] * cos_v - point[1] * sin_v, point[0] * sin_v + point[1] * cos_v],
        dtype=np.float32,
    )


def _get_3rd_point(point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray:
    direction = point_a - point_b
    return point_b + np.array([-direction[1], direction[0]], dtype=np.float32)


def get_warp_matrix(center: np.ndarray, scale: np.ndarray, rot: float, output_size: Tuple[int, int]) -> np.ndarray:
    src_w = scale[0]
    dst_w, dst_h = output_size
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0.0, src_w * -0.5], dtype=np.float32), rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5], dtype=np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + src_dir
    src[2] = _get_3rd_point(src[0], src[1])
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0] = [dst_w * 0.5, dst_h * 0.5]
    dst[1] = dst[0] + dst_dir
    dst[2] = _get_3rd_point(dst[0], dst[1])
    return cv2.getAffineTransform(src, dst)


def bbox_xyxy2cs(bbox: np.ndarray, padding: float) -> Tuple[np.ndarray, np.ndarray]:
    bbox = np.asarray(bbox, dtype=np.float32).reshape(-1, 4)
    center = np.stack([(bbox[:, 0] + bbox[:, 2]) * 0.5, (bbox[:, 1] + bbox[:, 3]) * 0.5], axis=-1)
    scale = np.stack([(bbox[:, 2] - bbox[:, 0]) * padding, (bbox[:, 3] - bbox[:, 1]) * padding], axis=-1)
    return center, scale


def fix_aspect_ratio(scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    width = scale[..., 0:1]
    height = scale[..., 1:2]
    return np.where(
        width > height * aspect_ratio,
        np.concatenate([width, width / aspect_ratio], axis=-1),
        np.concatenate([height * aspect_ratio, height], axis=-1),
    )


def crop_bbox_region(bbox: np.ndarray, region: Tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    width = x2 - x1
    height = y2 - y1
    rx1, ry1, rx2, ry2 = region
    return np.array([x1 + width * rx1, y1 + height * ry1, x1 + width * rx2, y1 + height * ry2], dtype=np.float32)


def center_scale_to_bbox(center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.array([
        center[0] - scale[0] * 0.5,
        center[1] - scale[1] * 0.5,
        center[0] + scale[0] * 0.5,
        center[1] + scale[1] * 0.5,
    ], dtype=np.float32)


def preprocess_pose_bbox(image_bgr: np.ndarray, bbox: np.ndarray, cfg: PosePreprocessConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    work_bbox = np.asarray(bbox, dtype=np.float32)
    if cfg.crop_region is not None:
        work_bbox = crop_bbox_region(work_bbox, cfg.crop_region)
    center, scale = bbox_xyxy2cs(work_bbox, padding=cfg.padding)
    center = center[0]
    scale = fix_aspect_ratio(scale[0], cfg.input_size[0] / cfg.input_size[1])
    warp_mat = get_warp_matrix(center, scale, 0.0, cfg.input_size)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) if cfg.bgr_to_rgb else image_bgr.copy()
    resized = cv2.warpAffine(image, warp_mat, cfg.input_size, flags=cv2.INTER_LINEAR)
    resized = (resized.astype(np.float32) - cfg.mean) / cfg.std
    return resized, center, scale, center_scale_to_bbox(center, scale)


def keypoints_to_original(keypoints_2d: np.ndarray, model_input_size: Tuple[int, int], center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    width, height = model_input_size
    keypoints = keypoints_2d.copy()
    keypoints[..., 0] = keypoints[..., 0] / width * scale[0] + center[0] - scale[0] / 2.0
    keypoints[..., 1] = keypoints[..., 1] / height * scale[1] + center[1] - scale[1] / 2.0
    return keypoints


def get_simcc_maximum_2d(simcc_x: np.ndarray, simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    batch_size, num_keypoints, _ = simcc_x.shape
    simcc_x = simcc_x.reshape(batch_size * num_keypoints, -1)
    simcc_y = simcc_y.reshape(batch_size * num_keypoints, -1)
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack([x_locs, y_locs], axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)
    max_val_x[max_val_x > max_val_y] = max_val_y[max_val_x > max_val_y]
    locs[max_val_x <= 0.0] = -1
    return locs.reshape(batch_size, num_keypoints, 2), max_val_x.reshape(batch_size, num_keypoints)


def get_simcc_maximum_3d(simcc_x: np.ndarray, simcc_y: np.ndarray, simcc_z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    batch_size, num_keypoints, _ = simcc_x.shape
    simcc_x = simcc_x.reshape(batch_size * num_keypoints, -1)
    simcc_y = simcc_y.reshape(batch_size * num_keypoints, -1)
    simcc_z = simcc_z.reshape(batch_size * num_keypoints, -1)
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    z_locs = np.argmax(simcc_z, axis=1)
    locs = np.stack([x_locs, y_locs, z_locs], axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)
    max_val_x[max_val_x > max_val_y] = max_val_y[max_val_x > max_val_y]
    locs[max_val_x <= 0.0] = -1
    return locs.reshape(batch_size, num_keypoints, 3), max_val_x.reshape(batch_size, num_keypoints)


def decode_simcc_2d(simcc_x: np.ndarray, simcc_y: np.ndarray, split_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    keypoints, scores = get_simcc_maximum_2d(simcc_x, simcc_y)
    return keypoints / split_ratio, scores


def decode_simcc_3d_raw(simcc_x: np.ndarray, simcc_y: np.ndarray, simcc_z: np.ndarray, split_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    keypoints, scores = get_simcc_maximum_3d(simcc_x, simcc_y, simcc_z)
    return keypoints / split_ratio, scores


def run_pose_model(sess: "ort.InferenceSession", image: np.ndarray) -> List[np.ndarray]:
    blob = image.transpose(2, 0, 1)[None].astype(np.float32)
    input_name = sess.get_inputs()[0].name
    output_names = [output.name for output in sess.get_outputs()]
    return sess.run(output_names, {input_name: blob})


def empty_pose_result(num_keypoints: int) -> PoseResult:
    return PoseResult(
        keypoints=np.zeros((0, num_keypoints, 3), dtype=np.float32),
        keypoint_scores=np.zeros((0, num_keypoints), dtype=np.float32),
        keypoints_2d=np.zeros((0, num_keypoints, 2), dtype=np.float32),
        keypoints_bbox_norm=np.zeros((0, num_keypoints, 3), dtype=np.float32),
        bboxes=np.zeros((0, 4), dtype=np.float32),
        input_bboxes=np.zeros((0, 4), dtype=np.float32),
        input_bbox_scores=np.zeros((0,), dtype=np.float32),
    )


def normalize_keypoints_to_bboxes(keypoints_2d: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    if len(keypoints_2d) == 0:
        return np.zeros((0, 0, 2), dtype=np.float32)
    bboxes = np.asarray(bboxes, dtype=np.float32)
    keypoints_2d = np.asarray(keypoints_2d, dtype=np.float32)
    widths = np.maximum(bboxes[:, 2] - bboxes[:, 0], 1e-6)[:, None, None]
    heights = np.maximum(bboxes[:, 3] - bboxes[:, 1], 1e-6)[:, None, None]
    norm_x = (keypoints_2d[..., 0:1] - bboxes[:, None, 0:1]) / widths
    norm_y = (keypoints_2d[..., 1:2] - bboxes[:, None, 1:2]) / heights
    return np.concatenate([norm_x, norm_y], axis=-1).astype(np.float32)


def infer_pose_instances(frame: np.ndarray, boxes: np.ndarray, spec: PoseModelSpec, box_scores: Optional[np.ndarray] = None) -> PoseResult:
    if len(boxes) == 0:
        return empty_pose_result(spec.num_keypoints)
    if box_scores is None:
        box_scores = np.ones((len(boxes),), dtype=np.float32)
    else:
        box_scores = np.asarray(box_scores, dtype=np.float32)
    all_keypoints = []
    all_scores = []
    all_keypoints_2d = []
    all_keypoints_bbox_norm = []
    all_bboxes = []
    all_input_bboxes = []
    for bbox in np.asarray(boxes, dtype=np.float32):
        image, center, scale, vis_bbox = preprocess_pose_bbox(frame, bbox, spec.preprocess_cfg)
        outputs = run_pose_model(spec.session, image)
        if len(outputs) >= 3:
            keypoints_input, scores = decode_simcc_3d_raw(outputs[0], outputs[1], outputs[2], split_ratio=POSE_SIMCC_SPLIT_RATIO)
            keypoints_2d = keypoints_to_original(keypoints_input[..., :2], spec.preprocess_cfg.input_size, center, scale)
            bbox_norm_xy = normalize_keypoints_to_bboxes(keypoints_2d, np.asarray(bbox, dtype=np.float32)[None])
            if spec.z_output_mode == "body":
                keypoints_z = (
                    keypoints_input[..., 2:3] / (spec.input_size_z / 2.0) - 1.0
                ) * spec.z_range + spec.root_z
                bbox_norm_z = keypoints_z.copy()
            elif spec.z_output_mode == "x_like":
                bbox_width = max(float(bbox[2] - bbox[0]), 1e-6)
                bbox_norm_z = (
                    keypoints_input[..., 2:3] / spec.preprocess_cfg.input_size[0] - 0.5
                ).astype(np.float32)
                keypoints_z = (bbox_norm_z * bbox_width).astype(np.float32)
            else:
                raise ValueError(f"Unsupported z_output_mode: {spec.z_output_mode}")
            keypoints = np.concatenate([keypoints_2d, keypoints_z], axis=-1)
            keypoints_bbox_norm = np.concatenate([bbox_norm_xy, bbox_norm_z], axis=-1)
        else:
            keypoints_input, scores = decode_simcc_2d(outputs[0], outputs[1], split_ratio=POSE_SIMCC_SPLIT_RATIO)
            keypoints_2d = keypoints_to_original(keypoints_input, spec.preprocess_cfg.input_size, center, scale)
            keypoints = np.concatenate([keypoints_2d, np.zeros_like(keypoints_2d[..., :1])], axis=-1)
            keypoints_bbox_norm = np.concatenate([
                normalize_keypoints_to_bboxes(keypoints_2d, np.asarray(bbox, dtype=np.float32)[None]),
                np.zeros_like(keypoints_2d[..., :1]),
            ], axis=-1)
        all_keypoints.append(keypoints.astype(np.float32))
        all_scores.append(scores.astype(np.float32))
        all_keypoints_2d.append(keypoints_2d.astype(np.float32))
        all_keypoints_bbox_norm.append(keypoints_bbox_norm.astype(np.float32))
        all_bboxes.append(vis_bbox[None].astype(np.float32))
        all_input_bboxes.append(np.asarray(bbox, dtype=np.float32)[None])
    return PoseResult(
        keypoints=np.concatenate(all_keypoints, axis=0),
        keypoint_scores=np.concatenate(all_scores, axis=0),
        keypoints_2d=np.concatenate(all_keypoints_2d, axis=0),
        keypoints_bbox_norm=np.concatenate(all_keypoints_bbox_norm, axis=0),
        bboxes=np.concatenate(all_bboxes, axis=0),
        input_bboxes=np.concatenate(all_input_bboxes, axis=0),
        input_bbox_scores=box_scores.copy(),
    )


def letterbox_rgb(image_bgr: np.ndarray, target_h: int, target_w: int, pad_value: int = DET_PAD_VALUE) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    src_h, src_w = image_bgr.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    resized_w = int(round(src_w * scale))
    resized_h = int(round(src_h * scale))
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    pad_left = (target_w - resized_w) // 2
    pad_top = (target_h - resized_h) // 2
    canvas[pad_top:pad_top + resized_h, pad_left:pad_left + resized_w] = resized
    return canvas, scale, (pad_left, pad_top)


def preprocess_detector(image_bgr: np.ndarray, target_h: int, target_w: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    letterboxed, scale, pads = letterbox_rgb(image_bgr, target_h, target_w)
    chw = np.transpose(letterboxed.astype(np.float32), (2, 0, 1))
    chw = (chw - DET_MEAN[:, None, None]) / DET_STD[:, None, None]
    return np.expand_dims(chw, axis=0), scale, pads


def decode_rtmdet_features(outputs: List[np.ndarray], score_thr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_boxes = []
    all_scores = []
    all_labels = []
    for stride, feat in zip(DET_STRIDES, outputs):
        feat = np.ascontiguousarray(feat[0].transpose(1, 2, 0))
        cls_logits = feat[..., :DET_NUM_CLASSES]
        box_deltas = feat[..., DET_NUM_CLASSES:]
        cls_scores = sigmoid(cls_logits)
        labels = cls_scores.argmax(axis=-1)
        scores = cls_scores.max(axis=-1)
        ys, xs = np.where(scores > score_thr)
        if ys.size == 0:
            continue
        chosen_scores = scores[ys, xs].astype(np.float32)
        chosen_labels = labels[ys, xs].astype(np.int32)
        chosen_boxes = box_deltas[ys, xs].astype(np.float32)
        x1 = (xs.astype(np.float32) - chosen_boxes[:, 0]) * stride
        y1 = (ys.astype(np.float32) - chosen_boxes[:, 1]) * stride
        x2 = (xs.astype(np.float32) + chosen_boxes[:, 2]) * stride
        y2 = (ys.astype(np.float32) + chosen_boxes[:, 3]) * stride
        all_boxes.append(np.stack([x1, y1, x2, y2], axis=1))
        all_scores.append(chosen_scores)
        all_labels.append(chosen_labels)
    if not all_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.concatenate(all_boxes, axis=0), np.concatenate(all_scores, axis=0), np.concatenate(all_labels, axis=0)


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter_w = np.maximum(0.0, xx2 - xx1)
    inter_h = np.maximum(0.0, yy2 - yy1)
    inter = inter_w * inter_h
    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = np.maximum(area1 + area2 - inter, 1e-6)
    return inter / union


def multiclass_nms(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_thr: float, max_det: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    kept_boxes = []
    kept_scores = []
    kept_labels = []
    for class_id in np.unique(labels):
        class_mask = labels == class_id
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        order = class_scores.argsort()[::-1]
        while order.size > 0:
            idx = order[0]
            kept_boxes.append(class_boxes[idx])
            kept_scores.append(class_scores[idx])
            kept_labels.append(int(class_id))
            if order.size == 1:
                break
            rest = order[1:]
            ious = compute_iou(class_boxes[idx], class_boxes[rest])
            order = rest[ious <= iou_thr]
    kept_boxes = np.asarray(kept_boxes, dtype=np.float32)
    kept_scores = np.asarray(kept_scores, dtype=np.float32)
    kept_labels = np.asarray(kept_labels, dtype=np.int32)
    order = kept_scores.argsort()[::-1]
    if max_det > 0:
        order = order[:max_det]
    return kept_boxes[order], kept_scores[order], kept_labels[order]


def scale_boxes(boxes: np.ndarray, scale: float, pads: Tuple[int, int], image_shape: Tuple[int, int]) -> np.ndarray:
    pad_left, pad_top = pads
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    boxes /= scale
    height, width = image_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
    return boxes


def run_detector_full(det_sess: "ort.InferenceSession", frame: np.ndarray, score_thr: float, iou_thr: float, max_det: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_shape = det_sess.get_inputs()[0].shape
    input_h = int(input_shape[2])
    input_w = int(input_shape[3])
    input_name = det_sess.get_inputs()[0].name
    tensor, scale, pads = preprocess_detector(frame, input_h, input_w)
    outputs = det_sess.run(None, {input_name: tensor})
    boxes, scores, labels = decode_rtmdet_features(outputs, score_thr)
    boxes, scores, labels = multiclass_nms(boxes, scores, labels, iou_thr, max_det)
    if len(boxes) > 0:
        boxes = scale_boxes(boxes, scale, pads, frame.shape[:2])
    return boxes.astype(np.float32), scores.astype(np.float32), labels.astype(np.int32)


def infer_frame(det_sess: "ort.InferenceSession", person_spec: PoseModelSpec, hand_spec: PoseModelSpec, frame: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, FrameResult]:
    boxes, scores, labels = run_detector_full(det_sess, frame, score_thr=args.det_score_thr, iou_thr=args.det_iou_thr, max_det=args.max_det)
    person_boxes = boxes[labels == DET_CLASS_PERSON]
    hand_boxes = boxes[labels == DET_CLASS_HAND]
    if args.det_only:
        frame_result = FrameResult(
            det_boxes=boxes,
            det_scores=scores,
            det_labels=labels,
            person_result=empty_pose_result(person_spec.num_keypoints),
            hand_result=empty_pose_result(hand_spec.num_keypoints),
        )
        return frame.copy(), frame_result
    person_result = infer_pose_instances(frame, person_boxes, person_spec, scores[labels == DET_CLASS_PERSON])
    hand_result = infer_pose_instances(frame, hand_boxes, hand_spec, scores[labels == DET_CLASS_HAND])
    return frame.copy(), FrameResult(
        det_boxes=boxes,
        det_scores=scores,
        det_labels=labels,
        person_result=person_result,
        hand_result=hand_result,
    )


@dataclass(frozen=True)
class ClipTask:
    clip_id: str
    clip_path: Optional[Path]
    meta_json_path: Optional[Path]
    source_video_path: Path
    start_frame: int
    end_frame: int
    output_name: str


def _camera_alias(path: Path) -> str:
    text = str(path)
    if "camera_base" in text:
        return "base"
    if "camera_top" in text:
        return "top"
    return "cam"


def _short_clip_dirname(clip_id: str, ref_path: Path) -> str:
    gesture = clip_id.split("_f", 1)[0]
    date_code = "unknown"
    time_code = "unknown"
    camera = _camera_alias(ref_path)
    for part in ref_path.parts:
        date_match = DATASET_DATE_RE.fullmatch(part)
        if date_match and date_code == "unknown":
            date_code = f"20{date_match.group(1)}"
        match = DEBUG_DIR_RE.fullmatch(part)
        if match:
            date_code = match.group(1)
            time_code = match.group(2)

    return f"{date_code}_{camera}_{time_code}_{clip_id}"


def _output_subdir(dst_dir: Path, clip_id: str, ref_path: Path) -> Path:
    if ARGS.output_layout == "flat":
        return dst_dir / _short_clip_dirname(clip_id, ref_path)
    return dst_dir / ref_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract hand keypoints from prepared gesture clips.")
    parser.add_argument("--src_dir", required=True, help="Prepared clips root.")
    parser.add_argument(
        "--dst_dir",
        default=None,
        help="Output root for unified RTM annotations. Default: <src_parent>/class8_rtm_kpts",
    )
    parser.add_argument(
        "--hand21_dst_dir",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--arm_hand27_dst_dir",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output_layout",
        choices=("flat", "mirror"),
        default="flat",
        help="flat: short clip dir names; mirror: keep source tree layout",
    )
    parser.add_argument("--det_onnx", default=DEFAULT_DET_ONNX)
    parser.add_argument("--person_pose_onnx", default=DEFAULT_PERSON_POSE_ONNX)
    parser.add_argument("--hand_pose_onnx", default=DEFAULT_HAND_POSE_ONNX)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--det_score_thr", type=float, default=0.3)
    parser.add_argument("--det_iou_thr", type=float, default=0.3)
    parser.add_argument("--max_det", type=int, default=100)
    parser.add_argument("--hand_kpt_thr", type=float, default=0.5)
    parser.add_argument("--max_hands", type=int, default=2)
    parser.add_argument("--track_dist_thr", type=float, default=0.18)
    parser.add_argument("--track_max_missing", type=int, default=8)
    parser.add_argument(
        "--include_labels",
        nargs="*",
        default=None,
        help="Only process clips whose label prefix is in this list, e.g. NONE1 NONE2.",
    )
    parser.add_argument(
        "--exclude_labels",
        nargs="*",
        default=None,
        help="Skip clips whose label prefix is in this list.",
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _ns(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        det_score_thr=args.det_score_thr,
        det_iou_thr=args.det_iou_thr,
        max_det=args.max_det,
        det_only=False,
        person_kpt_thr=1.0,
        hand_kpt_thr=args.hand_kpt_thr,
        show=False,
        print_pred=False,
    )


def _iter_video_tasks(src_dir: Path) -> Iterable[ClipTask]:
    for video_path in sorted(src_dir.rglob("*.mp4")):
        yield ClipTask(
            clip_id=video_path.stem,
            clip_path=video_path,
            meta_json_path=None,
            source_video_path=video_path,
            start_frame=0,
            end_frame=-1,
            output_name=str(_output_subdir(Path("."), video_path.stem, video_path)),
        )


def _find_int(payload: dict, keys: Sequence[str]) -> Optional[int]:
    for key in keys:
        value = payload.get(key)
        if value is not None and str(value) != "":
            return int(value)
    return None


def _resolve_source_video(json_path: Path, payload: dict) -> Path:
    for key in ("clip_video_path", "video_path", "source_video_path", "src_video_path"):
        value = payload.get(key)
        if value:
            return Path(str(value))
    recording_dir = json_path.parent.parent
    fallback = recording_dir.with_suffix(".mp4")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Cannot resolve source video for {json_path}")


def _parse_frame_range(json_path: Path, payload: dict) -> Tuple[int, int]:
    start = _find_int(payload, ("start_frame", "frame_start", "start", "start_idx"))
    end = _find_int(payload, ("end_frame", "frame_end", "end", "end_idx"))
    if start is not None and end is not None:
        return start, end
    match = CLIP_RANGE_RE.search(json_path.stem)
    if match:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Cannot resolve frame range from {json_path}")


def _iter_json_tasks(src_dir: Path) -> Iterable[ClipTask]:
    for json_path in sorted(src_dir.rglob("clips/*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        source_video = _resolve_source_video(json_path, payload)
        start_frame, end_frame = _parse_frame_range(json_path, payload)
        yield ClipTask(
            clip_id=json_path.stem,
            clip_path=None,
            meta_json_path=json_path,
            source_video_path=source_video,
            start_frame=start_frame,
            end_frame=end_frame,
            output_name=str(_output_subdir(Path("."), json_path.stem, json_path)),
        )


def collect_tasks(src_dir: Path) -> List[ClipTask]:
    json_tasks = list(_iter_json_tasks(src_dir))
    if json_tasks:
        return json_tasks
    return list(_iter_video_tasks(src_dir))


def _label_from_clip_id(clip_id: str) -> str:
    return clip_id.split("_f", 1)[0].upper()


def _filter_tasks_by_labels(
    tasks: List[ClipTask],
    include_labels: Optional[Sequence[str]],
    exclude_labels: Optional[Sequence[str]],
) -> List[ClipTask]:
    if not include_labels and not exclude_labels:
        return tasks
    wanted = {label.strip().upper() for label in include_labels or [] if label.strip()}
    blocked = {label.strip().upper() for label in exclude_labels or [] if label.strip()}
    filtered = []
    for task in tasks:
        label = _label_from_clip_id(task.clip_id)
        if wanted and label not in wanted:
            continue
        if blocked and label in blocked:
            continue
        filtered.append(task)
    return filtered


def _bbox_from_keypoints(kpts_xy: np.ndarray, img_w: int, img_h: int, pad: float = 0.25) -> List[int]:
    x_min, y_min = np.min(kpts_xy, axis=0)
    x_max, y_max = np.max(kpts_xy, axis=0)
    cx, cy = (x_min + x_max) * 0.5, (y_min + y_max) * 0.5
    side = max(x_max - x_min, y_max - y_min) * (1.0 + pad)
    half = max(side * 0.5, 2.0)
    x1 = int(np.clip(round(cx - half), 0, img_w - 1))
    y1 = int(np.clip(round(cy - half), 0, img_h - 1))
    x2 = int(np.clip(round(cx + half), 0, img_w - 1))
    y2 = int(np.clip(round(cy + half), 0, img_h - 1))
    return [x1, y1, x2, y2]


def _draw(frame: np.ndarray, hands: List[dict]) -> np.ndarray:
    vis = frame.copy()
    for hand in hands:
        kpts = np.asarray(hand["hand21_kpts_xy"], dtype=np.float32)
        bbox = _bbox_from_keypoints(kpts, vis.shape[1], vis.shape[0])
        cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        for u, v in HAND_SKELETON:
            p1 = tuple(np.round(kpts[u]).astype(np.int32))
            p2 = tuple(np.round(kpts[v]).astype(np.int32))
            cv2.line(vis, p1, p2, (255, 255, 0), 2, cv2.LINE_AA)
        for x, y in kpts:
            cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)
        cv2.putText(
            vis,
            f"id={hand['track_id']} {hand.get('handedness_name', 'Unknown')} score={hand['score']:.2f}",
            (bbox[0], max(12, bbox[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return vis


def _is_finite_xy(point: np.ndarray) -> bool:
    return point.shape == (2,) and bool(np.all(np.isfinite(point)))


def _track_point_from_hand(hand: dict) -> np.ndarray:
    upperbody_wrist_xyz = hand.get("upperbody_wrist_xyz")
    upperbody_wrist_score = hand.get("upperbody_wrist_score")
    if upperbody_wrist_xyz is not None and upperbody_wrist_score is not None and float(upperbody_wrist_score) >= 0.2:
        point = np.asarray(upperbody_wrist_xyz[:2], dtype=np.float32)
        if _is_finite_xy(point):
            return point

    kpts = np.asarray(hand["hand21_kpts_xy"], dtype=np.float32)
    if len(kpts) > 0 and _is_finite_xy(kpts[0]):
        return kpts[0]
    return np.mean(kpts, axis=0).astype(np.float32)


def _handedness_compatible(prev_label: int, cur_label: int) -> bool:
    if prev_label == UNKNOWN_HAND_LABEL or cur_label == UNKNOWN_HAND_LABEL:
        return True
    return prev_label == cur_label


def _assign_tracks(
    hands: List[dict],
    state: Dict[int, dict],
    next_id: int,
    img_diag: float,
    thr: float,
    max_missing: int,
) -> int:
    for track in state.values():
        track["missing"] += 1

    used: set[int] = set()
    for hand in hands:
        point = _track_point_from_hand(hand)
        handedness_label = int(hand.get("handedness_label", UNKNOWN_HAND_LABEL))
        best_id = None
        best_dist = None
        for allow_incompatible in (False, True):
            for track_id, track in state.items():
                if track_id in used:
                    continue
                compatible = _handedness_compatible(
                    int(track.get("handedness_label", UNKNOWN_HAND_LABEL)),
                    handedness_label,
                )
                if not compatible and not allow_incompatible:
                    continue
                dist = float(np.linalg.norm(point - track["point"]) / max(img_diag, 1.0))
                if dist <= thr and (best_dist is None or dist < best_dist):
                    best_id = track_id
                    best_dist = dist
            if best_id is not None:
                break
        if best_id is None:
            best_id = next_id
            next_id += 1
        hand["track_id"] = best_id
        used.add(best_id)
        state[best_id] = {
            "point": point,
            "missing": 0,
            "handedness_label": handedness_label,
        }
    for track_id in list(state):
        if state[track_id]["missing"] > max_missing:
            state.pop(track_id, None)
    return next_id


def _select_main_arm(frame_result) -> Optional[dict]:
    person_kpts_2d = frame_result.person_result.keypoints_2d
    person_kpts_3d = frame_result.person_result.keypoints
    person_scores = frame_result.person_result.keypoint_scores
    if len(person_kpts_2d) == 0:
        return None

    best_idx = 0
    best_score = -1.0
    for idx in range(len(person_kpts_2d)):
        cur_scores = person_scores[idx][ARM_ONLY_IDXS] if idx < len(person_scores) else None
        cur_score = float(np.mean(cur_scores)) if cur_scores is not None else 0.0
        if cur_score > best_score:
            best_idx = idx
            best_score = cur_score
    return {
        "xy": person_kpts_2d[best_idx][ARM_ONLY_IDXS].astype(np.float32),
        "xyz": person_kpts_3d[best_idx][ARM_ONLY_IDXS].astype(np.float32),
        "scores": person_scores[best_idx][ARM_ONLY_IDXS].astype(np.float32),
    }


def _handedness_name(label: int) -> str:
    if label == LEFT_HAND_LABEL:
        return "Left"
    if label == RIGHT_HAND_LABEL:
        return "Right"
    return "Unknown"


def _infer_handedness(hand21_xyz: np.ndarray, arm_info: Optional[dict]) -> int:
    if arm_info is None:
        return UNKNOWN_HAND_LABEL

    arm_xy = arm_info["xy"]
    arm_scores = arm_info["scores"]
    hand_wrist_xy = hand21_xyz[0, :2]

    def side_distance(wrist_idx: int, elbow_idx: int, shoulder_idx: int) -> float:
        weighted = 0.0
        total_w = 0.0
        for idx, weight in ((wrist_idx, 3.0), (elbow_idx, 1.5), (shoulder_idx, 0.75)):
            score = float(arm_scores[idx])
            if score < 0.2:
                continue
            dist = float(np.linalg.norm(hand_wrist_xy - arm_xy[idx]))
            weighted += dist * weight
            total_w += weight
        if total_w <= 0:
            return float("inf")
        return weighted / total_w

    left_dist = side_distance(LEFT_WRIST_ARM_IDX, LEFT_ELBOW_ARM_IDX, LEFT_SHOULDER_ARM_IDX)
    right_dist = side_distance(RIGHT_WRIST_ARM_IDX, RIGHT_ELBOW_ARM_IDX, RIGHT_SHOULDER_ARM_IDX)
    if not np.isfinite(left_dist) and not np.isfinite(right_dist):
        return UNKNOWN_HAND_LABEL
    if not np.isfinite(left_dist):
        return RIGHT_HAND_LABEL
    if not np.isfinite(right_dist):
        return LEFT_HAND_LABEL

    shoulder_width = None
    if arm_scores[LEFT_SHOULDER_ARM_IDX] >= 0.2 and arm_scores[RIGHT_SHOULDER_ARM_IDX] >= 0.2:
        shoulder_width = float(np.linalg.norm(arm_xy[LEFT_SHOULDER_ARM_IDX] - arm_xy[RIGHT_SHOULDER_ARM_IDX]))
    threshold = shoulder_width * 1.25 if shoulder_width is not None else None
    best_dist = min(left_dist, right_dist)
    if threshold is not None and best_dist > threshold:
        return UNKNOWN_HAND_LABEL
    if abs(left_dist - right_dist) < max(8.0, best_dist * 0.12):
        return UNKNOWN_HAND_LABEL
    return LEFT_HAND_LABEL if left_dist < right_dist else RIGHT_HAND_LABEL


def _build_upperbody_wrist_xyz(handedness_label: int, arm_info: Optional[dict]) -> tuple[Optional[List[float]], Optional[float], Optional[List[float]]]:
    if arm_info is None:
        return None, None, None
    arm_xyz = arm_info["xyz"]
    arm_scores = arm_info["scores"]
    if handedness_label == LEFT_HAND_LABEL:
        wrist_idx = LEFT_WRIST_ARM_IDX
        shoulder_idx = LEFT_SHOULDER_ARM_IDX
    elif handedness_label == RIGHT_HAND_LABEL:
        wrist_idx = RIGHT_WRIST_ARM_IDX
        shoulder_idx = RIGHT_SHOULDER_ARM_IDX
    else:
        return None, None, None

    wrist_xyz = arm_xyz[wrist_idx].copy()
    shoulder_xyz = arm_xyz[shoulder_idx].copy()
    wrist_xyz[2] = wrist_xyz[2] - shoulder_xyz[2]
    wrist_score = float(min(arm_scores[wrist_idx], arm_scores[shoulder_idx]))
    return wrist_xyz.astype(np.float32).tolist(), wrist_score, shoulder_xyz.astype(np.float32).tolist()


def process_task(
    task: ClipTask,
    det_sess: object,
    person_spec: PoseModelSpec,
    hand_spec: PoseModelSpec,
    args: argparse.Namespace,
) -> None:
    output_dir = args.dst_dir / task.output_name
    output_json = output_dir / "annotations.json"
    if output_json.exists() and not args.overwrite:
        print(f"[skip] {task.clip_id}")
        return

    cap = cv2.VideoCapture(str(task.source_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {task.source_video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = task.end_frame if task.end_frame >= 0 else max(total_frames_video - 1, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, task.start_frame)

    vis_writer = None
    if args.save_vis:
        vis_root = Path(str(args.dst_dir) + "_vis")
        vis_path = vis_root / task.output_name
        vis_path = vis_path.with_suffix(".mp4")
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        vis_writer = cv2.VideoWriter(
            str(vis_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (img_w, img_h),
        )

    infer_args = _ns(args)
    frames = []
    detected_frames = 0
    track_state: Dict[int, dict] = {}
    next_track_id = 0
    img_diag = float(np.hypot(img_w, img_h))

    try:
        for frame_idx in range(task.start_frame, end_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break
            _, frame_result = infer_frame(det_sess, person_spec, hand_spec, frame, infer_args)
            arm_kpts = _select_main_arm(frame_result)
            scores = frame_result.hand_result.keypoint_scores
            kpts_2d = frame_result.hand_result.keypoints_2d
            kpts_3d = frame_result.hand_result.keypoints
            hands = []
            for idx in range(len(kpts_2d)):
                score = float(np.mean(scores[idx])) if idx < len(scores) else 0.0
                if score < args.hand_kpt_thr:
                    continue
                hand21_kpts_xy = kpts_2d[idx].astype(np.float32)
                hand21_kpts_xyz = kpts_3d[idx].astype(np.float32)
                handedness_label = _infer_handedness(hand21_kpts_xyz, arm_kpts)
                upperbody_wrist_xyz, upperbody_wrist_score, upperbody_shoulder_xyz = _build_upperbody_wrist_xyz(
                    handedness_label,
                    arm_kpts,
                )
                hands.append(
                    {
                        "score": score,
                        "handedness_label": handedness_label,
                        "handedness_name": _handedness_name(handedness_label),
                        "hand21_kpts_xy": hand21_kpts_xy.tolist(),
                        "hand21_kpts_xyz": hand21_kpts_xyz.tolist(),
                        "hand21_kpt_scores": scores[idx].astype(np.float32).tolist(),
                        "upperbody_wrist_xyz": upperbody_wrist_xyz,
                        "upperbody_wrist_score": upperbody_wrist_score,
                        "upperbody_shoulder_xyz": upperbody_shoulder_xyz,
                    }
                )
            hands.sort(key=lambda item: item["score"], reverse=True)
            hands = hands[: args.max_hands]
            next_track_id = _assign_tracks(
                hands,
                track_state,
                next_track_id,
                img_diag,
                args.track_dist_thr,
                args.track_max_missing,
            )

            entry = {"frame_idx": frame_idx - task.start_frame, "num_hands": len(hands)}
            if hands:
                entry["hands"] = hands
                detected_frames += 1
            frames.append(entry)

            if vis_writer is not None:
                vis_writer.write(_draw(frame, hands))
    finally:
        cap.release()
        if vis_writer is not None:
            vis_writer.release()

    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "video": str(task.source_video_path),
        "clip_id": task.clip_id,
        "clip_path": str(task.clip_path) if task.clip_path else None,
        "clip_meta_json": str(task.meta_json_path) if task.meta_json_path else None,
        "source_frame_range": [task.start_frame, end_frame],
        "fps": fps,
        "total_frames": len(frames),
        "resolution": [img_w, img_h],
        "detected_frames": detected_frames,
        "track_config": {
            "track_dist_thr": args.track_dist_thr,
            "track_max_missing": args.track_max_missing,
            "match_priority": "same_handedness_first_upperbody_wrist_xy_then_hand21_wrist_xy",
        },
        "num_keypoints": {
            "hand21": 21,
            "upperbody_wrist": 1,
        },
        "coordinate_definition": {
            "hand21_kpts_xyz": {
                "x": "image pixel x on original frame",
                "y": "image pixel y on original frame",
                "z": "hand-model 2.5D relative depth from x_like decoding, scaled by hand bbox width",
            },
            "upperbody_wrist_xyz": {
                "x": "image pixel x on original frame",
                "y": "image pixel y on original frame",
                "z": "same-side wrist z minus same-side shoulder z from upperbody model body decoding",
            },
            "handedness_label": {
                "0": "Left",
                "1": "Right",
                "2": "Unknown",
            },
        },
        "frames": [
            {
                "frame_idx": frame["frame_idx"],
                "num_hands": frame["num_hands"],
                **(
                    {
                        "hands": [
                            {
                                "track_id": hand["track_id"],
                                "handedness_label": hand["handedness_label"],
                                "handedness_name": hand["handedness_name"],
                                "score": hand["score"],
                                "hand21_kpts_xy": hand["hand21_kpts_xy"],
                                "hand21_kpts_xyz": hand["hand21_kpts_xyz"],
                                "hand21_kpt_scores": hand["hand21_kpt_scores"],
                                "upperbody_wrist_xyz": hand["upperbody_wrist_xyz"],
                                "upperbody_wrist_score": hand["upperbody_wrist_score"],
                                "upperbody_shoulder_xyz": hand["upperbody_shoulder_xyz"],
                                "score": hand["score"],
                            }
                            for hand in frame["hands"]
                        ]
                    }
                    if frame["num_hands"] > 0
                    else {}
                ),
            }
            for frame in frames
        ],
    }
    output_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {task.clip_id} -> {output_json}")


def _build_model_specs(args: argparse.Namespace) -> tuple[object, PoseModelSpec, PoseModelSpec]:
    det_sess = build_session(args.det_onnx, args.device)
    person_spec = PoseModelSpec(
        session=build_session(args.person_pose_onnx, args.device),
        preprocess_cfg=PosePreprocessConfig(
            input_size=POSE_INPUT_SIZE,
            mean=POSE_MEAN.copy(),
            std=POSE_STD.copy(),
            bgr_to_rgb=True,
            crop_region=UPPERBODY_CROP_REGION,
            padding=POSE_PADDING,
        ),
        draw_cfg=PoseDrawConfig(
            skeleton=(),
            link_colors=np.zeros((0, 3), dtype=np.uint8),
            point_colors=np.zeros((0, 3), dtype=np.uint8),
        ),
        num_keypoints=79,
        input_size_z=256,
        z_range=2.1744869,
        root_z=0,
        z_output_mode="body",
    )
    hand_spec = PoseModelSpec(
        session=build_session(args.hand_pose_onnx, args.device),
        preprocess_cfg=PosePreprocessConfig(
            input_size=POSE_INPUT_SIZE,
            mean=POSE_MEAN.copy(),
            std=POSE_STD.copy(),
            bgr_to_rgb=True,
            crop_region=None,
            padding=POSE_PADDING,
        ),
        draw_cfg=PoseDrawConfig(
            skeleton=DEMO_HAND_SKELETON,
            link_colors=HAND_LINK_COLORS,
            point_colors=HAND_POINT_COLORS,
        ),
        num_keypoints=21,
        input_size_z=HAND_INPUT_SIZE_Z,
        z_range=HAND_Z_RANGE,
        root_z=HAND_ROOT_Z,
        z_output_mode="x_like",
    )
    return det_sess, person_spec, hand_spec


def _init_worker(args: argparse.Namespace) -> None:
    global WORKER_ARGS, WORKER_DET_SESS, WORKER_PERSON_SPEC, WORKER_HAND_SPEC
    WORKER_ARGS = args
    WORKER_DET_SESS, WORKER_PERSON_SPEC, WORKER_HAND_SPEC = _build_model_specs(args)


def _process_task_worker(task: ClipTask) -> str:
    process_task(task, WORKER_DET_SESS, WORKER_PERSON_SPEC, WORKER_HAND_SPEC, WORKER_ARGS)
    return task.clip_id


def main() -> None:
    global ARGS
    args = parse_args()
    ARGS = args
    src_dir = Path(args.src_dir).expanduser().resolve()
    src_parent = src_dir.parent
    args.dst_dir = Path(
        args.dst_dir or args.hand21_dst_dir or args.arm_hand27_dst_dir or (src_parent / "class8_rtm_kpts")
    ).expanduser().resolve()
    tasks = collect_tasks(src_dir)
    tasks = _filter_tasks_by_labels(tasks, args.include_labels, args.exclude_labels)
    if not tasks:
        filters = []
        if args.include_labels:
            filters.append(f"include_labels={args.include_labels}")
        if args.exclude_labels:
            filters.append(f"exclude_labels={args.exclude_labels}")
        label_hint = f" for {', '.join(filters)}" if filters else ""
        raise FileNotFoundError(f"No clips found under {src_dir}{label_hint}")

    for model_path in (args.det_onnx, args.person_pose_onnx, args.hand_pose_onnx):
        if not Path(model_path).exists():
            raise FileNotFoundError(model_path)

    if args.num_workers <= 1:
        det_sess, person_spec, hand_spec = _build_model_specs(args)
        for task in tasks:
            process_task(task, det_sess, person_spec, hand_spec, args)
        return

    mp_ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.num_workers,
        mp_context=mp_ctx,
        initializer=_init_worker,
        initargs=(args,),
    ) as executor:
        for _ in executor.map(_process_task_worker, tasks, chunksize=1):
            pass


if __name__ == "__main__":
    main()
