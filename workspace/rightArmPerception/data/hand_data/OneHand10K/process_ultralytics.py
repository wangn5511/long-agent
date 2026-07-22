#!/usr/bin/env python3
"""
OneHand10K Extracted Data -> Ultralytics Pose Format Converter
==============================================================

将 extract_image_label.py 提取出的 OneHand10K 数据转换为 Ultralytics YOLO-Pose 格式。

Ultralytics Pose 标签格式 (每行):
    <class_id> <cx> <cy> <w> <h> <x1> <y1> <v1> ... <x21> <y21> <v21>

    - class_id: 类别ID (0 = hand)
    - cx, cy, w, h: 归一化的边界框中心坐标和宽高 [0, 1]
    - xi, yi: 归一化的关键点坐标 [0, 1]
    - vi: 关键点可见性 (YOLO 官方定义):
          0 = 不标注 (该关键点缺失/无标注)
          1 = 标注但不可见 (被遮挡)
          2 = 标注且可见

输入:
    extract_image_label.py 的输出目录:
    extracted/
    ├── train/
    │   ├── images/          # {08d}.png
    │   ├── keypoints.json   # [ [[x0,y0],...,[x20,y20]], ... ]
    │   └── bboxes.json      # [ [x1,y1,x2,y2] | null, ... ]
    ├── valid/
    │   └── ...
    └── test/
        └── ...

输出:
    output_dir/
    ├── images/
    │   ├── train/   -> 符号链接到 extracted/train/images/*.png
    │   └── val/     -> 符号链接到 extracted/valid+test/images/*.png
    ├── labels/
    │   ├── train/   # {08d}.txt
    │   └── val/     # {08d}.txt
    └── dataset.yaml

注意:
    - images 不拷贝，使用符号链接
    - valid + test 合并为 val
    - 无效关键点 ([-1,-1]) 设置 visibility=0, 坐标写为 0 0

Usage:
    python process_ultralytics.py \
        --extracted_dir /home/models/rightArm_collection/OneHand10K/extracted \
        --output_dir /home/models/rightArm_collection/OneHand10K/ultralytics
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np


# ==================== 常量 ====================

IMAGE_SIZE = 256
NUM_KEYPOINTS = 21

KEYPOINT_NAMES = [
    'wrist',
    'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
    'index_mcp', 'index_pip', 'index_dip', 'index_tip',
    'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
    'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
    'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
]


# ==================== 工具函数 ====================

def load_json(path: str) -> list:
    """加载 JSON 文件"""
    with open(path, 'r') as f:
        return json.load(f)


def bbox_xyxy_to_yolo(bbox: list, img_w: int, img_h: int
                      ) -> Tuple[float, float, float, float]:
    """
    xyxy -> YOLO 归一化 (cx, cy, w, h)

    Args:
        bbox: [x_min, y_min, x_max, y_max] 像素坐标
        img_w, img_h: 图像宽高

    Returns:
        (cx, cy, w, h) 归一化到 [0, 1]
    """
    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return (cx, cy, w, h)


def format_yolo_pose_line(bbox_yolo: Tuple[float, float, float, float],
                          keypoints: list,
                          img_w: int, img_h: int) -> str:
    """
    生成一行 Ultralytics Pose 标签

    Args:
        bbox_yolo: (cx, cy, w, h) 归一化
        keypoints: 21 个 [x, y] 像素坐标, 无效为 [-1, -1]
        img_w, img_h: 图像宽高

    Returns:
        "0 cx cy w h x1 y1 v1 x2 y2 v2 ... x21 y21 v21"
    """
    cx, cy, w, h = bbox_yolo
    parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]

    for x, y in keypoints:
        if x < 0 or y < 0:
            # 无效关键点: visibility = 0 (不标注), 坐标写 0
            parts.extend(["0.000000", "0.000000", "0"])
        else:
            # 有效关键点: visibility = 2 (标注且可见)
            # OneHand10K 中无效关键点已标记为 [-1,-1],
            # 剩余的有效关键点都是图像中实际可见的
            nx = x / img_w
            ny = y / img_h
            # 防止超出 [0, 1]
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))
            parts.extend([f"{nx:.6f}", f"{ny:.6f}", "2"])

    return " ".join(parts)


# ==================== 主处理逻辑 ====================

def process_split(extracted_dir: str, split: str,
                  output_dir: str, yolo_split: str,
                  index_offset: int = 0) -> dict:
    """
    处理一个 split 的数据

    Args:
        extracted_dir: extract_image_label.py 输出根目录
        split: 源 split 名 ("train", "valid", "test")
        output_dir: ultralytics 输出根目录
        yolo_split: 目标 split 名 ("train" 或 "val")
        index_offset: 文件命名偏移 (用于 valid+test 合并)

    Returns:
        统计信息 dict
    """
    split_dir = Path(extracted_dir) / split
    images_dir = split_dir / "images"
    kps_path = split_dir / "keypoints.json"
    bbox_path = split_dir / "bboxes.json"

    # 检查源文件
    for p in [images_dir, kps_path, bbox_path]:
        if not p.exists():
            print(f"  [WARN] Not found: {p}, skipping {split}")
            return {'total': 0, 'skipped_no_bbox': 0, 'saved': 0}

    all_keypoints = load_json(str(kps_path))
    all_bboxes = load_json(str(bbox_path))
    total = len(all_keypoints)

    print(f"\n  [{split} -> {yolo_split}] Loaded {total} samples")

    # 输出目录
    out_images_dir = Path(output_dir) / "images" / yolo_split
    out_labels_dir = Path(output_dir) / "labels" / yolo_split
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped_no_bbox = 0

    for i in range(total):
        kps = all_keypoints[i]
        bbox = all_bboxes[i]

        # 无 bbox 的样本跳过 (不应该出现, 因 extract 已剔除异常)
        if bbox is None:
            skipped_no_bbox += 1
            continue

        # 源图像路径
        src_img = images_dir / f"{i:08d}.png"
        if not src_img.exists():
            continue

        # 目标文件名 (带偏移)
        out_idx = index_offset + saved
        dst_img = out_images_dir / f"{out_idx:08d}.png"
        dst_label = out_labels_dir / f"{out_idx:08d}.txt"

        # 符号链接图像 (不拷贝)
        if not dst_img.exists():
            dst_img.symlink_to(src_img.resolve())

        # 生成 YOLO 标签
        bbox_yolo = bbox_xyxy_to_yolo(bbox, IMAGE_SIZE, IMAGE_SIZE)
        label_line = format_yolo_pose_line(bbox_yolo, kps, IMAGE_SIZE, IMAGE_SIZE)

        with open(dst_label, 'w') as f:
            f.write(label_line + "\n")

        saved += 1

    stats = {
        'total': total,
        'skipped_no_bbox': skipped_no_bbox,
        'saved': saved,
    }

    # 关键点统计
    n_full = sum(1 for kps in all_keypoints if all(p[0] >= 0 for p in kps))
    n_partial = sum(1 for kps in all_keypoints
                    if any(p[0] >= 0 for p in kps) and any(p[0] < 0 for p in kps))
    stats['full_kps'] = n_full
    stats['partial_kps'] = n_partial

    print(f"    Saved: {saved}/{total}")
    if skipped_no_bbox:
        print(f"    Skipped (no bbox): {skipped_no_bbox}")
    print(f"    Full kps (v=2 all 21): {n_full}")
    print(f"    Partial kps (some v=0): {n_partial}")

    return stats


def generate_dataset_yaml(output_dir: str):
    """生成 dataset.yaml"""
    out = Path(output_dir)
    yaml_content = f"""# OneHand10K Hand Pose Dataset - Ultralytics Format
# Generated by process_ultralytics.py
# Source: OneHand10K TFRecords (cleaned by extract_image_label.py)

path: {out.resolve()}
train: images/train
val: images/val

# Keypoints configuration
kpt_shape: [{NUM_KEYPOINTS}, 3]  # [num_keypoints, (x, y, visibility)]

# Flip index for augmentation (left-right symmetry not applicable for single hand)
# flip_idx: []

# Classes
names:
  0: hand

# Keypoint names (MediaPipe compatible):
# {KEYPOINT_NAMES}

# Visibility values:
#   0: not labeled (keypoint missing/not annotated)
#   1: labeled but not visible (occluded) — not used in this dataset
#   2: labeled and visible

# Skeleton connections for visualization
# skeleton:
#   - [0, 1]   # wrist -> thumb_cmc
#   - [1, 2]   # thumb_cmc -> thumb_mcp
#   - [2, 3]   # thumb_mcp -> thumb_ip
#   - [3, 4]   # thumb_ip -> thumb_tip
#   - [0, 5]   # wrist -> index_mcp
#   - [5, 6]   # index_mcp -> index_pip
#   - [6, 7]   # index_pip -> index_dip
#   - [7, 8]   # index_dip -> index_tip
#   - [0, 9]   # wrist -> middle_mcp
#   - [9, 10]  # middle_mcp -> middle_pip
#   - [10, 11] # middle_pip -> middle_dip
#   - [11, 12] # middle_dip -> middle_tip
#   - [0, 13]  # wrist -> ring_mcp
#   - [13, 14] # ring_mcp -> ring_pip
#   - [14, 15] # ring_pip -> ring_dip
#   - [15, 16] # ring_dip -> ring_tip
#   - [0, 17]  # wrist -> pinky_mcp
#   - [17, 18] # pinky_mcp -> pinky_pip
#   - [18, 19] # pinky_pip -> pinky_dip
#   - [19, 20] # pinky_dip -> pinky_tip
"""

    yaml_path = out / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\nGenerated: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert extracted OneHand10K data to Ultralytics YOLO-Pose format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 基本用法 (读取 extract_image_label.py 的输出)
    python process_ultralytics.py \\
        --extracted_dir /home/models/rightArm_collection/OneHand10K/extracted \\
        --output_dir /home/models/rightArm_collection/OneHand10K/ultralytics

    # 自定义输出路径
    python process_ultralytics.py \\
        --extracted_dir ./extracted \\
        --output_dir ./yolo_pose_data
        """
    )

    parser.add_argument(
        "--extracted_dir", type=str,
        default="/home/models/rightArm_collection/OneHand10K/extracted",
        help="extract_image_label.py 输出目录 (含 train/ valid/ test/)"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/home/models/rightArm_collection/OneHand10K/ultralytics",
        help="Ultralytics 格式输出目录"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("OneHand10K -> Ultralytics Pose Format")
    print("=" * 60)
    print(f"  Extracted dir: {args.extracted_dir}")
    print(f"  Output dir:    {args.output_dir}")
    print(f"  Split mapping: train->train, valid+test->val")
    print(f"  Images:        symlink (not copy)")
    print(f"  Visibility:    0=not labeled, 2=visible (YOLO official)")
    print("=" * 60)

    all_stats = {}

    # 1) train -> train
    print(f"\n{'='*60}")
    print("Processing train split...")
    print("=" * 60)
    all_stats['train'] = process_split(
        args.extracted_dir, "train",
        args.output_dir, "train",
        index_offset=0
    )

    # 2) valid -> val
    print(f"\n{'='*60}")
    print("Processing valid + test -> val split...")
    print("=" * 60)
    all_stats['valid'] = process_split(
        args.extracted_dir, "valid",
        args.output_dir, "val",
        index_offset=0
    )

    # 3) test -> val (接在 valid 后面)
    val_offset = all_stats['valid']['saved']
    all_stats['test'] = process_split(
        args.extracted_dir, "test",
        args.output_dir, "val",
        index_offset=val_offset
    )

    # 生成 dataset.yaml
    generate_dataset_yaml(args.output_dir)

    # 总结
    train_saved = all_stats['train']['saved']
    val_saved = all_stats['valid']['saved'] + all_stats['test']['saved']

    print(f"\n{'='*60}")
    print("Summary")
    print("=" * 60)
    print(f"  train:  {train_saved} samples")
    print(f"  val:    {val_saved} samples "
          f"(valid={all_stats['valid']['saved']} + test={all_stats['test']['saved']})")
    print(f"  total:  {train_saved + val_saved} samples")
    print(f"\n  Output: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
