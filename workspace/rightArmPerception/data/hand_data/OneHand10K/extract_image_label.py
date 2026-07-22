#!/usr/bin/env python3
"""
OneHand10K TFRecords 数据提取脚本

功能：
  1. 从 TFRecords 中提取图像、关键点标注和 bbox 标注
  2. 图像存储为 PNG 格式，命名: {index:08d}.png (从 0 开始)
  3. 关键点标注转换为像素坐标，保存为 JSON (列表, 每个元素 21 个 [x, y])
  4. BBox 从 mask 提取，以 xyxy 格式保存为 JSON (列表, 每个元素 [x1, y1, x2, y2])
  5. 输出按 train / valid / test 分目录存储
  6. 从提取结果中抽样可视化验证

用法：
  # 提取数据
  python extract_image_label.py \
      --data_root /home/models/rightArm_collection/OneHand10K \
      --output_dir /home/models/rightArm_collection/OneHand10K/extracted \
      --splits train valid test

  # 仅可视化验证 (从已提取的结果)
  python extract_image_label.py \
      --output_dir /home/models/rightArm_collection/OneHand10K/extracted \
      --verify_only --n_verify 10

坐标说明：
  - TFRecord 中 label 为 42 个 float, 交错排列 [x0,y0,...,x20,y20]
  - 坐标范围: [-0.5, 0.5] 中心归一化
  - 反归一化: pixel = (value + 0.5) * 256
  - 无效关键点: 反归一化后 pixel < 0, 标记为 [-1, -1]
  - BBox 从 mask 提取 (非 skeleton), xyxy 格式, 无 mask 时为 null
"""

import os
import json
import glob
import random
import argparse
from typing import Optional

import cv2
import numpy as np

# 禁用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf


# ============================================================
# 常量
# ============================================================

IMAGE_SIZE = 256

# 骨架连接 (与 MediaPipe / eda.py 一致)
HAND_SKELETON = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# 每根手指的颜色 (BGR)
FINGER_COLORS = {
    'thumb': (0, 0, 255),    # 红色
    'index': (0, 255, 0),    # 绿色
    'middle': (255, 0, 0),   # 蓝色
    'ring': (0, 255, 255),   # 黄色
    'pinky': (255, 0, 255),  # 紫色
}


# ============================================================
# TFRecord 解析 (与 eda.py 保持一致)
# ============================================================

def parse_tfrecord_fn(serialized_example):
    """
    解析单条 TFRecord 记录

    Schema:
      - image_raw: bytes (196608 = 256*256*3, raw RGB pixels)
      - image_mask: bytes (65536 = 256*256, single-channel mask)
      - label: float_list (42 = 21*2, interleaved [x0,y0,x1,y1,...])
    """
    feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'image_mask': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([42], tf.float32),
    }

    parsed = tf.io.parse_single_example(serialized_example, feature_description)

    image = tf.io.decode_raw(parsed['image_raw'], tf.uint8)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])

    mask = tf.io.decode_raw(parsed['image_mask'], tf.uint8)
    mask = tf.reshape(mask, [IMAGE_SIZE, IMAGE_SIZE])

    label = parsed['label']

    return {
        'image': image,
        'mask': mask,
        'label': label,
    }


def load_tfrecord_dataset(tfrecord_dir: str, split: str):
    """加载指定 split 的所有 TFRecord 文件"""
    split_dir = os.path.join(tfrecord_dir, split)
    tfrecord_files = sorted(glob.glob(os.path.join(split_dir, "*.tfrecords")))

    if not tfrecord_files:
        raise FileNotFoundError(f"No TFRecord files found in {split_dir}")

    print(f"Found {len(tfrecord_files)} TFRecord files in {split_dir}")

    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset


# ============================================================
# 标注提取
# ============================================================

def unnormalize_keypoints(label: np.ndarray) -> list:
    """
    将 label (42 floats) 转换为像素坐标的 21 个 [x, y]

    - 输入: (42,) float32, 交错 [x0,y0,...,x20,y20], 范围 [-0.5, 0.5]
    - 反归一化: pixel = (value + 0.5) * 256
    - 无效关键点 (反归一化后 < 0) 标记为 [-1, -1]

    Returns:
        list of 21 [x, y], x/y 为 float (像素坐标) 或 -1 (无效)
    """
    kps = label.reshape(21, 2)
    pixel_kps = (kps + 0.5) * IMAGE_SIZE

    result = []
    for x, y in pixel_kps:
        if x < 0 or y < 0:
            result.append([-1, -1])
        else:
            result.append([round(float(x), 2), round(float(y), 2)])
    return result


def get_bbox_from_mask(mask: np.ndarray) -> Optional[list]:
    """
    从 mask 提取边界框, xyxy 格式

    Args:
        mask: (H, W) uint8, >0 为手部区域

    Returns:
        [x_min, y_min, x_max, y_max] 或 None
    """
    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return [int(x_min), int(y_min), int(x_max), int(y_max)]


# ============================================================
# 主提取逻辑
# ============================================================

def extract_split(data_root: str, output_dir: str, split: str):
    """
    提取指定 split 的数据

    输出结构:
      output_dir/{split}/
        images/
          00000000.png
          00000001.png
          ...
        masks/
          00000000.png
          00000001.png
          ...
        keypoints.json     # [ [[x0,y0],...,[x20,y20]], ... ]
        bboxes.json        # [ [x1,y1,x2,y2] | null, ... ]
    """
    # 异常样本阈值: bbox 面积占图像 >90% 视为异常
    # 原因: 原始图像可能不是 256x256, 制作 TFRecord 时 padding/resize 导致
    # mask 填满整图, keypoints 坐标未正确变换, 标注不可靠
    BBOX_AREA_THRESH = 0.9
    img_area = IMAGE_SIZE * IMAGE_SIZE

    tfrecord_dir = os.path.join(data_root, "onehand10k-tfrecords")
    dataset = load_tfrecord_dataset(tfrecord_dir, split)

    # 创建输出目录
    split_output = os.path.join(output_dir, split)
    images_dir = os.path.join(split_output, "images")
    masks_dir = os.path.join(split_output, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    all_keypoints = []
    all_bboxes = []
    saved_count = 0
    total_count = 0
    skipped_count = 0
    skipped_indices = []  # TFRecord 中的原始索引

    for sample in dataset:
        image_rgb = sample['image'].numpy()   # (256, 256, 3) RGB uint8
        mask = sample['mask'].numpy()          # (256, 256) uint8
        label = sample['label'].numpy()        # (42,) float32

        # 先检查 bbox 是否异常 (mask 几乎覆盖全图)
        bbox = get_bbox_from_mask(mask)
        is_suspicious = False
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area / img_area > BBOX_AREA_THRESH:
                is_suspicious = True

        if is_suspicious:
            skipped_indices.append(total_count)
            skipped_count += 1
            total_count += 1
            continue

        # 1) 保存图像为 PNG (RGB -> BGR for OpenCV)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        img_path = os.path.join(images_dir, f"{saved_count:08d}.png")
        cv2.imwrite(img_path, image_bgr)

        # 2) 保存 mask 为 PNG (单通道灰度)
        mask_path = os.path.join(masks_dir, f"{saved_count:08d}.png")
        cv2.imwrite(mask_path, mask)

        # 3) 关键点: 反归一化到像素坐标
        kps = unnormalize_keypoints(label)
        all_keypoints.append(kps)

        # 4) BBox: xyxy
        all_bboxes.append(bbox)

        saved_count += 1
        total_count += 1
        if total_count % 500 == 0:
            print(f"  [{split}] Scanned {total_count} samples "
                  f"(saved {saved_count}, skipped {skipped_count})...")

    # 保存 JSON
    kps_path = os.path.join(split_output, "keypoints.json")
    with open(kps_path, 'w') as f:
        json.dump(all_keypoints, f)

    bbox_path = os.path.join(split_output, "bboxes.json")
    with open(bbox_path, 'w') as f:
        json.dump(all_bboxes, f)

    # 保存被剔除样本的原始索引 (方便溯源)
    skipped_path = os.path.join(split_output, "skipped_indices.json")
    with open(skipped_path, 'w') as f:
        json.dump(skipped_indices, f)

    print(f"\n[{split}] Done!")
    print(f"  Total in TFRecords:  {total_count}")
    print(f"  Skipped (bad bbox):  {skipped_count}")
    print(f"  Saved (clean):       {saved_count}")
    print(f"  Images:    {images_dir}/")
    print(f"  Masks:     {masks_dir}/")
    print(f"  Keypoints: {kps_path}  ({len(all_keypoints)} entries)")
    print(f"  BBoxes:    {bbox_path}  ({len(all_bboxes)} entries)")
    print(f"  Skipped:   {skipped_path}  ({len(skipped_indices)} entries)")

    # 统计 (仅已保存的样本)
    n_valid_bbox = sum(1 for b in all_bboxes if b is not None)
    n_full_kps = sum(1 for kps in all_keypoints
                     if all(p[0] >= 0 for p in kps))
    n_partial_kps = sum(1 for kps in all_keypoints
                        if any(p[0] >= 0 for p in kps) and any(p[0] < 0 for p in kps))
    n_no_kps = sum(1 for kps in all_keypoints
                   if all(p[0] < 0 for p in kps))

    print(f"\n  Clean data statistics:")
    print(f"    Valid bboxes (from mask):  {n_valid_bbox}/{saved_count}")
    print(f"    Full keypoints (21/21):    {n_full_kps}/{saved_count}")
    print(f"    Partial keypoints:         {n_partial_kps}/{saved_count}")
    print(f"    No keypoints (all -1):     {n_no_kps}/{saved_count}")


# ============================================================
# 可视化验证 (从提取结果中抽样)
# ============================================================

def draw_skeleton_on_image(img: np.ndarray, keypoints: list,
                           bbox: Optional[list] = None) -> np.ndarray:
    """
    在图像上绘制骨架和 bbox

    Args:
        img: BGR 图像
        keypoints: 21 个 [x, y], 无效为 [-1, -1]
        bbox: [x1, y1, x2, y2] 或 None

    Returns:
        绘制后的图像
    """
    vis = img.copy()
    kps = np.array(keypoints, dtype=np.float32)

    # 绘制骨架连接
    for i, (start, end) in enumerate(HAND_SKELETON):
        if kps[start][0] < 0 or kps[end][0] < 0:
            continue
        if i < 4:
            color = FINGER_COLORS['thumb']
        elif i < 8:
            color = FINGER_COLORS['index']
        elif i < 12:
            color = FINGER_COLORS['middle']
        elif i < 16:
            color = FINGER_COLORS['ring']
        else:
            color = FINGER_COLORS['pinky']
        pt1 = (int(kps[start][0]), int(kps[start][1]))
        pt2 = (int(kps[end][0]), int(kps[end][1]))
        cv2.line(vis, pt1, pt2, color, 1)

    # 绘制关键点
    for i, (x, y) in enumerate(kps):
        if x < 0:
            continue
        if i == 0:
            color = (255, 255, 255)
        elif i <= 4:
            color = FINGER_COLORS['thumb']
        elif i <= 8:
            color = FINGER_COLORS['index']
        elif i <= 12:
            color = FINGER_COLORS['middle']
        elif i <= 16:
            color = FINGER_COLORS['ring']
        else:
            color = FINGER_COLORS['pinky']
        cv2.circle(vis, (int(x), int(y)), 2, color, -1, cv2.FILLED)
        cv2.circle(vis, (int(x), int(y)), 2, (255, 255, 255), 1)
        # 标注索引
        cv2.putText(vis, str(i), (int(x) + 3, int(y) + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1,
                    cv2.LINE_AA)

    # 绘制 bbox
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(vis, "bbox(mask)", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1,
                    cv2.LINE_AA)

    return vis


def verify_split(output_dir: str, split: str, n_verify: int = 10,
                 eda_output_dir: Optional[str] = None):
    """
    从提取结果中随机抽样进行可视化验证

    读取已提取的 (已清洗) images/ + masks/ + keypoints.json + bboxes.json,
    生成三列拼图: [原图+骨架+bbox | mask | mask叠加骨架],
    保存到 eda_output/{split}/

    Args:
        output_dir: 提取结果根目录 (含 train/ valid/ test/)
        split: 数据集 split
        n_verify: 随机抽样数量
        eda_output_dir: 可视化输出目录, 默认 output_dir 同级的 eda_output
    """
    split_dir = os.path.join(output_dir, split)
    images_dir = os.path.join(split_dir, "images")
    masks_dir = os.path.join(split_dir, "masks")
    kps_path = os.path.join(split_dir, "keypoints.json")
    bbox_path = os.path.join(split_dir, "bboxes.json")

    # 检查文件存在
    for p in [images_dir, kps_path, bbox_path]:
        if not os.path.exists(p):
            print(f"  [WARN] Not found: {p}, skipping {split}")
            return

    has_masks = os.path.isdir(masks_dir)

    with open(kps_path, 'r') as f:
        all_keypoints = json.load(f)
    with open(bbox_path, 'r') as f:
        all_bboxes = json.load(f)

    total = len(all_keypoints)
    print(f"\n[{split}] Loaded {total} clean entries for verification")

    # 加载被剔除样本信息 (仅用于统计打印)
    skipped_path = os.path.join(split_dir, "skipped_indices.json")
    if os.path.exists(skipped_path):
        with open(skipped_path, 'r') as f:
            skipped = json.load(f)
        print(f"  (original had {total + len(skipped)} samples, "
              f"{len(skipped)} were skipped due to bad bbox)")

    # 随机抽样索引
    n_verify = min(n_verify, total)
    indices = sorted(random.sample(range(total), n_verify))

    # 输出目录
    if eda_output_dir is None:
        eda_output_dir = os.path.join(os.path.dirname(output_dir.rstrip("/")),
                                      "eda_output")
    verify_dir = os.path.join(eda_output_dir, split)
    os.makedirs(verify_dir, exist_ok=True)

    for idx in indices:
        img_path = os.path.join(images_dir, f"{idx:08d}.png")
        if not os.path.exists(img_path):
            print(f"  [WARN] Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        kps = all_keypoints[idx]
        bbox = all_bboxes[idx]

        # 统计
        n_valid = sum(1 for p in kps if p[0] >= 0)
        bbox_ratio = 0.0
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            bbox_ratio = (x2 - x1) * (y2 - y1) / (IMAGE_SIZE * IMAGE_SIZE)

        # Panel 1: 原图 + 骨架 + bbox
        vis_skel = draw_skeleton_on_image(img, kps, bbox)
        title1 = f"idx={idx:08d}  kps={n_valid}/21  bbox={bbox_ratio:.0%}"
        cv2.putText(vis_skel, title1, (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 255), 1,
                    cv2.LINE_AA)

        # Panel 2: mask
        if has_masks:
            mask_file = os.path.join(masks_dir, f"{idx:08d}.png")
            if os.path.exists(mask_file):
                mask_gray = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            else:
                mask_gray = np.zeros((h, w), dtype=np.uint8)
        else:
            mask_gray = np.zeros((h, w), dtype=np.uint8)

        mask_vis = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
        mask_colored = np.zeros_like(img)
        mask_colored[mask_gray > 0] = (0, 200, 0)
        mask_vis = cv2.addWeighted(mask_vis, 0.5, mask_colored, 0.5, 0)
        cv2.putText(mask_vis, "mask", (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1,
                    cv2.LINE_AA)

        # Panel 3: 原图 + mask 叠加 + 骨架
        overlay = img.copy()
        green_mask = np.zeros_like(img)
        green_mask[mask_gray > 0] = (0, 180, 0)
        overlay = cv2.addWeighted(overlay, 0.7, green_mask, 0.3, 0)
        overlay = draw_skeleton_on_image(overlay, kps, bbox)
        cv2.putText(overlay, "img+mask+skel", (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1,
                    cv2.LINE_AA)

        canvas = np.hstack([vis_skel, mask_vis, overlay])
        out_path = os.path.join(verify_dir, f"verify_{idx:08d}.jpg")
        cv2.imwrite(out_path, canvas)

    print(f"  [{split}] Saved {n_verify} verification images to {verify_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Extract images, keypoints and bboxes from OneHand10K TFRecords"
    )
    parser.add_argument(
        "--data_root", type=str,
        default="/home/models/rightArm_collection/OneHand10K",
        help="Root directory containing onehand10k-tfrecords/"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/home/models/rightArm_collection/OneHand10K/extracted",
        help="Output directory for extracted data"
    )
    parser.add_argument(
        "--splits", nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test"],
        help="Which splits to extract (default: all)"
    )
    parser.add_argument(
        "--verify_only", action="store_true",
        help="Skip extraction, only run verification on existing results"
    )
    parser.add_argument(
        "--n_verify", type=int, default=10,
        help="Number of samples to visualize for verification (default: 10)"
    )
    parser.add_argument(
        "--eda_output_dir", type=str, default=None,
        help="Verification output directory (default: {data_root}/eda_output)"
    )

    args = parser.parse_args()

    # 默认 eda_output_dir 为 data_root/eda_output
    eda_output_dir = args.eda_output_dir
    if eda_output_dir is None:
        eda_output_dir = os.path.join(
            os.path.dirname(args.output_dir.rstrip("/")), "eda_output"
        )

    if not args.verify_only:
        print("=" * 60)
        print("OneHand10K TFRecord Extraction")
        print("=" * 60)
        print(f"  Data root:  {args.data_root}")
        print(f"  Output dir: {args.output_dir}")
        print(f"  Splits:     {args.splits}")
        print("=" * 60)

        for split in args.splits:
            print(f"\n{'='*60}")
            print(f"Extracting {split} split...")
            print(f"{'='*60}")
            extract_split(args.data_root, args.output_dir, split)

        print(f"\n{'='*60}")
        print("Extraction complete!")
        print(f"{'='*60}")

    # 可视化验证
    print(f"\n{'='*60}")
    print("Verification: sampling extracted results for visualization")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  EDA output dir: {eda_output_dir}")
    print(f"  N samples:      {args.n_verify}")
    print(f"{'='*60}")

    for split in args.splits:
        verify_split(args.output_dir, split, args.n_verify, eda_output_dir)

    print(f"\n{'='*60}")
    print("All done!")
    print(f"  Extracted data: {args.output_dir}")
    print(f"  Verification:   {eda_output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
