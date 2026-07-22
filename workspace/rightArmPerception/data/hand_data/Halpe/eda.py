"""
Halpe Full-Body 136 Keypoint Dataset - EDA (Exploratory Data Analysis)

Halpe 是一个全身 136 关键点数据集，标注覆盖身体、面部、手部和脚部。
标注格式为 COCO JSON。

数据目录结构:
  Halpe/
  ├── annotations/
  │   ├── halpe_train_v1.json
  │   └── halpe_val_v1.json
  ├── hico_20160224_det/           # 训练集图像 (来自 HICO-Det)
  │   └── images/train2015/
  ├── val2017/                     # 验证集图像 (来自 COCO val2017)
  └── person_detection_results/

136 关键点定义:
  0-25:    Body (26 个身体+脚部关键点, 扩展 COCO 17 body)
  26-93:   Face (68 个面部关键点)
  94-114:  Left Hand (21 个左手关键点, MediaPipe 风格)
  115-135: Right Hand (21 个右手关键点, MediaPipe 风格)

参考:
  - https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_wholebody_keypoint.html#halpe
  - https://github.com/Fang-Haoshu/Halpe-FullBody
"""

import os
import cv2
import json
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional, List, Dict, Tuple
import argparse

# MediaPipe
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ============================================================
# MediaPipe HandLandmarker 模型路径 (与 FreiHAND 共用)
# ============================================================
_HAND_LANDMARKER_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "FreiHAND", "hand_landmarker.task"
)


# ============================================================
# Halpe 136 关键点定义
# ============================================================

# Body (0-25): 26 个身体关键点
BODY_KEYPOINT_NAMES = [
    "Nose",          # 0
    "LEye",          # 1
    "REye",          # 2
    "LEar",          # 3
    "REar",          # 4
    "LShoulder",     # 5
    "RShoulder",     # 6
    "LElbow",        # 7
    "RElbow",        # 8
    "LWrist",        # 9
    "RWrist",        # 10
    "LHip",          # 11
    "RHip",          # 12
    "LKnee",         # 13
    "RKnee",         # 14
    "LAnkle",        # 15
    "RAnkle",        # 16
    "Head",          # 17
    "Neck",          # 18
    "Hip",           # 19 (center)
    "LBigToe",       # 20
    "RBigToe",       # 21
    "LSmallToe",     # 22
    "RSmallToe",     # 23
    "LHeel",         # 24
    "RHeel",         # 25
]

# Hand keypoint names (21 per hand, MediaPipe 风格)
HAND_KEYPOINT_NAMES = [
    "Wrist",         # +0
    "Thumb_CMC",     # +1
    "Thumb_MCP",     # +2
    "Thumb_IP",      # +3
    "Thumb_TIP",     # +4
    "Index_MCP",     # +5
    "Index_PIP",     # +6
    "Index_DIP",     # +7
    "Index_TIP",     # +8
    "Middle_MCP",    # +9
    "Middle_PIP",    # +10
    "Middle_DIP",    # +11
    "Middle_TIP",    # +12
    "Ring_MCP",      # +13
    "Ring_PIP",      # +14
    "Ring_DIP",      # +15
    "Ring_TIP",      # +16
    "Pinky_MCP",     # +17
    "Pinky_PIP",     # +18
    "Pinky_DIP",     # +19
    "Pinky_TIP",     # +20
]


# ============================================================
# 骨架连接定义
# ============================================================

# Body skeleton (26 个身体关键点之间的连接)
BODY_SKELETON = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Neck & Head
    (0, 17), (17, 18),
    # Shoulders -> Neck
    (5, 18), (6, 18),
    # Left arm
    (5, 7), (7, 9),
    # Right arm
    (6, 8), (8, 10),
    # Torso
    (5, 19), (6, 19),
    # Hips
    (19, 11), (19, 12),
    # Left leg
    (11, 13), (13, 15),
    # Right leg
    (12, 14), (14, 16),
    # Left foot
    (15, 20), (15, 24), (20, 22),
    # Right foot
    (16, 21), (16, 25), (21, 23),
]

# Hand skeleton (21 个手关键点之间的连接, 偏移量相对于手腕)
HAND_SKELETON_OFFSETS = [
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

# Face contour skeleton (68 个面部关键点)
FACE_SKELETON_OFFSETS = [
    # Jaw contour (0-16)
    *[(i, i + 1) for i in range(16)],
    # Left eyebrow (17-21)
    *[(i, i + 1) for i in range(17, 21)],
    # Right eyebrow (22-26)
    *[(i, i + 1) for i in range(22, 26)],
    # Nose bridge (27-30)
    *[(i, i + 1) for i in range(27, 30)],
    # Nose bottom (31-35)
    *[(i, i + 1) for i in range(31, 35)],
    # Left eye (36-41)
    *[(i, i + 1) for i in range(36, 41)], (41, 36),
    # Right eye (42-47)
    *[(i, i + 1) for i in range(42, 47)], (47, 42),
    # Outer lip (48-59)
    *[(i, i + 1) for i in range(48, 59)], (59, 48),
    # Inner lip (60-67)
    *[(i, i + 1) for i in range(60, 67)], (67, 60),
]

# 颜色定义 (BGR)
BODY_COLOR = (0, 255, 0)         # 绿色
BODY_LEFT_COLOR = (255, 128, 0)  # 蓝偏
BODY_RIGHT_COLOR = (0, 128, 255) # 橙偏
FACE_COLOR = (255, 200, 100)     # 浅蓝
LHAND_BBOX_COLOR = (0, 200, 255) # 橙黄
RHAND_BBOX_COLOR = (255, 0, 200) # 粉紫
BBOX_COLOR = (0, 255, 0)         # 绿色
MP_COLOR = (0, 165, 255)         # 橙色 (MediaPipe)

# 手指颜色 (BGR)
FINGER_COLORS = {
    'thumb':  (0, 0, 255),    # 红
    'index':  (0, 255, 0),    # 绿
    'middle': (255, 0, 0),    # 蓝
    'ring':   (0, 255, 255),  # 黄
    'pinky':  (255, 0, 255),  # 紫
}

# 手部大小分 bin 的阈值 (手部 bbox 面积占图像面积的比例 %)
# tiny:   < 0.5%
# small:  0.5% - 2%
# medium: 2% - 5%
# large:  >= 5%
HAND_SIZE_BINS = [
    ("tiny",   0.0, 0.5),
    ("small",  0.5, 2.0),
    ("medium", 2.0, 5.0),
    ("large",  5.0, 100.0),
]


# ============================================================
# 数据加载
# ============================================================

def load_annotations(anno_path: str) -> dict:
    """加载 COCO 格式的 Halpe 标注文件"""
    print(f"Loading annotations from {anno_path} ...")
    with open(anno_path, 'r') as f:
        data = json.load(f)
    print(f"  images: {len(data.get('images', []))}")
    print(f"  annotations: {len(data.get('annotations', []))}")
    print(f"  categories: {len(data.get('categories', []))}")
    return data


def build_img_id_to_info(data: dict) -> dict:
    """构建 image_id -> image_info 映射"""
    return {img['id']: img for img in data['images']}


def build_img_id_to_anns(data: dict) -> dict:
    """构建 image_id -> [annotations] 映射"""
    mapping = defaultdict(list)
    for ann in data['annotations']:
        mapping[ann['image_id']].append(ann)
    return mapping


# ============================================================
# 关键点/BBox 工具
# ============================================================

def parse_keypoints(kpts_flat: list) -> np.ndarray:
    """
    解析 COCO 格式的 keypoints 列表

    Args:
        kpts_flat: [x0,y0,v0, x1,y1,v1, ..., x135,y135,v135] 长度 = 136*3 = 408

    Returns:
        (136, 3) np.ndarray, 每行 [x, y, visibility]
    """
    n = len(kpts_flat) // 3
    return np.array(kpts_flat).reshape(n, 3)


def get_hand_bbox(kpts: np.ndarray, hand_offset: int,
                  min_visible: int = 3,
                  padding_ratio: float = 0.15) -> Optional[Tuple[int, int, int, int]]:
    """
    从可见的手部关键点计算手部外接 bbox

    Args:
        kpts: (136, 3) 关键点数组
        hand_offset: 94 (左手) 或 115 (右手)
        min_visible: 最少可见关键点数才返回有效 bbox
        padding_ratio: bbox 各边扩展比例 (相对于 bbox 大小)

    Returns:
        (x1, y1, x2, y2) 或 None (可见点太少)
    """
    hand_kpts = kpts[hand_offset:hand_offset + 21]
    visible_mask = hand_kpts[:, 2] > 0
    n_vis = visible_mask.sum()

    if n_vis < min_visible:
        return None

    vis_coords = hand_kpts[visible_mask, :2]  # (n_vis, 2)
    x_min, y_min = vis_coords.min(axis=0)
    x_max, y_max = vis_coords.max(axis=0)

    # 添加 padding
    w = x_max - x_min
    h = y_max - y_min
    pad_x = w * padding_ratio
    pad_y = h * padding_ratio

    x1 = int(max(0, x_min - pad_x))
    y1 = int(max(0, y_min - pad_y))
    x2 = int(x_max + pad_x)
    y2 = int(y_max + pad_y)

    return (x1, y1, x2, y2)


def get_hand_bbox_no_padding(kpts: np.ndarray, hand_offset: int,
                             min_visible: int = 3) -> Optional[Tuple[int, int, int, int]]:
    """不带 padding 的手部 bbox，用于面积统计"""
    hand_kpts = kpts[hand_offset:hand_offset + 21]
    visible_mask = hand_kpts[:, 2] > 0
    n_vis = visible_mask.sum()

    if n_vis < min_visible:
        return None

    vis_coords = hand_kpts[visible_mask, :2]
    x_min, y_min = vis_coords.min(axis=0)
    x_max, y_max = vis_coords.max(axis=0)

    return (float(x_min), float(y_min), float(x_max), float(y_max))


# ============================================================
# 手部大小分布分析
# ============================================================

def compute_all_hand_infos(data: dict, img_id_to_info: dict) -> List[Dict]:
    """
    遍历所有标注，提取每只手的 bbox 和大小信息

    Returns:
        hand_infos: list of dicts, 每项包含:
            image_id, ann_id, hand_side ('left'/'right'),
            hand_bbox (x1,y1,x2,y2), hand_area, img_area,
            area_ratio (%), max_side (px), n_visible
    """
    hand_infos = []
    annotations = data['annotations']

    for ann in annotations:
        kpts = parse_keypoints(ann['keypoints'])
        img_info = img_id_to_info.get(ann['image_id'])
        if img_info is None:
            continue
        img_w, img_h = img_info['width'], img_info['height']
        img_area = img_w * img_h

        for hand_side, offset in [('left', 94), ('right', 115)]:
            hand_kpts = kpts[offset:offset + 21]
            n_vis = int((hand_kpts[:, 2] > 0).sum())
            bbox = get_hand_bbox_no_padding(kpts, offset, min_visible=3)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            bw = x2 - x1
            bh = y2 - y1
            hand_area = bw * bh
            area_ratio = (hand_area / img_area) * 100  # 百分比
            max_side = max(bw, bh)

            hand_infos.append({
                'image_id': ann['image_id'],
                'ann_id': ann.get('id', None),
                'hand_side': hand_side,
                'hand_bbox': (x1, y1, x2, y2),
                'hand_area': hand_area,
                'img_area': img_area,
                'img_w': img_w,
                'img_h': img_h,
                'area_ratio': area_ratio,
                'max_side': max_side,
                'n_visible': n_vis,
            })

    return hand_infos


def categorize_hand_by_size(area_ratio: float) -> str:
    """根据面积占比分类手部大小"""
    for name, low, high in HAND_SIZE_BINS:
        if low <= area_ratio < high:
            return name
    return "large"  # fallback


def print_hand_size_stats(hand_infos: List[Dict], split: str):
    """打印手部大小分布统计"""
    if not hand_infos:
        print("  No valid hand bboxes found.")
        return

    ratios = [h['area_ratio'] for h in hand_infos]
    max_sides = [h['max_side'] for h in hand_infos]

    # 按 bin 统计
    bin_counts = defaultdict(int)
    for h in hand_infos:
        cat = categorize_hand_by_size(h['area_ratio'])
        bin_counts[cat] += 1

    total = len(hand_infos)
    left_count = sum(1 for h in hand_infos if h['hand_side'] == 'left')
    right_count = total - left_count

    print(f"\n{'='*60}")
    print(f"Hand Size Distribution - {split}")
    print(f"{'='*60}")
    print(f"  有效手部 bbox 总数: {total}")
    print(f"    左手: {left_count}, 右手: {right_count}")
    print(f"  面积占比 (area_ratio = hand_bbox_area / img_area):")
    print(f"    min:    {min(ratios):.3f}%")
    print(f"    max:    {max(ratios):.3f}%")
    print(f"    mean:   {np.mean(ratios):.3f}%")
    print(f"    median: {np.median(ratios):.3f}%")
    print(f"  手部 bbox 最大边 (pixels):")
    print(f"    min:    {min(max_sides):.0f} px")
    print(f"    max:    {max(max_sides):.0f} px")
    print(f"    mean:   {np.mean(max_sides):.0f} px")
    print(f"    median: {np.median(max_sides):.0f} px")
    print(f"  按大小分 bin:")
    for name, low, high in HAND_SIZE_BINS:
        cnt = bin_counts.get(name, 0)
        pct = cnt / total * 100 if total else 0
        print(f"    {name:8s} ({low:.1f}%-{high:.1f}%): "
              f"{cnt:6d} ({pct:5.1f}%)")
    print(f"{'='*60}")


def plot_hand_size_distribution(hand_infos: List[Dict], output_path: str, split: str):
    """
    绘制手部大小分布直方图 (面积占比 + 最大边像素)

    保存到 output_path
    """
    if not hand_infos:
        return

    ratios = [h['area_ratio'] for h in hand_infos]
    max_sides = [h['max_side'] for h in hand_infos]

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # 子图1: 面积占比分布
    ax = axs[0]
    # 使用 log-scale x 轴更清晰
    bins_ratio = np.logspace(np.log10(max(0.001, min(ratios))),
                             np.log10(max(ratios) * 1.1), 50)
    ax.hist(ratios, bins=bins_ratio, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('Hand BBox Area Ratio (%)')
    ax.set_ylabel('Count')
    ax.set_title(f'Hand Size Distribution (Area Ratio)\n'
                 f'N={len(ratios)}, median={np.median(ratios):.3f}%')
    # 画 bin 分界线
    for name, low, high in HAND_SIZE_BINS:
        if low > 0:
            ax.axvline(x=low, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax.text(low, ax.get_ylim()[1] * 0.95, f' {low}%',
                    color='red', fontsize=8, va='top')
    ax.grid(True, alpha=0.3)

    # 子图2: 最大边像素分布
    ax = axs[1]
    ax.hist(max_sides, bins=50, color='coral', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Hand BBox Max Side (pixels)')
    ax.set_ylabel('Count')
    ax.set_title(f'Hand Size Distribution (Max Side px)\n'
                 f'N={len(max_sides)}, median={np.median(max_sides):.0f}px')
    # 画参考线
    for px, label in [(32, '32px'), (64, '64px'), (128, '128px')]:
        ax.axvline(x=px, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.text(px, ax.get_ylim()[1] * 0.95, f' {label}',
                color='red', fontsize=8, va='top')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Halpe {split} - Hand Size Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Distribution plot saved: {output_path}")


# ============================================================
# 统计分析
# ============================================================

def compute_statistics(data: dict) -> dict:
    """计算数据集统计信息"""
    images = data['images']
    annotations = data['annotations']

    # 图像尺寸统计
    widths = [img['width'] for img in images]
    heights = [img['height'] for img in images]

    # 每张图像的标注数量
    img_ann_counts = defaultdict(int)
    for ann in annotations:
        img_ann_counts[ann['image_id']] += 1
    ann_counts = list(img_ann_counts.values())

    # 关键点可见性统计
    vis_counts = {0: 0, 1: 0, 2: 0}
    n_keypoints_per_ann = []
    num_kpts = 0

    body_visible = 0
    face_visible = 0
    lhand_visible = 0
    rhand_visible = 0

    for ann in annotations:
        kpts = ann['keypoints']
        num_kpts = len(kpts) // 3
        n_vis = 0
        for i in range(num_kpts):
            v = kpts[i * 3 + 2]
            vis_counts[int(v)] = vis_counts.get(int(v), 0) + 1
            if v > 0:
                n_vis += 1
                if i <= 25:
                    body_visible += 1
                elif i <= 93:
                    face_visible += 1
                elif i <= 114:
                    lhand_visible += 1
                else:
                    rhand_visible += 1
        n_keypoints_per_ann.append(n_vis)

    stats = {
        'num_images': len(images),
        'num_annotations': len(annotations),
        'num_keypoints_per_instance': num_kpts,
        'image_width_range': (min(widths), max(widths)),
        'image_height_range': (min(heights), max(heights)),
        'anns_per_image': {
            'min': min(ann_counts) if ann_counts else 0,
            'max': max(ann_counts) if ann_counts else 0,
            'mean': np.mean(ann_counts) if ann_counts else 0,
        },
        'visibility_counts': vis_counts,
        'visible_keypoints_per_ann': {
            'min': min(n_keypoints_per_ann) if n_keypoints_per_ann else 0,
            'max': max(n_keypoints_per_ann) if n_keypoints_per_ann else 0,
            'mean': np.mean(n_keypoints_per_ann) if n_keypoints_per_ann else 0,
        },
        'body_visible_total': body_visible,
        'face_visible_total': face_visible,
        'lhand_visible_total': lhand_visible,
        'rhand_visible_total': rhand_visible,
    }
    return stats


def print_statistics(stats: dict, split: str):
    """打印统计信息"""
    print(f"\n{'='*60}")
    print(f"Halpe Dataset Statistics - {split}")
    print(f"{'='*60}")
    print(f"  图像数量: {stats['num_images']}")
    print(f"  标注数量 (person instances): {stats['num_annotations']}")
    print(f"  每个实例关键点数: {stats['num_keypoints_per_instance']}")
    print(f"  图像宽度范围: {stats['image_width_range']}")
    print(f"  图像高度范围: {stats['image_height_range']}")
    print(f"  每张图像标注数: min={stats['anns_per_image']['min']}, "
          f"max={stats['anns_per_image']['max']}, "
          f"mean={stats['anns_per_image']['mean']:.2f}")
    print(f"  关键点可见性统计:")
    print(f"    未标注 (v=0): {stats['visibility_counts'].get(0, 0)}")
    print(f"    标注不可见 (v=1): {stats['visibility_counts'].get(1, 0)}")
    print(f"    标注可见 (v=2): {stats['visibility_counts'].get(2, 0)}")
    print(f"  每个实例可见关键点数: min={stats['visible_keypoints_per_ann']['min']}, "
          f"max={stats['visible_keypoints_per_ann']['max']}, "
          f"mean={stats['visible_keypoints_per_ann']['mean']:.1f}")
    print(f"  分部位可见关键点总数:")
    print(f"    Body (0-25):       {stats['body_visible_total']}")
    print(f"    Face (26-93):      {stats['face_visible_total']}")
    print(f"    Left Hand (94-114):  {stats['lhand_visible_total']}")
    print(f"    Right Hand (115-135): {stats['rhand_visible_total']}")
    print(f"{'='*60}")


# ============================================================
# MediaPipe 推理
# ============================================================

def run_mediapipe_inference(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    使用 MediaPipe Tasks API 对图像进行手部关键点检测

    Args:
        img_bgr: BGR 格式的图像 (可以是手部 crop 或完整图像)

    Returns:
        (21, 2) 的关键点坐标数组 [x, y] (像素坐标), 或 None
    """
    if not os.path.exists(_HAND_LANDMARKER_MODEL_PATH):
        return None

    base_options = mp_python.BaseOptions(
        model_asset_path=_HAND_LANDMARKER_MODEL_PATH
    )
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
    )
    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            h, w = img_bgr.shape[:2]
            keypoints = np.array([
                [lm.x * w, lm.y * h] for lm in landmarks
            ])
            return keypoints
        return None


# ============================================================
# 可视化工具
# ============================================================

def draw_body_skeleton(img: np.ndarray, kpts: np.ndarray,
                       line_thickness: int = 2, point_radius: int = 3) -> np.ndarray:
    """绘制身体骨架 (关键点 0-25)"""
    for start, end in BODY_SKELETON:
        if kpts[start, 2] > 0 and kpts[end, 2] > 0:
            pt1 = (int(kpts[start, 0]), int(kpts[start, 1]))
            pt2 = (int(kpts[end, 0]), int(kpts[end, 1]))
            if start in [1, 3, 5, 7, 9, 11, 13, 15, 20, 22, 24] or \
               end in [1, 3, 5, 7, 9, 11, 13, 15, 20, 22, 24]:
                color = BODY_LEFT_COLOR
            elif start in [2, 4, 6, 8, 10, 12, 14, 16, 21, 23, 25] or \
                 end in [2, 4, 6, 8, 10, 12, 14, 16, 21, 23, 25]:
                color = BODY_RIGHT_COLOR
            else:
                color = BODY_COLOR
            cv2.line(img, pt1, pt2, color, line_thickness)

    for i in range(26):
        if kpts[i, 2] > 0:
            x, y = int(kpts[i, 0]), int(kpts[i, 1])
            cv2.circle(img, (x, y), point_radius, BODY_COLOR, -1)
            cv2.circle(img, (x, y), point_radius, (255, 255, 255), 1)
    return img


def draw_face_keypoints(img: np.ndarray, kpts: np.ndarray,
                        point_radius: int = 1, line_thickness: int = 1) -> np.ndarray:
    """绘制面部关键点 (关键点 26-93)"""
    face_offset = 26
    for start_off, end_off in FACE_SKELETON_OFFSETS:
        start = start_off + face_offset
        end = end_off + face_offset
        if kpts[start, 2] > 0 and kpts[end, 2] > 0:
            pt1 = (int(kpts[start, 0]), int(kpts[start, 1]))
            pt2 = (int(kpts[end, 0]), int(kpts[end, 1]))
            cv2.line(img, pt1, pt2, FACE_COLOR, line_thickness)

    for i in range(26, 94):
        if kpts[i, 2] > 0:
            x, y = int(kpts[i, 0]), int(kpts[i, 1])
            cv2.circle(img, (x, y), point_radius, FACE_COLOR, -1)
    return img


def draw_hand_skeleton_on_img(img: np.ndarray, kpts: np.ndarray, hand_offset: int,
                              line_thickness: int = 2, point_radius: int = 3,
                              show_indices: bool = False) -> np.ndarray:
    """
    绘制手部骨架 (21 个关键点) 到完整图像上

    Args:
        kpts: (136, 3) 全身关键点数组
        hand_offset: 94 (左手) 或 115 (右手)
    """
    for conn_idx, (start_off, end_off) in enumerate(HAND_SKELETON_OFFSETS):
        start = start_off + hand_offset
        end = end_off + hand_offset
        if kpts[start, 2] > 0 and kpts[end, 2] > 0:
            pt1 = (int(kpts[start, 0]), int(kpts[start, 1]))
            pt2 = (int(kpts[end, 0]), int(kpts[end, 1]))
            if conn_idx < 4:
                color = FINGER_COLORS['thumb']
            elif conn_idx < 8:
                color = FINGER_COLORS['index']
            elif conn_idx < 12:
                color = FINGER_COLORS['middle']
            elif conn_idx < 16:
                color = FINGER_COLORS['ring']
            else:
                color = FINGER_COLORS['pinky']
            cv2.line(img, pt1, pt2, color, line_thickness)

    for i in range(21):
        idx = i + hand_offset
        if kpts[idx, 2] > 0:
            x, y = int(kpts[idx, 0]), int(kpts[idx, 1])
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
            cv2.circle(img, (x, y), point_radius, color, -1)
            cv2.circle(img, (x, y), point_radius, (255, 255, 255), 1)
            if show_indices:
                cv2.putText(img, str(i), (x + point_radius + 2, y + 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1,
                           cv2.LINE_AA)
    return img


def draw_hand_skeleton_21(img: np.ndarray, kpts_21: np.ndarray,
                          line_thickness: int = 2, point_radius: int = 4,
                          show_indices: bool = True) -> np.ndarray:
    """
    在图像上绘制 21 个手部关键点骨架 (通用, kpts_21 为 (21, 2) 或 (21, 3))

    适用于 GT crop 或 MediaPipe 输出
    """
    for conn_idx, (start, end) in enumerate(HAND_SKELETON_OFFSETS):
        # 检查有效性
        if kpts_21.shape[1] == 3:
            if kpts_21[start, 2] <= 0 or kpts_21[end, 2] <= 0:
                continue
        pt1 = (int(kpts_21[start, 0]), int(kpts_21[start, 1]))
        pt2 = (int(kpts_21[end, 0]), int(kpts_21[end, 1]))
        if conn_idx < 4:
            color = FINGER_COLORS['thumb']
        elif conn_idx < 8:
            color = FINGER_COLORS['index']
        elif conn_idx < 12:
            color = FINGER_COLORS['middle']
        elif conn_idx < 16:
            color = FINGER_COLORS['ring']
        else:
            color = FINGER_COLORS['pinky']
        cv2.line(img, pt1, pt2, color, line_thickness)

    for i in range(21):
        if kpts_21.shape[1] == 3 and kpts_21[i, 2] <= 0:
            continue
        x, y = int(kpts_21[i, 0]), int(kpts_21[i, 1])
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
        cv2.circle(img, (x, y), point_radius, color, -1)
        cv2.circle(img, (x, y), point_radius, (255, 255, 255), 1)
        if show_indices:
            cv2.putText(img, str(i), (x + point_radius + 2, y + 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1,
                       cv2.LINE_AA)
    return img


def draw_mp_skeleton_21(img: np.ndarray, kpts_21: np.ndarray,
                        line_thickness: int = 2, point_radius: int = 4) -> np.ndarray:
    """
    绘制 MediaPipe 检测结果的手部骨架 (统一用橙色区分于 GT)
    """
    for start, end in HAND_SKELETON_OFFSETS:
        pt1 = (int(kpts_21[start, 0]), int(kpts_21[start, 1]))
        pt2 = (int(kpts_21[end, 0]), int(kpts_21[end, 1]))
        cv2.line(img, pt1, pt2, MP_COLOR, line_thickness)

    for i in range(21):
        x, y = int(kpts_21[i, 0]), int(kpts_21[i, 1])
        cv2.circle(img, (x, y), point_radius, MP_COLOR, -1)
        cv2.circle(img, (x, y), point_radius, (255, 255, 255), 1)
    return img


def draw_hand_bbox_on_img(img: np.ndarray, bbox: Tuple[int, int, int, int],
                          label: str = "hand", color: tuple = LHAND_BBOX_COLOR,
                          thickness: int = 2) -> np.ndarray:
    """绘制手部 bbox (xyxy 格式)"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(img, label, (x1, y1 - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img


def draw_bbox(img: np.ndarray, bbox: list, label: str = "person",
              color: tuple = BBOX_COLOR, thickness: int = 2) -> np.ndarray:
    """绘制边界框 (COCO 格式: [x, y, w, h])"""
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    cv2.putText(img, label, (x, y - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img


def draw_all_annotations(img: np.ndarray, ann: dict,
                         draw_body: bool = True,
                         draw_face: bool = True,
                         draw_hands: bool = True,
                         draw_box: bool = True,
                         draw_hand_bbox: bool = True) -> np.ndarray:
    """
    在图像上绘制一个 person annotation 的所有标注 (含手部 bbox)
    """
    img_draw = img.copy()
    kpts = parse_keypoints(ann['keypoints'])

    # Person BBox
    if draw_box and 'bbox' in ann:
        draw_bbox(img_draw, ann['bbox'])

    # Body skeleton
    if draw_body:
        draw_body_skeleton(img_draw, kpts)

    # Face keypoints
    if draw_face:
        draw_face_keypoints(img_draw, kpts)

    # Left hand
    if draw_hands:
        draw_hand_skeleton_on_img(img_draw, kpts, hand_offset=94,
                                  line_thickness=2, point_radius=2)
    # Right hand
    if draw_hands:
        draw_hand_skeleton_on_img(img_draw, kpts, hand_offset=115,
                                  line_thickness=2, point_radius=2)

    # Hand bboxes (从关键点外接框)
    if draw_hand_bbox:
        lhand_bbox = get_hand_bbox(kpts, hand_offset=94, padding_ratio=0.15)
        rhand_bbox = get_hand_bbox(kpts, hand_offset=115, padding_ratio=0.15)
        if lhand_bbox is not None:
            lhand_area_ratio = _bbox_area_ratio(lhand_bbox, img_draw.shape)
            draw_hand_bbox_on_img(img_draw, lhand_bbox,
                                  label=f"LHand {lhand_area_ratio:.1f}%",
                                  color=LHAND_BBOX_COLOR)
        if rhand_bbox is not None:
            rhand_area_ratio = _bbox_area_ratio(rhand_bbox, img_draw.shape)
            draw_hand_bbox_on_img(img_draw, rhand_bbox,
                                  label=f"RHand {rhand_area_ratio:.1f}%",
                                  color=RHAND_BBOX_COLOR)

    return img_draw


def _bbox_area_ratio(bbox: Tuple, img_shape: Tuple) -> float:
    """计算 bbox xyxy 面积占图像面积百分比"""
    x1, y1, x2, y2 = bbox
    h, w = img_shape[:2]
    return (x2 - x1) * (y2 - y1) / (w * h) * 100


# ============================================================
# 图像路径
# ============================================================

def get_image_path(img_info: dict, data_root: str, split: str) -> str:
    """
    获取图像路径

    训练集图像来自 HICO-Det (hico_20160224_det/images/train2015/)
    验证集图像来自 COCO val2017 (val2017/)
    """
    file_name = img_info['file_name']
    if split == "train":
        img_path = os.path.join(data_root, "hico_20160224_det", "images",
                                "train2015", file_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(data_root, "hico_20160224_det", "images",
                                    file_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(data_root, file_name)
    else:
        img_path = os.path.join(data_root, "val2017", file_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(data_root, file_name)
    return img_path


# ============================================================
# 分层采样: 按手部大小 bin 采样
# ============================================================

def stratified_sample_hands(hand_infos: List[Dict],
                            n_per_bin: int = 3) -> Dict[str, List[Dict]]:
    """
    按手部大小 bin 分层采样

    Returns:
        { bin_name: [hand_info, ...] }
    """
    bins = defaultdict(list)
    for h in hand_infos:
        cat = categorize_hand_by_size(h['area_ratio'])
        bins[cat].append(h)

    sampled = {}
    for name, _, _ in HAND_SIZE_BINS:
        pool = bins.get(name, [])
        if not pool:
            sampled[name] = []
            continue
        k = min(n_per_bin, len(pool))
        sampled[name] = random.sample(pool, k=k)
    return sampled


# ============================================================
# EDA 主函数
# ============================================================

def eda_split(data_root: str, split: str = "val",
              n_samples: int = 10, n_per_bin: int = 3):
    """
    对指定 split 进行 EDA 可视化

    流程:
      1. 加载标注 & 计算基本统计
      2. 扫描所有手部 bbox, 计算手部大小分布
      3. 绘制分布直方图
      4. 随机采样 n_samples 张图, 输出 GT 标注 (含手部 bbox)
      5. 分层采样: 从 tiny/small/medium/large 各 bin 中选取样本,
         对 medium/large 执行 MediaPipe 手部关键点检测, 输出对比可视化

    Args:
        data_root: Halpe 数据根目录
        split: "train" 或 "val"
        n_samples: 基础采样数量 (全图标注可视化)
        n_per_bin: 每个手部大小 bin 采样数量 (手部 crop 可视化)
    """
    # 加载标注
    if split == "train":
        anno_file = os.path.join(data_root, "annotations", "halpe_train_v1.json")
    else:
        anno_file = os.path.join(data_root, "annotations", "halpe_val_v1.json")

    data = load_annotations(anno_file)

    # 基本统计
    stats = compute_statistics(data)
    print_statistics(stats, split)

    # 构建映射
    img_id_to_info = build_img_id_to_info(data)
    img_id_to_anns = build_img_id_to_anns(data)
    ann_id_to_ann = {}
    for ann in data['annotations']:
        ann_id_to_ann[ann.get('id', id(ann))] = ann

    # -------------------------------------------------------
    # 手部大小分布分析
    # -------------------------------------------------------
    print("\nComputing hand size distribution ...")
    hand_infos = compute_all_hand_infos(data, img_id_to_info)
    print_hand_size_stats(hand_infos, split)

    # 输出目录
    eda_output_path = os.path.join(data_root, "eda_output", split)
    if os.path.exists(eda_output_path):
        shutil.rmtree(eda_output_path)
    os.makedirs(eda_output_path, exist_ok=True)

    # 绘制分布直方图
    dist_plot_path = os.path.join(eda_output_path, "hand_size_distribution.png")
    plot_hand_size_distribution(hand_infos, dist_plot_path, split)

    # -------------------------------------------------------
    # Part 1: 全图标注可视化 (随机采样 n_samples 张图)
    # -------------------------------------------------------
    print(f"\n--- Part 1: Full image annotation visualization ({n_samples} samples) ---")
    samples_dir = os.path.join(eda_output_path, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    image_ids = list(img_id_to_anns.keys())
    selected_ids = random.sample(image_ids, k=min(n_samples, len(image_ids)))

    for idx, img_id in enumerate(selected_ids):
        img_info = img_id_to_info[img_id]
        anns = img_id_to_anns[img_id]

        img_path = get_image_path(img_info, data_root, split)
        if not os.path.exists(img_path):
            print(f"  [SKIP] Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # 绘制所有标注 (含手部 bbox)
        img_annotated = img.copy()
        for ann in anns:
            img_annotated = draw_all_annotations(img_annotated, ann)

        # 2 列子图: 原图 + 标注
        fig, axs = plt.subplots(1, 2, figsize=(20, 8))

        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0].set_title(f"Original ({w}x{h})")
        axs[0].axis('off')

        axs[1].imshow(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB))
        axs[1].set_title(f"GT Annotations ({len(anns)} person(s))\n"
                         f"+ Hand BBox (from keypoints)")
        axs[1].axis('off')

        plt.suptitle(f"Halpe {split} - {img_info['file_name']}", fontsize=14)
        plt.tight_layout()

        save_name = os.path.splitext(img_info['file_name'])[0] + ".jpg"
        save_path = os.path.join(samples_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [{idx}] Saved: {save_path}")

    # -------------------------------------------------------
    # Part 2: 按手部大小 bin 分层采样 + MediaPipe 对比
    # -------------------------------------------------------
    print(f"\n--- Part 2: Stratified hand size visualization ({n_per_bin} per bin) ---")

    sampled_bins = stratified_sample_hands(hand_infos, n_per_bin=n_per_bin)

    # 检查 MediaPipe 模型是否可用
    mp_available = os.path.exists(_HAND_LANDMARKER_MODEL_PATH)
    if not mp_available:
        print(f"  [WARN] MediaPipe model not found: {_HAND_LANDMARKER_MODEL_PATH}")
        print(f"         Will skip MediaPipe inference.")

    for bin_name, _, _ in HAND_SIZE_BINS:
        bin_samples = sampled_bins.get(bin_name, [])
        if not bin_samples:
            print(f"\n  [{bin_name}] No samples in this bin, skipping.")
            continue

        bin_dir = os.path.join(eda_output_path, f"hand_size_{bin_name}")
        os.makedirs(bin_dir, exist_ok=True)

        print(f"\n  [{bin_name}] Visualizing {len(bin_samples)} hand(s) ...")

        for hi, hand_info in enumerate(bin_samples):
            img_id = hand_info['image_id']
            img_info = img_id_to_info[img_id]
            hand_side = hand_info['hand_side']
            hand_offset = 94 if hand_side == 'left' else 115
            area_ratio = hand_info['area_ratio']
            max_side = hand_info['max_side']
            n_vis = hand_info['n_visible']

            img_path = get_image_path(img_info, data_root, split)
            if not os.path.exists(img_path):
                print(f"    [SKIP] Image not found: {img_path}")
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            img_h, img_w = img.shape[:2]

            # 找到对应的 annotation
            anns = img_id_to_anns[img_id]
            target_ann = None
            for ann in anns:
                kpts = parse_keypoints(ann['keypoints'])
                bbox_check = get_hand_bbox_no_padding(kpts, hand_offset, min_visible=3)
                if bbox_check is not None:
                    bx1, by1, bx2, by2 = bbox_check
                    hx1, hy1, hx2, hy2 = hand_info['hand_bbox']
                    if abs(bx1 - hx1) < 1 and abs(by1 - hy1) < 1:
                        target_ann = ann
                        break
            if target_ann is None:
                # fallback: 取第一个有该手标注的
                for ann in anns:
                    kpts = parse_keypoints(ann['keypoints'])
                    if get_hand_bbox_no_padding(kpts, hand_offset, min_visible=3) is not None:
                        target_ann = ann
                        break
            if target_ann is None:
                continue

            kpts = parse_keypoints(target_ann['keypoints'])

            # 手部 bbox (带 padding, 用于 crop)
            hand_bbox_padded = get_hand_bbox(kpts, hand_offset, padding_ratio=0.3)
            if hand_bbox_padded is None:
                continue
            cx1, cy1, cx2, cy2 = hand_bbox_padded
            # 限制在图像范围内
            cx1 = max(0, cx1)
            cy1 = max(0, cy1)
            cx2 = min(img_w, cx2)
            cy2 = min(img_h, cy2)

            crop_w = cx2 - cx1
            crop_h = cy2 - cy1
            if crop_w < 5 or crop_h < 5:
                continue

            # 手部 crop
            hand_crop = img[cy1:cy2, cx1:cx2].copy()

            # GT 关键点在 crop 坐标系中 (转为 float 以便后续缩放)
            hand_kpts_full = kpts[hand_offset:hand_offset + 21].astype(np.float64)
            hand_kpts_crop = hand_kpts_full.copy()
            hand_kpts_crop[:, 0] -= cx1
            hand_kpts_crop[:, 1] -= cy1

            # --- 子图 1: 全图 + 手部 bbox 高亮 ---
            img_with_bbox = img.copy()
            # 画所有标注 (轻量)
            for ann in anns:
                k = parse_keypoints(ann['keypoints'])
                draw_body_skeleton(img_with_bbox, k, line_thickness=1, point_radius=2)
            # 高亮当前手部 bbox
            hand_bbox_vis = get_hand_bbox(kpts, hand_offset, padding_ratio=0.15)
            if hand_bbox_vis:
                bbox_color = LHAND_BBOX_COLOR if hand_side == 'left' else RHAND_BBOX_COLOR
                draw_hand_bbox_on_img(img_with_bbox, hand_bbox_vis,
                                      label=f"{hand_side.upper()} {area_ratio:.2f}%",
                                      color=bbox_color, thickness=3)

            # --- 子图 2: 手部 crop + GT skeleton ---
            crop_gt = hand_crop.copy()
            # 放大 crop 以便看清
            scale = max(1, 256 / max(crop_w, crop_h))
            if scale > 1:
                crop_gt_resized = cv2.resize(crop_gt, None, fx=scale, fy=scale,
                                             interpolation=cv2.INTER_LINEAR)
                kpts_scaled = hand_kpts_crop.copy()
                kpts_scaled[:, :2] *= scale
            else:
                crop_gt_resized = crop_gt.copy()
                kpts_scaled = hand_kpts_crop.copy()
            draw_hand_skeleton_21(crop_gt_resized, kpts_scaled,
                                  line_thickness=2, point_radius=4, show_indices=True)

            # --- 子图 3: 手部 crop + MediaPipe ---
            run_mp = mp_available and (bin_name in ("medium", "large"))
            mp_detected = False
            if run_mp:
                # 对 crop 做 resize 以提高 MediaPipe 检出率
                mp_input = hand_crop.copy()
                mp_scale = max(1, 256 / max(crop_w, crop_h))
                if mp_scale > 1:
                    mp_input = cv2.resize(mp_input, None, fx=mp_scale, fy=mp_scale,
                                          interpolation=cv2.INTER_LINEAR)
                mp_kpts = run_mediapipe_inference(mp_input)
                if mp_kpts is not None:
                    crop_mp = mp_input.copy()
                    draw_mp_skeleton_21(crop_mp, mp_kpts,
                                        line_thickness=2, point_radius=4)
                    mp_detected = True
                else:
                    crop_mp = mp_input.copy()
                    cv2.putText(crop_mp, "No hand detected",
                               (5, crop_mp.shape[0] // 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                crop_mp = None

            # --- 子图 4: GT vs MediaPipe 叠加 (仅 medium/large) ---
            if run_mp:
                crop_compare = mp_input.copy() if mp_scale > 1 else hand_crop.copy()
                kpts_for_compare = hand_kpts_crop.copy()
                if mp_scale > 1:
                    kpts_for_compare[:, :2] *= mp_scale
                draw_hand_skeleton_21(crop_compare, kpts_for_compare,
                                      line_thickness=2, point_radius=3,
                                      show_indices=False)
                if mp_detected:
                    draw_mp_skeleton_21(crop_compare, mp_kpts,
                                        line_thickness=1, point_radius=3)

            # --- 绘制 figure ---
            if run_mp:
                fig, axs = plt.subplots(1, 4, figsize=(28, 6))
            else:
                fig, axs = plt.subplots(1, 2, figsize=(16, 6))

            # 全图
            axs[0].imshow(cv2.cvtColor(img_with_bbox, cv2.COLOR_BGR2RGB))
            axs[0].set_title(f"Full Image ({img_w}x{img_h})\n"
                             f"{hand_side} hand bbox: {area_ratio:.2f}%, "
                             f"max_side={max_side:.0f}px")
            axs[0].axis('off')

            # GT crop
            axs[1].imshow(cv2.cvtColor(crop_gt_resized, cv2.COLOR_BGR2RGB))
            axs[1].set_title(f"GT Hand Crop ({crop_w}x{crop_h})\n"
                             f"{n_vis}/21 visible keypoints")
            axs[1].axis('off')

            if run_mp:
                # MediaPipe crop
                axs[2].imshow(cv2.cvtColor(crop_mp, cv2.COLOR_BGR2RGB))
                mp_title = "MediaPipe Detection" if mp_detected else "MediaPipe (Failed)"
                axs[2].set_title(mp_title)
                axs[2].axis('off')

                # GT vs MP comparison
                axs[3].imshow(cv2.cvtColor(crop_compare, cv2.COLOR_BGR2RGB))
                axs[3].set_title("GT (color) vs MP (orange)" if mp_detected
                                 else "GT only (MP failed)")
                axs[3].axis('off')

            suptitle = (f"Halpe {split} - {bin_name.upper()} hand - "
                        f"{img_info['file_name']} - {hand_side}")
            plt.suptitle(suptitle, fontsize=13)
            plt.tight_layout()

            save_name = (f"{bin_name}_{hi:02d}_{hand_side}_"
                         f"{os.path.splitext(img_info['file_name'])[0]}.jpg")
            save_path = os.path.join(bin_dir, save_name)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    [{bin_name}][{hi}] {hand_side} hand, "
                  f"ratio={area_ratio:.2f}%, max_side={max_side:.0f}px"
                  f"{', MP=' + ('OK' if mp_detected else 'FAIL') if run_mp else ''}"
                  f" -> {save_path}")

    # -------------------------------------------------------
    # 完成
    # -------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"EDA for {split} split done!")
    print(f"  Output dir: {eda_output_path}")
    print(f"  - hand_size_distribution.png  (分布直方图)")
    print(f"  - samples/                    (全图标注可视化)")
    for name, _, _ in HAND_SIZE_BINS:
        print(f"  - hand_size_{name}/            "
              f"({name} 手部 crop + GT"
              f"{' + MediaPipe' if name in ('medium', 'large') else ''})")
    print(f"{'='*60}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Halpe Full-Body 136 Keypoint Dataset - EDA Visualization"
    )
    parser.add_argument("--data_root", type=str,
                        default="/home/models/rightArm_collection/Halpe/Halpe",
                        help="Halpe 数据集根目录")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val"],
                        help="数据集 split (default: val)")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="全图标注可视化采样数量 (default: 10)")
    parser.add_argument("--n_per_bin", type=int, default=3,
                        help="每个手部大小 bin 的采样数量 (default: 3)")
    args = parser.parse_args()

    eda_split(data_root=args.data_root, split=args.split,
              n_samples=args.n_samples, n_per_bin=args.n_per_bin)
