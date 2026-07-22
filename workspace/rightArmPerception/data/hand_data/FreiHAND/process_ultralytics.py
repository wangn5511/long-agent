"""
FreiHAND Dataset -> Ultralytics Pose Format Converter
======================================================

将 FreiHAND 数据集转换为 Ultralytics YOLO-Pose 格式。

Ultralytics Pose 标签格式 (每行):
    <class_id> <cx> <cy> <w> <h> <x1> <y1> <v1> <x2> <y2> <v2> ... <x21> <y21> <v21>
    
    - class_id: 类别ID (0 = hand)
    - cx, cy, w, h: 归一化的边界框中心坐标和宽高 [0, 1]
    - xi, yi: 归一化的关键点坐标 [0, 1]
    - vi: 关键点可见性 (0=不标注, 1=标注但不可见, 2=标注且可见)

输出目录结构:
    output_dir/
    ├── images/
    │   ├── train/
    │   │   ├── 00000000.jpg
    │   │   └── ...
    │   └── val/
    │       └── ...
    ├── labels/
    │   ├── train/
    │   │   ├── 00000000.txt
    │   │   └── ...
    │   └── val/
    │       └── ...
    └── dataset.yaml

Usage:
    python process_ultralytics.py --data_root /path/to/FreiHAND --output_dir ./freihand_ultralytics
    
    # 只处理原始样本 (不含增强)
    python process_ultralytics.py --data_root /path/to/FreiHAND --output_dir ./output --no-augmented
    
    # 使用符号链接而不是复制图像
    python process_ultralytics.py --data_root /path/to/FreiHAND --output_dir ./output --symlink
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from typing import Optional, Tuple, List
from tqdm import tqdm
import numpy as np
import cv2


# ==================== 常量定义 ====================

# 默认图像尺寸
IMAGE_SIZE = 224

# 21个手部关键点名称 (与 MediaPipe 一致)
KEYPOINT_NAMES = [
    'wrist',
    'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
    'index_mcp', 'index_pip', 'index_dip', 'index_tip',
    'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
    'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
    'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
]

NUM_KEYPOINTS = 21


# ==================== 工具函数 ====================

def load_json(path: str) -> list:
    """加载 JSON 文件"""
    with open(path, 'r') as f:
        return json.load(f)


def project_3d_to_2d(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    将3D坐标投影到2D图像平面
    
    Args:
        xyz: (21, 3) - 21个手部关键点的3D坐标
        K: (3, 3) - 相机内参矩阵
    
    Returns:
        (21, 2) - 2D关键点坐标 (像素坐标)
    """
    uv = K @ xyz.T
    uv = uv[:2, :] / uv[2:, :]
    return uv.T


def get_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    从mask图像中获取边界框
    
    Args:
        mask: mask图像 (可以是灰度或3通道)
    
    Returns:
        (x_min, y_min, x_max, y_max) 或 None
    """
    # 如果是3通道图像，转换为灰度图
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask
    
    coords = np.column_stack(np.where(mask_gray > 128))
    if coords.size == 0:
        return None
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return (x_min, y_min, x_max, y_max)


def get_bbox_from_keypoints(uv: np.ndarray, padding: float = 0.1, 
                            img_size: int = IMAGE_SIZE) -> Tuple[int, int, int, int]:
    """
    从2D关键点计算边界框
    
    Args:
        uv: (21, 2) - 2D关键点坐标
        padding: 边界框扩展比例
        img_size: 图像尺寸
    
    Returns:
        (x_min, y_min, x_max, y_max)
    """
    x_min, y_min = uv.min(axis=0)
    x_max, y_max = uv.max(axis=0)
    
    # 添加 padding
    width = x_max - x_min
    height = y_max - y_min
    x_min -= width * padding
    y_min -= height * padding
    x_max += width * padding
    y_max += height * padding
    
    # 裁剪到图像边界
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_size - 1, x_max)
    y_max = min(img_size - 1, y_max)
    
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def bbox_to_yolo(bbox: Tuple[int, int, int, int], 
                 img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    将边界框转换为 YOLO 格式 (归一化的中心坐标和宽高)
    
    Args:
        bbox: (x_min, y_min, x_max, y_max)
        img_width, img_height: 图像尺寸
    
    Returns:
        (cx, cy, w, h) - 归一化到 [0, 1]
    """
    x_min, y_min, x_max, y_max = bbox
    
    cx = (x_min + x_max) / 2 / img_width
    cy = (y_min + y_max) / 2 / img_height
    w = (x_max - x_min) / img_width
    h = (y_max - y_min) / img_height
    
    return (cx, cy, w, h)


def format_yolo_pose_label(class_id: int, 
                           bbox: Tuple[float, float, float, float],
                           keypoints: np.ndarray,
                           visibility: int = 2) -> str:
    """
    生成 Ultralytics Pose 格式的标签行
    
    Args:
        class_id: 类别ID
        bbox: (cx, cy, w, h) 归一化的边界框
        keypoints: (21, 2) 归一化的关键点坐标
        visibility: 所有关键点的可见性 (1=标注但不可见)
    
    Returns:
        标签行字符串
    """
    cx, cy, w, h = bbox
    
    # 构建标签: class_id cx cy w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...
    parts = [str(class_id), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
    
    for i in range(len(keypoints)):
        kp_x, kp_y = keypoints[i]
        parts.extend([f"{kp_x:.6f}", f"{kp_y:.6f}", str(visibility)])
    
    return " ".join(parts)


# ==================== 主处理类 ====================

class FreiHANDToUltralytics:
    """FreiHAND 数据集转 Ultralytics 格式"""
    
    def __init__(self, data_root: str, output_dir: str, 
                 include_augmented: bool = True,
                 use_symlink: bool = False,
                 visibility: int = 2):
        """
        初始化转换器
        
        Args:
            data_root: FreiHAND 数据集根目录
            output_dir: 输出目录
            include_augmented: 是否包含增强数据 (训练集)
            use_symlink: 是否使用符号链接而不是复制图像
            visibility: 关键点可见性标注 (默认1)
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.include_augmented = include_augmented
        self.use_symlink = use_symlink
        self.visibility = visibility
        
        # 创建输出目录
        self.images_train_dir = self.output_dir / "images" / "train"
        self.images_val_dir = self.output_dir / "images" / "val"
        self.labels_train_dir = self.output_dir / "labels" / "train"
        self.labels_val_dir = self.output_dir / "labels" / "val"
        
        for d in [self.images_train_dir, self.images_val_dir, 
                  self.labels_train_dir, self.labels_val_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        self.stats = {
            'train_total': 0,
            'train_with_mask': 0,
            'train_without_mask': 0,
            'val_total': 0,
            'val_with_mask': 0,
        }
    
    def load_annotations(self, split: str) -> Tuple[list, list]:
        """
        加载标注数据
        
        Args:
            split: 'training' 或 'evaluation'
        
        Returns:
            (xyz_list, K_list)
        """
        xyz_path = self.data_root / f"{split}_xyz.json"
        K_path = self.data_root / f"{split}_K.json"
        
        if not xyz_path.exists() or not K_path.exists():
            raise FileNotFoundError(f"Missing annotation files for {split}")
        
        xyz_list = load_json(str(xyz_path))
        K_list = load_json(str(K_path))
        
        print(f"Loaded {len(xyz_list)} annotations for {split}")
        return xyz_list, K_list
    
    def get_mask_path(self, split: str, idx: int, original_idx: int) -> Optional[Path]:
        """
        获取 mask 文件路径
        
        Args:
            split: 'training' 或 'evaluation'
            idx: 图像索引
            original_idx: 原始标注索引 (用于增强图像)
        
        Returns:
            mask 路径或 None
        """
        if split == 'training':
            # 训练集 mask 是 .jpg 格式
            mask_path = self.data_root / split / "mask" / f"{original_idx:08d}.jpg"
        else:
            # 评估集 mask 在 segmap/ 目录，是 .png 格式
            mask_path = self.data_root / split / "segmap" / f"{idx:08d}.png"
        
        if mask_path.exists():
            return mask_path
        return None
    
    def process_sample(self, split: str, idx: int, 
                       xyz_list: list, K_list: list,
                       num_original: int) -> Optional[str]:
        """
        处理单个样本
        
        Args:
            split: 'training' 或 'evaluation'
            idx: 图像索引
            xyz_list, K_list: 标注数据
            num_original: 原始样本数量
        
        Returns:
            标签行字符串，或 None (如果处理失败)
        """
        # 确定标注索引
        if split == 'training':
            anno_idx = idx % num_original
        else:
            anno_idx = idx
        
        # 加载 3D 关键点和相机参数
        xyz = np.array(xyz_list[anno_idx])
        K = np.array(K_list[anno_idx])
        
        # 投影到 2D
        uv = project_3d_to_2d(xyz, K)
        uv = np.clip(uv, 0, IMAGE_SIZE - 1)
        
        # 获取 mask 路径
        mask_path = self.get_mask_path(split, idx, anno_idx)
        
        # 获取边界框
        if mask_path is not None:
            mask = cv2.imread(str(mask_path))
            bbox = get_bbox_from_mask(mask)
            if bbox is None:
                bbox = get_bbox_from_keypoints(uv)
            if split == 'training':
                self.stats['train_with_mask'] += 1
            else:
                self.stats['val_with_mask'] += 1
        else:
            bbox = get_bbox_from_keypoints(uv)
            if split == 'training':
                self.stats['train_without_mask'] += 1
        
        # 转换为 YOLO 格式
        yolo_bbox = bbox_to_yolo(bbox, IMAGE_SIZE, IMAGE_SIZE)
        
        # 归一化关键点
        kp_normalized = uv / IMAGE_SIZE
        
        # 生成标签行
        label_line = format_yolo_pose_label(
            class_id=0,
            bbox=yolo_bbox,
            keypoints=kp_normalized,
            visibility=self.visibility
        )
        
        return label_line
    
    def copy_or_link_image(self, src: Path, dst: Path):
        """复制或创建符号链接"""
        if dst.exists():
            return
        
        if self.use_symlink:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
    
    def process_split(self, split: str):
        """
        处理一个数据集分割
        
        Args:
            split: 'training' 或 'evaluation'
        """
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print('='*60)
        
        # 加载标注
        xyz_list, K_list = self.load_annotations(split)
        num_original = len(xyz_list)
        
        # 确定图像目录和输出目录
        rgb_dir = self.data_root / split / "rgb"
        if split == 'training':
            images_out_dir = self.images_train_dir
            labels_out_dir = self.labels_train_dir
        else:
            images_out_dir = self.images_val_dir
            labels_out_dir = self.labels_val_dir
        
        # 获取图像列表
        image_files = sorted(rgb_dir.glob("*.jpg"))
        
        if split == 'training' and not self.include_augmented:
            # 只处理原始样本
            image_files = [f for f in image_files 
                          if int(f.stem) < num_original]
            print(f"Processing only original samples: {len(image_files)}")
        else:
            print(f"Processing all images: {len(image_files)}")
        
        # 处理每个样本
        for img_path in tqdm(image_files, desc=f"Processing {split}"):
            idx = int(img_path.stem)
            
            # 处理样本
            label_line = self.process_sample(split, idx, xyz_list, K_list, num_original)
            
            if label_line is None:
                continue
            
            # 保存标签
            label_path = labels_out_dir / f"{idx:08d}.txt"
            with open(label_path, 'w') as f:
                f.write(label_line + "\n")
            
            # 复制/链接图像
            img_dst = images_out_dir / img_path.name
            self.copy_or_link_image(img_path, img_dst)
            
            if split == 'training':
                self.stats['train_total'] += 1
            else:
                self.stats['val_total'] += 1
    
    def generate_dataset_yaml(self):
        """生成 dataset.yaml 配置文件"""
        yaml_content = f"""# FreiHAND Hand Pose Dataset - Ultralytics Format
# Generated by process_ultralytics.py

path: {self.output_dir.resolve()}
train: images/train
val: images/val

# Keypoints configuration
kpt_shape: [{NUM_KEYPOINTS}, 3]  # [num_keypoints, (x, y, visibility)]

# Classes
names:
  0: hand

# Number of keypoints
# Keypoint names (MediaPipe compatible):
# {KEYPOINT_NAMES}

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
        
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\nGenerated dataset.yaml: {yaml_path}")
    
    def print_stats(self):
        """打印统计信息"""
        print("\n" + "="*60)
        print("Processing Statistics")
        print("="*60)
        print(f"Training samples: {self.stats['train_total']}")
        print(f"  - With mask: {self.stats['train_with_mask']}")
        print(f"  - Without mask (bbox from keypoints): {self.stats['train_without_mask']}")
        print(f"Validation samples: {self.stats['val_total']}")
        print(f"  - With mask: {self.stats['val_with_mask']}")
        print(f"\nOutput directory: {self.output_dir.resolve()}")
    
    def run(self):
        """执行完整的转换流程"""
        print("FreiHAND -> Ultralytics Pose Format Converter")
        print(f"Data root: {self.data_root}")
        print(f"Output dir: {self.output_dir}")
        print(f"Include augmented: {self.include_augmented}")
        print(f"Use symlink: {self.use_symlink}")
        print(f"Visibility: {self.visibility}")
        
        # 处理训练集
        self.process_split('training')
        
        # 处理评估集 (作为验证集)
        self.process_split('evaluation')
        
        # 生成配置文件
        self.generate_dataset_yaml()
        
        # 打印统计
        self.print_stats()


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        description='Convert FreiHAND dataset to Ultralytics Pose format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 基本用法
    python process_ultralytics.py --data_root /path/to/FreiHAND --output_dir ./output
    
    # 只处理原始样本 (不含4倍增强)
    python process_ultralytics.py --data_root /path/to/FreiHAND --output_dir ./output --no-augmented
    
    # 使用符号链接节省空间
    python process_ultralytics.py --data_root /path/to/FreiHAND --output_dir ./output --symlink
    
    # 设置可见性为2 (标注且可见)
    python process_ultralytics.py --data_root /path/to/FreiHAND --output_dir ./output --visibility 2
        """
    )
    
    parser.add_argument('--data_root', type=str, required=True,
                        help='FreiHAND 数据集根目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--no-augmented', action='store_true',
                        help='不包含增强数据 (只处理原始32560个样本)')
    parser.add_argument('--symlink', action='store_true',
                        help='使用符号链接而不是复制图像')
    parser.add_argument('--visibility', type=int, default=2, choices=[0, 1, 2],
                        help='关键点可见性 (0=不标注, 1=标注但不可见, 2=标注且可见)')
    
    args = parser.parse_args()
    
    converter = FreiHANDToUltralytics(
        data_root=args.data_root,
        output_dir=args.output_dir,
        include_augmented=not args.no_augmented,
        use_symlink=args.symlink,
        visibility=args.visibility
    )
    
    converter.run()


if __name__ == '__main__':
    main()
