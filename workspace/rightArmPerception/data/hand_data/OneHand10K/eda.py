import os
import cv2
import numpy as np
import glob
import random
import shutil
from typing import Optional
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import matplotlib.pyplot as plt

# 禁用 GPU，只使用 CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 使用 TensorFlow 来读取 TFRecords
import tensorflow as tf


# MediaPipe HandLandmarker 模型路径 (与 FreiHAND 共用)
_HAND_LANDMARKER_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "FreiHAND", "hand_landmarker.task"
)


# ============================================================
# OneHand10K 数据集信息 (来自 Kaggle TFRecords 版本)
# ============================================================
# 数据集: OneHand10K - 单手 21 关键点 2D 手部姿态估计
#
# 关键点顺序 (与 MediaPipe 一致):
#   参考: https://www.kaggle.com/code/rustyelectron/inspect-onehand10k-tfrecords
#
#   OneHand10K    MediaPipe             名称
#   ---------    ---------    -----------------------
#        0           0        Wrist (手腕)
#        1           1        Thumb CMC (拇指掌腕关节)
#        2           2        Thumb MCP (拇指掌指关节)
#        3           3        Thumb IP  (拇指指间关节)
#        4           4        Thumb TIP (拇指指尖)
#        5           5        Index MCP (食指掌指关节)
#        6           6        Index PIP (食指近端指间关节)
#        7           7        Index DIP (食指远端指间关节)
#        8           8        Index TIP (食指指尖)
#        9           9        Middle MCP
#       10          10        Middle PIP
#       11          11        Middle DIP
#       12          12        Middle TIP
#       13          13        Ring MCP
#       14          14        Ring PIP
#       15          15        Ring DIP
#       16          16        Ring TIP
#       17          17        Pinky MCP
#       18          18        Pinky PIP
#       19          19        Pinky DIP
#       20          20        Pinky TIP
#
#   映射关系: 1:1 恒等映射, 无需重排
#   ONEHAND10K_TO_MEDIAPIPE = list(range(21))
#
# 骨架连接 (POSE_PAIRS, 来自 Kaggle notebook, 与 MediaPipe HAND_CONNECTIONS 一致):
#   [[0,1],[1,2],[2,3],[3,4],            # Thumb
#    [0,5],[5,6],[6,7],[7,8],            # Index
#    [0,9],[9,10],[10,11],[11,12],       # Middle
#    [0,13],[13,14],[14,15],[15,16],     # Ring
#    [0,17],[17,18],[18,19],[19,20]]     # Pinky
#
# TFRecords 实际结构 (通过 inspect_tfrecord 发现):
#   - image_raw: bytes (196608 = 256*256*3, 原始像素数据 RGB)
#   - image_mask: bytes (65536 = 256*256, 单通道 mask)
#   - label: float_list (42 = 21 keypoints * 2)
#            交错排列: [x0, y0, x1, y1, ..., x20, y20]
#            坐标为 [-0.5, 0.5] 中心归一化格式
#            反归一化: pixel = (value + 0.5) * 256
#            无效关键点原始值远 < -0.5 (如 -1),
#            反归一化后 pixel < 0, 用于标记无效
# ============================================================

# 图像固定尺寸
IMAGE_SIZE = 256


# MediaPipe 风格的骨架连接 (与 FreiHAND 保持一致)
HAND_SKELETON = [
    # Thumb (红色)
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index (绿色)
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle (蓝色)
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring (黄色)
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky (紫色)
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
# TFRecord 解析
# ============================================================

def inspect_tfrecord(tfrecord_path: str):
    """
    检查 TFRecord 文件中第一条记录的 feature 结构
    用于自动发现 TFRecords 的 schema
    """
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        print(f"=== TFRecord Features in {os.path.basename(tfrecord_path)} ===")
        for key, feature in example.features.feature.items():
            kind = feature.WhichOneof('kind')
            if kind == 'bytes_list':
                vals = feature.bytes_list.value
                print(f"  {key}: bytes_list (len={len(vals)}, first_len={len(vals[0]) if vals else 0})")
            elif kind == 'float_list':
                vals = feature.float_list.value
                print(f"  {key}: float_list (len={len(vals)})")
                if len(vals) == 42:
                    # label: 打印前半 (x) 和后半 (y) 的范围
                    x_vals = list(vals[:21])
                    y_vals = list(vals[21:])
                    print(f"    x[0:5]  = {x_vals[:5]}")
                    print(f"    x[16:21]= {x_vals[16:]}")
                    print(f"    y[0:5]  = {y_vals[:5]}")
                    print(f"    y[16:21]= {y_vals[16:]}")
                else:
                    print(f"    values={list(vals[:5])}...")
            elif kind == 'int64_list':
                vals = feature.int64_list.value
                print(f"  {key}: int64_list (len={len(vals)}, values={list(vals)})")
        print()
        return example


def parse_tfrecord_fn(serialized_example):
    """
    解析单条 TFRecord 记录
    
    实际 schema (通过 inspect_tfrecord 发现):
      - image_raw: bytes (196608 = 256*256*3, raw RGB pixels)
      - image_mask: bytes (65536 = 256*256, single-channel mask)
      - label: float_list (42 = 21*2, interleaved [x0,y0,x1,y1,...])
    
    Returns:
        dict with keys: image, mask, label
    """
    feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'image_mask': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([42], tf.float32),
    }
    
    parsed = tf.io.parse_single_example(serialized_example, feature_description)
    
    # 解码图像: raw bytes -> (256, 256, 3) RGB
    image = tf.io.decode_raw(parsed['image_raw'], tf.uint8)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    
    # 解码 mask: raw bytes -> (256, 256) single channel
    mask = tf.io.decode_raw(parsed['image_mask'], tf.uint8)
    mask = tf.reshape(mask, [IMAGE_SIZE, IMAGE_SIZE])
    
    # label: 42 floats -> (21, 2) 关键点坐标
    label = parsed['label']
    
    return {
        'image': image,
        'mask': mask,
        'label': label,
    }


def load_tfrecord_dataset(tfrecord_dir: str, split: str = "train"):
    """
    加载指定 split 的所有 TFRecord 文件
    
    Args:
        tfrecord_dir: TFRecords 根目录
        split: "train", "valid", 或 "test"
    
    Returns:
        tf.data.Dataset
    """
    split_dir = os.path.join(tfrecord_dir, split)
    tfrecord_files = sorted(glob.glob(os.path.join(split_dir, "*.tfrecords")))
    
    if not tfrecord_files:
        raise FileNotFoundError(f"No TFRecord files found in {split_dir}")
    
    print(f"Found {len(tfrecord_files)} TFRecord files in {split_dir}")
    
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset


def load_samples(tfrecord_dir: str, split: str = "train", n_samples: int = 10):
    """
    从 TFRecord 中随机加载 n 个样本
    
    先收集所有样本，然后随机采样
    """
    dataset = load_tfrecord_dataset(tfrecord_dir, split)
    
    all_samples = []
    count = 0
    for sample in dataset:
        all_samples.append({
            'image': sample['image'].numpy(),     # (256, 256, 3) RGB uint8
            'mask': sample['mask'].numpy(),        # (256, 256) uint8
            'label': sample['label'].numpy(),      # (42,) float32
        })
        count += 1
    
    print(f"Total {count} samples in {split} split")
    
    if count <= n_samples:
        return all_samples
    
    selected = random.sample(all_samples, k=n_samples)
    return selected


# ============================================================
# MediaPipe 推理
# ============================================================

def run_mediapipe_inference(img: np.ndarray) -> Optional[np.ndarray]:
    """
    使用 MediaPipe Tasks API 进行手部关键点检测
    
    Args:
        img: BGR 格式的图像
    
    Returns:
        (21, 2) 的关键点坐标数组，如果检测失败则返回 None
    """
    base_options = mp_python.BaseOptions(
        model_asset_path=_HAND_LANDMARKER_MODEL_PATH
    )
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
    )
    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        # MediaPipe 需要 RGB 格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)
        
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            h, w = img.shape[:2]
            keypoints = np.array([
                [lm.x * w, lm.y * h] for lm in landmarks
            ])
            return keypoints
        return None


# ============================================================
# 可视化工具
# ============================================================

def draw_skeleton(img: np.ndarray, keypoints: np.ndarray, 
                  line_thickness: int = 2, point_radius: int = 4) -> np.ndarray:
    """
    在图像上绘制手部骨架
    
    Args:
        img: 原始图像 (BGR)
        keypoints: (21, 2) 的关键点坐标 [x, y]
        line_thickness: 骨架线宽
        point_radius: 关键点半径
    
    Returns:
        绘制了骨架的图像
    """
    img_copy = img.copy()
    
    # 绘制骨架连接
    for i, (start, end) in enumerate(HAND_SKELETON):
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
        
        pt1 = (int(keypoints[start, 0]), int(keypoints[start, 1]))
        pt2 = (int(keypoints[end, 0]), int(keypoints[end, 1]))
        cv2.line(img_copy, pt1, pt2, color, line_thickness)
    
    # 绘制关键点
    for i, (x, y) in enumerate(keypoints):
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
        
        cv2.circle(img_copy, (int(x), int(y)), point_radius, color, -1)
        cv2.circle(img_copy, (int(x), int(y)), point_radius, (255, 255, 255), 1)
    
    return img_copy


def get_bbox_from_mask(mask: np.ndarray) -> tuple:
    """
    从 mask 图像中提取边界框
    
    Args:
        mask: 单通道 mask 图像 (H, W), uint8
    
    Returns:
        (x_min, y_min, x_max, y_max) 或 None
    """
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask
    
    coords = np.column_stack(np.where(mask_gray > 0))
    if coords.size == 0:
        return None
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def draw_gt_skeleton(img: np.ndarray, keypoints: np.ndarray,
                     line_thickness: int = 1, point_radius: int = 2,
                     show_indices: bool = False) -> np.ndarray:
    """
    参考 Kaggle notebook draw_keypoints 函数绘制 GT 骨架。
    跳过无效关键点 (x < 0)。
    
    Args:
        img: 原始图像 (BGR)
        keypoints: (21, 2) 的关键点坐标 [x, y], 像素坐标
        line_thickness: 骨架线宽
        point_radius: 关键点半径
        show_indices: 是否在关键点旁标注索引号
    
    Returns:
        绘制了骨架的图像
    """
    img_copy = img.copy()
    kps = keypoints.astype(np.int32)
    
    # 绘制骨架连接 (参考 Kaggle POSE_PAIRS)
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
        cv2.line(img_copy, tuple(kps[start]), tuple(kps[end]), color, line_thickness)
    
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
        
        cv2.circle(img_copy, (int(x), int(y)), point_radius, color, -1, cv2.FILLED)
        cv2.circle(img_copy, (int(x), int(y)), point_radius, (255, 255, 255), 1)
        if show_indices:
            cv2.putText(img_copy, str(i), (int(x) + point_radius + 2, int(y) + 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1,
                       cv2.LINE_AA)
    
    return img_copy


# ============================================================
# EDA 主函数
# ============================================================

def eda_split(split: str = "train", n_samples: int = 10):
    """
    对指定 split 进行 EDA 可视化
    
    对每个样本生成一张包含 5 个子图的可视化:
    1. 原始图像
    2. Mask Overlay
    3. GT 骨架 + BBox
    4. MediaPipe 检测结果
    5. GT vs MediaPipe 对比 (两者叠加)
    
    Args:
        split: "train", "valid", 或 "test"
        n_samples: 采样数量
    """
    data_root = "/home/models/rightArm_collection/OneHand10K"
    tfrecord_dir = os.path.join(data_root, "onehand10k-tfrecords")
    eda_output = "eda_output"

    eda_output_path = os.path.join(data_root, eda_output, split)
    if os.path.exists(eda_output_path):
        shutil.rmtree(eda_output_path)
    os.makedirs(eda_output_path, exist_ok=True)

    # 先检查 TFRecord 结构
    split_dir = os.path.join(tfrecord_dir, split)
    tfrecord_files = sorted(glob.glob(os.path.join(split_dir, "*.tfrecords")))
    if tfrecord_files:
        print(f"\n{'='*60}")
        print(f"Inspecting TFRecord structure for {split} split")
        print(f"{'='*60}")
        inspect_tfrecord(tfrecord_files[0])

    # 加载样本
    print(f"\nLoading {n_samples} samples from {split} split...")
    samples = load_samples(tfrecord_dir, split, n_samples)
    print(f"Selected {len(samples)} samples for visualization")

    for idx, sample in enumerate(samples):
        img_rgb = sample['image']  # (256, 256, 3) RGB uint8
        mask = sample['mask']      # (256, 256) uint8
        label = sample['label']    # (42,) float32

        # 转为 BGR (OpenCV 格式)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        # label: 42 floats, 交错排列 [x0,y0,x1,y1,...,x20,y20]
        # 坐标为 [-0.5, 0.5] 中心归一化格式
        # 反归一化: pixel = (value + 0.5) * IMAGE_SIZE
        # 无效关键点的原始值远 < -0.5 (如 -1), 反归一化后 pixel < 0
        label_reshaped = label.reshape(21, 2)  # (21, 2) -> [x, y]

        # 反归一化: (value + 0.5) * 256 → 像素坐标
        # 参考 Kaggle notebook: unnormalize_and_scale_labels
        gt_keypoints = (label_reshaped + 0.5) * IMAGE_SIZE

        # 区分有效/无效关键点 (反归一化后, pixel_x < 0 为无效)
        valid_mask = gt_keypoints[:, 0] >= 0
        n_valid = valid_mask.sum()
        n_invalid = (~valid_mask).sum()

        # 将无效关键点设为大负值, 防止 int32 截断后变 0
        gt_keypoints[~valid_mask] = -1000

        # 从 mask 计算 BBox (而非 skeleton)
        bbox = get_bbox_from_mask(mask)

        print(f"\n[{split}] Sample {idx}: img_shape={img_rgb.shape}, "
              f"mask_shape={mask.shape}")
        print(f"  valid/invalid keypoints: {n_valid}/{n_invalid}")
        if n_valid > 0:
            vkps = gt_keypoints[valid_mask]
            print(f"  gt_keypoints pixel range: x=[{vkps[:,0].min():.1f}, {vkps[:,0].max():.1f}], "
                  f"y=[{vkps[:,1].min():.1f}, {vkps[:,1].max():.1f}]")
        if bbox is not None:
            bx0, by0, bx1, by1 = bbox
            print(f"  mask bbox: ({bx0}, {by0}) -> ({bx1}, {by1}), "
                  f"center=({(bx0+bx1)//2}, {(by0+by1)//2}), "
                  f"size=({bx1-bx0}x{by1-by0})")
        else:
            print(f"  mask bbox: None (empty mask)")

        # --- Mask 转为 3 通道用于显示 ---
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # --- 子图 1: 原始图像 ---
        img_original = img_bgr.copy()

        # --- 子图 2: Mask Overlay + BBox (from mask) ---
        img_mask_overlay = cv2.addWeighted(img_bgr, 0.5, mask_3ch, 0.5, 0)
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(img_mask_overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(img_mask_overlay, "mask bbox", (x_min, y_min - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # --- 子图 3: GT 骨架 + BBox (from mask) ---
        # 使用 Kaggle notebook 的 POSE_PAIRS 绘制骨架 (与 MediaPipe 一致)
        img_gt_skeleton = draw_gt_skeleton(img_bgr, gt_keypoints,
                                            line_thickness=2, point_radius=3,
                                            show_indices=True)
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(img_gt_skeleton, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(img_gt_skeleton, "mask bbox", (x_min, y_min - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # --- 子图 4: MediaPipe 检测结果 ---
        mp_keypoints = run_mediapipe_inference(img_bgr)
        if mp_keypoints is not None:
            img_mediapipe = draw_skeleton(img_bgr, mp_keypoints)
            mp_title = "MediaPipe Detection"
        else:
            img_mediapipe = img_bgr.copy()
            cv2.putText(img_mediapipe, "No hand detected", 
                       (10, img_bgr.shape[0] // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            mp_title = "MediaPipe (Failed)"

        # --- 子图 5: GT vs MediaPipe 对比 ---
        img_compare = img_bgr.copy()
        # GT: 彩色骨架 (实线)
        img_compare = draw_gt_skeleton(img_compare, gt_keypoints,
                                        line_thickness=2, point_radius=4)
        if mp_keypoints is not None:
            # MediaPipe: 橙色骨架 (细线)
            mp_kps_int = mp_keypoints.astype(np.int32)
            for start, end in HAND_SKELETON:
                cv2.line(img_compare, tuple(mp_kps_int[start]),
                        tuple(mp_kps_int[end]), (0, 165, 255), 1)
            for i, (x, y) in enumerate(mp_kps_int):
                cv2.circle(img_compare, (int(x), int(y)), 3, (0, 165, 255), -1)
            compare_title = "GT (color) vs MP (orange)"
        else:
            compare_title = "GT only (MP failed)"

        # --- 绘制完整图 (5 子图) ---
        fig, axs = plt.subplots(1, 5, figsize=(30, 6))
        
        axs[0].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
        axs[0].set_title(f"Original ({w}x{h})")
        axs[0].axis('off')
        
        axs[1].imshow(cv2.cvtColor(img_mask_overlay, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Mask Overlay + BBox (from mask)")
        axs[1].axis('off')
        
        axs[2].imshow(cv2.cvtColor(img_gt_skeleton, cv2.COLOR_BGR2RGB))
        axs[2].set_title(f"GT Skeleton + BBox\n(valid: {n_valid}/21)")
        axs[2].axis('off')
        
        axs[3].imshow(cv2.cvtColor(img_mediapipe, cv2.COLOR_BGR2RGB))
        axs[3].set_title(mp_title)
        axs[3].axis('off')
        
        axs[4].imshow(cv2.cvtColor(img_compare, cv2.COLOR_BGR2RGB))
        axs[4].set_title(compare_title)
        axs[4].axis('off')
        
        plt.suptitle(f"OneHand10K - {split} - Sample {idx}", fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(eda_output_path, f"sample_{idx:04d}.jpg")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")

    print(f"\n{'='*60}")
    print(f"EDA for {split} split done! Output saved to: {eda_output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OneHand10K EDA - Sample Visualization & MediaPipe Comparison")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "valid", "test"],
                       help="Dataset split to inspect (default: train)")
    parser.add_argument("--n_samples", type=int, default=10,
                       help="Number of samples to visualize (default: 10)")
    args = parser.parse_args()
    
    eda_split(split=args.split, n_samples=args.n_samples)
