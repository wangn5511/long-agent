import os
import cv2
import numpy as np
import json
from typing import Optional, Tuple
import shutil
import random
import glob
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import matplotlib.pyplot as plt


# MediaPipe HandLandmarker 模型路径
_HAND_LANDMARKER_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"
)


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


def draw_mediapipe_skeleton(img: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """
    在图像上绘制 MediaPipe 风格的手部骨架
    
    Args:
        img: 原始图像
        keypoints: (21, 2) 的关键点坐标
    
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
        cv2.line(img_copy, pt1, pt2, color, 2)
    
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
        
        cv2.circle(img_copy, (int(x), int(y)), 4, color, -1)
        cv2.circle(img_copy, (int(x), int(y)), 4, (255, 255, 255), 1)
    
    return img_copy


"""
training:
There are total 130240 images
There are total 32560 mask files
There are 97680 images without mask

training_scale.json:
There are total 32560 scale data
用途: 用于将3D坐标从归一化空间转换到真实世界尺度
应用场景: 当需要恢复手部的真实物理尺寸时使用

training_mano.json:
There are total 32560 MANO data
MANO 是一个参数化的手部模型 (类似于人体的 SMPL 模型)
pose (48维): 16个关节 × 3个旋转角度 (axis-angle)，控制手指弯曲姿态
shape (10维): PCA 形状空间系数，控制手部的大小、粗细等形状特征

training_verts.json:
There are total 32560 verts data
形状: [N, 778, 3]
内容: MANO 手部模型的 778 个网格顶点的 3D 坐标
用途:
可直接用于 3D 手部重建的监督训练
配合 MANO 的面片定义可渲染完整的手部网格
用于手部分割、深度估计等任务

training_xyz.json:
There are total 32560 xyz data
形状: [32560, 21, 3]
坐标系: 相机坐标系 (Camera Coordinate System)
X轴 → 向右
Y轴 → 向下
Z轴 → 向前（远离相机）
单位: 米 (meters)

training_K.json:
There are total 32560 K data
形状: [32560, 3, 3]
参数含义:
fx, fy: 焦距（单位：像素）
cx, cy: 主点坐标（通常接近图像中心 112, 112）

img shape:  (224, 224, 3)
mask shape:  (224, 224, 3)

mask 是 3 个通道相同的灰度图
3 channels identical: True
Is binary (only 0 and 255): False
"""

# MediaPipe 风格的骨架连接
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


def project_3d_to_2d(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    将3D坐标投影到2D图像平面
    
    Args:
        xyz: (21, 3) - 21个手部关键点的3D坐标
        K: (3, 3) - 相机内参矩阵
    
    Returns:
        (21, 2) - 2D关键点坐标
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
    
    coords = np.column_stack(np.where(mask_gray > 0.5 * 255))
    if coords.size == 0:
        return None
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return (x_min, y_min, x_max, y_max)


def eda_train():
    data_root = "/home/models/rightArm_collection/FreiHAND/"
    split = "training"
    eda_output = "eda_output"

    img_dir = os.path.join(data_root, split, "rgb")
    mask_dir = os.path.join(data_root, split, "mask")

    img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
    mask_files = glob.glob(os.path.join(mask_dir, "*.jpg"))

    cnt = 0
    exist_img_files = []
    aug_img_files = []
    for img_f in img_files:
        name = os.path.basename(img_f)
        mask_path = os.path.join(mask_dir, name)
        if not os.path.exists(mask_path):
            cnt += 1
            aug_img_files.append(name)
        else:
            exist_img_files.append(name)

    num_origin_images = len(exist_img_files)
    print("There are total {} images".format(len(img_files)))
    print("There are total {} mask files".format(len(mask_files)))
    print(f"There are {cnt} images without mask")

    # # 尺度因子数据
    # scale_path = os.path.join(data_root, "training_scale.json")
    # scale_data = json.load(open(scale_path, "r"))
    # print("There are total {} scale data".format(len(scale_data)))

    # # MANO 参数数据
    # mano_path = os.path.join(data_root, "training_mano.json")
    # mano_data = json.load(open(mano_path, "r"))
    # print("There are total {} MANO data".format(len(mano_data)))

    # # 手部网格顶点数据
    # verts_path = os.path.join(data_root, "training_verts.json")
    # verts_data = json.load(open(verts_path, "r"))
    # print("There are total {} verts data".format(len(verts_data)))

    # 3D关键点坐标数据
    xyz_path = os.path.join(data_root, "training_xyz.json")
    xyz_data = json.load(open(xyz_path, "r"))
    print("There are total {} xyz data".format(len(xyz_data)))

    # 相机内参矩阵数据
    K_path = os.path.join(data_root, "training_K.json")
    K_data = json.load(open(K_path, "r"))
    print("There are total {} K data".format(len(K_data)))

    eda_output_path = os.path.join(data_root, eda_output, split)
    shutil.rmtree(eda_output_path)
    os.makedirs(eda_output_path, exist_ok=True)

    sample_names = random.sample(exist_img_files, k=10)
    for sample_name in sample_names:
        img_f = os.path.join(img_dir, sample_name)
        mask_f = os.path.join(mask_dir, sample_name)

        img = cv2.imread(img_f)
        mask = cv2.imread(mask_f)

        bbox = get_bbox_from_mask(mask)  # [x_min, y_min, x_max, y_max]

        idx = int(sample_name.split(".")[0])
        xyz = np.array(xyz_data[idx])
        K = np.array(K_data[idx])

        uv = project_3d_to_2d(xyz, K)
        uv = np.clip(uv, 0, 223).astype(int)  # 限制在图像范围内
        
        # 绘制骨架
        img_with_skeleton = img.copy()
        for i, (start, end) in enumerate(HAND_SKELETON):
            # 根据连接索引确定颜色
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
            
            pt1 = (int(uv[start, 0]), int(uv[start, 1]))
            pt2 = (int(uv[end, 0]), int(uv[end, 1]))
            cv2.line(img_with_skeleton, pt1, pt2, color, 2)
        
        # 绘制关键点
        for i, (x, y) in enumerate(uv):
            # 手腕用白色，其他用对应手指颜色
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
            
            cv2.circle(img_with_skeleton, (int(x), int(y)), 4, color, -1)
            cv2.circle(img_with_skeleton, (int(x), int(y)), 4, (255, 255, 255), 1)

        # 绘制边界框
        img_with_bbox = img.copy()
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(img_with_bbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # 添加标签
            cv2.putText(img_with_bbox, "hand", (x_min, y_min - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 同时在skeleton图上也画bbox
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(img_with_skeleton, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # MediaPipe 推理
        mp_keypoints = run_mediapipe_inference(img)
        if mp_keypoints is not None:
            img_with_mediapipe = draw_mediapipe_skeleton(img, mp_keypoints)
            mp_title = "MediaPipe Detection"
        else:
            img_with_mediapipe = img.copy()
            cv2.putText(img_with_mediapipe, "No hand detected", (10, 112),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            mp_title = "MediaPipe (Failed)"

        fig, axs = plt.subplots(1, 6, figsize=(30, 5))
        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original Image")
        axs[0].axis('off')
        
        axs[1].imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Mask")
        axs[1].axis('off')
        
        axs[2].imshow(cv2.cvtColor(cv2.addWeighted(img, 0.5, mask, 0.5, 0), cv2.COLOR_BGR2RGB))
        axs[2].set_title("Mask + Original Image")
        axs[2].axis('off')
        
        axs[3].imshow(cv2.cvtColor(img_with_skeleton, cv2.COLOR_BGR2RGB))
        axs[3].set_title("GT Skeleton + BBox")
        axs[3].axis('off')
        
        axs[4].imshow(cv2.cvtColor(img_with_mediapipe, cv2.COLOR_BGR2RGB))
        axs[4].set_title(mp_title)
        axs[4].axis('off')
        
        axs[5].imshow(cv2.cvtColor(img_with_bbox, cv2.COLOR_BGR2RGB))
        axs[5].set_title("Bounding Box")
        axs[5].axis('off')
        
        plt.tight_layout()

        plt.savefig(os.path.join(eda_output_path, sample_name))
        print("Save plot image to: ", os.path.join(eda_output_path, sample_name))

    # sample augmentation images
    sample_names = random.sample(aug_img_files, k=10)
    for sample_name in sample_names:
        img_f = os.path.join(img_dir, sample_name)
        origin_idx = int(sample_name.split(".")[0]) % num_origin_images
        mask_f = os.path.join(mask_dir, f"{origin_idx:08d}.jpg")

        img = cv2.imread(img_f)
        mask = cv2.imread(mask_f)

        bbox = get_bbox_from_mask(mask)  # [x_min, y_min, x_max, y_max]

        xyz = np.array(xyz_data[origin_idx])
        K = np.array(K_data[origin_idx])

        uv = project_3d_to_2d(xyz, K)
        uv = np.clip(uv, 0, 223).astype(int)  # 限制在图像范围内
        
        # 绘制骨架
        img_with_skeleton = img.copy()
        for i, (start, end) in enumerate(HAND_SKELETON):
            # 根据连接索引确定颜色
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
            
            pt1 = (int(uv[start, 0]), int(uv[start, 1]))
            pt2 = (int(uv[end, 0]), int(uv[end, 1]))
            cv2.line(img_with_skeleton, pt1, pt2, color, 2)
        
        # 绘制关键点
        for i, (x, y) in enumerate(uv):
            # 手腕用白色，其他用对应手指颜色
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
            
            cv2.circle(img_with_skeleton, (int(x), int(y)), 4, color, -1)
            cv2.circle(img_with_skeleton, (int(x), int(y)), 4, (255, 255, 255), 1)

        # 绘制边界框
        img_with_bbox = img.copy()
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(img_with_bbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # 添加标签
            cv2.putText(img_with_bbox, "hand", (x_min, y_min - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 同时在skeleton图上也画bbox
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(img_with_skeleton, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # MediaPipe 推理
        mp_keypoints = run_mediapipe_inference(img)
        if mp_keypoints is not None:
            img_with_mediapipe = draw_mediapipe_skeleton(img, mp_keypoints)
            mp_title = "MediaPipe Detection"
        else:
            img_with_mediapipe = img.copy()
            cv2.putText(img_with_mediapipe, "No hand detected", (10, 112),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            mp_title = "MediaPipe (Failed)"

        fig, axs = plt.subplots(1, 6, figsize=(30, 5))
        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original Image")
        axs[0].axis('off')
        
        axs[1].imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Mask")
        axs[1].axis('off')
        
        axs[2].imshow(cv2.cvtColor(cv2.addWeighted(img, 0.5, mask, 0.5, 0), cv2.COLOR_BGR2RGB))
        axs[2].set_title("Mask + Original Image")
        axs[2].axis('off')
        
        axs[3].imshow(cv2.cvtColor(img_with_skeleton, cv2.COLOR_BGR2RGB))
        axs[3].set_title("GT Skeleton + BBox")
        axs[3].axis('off')
        
        axs[4].imshow(cv2.cvtColor(img_with_mediapipe, cv2.COLOR_BGR2RGB))
        axs[4].set_title(mp_title)
        axs[4].axis('off')
        
        axs[5].imshow(cv2.cvtColor(img_with_bbox, cv2.COLOR_BGR2RGB))
        axs[5].set_title("Bounding Box")
        axs[5].axis('off')
        
        plt.tight_layout()

        plt.savefig(os.path.join(eda_output_path, sample_name.split('.')[0] + '_aug.jpg'))
        print("Save plot image to: ", os.path.join(eda_output_path, sample_name.split('.')[0] + '_aug.jpg'))


def eda_eval():
    data_root = "/home/models/rightArm_collection/FreiHAND/"
    split = "evaluation"
    eda_output = "eda_output"

    img_dir = os.path.join(data_root, split, "rgb")
    mask_dir = os.path.join(data_root, split, "segmap")

    img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
    mask_files = glob.glob(os.path.join(mask_dir, "*.png"))

    cnt = 0
    exist_img_files = []
    aug_img_files = []
    for img_f in img_files:
        name = os.path.basename(img_f)
        mask_path = os.path.join(mask_dir, name.replace('.jpg', '.png'))
        if not os.path.exists(mask_path):
            cnt += 1
            aug_img_files.append(name)
        else:
            exist_img_files.append(name)
    print(f"There are {cnt} images without mask")
    print("There are total {} images".format(len(img_files)))
    print("There are total {} mask files".format(len(mask_files)))
    print("augmented images: ", len(aug_img_files))

    # evaluation_xyz.json:
    xyz_path = os.path.join(data_root, "evaluation_xyz.json")
    xyz_data = json.load(open(xyz_path, "r"))
    print("There are total {} xyz data".format(len(xyz_data)))

    # evaluation_K.json:
    K_path = os.path.join(data_root, "evaluation_K.json")
    K_data = json.load(open(K_path, "r"))
    print("There are total {} K data".format(len(K_data)))

    eda_output_path = os.path.join(data_root, eda_output, split)
    shutil.rmtree(eda_output_path, ignore_errors=True)
    os.makedirs(eda_output_path, exist_ok=True)

    sample_names = random.sample(exist_img_files, k=10)
    for sample_name in sample_names:
        img_f = os.path.join(img_dir, sample_name)
        mask_f = os.path.join(mask_dir, sample_name.replace('.jpg', '.png'))

        img = cv2.imread(img_f)
        mask = cv2.imread(mask_f)

        bbox = get_bbox_from_mask(mask)  # [x_min, y_min, x_max, y_max]

        idx = int(sample_name.split(".")[0])
        xyz = np.array(xyz_data[idx])
        K = np.array(K_data[idx])

        uv = project_3d_to_2d(xyz, K)
        uv = np.clip(uv, 0, 223).astype(int)  # 限制在图像范围内
        
        # 绘制骨架
        img_with_skeleton = img.copy()
        for i, (start, end) in enumerate(HAND_SKELETON):
            # 根据连接索引确定颜色
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
            
            pt1 = (int(uv[start, 0]), int(uv[start, 1]))
            pt2 = (int(uv[end, 0]), int(uv[end, 1]))
            cv2.line(img_with_skeleton, pt1, pt2, color, 2)
        
        # 绘制关键点
        for i, (x, y) in enumerate(uv):
            # 手腕用白色，其他用对应手指颜色
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
            
            cv2.circle(img_with_skeleton, (int(x), int(y)), 4, color, -1)
            cv2.circle(img_with_skeleton, (int(x), int(y)), 4, (255, 255, 255), 1)

        # 绘制边界框
        img_with_bbox = img.copy()
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(img_with_bbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # 添加标签
            cv2.putText(img_with_bbox, "hand", (x_min, y_min - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 同时在skeleton图上也画bbox
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(img_with_skeleton, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # MediaPipe 推理
        mp_keypoints = run_mediapipe_inference(img)
        if mp_keypoints is not None:
            img_with_mediapipe = draw_mediapipe_skeleton(img, mp_keypoints)
            mp_title = "MediaPipe Detection"
        else:
            img_with_mediapipe = img.copy()
            cv2.putText(img_with_mediapipe, "No hand detected", (10, 112),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            mp_title = "MediaPipe (Failed)"

        fig, axs = plt.subplots(1, 6, figsize=(30, 5))
        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original Image")
        axs[0].axis('off')
        
        axs[1].imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Mask")
        axs[1].axis('off')
        
        axs[2].imshow(cv2.cvtColor(cv2.addWeighted(img, 0.5, mask, 0.5, 0), cv2.COLOR_BGR2RGB))
        axs[2].set_title("Mask + Original Image")
        axs[2].axis('off')
        
        axs[3].imshow(cv2.cvtColor(img_with_skeleton, cv2.COLOR_BGR2RGB))
        axs[3].set_title("GT Skeleton + BBox")
        axs[3].axis('off')
        
        axs[4].imshow(cv2.cvtColor(img_with_mediapipe, cv2.COLOR_BGR2RGB))
        axs[4].set_title(mp_title)
        axs[4].axis('off')
        
        axs[5].imshow(cv2.cvtColor(img_with_bbox, cv2.COLOR_BGR2RGB))
        axs[5].set_title("Bounding Box")
        axs[5].axis('off')
        
        plt.tight_layout()

        plt.savefig(os.path.join(eda_output_path, sample_name))
        print("Save plot image to: ", os.path.join(eda_output_path, sample_name))


if __name__ == "__main__":
    # eda_train()  # 训练集EDA
    eda_eval()  # 测试集EDA
