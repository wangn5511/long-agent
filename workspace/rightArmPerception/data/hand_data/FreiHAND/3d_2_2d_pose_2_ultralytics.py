import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def project_3d_to_2d(xyz, K):
    """
    将3D坐标投影到2D图像平面
    xyz: (21, 3) - 21个手部关键点的3D坐标
    K: (3, 3) - 相机内参矩阵
    返回: (21, 2) - 2D关键点坐标
    """
    # 将3D点投影到2D平面 (忽略畸变)
    uv = K @ xyz.T
    uv = uv[:2, :] / uv[2:, :]  # 齐次坐标转换
    return uv.T.astype(int)

def get_hbb_from_mask(mask_path):
    """
    从mask图像中获取HBB边界框
    mask_path: mask图像路径
    返回: [x_min, y_min, width, height] 或 None (如果mask不存在或为空)
    """
    if not os.path.exists(mask_path):
        return None
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    # 找到mask中非零像素的坐标
    coords = np.column_stack(np.where(mask > 0.8))
    
    if coords.size == 0:
        return None
    
    # 计算边界框
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    width = x_max - x_min
    height = y_max - y_min
    
    return [x_min, y_min, width, height]

def get_hbb_from_pose(uv_points):
    """
    从2D关键点获取HBB边界框
    uv_points: (21, 2) - 2D关键点坐标
    返回: [x_min, y_min, width, height]
    """
    x_coords = uv_points[:, 0]
    y_coords = uv_points[:, 1]
    
    x_min = int(np.min(x_coords))
    y_min = int(np.min(y_coords))
    x_max = int(np.max(x_coords))
    y_max = int(np.max(y_coords))
    
    width = x_max - x_min
    height = y_max - y_min
    
    return [x_min, y_min, width, height]




def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    将边界框从 [x_min, y_min, width, height] 转换为 YOLO 格式 [center_x, center_y, width, height]
    并归一化到 [0, 1]
    """
    x_min, y_min, width, height = bbox
    center_x = x_min + width / 2
    center_y = y_min + height / 2
    
    # 归一化
    center_x /= img_width
    center_y /= img_height
    width /= img_width
    height /= img_height
    
    return [center_x, center_y, width, height]

def save_yolo_label(label_path, yolo_bbox, keypoints=None):
    """
    保存YOLO格式的标签文件
    label_path: 标签文件保存路径
    yolo_bbox: YOLO格式的边界框 [center_x, center_y, width, height]
    keypoints: 关键点列表 [(x, y, visibility), ...]，visibility: 0=invisible, 1=visible, 2=occluded
    """
    with open(label_path, 'w') as f:
        # 写入边界框 (class_id = 0)
        line = f"0 {' '.join(map(str, yolo_bbox))}"
        
        # 如果有关键点信息，添加到行末
        if keypoints is not None:
            # 对关键点进行归一化
            normalized_keypoints = []
            for kp in keypoints:
                x_norm = kp[0]  # 已经是归一化后的值
                y_norm = kp[1]  # 已经是归一化后的值
                vis = kp[2]     # 可见性
                normalized_keypoints.extend([x_norm, y_norm, vis])
            line += " " + " ".join(map(str, normalized_keypoints))
        
        f.write(line + "\n")




def one_sample(data_root="/home/datasets/rest_gesture_dataset/FreiHAND", mode='evaluation'):
    # 假设根目录为 FreiHAND/
    img_dir = os.path.join(data_root, mode, "rgb")
    mask_dir = os.path.join(data_root, mode, "segmap")
    if not os.path.exists(mask_dir):
        mask_dir = os.path.join(data_root, mode, "mask")

    # 创建输出目录
    output_dir = os.path.join(data_root, "visualization_output")
    label_dir = os.path.join(data_root, "labels", mode)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)


    # 如果是评估集，则使用evaluation文件
    eval_xyz_path = os.path.join(data_root, f"{mode}_xyz.json")
    eval_k_path = os.path.join(data_root, f"{mode}_K.json")
    
    if os.path.exists(eval_xyz_path) and os.path.exists(eval_k_path):
        with open(eval_xyz_path, 'r') as f:
            xyz_list = json.load(f)
        with open(eval_k_path, 'r') as f:
            K_list = json.load(f)
    else:
        print("No pose data found!")
        raise FileNotFoundError


    for i in range(len(os.listdir(os.path.join(data_root, mode, "rgb")))):  # 仅处理前5个样本作为示例
        # 加载并投影 pose
        try:
            if False:
                pass
            else:
                # 获取第 i 个样本
                img_name = f"{i:08d}.jpg"          # '00000.jpg'
                img_path = os.path.join(img_dir, img_name)
                mask_path = os.path.join(mask_dir, img_name.replace(".jpg", ".png"))  # 假设mask和图片同名
                if not os.path.exists(mask_path):
                    mask_name = f"{i%32560:08d}.jpg"
                    mask_path = os.path.join(mask_dir, mask_name)  # 尝试PNG格式

                
                if m == 'training':
                    idx = i % 32560
                else:
                    idx = i

                xyz = np.array(xyz_list[idx])        # (21, 3)
                K = np.array(K_list[idx])            # (3, 3)
                uv = project_3d_to_2d(xyz, K)      # (21, 2)
                uv = np.clip(uv, 0, 223)

            # 尝试从mask获取HBB
            hbb_from_mask = get_hbb_from_mask(mask_path)
            
            
            # 加载图像
            img = cv2.imread(img_path)
            if img is None:
                # 如果JPG不存在，尝试PNG格式
                img_path = os.path.join(data_root, mode, "rgb", f"{i:08d}.png")
                img = cv2.imread(img_path)

            if img is not None:
                # 获取图像尺寸
                img_height, img_width = img.shape[:2]

                # 使用mask HBB作为主要边界框
                main_hbb = hbb_from_mask
                
                # 转换为YOLO格式
                yolo_bbox = convert_bbox_to_yolo(main_hbb, img_width, img_height)
                
                # 准备关键点数据 (对于Ultralytics格式，每个关键点包括 x, y, visibility)
                keypoints_data = []
                for j in range(len(uv)):
                    x_norm = uv[j][0] / img_width  # 归一化x坐标
                    y_norm = uv[j][1] / img_height  # 归一化y坐标
                    visibility = 2  # 假设所有关键点都是可见的，可根据实际数据调整
                    keypoints_data.append([x_norm, y_norm, visibility])
                
                # 保存YOLO格式标签文件
                label_filename = f"{i:08d}.txt"
                label_path = os.path.join(label_dir, label_filename)
                save_yolo_label(label_path, yolo_bbox, keypoints_data)
                
                print(f"Saved YOLO format label to: {label_path}")

        except FileNotFoundError as e:
            print(f"File not found: {e}")
        except Exception as e:
            print(f"Error occurred: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    data_root="/home/datasets/rest_gesture_dataset/FreiHAND"
    mode=['training', 'evaluation']
    for m in mode:
        one_sample(data_root=data_root, mode=m)

    
