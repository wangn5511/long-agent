# Head Pose Estimation 数据集文档

本目录包含头部姿态估计相关数据集的 EDA (探索性数据分析) 脚本。

## 数据集概览

| 数据集 | 样本数 | 图像尺寸 | 场景 | 传感器 | 关键点 | 角度范围 | 深度图 | 需预裁剪 |
|--------|--------|----------|------|--------|--------|----------|--------|----------|
| AFLW2000-3D | 2,000 | 450×450 | 真实野外 | 普通相机 | 68个 3D | ±90° | ❌ | ❌ (提供ROI) |
| BIWI | 15,677 | 640×480 | 实验室 | Kinect | ❌ | Pitch ±60°, Yaw ±75° | ✅ | ❌ (整张脸) |
| 300W-LP | 122,450 | 450×450 | 合成 | 普通相机 | 68个 2D/3D | ±90° | ❌ | ❌ (提供ROI) |
| UPNA | 36,000 | 1280×720 | 合成视频 | 虚拟相机 | 43个 2D | ±46° | ❌ | ✅ (提供BBox) |

### 关键特性对比

| 特性 | AFLW2000-3D | BIWI | 300W-LP | UPNA |
|------|-------------|------|---------|------|
| **RPY单位** | 弧度 (需转换) | 角度 | 弧度 (需转换) | 角度 |
| **RPY顺序** | Pitch, Yaw, Roll | Pitch, Yaw, Roll | Pitch, Yaw, Roll | Roll, Yaw, Pitch |
| **数据类型** | 真实图像 | 真实视频 | 合成扩增 | 合成视频 |
| **适用场景** | 评估/测试 | 训练/评估 | 训练 | 训练/评估 |

---

## 1. AFLW2000-3D 数据集

### 1.1 下载地址

| 存储 | 路径 |
|------|------|
| 火山云 | `tos://cytoderm-embodied-ai/datasets/AFLW2000-3D.zip` |
| 腾讯云 | `cos://cytoderm-ai-shanghai-1389948562/datasets/AFLW2000-3D.zip` |

### 1.2 基本信息

| 属性 | 值 |
|------|-----|
| **样本数** | 2,000 张图像 |
| **图像尺寸** | 450×450 |
| **场景** | 真实野外 (Wild) |
| **关键点** | 68个 3D 关键点 ✅ |
| **深度图** | ❌ 无 |
| **边界框** | ✅ 提供 ROI (但可能有负值) |
| **需预裁剪** | ❌ 图像已是人脸区域 |

### 1.3 文件组织结构

```
AFLW2000/
├── image00002.jpg          # 人脸图像
├── image00002.mat          # 对应的标注文件 (MATLAB格式)
├── image00004.jpg
├── image00004.mat
├── ...
└── (共 2000 对 jpg + mat 文件)
```

### 1.4 标注文件格式 (.mat)

每个 `.mat` 文件包含以下字段:

| 字段名 | 形状 | 说明 |
|--------|------|------|
| `Pose_Para` | (1, 7) | **姿态参数**: [pitch, yaw, roll, tx, ty, tz, scale] |
| `pt3d_68` | (3, 68) | 68个 3D 人脸关键点坐标 |
| `pt2d` | (2, 21) | 21个 2D 人脸关键点 |
| `roi` | (1, 4) | 人脸边界框 [x1, y1, x2, y2] |
| `Shape_Para` | (199, 1) | 3DMM 形状参数 |
| `Exp_Para` | (29, 1) | 表情参数 |

### 1.5 RPY 角度定义

| 属性 | 值 |
|------|-----|
| **存储单位** | ⚠️ **弧度制** |
| **使用时需转换** | ✅ 需要先标准化到 [-π, π]，再转角度 |
| **顺序** | [Pitch, Yaw, Roll] |
| **角度范围** | 过滤后: Pitch [-83°, 88°], Yaw [-84°, 87°], Roll [-87°, 84°] |

### 1.6 读取 RPY 角度

**关键点**: `Pose_Para` 的前 3 个值是 **弧度制** 的欧拉角，需要标准化后再转换为度数。

```python
import numpy as np
from scipy.io import loadmat

def map_to_pi(angle_rad):
    """将弧度角度标准化到 [-π, π] 范围"""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi

def parse_aflw_mat(mat_path):
    """
    解析 AFLW2000 的 MAT 标注文件
    
    返回:
        landmarks: (68, 3) 3D关键点
        pose: [pitch, yaw, roll] 度数，已标准化
    """
    mat_data = loadmat(mat_path)
    
    # 提取 3D 关键点 (原始形状 3×68，需转置)
    landmarks = mat_data['pt3d_68'].T  # (68, 3)
    
    # 提取姿态参数
    pose_para = mat_data['Pose_Para'].flatten()
    pitch_rad, yaw_rad, roll_rad = pose_para[0], pose_para[1], pose_para[2]
    
    # 1. 先在弧度域标准化到 [-π, π]
    pitch_rad = map_to_pi(pitch_rad)
    yaw_rad = map_to_pi(yaw_rad)
    roll_rad = map_to_pi(roll_rad)
    
    # 2. 再转换为度数
    pitch = np.degrees(pitch_rad)
    yaw = np.degrees(yaw_rad)
    roll = np.degrees(roll_rad)
    
    return landmarks, np.array([pitch, yaw, roll])

# 使用示例
landmarks, pose = parse_aflw_mat('image00002.mat')
print(f"Pitch: {pose[0]:.1f}°, Yaw: {pose[1]:.1f}°, Roll: {pose[2]:.1f}°")
```

### 1.7 注意事项

- ⚠️ **约 1.8% 的样本** (36个) 有极端角度 (|angle| > 90°)，可能是标注问题
- ✅ 过滤到 |angle| ≤ 90° 后，范围约为 Pitch [-83°, 88°], Yaw [-84°, 87°], Roll [-87°, 84°]
- 📦 ROI 边界框可能包含负值坐标，需要 clip 处理
- 参考: [Kaggle head-pose-estimation](https://www.kaggle.com/code/raedalkhatib/head-pose-estimation)

---

## 2. BIWI Kinect Head Pose Database

### 2.1 下载地址

| 存储 | 路径 |
|------|------|
| 火山云 | `tos://cytoderm-embodied-ai/datasets/biwi-kinect-head-pose-database.tar` |
| 腾讯云 | `cos://cytoderm-ai-shanghai-1389948562/datasets/rightArm/biwi-kinect-head-pose-database.tar` |

### 2.2 基本信息

| 属性 | 值 |
|------|-----|
| **样本数** | 15,677 帧 (24个被试 × 多个视频) |
| **图像尺寸** | 640×480 |
| **场景** | 实验室环境 |
| **传感器** | Microsoft Kinect |
| **关键点** | ❌ 无 |
| **深度图** | ✅ 有 (depth_mask.png) |
| **边界框** | ❌ 无 (整张图包含完整人脸) |
| **需预裁剪** | ❌ 图像以人脸为主体 |

### 2.3 文件组织结构

```
biwi-kinect-head-pose-database/
├── db_annotations/           # 姿态标注 (15,677 个 .bin 文件)
│   ├── 01/                   # 被试 01
│   │   ├── frame_00004_pose.bin
│   │   ├── frame_00005_pose.bin
│   │   └── ...
│   ├── 02/
│   └── ... (共 24 个被试)
│
├── faces_0/                  # 人脸图像 (47,107 个文件)
│   ├── 01/
│   │   ├── frame_00003_rgb.png      # RGB 图像
│   │   ├── frame_00004_rgb.png
│   │   ├── frame_00004_pose.txt     # 文本格式的姿态 (旋转矩阵)
│   │   └── ...
│   └── ...
│
└── head_pose_masks/          # 深度掩码 (15,677 个文件)
    ├── 01/
    │   ├── frame_00004_depth_mask.png
    │   └── ...
    └── ...
```

**文件比例**: 每个标注对应约 3 种文件 (RGB + Depth + Mask)

### 2.4 RPY 角度定义

| 属性 | 值 |
|------|-----|
| **存储单位** | ✅ **角度制** (直接使用) |
| **顺序** | [Pitch, Yaw, Roll] |
| **角度范围** | Pitch [-84°, 54°], Yaw [-67°, 77°], Roll [-70°, 63°] |

### 2.5 标注文件格式

BIWI 有两种标注格式 (内容完全一致)：

#### 方式一：二进制格式 (.bin) - 推荐

位于 `db_annotations/` 目录，文件大小 24 bytes (6 个 float32)：

```
[center_x, center_y, center_z, pitch, yaw, roll]
```

| 索引 | 字段 | 单位 | 说明 |
|------|------|------|------|
| 0-2 | center | mm | 头部中心 3D 坐标，z 约 800-1000mm |
| 3 | pitch | degrees | 俯仰角 (点头) |
| 4 | yaw | degrees | 偏航角 (摇头) |
| 5 | roll | degrees | 滚转角 (歪头) |

#### 方式二：文本格式 (.txt)

位于 `faces_0/` 目录，包含旋转矩阵和头部中心：

```
R11 R12 R13
R21 R22 R23
R31 R32 R33

tx ty tz
```

- 前 3 行：3×3 旋转矩阵
- 第 4 行：头部中心坐标 (mm)

### 2.6 读取 RPY 角度

#### 从 .bin 文件读取 (推荐)

```python
import numpy as np

def parse_biwi_bin(bin_path):
    """
    解析 BIWI 的 .bin 姿态标注文件
    
    返回:
        head_center: (3,) 头部中心 3D 坐标 (mm)
        euler: [pitch, yaw, roll] 度数 (直接可用，无需转换)
    """
    data = np.fromfile(bin_path, dtype=np.float32)
    
    if len(data) == 6:
        head_center = data[:3]  # [x, y, z] in mm
        euler = data[3:6]       # [pitch, yaw, roll] in degrees
        return head_center, euler
    else:
        raise ValueError(f"Unexpected format: {len(data)} float32 values")

# 使用示例
center, euler = parse_biwi_bin('frame_00008_pose.bin')
print(f"Head Center: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}] mm")
print(f"Pitch: {euler[0]:.1f}°, Yaw: {euler[1]:.1f}°, Roll: {euler[2]:.1f}°")
```

#### 从 .txt 文件读取

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

def parse_biwi_txt(txt_path):
    """
    解析 BIWI 的 .txt 姿态标注文件 (旋转矩阵格式)
    
    返回:
        head_center: (3,) 头部中心 3D 坐标 (mm)
        euler: [pitch, yaw, roll] 度数
    """
    with open(txt_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    
    # 解析 3×3 旋转矩阵
    rot_matrix = np.array([
        [float(x) for x in lines[0].split()],
        [float(x) for x in lines[1].split()],
        [float(x) for x in lines[2].split()]
    ])
    
    # 解析头部中心
    head_center = np.array([float(x) for x in lines[3].split()])
    
    # 从旋转矩阵提取欧拉角
    r = R.from_matrix(rot_matrix)
    euler = r.as_euler('xyz', degrees=True)  # [pitch, yaw, roll]
    
    return head_center, euler

# 使用示例
center, euler = parse_biwi_txt('frame_00100_pose.txt')
print(f"Pitch: {euler[0]:.1f}°, Yaw: {euler[1]:.1f}°, Roll: {euler[2]:.1f}°")
```

### 2.7 注意事项

- ✅ 两种格式 (.bin 和 .txt) 的标注**完全一致** (差异为 0)
- ✅ **无需角度转换**，直接读取即为角度制
- ✅ **无极端异常值**，所有角度都在 ±90° 范围内
- 🎯 官方角度范围: Pitch ±60°, Yaw ±75°, Roll 较小
- 📷 图像尺寸统一为 640×480
- 🔧 Kinect 相机内参: fx=fy=517.679, cx=320, cy=240.5

---

## 3. 300W-LP (300W Large Pose) 数据集

### 3.1 下载地址

| 存储 | 路径 |
|------|------|
| 火山云 | `tos://cytoderm-embodied-ai/datasets/300W-LP.zip` |
| 腾讯云 | `cos://cytoderm-ai-shanghai-1389948562/datasets/300W-LP.zip` |

### 3.2 基本信息

| 属性 | 值 |
|------|-----|
| **样本数** | 122,450 张 (原始 61,225 + 翻转 61,225) |
| **图像尺寸** | 450×450 |
| **场景** | 合成 (基于 300W 数据集通过 3D 变换生成) |
| **关键点** | 68个 2D/3D 关键点 ✅ |
| **深度图** | ❌ 无 |
| **边界框** | ✅ 提供 ROI |
| **需预裁剪** | ❌ 图像已裁剪 |

### 3.3 文件组织结构

```
300W_LP/
├── Code/                    # 官方读取代码 (MATLAB)
├── landmarks/               # 关键点标注
│   ├── AFW/                 # *_pts.mat 文件 (68个 2D/3D 关键点)
│   ├── HELEN/
│   ├── IBUG/
│   └── LFPW/
├── AFW/                     # 图像和参数 (.jpg + .mat)
├── AFW_Flip/                # 翻转版本 (数据增强)
├── HELEN/
├── HELEN_Flip/
├── IBUG/
├── IBUG_Flip/
├── LFPW/
└── LFPW_Flip/
```

### 3.4 子数据集统计

| 子数据集 | 原始样本数 | 翻转样本数 | 总计 |
|----------|-----------|-----------|------|
| AFW | 5,207 | 5,207 | 10,414 |
| HELEN | 37,676 | 37,676 | 75,352 |
| IBUG | 1,786 | 1,786 | 3,572 |
| LFPW | 16,556 | 16,556 | 33,112 |
| **总计** | **61,225** | **61,225** | **122,450** |

### 3.5 RPY 角度定义

| 属性 | 值 |
|------|-----|
| **存储单位** | ⚠️ **弧度制** |
| **使用时需转换** | ✅ 需转换为角度 |
| **顺序** | [Pitch, Yaw, Roll] (在 `Pose_Para` 的前 3 个值) |
| **角度范围** | Yaw 扩展至 [-90°, +90°]，适合大姿态训练 |

### 3.6 标注文件格式

#### 参数文件 (.mat) - 位于子数据集目录

| 字段名 | 形状 | 说明 |
|--------|------|------|
| `Pose_Para` | (1, 7) | **姿态参数**: [pitch, yaw, roll, tx, ty, tz, scale] (弧度) |
| `roi` | (1, 4) | 人脸边界框 |
| `Shape_Para` | (199, 1) | 3DMM 形状参数 |
| `Exp_Para` | (29, 1) | 表情参数 |
| `pt2d` | (2, 68) | 68个 2D 关键点 (部分文件) |

#### 关键点文件 (*_pts.mat) - 位于 landmarks/ 目录

| 字段名 | 形状 | 说明 |
|--------|------|------|
| `pts_2d` | (68, 2) | 68个 2D 关键点坐标 |
| `pts_3d` | (68, 2) | 68个 3D 关键点坐标 (投影后) |

### 3.7 读取 RPY 角度

```python
import numpy as np
from scipy.io import loadmat

def parse_300wlp_mat(mat_path, landmark_path=None):
    """
    解析 300W-LP 的 MAT 标注文件
    
    Args:
        mat_path: 参数 .mat 文件路径
        landmark_path: 关键点 *_pts.mat 文件路径 (可选)
    
    返回:
        pose: [pitch, yaw, roll] 度数
        landmarks: (68, 2) 2D关键点 (如果提供了 landmark_path)
    """
    mat_data = loadmat(mat_path)
    
    # 提取姿态参数 (弧度制)
    pose_para = mat_data['Pose_Para'].flatten()
    pitch_rad, yaw_rad, roll_rad = pose_para[0], pose_para[1], pose_para[2]
    
    # 转换为度数
    pitch = np.degrees(pitch_rad)
    yaw = np.degrees(yaw_rad)
    roll = np.degrees(roll_rad)
    
    result = {'pose': np.array([pitch, yaw, roll])}
    
    # 加载关键点 (如果提供)
    if landmark_path:
        lm_data = loadmat(landmark_path)
        result['pts_2d'] = lm_data.get('pts_2d')
        result['pts_3d'] = lm_data.get('pts_3d')
    
    return result

# 使用示例
result = parse_300wlp_mat('AFW/sample.mat', 'landmarks/AFW/sample_pts.mat')
print(f"Pitch: {result['pose'][0]:.1f}°, Yaw: {result['pose'][1]:.1f}°, Roll: {result['pose'][2]:.1f}°")
```

### 3.8 注意事项

- ✅ 主要特点是 **Yaw 角度扩展至 ±90°**，适合训练大姿态模型
- ⚠️ **角度为弧度制**，需要转换
- 📂 关键点在单独的 `landmarks/` 目录，需要拼接文件名
- 🔄 翻转版本 (_Flip) 用于数据增强，翻转后角度符号可能需要调整

---

## 4. UPNA Synthetic Head Pose Database

### 4.1 基本信息

| 属性 | 值 |
|------|-----|
| **样本数** | 36,000 帧 (10用户 × 12视频 × 300帧) |
| **图像尺寸** | 1280×720 (视频帧) |
| **场景** | 合成视频 |
| **关键点** | 43个 2D 关键点 ✅ (iBUG 68点子集) |
| **深度图** | ❌ 无 |
| **边界框** | ✅ 提供 BBox [x, y, w, h] |
| **需预裁剪** | ✅ 需要使用 BBox 裁剪人脸区域 |

### 4.2 文件组织结构

```
Synthetic_Head_Pose_Database_UPNA/
├── Camera_parameters.mat     # 相机参数
├── tl.mat                    # 三角化参数
├── README.txt                # 数据格式说明
└── User_XX/                  # 10个用户 (01-10)
    ├── head_XX.mat           # 3D模型数据
    ├── user_XX_video_YY.mp4  # 视频文件 (12个/用户)
    ├── user_XX_video_YY_groundtruth3D.txt   # 3D姿态标注
    ├── user_XX_video_YY_groundtruth2D.txt   # 2D关键点标注  
    └── user_XX_video_YY_bbox.txt            # 边界框标注
```

### 4.3 RPY 角度定义

| 属性 | 值 |
|------|-----|
| **存储单位** | ✅ **角度制** (直接使用) |
| **顺序** | ⚠️ [Tx, Ty, Tz, **Roll, Yaw, Pitch**] (注意顺序不同!) |
| **角度范围** | Roll [-46°, 36°], Yaw [-43°, 36°], Pitch [-30°, 26°] |

### 4.4 标注文件格式

#### 3D 姿态标注 (*_groundtruth3D.txt)

每行对应一帧，共 300 行，6 列 (Tab 分隔):

```
Tx	Ty	Tz	Roll	Yaw	Pitch
```

| 索引 | 字段 | 单位 | 说明 |
|------|------|------|------|
| 0 | Tx | mm | 水平位移 |
| 1 | Ty | mm | 垂直位移 |
| 2 | Tz | mm | 深度距离 (约 424-927mm) |
| 3 | Roll | degrees | 滚转角 (歪头) |
| 4 | Yaw | degrees | 偏航角 (摇头) |
| 5 | Pitch | degrees | 俯仰角 (点头) |

#### 2D 关键点标注 (*_groundtruth2D.txt)

每行对应一帧，共 300 行，86 列 (43 个点 × 2 坐标):

```
x1 y1 x2 y2 ... x43 y43
```

#### 边界框标注 (*_bbox.txt)

每行对应一帧，共 300 行，4 列:

```
minX	minY	width	height
```

### 4.5 读取 RPY 角度

```python
import numpy as np
import cv2

def parse_upna_groundtruth3d(file_path):
    """
    解析 UPNA 3D ground truth 文件
    
    返回:
        data: (300, 6) - [Tx, Ty, Tz, Roll, Yaw, Pitch]
    """
    data = np.loadtxt(file_path, delimiter='\t')
    return data

def parse_upna_groundtruth2d(file_path):
    """
    解析 UPNA 2D 关键点文件
    
    返回:
        landmarks: (300, 43, 2) - 每帧 43 个关键点的 xy 坐标
    """
    data = np.loadtxt(file_path, delimiter='\t')
    return data.reshape(-1, 43, 2)

def parse_upna_bbox(file_path):
    """
    解析 UPNA bbox 文件
    
    返回:
        bboxes: (300, 4) - [x, y, width, height]
    """
    return np.loadtxt(file_path, delimiter='\t')

def extract_frame(video_path, frame_idx):
    """从视频中提取指定帧"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None

# 使用示例
gt3d = parse_upna_groundtruth3d('user_01_video_01_groundtruth3D.txt')
frame_idx = 0
tx, ty, tz, roll, yaw, pitch = gt3d[frame_idx]
print(f"Frame {frame_idx}: Roll={roll:.1f}°, Yaw={yaw:.1f}°, Pitch={pitch:.1f}°")
```

### 4.6 注意事项

- ⚠️ **角度顺序与其他数据集不同**: [Roll, Yaw, Pitch] 而非 [Pitch, Yaw, Roll]
- ✅ **角度直接为度数**，无需转换
- 📦 **需要使用 BBox 裁剪**，原始视频帧较大 (1280×720)
- 📐 43 个关键点是 iBUG 68 点的子集
- 🎥 数据来自合成视频，每视频 300 帧，共 120 个视频

---

## 5. 绘制姿态坐标轴

可视化头部姿态的 3D 坐标轴：

```python
import cv2
import numpy as np
from math import cos, sin

def draw_axis(img, pitch, yaw, roll, tdx, tdy, size=80):
    """
    在图像上绘制头部姿态的 3D 坐标轴
    
    Args:
        img: BGR 图像
        pitch, yaw, roll: 欧拉角 (弧度)
        tdx, tdy: 坐标轴原点 (通常是鼻尖位置)
        size: 坐标轴长度 (像素)
    
    坐标轴颜色:
        - 红色 (X): 指向右侧
        - 绿色 (Y): 指向下方
        - 蓝色 (Z): 指向前方 (从脸部射出)
    """
    img_draw = img.copy()
    yaw = -yaw  # 符号约定
    
    # X-Axis (红色)
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
    
    # Y-Axis (绿色)
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
    
    # Z-Axis (蓝色)
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy
    
    cv2.line(img_draw, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img_draw, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img_draw, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 3)
    
    return img_draw

# 使用示例 (注意: 需要将角度转换为弧度)
pitch_rad = np.radians(pitch_deg)
yaw_rad = np.radians(yaw_deg)
roll_rad = np.radians(roll_deg)
img_with_axes = draw_axis(img, pitch_rad, yaw_rad, roll_rad, nose_x, nose_y)
```

---

## 6. EDA 脚本

| 文件 | 说明 |
|------|------|
| `eda_aflw_dataset.ipynb` | AFLW2000-3D 数据集探索分析 |
| `eda_biwi_dataset.ipynb` | BIWI 数据集探索分析 |
| `300w-lp_eda.ipynb` | 300W-LP 数据集探索分析 |
| `upna_head_pose_eda.ipynb` | UPNA 数据集探索分析 |
| `head-pose-estimation.ipynb` | Kaggle 参考脚本 |

---

## 7. 参考链接

- [AFLW2000-3D 官方](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
- [BIWI Kinect Head Pose Database](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html)
- [300W-LP Paper](https://arxiv.org/abs/1511.07212) - "Face Alignment Across Large Poses: A 3D Solution" (CVPR 2016)
- [UPNA Head Pose Database](http://gi4e.unavarra.es/databases/hpdb/)
- [Kaggle Head Pose Estimation](https://www.kaggle.com/code/raedalkhatib/head-pose-estimation)

