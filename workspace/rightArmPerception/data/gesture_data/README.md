# 手势数据集 (Gesture Dataset)

## 1. 数据概述

目前共有 **4 个数据来源**，均为视频数据，包含以下 12 个手势类别：

| 手势 | 说明 |
|------|------|
| BYEBYE | 再见 |
| FIST_MOVE | 握拳移动 |
| NONE | 无手势 |
| NUMBER | 数字 |
| OK | OK 手势 |
| THUMBS_UP | 竖大拇指 |
| WAVE_DOWN | 向下挥手 |
| WAVE_IN | 向内挥手 |
| WAVE_LEFT | 向左挥手 |
| WAVE_OUT | 向外挥手 |
| WAVE_RIGHT | 向右挥手 |
| WAVE_UP | 向上挥手 |

> **注意**：所有数据源均为原始视频，使用前需要通过 `processing.py` 提取视频帧（见第 4 节）。

## 2. 数据集总览（TOS / COS）

| # | 数据来源 | 采集方式 | 视频数据 (TOS) | 视频数据 (COS) | 标注数据 (TOS) | 标注数据 (COS) |
|---|---------|---------|---------------|---------------|---------------|---------------|
| 1 | 搜狐大厦员工数采 (v1) | 手持设备录制 | `tos://cybopal/sohu_1209_gesture_collection/v1/videos.tar` | `cos://cytoderm-ai-shanghai-1389948562/datasets/rest_gesture_dataset/datasets/rest_gesture_dataset/gesture_v1_new/videos.tar` | `tos://cybopal/sohu_1209_gesture_collection/gesture_merged_annotation.csv`（v1+v2 合并） | `cos://cytoderm-ai-shanghai-1389948562/datasets/rest_gesture_dataset/datasets/rest_gesture_dataset/gesture_merged_annotation.csv` |
| 2 | 搜狐大厦员工数采 (v2) | 手持设备录制 | `tos://cybopal/sohu_1209_gesture_collection/v2/videos.tar` | `cos://cytoderm-ai-shanghai-1389948562/datasets/rest_gesture_dataset/datasets/rest_gesture_dataset/gesture_v2_new/videos.tar` | 同上 | 同上 |
| 3 | 数据堂外采 | 外部数据公司采集 | `tos://cybopal/shujutang_collection/gesture_collection_shujutang.tar` | `cos://cytoderm-ai-shanghai-1389948562/datasets/rightArm/gesture_collection_shujutang.tar` | `tos://cybopal/shujutang_collection/gesture_collection_shujutang_annotation.xlsx`（原始） | `cos://cytoderm-ai-shanghai-1389948562/datasets/rightArm/gesture_collection_shujutang_annotation.csv` |
| 4 | ONE v0 真机摄像头采集 | cybopal ONE v0 机器人摄像头 | `tos://cybopal/ONE_v0/video_dataset/shihao_collection_sohu/cybopal_camera_base_recordings.tar`（底部）<br>`tos://cybopal/ONE_v0/video_dataset/shihao_collection_sohu/cybopal_camera_top_recordings.tar`（顶部） | `cos://cytoderm-ai-shanghai-1389948562/datasets/rightArm/cybopal_camera_base_recordings.tar`（底部）<br>`cos://cytoderm-ai-shanghai-1389948562/datasets/rightArm/cybopal_camera_top_recordings.tar`（顶部） | `tos://cybopal/ONE_v0/video_dataset/shihao_collection_sohu/annotations.csv` | `cos://cytoderm-ai-shanghai-1389948562/datasets/rightArm/annotations.csv` |

### 2.1 统一标注 CSV 格式（所有手势数据）

无论原始标注来源是 `.xlsx` 还是 `.csv`，训练和数据处理前统一转换为以下 CSV 列结构：

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `视频名` | string | 视频文件名（带后缀） | `recording_20250902_152617.mp4` |
| `起始帧号` | int | 片段起始帧（闭区间） | `46` |
| `终止帧号` | int | 片段终止帧（闭区间，且 `终止帧号 >= 起始帧号`） | `66` |
| `类别名` | string | 手势类别名 | `WAVE_IN` |

统一 CSV 示例：

```csv
视频名,起始帧号,终止帧号,类别名
recording_20250902_152617.mp4,46,66,BYEBYE
001-0003.mp4,247,256,WAVE_IN
001-0003.mp4,346,355,WAVE_OUT
```

## 3. 各数据集详情

### 3.1 搜狐大厦员工数采 (v1 & v2)

两次在搜狐大厦进行的员工手势数据采集。

- 数采流程文档：https://v7lgmr65fj.feishu.cn/wiki/ETXswro66iswLrkWv6rcv6utnTh
- 标注规范文档：https://v7lgmr65fj.feishu.cn/wiki/MhDNwqKtMilWXWksEpYcIhVYnwg

视频数据：

- **TOS v1**: `tos://cybopal/sohu_1209_gesture_collection/v1/videos.tar`
- **COS v1**: `cos://cytoderm-ai-shanghai-1389948562/datasets/rest_gesture_dataset/datasets/rest_gesture_dataset/gesture_v1_new/videos.tar`
- **TOS v2**: `tos://cybopal/sohu_1209_gesture_collection/v2/videos.tar`
- **COS v2**: `cos://cytoderm-ai-shanghai-1389948562/datasets/rest_gesture_dataset/datasets/rest_gesture_dataset/gesture_v2_new/videos.tar`

标注结果（v1 + v2 合并，由标注公司完成视频精确切割标注）：

- **TOS**: `tos://cybopal/sohu_1209_gesture_collection/gesture_merged_annotation.csv`
- **COS**: `cos://cytoderm-ai-shanghai-1389948562/datasets/rest_gesture_dataset/datasets/rest_gesture_dataset/gesture_merged_annotation.csv`

原始标签统计如下。

**v1**

| 标签 | 数量 |
|------|------|
| BYEBYE | 18 |
| NUMBER | 1 |
| OK | 21 |
| THUMBS_UP | 7 |
| WAVE_DOWN | 24 |
| WAVE_IN | 24 |
| WAVE_LEFT | 24 |
| WAVE_OUT | 22 |
| WAVE_RIGHT | 24 |
| WAVE_UP | 24 |

**v2**

| 标签 | 数量 |
|------|------|
| BYEBYE | 18 |
| OK | 18 |
| THUMBS_UP | 18 |
| WAVE_DOWN | 24 |
| WAVE_IN | 24 |
| WAVE_LEFT | 24 |
| WAVE_OUT | 24 |
| WAVE_RIGHT | 24 |
| WAVE_UP | 24 |

解压后的目录结构：

```
videos/
├── BYEBYE/
│   ├── recording_20250902_152101.mp4
│   ├── recording_20250902_152222.mp4
│   └── ...
├── FIST_MOVE/
│   └── ...
├── NONE/
│   └── ...
├── NUMBER/
│   └── ...
├── OK/
│   └── ...
├── THUMBS_UP/
│   └── ...
├── WAVE_DOWN/
│   └── ...
├── WAVE_IN/
│   └── ...
├── WAVE_LEFT/
│   └── ...
├── WAVE_OUT/
│   └── ...
├── WAVE_RIGHT/
│   └── ...
└── WAVE_UP/
    └── ...
```

### 3.2 数据堂外采

由数据堂公司采集的手势数据及对应标注：

- 视频数据：
  - **TOS**: `tos://cybopal/shujutang_collection/gesture_collection_shujutang.tar`
  - **COS**: `cos://cytoderm-ai-shanghai-1389948562/datasets/rightArm/gesture_collection_shujutang.tar`
- 标注结果：
  - **TOS（原始）**: `tos://cybopal/shujutang_collection/gesture_collection_shujutang_annotation.xlsx`
  - **COS（统一 CSV）**: `cos://cytoderm-ai-shanghai-1389948562/datasets/rightArm/gesture_collection_shujutang_annotation.csv`

原始标签统计如下。

| 标签 | 数量 |
|------|------|
| BYEBYE | 86 |
| OK | 88 |
| THUMBS_UP | 87 |
| WAVE_DOWN | 86 |
| WAVE_IN | 87 |
| WAVE_LEFT | 87 |
| WAVE_OUT | 87 |
| WAVE_RIGHT | 88 |
| WAVE_UP | 86 |

备注：

- 数据堂原始标注里曾出现过 `THUNBS UP` 这类非标准写法，当前文档已统一记为 `THUMBS_UP`
- 后续需要同步检查并处理云上的原始标注/数据目录命名，确保线上数据与本文档统计口径一致

### 3.3 ONE v0 真机摄像头采集

针对 cybopal ONE v0 版本，在搜狐大厦使用机器人自身摄像头进行的数据采集，包含底部和顶部两个摄像头视角：

- 底部摄像头视频：
  - **TOS**: `tos://cybopal/ONE_v0/video_dataset/shihao_collection_sohu/cybopal_camera_base_recordings.tar`
  - **COS**: `cos://cytoderm-ai-shanghai-1389948562/datasets/rightArm/cybopal_camera_base_recordings.tar`
- 顶部摄像头视频：
  - **TOS**: `tos://cybopal/ONE_v0/video_dataset/shihao_collection_sohu/cybopal_camera_top_recordings.tar`
  - **COS**: `cos://cytoderm-ai-shanghai-1389948562/datasets/rightArm/cybopal_camera_top_recordings.tar`
- 标注结果（视频动作精确切割，统一 CSV）：
  - **TOS**: `tos://cybopal/ONE_v0/video_dataset/shihao_collection_sohu/annotations.csv`
  - **COS**: `cos://cytoderm-ai-shanghai-1389948562/datasets/rightArm/annotations.csv`

ONE v0 使用另一套原始标签体系，统计如下：

| 标签 | 数量 |
|------|------|
| FIST | 263 |
| OK | 264 |
| THUMBS_UP | 264 |
| WAVE_IN | 266 |
| WAVE_OUT | 264 |
| ONE | 265 |
| TWO | 268 |
| THREE | 266 |
| FOUR | 265 |
| FIVE | 266 |
| GUN | 267 |

## 4. 数据处理：提取视频帧

所有数据集均为原始视频格式，训练或分析前需使用 `processing.py` 将视频逐帧提取为 PNG 图片。

### 4.1 前置依赖

- Python 3
- ffmpeg（需在系统 PATH 中可用）

### 4.2 使用方法

```bash
# 默认模式（跳过格式转换，直接抽帧）
python processing.py extract_frames \
    --src_dir <视频目录> \
    --dst_dir <帧输出目录>

# 示例：提取 v1 视频帧
python processing.py extract_frames \
    --src_dir /home/models/rightArm_collection/self_gesture_dataset/v1/videos \
    --dst_dir /home/models/rightArm_collection/self_gesture_dataset/v1/images
```

如果原始视频编码格式不兼容，可加 `--no_skip_convert` 先用 libx264 + aac 转码再抽帧：

```bash
python processing.py extract_frames \
    --src_dir /home/models/rightArm_collection/self_gesture_dataset/v1/videos \
    --dst_dir /home/models/rightArm_collection/self_gesture_dataset/v1/images \
    --no_skip_convert
```

### 4.3 输入输出结构

```
# 输入（视频）
<src_dir>/<label>/<video>.mp4

# 输出（帧图片）
<dst_dir>/<label>/<video>/frame_0001.png
<dst_dir>/<label>/<video>/frame_0002.png
...
```

示例：

```
# 输入
videos/NONE/recording_20250902_152101.mp4

# 输出
images/NONE/recording_20250902_152101/frame_0001.png
images/NONE/recording_20250902_152101/frame_0002.png
...
```

### 4.4 注意事项

- 已存在的视频帧目录（非空）会自动跳过，不会重复抽帧
- 使用 `--no_skip_convert` 时，转码后的视频会存放在 `<dst_dir>_h264_aac/` 目录下
