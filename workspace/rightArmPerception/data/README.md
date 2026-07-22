# 数据处理模块

本模块用于右臂感知相关的数据处理工作，包括数据集管理、数据预处理、数据增强及标注工具等。

## 主要功能

- 数据集下载与管理
- 数据预处理流程
- 数据增强策略
- 标注工具与格式转换

## 手势识别录制数据处理
```bash

# 原始数据位置：

# 1209 D405 采集1：
# - 腾讯云存储 cos://cytoderm-ai-shanghai-1389948562/datasets/rest_gesture_dataset/datasets/rest_gesture_dataset/gesture_v1_new/videos.tar
# - 腾讯云机器 /home/datasets/gesture_dataset/gesture_raw_data/right_arm_collection/experimental_collection_v1/videos
# - 标注数据参考: /home/datasets/rest_gesture_dataset/gesture_v1_new/gesture_v1_train.txt
# - 外包标注人员修正 (仅部分): cos://cytoderm-ai-shanghai-1389948562/datasets/rest_gesture_dataset/datasets/rest_gesture_dataset/第一批V1手势-更新.xlsx

# 1209 D405 采集2：cos://cytoderm-ai-shanghai-1389948562/datasets/rest_gesture_dataset/datasets/rest_gesture_dataset/gesture_v2_new/videos.tar
# - 腾讯云存储 cos://cytoderm-ai-shanghai-1389948562/datasets/rest_gesture_dataset/datasets/rest_gesture_dataset/gesture_v2_new/videos.tar
# - 腾讯云机器 /home/datasets/gesture_dataset/gesture_raw_data/right_arm_collection/experimental_collection_v2/videos
# - 标注数据参考 ( 101.126.137.134 )：/home/datasets/gesture_dataset/v2/gesture_v2_train.txt
# - 外包标注人员修正 (仅部分): cos://cytoderm-ai-shanghai-1389948562/datasets/rest_gesture_dataset/datasets/rest_gesture_dataset/第一批V2手势-更新.xlsx

# 1209 D405 数据 外包标注人员切分（片段切分结果）
# - cos://cytoderm-ai-shanghai-1389948562/datasets/rest_gesture_dataset/datasets/rest_gesture_dataset/batch1-更新-图片.xlsx1.xlsx
# - cos://cytoderm-ai-shanghai-1389948562/datasets/rest_gesture_dataset/datasets/rest_gesture_dataset/batch2.xlsx1.xlsx

# 标注人员采集
# - 腾讯云存储 cos://cytoderm-ai-shanghai-1389948562/datasets/rest_gesture_dataset/datasets/rest_gesture_dataset/batch1_process/images
# - 腾讯云机器 /home/datasets/rest_gesture_dataset/batch1_process/images
# - 腾讯云存储 cos://cytoderm-ai-shanghai-1389948562/datasets/rest_gesture_dataset/datasets/rest_gesture_dataset/batch2_process/images
# - 腾讯云机器 /home/datasets/rest_gesture_dataset/batch2_process/images

# cybopal 世豪采集
# base 相机
# - 腾讯云存储 cos://cytoderm-ai-shanghai-1389948562/datasets/rightArm/cybopal_camera_base_recordings.tar
# - 腾讯云机器 /home/datasets/gesture_dataset/gesture_raw_data/right_arm_collection/cybopal_collection/cybopal_camera_base_recordings
# top 相机
# - 腾讯云存储 cos://cytoderm-ai-shanghai-1389948562/datasets/rightArm/cybopal_camera_top_recordings.tar
# - 腾讯云机器 /home/datasets/gesture_dataset/gesture_raw_data/right_arm_collection/cybopal_collection/cybopal_camera_top_recordings

# 【参考】将录制的原始mp4 抽帧转换为 jpg
python processing.py prepare_gesture_image_data --src_dir=/home/datasets/gesture_dataset/gesture_raw_data/right_arm_collection/experimental_collection_v1/videos --dst_dir=/home/datasets/gesture_dataset/gesture_raw_data/right_arm_collection/experimental_collection_v1/images --skip_convert=True

# 【参考】将标签和数据制定为mmaction2 的官方dataset接受格式
python processing.py prepare_mmaction_rawframe_dataset_file  # TODO: 待实现
# 参考 /home/datasets/rest_gesture_dataset/gesture_v1_new/gesture_v1_train.txt 和 官方dataset实现 training/mmaction2/mmaction/datasets/rawframe_dataset.py —— RawframeDataset

```