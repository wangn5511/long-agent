# 模型训练模块

本模块用于右臂感知相关模型的训练工作，包括模型定义、训练流程、验证评估等。

## 主要功能

- 模型架构定义与配置
- 训练流程管理
- 模型验证与评估
- 训练日志与可视化

## mmaction2
包含 mmaction2 官方代码库的 clone 版本，其中做了细微的改动，以配合右臂产品的动态手势识别模型训练工作。

> 官方仓库：https://github.com/open-mmlab/mmaction2

主要功能：
- 动态手势识别模型的训练与验证
- 视频分类模型（如 TSM）的训练流程
- 数据集预处理与数据增强
- 模型评估与可视化

启动训练：
```bash
# 训练入口文件：
configs/recognition/tsm/tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb-cytoderm-gesture_pipeline4_large_lr.py

# 启动方式和mm训练系列一致
```


### TSM 训练：
TSM的训练分为两个阶段，分别是：
1. 第一阶段：使用全部27个类别的数据集进行预训练，训练完成后，模型会保存到 `work_dirs/tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb-cytoderm-gesture_pipeline4_v1v2_27_class_jester` 目录下。
2. 第二阶段：使用2个类别（wave in和wave out）的数据集进行训练，训练完成后，模型会保存到 `work_dirs/tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb-cytoderm-gesture_pipeline4_v1v2_2_class_stage_1_clip_none_from_17_jester` 目录下。

单卡启动命令如下：
```bash
# 第一阶段训练 
python tools/train.py configs/recognition/tsm/tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb-cytoderm-gesture_pipeline4_v1v2_27_class_jester.py

# 第二阶段训练
python tools/train.py configs/recognition/tsm/tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb-cytoderm-gesture_pipeline4_v1v2_2_class_stage_1_clip_none_from_17_jester.py
```

多卡启动命令如下：
```bash
# 第一阶段训练 8卡训练
bash tools/dist_train.sh configs/recognition/tsm/tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb-cytoderm-gesture_pipeline4_v1v2_27_class_jester.py 8

# 第二阶段训练 8卡训练
bash tools/dist_train.sh configs/recognition/tsm/tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb-cytoderm-gesture_pipeline4_v1v2_2_class_stage_1_clip_none_from_17_jester.py 8
```

训练注意事项：
- 每次训练需要根据实际情况修改配置文件中的数据集配置 dataset_configs 、类别数目 num_classes 、类别与idx映射表 class_mapping;
- 第二阶段训练需要根据第一阶段训练的模型路径，修改配置文件中的 load_from 字段。


## ultralytics：
基于 详细请见training/ultralytics/README.md; 


## mobilenetv2训练流程
目前训练数据、预训练权重文件、docker均在134这台机子上

详见 training/mobilenetv2/训练流程.md


## skeleton_classifier:
基于 21 个手部关键点的手势mlp分类模型训练与评测。