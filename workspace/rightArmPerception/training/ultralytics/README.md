# ultralytics：
包含 ultralytics 官方代码库的 clone 版本，其中做了细微的改动，以配合右臂产品的head、hand检测模型训练和hand-pose模型训练工作

> 官方仓库：GitHub - ultralytics/ultralytics: Ultralytics YOLO 🚀

主要功能：
- hand与head检测模型的训练与验证
- hand、head检测、hand-pose模型的训练与验证
- 数据集预处理与数据增强
- 模型评估与可视化

具体的脚本使用说明见[fanchen_README.md](./ultralytics/fanchen_README.md)文件

hand、head检测模型训练启动命令如下：
```bash
# 单卡启动
python ultralytics/my_train.py --dataset ultralytics/cfg/datasets/open_images_hand_head.yaml --model ultralytics/cfg/models/12/yolo12n-head_hand.yaml --weights yolov8n.pt --device 1 --batch 32

# 多卡启动
python ultralytics/my_train.py --dataset ultralytics/cfg/datasets/open_images_hand_head.yaml --model ultralytics/cfg/models/12/yolo12n-head_hand.yaml --weights yolov8n.pt --device 0,1,2,3 --batch 128
```
注意：
- 多卡启动时，需要设置 `PYTHONPATH` 环境变量，指向 `training/ultralytics` 目录。
    ```bash
    export PYTHONPATH=$PYTHONPATH:rightArmPerception/training/ultralytics
    ```
- ultralytics的数据集目录结构如下：
    ```
    ├── open_images_hand_head_dataset
        ├── images
        │   ├── test
        │   ├── train
        │   └── val
        └── labels
            ├── test
            ├── train
            └── val
    ```
- 训练时需更改数据配置文件中的 `path` 变量：
  - 火山云数据集路径：`/home/datasets/gesture_dataset/open_images_hand_head_dataset`
  - 腾讯云数据集路径：`/home/datasets/rest_gesture_dataset/open_images_hand_head_dataset`

hand、head检测、hand-pose模型的训练启动命令如下：
```bash
# 单卡启动
python ultralytics/my_train_load_before_keypoints.py --dataset ultralytics/cfg/datasets/open_images_hand_head.yaml --model ultralytics/cfg/models/12/yolo12n-pose_my_add_RTMO.yaml --weights yolov12n-pose_my_add_RTMO.pt --device 1 --batch 32

# 多卡启动
python ultralytics/my_train_load_before_keypoints.py --dataset ultralytics/cfg/datasets/open_images_hand_head.yaml --model ultralytics/cfg/models/12/yolo12n-pose_my_add_RTMO.yaml --weights yolov12n-pose_my_add_RTMO.pt --device 0,1,2,3 --batch 128
```
注意：
- 在启动pose训练前，需要先训练hand、head检测模型，获取预训练权重 `yolov12n-origion.pt`; 之后借助`add_hand_head_pretrain_weight.py`脚本，将预训练权重添加到pose模型的权重中，以获得 `yolov12n-pose_my_add_RTMO.pt` 权重。
- 多卡启动时，需要设置 `PYTHONPATH` 环境变量，指向 `training/ultralytics` 目录。
    ```bash
    export PYTHONPATH=$PYTHONPATH:rightArmPerception/training/ultralytics
    ```
- ultralytics的数据集目录结构如下：
    ```
    ├── open_images_hand_head_dataset
        ├── images
        │   ├── test
        │   ├── train
        │   └── val
        └── labels
            ├── test
            ├── train
            └── val
    ```
- 在多数据集训练时，需检查数据集目录结构是否一致，包括 `images` 和 `labels` 目录下的子目录结构。
