


          
# 脚本功能说明文档


## 1. my_train.py - 目标检测训练脚本

该脚本用于训练通用的目标检测模型。

### 主要功能
- **模型训练**：在手部和头部检测数据集上训练YOLO模型
- **模型验证**：训练完成后自动验证模型性能
- **结果保存**：保存训练过程中的模型权重和日志

### 配置参数
- epochs: 300轮训练
- imgsz: 640x640图像尺寸
- batch: 1024批次大小
- device: 使用4个GPU([0, 1, 2, 3])   （多个device时，默认DDP训练，需要设置环境变量`PYTHONPATH`，让python找到ultralytics文件）

### 使用方式
```bash
python my_train.py   
```
注意训练过程中调整对应的模型配置文件和数据集配置文件。
- 模型配置文件：`ultralytics/cfg/models/12/yolo12-head_hand.yaml`
- 数据集配置文件：`ultralytics/cfg/datasets/open_images_hand_head.yaml`

## 2. my_pred.py - 目标检测推理脚本

该脚本用于执行通用的目标检测推理任务。

### 主要功能
- **模型推理**：加载训练好的目标检测模型，对图像进行推理
- **结果可视化**：绘制检测框和标签
- **结果保存**：将检测结果保存到指定目录

### 核心函数
- `inference_single_image()`：执行单张图像的目标检测推理

### 配置参数
- `model_path`: 预训练的YOLO模型路径（如yolov8n.pt）
- `image_path`: 输入的单张图像路径（如test.jpg）


### 使用方式
```bash
python my_pred.py
```

## 3. my_test.py - 模型测试和性能评估脚本

该脚本用于全面测试和评估YOLO模型的性能。

### 主要功能
- **模型测试**：支持多种输入源（图像、视频、摄像头）的检测测试
- **性能评估**：计算模型的参数量、计算量(FLOPs)和推理时间
- **指标验证**：计算mAP、Precision、Recall等评估指标

### 核心函数
- `test_model()`：测试YOLO模型在不同数据源上的表现
- `validate_model()`：验证模型性能，包括计算量统计和指标评估
- `measure_inference_time()`：测量模型推理时间
- `get_model_compute_stats()`：获取模型计算量和参数量

### 使用方式
```bash
python my_test.py --model [模型路径] --source [数据源] --conf [置信度阈值] --save --show
```




## 4. pth_2_onnx.py - 模型格式转换脚本

该脚本用于将PyTorch(.pt)格式的YOLO模型转换为ONNX格式。

### 主要功能
- **格式转换**：将.pth模型文件转换为.onnx格式
- **模型导出**：使用YOLO内置的export方法进行转换
- **路径管理**：自动处理输出路径和目录创建

### 核心函数
- `convert_yolo_to_onnx()`：执行YOLO到ONNX的转换

### 技术特点
- 支持指定输入图像尺寸
- 自动生成简化版ONNX模型
- 自动重命名和移动生成的ONNX文件

### 使用方式
```bash
python pth_2_onnx.py
```

## 总结

这组脚本构成了一个完整的YOLO模型开发和部署流程：

1. **训练阶段**：使用`my_train.py`进行head和hand目标检测模型训练
2. **测试评估**：使用`my_test.py`对模型进行全面的性能评估
3. **推理应用**：使用`my_pred.py`进行实际的检测推理
4. **模型部署**：使用`pth_2_onnx.py`将模型转换为ONNX格式以便部署到生产环境

        