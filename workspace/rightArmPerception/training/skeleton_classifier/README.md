# 基于 21 维手部关键点的手势分类训练

基于 21 个手部关键点的手势mlp分类模型训练与评测。

原始项目为https://github.com/kinivi/hand-gesture-recognition-mediapipe.git


## 项目内容

- 将手部关键点 JSON 标注转换为定长 skeleton 训练数据
- 基于 21 个关键点的时序特征训练手势分类模型
- 导出 `.keras` 和 `.tflite` 模型
- 对保存的 JSON 样本进行推理速度 benchmark

## 环境依赖

[environment.yml](environment.yml) 创建运行环境。

## 使用流程

### 1. 将 JSON 数据集转换为 skeleton CSV

使用 [tools/json_dataset_to_skeleton_csv.py](tools/json_dataset_to_skeleton_csv.py) 将原始 JSON 标注转换为训练集和验证集 CSV。


### 2. 训练 skeleton 分类模型

使用 [skeleton_classification_dataset_v1.py](skeleton_classification_dataset_v1.py) 进行训练。

示例：

```bash
python skeleton_classification_dataset_v1.py \
  --train-csv model/skeleton_classifier/skeleton_train.csv \
  --val-csv model/skeleton_classifier/skeleton_val.csv \
  --label-csv model/skeleton_classifier/skeleton_classifier_label.csv
```

该脚本会完成以下工作：

- 使用 16 帧、21 个关键点的时序特征训练分类模型
- 保存训练后的 Keras 模型
- 在验证集上进行评估
- 生成混淆矩阵图
- 导出 TFLite 模型

如果希望使用 LSTM 版本的模型头，可以执行：

```bash
python skeleton_classification_dataset_v1.py --use-lstm
```

### 3. 进行推理 FPS 测试

使用 [tools/benchmark_skeleton_inference_fps.py](tools/benchmark_skeleton_inference_fps.py) 测试模型推理速度。

TFLite 模型示例：

```bash
python tools/benchmark_skeleton_inference_fps.py \
  --model-path model/skeleton_classifier/skeleton_classifier_v1.tflite \
  --model-type tflite \
  --input-json /path/to/sample.json
```

Keras 模型示例：

```bash
python tools/benchmark_skeleton_inference_fps.py \
  --model-path model/skeleton_classifier/skeleton_classifier_v1.keras \
  --model-type keras \
  --input-json /path/to/sample.json
```

### 4. 统计原始 JSON 样本数量

使用 [tools/count_json_samples.py](tools/count_json_samples.py) 查看数据集各类别样本分布。

示例：

```bash
python tools/count_json_samples.py --dataset-root /path/to/dataset
```

## 数据表示方式

- 每个样本会被归一化到 `16` 个时间步
- 每一帧包含 `21` 个手部关键点
- 每个关键点使用 `x, y` 两个坐标表示
- 模型输入维度为 `16 x 21 x 2 = 672`

坐标会以前一序列首帧的第一个关键点为参考原点，并按图像宽高进行归一化。

