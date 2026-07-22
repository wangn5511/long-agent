# hand21 数据集与处理流程

## 1. 数据概述

本目录维护 hand21 骨架手势数据准备链路里使用的自定义脚本，覆盖以下阶段：

- 原始视频级 train/val split 生成
- 标注覆盖检查
- 按标注切片并提取 hand21 关键点
- 标签白名单清洗
- 从长视频池挖掘 `NONE` 片段
- 数据集构造完成后的格式转换边界步骤

当前主要脚本如下：

| 脚本 | 作用 |
|------|------|
| `generate_hand21_video_split.py` | 基于标注和额外长视频池生成 source-video 级 train/val split |
| `report_annotation_video_name_coverage.py` | 检查标注中的视频名是否都能在视频目录中找到 |
| `prepare_hand21_from_excel_segments.py` | 单进程按标注裁剪、提取并生成 hand21 clip JSON |
| `prepare_hand21_from_excel_segments_parallel.py` | 多进程版本的数据准备入口 |
| `filter_prepared_hand21_by_label_whitelist.py` | 对 prepared 数据集做标签白名单清洗 |
| `mine_none_clips_from_long_videos.py` | 从长视频池中额外挖掘 `NONE` 样本 |
| `convert_hand21_json_to_pkl.py` | 将 prepared hand21 JSON 转成 MMAction2 `pkl`

## 2. 数据集总览（COS）

当前 hand21 数据集构造链路对应的云上数据来自：

- `cos://cytoderm-ai-shanghai-1389948562/datasets/dynamic_gestures_2603/`

| KEY | SIZE | 说明 |
|-----|------|------|
| `datasets/dynamic_gestures_2603/annotations.csv` | `160.72 KB` | 主标注文件 |
| `datasets/dynamic_gestures_2603/dynamic_gestures_260319.tar.gz` | `9.74 GB` | 主数据包之一 |
| `datasets/dynamic_gestures_2603/dynamic_gestures_260325.tar.gz` | `18.94 GB` | 主数据包之一 |
| `datasets/dynamic_gestures_2603/dynamic_gestures_260327.tar.gz` | `13.35 GB` | 主数据包之一 |
| `datasets/dynamic_gestures_2603/dynamic_gestures_260330.tar.gz` | `9.92 GB` | 主数据包之一 |
| `datasets/dynamic_gestures_2603/dynamic_gestures_260331.tar.gz` | `8.96 GB` | 主数据包之一 |
| `datasets/dynamic_gestures_2603/dynamic_gestures_260413_NONE.tar.gz` | `2.84 GB` | `NONE` 长视频池 |
| `datasets/dynamic_gestures_2603/dynamic_gestures_260415_NONE.tar.gz` | `788.39 MB` | 补充 `NONE` 长视频池 |
| `datasets/dynamic_gestures_2603/dynamic_gestures_260416_low_light_val.tar.gz` | `3.07 GB` | 低照度验证集补充视频 |

## 3. 统一标注与标签约定

### 3.1 统一标注 CSV 格式

当前流程使用 `compact_4col` 标注格式，对应 4 列：

| 列顺序 | 说明 |
|--------|------|
| 第 1 列 | 视频名；若为空则继承上一条非空视频名 |
| 第 2 列 | 起始帧 |
| 第 3 列 | 结束帧 |
| 第 4 列 | 标签名 |

示例：

```csv
recording_20260416_101023.mp4,46,66,WAVE_IN
,88,110,CLICK
recording_20260416_101155.mp4,25,40,NONE
```

### 3.2 标签约定


```text
WAVE_IN FIVE CLICK ROTATE SLIDE_DOWN SLIDE_UP OK CURSOR NONE
```

注意：

- `--allowed-labels` 请始终与项目实际使用的类别集合保持一致

## 4. 目录职责与变量约定

本目录聚焦 hand21 数据集的构造过程，不展开模型训练、评估、导出等后续步骤。

数据集构造阶段：

- `generate_hand21_video_split.py`
- `report_annotation_video_name_coverage.py`
- `prepare_hand21_from_excel_segments.py`
- `prepare_hand21_from_excel_segments_parallel.py`
- `filter_prepared_hand21_by_label_whitelist.py`
- `mine_none_clips_from_long_videos.py`


 `convert_hand21_json_to_pkl.py` 交付最终 hand21 数据`pkl`。


| 变量 | 含义 |
|------|------|
| `PROJECT_ROOT` | 项目根目录 |
| `DATA_ROOT` | 数据集落盘根目录 |
| `ANNOTATION_ROOT` | 主标注视频目录 |
| `ANNOTATION_FILE` | 主标注 CSV 路径 |
| `SPLIT_ROOT` | split 文件输出目录 |
| `NONE_POOL_ROOT` | `NONE` 长视频池目录 |
| `PREPARED_ROOT` | hand21 prepared 数据输出目录 |

示例：

```bash
export PROJECT_ROOT=/path/to/rightArmPerception
export DATA_ROOT=/path/to/data_root
export ANNOTATION_ROOT=${DATA_ROOT}/annotation
export ANNOTATION_FILE=${ANNOTATION_ROOT}/annotations.csv
export SPLIT_ROOT=${DATA_ROOT}/video_split
export NONE_POOL_ROOT=${DATA_ROOT}/dynamic_gestures_260413_NONE
export PREPARED_ROOT=${DATA_ROOT}/gesture_class9
```

## 5. 数据处理流程


### 5.1 生成原始视频级 train/val split

```bash
python ${PROJECT_ROOT}/data/hand21/generate_hand21_video_split.py \
  --video-root ${ANNOTATION_ROOT} \
  --annotation-file ${ANNOTATION_FILE} \
  --output-json ${SPLIT_ROOT}/video_split.json \
  --annotation-format compact_4col \
  --frame-base 1 \
  --val-ratio 0.2 \
  --seed 42 \
  --extra-video-root ${NONE_POOL_ROOT}
```

产物：

- `${SPLIT_ROOT}/video_split.json`
- `${SPLIT_ROOT}/train_video_names.txt`
- `${SPLIT_ROOT}/val_video_names.txt`

### 5.2 检查标注覆盖情况

```bash
python ${PROJECT_ROOT}/data/hand21/report_annotation_video_name_coverage.py \
  --annotation-file ${ANNOTATION_FILE} \
  --video-root ${ANNOTATION_ROOT} \
  --output-csv ${ANNOTATION_ROOT}/annotation_video_name_coverage_report.csv \
  --summary-txt ${ANNOTATION_ROOT}/annotation_video_name_coverage_summary.txt \
  --annotated-missing-txt ${ANNOTATION_ROOT}/annotated_but_missing_on_disk.txt \
  --unannotated-videos-txt ${ANNOTATION_ROOT}/unannotated_video_names.txt
```

建议先修正漏标、错名、落盘缺失等问题，再进入 prepare 阶段。

### 5.3 准备 train 数据

```bash
python ${PROJECT_ROOT}/data/hand21/prepare_hand21_from_excel_segments_parallel.py \
  --video-root ${ANNOTATION_ROOT} \
  --annotation-file ${ANNOTATION_FILE} \
  --output-root ${PREPARED_ROOT}/gesture_class9_prepared_train \
  --crop-first \
  --annotation-format compact_4col \
  --frame-base 1 \
  --min-frames 8 \
  --min-valid-frames 6 \
  --min-valid-ratio 0.5 \
  --low-quality-valid-ratio 0.9 \
  --other-track-coverage-threshold 0.3 \
  --max-consecutive-missing-threshold 5 \
  --max-interpolation-gap 5 \
  --video-split-json ${SPLIT_ROOT}/video_split.json \
  --split-name train \
  --jobs 4
```

### 5.4 准备 val 数据

```bash
python ${PROJECT_ROOT}/data/hand21/prepare_hand21_from_excel_segments_parallel.py \
  --video-root ${ANNOTATION_ROOT} \
  --annotation-file ${ANNOTATION_FILE} \
  --output-root ${PREPARED_ROOT}/gesture_class9_prepared_val \
  --crop-first \
  --annotation-format compact_4col \
  --frame-base 1 \
  --min-frames 8 \
  --min-valid-frames 6 \
  --min-valid-ratio 0.5 \
  --low-quality-valid-ratio 0.9 \
  --other-track-coverage-threshold 0.3 \
  --max-consecutive-missing-threshold 5 \
  --max-interpolation-gap 5 \
  --video-split-json ${SPLIT_ROOT}/video_split.json \
  --split-name val \
  --jobs 4
```

### 5.5 清理错误标签

```bash
python ${PROJECT_ROOT}/data/hand21/filter_prepared_hand21_by_label_whitelist.py \
  --prepared-roots \
    ${PREPARED_ROOT}/gesture_class9_prepared_train \
    ${PREPARED_ROOT}/gesture_class9_prepared_val \
  --allowed-labels \
    WAVE_IN FIVE CLICK ROTATE SLIDE_DOWN SLIDE_UP OK CURSOR NONE \
  --cleanup-files
```

如果只想先检查将要删除哪些样本，可临时加上 `--dry-run`。

### 5.6 从长视频池挖掘 `NONE` train

```bash
python ${PROJECT_ROOT}/data/hand21/mine_none_clips_from_long_videos.py \
  --video-split-json ${SPLIT_ROOT}/video_split.json \
  --split-name train \
  --output-root ${PREPARED_ROOT}/none_prepared_train \
  --jobs 4 \
  --source-kind extra_video_pool \
  --source-groups $(basename "${NONE_POOL_ROOT}") \
  --label NONE \
  --crop-first \
  --window-seconds-min 2 \
  --window-seconds-max 4 \
  --stride-seconds 1 \
  --sample-windows-per-video 500 \
  --min-valid-frames 4 \
  --min-valid-ratio 0.3 \
  --low-quality-valid-ratio 0.7 \
  --other-track-coverage-threshold 0.3 \
  --max-consecutive-missing-threshold 12 \
  --max-interpolation-gap 12
```

### 5.7 从长视频池挖掘 `NONE` val

```bash
python ${PROJECT_ROOT}/data/hand21/mine_none_clips_from_long_videos.py \
  --video-split-json ${SPLIT_ROOT}/video_split.json \
  --split-name val \
  --output-root ${PREPARED_ROOT}/none_prepared_val \
  --jobs 4 \
  --source-kind extra_video_pool \
  --source-groups $(basename "${NONE_POOL_ROOT}") \
  --label NONE \
  --crop-first \
  --window-seconds-min 2 \
  --window-seconds-max 4 \
  --stride-seconds 1 \
  --sample-windows-per-video 500 \
  --min-valid-frames 4 \
  --min-valid-ratio 0.3 \
  --low-quality-valid-ratio 0.7 \
  --other-track-coverage-threshold 0.3 \
  --max-consecutive-missing-threshold 12 \
  --max-interpolation-gap 12
```

### 5.8 转成 MMAction2 的 `pkl`

输出最终数据集文件：

- `${PREPARED_ROOT}/gesture_class9_hand21.pkl`

执行转换：

```bash
python ${PROJECT_ROOT}/data/hand21/convert_hand21_json_to_pkl.py \
  --train-root ${PREPARED_ROOT}/gesture_class9_prepared_train \
  --val-root ${PREPARED_ROOT}/gesture_class9_prepared_val \
  --train-extra-roots ${PREPARED_ROOT}/none_prepared_train \
  --val-extra-roots ${PREPARED_ROOT}/none_prepared_val \
  --output ${PREPARED_ROOT}/gesture_class9_hand21.pkl
```

## 6. 注意事项

- `prepare_hand21_from_excel_segments_parallel.py` 和 `mine_none_clips_from_long_videos.py` 依赖同目录下的基础脚本，请不要单独移动其中某一个文件
- 如果新增长视频池，需要同时考虑：
  - 是否加入 `generate_hand21_video_split.py --extra-video-root`
  - 是否加入 `mine_none_clips_from_long_videos.py --source-groups`
- 若压缩包解压后目录权限异常，建议使用 `tar --no-same-owner --no-same-permissions ...`
- 运行 `prepare_*` 和 `mine_none_*` 脚本前，需要确保运行环境里已经安装 `numpy` 以及对应的提取依赖
