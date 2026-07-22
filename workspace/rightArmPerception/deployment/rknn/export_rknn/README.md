### 执行导出

```bash
conda activate rknn
python export_rknn.py --config example.yaml
```

查看解析后的配置：

```bash
conda activate rknn
python export_rknn.py --config example.yaml --print-config
```

### 命令行参数

| 参数 | 说明 |
|------|------|
| `--config <path>` | YAML 配置文件路径，必填 |
| `--set KEY=VALUE` | 覆盖配置值，支持点号分隔的嵌套键，可多次使用 |
| `--dry-run` | 仅验证配置和输入文件，不执行导出 |
| `--print-config` | 打印解析后的完整配置 |

`--set` 示例：

```bash
python export_rknn.py --config my.yaml --set model.model_path=./new.onnx
python export_rknn.py --config my.yaml --set build.do_quantization=false
```

---

## 配置文件参数说明

YAML 里的相对路径均相对于配置文件所在目录解析。
校准数据集 `.txt` 清单中的相对路径会相对于清单文件所在目录解析。

### `meta`

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | string | 任务名称，仅用于日志 |
| `verbose` | bool | 是否开启 RKNN 详细日志 |
| `verbose_file` | string | 日志输出文件路径 |

### `model`

| 字段 | 类型 | 说明 |
|------|------|------|
| `framework` | string | 模型框架：`onnx` / `pytorch` |
| `model_path` | string | 模型文件路径 |
| `inputs` | list[string] | 输入 tensor 名称；多输入模型按模型输入顺序填写 |
| `outputs` | list[string] | 输出 tensor 名称，通常仅在需要裁剪或手动指定子图时填写 |
| `input_size_list` | list[list[int]] | 输入形状；多输入模型按模型输入顺序填写，`pytorch` 必填，`onnx` 一般按需填写 |
| `input_initial_val_files` | list[string] | 初始值 `.npy` 文件，动态输入模型可能需要；支持单个路径或列表，列表顺序需和输入顺序一致 |

### `rknn_config`

映射到 `rknn.config()` 参数。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target_platform` | string | — | 目标平台：`rk3588` / `rk3568` / `rv1106` 等 |
| `mean_values` | list | — | 输入均值，如 `[[123.675, 116.28, 103.53]]` |
| `std_values` | list | — | 输入标准差，如 `[[58.395, 57.12, 57.375]]` |
| `quantized_dtype` | string | `w8a8` | 量化类型 |
| `quantized_algorithm` | string | `normal` | 量化算法：`normal` / `mmse` / `kl_divergence` |
| `quantized_method` | string | `channel` | 量化粒度：`channel` / `layer` / `tensor` |
| `quantized_hybrid_level` | int | `0` | 混合量化级别 |
| `quant_img_RGB2BGR` | bool | `false` | 量化时是否做 RGB→BGR 转换 |
| `float_dtype` | string | `float16` | 非量化层浮点精度 |
| `optimization_level` | int | `3` | 编译优化等级（0-3） |
| `custom_string` | string | — | 嵌入 RKNN 模型的自定义字符串 |
| `dynamic_input` | — | — | 动态输入配置 |
| `auto_hybrid_cos_thresh` | float | `0.98` | 自动混合量化余弦相似度阈值 |

### `build`

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `do_quantization` | bool | 自动 | 是否量化，省略时有数据集则自动启用 |
| `rknn_batch_size` | int | — | 推理批大小 |
| `auto_hybrid` | bool | `false` | 自动混合量化 |
| `dataset` | string | — | 校准数据集 `.txt` 文件路径；单输入每行一个文件，多输入每行多个文件 |

说明：

- 开启量化时，脚本会检查校准清单是否存在、是否为空，以及清单中的样本文件是否存在。
- 如果 `model.inputs` 或 `model.input_size_list` 声明了输入数量，脚本会校验校准清单每行的文件数量是否一致。

### `export`

| 字段 | 类型 | 说明 |
|------|------|------|
| `output_path` | string | 输出 `.rknn` 文件路径 |
| `accuracy_analysis.inputs` | list[string] | 精度分析用的样本路径 |
| `accuracy_analysis.output_dir` | string | 精度分析结果输出目录 |

---

## 示例配置说明

仓库里带了一份最小可跑的示例：

- `example.yaml`
- `example_assets/test.onnx`
- `example_assets/test.txt`
- `example_assets/sample_0000.npy` 到 `sample_0002.npy`

这份示例主要用于验证脚本和 `rknn-toolkit2` 环境，不是实际模型。

多输入示例：

- `multi_input_example.yaml`
- `example_assets/multi_input_test.txt`
- `README_multi_input.md`

`multi_input_example.yaml` 默认指向仓库内的单输入 `test.onnx`，仅用于 `--dry-run` 验证多输入 YAML 和清单解析格式。正式导出前需要替换为真实多输入 ONNX，并改成真实 `inputs` 和 `input_size_list`。

---

## 校准数据集准备

INT8 量化需要校准数据集来统计激活分布。需要准备一个 `.txt` 文件，每行写一个样本文件的路径。

### 图像模型

图像模型可以直接使用 `.jpg` / `.png` 图片，RKNN toolkit 会根据 `rknn_config` 中配置的 `mean_values` / `std_values` 自动完成预处理。

`.txt` 文件示例：

```text
/path/to/calibration_images/img_001.jpg
/path/to/calibration_images/img_002.jpg
/path/to/calibration_images/img_003.jpg
```

### 非图像模型

对于非图像输入的模型，比如骨骼点、时序特征或传感器数据，更推荐保存为 `.npy` 文件，每个文件包含一个样本，shape 和 dtype 需与模型输入一致。

`.txt` 文件示例：

```text
/path/to/calibration_data/sample_0000.npy
/path/to/calibration_data/sample_0001.npy
/path/to/calibration_data/sample_0002.npy
```

通常准备 50-100 个样本即可，样本应尽量覆盖真实推理场景的数据分布。

### 多输入模型

多输入模型的校准清单仍然是一个 `.txt` 文件，但每行代表一个完整样本，行内按模型输入顺序写多个输入文件，使用空格分隔。
因此校准文件路径本身不要包含空格。

两输入模型 `.txt` 示例：

```text
/path/to/calib/template_0000.npy /path/to/calib/search_0000.npy
/path/to/calib/template_0001.npy /path/to/calib/search_0001.npy
```

对应 YAML 里的输入声明也需要按同一顺序填写：

```yaml
model:
  inputs:
    - template
    - search
  input_size_list:
    - [1, 3, 128, 128]
    - [1, 3, 256, 256]
```

如果配置中声明了两个输入，但校准清单某一行只有一个或三个文件，`--dry-run` 会直接报错。

---

## 备注

- 当前本地已验证环境为 `rknn-toolkit2 2.3.2`。
- 当前本地环境对 ONNX 的支持上限是 opset 19，过高会在 `load_onnx` 阶段报错。
