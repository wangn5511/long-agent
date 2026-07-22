# STGCN RGB+Gated 融合模型 RKNN 转换 / 部署踩坑记录

记录 4 输入 RGB+骨架后融合手势模型从 ONNX 转 RKNN、再部署到 RK3588 (Radxa Rock 5A)
真实 NPU 上的全过程，以及中间踩到的坑。给后续做类似多输入 / 非图像 RKNN 模型的人参考。

> 适用对象：STGCN 3D RGB+gated 10 类手势模型
> （`gesture_recognition/stgcn_3d_rgb_gated_kproi_abspos_4ch`）。
> 大部分结论对其他多输入 / 含 4D 非图像张量的 RKNN 模型同样适用。

---

## 1. 模型概况

4 输入 2 输出：

| 张量 | shape | dtype | 含义 |
|------|-------|-------|------|
| `keypoint` | `(1, 16, 21, 4)` | float32 | 手部骨架：16 帧 × 21 关节 × (x,y,z,score)，wrist 相对 + 尺度归一 |
| `imgs` | `(1, 16, 3, 96, 96)` | float32 | 手部 ROI RGB 序列，原始 `[0,255]`，ImageNet 归一化在 ONNX 图内做 |
| `handedness` | `(1,)` | int64 | 0/1/2 = Left/Right/Unknown |
| `global_traj` | `(1, 16, 5)` | float32 | 手中心轨迹：delta_xyz + abs_xy |
| `cls_score` | `(1, 10)` | float32 | 10 类原始 logits |
| `gate_logit` | `(1, 1)` | float32 | NONE 门控原始标量，部署侧做 `sigmoid` + 阈值 |

转换链路与相关文件：

```
训练 ckpt
  └─ tools/export_onnx_gcn_3d_rgb.py        → model.onnx (FP32)
       └─ deployment/rknn/export_rknn/
            ├─ build_stgcn_3d_rgb_gated_calibration.py  → 校准 npy + 清单
            ├─ stgcn_3d_rgb_gated_rk3588.yaml            → RKNN 转换配置
            ├─ export_rknn.py                            → model.rknn
            └─ verify_stgcn_3d_rgb_gated_onnx_vs_rknn.py → ONNX vs RKNN 一致性
```

---

## 2. 踩坑记录

### 坑 1 ⭐⭐⭐ — 4D `keypoint` 张量被 RKNN 当成图像，NHWC 物理布局转置

**这是最隐蔽、最致命的坑。整套链路在真实 NPU 上输出全错，根因就是它。**

`keypoint` 是 rank-4 张量 `(1, T=16, V=21, C=4)`，是自然的 ONNX/mmaction 顺序
`(batch, time, joints, feature)`。RKNN toolkit 遇到 rank-4 输入会**按图像处理**，
编译出来的 NPU 图内部按 **NHWC** 排布。我们这份输入按 NCHW 解释
`(N=1, C=16, H=21, W=4)`，对应的 NHWC 物理布局是 `(1, 21, 4, 16)` = `(N, H, W, C)`。

**关键：用什么 API 喂数据，转置行为完全不同：**

| 路径 | 行为 |
|------|------|
| Python `rknn.inference(inputs=[...], data_format=['nchw',...])` | toolkit 把你的 buffer 当 NCHW，**自动转置**到 NHWC。结果正确。 |
| Python `rknn.inference(...)` 不传 `data_format` | toolkit 可能误判布局，**静默转错** V/C 轴。结果错误。 |
| C++ 裸 C API `rknn_inputs_set`，`rknn_input.fmt = RKNN_TENSOR_NHWC` | runtime **不转置**，直接当你 buffer 已经是 NHWC 物理布局。若你的 buffer 实际是 `(1,16,21,4)` NCHW 顺序 → NPU 收到错位张量 → **静默输出错误，不报错**。 |
| C++ 裸 C API，`rknn_input.fmt = RKNN_TENSOR_NCHW` | runtime 尝试转置，但其 `normalize` 算子对 rank-4 输入只支持 NHWC src → **直接报错** `Meet unsupported src layout for normalize: NCHW`。 |

**结论：**

- Python 验证脚本里 **必须** 给每个输入显式传 `data_format`（见
  `verify_stgcn_3d_rgb_gated_onnx_vs_rknn.py`，对 4 个输入都传了 `'nchw'`）。
- C++ 部署侧（cybopal perception）**必须自己物理转置** `(1,T,V,C) → (1,V,C,T)`
  之后再 `rknn_inputs_set(fmt=NHWC)`。裸 C API 没有 Python wrapper 的自动转置。
- 错误布局**不会报错**，只会静默输出错乱的预测。最难查。

**症状：** 同一份输入、同一个 `.rknn` 文件，Python `rknnlite`（传 `data_format='nchw'`）
输出正确，C++ 部署输出完全不同的类别。

**定位方法：** 把 C++ 实际喂的 4 个输入 dump 成二进制，Python 端用三种
`data_format` 组合 + 预转置 buffer 复现，能直接观察到是不是转置问题。

---

### 坑 2 ⭐⭐ — INT8 `normal` 校准掉点 7.5pp，要换 `mmse`

第一次 INT8 量化用默认 `quantized_algorithm: normal`（min/max 校准），
top1 相对 ONNX 掉了约 7.5pp。

`normal` 的 min/max 校准对离群权重很敏感。这个融合模型里
`Conv_901` 有离群值 11.1、`Conv_919` 有 -36.9，被 min/max 拉爆了量化范围。

换成 `quantized_algorithm: mmse`（MSE 最优校准）后，离线 ONNX↔RKNN 一致性恢复。
代价是 build 更慢。`stgcn_3d_rgb_gated_rk3588.yaml` 已固定为 `mmse` + `quantized_method: channel`。

> 融合模型各分支数值范围差异很大（RGB 分支 / 骨架分支 / 轨迹分支 /
> handedness embedding / gate 头），`channel` 粒度 + `mmse` 是必要的。

---

### 坑 3 ⭐⭐ — 验证脚本跑的是 CPU 模拟器，盖不住部署侧的布局 bug

`verify_stgcn_3d_rgb_gated_onnx_vs_rknn.py` 用 `rknn.api.RKNN` +
`init_runtime()`（**没传 `target='rk3588'`**），跑的是 **CPU 模拟器**，不是真实 NPU。

模拟器对**模型本身**的 ONNX↔RKNN 一致性验证是有效的。但它有两个盲区：

1. **盖不住部署侧 C++ 代码的输入布局 bug**（坑 1）—— 那个 bug 在 C++ 喂数据的代码里，
   不在模型里，模拟器和验证脚本都用 Python wrapper 正确转置了，自然测不出来。
2. 模拟器与真实 NPU 在极端量化下偶尔有细微差异。

**结论：** 模型验证不能只信模拟器 / Python 脚本。必须在板子上用**真实推理代码路径**
（C++ 部署 binary）做端到端验证。Python `rknnlite` 真机测试 + C++ 部署测试都要做。

---

### 坑 4 ⭐ — FP16（`do_quantization=false`）是稳妥兜底

二分定位时建议先转一版 FP16 RKNN：

```bash
python export_rknn.py --config stgcn_3d_rgb_gated_rk3588.yaml \
    --set build.do_quantization=false \
    --set export.output_path=/path/to/model_fp16.rknn
```

FP16 RKNN（`float_dtype: float16`，仍跑在 NPU 上）与 ONNX FP32 数值上几乎等价
（实测各类 logit 差 <0.03，gate_logit 差 <0.04）。用途：

- **二分诊断**：FP16 对 ≈ ONNX 而 INT8 对不上 → 问题在量化；FP16 也对不上 → 问题在
  图转换 / runtime / 部署代码。
- **兜底部署**：FP16 比 INT8 慢约 1.5~2×、文件大约 1.5×（8.9MB vs 6.0MB），
  但比 CPU 跑 ONNX 快 5~10×，且精度无损。INT8 调不动时可直接 ship FP16。

---

### 坑 5 — 杂项（部署侧，cybopal 仓库）

把 FP32 ONNX 直接部署到板子做对照时遇到的，记录备查：

- **manifest `provider_family` 必须精确匹配**：host CPU ONNX backend 的 descriptor 是
  `onnxruntime-cpu`，manifest 里写 `onnxruntime` 会被判不兼容
  （`Manifest is not compatible with the selected provider`）。
- **ONNX Runtime `.so` 加载**：板子上 onnxruntime 装在
  `/usr/local/lib/python3.x/site-packages/onnxruntime/capi/`，cytoderm backend 默认
  搜不到，需设 `CYTODERM_ONNXRUNTIME_LIB` 环境变量指向 `libonnxruntime.so.<ver>`。

---

## 3. 诊断方法论（二分定位）

RKNN「ONNX 对、RKNN 错」时，不要先调量化。按下面顺序二分：

```
1. 同一份 .npy 输入，分别跑 ONNX / RKNN-FP16 / RKNN-INT8，对比 top3 + gate。
2. FP16 ≈ ONNX 但 INT8 偏 → 量化问题（坑 2，调 mmse / channel / 校准集）。
3. FP16 也 ≠ ONNX → 图转换 / 输入布局 / data_format 问题（坑 1、坑 3）。
4. Python 模拟器对、真机 C++ 错 → 部署侧输入布局 / 预处理 bug（坑 1）。
5. dump C++ 实际喂的输入二进制，Python 端复现，逐输入对比 shape/dtype/min/max/mean。
```

每一步都用**完全相同**的 `.npy` 输入，才能逐段锁定问题。

---

## 4. 最终结论

| 产物 | 文件 | 精度 vs ONNX | 跑在 | 备注 |
|------|------|--------------|------|------|
| ONNX FP32 | `model.onnx` | 基准 | CPU | 慢，仅做对照 / 兜底 |
| RKNN FP16 | `model.rknn` (`do_quantization=false`) | logit 差 <0.03 | NPU | 稳妥，精度无损 |
| RKNN INT8 mmse | `model.rknn` (`do_quantization=true`) | top1 一致，logit 偏弱 | NPU | 最快、最小，需坑 1 修好后才可用 |

**核心教训：** RKNN 部署「输出全错」未必是量化问题。这次的根因是 4D 非图像张量
（`keypoint`）的 NHWC 布局转置 —— C++ 裸 C API 不会像 Python wrapper 那样自动转置，
必须在部署侧手动转。错误布局静默不报错，是排查里最费时的部分。
