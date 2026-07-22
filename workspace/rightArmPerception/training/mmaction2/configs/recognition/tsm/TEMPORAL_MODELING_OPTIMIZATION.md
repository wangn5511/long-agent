# TSM 时序建模能力优化指南

## 📋 问题诊断

### 当前问题
- ❌ **模型时序特征不强**：模型主要依赖单帧RGB特征进行分类
- ❌ **RGB特征过于明显**：对静态外观特征过度敏感，忽视时序信息
- ❌ **动态手势识别困难**：需要时序理解的手势动作识别准确率低

### 根本原因分析
1. **采样策略问题**：
   - 原配置使用 `SampleFrames` uniform采样
   - frame_interval=1，时序跨度不足
   - 无法充分捕捉动作的时序变化

2. **学习率策略问题**：
   - fc_lr5=True 导致分类头学习率过高（0.01）
   - 分类头快速收敛到RGB特征捷径
   - Backbone 的 Temporal Shift 模块学习不充分

3. **评估策略问题**：
   - 单次采样评估，时序信息利用不充分
   - 无法全面评估模型的时序建模能力

---

## ✅ 优化方案详解

### 1. 采样策略优化（最关键）

#### 训练阶段：Dense 采样
```python
dict(type='CustomSampleFrames', 
    clip_len=1, 
    frame_interval=3,  # ← 关键参数
    num_clips=8,
    sample_mode='dense',  # ← 关键参数
    sample_range=64,  # ← 关键参数
    temporal_jitter=True,  # ← 增强鲁棒性
    test_mode=False)
```

**参数说明**：
- `sample_mode='dense'`：
  - Dense采样在指定范围内密集采样，覆盖更多时序信息
  - 相比uniform采样，能更好捕捉连续动作
  
- `sample_range=64`：
  - 在64帧范围内进行采样
  - 对于30fps视频，约覆盖2秒时长
  - 足够捕捉大部分手势动作的完整时序

- `frame_interval=3`：
  - 每隔3帧采样一次
  - 8帧 × 3间隔 = 24帧跨度
  - 平衡时序跨度和细节信息

- `temporal_jitter=True`：
  - 在采样时加入随机时序扰动
  - 增强模型对时序变化的鲁棒性
  - 防止过拟合特定时间点

**效果对比**：
```
Uniform采样（原配置）：
[0, 1, 2, 3, 4, 5, 6, 7] - 8帧连续采样
问题：时序跨度小，只看到局部动作

Dense采样（新配置）：
在64帧范围内采样8个关键帧，间隔3
例如：[5, 13, 21, 29, 37, 45, 53, 61]
优势：时序跨度大，看到完整动作过程
```

#### 验证/测试阶段：Dense Sliding

```python
dict(
    type='CustomSampleFrames',
    clip_len=1,
    frame_interval=3,
    num_clips=8,
    sample_mode='dense_sliding',  # ← 滑动窗口
    window_stride=3,  # ← 步长
    test_mode=True)
```

**参数说明**：
- `sample_mode='dense_sliding'`：
  - 滑动窗口遍历整个视频
  - 多次采样后聚合预测结果
  - 充分利用时序信息

- `window_stride=3`：
  - 每次窗口移动3帧
  - 保证时序连续性和充分覆盖
  - 平衡精度和计算成本

**Dense Sliding 工作流程**：
```
视频总帧数：100帧
窗口大小：8帧 × 3间隔 = 24帧

窗口1: [0-23]   → 预测1
窗口2: [3-26]   → 预测2
窗口3: [6-29]   → 预测3
...
窗口N: [76-99]  → 预测N

最终预测 = 聚合所有窗口预测（投票或平均）
```

**为什么需要 batch_size=1？**
- Dense Sliding 为每个视频生成多个clips
- 不同视频的clips数量不同（取决于视频长度）
- batch_size=1 保证每个视频独立处理和聚合

---

### 2. 学习率策略优化

#### fc_lr5 关闭
```python
optim_wrapper = dict(
    constructor='TSMOptimWrapperConstructor',
    paramwise_cfg=dict(
        fc_lr5=False,  # ← 关键：关闭5倍学习率
    ),
    optimizer=dict(
        lr=0.002,  # 统一学习率
        ...
    )
)
```

#### 为什么关闭 fc_lr5？

**fc_lr5=True 的问题**：
```
backbone_lr = 0.002    (Temporal Shift模块)
cls_head_lr = 0.01     (分类头，5倍)

问题：
1. 分类头学习速度 >> backbone学习速度
2. 分类头快速收敛到RGB特征（容易学习）
3. Temporal Shift模块学习不充分（难学习）
4. 结果：模型过度依赖RGB，忽视时序
```

**fc_lr5=False 的优势**：
```
backbone_lr = 0.002    (Temporal Shift模块)
cls_head_lr = 0.002    (分类头，统一)

优势：
1. backbone和分类头协同学习
2. 强制模型学习时序特征
3. 避免RGB特征捷径
4. 结果：更强的时序建模能力
```

#### TSM 工作原理回顾
```
输入视频: [F1, F2, F3, F4, F5, F6, F7, F8]
         ↓
    Temporal Shift（在backbone中）
         ↓
    时序特征融合
         ↓
    分类头预测
         ↓
    输出类别
```

- **Temporal Shift** 是TSM的核心，位于backbone
- **需要backbone充分学习**才能捕捉时序信息
- **分类头学习过快**会导致backbone懒惰，不学时序

---

### 3. 其他配置建议

#### 数据增强
保持当前配置：
```python
dict(type='ColorJitter', 
    brightness=0.5, 
    contrast=0.5, 
    saturation=0.5, 
    hue=0.1)
```
- 适度的颜色增强可以防止过拟合RGB特征
- 鼓励模型关注时序和运动特征

#### 训练配置
```python
batch_size = 16  # 训练时保持
lr = 0.002       # 适中的学习率
warmup_epochs = 5
MultiStepLR: [40, 80]
```

#### 数据加载
```python
训练:
  batch_size=16
  num_workers=16
  prefetch_factor=4

验证/测试:
  batch_size=1      # Dense Sliding必须
  num_workers=8
  prefetch_factor=2
```

---

## 📊 预期效果

### 量化指标改善预期
| 指标 | 优化前 | 优化后（预期） | 提升 |
|------|--------|---------------|------|
| **整体准确率** | 基准 | +3~5% | ⬆️ |
| **动态手势准确率** | 低 | +10~15% | ⬆️⬆️ |
| **静态手势准确率** | 高 | 持平或略降 | → |
| **时序混淆错误** | 多 | 显著减少 | ⬇️ |

### 定性改善预期
✅ **时序理解增强**
- 能区分相似外观但不同运动的手势
- 对动作速度和节奏更敏感

✅ **RGB依赖降低**
- 不再过度依赖单帧外观
- 背景干扰鲁棒性提升

✅ **泛化能力提升**
- 对新的动作变体适应更好
- 跨场景性能更稳定

---

## 🔍 训练监控指标

### 关键指标

1. **训练Loss vs 验证Loss**
   ```
   理想状态：
   - 训练loss稳定下降
   - 验证loss也持续下降
   - Gap不要过大（<0.5）
   
   如果gap过大（>1.0）：
   - 考虑增加正则化
   - 降低学习率
   - 增加数据增强
   ```

2. **Top-1 Accuracy**
   ```
   监控趋势：
   - 训练acc快速上升到80%+
   - 验证acc稳步提升
   
   如果验证acc不提升：
   - 检查是否过拟合RGB特征
   - 考虑调整frame_interval
   ```

3. **混淆矩阵**
   ```
   重点关注：
   - 时序相关的手势对（如"挥手"vs"招手"）
   - 之前容易混淆的动作对
   
   优化成功的标志：
   - 时序相关的混淆明显减少
   ```

### TensorBoard 关键曲线

```bash
tensorboard --logdir=work_dirs/...
```

**必看曲线**：
- `loss/train` vs `loss/val`：是否过拟合
- `acc/top1_train` vs `acc/top1_val`：泛化能力
- `lr`：学习率变化
- `grad_norm`：梯度是否稳定

---

## 🚀 进阶优化建议

### 如果效果仍不理想

#### 选项 A：增加时序跨度
```python
frame_interval=5,  # 从3增加到5
sample_range=80,   # 从64增加到80
```
- 适用于：动作较慢的手势
- 代价：可能丢失细节信息

#### 选项 B：增加采样帧数
```python
num_frames = 16,   # 从8增加到16
```
- 优势：更多时序信息
- 代价：
  - 显存占用增加（约2倍）
  - 训练速度变慢
  - 需要降低batch_size

#### 选项 C：调整学习率
```python
# 方案1：更保守
lr = 0.001  # 从0.002降低

# 方案2：分阶段学习率
custom_keys={
    'backbone': dict(lr_mult=1.0),
    'cls_head': dict(lr_mult=1.5),  # 轻微提升
}
```

#### 选项 D：增强时序正则化
```python
# 在backbone中增加dropout
model = dict(
    backbone=dict(
        dropout_ratio=0.5,  # 增加dropout
    )
)
```

---

## 📈 消融实验建议

为了验证各个优化的有效性，建议进行以下消融实验：

### 实验1：采样策略对比
```
A: Uniform + interval=1 (基准)
B: Dense + interval=1
C: Dense + interval=3 (当前配置)
D: Dense + interval=5
```

### 实验2：fc_lr5 影响
```
A: fc_lr5=True + lr=0.002
B: fc_lr5=False + lr=0.002 (当前配置)
C: fc_lr5=False + lr=0.001
```

### 实验3：评估策略对比
```
A: Uniform采样评估
B: Dense Sliding评估 (当前配置)
C: Dense Sliding + TenCrop
```

---

## 🔧 实施检查清单

### 配置文件检查
- [ ] `train_pipeline` 使用 `CustomSampleFrames`
- [ ] `sample_mode='dense'`, `sample_range=64`, `frame_interval=3`
- [ ] `temporal_jitter=True`
- [ ] `val/test_pipeline` 使用 `dense_sliding`
- [ ] `window_stride=3`
- [ ] `fc_lr5=False`
- [ ] `lr=0.002`
- [ ] `val/test batch_size=1`
- [ ] `ValLossLoop` 记录验证loss

### 训练前确认
- [ ] CustomSampleFrames 已注册到transforms
- [ ] 数据集路径正确
- [ ] GPU显存充足（约6-8GB）
- [ ] TensorBoard配置正确

### 训练中监控
- [ ] 验证loss正常下降
- [ ] 验证acc稳步提升
- [ ] 没有严重过拟合（train-val gap < 0.5）
- [ ] data_time占比正常（< 50%）

---

## 📚 参考资料

### 论文
1. **TSM: Temporal Shift Module for Efficient Video Understanding**
   - Link: https://arxiv.org/abs/1811.08383
   - 关键点：Temporal Shift原理，fc_lr5策略

2. **Temporal Segment Networks (TSN)**
   - Link: https://arxiv.org/abs/1608.00859
   - 关键点：Sparse采样 vs Dense采样

3. **SlowFast Networks for Video Recognition**
   - Link: https://arxiv.org/abs/1812.03982
   - 关键点：多尺度时序建模

### MMAction2 文档
- **微调指南**: https://mmaction2.readthedocs.io/en/latest/user_guides/finetune.html
- **数据流水线**: https://mmaction2.readthedocs.io/en/latest/user_guides/prepare_dataset.html
- **配置系统**: https://mmaction2.readthedocs.io/en/latest/user_guides/config.html

### 相关讨论
- **TSM官方repo**: https://github.com/mit-han-lab/temporal-shift-module
- **MMAction2 Issues**: https://github.com/open-mmlab/mmaction2/issues

---

## ❓ 常见问题

### Q1: 为什么验证很慢？
**A:** Dense Sliding模式需要对每个视频生成多个clips，这是正常的。可以通过调整`window_stride`来平衡精度和速度：
- `stride=1`: 最精确，最慢
- `stride=3`: 平衡（当前配置）
- `stride=8`: 快速，精度略降

### Q2: 显存不够怎么办？
**A:** 按优先级尝试：
1. 降低`train_batch_size`到8或更小
2. 减少`num_workers`和`prefetch_factor`
3. 降低`sample_range`到32
4. 使用gradient accumulation

### Q3: 训练时间太长？
**A:** 
- Dense采样增加了约10-20%的训练时间（可接受）
- 验证时间增加较多（因为Dense Sliding），可以降低`val_interval`
- H100应该能维持较快速度

### Q4: 精度没有提升？
**A:** 检查：
1. 数据集是否足够大（至少500+ samples）
2. 手势是否确实需要时序信息（静态手势改善有限）
3. 是否出现过拟合（查看train-val gap）
4. 尝试更小的学习率（0.001）

### Q5: 什么时候应该用fc_lr5=True？
**A:** 
- 从头训练且数据集很大（>10K samples）
- 模型在验证集上欠拟合（acc < 60%）
- 经过多次实验发现时序能力已经足够

---

## 🎯 总结

### 核心要点
1. **Dense采样是关键**：大幅提升时序建模能力
2. **关闭fc_lr5很重要**：避免RGB特征捷径
3. **Dense Sliding评估更准确**：充分利用时序信息
4. **统一学习率更稳定**：backbone和分类头协同学习

### 预期结果
- ✅ 时序特征显著增强
- ✅ 动态手势识别提升10-15%
- ✅ 模型更关注运动而非外观
- ✅ 训练更稳定，不易过拟合

### 下一步
1. 用新配置重新训练
2. 监控TensorBoard曲线
3. 分析混淆矩阵，关注时序相关手势对
4. 根据结果进行微调

---

**配置文件**: `tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb-cytoderm-gesture_pipeline4_v1v2.py`

**最后更新**: 2025-10-20

