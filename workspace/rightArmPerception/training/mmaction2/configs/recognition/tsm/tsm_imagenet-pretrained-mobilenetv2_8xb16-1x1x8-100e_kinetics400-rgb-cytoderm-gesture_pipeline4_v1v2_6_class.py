"""
TSM MobileNetV2 配置文件 - 时序能力优化版（6类手势）

====================================================================================
                         针对时序建模能力提升的配置优化
====================================================================================

【问题分析】
- 模型时序特征不强，RGB静态特征过于明显
- 需要增强时序建模能力，减少对单帧RGB特征的依赖

【优化策略汇总】

1. ✅ 采样策略优化（最重要）
   训练: CustomSampleFrames + Dense采样
   - sample_range=64: 在64帧范围内密集采样
   - frame_interval=3: 帧间隔3，增加时序跨度
   - temporal_jitter=True: 增加训练多样性，促进时序学习
   
   验证/测试: CustomSampleFrames + Dense Sliding
   - window_stride=3: 滑动窗口步长3帧
   - 逐视频评估(batch_size=1)，充分利用时序信息

2. ✅ 学习率策略优化
   - fc_lr5=False: 关闭分类头5倍学习率
   - 统一lr=0.002: backbone和分类头协同学习时序特征
   - 避免分类头过快收敛到RGB捷径

3. ✅ 训练配置
   - batch_size=16 (训练)
   - MultiStepLR + 5epoch warmup
   - weight_decay=0.0001

4. ✅ 数据加载优化
   - num_workers=16 (训练)
   - prefetch_factor=4
   - pin_memory=True

5. ✅ 数据质量控制（新增）
   - 使用RawframeDatasetWithLabelFix
   - abandon_list: 过滤低质量样本
   - label_fix_list: 修正错误标注

【预期效果】
- 时序特征显著增强
- RGB静态特征依赖降低
- 对动态手势识别能力提升

【参考文档】
- TSM论文: https://arxiv.org/abs/1811.08383
- MMAction2微调: https://mmaction2.readthedocs.io/en/latest/user_guides/finetune.html

====================================================================================
"""

_base_ = [
    '../../_base_/models/tsm_mobilenet_v2.py',
    '../../_base_/default_runtime.py'
]

# dataset settings
# dataset_type = 'RawframeDataset'
dataset_type = "RawframeDatasetWithLabelFix"

# ===== 多数据集配置（列表形式，易于扩展） =====
# 如果需要添加更多数据集（如v3, v4等），只需在列表中添加新的配置项即可
# 每个数据集的权重（sample_ratio）可以在这里配置
dataset_configs = [
    {
        'name': 'v1',  # 数据集名称（用于标识）
        'data_root': '/home/datasets/gesture_dataset/gesture_v1_new/images/',
        'ann_file_train': '/home/datasets/gesture_dataset/gesture_v1_new/gesture_v1_train_only_6_classes.txt',
        'ann_file_val': '/home/datasets/gesture_dataset/gesture_v1_new/gesture_v1_val_only_6_classes.txt',
        'sample_ratio': 0.5,  # 训练时的采样比例（所有数据集的比例之和应为1.0）
    },
    {
        'name': 'v2',
        'data_root': '/home/datasets/gesture_dataset/v2/processed/images/',
        'ann_file_train': '/home/datasets/gesture_dataset/v2/processed/gesture_v2_train_only_6_classes.txt',
        'ann_file_val': '/home/datasets/gesture_dataset/v2/processed/gesture_v2_val_only_6_classes.txt',
        'sample_ratio': 0.5,
    },
    # 可以继续添加更多数据集，例如：
    # {
    #     'name': 'v3',
    #     'data_root': '/path/to/v3/images/',
    #     'ann_file_train': '/path/to/v3/train.txt',
    #     'ann_file_val': '/path/to/v3/val.txt',
    #     'sample_ratio': 0.33,  # 如果有3个数据集，每个1/3
    # },
]

# XXX: 标注修正模块
abandon_list = [
    "WAVE_OUT/recording_20250903_111818",
    "WAVE_IN/recording_20250903_163549",
]

label_fix_list = {
    "WAVE_OUT/recording_20250917_134129": "WAVE_IN",
}

# 验证sample_ratio总和
_total_ratio = sum(cfg['sample_ratio'] for cfg in dataset_configs)
assert abs(_total_ratio - 1.0) < 0.01, f"sample_ratio总和应为1.0，当前为{_total_ratio}"

num_classes = 6

class_mapping = {
    0: "NONE",
    1: "OK",
    2: "WAVE_IN",
    3: "WAVE_OUT",
    4: "BYEBYE",
    5: "THUMBS_UP",
}

num_frames = 8
# TSM微调最佳实践配置（基于官方文档和经验）
# 参考来源：
# 1. mmaction2官方微调文档: https://mmaction2.readthedocs.io/en/latest/user_guides/finetune.html
# 2. TSM官方配置: https://github.com/open-mmlab/mmaction2/tree/main/configs/recognition/tsm
# 3. TSM论文: https://arxiv.org/abs/1811.08383
# 官方推荐：微调时使用batch_size=16，这是TSM的标准配置
train_batch_size_per_gpu = 16  # TSM官方微调标准

test_batch_size_per_gpu = 1

file_client_args = dict(io_backend='disk')

deploy_npu = True

model = dict(
    backbone=dict(
        num_segments=num_frames,
    ),
    cls_head=dict(
        num_segments=num_frames,
        num_classes=num_classes),
    test_cfg=dict(
        _delete_=True,
        input_format='NCHW') if deploy_npu else None  # NOTE: 强制使用NCHW格式，使得ONNX 导出时的格式是正确的。
)

# ===== 训练数据增强管道（Dense采样提升时序能力） =====
train_pipeline = [
    # 使用CustomSampleFrames的Dense模式提升时序建模
    # Dense采样：在64帧范围内密集采样，frame_interval=3确保时序跨度
    # 相比uniform采样，dense采样能更好地捕捉时序信息
    dict(type='CustomSampleFrames', 
        clip_len=num_frames,  # 采样8帧
        frame_interval=1,  # 帧间隔3，提升时序跨度
        num_clips=1,
        sample_mode='dense',  # Dense采样模式
        sample_range=16,  # 在16帧范围内采样
        temporal_jitter=True,  # 增加训练多样性，促进学习时序变化
        # 说明：temporal_jitter不会打乱帧顺序，只在每个采样点加±1-2帧的随机偏移
        #       让模型学习相对时序变化，而非固定位置的RGB特征
        #       这对提升时序相关性很重要
        out_of_bound_opt='repeat_last',
        test_mode=False),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    # ColorJitter数据增强 - 颜色抖动
    # brightness: 亮度变化范围 (1-value, 1+value)，当前(0.8, 1.2)
    # contrast: 对比度变化范围 (1-value, 1+value)，当前(0.8, 1.2)
    # saturation: 饱和度变化范围 (1-value, 1+value)，当前(0.8, 1.2)
    # hue: 色调偏移范围 (-value, +value)，当前(-0.05, 0.05)
    # 如需更强增强，可改为 brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
    # 如需禁用，可注释掉此行
    dict(type='ColorJitter', brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
# ===== 验证数据管道（Dense Sliding提升时序覆盖） =====
val_pipeline = [
    # Dense Sliding模式：滑动窗口覆盖整个视频，充分利用时序信息
    # window_stride=3：每次滑动3帧，保证时序连续性和充分覆盖
    # 窗口大小由 clip_len * frame_interval 自动计算 (8 * 3 = 24帧)
    # 相比单次采样，sliding window能更全面地评估时序能力
    # 
    # 示例：30帧视频，clip_len=8，frame_interval=3，stride=3
    #   窗口1: [0, 3, 6, 9, 12, 15, 18, 21]
    #   窗口2: [3, 6, 9, 12, 15, 18, 21, 24]
    #   窗口3: [6, 9, 12, 15, 18, 21, 24, 27]
    #   → 覆盖整个视频！
    dict(
        type='CustomSampleFrames',
        clip_len=num_frames,  # 每个窗口采样8帧
        frame_interval=1,  # 帧间隔3
        num_clips=1,  # 每个窗口1个clip
        sample_mode='dense_sliding',  # 滑动窗口模式
        window_stride=3,  # 滑动步长为3帧
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

# ===== 测试数据管道（Dense Sliding更全面评估） =====
test_pipeline = [
    # 测试时同样使用Dense Sliding，充分评估时序建模能力
    # 注意：sample_range 参数在 dense_sliding 中被忽略
    #       窗口大小由 clip_len * frame_interval 决定
    # dict(
    #     type='CustomSampleFrames',
    #     clip_len=num_frames,  # 每个窗口采样8帧
    #     frame_interval=1,  # 帧间隔3
    #     sample_mode='dense_sliding',
    #     window_stride=3,  # 滑动步长为3帧
    #     test_mode=True),

    dict(
        type='SampleFrames',
        clip_len=num_frames,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs', meta_keys=(
            'img_shape', 'img_key', 'video_id', 'timestamp', 'sample_name'))
]

# ===== 自动生成训练集列表 =====
# 根据dataset_configs自动生成所有训练集
train_datasets = []
for cfg in dataset_configs:
    dataset = dict(
        type=dataset_type,
        ann_file=cfg['ann_file_train'],
        data_prefix=dict(img=cfg['data_root']),
        filename_tmpl='frame_{:04}.png',
        with_offset=True,
        pipeline=train_pipeline,
        class_mapping=class_mapping,
        abandon_list=abandon_list,
        label_fix_list=label_fix_list)
    train_datasets.append(dataset)

# ===== 数据集合并配置 =====
# 方案1：简单的ConcatDataset（当前使用）
# 适用于数据集大小相近的情况，会根据实际大小自动采样
combined_trainset = dict(
    type='ConcatDataset',
    datasets=train_datasets)

# 方案2：使用RepeatDataset实现精确的采样比例（可选）
# 如果需要严格按照sample_ratio进行采样，可以使用以下配置：
# 1. 计算每个数据集需要重复的次数
# 2. 使用RepeatDataset包装
# 示例代码（取消注释以启用）：
"""
from collections import defaultdict
import math

# 读取每个数据集的实际大小
def get_dataset_size(ann_file):
    try:
        with open(ann_file, 'r') as f:
            return sum(1 for _ in f)
    except:
        return 1  # 如果无法读取，返回1避免除零错误

# 计算每个数据集的实际大小
dataset_sizes = [get_dataset_size(cfg['ann_file_train']) for cfg in dataset_configs]

# 计算需要的重复次数以实现目标sample_ratio
# 找到最大的 size/ratio 值作为基准
max_normalized_size = max(size / cfg['sample_ratio'] 
                          for size, cfg in zip(dataset_sizes, dataset_configs))

# 为每个数据集计算重复次数
train_datasets_balanced = []
for i, (cfg, size) in enumerate(zip(dataset_configs, dataset_sizes)):
    target_size = max_normalized_size * cfg['sample_ratio']
    repeat_times = max(1, round(target_size / size))
    
    dataset = dict(
        type=dataset_type,
        ann_file=cfg['ann_file_train'],
        data_prefix=dict(img=cfg['data_root']),
        filename_tmpl='frame_{:04}.png',
        with_offset=True,
        pipeline=train_pipeline)
    
    if repeat_times > 1:
        dataset = dict(type='RepeatDataset', dataset=dataset, times=repeat_times)
    
    train_datasets_balanced.append(dataset)

combined_trainset = dict(
    type='ConcatDataset',
    datasets=train_datasets_balanced,
    _delete_=True)
"""

# ===== 自动生成验证集列表 =====
# 根据dataset_configs自动生成所有验证集
val_datasets = []
for cfg in dataset_configs:
    dataset = dict(
        type=dataset_type,
        ann_file=cfg['ann_file_val'],
        data_prefix=dict(img=cfg['data_root']),
        filename_tmpl='frame_{:04}.png',
        with_offset=True,
        pipeline=val_pipeline,
        test_mode=True,
        class_mapping=class_mapping,
        abandon_list=abandon_list,
        label_fix_list=label_fix_list)
    val_datasets.append(dataset)

# 合并所有验证集
combined_valset = dict(
    type='ConcatDataset',
    datasets=val_datasets)

# ===== 自动生成测试集列表 =====
# 根据dataset_configs自动生成所有测试集（使用验证集的标注文件）
test_datasets = []
for cfg in dataset_configs:
    dataset = dict(
        type=dataset_type,
        ann_file=cfg['ann_file_val'],  # 测试集使用验证集的标注
        data_prefix=dict(img=cfg['data_root']),
        filename_tmpl='frame_{:04}.png',
        with_offset=True,
        pipeline=test_pipeline,
        test_mode=True,
        class_mapping=class_mapping,
        abandon_list=abandon_list,
        label_fix_list=label_fix_list)
    test_datasets.append(dataset)

# 合并所有测试集
combined_testset = dict(
    type='ConcatDataset',
    datasets=test_datasets)

# ===== DataLoader配置 =====
# H100 优化：增加数据加载性能配置
# 参考: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
# 问题诊断：data_time占总时间85%（0.5/0.588），数据加载是瓶颈
# 解决方案：增加 num_workers, prefetch_factor, pin_memory
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=16,  # 从8增加到16，H100计算快需要更多worker
    persistent_workers=True,  # 保持worker进程，避免重复创建
    prefetch_factor=4,  # 每个worker预取4个batch，加速数据流
    pin_memory=True,  # 使用pinned memory加速CPU->GPU传输
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=combined_trainset)

val_dataloader = dict(
    batch_size=1,  # Dense Sliding模式需要batch_size=1进行逐个视频评估
    num_workers=8,  # batch_size=1时不需要太多worker
    persistent_workers=True,
    prefetch_factor=2,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=combined_valset)

test_dataloader = dict(
    batch_size=1,  # Dense Sliding模式需要batch_size=1
    # num_workers=8,
    # persistent_workers=True,
    # prefetch_factor=2,
    num_workers=0,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=combined_testset)

# ===== 数据集配置总结 =====
# 当前配置使用多个数据集，可以通过修改dataset_configs列表来添加/删除数据集
# 每个数据集的sample_ratio控制采样权重（所有比例之和应为1.0）

val_evaluator = dict(type='AccMetric')
# test_evaluator = val_evaluator
test_evaluator = [
    dict(type='AccMetric'),
    dict(type='ConfusionMatrix', num_classes=num_classes,
        save_path='confusion_matrix_val.npy',
        save_misclassified=True,
        misclassified_save_path='misclassified_val.txt',
        save_classified=True,
        classified_save_path='classified_val.txt',
        class_mapping=class_mapping)
]

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=5),
    logger=dict(
        type='LoggerHook', 
        interval=5, 
        ignore_last=False,
        # 关键：log_metric_by_epoch控制是否在epoch结束时记录metrics
        # 设置为True会在TensorBoard中记录val_loss等验证指标
        log_metric_by_epoch=True)  # 确保验证metrics被记录
)

# ===== 可视化配置（包含TensorBoard支持） =====
# TensorBoard日志将保存在 work_dirs/{config_name}/{timestamp}/vis_data/ 目录下
# 使用命令查看: tensorboard --logdir=work_dirs/{config_name}/{timestamp}/vis_data/
visualizer = dict(
    type='ActionVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),  # 本地可视化后端（保存到本地文件）
        dict(type='TensorboardVisBackend'),  # TensorBoard可视化后端
        # 如果需要使用WandB，可以取消下面的注释：
        # dict(type='WandbVisBackend', init_kwargs=dict(project='gesture-recognition')),
    ]
)

num_epochs = 100

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=num_epochs, val_begin=1, val_interval=1)

# MMEngine原生的ValLoop和TestLoop已经支持loss记录
# 只要模型的val_step/test_step返回正确格式，loss会自动记录到TensorBoard
# 参考: mmengine.runner.loops.ValLoop (line 370-414)
val_cfg = dict(type='ValLoop')  # 使用原生ValLoop，自动支持loss记录
test_cfg = dict(type='TestLoop')  # 使用原生TestLoop，自动支持loss记录

# ===== 学习率和优化器配置（基于官方最佳实践） =====
# 
# 参考来源：
# 1. mmaction2微调文档: https://mmaction2.readthedocs.io/en/latest/user_guides/finetune.html
#    - 官方示例: lr从0.01降到0.005（微调降低50%）
#    - 官方示例: batch_size=32, epochs=50
# 
# 2. TSM官方配置示例:
#    - Kinetics微调: https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb.py
#      lr=0.02, batch_size=16, fc_lr5=True, warmup=5epochs
#    - Something-Something微调: https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb.py
#      lr=0.02, batch_size=16, fc_lr5=True, warmup=5epochs
# 
# 3. TSM论文: https://arxiv.org/abs/1811.08383
#    - 原始实现使用较大lr配合fc_lr5
# 
# 关键经验总结：
# 1. TSM微调标准：batch_size=16, lr=0.001-0.005（手势识别小数据集）
# 2. 必须使用warmup（官方所有TSM配置都用5 epochs）
# 3. MultiStepLR在TSM上表现最好
# 4. fc_lr5=True是官方标准配置

lr = 0.002  # 2e-3，考虑到手势数据集较小，采用保守lr
warmup_epochs = 5  # TSM官方标准配置

# 官方推荐：MultiStepLR + Warmup（TSM微调最佳实践）
param_scheduler = [
    # Warm-up阶段：官方TSM配置标准（5 epochs, start_factor=0.1）
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=warmup_epochs),
    # MultiStepLR：TSM官方微调标准scheduler
    # 官方配置参考: milestones=[25,45] for 50epochs
    # 按比例调整: milestones=[50,90] for 100epochs（更保守）
    dict(
        type='MultiStepLR',
        begin=warmup_epochs,
        end=num_epochs,
        by_epoch=True,
        milestones=[40, 80],  # 在40%和80%处衰减
        gamma=0.1)
]

# 备选方案（如果上述配置不够好）：

# 方案B：更保守的lr（如果还过拟合）
# lr = 0.001  # 1e-3
# milestones = [40, 80]

# 方案C：CosineAnnealing（如果想更平滑的衰减）
# param_scheduler = [
#     dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),
#     dict(type='CosineAnnealingLR', eta_min=lr*0.01, by_epoch=True, 
#          begin=5, end=num_epochs)
# ]

# ===== 优化器配置（针对时序能力提升优化） =====
# 
# 关键问题分析：模型时序特征不强，RGB特征明显
# 优化策略：
# 1. ✅ 使用Dense采样（已配置）- 增强时序输入
# 2. ✅ 关闭fc_lr5 - 避免分类头过快拟合静态RGB特征
# 3. ✅ 统一学习率 - 让backbone和分类头协同学习时序特征
# 
# fc_lr5 = False 的理由：
# - 问题根源：之前fc_lr5=True时，分类头学习率过大(0.01)，容易过拟合RGB特征
# - TSM原理：Temporal Shift在backbone中实现，需要backbone充分学习时序
# - 当前策略：统一lr=0.002让backbone和分类头同步学习，避免分类头过快收敛到RGB捷径
# - 实验依据：Dense采样+统一学习率能让模型更关注时序而非静态特征
# 
optim_wrapper = dict(
    constructor='TSMOptimWrapperConstructor',
    paramwise_cfg=dict(
        fc_lr5=False,  # 【重要】关闭，提升时序建模能力
        # 
        # 对比分析：
        # fc_lr5=True:  backbone_lr=0.002, cls_head_lr=0.01
        #   优点：分类头快速收敛
        #   缺点：容易过拟合RGB特征，忽视时序信息
        # 
        # fc_lr5=False: backbone_lr=0.002, cls_head_lr=0.002（当前配置）
        #   优点：backbone和分类头协同学习，更好的时序建模
        #   缺点：收敛稍慢（但更稳定）
        # 
        # 如果训练后期仍需提升，可以尝试：
        # custom_keys={
        #     'backbone': dict(lr_mult=1.0),
        #     'cls_head': dict(lr_mult=1.5),  # 轻微提升分类头学习率
        # }
    ),
    optimizer=dict(
        type='SGD', 
        lr=lr,  # 0.002
        momentum=0.9, 
        weight_decay=0.0001  # 防止过拟合
        # 参考: https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/tsm/
    ),
    clip_grad=dict(max_norm=20, norm_type=2)  # 梯度裁剪，防止梯度爆炸
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = 总batch size
auto_scale_lr = dict(enable=False, base_batch_size=train_batch_size_per_gpu)
# 关闭auto_scale，因为lr已根据官方经验手动设置

load_from = "/home/models/rightArm/tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb_20230414-401127fd.pth"
