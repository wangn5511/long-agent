_base_ = [
    '../../_base_/models/tsm_mobilenet_v2.py',
    '../../_base_/default_runtime.py',
]

# 该配置用于和当前 ST-GCN 3 类版本做公平对比：
# 1. 训练/验证样本来自 MediaPipe Stage3 最终保留的同一批 clip
# 2. 输入为原始 mp4（VideoDataset），不再依赖骨架
# 3. 训练超参尽量沿用此前 2-class TSM 配置的思路

dataset_type = 'VideoDatasetWithSampleName'

class_mapping = {
    0: 'WAVE_IN',
    1: 'FIVE',
    2: 'NONE',
}
num_classes = len(class_mapping)

dataset_configs = [
    {
        'name': 'mediapipe_kept',
        'ann_file_train': '/home/datasets/cybopal_dataset/final_gesture_tsm_v1/train_list_videos.txt',
        'ann_file_val': '/home/datasets/cybopal_dataset/final_gesture_tsm_v1/val_list_videos.txt',
        'sample_ratio': 1.0,
    }
]

_total_ratio = sum(cfg['sample_ratio'] for cfg in dataset_configs)
assert abs(_total_ratio - 1.0) < 0.01, f'sample_ratio总和应为1.0，当前为{_total_ratio}'

num_frames = 8
train_batch_size_per_gpu = 8
test_batch_size_per_gpu = 1
file_client_args = dict(io_backend='disk')

preprocess_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
)

model = dict(
    backbone=dict(
        num_segments=num_frames,
    ),
    cls_head=dict(
        num_segments=num_frames,
        num_classes=num_classes,
        loss_cls=dict(type='CrossEntropyLoss'),
    ),
    data_preprocessor=dict(type='ActionDataPreprocessor', **preprocess_cfg),
)

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='CustomSampleFrames',
        clip_len=num_frames,
        frame_interval=1,
        num_clips=1,
        sample_mode='dense',
        sample_range=8,
        temporal_jitter=True,
        out_of_bound_opt='repeat_last',
        test_mode=False),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='ColorJitter', brightness=0.5, contrast=0.5, saturation=0.5, hue=0.0),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs'),
]

val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='CustomSampleFrames',
        clip_len=num_frames,
        frame_interval=1,
        num_clips=1,
        sample_mode='dense_sliding',
        window_stride=3,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs', meta_keys=(
        'img_shape', 'img_key', 'video_id', 'timestamp', 'sample_name')),
]

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='CustomSampleFrames',
        clip_len=num_frames,
        frame_interval=1,
        num_clips=1,
        sample_mode='dense_sliding',
        window_stride=3,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs', meta_keys=(
        'img_shape', 'img_key', 'video_id', 'timestamp', 'sample_name')),
]

train_datasets = []
for cfg in dataset_configs:
    train_datasets.append(
        dict(
            type=dataset_type,
            ann_file=cfg['ann_file_train'],
            data_prefix=dict(video=''),
            pipeline=train_pipeline,
        ))

val_datasets = []
for cfg in dataset_configs:
    val_datasets.append(
        dict(
            type=dataset_type,
            ann_file=cfg['ann_file_val'],
            data_prefix=dict(video=''),
            pipeline=val_pipeline,
            test_mode=True,
        ))

test_datasets = []
for cfg in dataset_configs:
    test_datasets.append(
        dict(
            type=dataset_type,
            ann_file=cfg['ann_file_val'],
            data_prefix=dict(video=''),
            pipeline=test_pipeline,
            test_mode=True,
        ))

combined_trainset = dict(type='ConcatDataset', datasets=train_datasets)
combined_valset = dict(type='ConcatDataset', datasets=val_datasets)
combined_testset = dict(type='ConcatDataset', datasets=test_datasets)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=16,
    persistent_workers=True,
    prefetch_factor=4,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=combined_trainset,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=2,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=combined_valset,
)

test_dataloader = dict(
    batch_size=test_batch_size_per_gpu,
    num_workers=1,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=combined_testset,
)

val_evaluator = [
    dict(type='AccMetric', prefix='overall_acc'),
    dict(
        type='AccMetric',
        metric_options=dict(topk=(1,), classwise=True),
        prefix='classwise_acc'),
]

test_evaluator = [
    dict(type='AccMetric', prefix='overall_acc'),
    dict(
        type='AccMetric',
        metric_options=dict(topk=(1,), classwise=True),
        prefix='classwise_acc'),
    dict(
        type='ConfusionMatrix',
        num_classes=num_classes,
        save_path='confusion_matrix_val.npy',
        save_misclassified=True,
        misclassified_save_path='misclassified_val.txt',
        save_classified=True,
        classified_save_path='classified_val.txt',
        class_mapping=class_mapping),
]

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=5),
    logger=dict(
        type='LoggerHook',
        interval=5,
        ignore_last=False,
        log_metric_by_epoch=True),
)

visualizer = dict(
    type='ActionVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ],
)

num_epochs = 80
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=num_epochs, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

lr = 0.002
warmup_epochs = 5
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=warmup_epochs),
    dict(
        type='MultiStepLR',
        begin=warmup_epochs,
        end=num_epochs,
        by_epoch=True,
        milestones=[40, 60],
        gamma=0.1),
]

optim_wrapper = dict(
    constructor='TSMOptimWrapperConstructor',
    paramwise_cfg=dict(fc_lr5=False),
    optimizer=dict(
        type='SGD',
        lr=lr,
        momentum=0.9,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=20, norm_type=2),
)

auto_scale_lr = dict(enable=False, base_batch_size=train_batch_size_per_gpu)

load_from = '/home/datasets/rest_gesture_dataset/jester_epoch_100_backbone.pth'

