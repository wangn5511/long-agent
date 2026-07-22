_base_ = [
    '../../_base_/models/tsm_mobilenet_v2.py',
    '../../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'RawframeDataset'

data_root = '/home/datasets/gesture_dataset/gesture_v1_processed/images'
data_root_val = data_root

# ann_file_train = 'data/gesture_dataset/gesture_v1_processed/annotations/train.txt'
# ann_file_train = 'data/21bn-jester/21bn-jester-train-rawframedataset_debug.txt'
ann_file_train = "/home/datasets/gesture_dataset/gesture_v1_processed/train_file_raw_frame_dataset_hand_clipped.txt"
ann_file_val = '/home/datasets/gesture_dataset/gesture_v1_processed/val_file_raw_frame_dataset_hand_clipped.txt'
# ann_file_val = "/home/datasets/gesture_dataset/gesture_v1_processed/train_file_raw_frame_dataset.txt"
ann_file_test = ann_file_val

num_classes = 11

num_frames = 16
train_batch_size_per_gpu = 16

test_batch_size_per_gpu = 1

file_client_args = dict(io_backend='disk')

model = dict(
    backbone=dict(
        num_segments=num_frames,
    ),
    cls_head=dict(
        num_segments=num_frames,
        num_classes=num_classes)
)

train_pipeline = [
    dict(type='SampleFrames', 
        clip_len=1, 
        frame_interval=1, 
        num_clips=num_frames,
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
    dict(type='ColorJitter', brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=num_frames,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=num_frames,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        filename_tmpl='frame_{:04}.png',
        with_offset=True,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        filename_tmpl='frame_{:04}.png',
        with_offset=True,
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=test_batch_size_per_gpu,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        filename_tmpl='frame_{:04}.png',
        with_offset=True,
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
# test_evaluator = val_evaluator
test_evaluator = [
    dict(type='AccMetric'),
    dict(type='ConfusionMatrix', num_classes=num_classes,
        save_path='confusion_matrix_val.npy',
        save_misclassified=True,
        misclassified_save_path='misclassified_val.txt',
        save_classified=True,
        classified_save_path='classified_val.txt')
]

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=5),
    logger=dict(type='LoggerHook', interval=5, ignore_last=False)
)

num_epochs = 100

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=num_epochs, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    # dict(
    #     type='MultiStepLR',
    #     begin=0,
    #     end=100,
    #     by_epoch=True,
    #     milestones=[40, 80],
    #     gamma=0.1)
    dict(
        type="OneCycleLR",
        # total_steps=num_epochs * len(train_dataloader["dataset"]),
        by_epoch=True,
        eta_max=0.1,
        div_factor=25,  # default
        final_div_factor=100,  # default is 1e4
        verbose=True,
    )
]

optim_wrapper = dict(
    constructor='TSMOptimWrapperConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.00002),
    clip_grad=dict(max_norm=20, norm_type=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=train_batch_size_per_gpu)

load_from = "/home/models/rightArm/tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb_20230414-401127fd.pth"
