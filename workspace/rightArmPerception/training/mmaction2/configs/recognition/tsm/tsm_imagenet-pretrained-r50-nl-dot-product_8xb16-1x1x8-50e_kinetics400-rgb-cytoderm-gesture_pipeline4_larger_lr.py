_base_ = [
    # '../../_base_/models/tsm_mobilenet_v2.py',
    # '../../_base_/models/tsm_r50.py',
    '../../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'RawframeDataset'

data_root = '/home/datasets/gesture_dataset/gesture_v1_processed/images'
data_root_val = data_root

# ann_file_train = 'data/gesture_dataset/gesture_v1_processed/annotations/train.txt'
# ann_file_train = 'data/21bn-jester/21bn-jester-train-rawframedataset_debug.txt'
ann_file_train = "/home/datasets/gesture_dataset/gesture_v1_processed/train_file_raw_frame_dataset_hand_clipped.txt"
ann_file_val = "/home/datasets/gesture_dataset/gesture_v1_processed/train_file_raw_frame_dataset_hand_clipped.txt"
# ann_file_val = '/home/datasets/gesture_dataset/gesture_v1_processed/val_file_raw_frame_dataset_hand_clipped.txt'
# ann_file_val = "/home/datasets/gesture_dataset/gesture_v1_processed/train_file_raw_frame_dataset.txt"
ann_file_test = ann_file_val

num_classes = 11

num_frames = 16
train_batch_size_per_gpu = 8

test_batch_size_per_gpu = 1

file_client_args = dict(io_backend='disk')

# copied from _base_/models/tsm_r50.py
preprocess_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNetTSM',
        # pretrained='torchvision://resnet50',
        pretrained2d=False,
        depth=50,
        norm_eval=False,
        shift_div=8,
        num_segments=num_frames,
        non_local=((0, 0, 0), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 0, 0)),
        non_local_cfg=dict(
            sub_sample=True,
            use_scale=False,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            mode='dot_product')
    ),
    cls_head=dict(
        type='TSMHead',
        num_classes=num_classes,
        num_segments=num_frames,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True,
        average_clips='prob'),
    data_preprocessor=dict(type='ActionDataPreprocessor', **preprocess_cfg),
    train_cfg=None,
    test_cfg=None)

"""
SampleFrames defines sample strategy for input frames. Sample strategy is defined as clip_len x frame_interval x num_clips. There are two sample strategy, uniform sampling and dense sampling.

Dense strategy samples continuous frames. To be more specific, samples one frame every frame_interval frames and finally get clip_len frames.
Uniform strategy(clip_len == 1) devides all frames into num_clips parts and samples one frame from each part.

If there are 96 frames for one sample, frame id in range [1, 96].
Dense strategy demo: 32x2x1 may select no.1, 3, 5, ..., 63 frames.
Uniform strategy demo:1x1x8 divides 100 frames into [1, 12], [13, 24], [25, 36], [37, 48], ..., [85, 96] and may select no.1, 13, 25, ..., 85 frames.
"""

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
    # num_workers=0,
    # persistent_workers=False,
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
        pipeline=test_pipeline,
        with_offset=True,
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
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.00002),  # here lr is not used, used in OneCycleLR
    clip_grad=dict(max_norm=20, norm_type=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=train_batch_size_per_gpu)

# load_from = "/home/models/rightArm/tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb_20230414-401127fd.pth"
load_from = "/home/models/rightArm/tsm_imagenet-pretrained-r50-nl-dot-product_8xb16-1x1x8-50e_kinetics400-rgb_20220831-108bfde5.pth"
