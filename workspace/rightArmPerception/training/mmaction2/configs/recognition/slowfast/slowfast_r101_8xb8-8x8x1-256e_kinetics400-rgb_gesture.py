_base_ = [
    'slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb.py'
]

# dataset settings
dataset_type = 'RawframeDataset'

data_root = '/home/datasets/gesture_dataset/gesture_v1_processed/images'
data_root_val = data_root

ann_file_train = "/home/datasets/gesture_dataset/gesture_v1_processed/train_file_raw_frame_dataset_hand_clipped.txt"
ann_file_val = '/home/datasets/gesture_dataset/gesture_v1_processed/val_file_raw_frame_dataset_hand_clipped.txt'
ann_file_test = ann_file_val

num_classes = 11
num_frames = 16
train_batch_size_per_gpu = 16
# test_batch_size_per_gpu = 32
file_client_args = dict(io_backend='disk')

# TODO: model number of classes
model = dict(
    cls_head=dict(
        num_classes=num_classes,
    )
)

train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
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
    batch_size=8,
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
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
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

num_epochs = 100
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=num_epochs, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=40, norm_type=2))

param_scheduler = [
    # dict(
    #     type='LinearLR',
    #     start_factor=0.1,
    #     by_epoch=True,
    #     begin=0,
    #     end=34,
    #     convert_to_iter_based=True),
    # dict(
    #     type='CosineAnnealingLR',
    #     T_max=256,
    #     eta_min=0,
    #     by_epoch=True,
    #     begin=0,
    #     end=256)

    dict(
        # 默认峰值是0.3个训练过程达到顶峰（33个epoch 左右）
        type="OneCycleLR",
        # total_steps=num_epochs * len(train_dataloader["dataset"]),
        by_epoch=True,
        eta_max=0.01,
        div_factor=25,  # default
        final_div_factor=100,  # default is 1e4
        verbose=True,
    )
]

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=5),
    logger=dict(type='LoggerHook', interval=5, ignore_last=False)
)

load_from = "/home/models/rightArm/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb_20220818-9c0e09bd.pth"
