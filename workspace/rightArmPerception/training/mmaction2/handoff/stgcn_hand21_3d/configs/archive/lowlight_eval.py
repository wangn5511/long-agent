_base_ = '../../../configs/_base_/default_runtime.py'

custom_imports = dict(
    imports=[
        'handoff.stgcn_hand21_4d.code.backbones',
        'handoff.stgcn_hand21_4d.code.heads',
        'handoff.stgcn_hand21_4d.code.transforms',
    ],
    allow_failed_imports=False)

categories = [
    'WAVE_IN',
    'FIVE',
    'CLICK',
    'ROTATE',
    'SLIDE_DOWN',
    'SLIDE_UP',
    'OK',
    'CURSOR',
    'NONE'
]
num_classes = len(categories)
class_mapping = {idx: name for idx, name in enumerate(categories)}

hand21_layout = dict(
    num_node=21,
    inward=[
        (1, 0),
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 0),
        (6, 5),
        (7, 6),
        (8, 7),
        (9, 0),
        (10, 9),
        (11, 10),
        (12, 11),
        (13, 0),
        (14, 13),
        (15, 14),
        (16, 15),
        (17, 0),
        (18, 17),
        (19, 18),
        (20, 19),
    ],
    center=0)

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN4D',
        graph_cfg=dict(layout=hand21_layout, mode='stgcn_spatial')),
    cls_head=dict(type='GCNHead4D', num_classes=num_classes, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = '/home/results/6t1/gesture_class9_lowlight_eval/gesture_class9_lowlight_eval.pkl'

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='UniformSampleFrames', clip_len=16),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput4D', num_person=1),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='UniformSampleFrames', clip_len=16, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput4D', num_person=1),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='UniformSampleFrames', clip_len=16, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput4D', num_person=1),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            pipeline=train_pipeline,
            split='train')))
val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split='val',
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        split='val',
        test_mode=True))

val_evaluator = [dict(type='AccMetric')]
test_evaluator = [
    dict(type='AccMetric'),
    dict(
        type='ConfusionMatrix',
        num_classes=num_classes,
        save_path='confusion_matrix_test.npy',
        save_misclassified=True,
        misclassified_save_path='misclassified_test.txt',
        save_classified=True,
        classified_save_path='classified_test.txt',
        class_mapping=class_mapping),
]

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=100, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=100,
        by_epoch=True,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005, nesterov=True))

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        save_best='acc/top1',
        rule='greater',
        save_last=True,
        max_keep_ckpts=1),
    logger=dict(interval=100))

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='acc/top1',
        rule='greater',
        min_delta=0.001,
        patience=5,
        strict=False)
]

auto_scale_lr = dict(enable=False, base_batch_size=64)
