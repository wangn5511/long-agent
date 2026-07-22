ann_file = '/home/results/6t1/gesture_data_v2/rtm_class10_WAVE_OUT.pkl'
auto_scale_lr = dict(base_batch_size=32, enable=False)
categories = [
    'WAVE_IN',
    'FIVE',
    'CLICK',
    'ROTATE',
    'SLIDE_DOWN',
    'SLIDE_UP',
    'OK',
    'CURSOR',
    'NONE',
    'WAVE_OUT',
]
class_mapping = dict({
    0: 'WAVE_IN',
    1: 'FIVE',
    2: 'CLICK',
    3: 'ROTATE',
    4: 'SLIDE_DOWN',
    5: 'SLIDE_UP',
    6: 'OK',
    7: 'CURSOR',
    8: 'NONE',
    9: 'WAVE_OUT'
})
custom_hooks = [
    dict(
        min_delta=0.001,
        monitor='acc/top1',
        patience=8,
        rule='greater',
        strict=False,
        type='EarlyStoppingHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'handoff.stgcn_hand21_3d_rgb_wave_out.code.backbones',
        'handoff.stgcn_hand21_3d_rgb_wave_out.code.heads',
        'handoff.stgcn_hand21_3d_rgb_wave_out.code.recognizers',
        'handoff.stgcn_hand21_3d_rgb_wave_out.code.transforms',
        'handoff.stgcn_hand21_3d_rgb_wave_out.code.preprocessors',
        'mmaction.datasets.transforms.video_roi_decode',
    ])
dataset_type = 'PoseDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=1,
        rule='greater',
        save_best='acc/top1',
        save_last=True,
        type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=25, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
hand21_layout = dict(
    center=0,
    inward=[
        (
            1,
            0,
        ),
        (
            2,
            1,
        ),
        (
            3,
            2,
        ),
        (
            4,
            3,
        ),
        (
            5,
            0,
        ),
        (
            6,
            5,
        ),
        (
            7,
            6,
        ),
        (
            8,
            7,
        ),
        (
            9,
            0,
        ),
        (
            10,
            9,
        ),
        (
            11,
            10,
        ),
        (
            12,
            11,
        ),
        (
            13,
            0,
        ),
        (
            14,
            13,
        ),
        (
            15,
            14,
        ),
        (
            16,
            15,
        ),
        (
            17,
            0,
        ),
        (
            18,
            17,
        ),
        (
            19,
            18,
        ),
        (
            20,
            19,
        ),
    ],
    num_node=21)
launcher = 'none'
load_from = 'handoff/stgcn_hand21_3d_rgb_wave_out/weights/best_acc_top1_epoch_22.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
meta_keys = (
    'img_shape',
    'handedness_label',
    'handedness_label_id',
    'track_id',
    'video_name',
)
model = dict(
    backbone=dict(
        graph_cfg=dict(
            layout=dict(
                center=0,
                inward=[
                    (
                        1,
                        0,
                    ),
                    (
                        2,
                        1,
                    ),
                    (
                        3,
                        2,
                    ),
                    (
                        4,
                        3,
                    ),
                    (
                        5,
                        0,
                    ),
                    (
                        6,
                        5,
                    ),
                    (
                        7,
                        6,
                    ),
                    (
                        8,
                        7,
                    ),
                    (
                        9,
                        0,
                    ),
                    (
                        10,
                        9,
                    ),
                    (
                        11,
                        10,
                    ),
                    (
                        12,
                        11,
                    ),
                    (
                        13,
                        0,
                    ),
                    (
                        14,
                        13,
                    ),
                    (
                        15,
                        14,
                    ),
                    (
                        16,
                        15,
                    ),
                    (
                        17,
                        0,
                    ),
                    (
                        18,
                        17,
                    ),
                    (
                        19,
                        18,
                    ),
                    (
                        20,
                        19,
                    ),
                ],
                num_node=21),
            mode='stgcn_spatial'),
        in_channels=3,
        type='STGCN4D'),
    cls_head=dict(
        contrast_loss_weight=0.01,
        contrast_proj_dim=128,
        contrast_temperature=0.1,
        dropout=0.35,
        enable_contrast_loss=True,
        handedness_embed_dim=8,
        hidden_channels=256,
        motion_channels=8,
        num_classes=10,
        rgb_channels=48,
        skel_channels=256,
        type='GCNHead3DGlobalMotionHandednessRGB',
        use_handedness_score=False),
    data_preprocessor=dict(type='BundleGCNRGBDataPreprocessor'),
    motion_branch=dict(
        in_channels=3, proj_channels=8, type='GlobalMotionBranch'),
    rgb_branch=dict(freeze_cnn=True, proj_channels=48, type='RGBBranch'),
    type='RecognizerGCNGlobalMotionHandednessRGB')
num_classes = 10
optim_wrapper = dict(
    loss_scale='dynamic',
    optimizer=dict(
        lr=0.2, momentum=0.9, nesterov=True, type='SGD', weight_decay=0.0005),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        T_max=100,
        by_epoch=True,
        convert_to_iter_based=True,
        eta_min=0,
        type='CosineAnnealingLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='/home/results/6t1/gesture_data_v2/rtm_class10_WAVE_OUT.pkl',
        pipeline=[
            dict(
                abs_energy_floor=0.003,
                clip_len=16,
                core_classes=(2, ),
                crop_mode='center',
                enforce_quality_gate=False,
                num_clips=1,
                onset_threshold=0.4,
                peak_smooth_window=3,
                pre_context=1,
                primary_margin_frames_thr=2,
                primary_min_longest_run=8,
                primary_valid_ratio_thr=0.75,
                secondary_max_ratio_thr=0.25,
                secondary_total_ratio_thr=0.375,
                track_valid_score_threshold=0.0,
                type='TrackAwareClickDeterministicOnsetSampleFrames'),
            dict(
                allow_fallback=True,
                auto_select_if_missing=True,
                type='SelectLocalPrimaryTrack',
                valid_score_threshold=0.0),
            dict(
                padding_ratio=0.1,
                target_size=(
                    96,
                    96,
                ),
                type='DecodeVideoROI'),
            dict(type='PoseDecode'),
            dict(type='ExtractGlobalCenterTrajectory'),
            dict(type='Hand21WorldNormalize'),
            dict(type='FormatGlobalTrajectory'),
            dict(include_score=False, num_person=1, type='FormatGCNInput3D'),
            dict(
                algorithm_keys=('global_traj', ),
                collect_keys=(
                    'keypoint',
                    'imgs',
                ),
                meta_keys=(
                    'img_shape',
                    'handedness_label',
                    'handedness_label_id',
                    'track_id',
                    'video_name',
                ),
                type='PackActionInputs'),
        ],
        split='val',
        test_mode=True,
        type='PoseDataset'),
    num_workers=12,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(type='AccMetric'),
    dict(
        class_mapping=dict({
            0: 'WAVE_IN',
            1: 'FIVE',
            2: 'CLICK',
            3: 'ROTATE',
            4: 'SLIDE_DOWN',
            5: 'SLIDE_UP',
            6: 'OK',
            7: 'CURSOR',
            8: 'NONE',
            9: 'WAVE_OUT'
        }),
        classified_save_path=
        'classified_test.txt',
        misclassified_save_path=
        'misclassified_test.txt',
        num_classes=10,
        save_classified=True,
        save_misclassified=True,
        save_path=
        'confusion_matrix_test.npy',
        type='ConfusionMatrix'),
]
test_pipeline = [
    dict(
        abs_energy_floor=0.003,
        clip_len=16,
        core_classes=(2, ),
        crop_mode='center',
        enforce_quality_gate=False,
        num_clips=1,
        onset_threshold=0.4,
        peak_smooth_window=3,
        pre_context=1,
        primary_margin_frames_thr=2,
        primary_min_longest_run=8,
        primary_valid_ratio_thr=0.75,
        secondary_max_ratio_thr=0.25,
        secondary_total_ratio_thr=0.375,
        track_valid_score_threshold=0.0,
        type='TrackAwareClickDeterministicOnsetSampleFrames'),
    dict(
        allow_fallback=True,
        auto_select_if_missing=True,
        type='SelectLocalPrimaryTrack',
        valid_score_threshold=0.0),
    dict(padding_ratio=0.1, target_size=(
        96,
        96,
    ), type='DecodeVideoROI'),
    dict(type='PoseDecode'),
    dict(type='ExtractGlobalCenterTrajectory'),
    dict(type='Hand21WorldNormalize'),
    dict(type='FormatGlobalTrajectory'),
    dict(include_score=False, num_person=1, type='FormatGCNInput3D'),
    dict(
        algorithm_keys=('global_traj', ),
        collect_keys=(
            'keypoint',
            'imgs',
        ),
        meta_keys=(
            'img_shape',
            'handedness_label',
            'handedness_label_id',
            'track_id',
            'video_name',
        ),
        type='PackActionInputs'),
]
track_policy = dict(
    primary_margin_frames_thr=2,
    primary_min_longest_run=8,
    primary_valid_ratio_thr=0.75,
    secondary_max_ratio_thr=0.25,
    secondary_total_ratio_thr=0.375,
    track_valid_score_threshold=0.0)
train_cfg = dict(
    max_epochs=100, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        dataset=dict(
            ann_file=
            '/home/results/6t1/gesture_data_v2/rtm_class10_WAVE_OUT.pkl',
            pipeline=[
                dict(
                    abs_energy_floor=0.003,
                    clip_len=16,
                    continuous_ratio=0.8,
                    continuous_start_min_ratio=0.2,
                    core_classes=(2, ),
                    fallback_policy='best_quality',
                    full_clip_ratio=0.0,
                    jitter_range=2,
                    local_max_coverage=24,
                    local_max_coverage_ratio=1.0,
                    local_min_coverage=20,
                    local_ratio=0.2,
                    max_resample_attempts=8,
                    onset_threshold=0.4,
                    peak_smooth_window=3,
                    pre_context=1,
                    primary_margin_frames_thr=2,
                    primary_min_longest_run=8,
                    primary_valid_ratio_thr=0.75,
                    secondary_max_ratio_thr=0.25,
                    secondary_total_ratio_thr=0.375,
                    track_valid_score_threshold=0.0,
                    type='TrackAwareCoreAwareSampleFrames'),
                dict(
                    allow_fallback=True,
                    auto_select_if_missing=True,
                    type='SelectLocalPrimaryTrack',
                    valid_score_threshold=0.0),
                dict(
                    padding_ratio=0.1,
                    target_size=(
                        96,
                        96,
                    ),
                    type='DecodeVideoROI'),
                dict(type='PoseDecode'),
                dict(type='ExtractGlobalCenterTrajectory'),
                dict(type='Hand21WorldNormalize'),
                dict(type='FormatGlobalTrajectory'),
                dict(
                    include_score=False, num_person=1,
                    type='FormatGCNInput3D'),
                dict(
                    algorithm_keys=('global_traj', ),
                    collect_keys=(
                        'keypoint',
                        'imgs',
                    ),
                    meta_keys=(
                        'img_shape',
                        'handedness_label',
                        'handedness_label_id',
                        'track_id',
                        'video_name',
                    ),
                    type='PackActionInputs'),
            ],
            split='train',
            type='PoseDataset'),
        times=2,
        type='RepeatDataset'),
    num_workers=12,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        abs_energy_floor=0.003,
        clip_len=16,
        continuous_ratio=0.8,
        continuous_start_min_ratio=0.2,
        core_classes=(2, ),
        fallback_policy='best_quality',
        full_clip_ratio=0.0,
        jitter_range=2,
        local_max_coverage=24,
        local_max_coverage_ratio=1.0,
        local_min_coverage=20,
        local_ratio=0.2,
        max_resample_attempts=8,
        onset_threshold=0.4,
        peak_smooth_window=3,
        pre_context=1,
        primary_margin_frames_thr=2,
        primary_min_longest_run=8,
        primary_valid_ratio_thr=0.75,
        secondary_max_ratio_thr=0.25,
        secondary_total_ratio_thr=0.375,
        track_valid_score_threshold=0.0,
        type='TrackAwareCoreAwareSampleFrames'),
    dict(
        allow_fallback=True,
        auto_select_if_missing=True,
        type='SelectLocalPrimaryTrack',
        valid_score_threshold=0.0),
    dict(padding_ratio=0.1, target_size=(
        96,
        96,
    ), type='DecodeVideoROI'),
    dict(type='PoseDecode'),
    dict(type='ExtractGlobalCenterTrajectory'),
    dict(type='Hand21WorldNormalize'),
    dict(type='FormatGlobalTrajectory'),
    dict(include_score=False, num_person=1, type='FormatGCNInput3D'),
    dict(
        algorithm_keys=('global_traj', ),
        collect_keys=(
            'keypoint',
            'imgs',
        ),
        meta_keys=(
            'img_shape',
            'handedness_label',
            'handedness_label_id',
            'track_id',
            'video_name',
        ),
        type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='/home/results/6t1/gesture_data_v2/rtm_class10_WAVE_OUT.pkl',
        pipeline=[
            dict(
                abs_energy_floor=0.003,
                clip_len=16,
                core_classes=(2, ),
                crop_mode='center',
                enforce_quality_gate=False,
                num_clips=1,
                onset_threshold=0.4,
                peak_smooth_window=3,
                pre_context=1,
                primary_margin_frames_thr=2,
                primary_min_longest_run=8,
                primary_valid_ratio_thr=0.75,
                secondary_max_ratio_thr=0.25,
                secondary_total_ratio_thr=0.375,
                track_valid_score_threshold=0.0,
                type='TrackAwareClickDeterministicOnsetSampleFrames'),
            dict(
                allow_fallback=True,
                auto_select_if_missing=True,
                type='SelectLocalPrimaryTrack',
                valid_score_threshold=0.0),
            dict(
                padding_ratio=0.1,
                target_size=(
                    96,
                    96,
                ),
                type='DecodeVideoROI'),
            dict(type='PoseDecode'),
            dict(type='ExtractGlobalCenterTrajectory'),
            dict(type='Hand21WorldNormalize'),
            dict(type='FormatGlobalTrajectory'),
            dict(include_score=False, num_person=1, type='FormatGCNInput3D'),
            dict(
                algorithm_keys=('global_traj', ),
                collect_keys=(
                    'keypoint',
                    'imgs',
                ),
                meta_keys=(
                    'img_shape',
                    'handedness_label',
                    'handedness_label_id',
                    'track_id',
                    'video_name',
                ),
                type='PackActionInputs'),
        ],
        split='val',
        test_mode=True,
        type='PoseDataset'),
    num_workers=12,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(type='AccMetric'),
]
val_pipeline = [
    dict(
        abs_energy_floor=0.003,
        clip_len=16,
        core_classes=(2, ),
        crop_mode='center',
        enforce_quality_gate=False,
        num_clips=1,
        onset_threshold=0.4,
        peak_smooth_window=3,
        pre_context=1,
        primary_margin_frames_thr=2,
        primary_min_longest_run=8,
        primary_valid_ratio_thr=0.75,
        secondary_max_ratio_thr=0.25,
        secondary_total_ratio_thr=0.375,
        track_valid_score_threshold=0.0,
        type='TrackAwareClickDeterministicOnsetSampleFrames'),
    dict(
        allow_fallback=True,
        auto_select_if_missing=True,
        type='SelectLocalPrimaryTrack',
        valid_score_threshold=0.0),
    dict(padding_ratio=0.1, target_size=(
        96,
        96,
    ), type='DecodeVideoROI'),
    dict(type='PoseDecode'),
    dict(type='ExtractGlobalCenterTrajectory'),
    dict(type='Hand21WorldNormalize'),
    dict(type='FormatGlobalTrajectory'),
    dict(include_score=False, num_person=1, type='FormatGCNInput3D'),
    dict(
        algorithm_keys=('global_traj', ),
        collect_keys=(
            'keypoint',
            'imgs',
        ),
        meta_keys=(
            'img_shape',
            'handedness_label',
            'handedness_label_id',
            'track_id',
            'video_name',
        ),
        type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/home/results/6t1/stgcn_3d_rgb_wave_out'
