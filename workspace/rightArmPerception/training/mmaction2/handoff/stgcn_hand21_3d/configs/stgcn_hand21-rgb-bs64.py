default_scope = "mmaction"

default_hooks = dict(
    runtime_info=dict(type="RuntimeInfoHook"),
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=25, ignore_last=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        save_best="acc/top1",
        rule="greater",
        save_last=True,
        max_keep_ckpts=1,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    sync_buffers=dict(type="SyncBuffersHook"),
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

log_processor = dict(type="LogProcessor", window_size=20, by_epoch=True)

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="ActionVisualizer", vis_backends=vis_backends)

log_level = "INFO"
load_from = None
resume = False

custom_imports = dict(
    imports=[
        "handoff.stgcn_hand21_3d.code.backbones",
        "handoff.stgcn_hand21_3d.code.heads",
        "handoff.stgcn_hand21_3d.code.recognizers",
        "handoff.stgcn_hand21_3d.code.transforms",
        "mmaction.datasets.transforms.video_roi_decode",
        "mmaction.models.data_preprocessors.gcn_rgb_preprocessor",
    ],
    allow_failed_imports=False,
)

categories = [
    "WAVE_IN",
    "FIVE",
    "CLICK",
    "ROTATE",
    "SLIDE_DOWN",
    "SLIDE_UP",
    "OK",
    "CURSOR",
    "NONE",
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
    center=0,
)

dataset_type = "PoseDataset"
ann_file = "/home/results/6t1/gesture_class9/rtm_class9_hand21_3d_windowtrack_v2supp_click_cursor_none.pkl"

meta_keys = (
    "img_shape",
    "handedness_label",
    "handedness_label_id",
    "track_id",
    "video_name",
)

track_policy = dict(
    track_valid_score_threshold=0.0,
    primary_valid_ratio_thr=0.75,
    primary_min_longest_run=8,
    secondary_max_ratio_thr=0.25,
    secondary_total_ratio_thr=0.375,
    primary_margin_frames_thr=2,
)

model = dict(
    type="RecognizerGCNGlobalMotionHandednessRGB",
    data_preprocessor=dict(type="GCNRGBDataPreprocessor"),
    backbone=dict(
        type="STGCN4D",
        in_channels=3,
        graph_cfg=dict(layout=hand21_layout, mode="stgcn_spatial"),
    ),
    motion_branch=dict(
        type="GlobalMotionBranch",
        in_channels=3,
        proj_channels=8,
    ),
    rgb_branch=dict(
        type="RGBBranch",
        proj_channels=48,
        freeze_cnn=True,
    ),
    cls_head=dict(
        type="GCNHead3DGlobalMotionHandednessRGB",
        num_classes=num_classes,
        skel_channels=256,
        motion_channels=8,
        rgb_channels=48,
        hidden_channels=256,
        handedness_embed_dim=8,
        use_handedness_score=False,
        dropout=0.35,
        enable_contrast_loss=True,
        contrast_proj_dim=128,
        contrast_temperature=0.1,
        contrast_loss_weight=0.01,
    ),
)

train_pipeline = [
    dict(
        type="TrackAwareCoreAwareSampleFrames",
        clip_len=16,
        continuous_ratio=0.8,
        local_ratio=0.2,
        full_clip_ratio=0.0,
        local_min_coverage=20,
        local_max_coverage=24,
        local_max_coverage_ratio=1.0,
        continuous_start_min_ratio=0.2,
        core_classes=(2,),
        peak_smooth_window=3,
        onset_threshold=0.4,
        abs_energy_floor=0.003,
        pre_context=1,
        jitter_range=2,
        max_resample_attempts=8,
        fallback_policy="best_quality",
        **track_policy,
    ),
    dict(
        type="SelectLocalPrimaryTrack",
        allow_fallback=True,
        auto_select_if_missing=True,
        valid_score_threshold=0.0,
    ),
    dict(type="DecodeVideoROI", target_size=(96, 96), padding_ratio=0.1),
    dict(type="PoseDecode"),
    dict(type="ExtractGlobalCenterTrajectory"),
    dict(type="Hand21WorldNormalize"),
    dict(type="FormatGlobalTrajectory"),
    dict(type="FormatGCNInput3D", num_person=1, include_score=False),
    dict(
        type="PackActionInputs",
        meta_keys=meta_keys,
        collect_keys=("keypoint", "imgs"),
        algorithm_keys=("global_traj",),
    ),
]

val_pipeline = [
    dict(
        type="TrackAwareClickDeterministicOnsetSampleFrames",
        clip_len=16,
        num_clips=1,
        crop_mode="center",
        core_classes=(2,),
        peak_smooth_window=3,
        onset_threshold=0.4,
        abs_energy_floor=0.003,
        pre_context=1,
        enforce_quality_gate=False,
        **track_policy,
    ),
    dict(
        type="SelectLocalPrimaryTrack",
        allow_fallback=True,
        auto_select_if_missing=True,
        valid_score_threshold=0.0,
    ),
    dict(type="DecodeVideoROI", target_size=(96, 96), padding_ratio=0.1),
    dict(type="PoseDecode"),
    dict(type="ExtractGlobalCenterTrajectory"),
    dict(type="Hand21WorldNormalize"),
    dict(type="FormatGlobalTrajectory"),
    dict(type="FormatGCNInput3D", num_person=1, include_score=False),
    dict(
        type="PackActionInputs",
        meta_keys=meta_keys,
        collect_keys=("keypoint", "imgs"),
        algorithm_keys=("global_traj",),
    ),
]

test_pipeline = [
    dict(
        type="TrackAwareClickDeterministicOnsetSampleFrames",
        clip_len=16,
        num_clips=1,
        crop_mode="center",
        core_classes=(2,),
        peak_smooth_window=3,
        onset_threshold=0.4,
        abs_energy_floor=0.003,
        pre_context=1,
        enforce_quality_gate=False,
        **track_policy,
    ),
    dict(
        type="SelectLocalPrimaryTrack",
        allow_fallback=True,
        auto_select_if_missing=True,
        valid_score_threshold=0.0,
    ),
    dict(type="DecodeVideoROI", target_size=(96, 96), padding_ratio=0.1),
    dict(type="PoseDecode"),
    dict(type="ExtractGlobalCenterTrajectory"),
    dict(type="Hand21WorldNormalize"),
    dict(type="FormatGlobalTrajectory"),
    dict(type="FormatGCNInput3D", num_person=1, include_score=False),
    dict(
        type="PackActionInputs",
        meta_keys=meta_keys,
        collect_keys=("keypoint", "imgs"),
        algorithm_keys=("global_traj",),
    ),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=12,
    prefetch_factor=4,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="RepeatDataset",
        times=2,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            pipeline=train_pipeline,
            split="train",
        ),
    ),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=12,
    prefetch_factor=4,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split="val",
        test_mode=True,
    ),
)

test_dataloader = dict(
    batch_size=32,
    num_workers=12,
    prefetch_factor=4,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        split="val",
        test_mode=True,
    ),
)

val_evaluator = [dict(type="AccMetric")]
test_evaluator = [
    dict(type="AccMetric"),
    dict(
        type="ConfusionMatrix",
        num_classes=num_classes,
        save_path="confusion_matrix_test.npy",
        save_misclassified=True,
        misclassified_save_path="misclassified_test.txt",
        save_classified=True,
        classified_save_path="classified_test.txt",
        class_mapping=class_mapping,
    ),
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=100, val_begin=1, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [
    dict(type="CosineAnnealingLR", eta_min=0, T_max=100, by_epoch=True, convert_to_iter_based=True)
]

optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="SGD", lr=0.20, momentum=0.9, weight_decay=0.0005, nesterov=True),
    loss_scale="dynamic",
)

custom_hooks = [
    dict(
        type="EarlyStoppingHook",
        monitor="acc/top1",
        rule="greater",
        min_delta=0.001,
        patience=8,
        strict=False,
    )
]

auto_scale_lr = dict(enable=False, base_batch_size=32)
