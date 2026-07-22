# Augmentation variant of stgcn_hand21-rgb-bs64-gated.py -- adds 3 train-only
# transforms to address two real-world failure modes:
#   1. Camera rotation (esp. 90 deg): JointRandomRotate
#      (handoff.stgcn_hand21_3d.code.transforms.geom_aug) rotates frames +
#      keypoints + global_traj_source synchronously.  Discrete
#      {0, 90, 180, 270} deg + continuous +/- 15 deg by default.
#   2. Far hand-camera distance: JointKeypointJitter (Gaussian xy noise) +
#      RGBQualityJitter (downsample-then-upsample) simulate degraded detector
#      accuracy and low-detail RGB.
# JointKeypointJitter is placed BEFORE DecodeVideoROI so noisy keypoints also
# perturb the ROI bbox -- this matches what would happen with a real noisy
# detector at deployment.  val/test pipelines unchanged.
# ann_file already points at the v2 dataset.
#
# 4-stream fusion (skeleton + global-motion + RGB + handedness) with a binary
# gesture/NONE gate head and absolute-position-aware motion branch.
#
# Combines:
#   * (A) stgcn_hand21-rgb-bs64.py     -> 4-stream concat fusion with RGB ROI
#   * (B) stgcn_3d-4ch_..._gated-head_mc16.py -> sigmoid gate head + BCE loss
#
# Changes vs. stgcn_hand21-rgb-bs64.py:
#   1. cls_head: GCNHead3DGlobalMotionHandednessRGBGated  (adds gate_head)
#   2. motion_branch.in_channels: 3 -> 5  (delta xyz + absolute xy)
#   3. FormatGlobalTrajectory(num_traj_channels=5)
#   4. DecodeVideoROI now imported from this handoff's local override
#      (handoff.stgcn_hand21_3d.code.transforms.video_roi_decode), which
#      derives the hand bbox from the 21 keypoints instead of relying on the
#      pkl's all-None bbox_per_frame field.
#   5. 10-class dataset (rtm_3d_class10.pkl, WAVE_OUT included).
#
# Fusion shape (post-pool, pre-fc1):
#     skel(256) || motion(8) || rgb(48) || hand(8) = 320 -> fc1 -> 256 -> fc2 -> 10
#     gate_head: 256 -> 1 (sigmoid, BCE vs is_gesture)
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
        # NOTE: do NOT also import the upstream mmaction version of
        # DecodeVideoROI here.  Our local version registers with
        # force=True so whichever loads last wins; keeping only ours
        # avoids depending on import order.
        "mmaction.models.data_preprocessors.gcn_rgb_preprocessor",
    ],
    allow_failed_imports=False,
)

categories = [
    "WAVE_IN",
    "WAVE_OUT",
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
none_class_id = categories.index("NONE")
click_class_id = categories.index("CLICK")

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
ann_file = "/home/results/6t1/gesture_data/rtm_3d_class10_v2.pkl"

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

# Trajectory layout (in order):
#   ch 0,1,2 -> delta xyz (relative to first valid frame)
#   ch 3,4   -> absolute xy normalized by (img_w, img_h)
num_traj_channels = 5
motion_in_channels = num_traj_channels  # 5

model = dict(
    type="RecognizerGCNGlobalMotionHandednessRGB",
    data_preprocessor=dict(type="GCNRGBDataPreprocessor"),
    backbone=dict(
        type="STGCN4D",
        # 4 channels: (x, y, z, keypoint_score) - the score channel lets the
        # GCN attend to / down-weight low-confidence joints (e.g. occluded
        # fingers).  Matches the mc16 gated-head baseline so this experiment
        # only changes the fusion side (+ RGB, + abs_xy), not the skeleton
        # input format.
        in_channels=4,
        graph_cfg=dict(layout=hand21_layout, mode="stgcn_spatial"),
    ),
    motion_branch=dict(
        type="GlobalMotionBranch",
        in_channels=motion_in_channels,  # 5: delta xyz + abs xy
        proj_channels=8,
    ),
    rgb_branch=dict(
        type="RGBBranch",
        proj_channels=48,
        freeze_cnn=True,
    ),
    cls_head=dict(
        type="GCNHead3DGlobalMotionHandednessRGBGated",
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
        # --- gate head (matches gated-head mc16 defaults) ---
        none_class_id=none_class_id,
        gate_loss_weight=0.5,
        gate_threshold=0.45,
        use_gate_at_inference=True,
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
        core_classes=(click_class_id,),  # CLICK = 3 in this 10-class layout
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
    # aug: rotate frame + keypoints + traj source consistently
    dict(type="JointRandomRotate"),
    # aug: simulate noisy keypoints (placed BEFORE DecodeVideoROI so noisy
    # keypoints also affect the ROI bbox derivation, matching real
    # detector behaviour at far distance)
    dict(type="JointKeypointJitter"),
    dict(type="DecodeVideoROI", target_size=(96, 96), padding_ratio=0.25),
    # aug: simulate low-resolution RGB ROI at far hand-camera distance
    dict(type="RGBQualityJitter"),
    dict(type="PoseDecode"),
    dict(type="ExtractGlobalCenterTrajectory", emit_abs_xy=True),
    dict(type="Hand21WorldNormalize"),
    dict(type="FormatGlobalTrajectory", num_traj_channels=num_traj_channels),
    dict(type="FormatGCNInput3D", num_person=1, include_score=True),
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
        core_classes=(click_class_id,),
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
    dict(type="DecodeVideoROI", target_size=(96, 96), padding_ratio=0.25),
    dict(type="PoseDecode"),
    dict(type="ExtractGlobalCenterTrajectory", emit_abs_xy=True),
    dict(type="Hand21WorldNormalize"),
    dict(type="FormatGlobalTrajectory", num_traj_channels=num_traj_channels),
    dict(type="FormatGCNInput3D", num_person=1, include_score=True),
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
        core_classes=(click_class_id,),
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
    dict(type="DecodeVideoROI", target_size=(96, 96), padding_ratio=0.25),
    dict(type="PoseDecode"),
    dict(type="ExtractGlobalCenterTrajectory", emit_abs_xy=True),
    dict(type="Hand21WorldNormalize"),
    dict(type="FormatGlobalTrajectory", num_traj_channels=num_traj_channels),
    dict(type="FormatGCNInput3D", num_person=1, include_score=True),
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
