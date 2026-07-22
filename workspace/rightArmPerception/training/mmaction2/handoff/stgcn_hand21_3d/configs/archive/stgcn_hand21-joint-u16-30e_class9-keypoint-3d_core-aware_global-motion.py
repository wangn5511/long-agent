_base_ = "./stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware.py"

meta_keys = (
    "img_shape",
    "handedness_label",
    "handedness_label_id",
    "handedness_score",
    "track_id",
    "video_name",
)

model = dict(
    type="RecognizerGCNGlobalMotionHandedness",
    motion_branch=dict(
        type="GlobalMotionBranch",
        in_channels=3,
        proj_channels=16,
    ),
    cls_head=dict(
        type="GCNHead3DGlobalMotionHandedness",
        num_classes=9,
        skel_channels=256,
        motion_channels=16,
        hidden_channels=256,
        handedness_embed_dim=8,
        use_handedness_score=True,
        dropout=0.6,
    ),
)

train_pipeline = [
    dict(
        type="CoreAwareSampleFrames",
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
    ),
    dict(type="PoseDecode"),
    dict(type="ExtractGlobalCenterTrajectory"),
    dict(type="Hand21WorldNormalize"),
    dict(type="FormatGlobalTrajectory"),
    dict(type="FormatGCNInput3D", num_person=1, include_score=False),
    dict(type="PackActionInputs", meta_keys=meta_keys, algorithm_keys=("global_traj",)),
]

val_pipeline = [
    dict(
        type="ClickDeterministicOnsetSampleFrames",
        clip_len=16,
        num_clips=1,
        crop_mode="center",
        core_classes=(2,),
        peak_smooth_window=3,
        onset_threshold=0.4,
        abs_energy_floor=0.003,
        pre_context=1,
    ),
    dict(type="PoseDecode"),
    dict(type="ExtractGlobalCenterTrajectory"),
    dict(type="Hand21WorldNormalize"),
    dict(type="FormatGlobalTrajectory"),
    dict(type="FormatGCNInput3D", num_person=1, include_score=False),
    dict(type="PackActionInputs", meta_keys=meta_keys, algorithm_keys=("global_traj",)),
]

test_pipeline = [
    dict(
        type="ClickDeterministicOnsetSampleFrames",
        clip_len=16,
        num_clips=1,
        crop_mode="center",
        core_classes=(2,),
        peak_smooth_window=3,
        onset_threshold=0.4,
        abs_energy_floor=0.003,
        pre_context=1,
    ),
    dict(type="PoseDecode"),
    dict(type="ExtractGlobalCenterTrajectory"),
    dict(type="Hand21WorldNormalize"),
    dict(type="FormatGlobalTrajectory"),
    dict(type="FormatGCNInput3D", num_person=1, include_score=False),
    dict(type="PackActionInputs", meta_keys=meta_keys, algorithm_keys=("global_traj",)),
]

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            pipeline=train_pipeline,
        )
    )
)

val_dataloader = dict(
    dataset=dict(
        pipeline=val_pipeline,
    )
)

test_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
    )
)

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=100, val_begin=1, val_interval=1)
