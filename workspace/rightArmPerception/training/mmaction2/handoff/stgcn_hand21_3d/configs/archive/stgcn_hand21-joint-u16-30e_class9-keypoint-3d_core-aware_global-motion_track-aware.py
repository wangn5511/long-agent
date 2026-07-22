_base_ = "./stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion.py"

ann_file = "/home/results/6t1/gesture_class9/rtm_class9_hand21_3d_windowtrack.pkl"

meta_keys = (
    "img_shape",
    "handedness_label",
    "handedness_label_id",
    "handedness_score",
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
    dict(type="SelectLocalPrimaryTrack", allow_fallback=True, auto_select_if_missing=True, valid_score_threshold=0.0),
    dict(type="PoseDecode"),
    dict(type="ExtractGlobalCenterTrajectory"),
    dict(type="Hand21WorldNormalize"),
    dict(type="FormatGlobalTrajectory"),
    dict(type="FormatGCNInput3D", num_person=1, include_score=False),
    dict(type="PackActionInputs", meta_keys=meta_keys, algorithm_keys=("global_traj",)),
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
    dict(type="SelectLocalPrimaryTrack", allow_fallback=True, auto_select_if_missing=True, valid_score_threshold=0.0),
    dict(type="PoseDecode"),
    dict(type="ExtractGlobalCenterTrajectory"),
    dict(type="Hand21WorldNormalize"),
    dict(type="FormatGlobalTrajectory"),
    dict(type="FormatGCNInput3D", num_person=1, include_score=False),
    dict(type="PackActionInputs", meta_keys=meta_keys, algorithm_keys=("global_traj",)),
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
    dict(type="SelectLocalPrimaryTrack", allow_fallback=True, auto_select_if_missing=True, valid_score_threshold=0.0),
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
            ann_file=ann_file,
            pipeline=train_pipeline,
        )
    )
)

val_dataloader = dict(
    dataset=dict(
        ann_file=ann_file,
        pipeline=val_pipeline,
    )
)

test_dataloader = dict(
    dataset=dict(
        ann_file=ann_file,
        pipeline=test_pipeline,
    )
)
