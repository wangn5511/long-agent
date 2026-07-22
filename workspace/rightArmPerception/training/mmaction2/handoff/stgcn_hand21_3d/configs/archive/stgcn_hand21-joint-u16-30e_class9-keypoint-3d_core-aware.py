_base_ = "./stgcn_hand21-joint-u16-30e_class9-keypoint-3d.py"

meta_keys = (
    "img_shape",
    "handedness_label",
    "handedness_label_id",
    "handedness_score",
    "track_id",
    "video_name",
)

train_pipeline = [
    dict(
        type="CoreAwareSampleFrames",
        clip_len=16,
        continuous_ratio=0.6,
        local_ratio=0.3,
        full_clip_ratio=0.1,
        local_min_coverage=20,
        local_max_coverage=30,
        local_max_coverage_ratio=0.8,
        continuous_start_min_ratio=0.2,
        core_classes=(2,),
        peak_smooth_window=3,
        onset_threshold=0.4,
        abs_energy_floor=0.003,
        pre_context=1,
        jitter_range=2,
    ),
    dict(type="PoseDecode"),
    dict(type="Hand21WorldNormalize"),
    dict(type="FormatGCNInput3D", num_person=1, include_score=False),
    dict(type="PackActionInputs", meta_keys=meta_keys),
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
    dict(type="Hand21WorldNormalize"),
    dict(type="FormatGCNInput3D", num_person=1, include_score=False),
    dict(type="PackActionInputs", meta_keys=meta_keys),
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
    dict(type="Hand21WorldNormalize"),
    dict(type="FormatGCNInput3D", num_person=1, include_score=False),
    dict(type="PackActionInputs", meta_keys=meta_keys),
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
