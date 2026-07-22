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
        type="ClickAwareMixedTSNSampleFrames",
        clip_len=16,
        continuous_ratio=0.6,
        local_ratio=0.3,
        full_clip_ratio=0.1,
        local_min_coverage=20,
        local_max_coverage=30,
        local_max_coverage_ratio=0.8,
        continuous_start_min_ratio=0.2,
        force_continuous_labels=(2,),
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
        ),
    ),
)
