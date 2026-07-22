_base_ = "./stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion_track-aware.py"

model = dict(
    cls_head=dict(
        type="GCNHead3DGlobalMotionHandedness",
        enable_contrast_loss=True,
        contrast_proj_dim=128,
        contrast_temperature=0.1,
        contrast_loss_weight=0.02,
        use_handedness_score=False,
    ),
)
