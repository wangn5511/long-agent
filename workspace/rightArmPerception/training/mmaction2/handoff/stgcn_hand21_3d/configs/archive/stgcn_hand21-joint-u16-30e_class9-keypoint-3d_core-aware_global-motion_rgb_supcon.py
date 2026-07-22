_base_ = "./stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion_rgb.py"

model = dict(
    cls_head=dict(
        type="GCNHead3DGlobalMotionHandednessRGB",
        enable_contrast_loss=True,
        contrast_proj_dim=128,
        contrast_temperature=0.07,
        contrast_loss_weight=0.1,
    ),
)
