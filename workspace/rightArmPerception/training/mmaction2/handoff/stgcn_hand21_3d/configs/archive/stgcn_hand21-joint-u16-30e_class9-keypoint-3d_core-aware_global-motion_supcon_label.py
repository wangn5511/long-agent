_base_ = "./stgcn_hand21-joint-u16-30e_class9-keypoint-3d_core-aware_global-motion_supcon.py"

model = dict(
    cls_head=dict(
        use_handedness_score=False,
    ),
)
