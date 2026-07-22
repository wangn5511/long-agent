# Copyright (c) OpenMMLab. All rights reserved.
"""GCN recognizer with skeleton, global-motion, RGB, and handedness fusion."""

from typing import Optional, Tuple

import torch

from mmaction.models.recognizers.base import BaseRecognizer
from mmaction.registry import MODELS
from mmaction.utils import SampleList

from .recognizer_gcn_motion import (
    _collect_global_traj_tensors,
    _collect_handedness_tensors,
)


@MODELS.register_module()
class RecognizerGCNGlobalMotionHandednessRGB(BaseRecognizer):
    """GCN recognizer fusing ST-GCN skeleton, global-motion trajectory,
    RGB hand-ROI frames, and handedness metadata.

    Args:
        backbone (dict): Config for STGCN4D skeleton backbone.
        motion_branch (dict): Config for GlobalMotionBranch.
        rgb_branch (dict): Config for RGBBranch (expects ``imgs`` key).
        cls_head (dict): Config for the fusion classification head.
    """

    def __init__(
        self,
        backbone,
        motion_branch,
        rgb_branch,
        cls_head=None,
        neck=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
    ) -> None:
        if data_preprocessor is None:
            data_preprocessor = dict(type="BundleGCNRGBDataPreprocessor")
        super().__init__(
            backbone=backbone,
            cls_head=cls_head,
            neck=neck,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
        )
        self.motion_branch = MODELS.build(motion_branch)
        self.rgb_branch = MODELS.build(rgb_branch)

    def extract_feat(
        self,
        inputs,
        stage: str = "backbone",
        data_samples: Optional[SampleList] = None,
        **kwargs,
    ) -> Tuple:
        loss_predict_kwargs = dict()

        skel = inputs["keypoint"]  # (N, nc, T, V, C)
        imgs = inputs["imgs"]      # (N, T, 3, H, W), nc=1 assumed for RGB

        bs, nc = skel.shape[:2]
        skel_inputs = skel.reshape((bs * nc,) + skel.shape[2:])
        skel_feat = self.backbone(skel_inputs)  # (N*nc, 256, T', V)

        global_traj = _collect_global_traj_tensors(data_samples, skel_feat.device, nc)
        motion_feat = self.motion_branch(
            global_traj.reshape(bs * nc, *global_traj.shape[2:])
        )  # (N*nc, motion_channels)

        rgb_feat = self.rgb_branch(imgs)  # (N, rgb_channels)

        if nc > 1:
            _, c_s, t_s, v_s = skel_feat.shape
            skel_feat = skel_feat.view(bs, nc, c_s, t_s, v_s).mean(dim=1)
            motion_feat = motion_feat.view(bs, nc, -1).mean(dim=1)

        loss_predict_kwargs.update(
            _collect_handedness_tensors(data_samples, skel_feat.device, 1)
        )
        feats = (skel_feat, motion_feat, rgb_feat)

        if stage == "backbone":
            return feats, loss_predict_kwargs
        if self.with_cls_head and stage == "head":
            feats = self.cls_head(feats, **loss_predict_kwargs)
            return feats, loss_predict_kwargs
        return feats, loss_predict_kwargs

    def loss(self, inputs, data_samples: SampleList, **kwargs) -> dict:
        feats, loss_kwargs = self.extract_feat(inputs, data_samples=data_samples)
        return self.cls_head.loss(feats, data_samples, **loss_kwargs)

    def predict(self, inputs, data_samples: SampleList, **kwargs) -> SampleList:
        feats, predict_kwargs = self.extract_feat(
            inputs, data_samples=data_samples, test_mode=True
        )
        return self.cls_head.predict(feats, data_samples, **predict_kwargs)

    def _forward(self, inputs, stage: str = "backbone", **kwargs):
        feats, _ = self.extract_feat(inputs, stage=stage)
        return feats

    def forward(self, inputs, data_samples=None, mode: str = "tensor", **kwargs):
        if mode == "tensor":
            return self._forward(inputs, **kwargs)
        if mode == "predict":
            return self.predict(inputs, data_samples, **kwargs)
        if mode == "loss":
            return self.loss(inputs, data_samples, **kwargs)
        raise RuntimeError(f'Invalid mode "{mode}".')
