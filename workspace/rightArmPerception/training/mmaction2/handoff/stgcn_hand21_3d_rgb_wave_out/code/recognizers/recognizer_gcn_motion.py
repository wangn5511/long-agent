# Copyright (c) OpenMMLab. All rights reserved.
"""GCN recognizer with whole-hand global-motion fusion."""

from typing import Optional, Tuple

import torch

from mmaction.models.recognizers.base import BaseRecognizer
from mmaction.registry import MODELS
from mmaction.utils import SampleList


def _collect_handedness_tensors(
    data_samples: Optional[SampleList],
    device: torch.device,
    num_clips: int,
) -> dict:
    if not data_samples:
        return {}

    labels = []
    scores = []
    for sample in data_samples:
        meta = getattr(sample, "metainfo", {}) or {}
        labels.append(int(meta.get("handedness_label_id", 2)))
        scores.append(float(meta.get("handedness_score", 0.0)))

    hand_label = torch.tensor(labels, dtype=torch.long, device=device).repeat_interleave(num_clips)
    hand_score = torch.tensor(scores, dtype=torch.float32, device=device).repeat_interleave(num_clips)
    return {"handedness": hand_label, "handedness_score": hand_score}


def _collect_global_traj_tensors(
    data_samples: Optional[SampleList],
    device: torch.device,
    num_clips: int,
) -> torch.Tensor:
    if not data_samples:
        raise ValueError("Global-motion recognizer requires data_samples with global_traj fields.")

    traj_list = []
    for sample in data_samples:
        global_traj = getattr(sample, "global_traj", None)
        if global_traj is None:
            raise ValueError("Missing `global_traj` in data sample. Check PackActionInputs algorithm_keys.")
        traj = torch.as_tensor(global_traj, dtype=torch.float32, device=device)
        if traj.ndim == 2:
            traj = traj.unsqueeze(0)
        if traj.ndim != 3:
            raise ValueError(
                f"Expected per-sample global_traj with shape (num_clips, T, C) or (T, C), got {tuple(traj.shape)}"
            )
        if traj.shape[0] != num_clips:
            raise ValueError(f"Expected {num_clips} trajectory clips, got {traj.shape[0]}")
        traj_list.append(traj)

    return torch.stack(traj_list, dim=0)


@MODELS.register_module()
class RecognizerGCNGlobalMotionHandedness(BaseRecognizer):
    """GCN recognizer that fuses ST-GCN and whole-hand center trajectory."""

    def __init__(
        self,
        backbone,
        motion_branch,
        cls_head=None,
        neck=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
    ) -> None:
        super().__init__(
            backbone=backbone,
            cls_head=cls_head,
            neck=neck,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
        )
        self.motion_branch = MODELS.build(motion_branch)

    def extract_feat(
        self,
        inputs: torch.Tensor,
        stage: str = "backbone",
        data_samples: Optional[SampleList] = None,
        **kwargs,
    ) -> Tuple:
        loss_predict_kwargs = dict()

        bs, nc = inputs.shape[:2]
        skel_inputs = inputs.reshape((bs * nc,) + inputs.shape[2:])
        skel_feat = self.backbone(skel_inputs)

        global_traj = _collect_global_traj_tensors(data_samples, skel_feat.device, nc)
        motion_feat = self.motion_branch(global_traj.reshape(bs * nc, *global_traj.shape[2:]))

        if nc > 1:
            _, c_s, t_s, v_s = skel_feat.shape
            skel_feat = skel_feat.view(bs, nc, c_s, t_s, v_s).mean(dim=1)
            motion_feat = motion_feat.view(bs, nc, -1).mean(dim=1)

        loss_predict_kwargs.update(_collect_handedness_tensors(data_samples, skel_feat.device, 1))
        feats = (skel_feat, motion_feat)

        if stage == "backbone":
            return feats, loss_predict_kwargs
        if self.with_cls_head and stage == "head":
            feats = self.cls_head(feats, **loss_predict_kwargs)
            return feats, loss_predict_kwargs
        return feats, loss_predict_kwargs

    def predict(self, inputs: torch.Tensor, data_samples: SampleList, **kwargs):
        feats, predict_kwargs = self.extract_feat(inputs, data_samples=data_samples, test_mode=True)
        return self.cls_head.predict(feats, data_samples, **predict_kwargs)
