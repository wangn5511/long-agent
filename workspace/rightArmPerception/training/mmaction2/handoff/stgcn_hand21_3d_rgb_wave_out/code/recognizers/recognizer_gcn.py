# Copyright (c) OpenMMLab. All rights reserved.
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


@MODELS.register_module()
class RecognizerGCNHandedness(BaseRecognizer):
    """GCN recognizer that forwards handedness metadata to the head."""

    def extract_feat(
        self,
        inputs: torch.Tensor,
        stage: str = "backbone",
        data_samples: Optional[SampleList] = None,
        **kwargs,
    ) -> Tuple:
        loss_predict_kwargs = dict()

        bs, nc = inputs.shape[:2]
        inputs = inputs.reshape((bs * nc,) + inputs.shape[2:])
        x = self.backbone(inputs)
        loss_predict_kwargs.update(_collect_handedness_tensors(data_samples, x.device, nc))

        if stage == "backbone":
            return x, loss_predict_kwargs
        if self.with_cls_head and stage == "head":
            x = self.cls_head(x, **loss_predict_kwargs)
            return x, loss_predict_kwargs
        return x, loss_predict_kwargs

    def predict(self, inputs: torch.Tensor, data_samples: SampleList, **kwargs):
        feats, predict_kwargs = self.extract_feat(inputs, data_samples=data_samples, test_mode=True)
        predictions = self.cls_head.predict(feats, data_samples, **predict_kwargs)
        return predictions
