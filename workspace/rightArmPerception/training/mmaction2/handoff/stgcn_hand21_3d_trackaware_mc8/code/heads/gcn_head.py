# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from mmaction.models.heads.base import BaseHead
from mmaction.registry import MODELS


@MODELS.register_module()
class GCNHead3DHandedness(BaseHead):
    """GCN head with handedness fusion before the final classifier."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        loss_cls: Dict = dict(type="CrossEntropyLoss"),
        dropout: float = 0.0,
        average_clips: str = "prob",
        handedness_embed_dim: int = 8,
        use_handedness_score: bool = True,
        num_handedness_classes: int = 3,
        init_cfg: Union[Dict, List[Dict]] = dict(type="Normal", layer="Linear", std=0.01),
        **kwargs,
    ) -> None:
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            average_clips=average_clips,
            init_cfg=init_cfg,
            **kwargs,
        )
        self.dropout_ratio = dropout
        self.dropout = nn.Dropout(p=self.dropout_ratio) if self.dropout_ratio != 0 else None
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.hand_embedding = nn.Embedding(num_handedness_classes, handedness_embed_dim)
        self.use_handedness_score = use_handedness_score
        self.score_proj = nn.Linear(1, handedness_embed_dim) if use_handedness_score else None
        self.fc = nn.Linear(self.in_channels + handedness_embed_dim, self.num_classes)

    def forward(
        self,
        x: torch.Tensor,
        handedness: Optional[torch.Tensor] = None,
        handedness_score: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        n, c, _, _ = x.shape
        x = self.pool(x).view(n, c)
        if x.shape[1] != self.in_channels:
            raise AssertionError(f"Expected {self.in_channels} channels, got {x.shape[1]}")

        if handedness is None:
            handedness = torch.full((n,), 2, dtype=torch.long, device=x.device)
        else:
            handedness = handedness.to(device=x.device, dtype=torch.long).view(-1)
            if handedness.numel() != n:
                raise ValueError(f"Expected {n} handedness labels, got {handedness.numel()}")
        hand_feat = self.hand_embedding(torch.clamp(handedness, min=0, max=self.hand_embedding.num_embeddings - 1))

        if self.score_proj is not None and handedness_score is not None:
            handedness_score = handedness_score.to(device=x.device, dtype=torch.float32).view(-1, 1)
            if handedness_score.shape[0] != n:
                raise ValueError(f"Expected {n} handedness scores, got {handedness_score.shape[0]}")
            hand_feat = hand_feat + self.score_proj(handedness_score)

        x = torch.cat([x, hand_feat], dim=1)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc(x)
