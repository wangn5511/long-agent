# Copyright (c) OpenMMLab. All rights reserved.
"""GCN head with global-motion and handedness fusion."""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.evaluation import top_k_accuracy
from mmaction.models.heads.base import BaseHead
from mmaction.registry import MODELS


class SupConLoss(nn.Module):
    """Supervised contrastive loss for single-view embeddings."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape {tuple(features.shape)}")
        if labels.ndim != 1:
            labels = labels.view(-1)
        if features.size(0) != labels.size(0):
            raise ValueError(
                "Features and labels must share the same batch dimension: "
                f"{features.size(0)} vs {labels.size(0)}"
            )

        batch_size = features.size(0)
        if batch_size < 2:
            # Keep the zero loss connected to the current graph so every DDP
            # rank touches the projection head parameters consistently.
            return features.sum() * 0.0

        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        labels = labels.contiguous().view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).to(device=features.device, dtype=features.dtype)
        logits_mask = torch.ones_like(positive_mask) - torch.eye(
            batch_size, device=features.device, dtype=features.dtype
        )
        positive_mask = positive_mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positive_count = positive_mask.sum(dim=1)
        valid_mask = positive_count > 0
        if not torch.any(valid_mask):
            # Some local batches can contain only singleton labels. Returning a
            # graph-connected zero avoids DDP unused-parameter divergence.
            return features.sum() * 0.0

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_count.clamp_min(1.0)
        return -mean_log_prob_pos[valid_mask].mean()


@MODELS.register_module()
class GCNHead3DGlobalMotionHandedness(BaseHead):
    """Fuse ST-GCN, global-motion, and handedness features before classification."""

    def __init__(
        self,
        num_classes: int,
        skel_channels: int,
        motion_channels: int = 32,
        hidden_channels: int = 256,
        loss_cls: Dict = dict(type="CrossEntropyLoss"),
        enable_contrast_loss: bool = False,
        contrast_proj_dim: int = 128,
        contrast_temperature: float = 0.07,
        contrast_loss_weight: float = 0.1,
        dropout: float = 0.5,
        average_clips: str = "prob",
        handedness_embed_dim: int = 8,
        use_handedness_score: bool = True,
        num_handedness_classes: int = 3,
        init_cfg: Union[Dict, List[Dict]] = dict(type="Normal", layer="Linear", std=0.01),
        **kwargs,
    ) -> None:
        legacy_in_channels = kwargs.pop("in_channels", None)
        if legacy_in_channels is not None and int(legacy_in_channels) != int(skel_channels):
            raise ValueError(
                f"Legacy in_channels={legacy_in_channels} does not match skel_channels={skel_channels}"
            )
        super().__init__(
            num_classes,
            in_channels=skel_channels + motion_channels + handedness_embed_dim,
            loss_cls=loss_cls,
            average_clips=average_clips,
            init_cfg=init_cfg,
            **kwargs,
        )
        self.skel_channels = skel_channels
        self.motion_channels = motion_channels
        self.hidden_channels = hidden_channels
        self.enable_contrast_loss = bool(enable_contrast_loss)
        self.contrast_loss_weight = float(contrast_loss_weight)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.hand_embedding = nn.Embedding(num_handedness_classes, handedness_embed_dim)
        self.use_handedness_score = use_handedness_score
        self.score_proj = nn.Linear(1, handedness_embed_dim) if use_handedness_score else None

        fusion_dim = skel_channels + motion_channels + handedness_embed_dim
        self.fc1 = nn.Linear(fusion_dim, hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_channels, num_classes)
        self.proj_head = None
        self.loss_contrast = None
        if self.enable_contrast_loss:
            self.proj_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, contrast_proj_dim),
            )
            self.loss_contrast = SupConLoss(temperature=contrast_temperature)

    def _compute_hidden(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
        handedness: Optional[torch.Tensor] = None,
        handedness_score: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        skel_feat, motion_feat = x
        n, c, _, _ = skel_feat.shape
        skel_feat = self.pool(skel_feat).view(n, c)
        if c != self.skel_channels:
            raise AssertionError(f"Expected {self.skel_channels} skeleton channels, got {c}")

        if motion_feat.ndim != 2 or motion_feat.shape[0] != n:
            raise ValueError(
                f"Expected motion_feat with shape (N, C), got {tuple(motion_feat.shape)}"
            )
        if motion_feat.shape[1] != self.motion_channels:
            raise AssertionError(
                f"Expected {self.motion_channels} motion channels, got {motion_feat.shape[1]}"
            )

        if handedness is None:
            handedness = torch.full((n,), 2, dtype=torch.long, device=skel_feat.device)
        else:
            handedness = handedness.to(device=skel_feat.device, dtype=torch.long).view(-1)
            if handedness.numel() != n:
                raise ValueError(f"Expected {n} handedness labels, got {handedness.numel()}")
        hand_feat = self.hand_embedding(
            torch.clamp(handedness, min=0, max=self.hand_embedding.num_embeddings - 1)
        )

        if self.score_proj is not None and handedness_score is not None:
            handedness_score = handedness_score.to(
                device=skel_feat.device, dtype=torch.float32
            ).view(-1, 1)
            if handedness_score.shape[0] != n:
                raise ValueError(f"Expected {n} handedness scores, got {handedness_score.shape[0]}")
            hand_feat = hand_feat + self.score_proj(handedness_score)

        fused = torch.cat([skel_feat, motion_feat, hand_feat], dim=1)
        return self.relu(self.fc1(fused))

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
        handedness: Optional[torch.Tensor] = None,
        handedness_score: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden = self._compute_hidden(
            x,
            handedness=handedness,
            handedness_score=handedness_score,
        )
        return self.fc2(self.dropout(hidden))

    def loss(
        self,
        feats: Tuple[torch.Tensor, torch.Tensor],
        data_samples,
        **kwargs,
    ) -> Dict:
        hidden = self._compute_hidden(feats, **kwargs)
        cls_scores = self.fc2(self.dropout(hidden))

        labels = [x.gt_label for x in data_samples]
        labels = torch.stack(labels).to(cls_scores.device).squeeze()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)

        losses: Dict[str, torch.Tensor] = {}
        if cls_scores.size() != labels.size():
            top_k_acc = top_k_accuracy(
                cls_scores.detach().cpu().numpy(),
                labels.detach().cpu().numpy(),
                self.topk,
            )
            for k, acc in zip(self.topk, top_k_acc):
                losses[f"top{k}_acc"] = torch.tensor(acc, device=cls_scores.device)

        loss_cls = self.loss_cls(cls_scores, labels)
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses["loss_cls"] = loss_cls

        if self.enable_contrast_loss and self.proj_head is not None and self.loss_contrast is not None:
            z = F.normalize(self.proj_head(hidden), dim=1)
            losses["loss_contrast"] = self.contrast_loss_weight * self.loss_contrast(z, labels)

        return losses
