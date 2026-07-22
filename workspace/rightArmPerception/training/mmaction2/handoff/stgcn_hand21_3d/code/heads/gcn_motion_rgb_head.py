# Copyright (c) OpenMMLab. All rights reserved.
"""GCN head fusing skeleton, global-motion, RGB, and handedness features."""

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

        device = features.device
        batch_size = features.size(0)
        if batch_size < 2:
            # Keep the zero loss connected to the current graph so every DDP
            # rank touches the projection head parameters consistently.
            return features.sum() * 0.0

        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        labels = labels.contiguous().view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).to(device=device, dtype=features.dtype)
        logits_mask = torch.ones_like(positive_mask) - torch.eye(
            batch_size, device=device, dtype=features.dtype
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
        loss = -mean_log_prob_pos[valid_mask].mean()
        return loss


@MODELS.register_module()
class GCNHead3DGlobalMotionHandednessRGB(BaseHead):
    """Classification head fusing ST-GCN, global-motion, RGB, and handedness.

    Fusion: pool(skel_feat) ‖ motion_feat ‖ rgb_feat ‖ hand_embed
            → fc1 → ReLU → Dropout → fc2 → logits

    Args:
        num_classes (int): Number of output classes.
        skel_channels (int): Channel depth of STGCN4D output. Default: 256.
        motion_channels (int): Channel depth of GlobalMotionBranch. Default: 16.
        rgb_channels (int): Channel depth of RGBBranch. Default: 96.
        hidden_channels (int): Width of intermediate FC layer. Default: 256.
        enable_contrast_loss (bool): Whether to add supervised contrastive
            loss on top of the fused hidden embedding. Default: False.
        contrast_proj_dim (int): Output dim of the projection head used by
            supervised contrastive loss. Default: 128.
        contrast_temperature (float): Temperature used by supervised
            contrastive loss. Default: 0.07.
        contrast_loss_weight (float): Weight applied to the contrastive loss.
            Default: 0.1.
        handedness_embed_dim (int): Embedding dim for handedness. Default: 8.
        use_handedness_score (bool): Whether to add score-projected correction
            to the handedness embedding. Default: True.
        num_handedness_classes (int): Size of the handedness vocabulary
            (Left / Right / Unknown → 3). Default: 3.
        dropout (float): Dropout probability before the final linear. Default: 0.5.
    """

    def __init__(
        self,
        num_classes: int,
        skel_channels: int,
        motion_channels: int = 16,
        rgb_channels: int = 96,
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
        fusion_dim = skel_channels + motion_channels + rgb_channels + handedness_embed_dim
        super().__init__(
            num_classes,
            in_channels=fusion_dim,
            loss_cls=loss_cls,
            average_clips=average_clips,
            init_cfg=init_cfg,
            **kwargs,
        )
        self.skel_channels = skel_channels
        self.motion_channels = motion_channels
        self.rgb_channels = rgb_channels
        self.hidden_channels = hidden_channels
        self.enable_contrast_loss = bool(enable_contrast_loss)
        self.contrast_loss_weight = float(contrast_loss_weight)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.hand_embedding = nn.Embedding(num_handedness_classes, handedness_embed_dim)
        self.use_handedness_score = use_handedness_score
        self.score_proj = nn.Linear(1, handedness_embed_dim) if use_handedness_score else None

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

    def _compute_fused_hidden(
        self,
        x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        handedness: Optional[torch.Tensor] = None,
        handedness_score: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        skel_feat, motion_feat, rgb_feat = x
        n, c, _, _ = skel_feat.shape
        skel_pooled = self.pool(skel_feat).view(n, c)
        if c != self.skel_channels:
            raise AssertionError(f"Expected {self.skel_channels} skeleton channels, got {c}")

        if motion_feat.shape != (n, self.motion_channels):
            raise AssertionError(
                f"Expected motion_feat ({n}, {self.motion_channels}), got {tuple(motion_feat.shape)}"
            )
        if rgb_feat.shape != (n, self.rgb_channels):
            raise AssertionError(
                f"Expected rgb_feat ({n}, {self.rgb_channels}), got {tuple(rgb_feat.shape)}"
            )

        if handedness is None:
            handedness = torch.full((n,), 2, dtype=torch.long, device=skel_pooled.device)
        else:
            handedness = handedness.to(device=skel_pooled.device, dtype=torch.long).view(-1)
        hand_feat = self.hand_embedding(
            torch.clamp(handedness, min=0, max=self.hand_embedding.num_embeddings - 1)
        )
        if self.score_proj is not None and handedness_score is not None:
            hs = handedness_score.to(device=skel_pooled.device, dtype=torch.float32).view(-1, 1)
            hand_feat = hand_feat + self.score_proj(hs)

        fused = torch.cat([skel_pooled, motion_feat, rgb_feat, hand_feat], dim=1)
        return self.relu(self.fc1(fused))

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        handedness: Optional[torch.Tensor] = None,
        handedness_score: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden = self._compute_fused_hidden(
            x,
            handedness=handedness,
            handedness_score=handedness_score,
        )
        return self.fc2(self.dropout(hidden))

    def loss(
        self,
        feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        data_samples,
        **kwargs,
    ) -> Dict:
        hidden = self._compute_fused_hidden(feats, **kwargs)
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


@MODELS.register_module()
class GCNHead3DGlobalMotionHandednessRGBGated(GCNHead3DGlobalMotionHandednessRGB):
    """Extends GCNHead3DGlobalMotionHandednessRGB with a binary gesture/NONE gate.

    Mirrors the design of ``GCNHead3DGlobalMotionHandednessGated`` but on the
    4-stream fused hidden (skel + motion + rgb + handedness).  A single linear
    ``gate_head`` branches off the shared hidden vector and outputs a sigmoid
    score.  At inference, samples whose gate score falls below
    ``gate_threshold`` are forced to predict the NONE class.  During training
    the gate is supervised with BCE loss (is_gesture = label != none_class_id).

    Args:
        none_class_id (int): The class index to force when the gate rejects a
            sample at inference.
        gate_loss_weight (float): Weight applied to the BCE gate loss.
            Default: 0.5.
        gate_threshold (float): Sigmoid threshold below which inference output
            is forced to NONE. Default: 0.5.
        use_gate_at_inference (bool): Whether to apply the hard NONE mask
            during ``forward``. Default: True.
    """

    def __init__(
        self,
        none_class_id: int,
        gate_loss_weight: float = 0.5,
        gate_threshold: float = 0.5,
        use_gate_at_inference: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.none_class_id = int(none_class_id)
        self.gate_loss_weight = float(gate_loss_weight)
        self.gate_threshold = float(gate_threshold)
        self.use_gate_at_inference = bool(use_gate_at_inference)
        self.gate_head = nn.Linear(self.hidden_channels, 1)

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        handedness: Optional[torch.Tensor] = None,
        handedness_score: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden = self._compute_fused_hidden(
            x,
            handedness=handedness,
            handedness_score=handedness_score,
        )
        cls_score = self.fc2(self.dropout(hidden))

        if self.use_gate_at_inference:
            is_gesture = torch.sigmoid(self.gate_head(hidden)).squeeze(-1) >= self.gate_threshold
            if not is_gesture.all():
                forced_none = torch.full_like(cls_score, -1e9)
                forced_none[:, self.none_class_id] = 1e9
                cls_score = torch.where(is_gesture.unsqueeze(1), cls_score, forced_none)

        return cls_score

    def loss(
        self,
        feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        data_samples,
        **kwargs,
    ) -> Dict:
        hidden = self._compute_fused_hidden(feats, **kwargs)
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

        gate_logits = self.gate_head(hidden)  # (N, 1)
        is_gesture = (labels != self.none_class_id).float().unsqueeze(-1)  # (N, 1)
        losses["loss_gate"] = self.gate_loss_weight * F.binary_cross_entropy_with_logits(
            gate_logits, is_gesture
        )

        return losses
