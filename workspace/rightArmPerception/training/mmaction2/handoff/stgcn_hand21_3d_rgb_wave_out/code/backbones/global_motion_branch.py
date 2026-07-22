# Copyright (c) OpenMMLab. All rights reserved.
"""Whole-hand global motion branch for 3D gesture recognition."""

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmaction.registry import MODELS


class _DepthwiseSepConv1dBlock(nn.Module):
    """Depthwise-separable temporal block with residual connection."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.dw = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pw(self.dw(x))
        x = self.act(self.bn(x))
        return x + residual


class _TemporalAttnPool(nn.Module):
    """Learnable temporal attention pooling."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.score(x).squeeze(-1)
        weights = torch.softmax(attn, dim=-1)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


@MODELS.register_module()
class GlobalMotionBranch(BaseModule):
    """Lightweight TCN branch for whole-hand palm-center velocity.

    Input:
        global_traj: (N, T, 2), where each frame is
        delta(normalized_palm_center_xy[t]).
    Output:
        motion_feat: (N, proj_channels)
    """

    def __init__(
        self,
        in_channels: int = 2,
        proj_channels: int = 32,
        init_cfg: Optional[Union[Dict, List[Dict]]] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.proj_channels = proj_channels

        self.proj = nn.Sequential(
            nn.Linear(in_channels, proj_channels, bias=False),
            nn.BatchNorm1d(proj_channels),
            nn.ReLU(inplace=True),
        )
        self.tcn = nn.Sequential(
            _DepthwiseSepConv1dBlock(proj_channels, dilation=1),
            _DepthwiseSepConv1dBlock(proj_channels, dilation=2),
            _DepthwiseSepConv1dBlock(proj_channels, dilation=4),
        )
        self.attn_pool = _TemporalAttnPool(proj_channels)

    def forward(self, global_traj: torch.Tensor) -> torch.Tensor:
        if global_traj.ndim != 3:
            raise ValueError(
                f"Expected global_traj with shape (N, T, C), got {tuple(global_traj.shape)}"
            )
        n, t, c = global_traj.shape
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} trajectory channels, got {c}")

        x = global_traj.reshape(n * t, c)
        x = self.proj(x)
        x = x.view(n, t, self.proj_channels).permute(0, 2, 1).contiguous()
        x = self.tcn(x)
        x = x.permute(0, 2, 1).contiguous()
        return self.attn_pool(x)
