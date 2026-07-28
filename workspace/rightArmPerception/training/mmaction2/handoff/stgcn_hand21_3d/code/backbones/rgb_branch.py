# Copyright (c) OpenMMLab. All rights reserved.
"""Lightweight RGB branch for skeleton-RGB fusion gesture recognition.

Architecture (per the design spec in rgb_stgcn.txt):
  (N, T, 3, 96, 96)
    -> reshape (N*T, 3, 96, 96)
    -> MobileNetV3-Small features[0:6] [pretrained, frozen by default]
    -> AdaptiveAvgPool2d(1) -> (N*T, 40)
    -> reshape (N, T, 40)
    -> Linear(40->96) + BN + ReLU                [projection]
    -> 3x depthwise-sep Conv1d, dilation=[1,2,4] [TCN]
    -> temporal mean pooling
    -> rgb_feat: (N, 96)
"""

from typing import Optional

import torch
import torch.nn as nn
from mmaction.registry import MODELS
from mmengine.model import BaseModule


class _DepthwiseSepConv1dBlock(nn.Module):
    """Depthwise-separable Conv1d block with residual connection."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.dw = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=pad,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pw = nn.Conv1d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pw(self.dw(x))
        x = self.act(self.bn(x))
        return x + residual


@MODELS.register_module()
class RGBBranch(BaseModule):
    """Lightweight RGB branch for hand-gesture recognition."""

    _CNN_OUT_CHANNELS: int = 40

    def __init__(
        self,
        proj_channels: int = 96,
        freeze_cnn: bool = True,
        pretrained: bool = True,
        init_cfg: Optional[dict | list[dict]] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        import torchvision.models as tvm

        if pretrained:
            try:
                weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                mobilenet = tvm.mobilenet_v3_small(weights=weights)
            except (AttributeError, TypeError):
                mobilenet = tvm.mobilenet_v3_small(pretrained=True)
        else:
            try:
                mobilenet = tvm.mobilenet_v3_small(weights=None)
            except TypeError:
                mobilenet = tvm.mobilenet_v3_small(pretrained=False)

        self.cnn = nn.Sequential(*list(mobilenet.features.children())[:6])

        self.freeze_cnn = freeze_cnn
        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(self._CNN_OUT_CHANNELS, proj_channels, bias=False),
            nn.BatchNorm1d(proj_channels),
            nn.ReLU(inplace=True),
        )
        self.tcn = nn.Sequential(
            _DepthwiseSepConv1dBlock(proj_channels, dilation=1),
            _DepthwiseSepConv1dBlock(proj_channels, dilation=2),
            _DepthwiseSepConv1dBlock(proj_channels, dilation=4),
        )

    def train(self, mode: bool = True) -> 'RGBBranch':
        super().train(mode)
        if self.freeze_cnn:
            self.cnn.eval()
        return self

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        n, t, c, h, w = imgs.shape

        x = imgs.view(n * t, c, h, w)
        x = self.cnn(x)
        x = self.gap(x).view(n * t, -1)

        x = self.proj(x)
        x = x.view(n, t, -1)

        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)

        return x.mean(dim=1)
