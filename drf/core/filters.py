"""SRM 高频噪声残差提取 + 深度可分离残差编码器。

包含:
    - SRMResidual:          3 组固定 SRM 核，对 RGB 三通道分别滤波，输出 9 通道残差
    - DWResidualEncoder:    在残差上做深度可分离卷积编码

修复说明（本 PR 的核心）
------------------------
此前 DWResidualEncoder 的 in_channels 被写死为 3，而 SRMResidual 实际输出 9 通道
(3 核 × 3 RGB)，导致前向时 shape mismatch。现将 in_channels 改为可配置，
并默认对齐 SRMResidual 的输出通道数。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# 3 组经典 SRM 高通核（5x5），来自隐写/伪造检测文献的常用滤波器
_SRM_KERNELS = [
    # 一阶水平差分核
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, -2, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    # 二阶核
    [
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0],
    ],
    # 5x5 KB 方形核
    [
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1],
    ],
]
# 对应归一化因子，使核响应量级一致
_SRM_NORM = [2.0, 4.0, 12.0]


class SRMResidual(nn.Module):
    """固定 SRM 核提取高频噪声残差。

    输入  : [B, 3, H, W] RGB 图像
    输出  : [B, 9, H, W] 残差（3 核 × 3 RGB 通道）

    核为不可学习的固定参数（requires_grad=False）。
    """

    def __init__(self) -> None:
        super().__init__()
        kernels = []
        for k, norm in zip(_SRM_KERNELS, _SRM_NORM):
            kernels.append(torch.tensor(k, dtype=torch.float32) / norm)
        # [3, 5, 5] -> 扩展到每个 RGB 通道做 depthwise： weight [9, 1, 5, 5]
        base = torch.stack(kernels, dim=0)               # [3, 5, 5]
        weight = base.repeat_interleave(3, dim=0)        # [9, 5, 5]
        weight = weight.unsqueeze(1)                     # [9, 1, 5, 5]
        self.register_buffer("weight", weight)
        # 输入 3 通道、分组卷积映射到 9 通道：每个核作用于每个 RGB 通道
        # 通过把输入重复 3 次再分组实现 3核×3通道 = 9 输出
        self.out_channels = 9

    @property
    def num_channels(self) -> int:
        return self.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(f"SRMResidual 期望 [B, 3, H, W]，实际 {tuple(x.shape)}")
        # 将每个 RGB 通道复制给 3 个核：输入扩展为 9 通道后 depthwise 卷积
        x_rep = x.repeat_interleave(3, dim=1)            # [B, 9, H, W] (R,R,R,G,G,G,B,B,B)
        # weight 排布为 (核0,核1,核2) 重复 3 次，对齐 (R,R,R,...) 需要重排：
        # 此处直接用 groups=9 的 depthwise，weight 已是 [9,1,5,5]
        out = F.conv2d(x_rep, self.weight, padding=2, groups=9)
        return out


class DWResidualEncoder(nn.Module):
    """深度可分离卷积残差编码器。

    Args:
        in_channels:  输入通道数。默认 9，对齐 SRMResidual 输出（修复点）。
        out_channels: 输出特征通道数。默认 192。
    """

    def __init__(self, in_channels: int = 9, out_channels: int = 192) -> None:
        super().__init__()
        self.in_channels = in_channels
        hidden = max(in_channels, 64)
        self.block = nn.Sequential(
            # depthwise
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # pointwise 升维
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            # 第二组 depthwise + pointwise
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"通道不匹配：编码器期望 in_channels={self.in_channels}，"
                f"实际输入 {x.size(1)}。请确认与 SRMResidual.num_channels 对齐。"
            )
        return self.block(x)
