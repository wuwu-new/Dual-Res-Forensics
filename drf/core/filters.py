"""
高频残差提取：SRM (Spatial Rich Model) 三组固定噪声残差核
==========================================================
相对旧版的单个 Laplacian：
  - 这里使用 SRM 论文中最常用的 3 个核 (5x5 KV, 5x5 SQUARE, 3x3 EDGE)
  - 对 RGB 三个通道分别做组卷积，输出 3*3 = 9 通道残差图
  - 仍是"固定核 + 不参与训练"，但残差信号更丰富，便于下游 DW 编码器学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _srm_kernels():
    """构造 SRM 论文里 3 个经典残差核，全部 5x5（小核用 0 填充对齐）。"""
    k1 = torch.tensor([
        [-1,  2, -2,  2, -1],
        [ 2, -6,  8, -6,  2],
        [-2,  8, -12, 8, -2],
        [ 2, -6,  8, -6,  2],
        [-1,  2, -2,  2, -1],
    ], dtype=torch.float32) / 12.0

    k2 = torch.tensor([
        [ 0,  0,  0,  0,  0],
        [ 0, -1,  2, -1,  0],
        [ 0,  2, -4,  2,  0],
        [ 0, -1,  2, -1,  0],
        [ 0,  0,  0,  0,  0],
    ], dtype=torch.float32) / 4.0

    k3 = torch.tensor([
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  1, -2,  1,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
    ], dtype=torch.float32) / 2.0

    return torch.stack([k1, k2, k3], dim=0)  # (3, 5, 5)


class SRMResidual(nn.Module):
    """
    输入: (B, 3, H, W) 归一化后的 RGB
    输出: (B, 9, H, W) 高频残差 (3 个 SRM 核 × 3 个 RGB 通道, group=3)

    关键点：
      - register_buffer 固化卷积核，不参与训练 (与旧 HighPassFilter 思路一致)
      - 通过 groups=3 让每个 RGB 通道独立卷积，再 concat —— 比 Laplacian 多 3 倍信号
    """

    def __init__(self):
        super().__init__()
        srm = _srm_kernels()                       # (3, 5, 5)
        # 想要 conv2d(weight=(C_out, C_in/groups, kH, kW)), groups=3
        # 每个输入通道对应 3 个输出 (3 个核) → C_out = 9, in/groups = 1
        weight = srm.unsqueeze(1).repeat(3, 1, 1, 1)   # (9, 1, 5, 5)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, bias=None, stride=1, padding=2, groups=3)

    @property
    def out_channels(self) -> int:
        return 9
