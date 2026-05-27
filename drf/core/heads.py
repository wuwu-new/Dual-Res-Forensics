"""
头部模块
==========
- BinaryClassifierHead: 单 logit (BCE) —— 旧版用 2-class CE
- BoundaryDecoder     : 轻量 up-sample 解码器, 输出 (B,1,H_out,W_out) 的边界图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryClassifierHead(nn.Module):
    """fused_feat (B, D) → 单 logit (B, 1)。BCEWithLogits + label smoothing 配合使用。"""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)        # (B,)


class BoundaryDecoder(nn.Module):
    """
    边界图预测头 (可选).
    输入 : 残差特征图 (B, C, h, w) (来自 DWResidualEncoder 的 fmap)
    输出 : (B, 1, h*4, w*4) 上采样到更大尺寸的边界图
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 2),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(in_ch // 2, in_ch // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 4),
            nn.GELU(),
        )
        self.out = nn.Conv2d(in_ch // 4, 1, kernel_size=1)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(fmap, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x)
        return self.out(x)
