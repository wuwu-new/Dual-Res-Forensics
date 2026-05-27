"""
深度可分离的残差编码器 (替换旧版纯 Conv→ReLU→AvgPool)
=========================================================
- 输入: SRM 高频残差图 (B, 9, H, W)
- 输出: token 序列 (B, N, C_token) —— 喂给后续 cross-attention
- 风格: MobileNet-V2 风格 DW + PW + 残差，比朴素 CNN 更轻、表达更强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWSepBlock(nn.Module):
    """深度可分离卷积 + 1x1 PW + 残差 (in==out 时短路, 否则用 1x1 投影)。"""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expansion: int = 2):
        super().__init__()
        hidden = in_ch * expansion
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3,
                      stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.use_short = (stride == 1 and in_ch == out_ch)
        if not self.use_short:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        y = self.pw(self.dw(self.expand(x)))
        s = x if self.use_short else self.proj(x)
        return F.gelu(y + s)


class DWResidualEncoder(nn.Module):
    """
    SRM 残差 (B, in_channels, H, W) → token (B, N, C)
    阶段:
      stem (in_channels→32, /2)
        → DW-blk1 (32→64, /2)
        → DW-blk2 (64→C_token, /2)
        → DW-blk3 (C_token→C_token)
        → 自适应到 token_grid × token_grid 网格 → 展平为 N=grid^2 个 token

    in_channels 默认 9 (对应 SRM 输出)；消融实验时可设 3 (直接喂 RGB)。
    """

    def __init__(self, c_token: int = 192, token_grid: int = 8, in_channels: int = 9):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.blk1 = DWSepBlock(32, 64, stride=2)
        self.blk2 = DWSepBlock(64, c_token, stride=2)
        self.blk3 = DWSepBlock(c_token, c_token, stride=1)
        self.token_grid = token_grid
        self.pool = nn.AdaptiveAvgPool2d((token_grid, token_grid))

        self.c_token = c_token
        self.num_tokens = token_grid * token_grid

    def forward(self, residual: torch.Tensor):
        """returns (tokens, feature_map):
            tokens : (B, N, C)
            fmap   : (B, C, token_grid, token_grid)  —— 留给可选的边界解码器
        """
        x = self.stem(residual)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.pool(x)
        tokens = x.flatten(2).transpose(1, 2).contiguous()  # (B, N, C)
        return tokens, x
