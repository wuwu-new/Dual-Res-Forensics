"""
DCT 频域分支 (Frequency Branch)
================================
动机
----
SRM 是空间域高通滤波, 抓的是像素邻域的噪声残差;
但深度伪造在 **频域** (尤其是 8x8 块的 DCT 中高频区) 留下的指纹
是 SRM 抓不到的: GAN/diffusion 频谱孔洞、JPEG 块伪影、
重压缩造成的能量异常等。F3-Net / SPSL / FreqNet 等 SOTA 方法
都依赖 DCT 频域特征做跨域泛化。

实现
----
- 输入 : RGB 图像  (B, 3, H, W),  H=W=224
- 分块 : 8x8 不重叠块,  得到 28x28 个块 (224/8)
- 变换 : 对每块做 2D DCT-II,  得到 (B, 3, 28, 28, 8, 8) 频率系数
- 归一 : 符号对称的 log1p 压缩动态范围 (DCT 直流分量数量级远大于交流分量)
- 重排 : (B, 192, 28, 28)  把 3 通道 × 64 频率拼成 channel 维
- 编码 : 轻量 DW-Sep CNN  →  (B, c_token, 8, 8)
- 输出 : token 序列 (B, 64, c_token)  和  fmap (B, c_token, 8, 8)
        与 DWResidualEncoder 输出形状对齐, 方便后续融合
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_dct_matrix(n: int = 8) -> torch.Tensor:
    """构造 n×n 的 DCT-II 正交基矩阵 M, 满足 DCT(x) = M @ x。"""
    k = torch.arange(n, dtype=torch.float64).view(-1, 1)        # (n,1) 频率索引
    i = torch.arange(n, dtype=torch.float64).view(1, -1)        # (1,n) 空间索引
    M = torch.cos(math.pi * (2 * i + 1) * k / (2 * n))
    M[0, :] *= 1.0 / math.sqrt(n)
    M[1:, :] *= math.sqrt(2.0 / n)
    return M.to(torch.float32)                                   # (n, n)


class _DWBlock(nn.Module):
    """与 residual_encoder 同款的 DW-Sep 块, 复制一份避免循环依赖。"""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expansion: int = 2):
        super().__init__()
        hidden = in_ch * expansion
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden), nn.GELU(),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.GELU(),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.use_short = (stride == 1 and in_ch == out_ch)
        if not self.use_short:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        y = self.pw(self.dw(self.expand(x)))
        s = x if self.use_short else self.proj(x)
        return F.gelu(y + s)


class DCTFrequencyBranch(nn.Module):
    """
    RGB → 8x8 块 DCT → log1p 归一 → 轻量 CNN → token 序列。

    Args:
        c_token   : 输出 token 通道数 (与 DWResidualEncoder 对齐, 默认 192)
        token_grid: 输出 token 网格边长 (默认 8 → 64 个 token)
        block_size: DCT 分块大小 (默认 8, 标准 JPEG 设置)
        drop_dc   : 是否把每块的直流分量 (低频) 置 0 (默认 True,
                    DC 主要编码亮度均值, 对伪造检测意义不大且数量级巨大)
    """

    def __init__(
        self,
        c_token: int = 192,
        token_grid: int = 8,
        block_size: int = 8,
        drop_dc: bool = True,
    ):
        super().__init__()
        self.bs = int(block_size)
        self.c_token = int(c_token)
        self.token_grid = int(token_grid)
        self.drop_dc = bool(drop_dc)

        # DCT 基矩阵 (8x8) — 注册为 buffer 跟模型一起搬到 GPU
        self.register_buffer("dct_M", _build_dct_matrix(self.bs), persistent=False)

        in_ch = 3 * self.bs * self.bs                            # 3 × 64 = 192

        # 轻量 CNN 编码器:  192→128→c_token, 两次 stride=2 把 28×28 降到 7×7
        # 再 AdaptiveAvgPool 到 token_grid×token_grid
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 128, 1, bias=False),
            nn.BatchNorm2d(128), nn.GELU(),
        )
        self.blk1 = _DWBlock(128, 128, stride=2)                 # 28 → 14
        self.blk2 = _DWBlock(128, c_token, stride=2)             # 14 →  7
        self.blk3 = _DWBlock(c_token, c_token, stride=1)
        self.pool = nn.AdaptiveAvgPool2d((token_grid, token_grid))

        self.num_tokens = token_grid * token_grid

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _blockwise_dct(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W),  H = W = 整数倍 of bs
        返回: (B, 3*bs*bs, H/bs, W/bs)  —— 每块 DCT 系数沿 channel 维展开
        """
        B, C, H, W = x.shape
        bs = self.bs
        assert H % bs == 0 and W % bs == 0, f"H,W 必须是 {bs} 的整数倍"
        nH, nW = H // bs, W // bs

        # (B, C, H, W) → (B, C, nH, bs, nW, bs) → (B, C, nH, nW, bs, bs)
        blocks = x.view(B, C, nH, bs, nW, bs).permute(0, 1, 2, 4, 3, 5).contiguous()

        # 2D DCT:  M @ block @ M^T   per block
        M = self.dct_M                                            # (bs, bs)
        # blocks: (..., bs, bs);  einsum:  c[i,j] = sum_{p,q} M[i,p] * x[p,q] * M[j,q]
        coeffs = torch.einsum("ip,bcnhpq,jq->bcnhij", M, blocks, M)

        if self.drop_dc:
            # 把每块 (0,0) 直流分量置 0, 强迫网络看交流频率
            coeffs = coeffs.clone()
            coeffs[..., 0, 0] = 0.0

        # 符号对称 log1p 压缩动态范围:  sign(x) * log(1 + |x|)
        coeffs = torch.sign(coeffs) * torch.log1p(coeffs.abs())

        # (B, C, nH, nW, bs, bs) → (B, C, bs*bs, nH, nW) → (B, C*bs*bs, nH, nW)
        coeffs = coeffs.permute(0, 1, 4, 5, 2, 3).contiguous()
        coeffs = coeffs.view(B, C * bs * bs, nH, nW)
        return coeffs

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """returns (tokens, fmap):
            tokens : (B, N, C_token)
            fmap   : (B, C_token, token_grid, token_grid)
        """
        # DCT 本身不带可学习参数 → 放在 no_grad 里更省显存
        feat = self._blockwise_dct(x)                            # (B, 192, 28, 28)
        x = self.stem(feat)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.pool(x)                                         # (B, C, g, g)
        tokens = x.flatten(2).transpose(1, 2).contiguous()       # (B, N, C)
        return tokens, x


if __name__ == "__main__":
    print("=" * 60)
    print("DCTFrequencyBranch 自检")
    print("=" * 60)
    m = DCTFrequencyBranch()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"参数量: {n_params/1e6:.3f}M")
    x = torch.randn(2, 3, 224, 224)
    t, f = m(x)
    print(f"tokens: {tuple(t.shape)}   期望 (2, 64, 192)")
    print(f"fmap  : {tuple(f.shape)}   期望 (2, 192, 8, 8)")
    print("✅ OK")
