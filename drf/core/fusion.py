"""
门控残差引导交叉注意力 (Gated Residual-guided Cross-Attention, GRCA)
===================================================================
核心思想未变：以高频残差作为 Query，主动查询 CLIP 全局语义 K/V。
具体方法的改动相对旧版：
  1) 手写 Scaled-Dot-Product Attention，而不是 nn.MultiheadAttention
  2) 采用 Pre-LayerNorm (旧版是 Post-LN)
  3) FFN 改为 SwiGLU
  4) 池化改为 可学习 [CLS] Query Attention Pooling (旧版是 mean-pool)
  5) 新增可学习标量门 g，把残差全局特征和 attended-feat 加权融合
  6) 模块输出 (B, out_dim)，与旧版接口兼容
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU FFN: y = Linear( SiLU(W1 x) * W2 x )."""

    def __init__(self, dim: int, hidden_mult: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class _SDPAttention(nn.Module):
    """手写的多头 scaled dot-product attention (Q 来自一处, K/V 来自另一处)。"""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.dh = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.dh)

    def _split(self, x):
        # (B, N, D) → (B, h, N, dh)
        b, n, _ = x.shape
        return x.view(b, n, self.h, self.dh).transpose(1, 2)

    def forward(self, x_q, x_kv):
        q = self._split(self.q(x_q))
        k = self._split(self.k(x_kv))
        v = self._split(self.v(x_kv))
        attn = (q @ k.transpose(-2, -1)) * self.scale          # (B, h, Nq, Nk)
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = attn @ v                                         # (B, h, Nq, dh)
        b, h, n, dh = out.shape
        out = out.transpose(1, 2).contiguous().view(b, n, h * dh)
        return self.o(out)


class _AttnPool(nn.Module):
    """可学习 [CLS] query 的 attention pooling: 把 token 序列聚合成单一向量。"""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.attn = _SDPAttention(dim, num_heads=num_heads, dropout=dropout)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, tokens):
        b = tokens.size(0)
        cls = self.cls.expand(b, -1, -1)                       # (B, 1, D)
        out = self.attn(self.norm_q(cls), self.norm_kv(tokens))  # (B, 1, D)
        return out.squeeze(1)                                  # (B, D)


class GatedResidualCrossAttention(nn.Module):
    """
    输入:
      adapter_tokens: (B, N_a, C_a)   高频残差 token 序列 (Query)
      clip_tokens   : (B, N_c, C_c)   CLIP patch token 序列 (K/V)
    输出:
      fused: (B, out_dim)             融合后的判别特征

    执行顺序 (与旧版完全不同):
      proj Q/K/V → Pre-LN → 手写 cross-attn → 残差
        → Pre-LN → SwiGLU FFN → 残差
        → 可学习 [CLS] query attention pooling     (cross-attended 序列)
        → 同样的 pool 单独跑一遍 残差 token (作为旁路)
        → 门控加权融合: out = g * f_cross + (1-g) * f_residual
        → Linear 投影到 out_dim
    """

    def __init__(
        self,
        adapter_dim: int,
        clip_dim: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        out_dim: int = 128,
        dropout: float = 0.1,
        use_gate: bool = True,
        freq_dim: int = 0,
    ):
        super().__init__()
        self.use_gate = bool(use_gate)
        self.use_freq = bool(freq_dim > 0)

        # ① 输入投影到统一维度 (Q 来自残差, K/V 来自 CLIP + 可选频域)
        self.proj_q = nn.Linear(adapter_dim, embed_dim, bias=False)
        self.proj_kv = nn.Linear(clip_dim, embed_dim, bias=False)
        if self.use_freq:
            # 频域 token 独立线性投影 (频域统计量级与 CLIP 差很多, 不能共享权重)
            self.proj_kv_freq = nn.Linear(freq_dim, embed_dim, bias=False)
            # 可学习类型嵌入: 区分 K/V 来源是 CLIP 还是频域 (类似 BERT segment emb)
            self.kv_type = nn.Parameter(torch.zeros(2, embed_dim))
            nn.init.trunc_normal_(self.kv_type, std=0.02)

        # ② Pre-LN + 手写 cross-attention
        self.norm_q1 = nn.LayerNorm(embed_dim)
        self.norm_kv1 = nn.LayerNorm(embed_dim)
        self.cross = _SDPAttention(embed_dim, num_heads=num_heads, dropout=dropout)

        # ③ Pre-LN + SwiGLU
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim, hidden_mult=2.0, dropout=dropout)

        # ④ 两路 attention pooling: cross 序列 + 残差旁路
        self.pool_cross = _AttnPool(embed_dim, num_heads=4, dropout=dropout)
        self.pool_resid = _AttnPool(embed_dim, num_heads=4, dropout=dropout) if self.use_gate else None

        # ⑤ 标量门 g (sigmoid 控制 cross / residual 占比); 关闭时 g 固定=1
        if self.use_gate:
            self.gate = nn.Parameter(torch.zeros(1))            # 初始 g=0.5
        else:
            self.register_buffer("gate", torch.tensor(float("inf")))  # sigmoid(∞)=1

        # ⑥ 输出投影
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

    def forward(
        self,
        adapter_tokens: torch.Tensor,
        clip_tokens: torch.Tensor,
        freq_tokens: torch.Tensor = None,
    ):
        q = self.proj_q(adapter_tokens)            # (B, N_a, D)
        kv = self.proj_kv(clip_tokens)             # (B, N_c, D)
        if self.use_freq and freq_tokens is not None:
            kv_f = self.proj_kv_freq(freq_tokens)  # (B, N_f, D)
            # 加上可学习的来源嵌入, 让 attention 能区分两路 K/V
            kv = kv + self.kv_type[0]              # type 0 = CLIP
            kv_f = kv_f + self.kv_type[1]          # type 1 = freq
            kv = torch.cat([kv, kv_f], dim=1)      # (B, N_c+N_f, D)

        # Pre-LN cross-attention + 残差
        attn_out = self.cross(self.norm_q1(q), self.norm_kv1(kv))
        x = q + attn_out

        # Pre-LN SwiGLU + 残差
        x = x + self.ffn(self.norm2(x))

        # 两路 pooling + 门控融合 (use_gate=False 时 g≡1, 退化为纯 cross)
        f_cross = self.pool_cross(x)           # (B, D)
        if self.use_gate and self.pool_resid is not None:
            f_resid = self.pool_resid(q)
            g = torch.sigmoid(self.gate)
            fused = g * f_cross + (1.0 - g) * f_resid
        else:
            fused = f_cross

        return self.head(fused)                # (B, out_dim)
