"""
DRF v2 主模型 (Orchestrator)
=============================
管线顺序:
  x (B,3,H,W)
   ├──→ CLIP-ViT (frozen)              → clip_tokens (B, N_c, C_c)
   └──→ SRM 高频残差 (固定)            → residual (B, 9, H, W)
            → DWResidualEncoder        → (residual_tokens, fmap)
   → GatedResidualCrossAttention(Q=residual_tokens, K/V=clip_tokens)
   → fused_feat (B, fused_dim)
   ├──→ BinaryClassifierHead           → logit (B,)
   └──(可选)→ BoundaryDecoder(fmap)    → boundary_map (B, 1, h, w)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import CLIPVisionModel

from .filters import SRMResidual
from .residual_encoder import DWResidualEncoder
from .fusion import GatedResidualCrossAttention
from .heads import BinaryClassifierHead, BoundaryDecoder
from .freq_branch import DCTFrequencyBranch


class DRFModel(nn.Module):
    def __init__(
        self,
        clip_name: str = "openai/clip-vit-base-patch32",
        c_token: int = 192,
        token_grid: int = 8,
        fused_dim: int = 128,
        fusion_embed_dim: int = 256,
        fusion_num_heads: int = 8,
        dropout: float = 0.1,
        use_boundary_head: bool = False,
        freeze_clip: bool = True,
        use_srm: bool = True,
        use_gate: bool = True,
        use_freq: bool = False,
        freq_c_token: int = 192,
        freq_token_grid: int = 8,
        proj_dim: int = 0,
    ):
        super().__init__()

        # ---- 主通道：CLIP (默认冻结) ----
        self.clip = CLIPVisionModel.from_pretrained(clip_name)
        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False
            self.clip.eval()                          # 锁住 BN/Dropout 行为
        self.clip_dim = self.clip.config.hidden_size

        # ---- 残差通道 ----
        # 消融开关: use_srm=False 时跳过 SRM, 直接用 RGB (3ch),
        # 相应地把 DW 编码器输入通道从 9 改为 3。
        self.use_srm = bool(use_srm)
        if self.use_srm:
            self.srm = SRMResidual()                  # 9-ch 残差
            in_ch = 9
        else:
            self.srm = None
            in_ch = 3
        self.residual_encoder = DWResidualEncoder(
            c_token=c_token, token_grid=token_grid, in_channels=in_ch,
        )

        # ---- 频域分支 (可选) ----
        self.use_freq = bool(use_freq)
        if self.use_freq:
            self.freq_branch = DCTFrequencyBranch(
                c_token=freq_c_token, token_grid=freq_token_grid,
            )
            freq_dim_for_fusion = freq_c_token
        else:
            self.freq_branch = None
            freq_dim_for_fusion = 0

        # ---- 融合 ----
        self.fusion = GatedResidualCrossAttention(
            adapter_dim=self.residual_encoder.c_token,
            clip_dim=self.clip_dim,
            embed_dim=fusion_embed_dim,
            num_heads=fusion_num_heads,
            out_dim=fused_dim,
            dropout=dropout,
            use_gate=use_gate,
            freq_dim=freq_dim_for_fusion,
        )

        # ---- 头 ----
        self.classifier = BinaryClassifierHead(fused_dim, hidden_dim=64, dropout=dropout)
        self.use_boundary_head = use_boundary_head
        if use_boundary_head:
            self.boundary = BoundaryDecoder(self.residual_encoder.c_token)

        # ---- 对比学习投影头 (可选) ----
        # 以前直接拿 fused_feat 算 SupCon, 会被 BCE 拉到 logit 极端, 导致对比损失几乎不动
        # 追加一个 MLP 投影头 (SimCLR/SupCon 标准做法), 让对比任务在独立空间优化
        self.proj_dim = int(proj_dim)
        if self.proj_dim > 0:
            self.proj_head = nn.Sequential(
                nn.Linear(fused_dim, fused_dim, bias=False),
                nn.BatchNorm1d(fused_dim),
                nn.GELU(),
                nn.Linear(fused_dim, self.proj_dim, bias=False),
            )
        else:
            self.proj_head = None

    # ----- CLIP 主通道单独前向 (始终 no_grad，省显存) -----
    @torch.no_grad()
    def _forward_clip(self, x: torch.Tensor) -> torch.Tensor:
        was_train = self.clip.training
        self.clip.eval()
        out = self.clip(pixel_values=x).last_hidden_state   # (B, N_c, C_c)
        if was_train:
            self.clip.train()
        return out

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 主通道
        clip_tokens = self._forward_clip(x)

        # 残差通道 (可选 SRM)
        residual = self.srm(x) if self.srm is not None else x   # (B, 9 or 3, H, W)
        residual_tokens, fmap = self.residual_encoder(residual)

        # 频域通道 (可选)
        freq_tokens = None
        if self.freq_branch is not None:
            freq_tokens, _ = self.freq_branch(x)

        # 融合
        fused = self.fusion(residual_tokens, clip_tokens, freq_tokens)   # (B, fused_dim)

        # 分类
        logit = self.classifier(fused)                      # (B,)
        out: Dict[str, torch.Tensor] = {"logit": logit, "fused_feat": fused}
        if self.proj_head is not None:
            # BN 需要 batch≥2, eval 时会需要 .eval() 模式 (本身训练路径不走这里)
            out["proj_feat"] = self.proj_head(fused)
        if self.use_boundary_head:
            out["boundary_map"] = self.boundary(fmap)
        return out

    # 训练时只回传非冻结参数，便于在 main 里写：
    #   optimizer = AdamW(model.trainable_parameters(), ...)
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


if __name__ == "__main__":
    print("=" * 60)
    print("DRF v2 模型自检")
    print("=" * 60)
    model = DRFModel()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {n_train/1e6:.2f}M | 总参数: {n_total/1e6:.2f}M")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"logit shape      : {out['logit'].shape}      期望 [2]")
    print(f"fused_feat shape : {out['fused_feat'].shape}  期望 [2, 128]")
    print("✅ OK")
