"""DRF v2 损失函数模块。

包含:
    - HardNegSupConLoss: 带难负例挖掘（top-k）的有监督对比损失
    - BCESupConLoss:     BCEWithLogits + Hard-Neg SupCon 的联合损失

设计动机
--------
Deepfake A/B 图像差异极小（FF++ 全局一致），普通 BCE 在难样本上梯度饱和。
Hard-Negative SupCon 强制 anchor 只与「最难区分」的 top-k 负样本对比，
把对比信号集中到决策边界附近的样本上，提升跨域 AUC。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardNegSupConLoss(nn.Module):
    """带难负例挖掘的有监督对比损失（Supervised Contrastive Learning, NeurIPS'20 变体）。

    与原始 SupCon 的区别：分母不再聚合「全部负样本」，而是只保留每个 anchor
    相似度最高（即最难区分）的 ``top_k`` 个负样本，使梯度集中在难例上。

    Args:
        temperature: 温度系数 τ，越小对难例越敏感。默认 0.3。
        top_k:       每个 anchor 参与对比的难负例数量。默认 16。
        base_temperature: 损失缩放基准，沿用原论文实现，默认与 temperature 一致。
    """

    def __init__(
        self,
        temperature: float = 0.3,
        top_k: int = 16,
        base_temperature: float | None = None,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature 必须为正数")
        self.temperature = temperature
        self.top_k = top_k
        self.base_temperature = base_temperature or temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算损失。

        Args:
            features: [B, D] 的特征向量（建议来自投影头，会在内部做 L2 归一化）。
            labels:   [B] 的整型标签（0=真, 1=伪）。

        Returns:
            标量损失。
        """
        if features.dim() != 2:
            raise ValueError(f"features 期望 [B, D]，实际 {tuple(features.shape)}")
        device = features.device
        batch_size = features.size(0)

        features = F.normalize(features, dim=1)
        labels = labels.contiguous().view(-1, 1)

        # 相似度矩阵 [B, B]
        sim = torch.matmul(features, features.t()) / self.temperature
        # 数值稳定：减去每行最大值
        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()

        # 正样本掩码（同标签且非自身）
        pos_mask = torch.eq(labels, labels.t()).float().to(device)
        self_mask = torch.eye(batch_size, device=device)
        pos_mask = pos_mask - self_mask                       # 去掉对角线
        neg_mask = 1.0 - torch.eq(labels, labels.t()).float() # 异标签 = 负样本

        # ---- 难负例挖掘：每行只保留相似度最高的 top_k 个负样本 ----
        neg_sim = sim.masked_fill(neg_mask == 0, float("-inf"))
        k = min(self.top_k, batch_size - 1)
        # 取 top_k 难负例的列索引
        _, hard_idx = neg_sim.topk(k, dim=1)
        hard_neg_mask = torch.zeros_like(sim)
        hard_neg_mask.scatter_(1, hard_idx, 1.0)
        hard_neg_mask = hard_neg_mask * neg_mask              # 防止 -inf 行污染

        # 分母 = 正样本 + 难负样本
        logits_mask = pos_mask + hard_neg_mask
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        # 仅对「有正样本」的 anchor 求均值，避免除零
        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        if valid.sum() == 0:
            return torch.zeros((), device=device)

        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[valid] / pos_count[valid]
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


class BCESupConLoss(nn.Module):
    """联合损失：BCEWithLogits（分类） + 加权 Hard-Neg SupCon（表征）。

    total = bce + supcon_weight * supcon
    """

    def __init__(
        self,
        supcon_weight: float = 0.3,
        temperature: float = 0.3,
        top_k: int = 16,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.supcon_weight = supcon_weight
        self.label_smoothing = label_smoothing
        self.bce = nn.BCEWithLogitsLoss()
        self.supcon = HardNegSupConLoss(temperature=temperature, top_k=top_k)

    def forward(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """返回包含各分项的字典，便于日志记录与消融。"""
        target = labels.float()
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce_loss = self.bce(logits.view(-1), target.view(-1))
        supcon_loss = self.supcon(features, labels)
        total = bce_loss + self.supcon_weight * supcon_loss
        return {"loss": total, "bce": bce_loss, "supcon": supcon_loss}
