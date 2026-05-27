"""
损失函数
==========
- BinaryClsLoss     : BCEWithLogits + label smoothing (替换旧版 2-class CE)
- HardNegSupConLoss : 难负例挖掘的样本级监督对比损失
                      (相对旧版 SupCon: 每个 anchor 只保留 top-k 难负例,
                       正例不变. 对 fake-fake 仍计入正对, 保持核心不变.)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryClsLoss(nn.Module):
    """logit (B,) + label (B,) ∈ {0,1} → BCEWithLogits with label smoothing."""

    def __init__(self, label_smoothing: float = 0.05, pos_weight: Optional[float] = None):
        super().__init__()
        self.eps = float(label_smoothing)
        self.pos_weight = (None if pos_weight is None
                           else torch.tensor([float(pos_weight)]))

    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        y = label.float()
        if self.eps > 0:
            y = y * (1.0 - self.eps) + 0.5 * self.eps          # 标签平滑
        pw = (self.pos_weight.to(logit.device) if self.pos_weight is not None else None)
        return F.binary_cross_entropy_with_logits(logit, y, pos_weight=pw)


class HardNegSupConLoss(nn.Module):
    """
    Hard-Negative Supervised Contrastive Loss.

    feat (B, D), label (B,) → scalar loss
    步骤:
      1. L2 归一化 → 余弦相似度 / 温度
      2. 屏蔽自身相似度
      3. 正例: 同类 (去自身)
      4. 负例: 异类中相似度 top-k 高的样本 (难负例),
              其余负例位置在 logits 中置 -inf, 不参与 softmax
      5. 标准 SupCon log-prob 平均
    """

    def __init__(self, temperature: float = 0.1, num_hard_neg: int = 16):
        super().__init__()
        self.tau = float(temperature)
        self.k_neg = int(num_hard_neg)

    def forward(self, feat: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        B = feat.size(0)
        if B < 2:
            return feat.new_zeros(())
        feat = F.normalize(feat, dim=1)
        sim = feat @ feat.t() / self.tau                       # (B, B)

        eye = torch.eye(B, device=feat.device, dtype=torch.bool)
        same = label.view(-1, 1).eq(label.view(1, -1))          # (B, B) bool
        pos_mask = same & ~eye
        neg_mask = ~same                                        # 异类

        # ----- 难负例挖掘: 每行仅保留 top-k 负例, 其余置 -inf -----
        sim_for_neg = sim.masked_fill(~neg_mask, float("-inf"))
        k = min(self.k_neg, max(int(neg_mask.sum(dim=1).max().item()), 1))
        topk_vals, topk_idx = sim_for_neg.topk(k, dim=1)
        hard_neg_mask = torch.zeros_like(sim, dtype=torch.bool)
        hard_neg_mask.scatter_(1, topk_idx, True)
        hard_neg_mask &= neg_mask                               # 防止 -inf 也被 scatter 进来

        # ----- 构造 logits: 自身 -inf, 非 (正例 ∪ 难负例) 也 -inf -----
        keep = pos_mask | hard_neg_mask
        logits = sim.masked_fill(~keep, float("-inf"))
        logits = logits.masked_fill(eye, float("-inf"))

        # 没有正例的行跳过 (clamp 防除零)
        log_prob = F.log_softmax(logits, dim=1)
        # 关键: 一整行全 -inf (没有正例也没有难负例, 或者 batch 全单类) 时
        # log_softmax 会得到 NaN, 与 mask 相乘后 0*NaN=NaN 污染整批 loss。
        # 这里把 NaN/Inf 全部替换为 0, 再靠下面的 valid mask 把这些行剔除。
        log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0)
        pos_log_prob = (pos_mask.float() * log_prob).sum(dim=1)
        n_pos = pos_mask.float().sum(dim=1).clamp(min=1.0)
        loss = -(pos_log_prob / n_pos)

        # 完全没有正例的行 (pos_mask 全 0) 直接置 0, 不影响均值
        valid = pos_mask.any(dim=1).float()
        loss = loss * valid
        denom = valid.sum().clamp(min=1.0)
        return loss.sum() / denom
