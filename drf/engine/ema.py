"""
EMA: 模型参数指数移动平均
============================
对所有可训练参数维护一份 EMA 副本; 评估时使用 EMA 模型 (通常更稳).
"""

from copy import deepcopy
import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])

    @torch.no_grad()
    def to(self, device):
        self.ema.to(device)
        return self
