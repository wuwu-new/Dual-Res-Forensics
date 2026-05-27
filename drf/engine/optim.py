"""
优化器 / 调度器构造
=====================
- 优化器: AdamW (取代旧版 Adam)
- 调度器: 线性 warmup → cosine annealing, 按 step (而非 epoch) 调度
"""

import math
from typing import Iterable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def _lr_lambda(step: int, warmup_steps: int, total_steps: int, min_ratio: float) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cos


def build_optimizer_and_scheduler(
    params: Iterable[torch.nn.Parameter],
    lr: float = 2e-4,
    weight_decay: float = 5e-4,
    betas=(0.9, 0.999),
    total_steps: int = 1000,
    warmup_steps: int = 100,
    min_lr_ratio: float = 0.01,
):
    optimizer = AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda s: _lr_lambda(s, warmup_steps, total_steps, min_lr_ratio),
    )
    return optimizer, scheduler
