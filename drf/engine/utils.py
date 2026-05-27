"""杂项工具: 随机种子 / 设备 / 检查点 IO"""

from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """固定 Python / NumPy / PyTorch (CPU+GPU) 的随机种子。
    deterministic=True 会进一步开启 cudnn 确定性算法 (会变慢)。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def pick_device(prefer_cuda: bool = True) -> torch.device:
    return torch.device("cuda" if (prefer_cuda and torch.cuda.is_available()) else "cpu")


def safe_load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """统一加载 ckpt; 不存在则抛 FileNotFoundError。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)
