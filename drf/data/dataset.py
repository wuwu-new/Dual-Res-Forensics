"""
JSON 驱动的人脸伪造数据集
============================
JSON 格式 (list of dict):
  [
    {"image_path": "/abs/path/real.jpg", "label": 0},
    {"image_path": "/abs/path/fake.jpg", "label": 1, "mask_path": "..."}
  ]

label: 0 = 真, 1 = 伪
"""

import json
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import build_train_transforms, build_test_transforms


class ForgeryDataset(Dataset):
    def __init__(self, json_path: str, image_size: int = 224,
                 mode: str = "train", augment_strength: float = 1.0):
        assert mode in ("train", "test")
        with open(json_path, "r", encoding="utf-8") as f:
            self.samples: List[Dict] = json.load(f)
        self.mode = mode
        self.transform = (
            build_train_transforms(image_size, augment_strength)
            if mode == "train"
            else build_test_transforms(image_size)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = np.array(Image.open(s["image_path"]).convert("RGB"))
        out = self.transform(image=img)
        return {
            "image": out["image"],
            "label": torch.tensor(int(s["label"]), dtype=torch.long),
            "image_path": s["image_path"],
        }


def collate_fn(batch):
    return {
        "image":      torch.stack([b["image"] for b in batch]),
        "label":      torch.stack([b["label"] for b in batch]),
        "image_path": [b["image_path"] for b in batch],
    }
