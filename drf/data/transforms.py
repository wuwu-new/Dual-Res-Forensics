"""
增强管线 (相对旧版重新排序、并加入 Cutout / GaussNoise, 支持强度调节)
=========================================================================
顺序:
  spatial      : HFlip → Rotate → RandomResizedCrop
  photometric  : ColorJitter (brightness/contrast/sat/hue)
  noise        : GaussNoise
  compression  : ImageCompression (JPEG)
  occlusion    : CoarseDropout (Cutout)
  finalize     : Resize → Normalize → ToTensor

augment_strength ∈ [0,1] 统一缩放各项概率与幅度.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


# CLIP 官方使用的归一化, 与预训练保持一致
CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def build_train_transforms(image_size: int = 224, augment_strength: float = 1.0):
    s = float(max(0.0, min(1.0, augment_strength)))
    return A.Compose([
        # spatial
        A.HorizontalFlip(p=0.5 * s),
        A.Rotate(limit=int(10 * s), border_mode=0, p=0.5 * s),
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(1.0 - 0.2 * s, 1.0),
            ratio=(0.9, 1.1),
            p=0.7 * s + 0.3,
        ),
        # photometric
        A.ColorJitter(
            brightness=0.1 * s, contrast=0.1 * s,
            saturation=0.1 * s, hue=0.02 * s, p=0.5 * s,
        ),
        # noise
        A.GaussNoise(std_range=(0.02, 0.05 + 0.05 * s), p=0.3 * s),
        # compression
        A.ImageCompression(quality_range=(int(40 + (1 - s) * 30), 100), p=0.5 * s),
        # occlusion (cutout)
        A.CoarseDropout(
            num_holes_range=(1, 1 + int(2 * s)),
            hole_height_range=(8, int(8 + 24 * s)),
            hole_width_range=(8, int(8 + 24 * s)),
            p=0.3 * s,
        ),
        # finalize
        A.Resize(image_size, image_size),
        A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ToTensorV2(),
    ])


def build_test_transforms(image_size: int = 224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ToTensorV2(),
    ])
