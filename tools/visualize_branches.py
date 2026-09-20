"""Visualize semantic, residual, and fused DRF evidence for paper figures."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drf.data.transforms import build_test_transforms
from drf.engine import pick_device, safe_load_checkpoint
from drf.core import build_model
from tools.test import _load_compact_state


def _resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


def _normalize_map(value: torch.Tensor) -> np.ndarray:
    value = value.float()
    value = value - value.min()
    value = value / value.max().clamp_min(1e-8)
    return value.cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefer", choices=["model", "ema"], default="model")
    args = ap.parse_args()

    with open(_resolve(args.config), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = pick_device()
    model = build_model(cfg).to(device)
    ckpt = safe_load_checkpoint(str(_resolve(args.ckpt)), map_location=device)
    key = "ema_state" if args.prefer == "ema" and ckpt.get("ema_state") else "model_state"
    _load_compact_state(model, ckpt[key], f"{args.ckpt}:{key}")
    model.eval()

    image = Image.open(_resolve(args.image)).convert("RGB")
    rgb = np.asarray(image.resize((cfg["data"]["image_size"], cfg["data"]["image_size"])))
    tensor = build_test_transforms(
        cfg["data"]["image_size"], cfg["data"].get("normalization", "clip")
    )(image=rgb)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor, return_aux=True)
        probability = torch.sigmoid(output["logit"])[0].item()

    residual_map = output.get("residual_map")
    if residual_map is None:
        raise RuntimeError("当前配置没有残差分支，无法生成分支可视化")
    residual_energy = residual_map[0].abs().mean(dim=0)
    residual_energy = F.interpolate(
        residual_energy[None, None], size=rgb.shape[:2], mode="bilinear", align_corners=False
    )[0, 0]

    semantic_tokens = output.get("semantic_tokens")
    if semantic_tokens is None or semantic_tokens.size(1) < 2:
        raise RuntimeError("当前配置没有 CLIP patch tokens，无法生成语义分支可视化")
    cls_token = F.normalize(semantic_tokens[0, 0], dim=0)
    patch_tokens = F.normalize(semantic_tokens[0, 1:], dim=-1)
    semantic_relevance = patch_tokens @ cls_token
    semantic_side = int(math.isqrt(semantic_relevance.numel()))
    if semantic_side * semantic_side != semantic_relevance.numel():
        raise RuntimeError("CLIP patch token 数量不是平方数，无法还原二维语义图")
    semantic_relevance = semantic_relevance.reshape(semantic_side, semantic_side)
    semantic_relevance = F.interpolate(
        semantic_relevance[None, None], size=rgb.shape[:2], mode="bilinear", align_corners=False
    )[0, 0]

    aux = output.get("fusion_aux")
    if not aux or aux.get("cross_attention") is None:
        raise RuntimeError("当前配置没有 GRCA attention，无法生成融合可视化")
    attention = aux["cross_attention"][0].mean(dim=(0, 1))
    # The first CLIP token is CLS; paper heatmaps use patch tokens only.
    patch_attention = attention[1:]
    side = int(math.isqrt(patch_attention.numel()))
    if side * side != patch_attention.numel():
        patch_attention = attention
        side = int(math.isqrt(patch_attention.numel()))
    patch_attention = patch_attention[: side * side].reshape(side, side)
    patch_attention = F.interpolate(
        patch_attention[None, None], size=rgb.shape[:2], mode="bilinear", align_corners=False
    )[0, 0]

    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(rgb)
    axes[0].set_title(f"Input  P(fake)={probability:.3f}")
    axes[1].imshow(rgb)
    axes[1].imshow(_normalize_map(semantic_relevance), cmap="viridis", alpha=0.55)
    axes[1].set_title("CLIP semantic relevance")
    axes[2].imshow(rgb)
    axes[2].imshow(_normalize_map(residual_energy), cmap="magma", alpha=0.58)
    axes[2].set_title("SRM residual evidence")
    axes[3].imshow(rgb)
    axes[3].imshow(_normalize_map(patch_attention), cmap="jet", alpha=0.52)
    gate = float(aux["gate"])
    axes[3].set_title(f"GRCA fusion  cross={gate:.3f}, residual={1.0-gate:.3f}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
