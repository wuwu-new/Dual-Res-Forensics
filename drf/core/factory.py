"""Build paper models from one experiment configuration."""

from __future__ import annotations

import torch.nn as nn

from .model import DRFModel
from .xception_baseline import XceptionBaseline


def build_model(cfg: dict) -> nn.Module:
    model_cfg = cfg.get("model", {})
    architecture = str(model_cfg.get("architecture", "drf")).lower()
    if architecture == "xception":
        return XceptionBaseline(
            pretrained=bool(model_cfg.get("pretrained", True)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    if architecture != "drf":
        raise ValueError(f"未知 model.architecture={architecture!r}")

    backbone = cfg["backbone"]
    residual = cfg["residual"]
    fusion = cfg["fusion"]
    freq = cfg.get("freq", {})
    return DRFModel(
        clip_name=backbone["clip_name"],
        c_token=int(residual["c_token"]),
        token_grid=int(residual["token_grid"]),
        fused_dim=int(fusion["fused_dim"]),
        fusion_embed_dim=int(fusion["embed_dim"]),
        fusion_num_heads=int(fusion["num_heads"]),
        dropout=float(fusion.get("dropout", 0.1)),
        use_boundary_head=bool(model_cfg.get("use_boundary_head", False)),
        freeze_clip=bool(backbone.get("freeze_clip", True)),
        use_srm=bool(residual.get("use_srm", True)),
        use_gate=bool(fusion.get("use_gate", True)),
        use_freq=bool(freq.get("use_freq", residual.get("use_freq", False))),
        freq_c_token=int(freq.get("c_token", residual.get("freq_c_token", residual["c_token"]))),
        freq_token_grid=int(freq.get("token_grid", residual.get("freq_token_grid", residual["token_grid"]))),
        proj_dim=int(model_cfg.get("proj_dim", 0)),
        use_semantic=bool(model_cfg.get("use_semantic", True)),
        use_residual=bool(model_cfg.get("use_residual", True)),
    )
