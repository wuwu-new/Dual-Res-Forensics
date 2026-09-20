"""Paper baseline wrapper around timm's ImageNet-pretrained legacy Xception."""

from __future__ import annotations

import torch
import torch.nn as nn


class XceptionBaseline(nn.Module):
    """Xception with the same output contract used by the DRF trainer."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Xception baseline 需要 timm>=1.0.0；请先 pip install -e .") from exc
        self.backbone = timm.create_model(
            "legacy_xception",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = int(self.backbone.num_features)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, 1))

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        feature = self.backbone(x)
        logit = self.classifier(feature).squeeze(-1)
        return {"logit": logit, "fused_feat": feature}

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]
