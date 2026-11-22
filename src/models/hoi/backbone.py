"""
Shared Backbone for HOI: Reuses frozen HaMeR encoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.hamer.backbone import ViTBackbone


class SharedBackbone(nn.Module):
    """
    Shared feature extractor for hand and object.
    Uses HaMeR ViT backbone (frozen) + adaptive pooling + projection.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.hamer_backbone = ViTBackbone(
            model_name=config.hamer.backbone,
            pretrained=True,
            freeze=config.hoi.freeze_backbone,
        )
        feat_dim = self.hamer_backbone.embed_dim

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, config.hoi.feature_dim),
            nn.LayerNorm(config.hoi.feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) hand-centered RGB crop
        Returns:
            global_feat: (B, feature_dim) - shared latent features
        """
        raise NotImplementedError("Implement forward pass")
