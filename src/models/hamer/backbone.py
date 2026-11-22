"""
ViT Backbone for HaMeR (based on timm/vit_base_patch16_224).
Feature extractor for hand-centered crops.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


class ViTBackbone(nn.Module):
    """
    Wrapper ViT backbone for HaMeR.
    Extracts spatial feature maps from hand-centered RGB crops.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze: bool = False,
        out_indices: tuple[int, ...] = (11,),
    ):
        super().__init__()
        self.model_name = model_name
        self.out_indices = out_indices

        self.vit = VisionTransformer.from_pretrained(model_name) if pretrained else VisionTransformer(model_name)

        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False

        self.embed_dim = self.vit.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) - hand-centered crops
        Returns:
            feature_maps: (B, embed_dim, H/16, W/16)
        """
        raise NotImplementedError("Implement forward with feature extraction")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global features (CLS token or pooled)."""
        raise NotImplementedError("Implement forward_features")


def create_vit_backbone(config) -> ViTBackbone:
    """Factory from config."""
    return ViTBackbone(
        model_name=config.hamer.backbone,
        pretrained=True,
        freeze=config.hoi.freeze_backbone,
    )
