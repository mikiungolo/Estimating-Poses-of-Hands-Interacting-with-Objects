"""
Full HaMeR Model: ViT Backbone + MANO/Joints/Translation Heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .backbone import create_vit_backbone
from .heads import HandTranslationHead, JointsHead, MANOHead


class HaMeR(nn.Module):
    """
    HaMeR: Hand Mesh Reconstruction with Transformers.
    
    Input: Hand-centered RGB crop (B, 3, 256, 256)
    Output: MANO params, Joints 3D, Hand translation
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.backbone = create_vit_backbone(config)
        feat_dim = self.backbone.embed_dim

        self.mano_head = MANOHead(
            input_dim=feat_dim,
            hidden_dims=config.hamer.get("mano_head_hidden", [512, 256]),
            mano_params=config.hamer.mano_params,
        )
        self.joints_head = JointsHead(
            input_dim=feat_dim,
            hidden_dims=config.hamer.get("joints_head_hidden", [512, 256]),
            num_joints=config.hamer.num_joints,
        )
        self.trans_head = HandTranslationHead(
            input_dim=feat_dim,
            hidden_dims=config.hamer.get("trans_head_hidden", [256, 128]),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(feat_dim, feat_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) hand-centered crop
        Returns:
            dict with:
                - mano_params: (B, 48)
                - joints_3d: (B, 21, 3)
                - hand_trans: (B, 3)
                - features: (B, feat_dim) global features
        """
        raise NotImplementedError("Implement forward pass")

    @classmethod
    def load_from_checkpoint(cls, ckpt_path: str, config: DictConfig = None, map_location="cpu"):
        """Load model from checkpoint."""
        if config is None:
            from src.utils.config import load_config
            config = load_config("configs/config.yaml")

        model = cls(config)
        ckpt = torch.load(ckpt_path, map_location=map_location)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        return model
