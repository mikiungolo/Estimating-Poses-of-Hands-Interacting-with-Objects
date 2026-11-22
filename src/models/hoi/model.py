"""
Full HOI Model: Shared Backbone + Hand/Object Heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .backbone import SharedBackbone
from .heads import (
    HandJointsHead,
    HandMANOHead,
    HandTranslationHead,
    ObjectEmbedding,
    ObjectPoseHead,
)


class HOIModel(nn.Module):
    """
    HOI Pose Estimation Model.
    
    Input: Hand-centered RGB crop (B, 3, 256, 256) + object class (B,)
    Output: Hand MANO, Joints 3D, Hand Trans, Object Trans+Quat
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.backbone = SharedBackbone(config)

        self.hand_mano_head = HandMANOHead(config)
        self.hand_joints_head = HandJointsHead(config)
        self.hand_trans_head = HandTranslationHead(config)

        self.obj_embedding = ObjectEmbedding(config)
        self.obj_pose_head = ObjectPoseHead(config)

    def forward(
        self,
        rgb: torch.Tensor,
        obj_class: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            rgb: (B, 3, H, W) hand-centered crop
            obj_class: (B,) int64 object category [0-19]
        Returns:
            dict with:
                - hand_mano: (B, 48)
                - hand_joints_3d: (B, 21, 3)
                - hand_trans: (B, 3)
                - obj_trans: (B, 3)
                - obj_quat: (B, 4) normalized
                - obj_pose: (B, 7) [trans, quat]
                - features: (B, feature_dim) shared
        """
        raise NotImplementedError("Implement forward pass")
