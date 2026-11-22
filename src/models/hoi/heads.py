"""
Prediction Heads for HOI Model: Hand (MANO, Joints, Trans) + Object (Pose, Embedding).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class MLPHead(nn.Module):
    """Generic MLP."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HandMANOHead(nn.Module):
    """MANO params head (input: shared features)."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.mlp = MLPHead(
            input_dim=config.hoi.feature_dim,
            hidden_dims=config.hoi.hand_mano_head_hidden,
            output_dim=config.hamer.mano_params,
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(feat)


class HandJointsHead(nn.Module):
    """Joints 3D relative coords head."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.mlp = MLPHead(
            input_dim=config.hoi.feature_dim,
            hidden_dims=config.hoi.hand_joints_head_hidden,
            output_dim=config.hamer.num_joints * 3,
        )
        self.num_joints = config.hamer.num_joints

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        out = self.mlp(feat)
        return out.view(-1, self.num_joints, 3)


class HandTranslationHead(nn.Module):
    """Hand global translation head."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.mlp = MLPHead(
            input_dim=config.hoi.feature_dim,
            hidden_dims=[256, 128],
            output_dim=3,
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(feat)


class ObjectEmbedding(nn.Module):
    """Object category embedding (20 YCB classes)."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.hoi.num_object_classes,
            embedding_dim=config.hoi.obj_embed_dim,
        )

    def forward(self, obj_class: torch.Tensor) -> torch.Tensor:
        return self.embedding(obj_class)


class ObjectPoseHead(nn.Module):
    """Object pose head: translation (3) + quaternion (4)."""

    def __init__(self, config: DictConfig):
        super().__init__()
        input_dim = config.hoi.feature_dim + config.hoi.obj_embed_dim
        self.mlp = MLPHead(
            input_dim=input_dim,
            hidden_dims=config.hoi.obj_pose_head_hidden,
            output_dim=7,
        )

    def forward(self, feat: torch.Tensor, obj_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, feature_dim) shared features
            obj_embed: (B, embed_dim) object embedding
        Returns:
            (B, 7) = [tx, ty, tz, qx, qy, qz, qw] (normalized quaternion)
        """
        raise NotImplementedError("Implement forward pass")
