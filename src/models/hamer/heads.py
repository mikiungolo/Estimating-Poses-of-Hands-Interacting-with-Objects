"""
Prediction Heads for HaMeR: MANO params + Joints 3D.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """Generic MLP for regression."""

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


class MANOHead(nn.Module):
    """Head for MANO parameters (45 pose + 3 global orient)."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: list[int] = [512, 256],
        mano_params: int = 48,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mlp = MLPHead(input_dim, hidden_dims, mano_params, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class JointsHead(nn.Module):
    """Head for joint 3D coordinates relative (21 joints x 3)."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: list[int] = [512, 256],
        num_joints: int = 21,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mlp = MLPHead(input_dim, hidden_dims, num_joints * 3, dropout)
        self.num_joints = num_joints

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        return out.view(-1, self.num_joints, 3)


class HandTranslationHead(nn.Module):
    """Head for global hand translation (3)."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: list[int] = [256, 128],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mlp = MLPHead(input_dim, hidden_dims, 3, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
