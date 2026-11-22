"""
Multi-task Loss Functions for HOI.

Spec: "MSE across predicted and ground-truth parameters,
supervising 3D hand translations, 3D object translations and rotation
and relative hand-object translations."
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


def quat_mse_loss(pred_quat: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
    """MSE loss for quaternion considering double cover (q and -q)."""
    loss1 = F.mse_loss(pred_quat, target_quat)
    loss2 = F.mse_loss(pred_quat, -target_quat)
    return torch.min(loss1, loss2)


class HOILoss(nn.Module):
    """
    Multi-task MSE Loss for HOI.
    
    Components (weights from config):
    - hand_trans: Hand global translation (3)
    - hand_mano: MANO parameters (48)
    - hand_joints: Joints 3D relative (21x3)
    - obj_trans: Object translation (3)
    - obj_rot: Object rotation quaternion (4)
    - rel_trans: Relative hand-object translation (3) = hand_trans - obj_trans
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.loss_weights = config.train.hoi.loss_weights

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred: output from HOIModel.forward()
            target: ground truth from dataset
        Returns:
            dict with total loss and individual components
        """
        raise NotImplementedError("Implement loss computation")
