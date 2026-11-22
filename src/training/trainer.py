"""
Generic Training Loop for HaMeR/HOI.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.utils.checkpoint import resume_from_checkpoint


class Trainer:
    """
    Training loop with:
    - Mixed precision (AMP)
    - Gradient accumulation
    - Checkpoint saving
    - Resume support
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        config: DictConfig,
        device: torch.device,
        compute_metrics_fn: Callable = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.compute_metrics_fn = compute_metrics_fn

        self.scaler = torch.cuda.amp.GradScaler(enabled=config.train.get("mixed_precision", True))
        self.grad_accum_steps = config.train.get("grad_accum_steps", 1)

        self.checkpoint_dir = config.checkpoint_dir
        self.save_every = config.train.get("save_every", 5)

        self.start_epoch, _ = resume_from_checkpoint(
            self.checkpoint_dir, model, optimizer
        )

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """One training epoch."""
        raise NotImplementedError("Implement train_epoch")

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        """Validation loop."""
        raise NotImplementedError("Implement validate")

    def fit(self, epochs: int, run_name: str = None) -> None:
        """Full training loop."""
        raise NotImplementedError("Implement fit")
