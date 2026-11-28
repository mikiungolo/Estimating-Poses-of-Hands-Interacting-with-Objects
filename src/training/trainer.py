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
        checkpoint_dir: str | None = None,
        train_cfg: DictConfig | None = None,
    ):
        """
        Args:
            checkpoint_dir: where THIS model saves/resumes its checkpoints, e.g.
                f"{config.checkpoint_dir}/{config.train.hoi.checkpoint_subdir}".
                One folder per model, so a run never resumes from another model.
            train_cfg: training section of this model, e.g. config.train.hoi
                (config.train holds one sub-section per model).
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.compute_metrics_fn = compute_metrics_fn

        self.train_cfg = train_cfg if train_cfg is not None else config.train
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.train_cfg.get("mixed_precision", True))
        self.grad_accum_steps = self.train_cfg.get("grad_accum_steps", 1)

        self.checkpoint_dir = checkpoint_dir or config.checkpoint_dir
        self.save_every = self.train_cfg.get("save_every", 5)

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
