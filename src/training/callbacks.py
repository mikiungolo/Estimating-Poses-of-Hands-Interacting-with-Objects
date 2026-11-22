"""
Training Callbacks: EarlyStopping, LR Scheduler, ModelCheckpoint.
"""
from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class EarlyStopping:
    """Early stopping based on validation metric."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def create_scheduler(optimizer: torch.optim.Optimizer, config: DictConfig):
    """Create scheduler from config (cosine + warmup)."""
    train_cfg = config.train

    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    main_epochs = train_cfg.epochs - warmup_epochs
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=main_epochs,
        eta_min=train_cfg.lr * 0.01,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    return scheduler


class ModelCheckpoint:
    """Callback to save best model."""

    def __init__(self, checkpoint_dir: str, monitor: str = "val_loss", mode: str = "min"):
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.best_score = None

    def __call__(self, model, optimizer, epoch, metrics, config):
        raise NotImplementedError("Implement ModelCheckpoint")
