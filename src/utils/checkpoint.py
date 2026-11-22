"""
Checkpoint save/load utilities with resume support.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf


def save_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    config: DictConfig,
    name: str = None,
) -> str:
    """Save full checkpoint. Returns saved file path."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if name is None:
        name = f"epoch_{epoch:04d}.pth"

    path = Path(checkpoint_dir) / name

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": OmegaConf.to_container(config, resolve=True),
    }, path)

    return str(path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    map_location: str = "cpu",
) -> tuple[int, dict]:
    """Load checkpoint. Returns (epoch, metrics)."""
    ckpt = torch.load(checkpoint_path, map_location=map_location)

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    epoch = ckpt.get("epoch", 0)
    metrics = ckpt.get("metrics", {})

    return epoch, metrics


def find_latest_checkpoint(checkpoint_dir: str, pattern: str = "epoch_*.pth") -> str | None:
    """Find latest checkpoint for resume."""
    files = glob.glob(os.path.join(checkpoint_dir, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def resume_from_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    pattern: str = "epoch_*.pth",
) -> tuple[int, dict]:
    """Auto-resume from latest checkpoint. Returns (start_epoch, metrics)."""
    latest = find_latest_checkpoint(checkpoint_dir, pattern)

    if latest is None:
        print("No checkpoint found, starting from scratch.")
        return 0, {}

    print(f"Resuming from {latest}")
    epoch, metrics = load_checkpoint(latest, model, optimizer)
    return epoch + 1, metrics
