"""
Split logic for DexYCB: train/val/test 80/10/10 deterministic.
Indices saved in dataset/splits/ as .npy (committable).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import DictConfig


def create_splits(
    dataset_len: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    splits_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create deterministic splits and save indices.
    
    Returns:
        (train_indices, val_indices, test_indices)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    rng = np.random.default_rng(seed)
    indices = np.arange(dataset_len)
    rng.shuffle(indices)

    n_train = int(dataset_len * train_ratio)
    n_val = int(dataset_len * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    if splits_dir:
        splits_dir = Path(splits_dir)
        splits_dir.mkdir(parents=True, exist_ok=True)
        np.save(splits_dir / "train_indices.npy", train_idx)
        np.save(splits_dir / "val_indices.npy", val_idx)
        np.save(splits_dir / "test_indices.npy", test_idx)
        print(f"Splits saved to {splits_dir}")
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    return train_idx, val_idx, test_idx


def load_splits(splits_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load saved split indices."""
    splits_dir = Path(splits_dir)
    train_idx = np.load(splits_dir / "train_indices.npy")
    val_idx = np.load(splits_dir / "val_indices.npy")
    test_idx = np.load(splits_dir / "test_indices.npy")
    return train_idx, val_idx, test_idx


def get_split_indices(
    dataset_len: int,
    config: DictConfig,
    splits_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get split indices: load if exist, else create.
    """
    if splits_dir is None:
        splits_dir = Path(config.project_root) / "dataset" / "splits"

    splits_dir = Path(splits_dir)
    train_file = splits_dir / "train_indices.npy"

    if train_file.exists():
        print(f"Loading splits from {splits_dir}")
        return load_splits(splits_dir)
    else:
        print(f"Creating new splits in {splits_dir}")
        return create_splits(
            dataset_len=dataset_len,
            train_ratio=config.data.train_split,
            val_ratio=config.data.val_split,
            test_ratio=config.data.test_split,
            seed=config.seed,
            splits_dir=splits_dir,
        )
