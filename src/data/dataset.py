"""
DexYCB Dataset Class for Hand-Object Interaction.

Expected data structure (DexYCB subset 10%):
- RGB images: (H, W, 3) uint8
- Depth images: (H, W) float32 (mm)
- MANO params: (48,) - 45 pose + 3 global orient
- Joints 2D: (21, 2) - pixel coordinates
- Joints 3D: (21, 3) - camera coordinates (mm)
- Object 6D pose: rotation (3,3) or quaternion (4,), translation (3,)
- Object class: int [0-19] (20 YCB classes)
- Camera intrinsics: (3, 3)
- Contact/active object flag: bool

Active Object Definition:
"An active object is strictly defined as an object in direct contact 
with the hand in a given frame. Frames without active contact must 
be identified and handled accordingly."
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset


class DexYCBDataset(Dataset):
    """
    DexYCB Dataset for HOI.
    
    Args:
        split: "train" | "val" | "test"
        config: OmegaConf config (data section)
        transform: Transform to apply (from get_transforms)
        data_root: Override dataset path (default from config)
    """

    def __init__(
        self,
        split: str,
        config: DictConfig,
        transform=None,
        data_root: str | None = None,
    ):
        assert split in ("train", "val", "test"), f"Invalid split: {split}"

        self.split = split
        self.config = config
        self.transform = transform

        self.data_root = Path(data_root or config.data_root)
        self.splits_dir = Path(config.project_root) / "dataset" / "splits"

        self._load_annotations()
        self._load_split_indices()

        self.camera_intrinsics = self._get_camera_intrinsics()

        print(f"DexYCBDataset({split}): {len(self)} samples")

    def _load_annotations(self) -> None:
        """Load annotations (pkl/json/npy from raw dataset)."""
        raise NotImplementedError("Implement DexYCB annotation loading")

    def _load_split_indices(self) -> None:
        """Load split indices (or create if first time)."""
        raise NotImplementedError("Implement load/create split indices")

    def _get_camera_intrinsics(self) -> np.ndarray:
        """Return camera intrinsics matrix (3x3)."""
        raise NotImplementedError("Implement camera intrinsics loading")

    def _is_active_object(self, idx: int) -> bool:
        """
        Check if object in frame is active (in contact with hand).
        
        Logic: distance hand vertices - object mesh < threshold (e.g., 5mm)
        Or use contact flag if present in annotations.
        """
        raise NotImplementedError("Implement active object logic")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns dict with:
        {
            'rgb': Tensor[3, H, W] float32 [0,1],
            'depth': Tensor[1, H, W] float32 (mm),
            'mano_params': Tensor[48] float32,
            'joints_2d': Tensor[21, 2] float32,
            'joints_3d': Tensor[21, 3] float32,
            'obj_rot': Tensor[3, 3] or Tensor[4] (quat) float32,
            'obj_trans': Tensor[3] float32,
            'obj_class': Tensor[] int64,
            'camera_intrinsics': Tensor[3, 3] float32,
            'is_active': Tensor[] bool,
            'sample_idx': int,
        }
        """
        raise NotImplementedError("Implement __getitem__")

    def get_sample_info(self, idx: int) -> dict[str, Any]:
        """Metadata for debugging/visualization (no transform)."""
        raise NotImplementedError("Implement get_sample_info")


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Custom collate for variable size tensors if needed."""
    return torch.utils.data.default_collate(batch)
