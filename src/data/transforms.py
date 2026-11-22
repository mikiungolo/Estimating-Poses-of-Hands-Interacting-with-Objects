"""
Data Augmentation for HOI training.

Project spec: "Color Jitter and Gaussian Blur"
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from omegaconf import DictConfig


class ColorJitter:
    """Color jitter for RGB images (brightness, contrast, saturation, hue)."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
    ):
        self.transform = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8)
        pil_img = TF.to_pil_image(rgb)
        pil_img = self.transform(pil_img)
        return np.array(pil_img).astype(np.float32) / 255.0


class GaussianBlur:
    """Gaussian blur for RGB images."""

    def __init__(
        self,
        kernel_size: int = 3,
        sigma: tuple[float, float] = (0.1, 2.0),
    ):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8)
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        return cv2.GaussianBlur(rgb, (self.kernel_size, self.kernel_size), sigma).astype(np.float32) / 255.0


class ToTensor:
    """Convert numpy (H,W,C) -> torch (C,H,W) float32 [0,1]."""

    def __call__(self, rgb: np.ndarray) -> torch.Tensor:
        if rgb.dtype != np.float32:
            rgb = rgb.astype(np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1)


class Normalize:
    """Normalize with ImageNet mean/std."""

    def __init__(
        self,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std


def get_transforms(split: str, config: DictConfig) -> T.Compose:
    """
    Get transform pipeline for split.
    
    Args:
        split: "train" | "val" | "test"
        config: Config data.augment section
    
    Returns:
        Compose of transforms
    """
    augment = config.data.augment

    if split == "train":
        transforms = []

        if augment.get("color_jitter", False):
            transforms.append(ColorJitter(
                brightness=augment.get("jitter_brightness", 0.2),
                contrast=augment.get("jitter_contrast", 0.2),
                saturation=augment.get("jitter_saturation", 0.2),
                hue=augment.get("jitter_hue", 0.1),
            ))

        if augment.get("gaussian_blur", False):
            transforms.append(GaussianBlur(
                kernel_size=augment.get("blur_kernel_size", 3),
                sigma=augment.get("blur_sigma", (0.1, 2.0)),
            ))

        transforms.extend([ToTensor(), Normalize()])

    else:
        transforms = [ToTensor(), Normalize()]

    return T.Compose(transforms)
