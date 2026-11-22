"""
Config loading utilities.
"""
from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str = "configs/config.yaml") -> DictConfig:
    """Load YAML config with OmegaConf. Resolves env vars ${oc.env:VAR, default}."""
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)
    return config


def merge_config(base: DictConfig, overrides: dict) -> DictConfig:
    """Merge base config with overrides (e.g., from CLI)."""
    return OmegaConf.merge(base, overrides)


def save_config(config: DictConfig, path: str) -> None:
    """Save config to YAML file."""
    OmegaConf.save(config, path)
