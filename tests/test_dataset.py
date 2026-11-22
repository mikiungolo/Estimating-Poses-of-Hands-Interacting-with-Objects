"""
Tests for DexYCBDataset.
"""
import pytest
from omegaconf import OmegaConf

from src.data import DexYCBDataset, get_transforms


class TestDexYCBDataset:
    """Test dataset loading and transforms."""

    @pytest.fixture
    def config(self):
        return OmegaConf.create({
            "data": {
                "batch_size": 4,
                "num_workers": 0,
                "image_size": 256,
                "augment": {
                    "color_jitter": True,
                    "gaussian_blur": True,
                }
            },
            "project_root": ".",
            "data_root": "./dataset/raw",
        })

    def test_transforms_train(self, config):
        transform = get_transforms("train", config)
        assert transform is not None

    def test_transforms_val(self, config):
        transform = get_transforms("val", config)
        assert transform is not None

    @pytest.mark.skipif(True, reason="Requires downloaded dataset")
    def test_dataset_load(self, config):
        dataset = DexYCBDataset("train", config)
        assert len(dataset) > 0

        sample = dataset[0]
        assert "rgb" in sample
        assert "mano_params" in sample
        assert sample["rgb"].shape == (3, 256, 256)
