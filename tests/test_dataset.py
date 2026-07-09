import pytest
import torch
import numpy as np
from src.data.dataset import MedicalSegmentationDataset

def test_dataset_synthetic_generation():
    dataset = MedicalSegmentationDataset(
        generate_synthetic=True,
        num_synthetic_samples=5,
        image_size=128,
        is_train=True
    )
    
    assert len(dataset) == 5
    
    # Fetch item
    sample = dataset[0]
    assert "image" in sample
    assert "mask" in sample
    
    # Check shapes
    # image shape: [3, H, W]
    # mask shape: [1, H, W]
    assert sample["image"].shape == (3, 128, 128)
    assert sample["mask"].shape == (1, 128, 128)
    
    # Values check
    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["mask"], torch.Tensor)
    assert sample["mask"].max() <= 1.0
    assert sample["mask"].min() >= 0.0

def test_dataset_eval_transforms():
    dataset = MedicalSegmentationDataset(
        generate_synthetic=True,
        num_synthetic_samples=2,
        image_size=128,
        is_train=False
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 128, 128)
