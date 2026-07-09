import pytest
import torch
from src.models.unet import UNet
from src.models.segformer import SegFormerWrapper

def test_unet_output_shape():
    model = UNet(in_channels=3, num_classes=1, bilinear=True)
    dummy_input = torch.randn(1, 3, 256, 256)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    assert output.shape == (1, 1, 256, 256)

def test_segformer_output_shape():
    model = SegFormerWrapper(
        pretrained_model_name="nvidia/mit-b0",
        num_classes=1,
        from_pretrained=False
    )
    dummy_input = torch.randn(1, 3, 128, 128)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    assert output.shape == (1, 1, 128, 128)
