import torch
import pytest
import os
import sys

# Ensure the core module is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import Unet
from core.discriminator import Discriminator

def test_unet_output_shape():
    """
    Test that the U-Net Generator outputs a tensor of the exact same 
    spatial dimensions as the input tensor.
    """
    model = Unet()
    model.eval()
    
    # Batch size 2, 3 channels (RGB), 256x256 image
    dummy_input = torch.randn(2, 3, 256, 256)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    assert output.shape == (2, 3, 256, 256), f"Expected (2, 3, 256, 256), got {output.shape}"
    assert torch.max(output) <= 1.0, "Output values should be bounded by Tanh (max 1.0 for normalized)"
    assert torch.min(output) >= -1.0, "Output values should be bounded by Tanh (min -1.0 for normalized)"

def test_discriminator_output_shape():
    """
    Test that the PatchGAN Discriminator outputs a patch probability map.
    For a 256x256 input, the typical PatchGAN output is a 16x16 grid (if using 4 layers).
    """
    model = Discriminator()
    model.eval()
    
    dummy_input = torch.randn(2, 3, 256, 256)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    # Discriminator should output a probability map, not a single scalar
    # 256 / 2^4 = 16. The patch size grid is 16x16.
    assert output.shape == (2, 1, 16, 16), f"Expected PatchGAN grid (2, 1, 16, 16), got {output.shape}"
