import os
import sys

# Ensure the core module is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from core.model import Unet

# 1. Load the PyTorch model
device = "cpu"
model = Unet().to(device)
weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "weights", "unet_v3_gan_12k.pth")
model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
model.eval()

# 2. Create a dummy input tensor (e.g., 512x512 resolution)
dummy_input = torch.randn(1, 3, 512, 512, device=device)

# 3. Export to ONNX format
torch.onnx.export(
    model, 
    dummy_input, 
    "generator.onnx", 
    export_params=True, 
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                  'output': {0: 'batch_size', 2: 'height', 3: 'width'}}
)
print("[SUCCESS] Model successfully exported to ONNX format!")
