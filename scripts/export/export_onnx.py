import os
import sys

# Ensure the core module is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from core.model import Unet

# 1. Chargement du modèle PyTorch
device = "cpu"
model = Unet().to(device)
weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "weights", "generator_final.pth")
model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
model.eval()

# 2. Créer un tenseur d'entrée fictif (résolution 512x512 par exemple)
dummy_input = torch.randn(1, 3, 512, 512, device=device)

# 3. Exporter au format ONNX
out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "weights", "generator_final.onnx")
torch.onnx.export(
    model, 
    dummy_input, 
    out_path, 
    export_params=True, 
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                  'output': {0: 'batch_size', 2: 'height', 3: 'width'}}
)
print("[SUCCESS] Model successfully exported to ONNX format!")
