import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
# This will silently fail if .env does not exist, which is fine for Colab where we might rely on default values
load_dotenv()

# --- HARDWARE CONFIGURATION ---
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
print(f"[INFO] Hardware accelerator selected: {DEVICE}")

# --- DATA PATHS ---
# Using os.getenv with fallbacks if .env is missing or not configured
CHEMIN_RAW = os.getenv("RAW_DATA_PATH", "/content/dataset_complet/raw")
CHEMIN_REF = os.getenv("REF_DATA_PATH", "/content/dataset_complet/ref")
MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "weights/unet_v3_gan_12k.pth")
ONNX_WEIGHTS_PATH = os.getenv("ONNX_WEIGHTS_PATH", "weights/generator.onnx")

# --- HYPERPARAMETERS ---
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.0001"))
EPOCHS = int(os.getenv("EPOCHS", "200"))
VAL_SPLIT = float(os.getenv("VAL_SPLIT", "0.1"))

# --- DATASET STATISTICS (Normalization) ---
# Parse comma-separated strings from .env into lists of floats
def parse_stats(env_var, default_value):
    val = os.getenv(env_var)
    if val:
        return [float(x.strip()) for x in val.split(',')]
    return default_value

# Default to V3 stats (12k dataset) if nothing is provided
MEAN = parse_stats("MEAN", [0.237, 0.476, 0.464])
STD = parse_stats("STD", [0.197, 0.239, 0.239])

print(f"[INFO] Config loaded | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")