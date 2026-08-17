import torch

# --- DATA PATHS ---
# REPLACE THESE PATHS WITH YOUR LOCAL DIRECTORIES BEFORE RUNNING
# Do NOT commit your local absolute paths to GitHub!
CHEMIN_RAW = "path/to/your/raw/images"
CHEMIN_REF = "path/to/your/reference/images"

# --- TRAINING HYPERPARAMETERS ---
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 10

# --- HARDWARE (DEVICE) ---
# Automatically selects Apple Silicon (MPS), Nvidia GPU (CUDA), or CPU.
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

# --- DATASET STATISTICS (Normalization) ---
# These values represent the mean and standard deviation of the underwater images.
# Note: The Red channel (index 0) is typically lower due to water light absorption!
MEAN = [0.285, 0.421, 0.462]  # R, G, B
STD = [0.182, 0.165, 0.141]   # R, G, B
