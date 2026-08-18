import os
import torch

# --- DATA PATHS & HYPERPARAMETERS ---
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 200
VAL_SPLIT = 0.1 
if os.path.exists('/content'):
    print("[INFO] Google Colab environment detected.")
    CHEMIN_RAW = "/content/dataset_complet/raw"
    CHEMIN_REF = "/content/dataset_complet/ref"
    
elif os.path.exists('/Users/moutassim'):
    print("[INFO] Local Environment (Mac M4) detected.")
    CHEMIN_RAW = "/Users/moutassim/Documents/Cours/Internationale/OUC/Underwater/dataset/raw"
    CHEMIN_REF = "/Users/moutassim/Documents/Cours/Internationale/OUC/Underwater/dataset/ref"
    
else:
    print("[INFO] Remote Server Environment detected.")
    CHEMIN_RAW = "./dataset_complet/raw"
    CHEMIN_REF = "./dataset_complet/ref"
    BATCH_SIZE = 16

# --- HARDWARE CONFIGURATION ---
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
print(f"[INFO] Hardware accelerator selected for training: {DEVICE}")

# --- DATASET STATISTICS (Normalization) ---
# [CRITICAL] The normalization stats MUST match the dataset the model was trained on!
# If you are evaluating a V1 or V2 model, uncomment the 890-images stats.
# If you are evaluating a V3 model, use the 12k-images stats.

# ---------------------------------------------------------
# OPTION A: Stats for V1 & V2 Models (890 images dataset)
# ---------------------------------------------------------
# MEAN = [0.269, 0.491, 0.496]
# STD = [0.213, 0.197, 0.216]

# ---------------------------------------------------------
# OPTION B: Stats for V3 Models (12k images full dataset)
# ---------------------------------------------------------
MEAN = [0.237, 0.476, 0.464]
STD = [0.197, 0.239, 0.239]