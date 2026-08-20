import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

from core.dataset import Underwaterdataset
from core.model import Unet
from core.metrics import calculate_psnr 
from core.config import (CHEMIN_RAW, CHEMIN_REF, BATCH_SIZE, 
                    LEARNING_RATE, EPOCHS, VAL_SPLIT, DEVICE, MEAN, STD)

def train():
    # --- MODIFICATION V1 : Forcer le chemin vers les 890 images ---
    CHEMIN_RAW_V1 = "uieb_data/raw"
    CHEMIN_REF_V1 = "uieb_data/ref"
    
    # 1. Dataset Initialization (On utilise les nouveaux chemins)
    full_dataset = Underwaterdataset(CHEMIN_RAW_V1, CHEMIN_REF_V1)
    # 1. Dataset Initialization
    #full_dataset = Underwaterdataset(CHEMIN_RAW, CHEMIN_REF)
    
    # --- DYNAMIC TRAIN/VAL SPLIT ---
    total_size = len(full_dataset)
    # Calculate validation size (e.g., 10% of total)
    val_size = int(total_size * VAL_SPLIT)
    train_size = total_size - val_size 
    
    # Set random seed for deterministic splitting
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # Initialize Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Répartition du Dataset : Train={len(train_dataset)}, Val={len(val_dataset)}")
    # ----------------------------------------------

    # 2. Model, Loss, and Optimizer Initialization
    model = Unet().to(DEVICE)
    chemin_modele = "unet_underwater_model.pth"
    if os.path.exists(chemin_modele):
        print(f"[INFO] Loading existing model weights from {chemin_modele}...")
        model.load_state_dict(torch.load(chemin_modele, weights_only=True, map_location=DEVICE))
    else:
        print("[INFO] Initializing model weights from scratch.")

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Training Loop
    for epoch in range(EPOCHS):
        # ==========================================
        # PHASE 1: TRAINING
        # Forward pass, L1 Loss calculation, and backpropagation.
        # ==========================================
        model.train() 
        train_loss = 0.0
        
        for batch_idx, (raw_images, ref_images) in enumerate(train_loader):
            raw_images, ref_images = raw_images.to(DEVICE), ref_images.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(raw_images)
            loss = criterion(outputs, ref_images)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Save qualitative visual results on the first batch
            if batch_idx == 0:
                os.makedirs("test", exist_ok=True)
                mean_tensor = torch.tensor(MEAN).view(1, 3, 1, 1).to(DEVICE)
                std_tensor = torch.tensor(STD).view(1, 3, 1, 1).to(DEVICE)
                
                raw_denorm = (raw_images * std_tensor) + mean_tensor
                outputs_denorm = (outputs * std_tensor) + mean_tensor
                ref_denorm = (ref_images * std_tensor) + mean_tensor
                
                comparaison = torch.cat((raw_denorm[:1], outputs_denorm[:1], ref_denorm[:1]), dim=3)
                save_image(comparaison, f"test/comparaison_epoque_{epoch+1}.png")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # ==========================================
        # PHASE 2: VALIDATION
        # Evaluate model generalization on the hold-out set without updating weights.
        # ==========================================
        model.eval() 
        val_loss = 0.0
        val_psnr = 0.0
        
        with torch.no_grad(): 
            for val_raw, val_ref in val_loader:
                val_raw, val_ref = val_raw.to(DEVICE), val_ref.to(DEVICE)
                
                val_outputs = model(val_raw)
                val_loss += criterion(val_outputs, val_ref).item()
                
                # Calculate Peak Signal-to-Noise Ratio (PSNR) for evaluation
                val_psnr += calculate_psnr(val_outputs, val_ref).item()
                
        # Moyenne du PSNR sur l'ensemble de validation
        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr / len(val_loader)
        
        # Epoch Summary
        print(f"[Epoch {epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val PSNR: {avg_val_psnr:.2f} dB")
                
        # Save model checkpoint
        torch.save(model.state_dict(), chemin_modele)

if __name__ == "__main__":
    train()