import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import Underwaterdataset
from model import Unet
from discriminator import Discriminator
from config import CHEMIN_RAW, CHEMIN_REF, BATCH_SIZE, EPOCHS, DEVICE, MEAN, STD

def train_gan():
    print(f"Initialisation du pipeline GAN sur {DEVICE}...")
    
    dataset = Underwaterdataset(CHEMIN_RAW, CHEMIN_REF)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 1. Initialisation des deux réseaux
    generator = Unet().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    
    # --- MÉCANIQUE DE REPRISE (ANTI-COUPURE INTERNET) ---
    if os.path.exists("generator_final.pth") and os.path.exists("discriminator_final.pth"):
        print("Chargement des poids existants (Générateur et Discriminateur)...")
        generator.load_state_dict(torch.load("generator_final.pth", map_location=DEVICE, weights_only=True))
        discriminator.load_state_dict(torch.load("discriminator_final.pth", map_location=DEVICE, weights_only=True))
    else:
        print("Initialisation des poids du réseau.")
    # ----------------------------------------------------
    
    # 2. Fonctions de perte (Losses)
    criterion_GAN = nn.MSELoss()
    criterion_pixelwise = nn.L1Loss()
    lambda_pixel = 100 # On donne 100x plus d'importance aux vraies couleurs
    
    # 3. Optimiseurs
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    print("Début de la phase d'entraînement adversarial.")
    
    for epoch in range(EPOCHS):
        for batch_idx, (raw_images, ref_images) in enumerate(train_loader):
            raw_images = raw_images.to(DEVICE)
            ref_images = ref_images.to(DEVICE)

            # ==========================================
            # ENTRAÎNEMENT DU GÉNÉRATEUR
            # ==========================================
            optimizer_G.zero_grad()
            gen_images = generator(raw_images)
            pred_fake = discriminator(gen_images)
            
            # CRITIQUE : Création d'étiquettes de la taille exacte du PatchGAN (16x16)
            valid = torch.ones_like(pred_fake).to(DEVICE)
            fake = torch.zeros_like(pred_fake).to(DEVICE)
            
            loss_G_GAN = criterion_GAN(pred_fake, valid)
            loss_G_pixel = criterion_pixelwise(gen_images, ref_images)
            loss_G = loss_G_GAN + lambda_pixel * loss_G_pixel
            
            loss_G.backward()
            optimizer_G.step()

            # ==========================================
            # ENTRAÎNEMENT DU DISCRIMINATEUR
            # ==========================================
            optimizer_D.zero_grad()
            pred_real = discriminator(ref_images)
            loss_D_real = criterion_GAN(pred_real, valid)
            
            pred_fake_d = discriminator(gen_images.detach())
            loss_D_fake = criterion_GAN(pred_fake_d, fake)
            
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimizer_D.step()

            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch+1}/{EPOCHS}] [Batch {batch_idx}/{len(train_loader)}] "
                      f"[Loss D: {loss_D.item():.4f}] [Loss G: {loss_G.item():.4f}]")
                
            if batch_idx == 0:
                os.makedirs("test_gan", exist_ok=True)
                mean_t = torch.tensor(MEAN).view(1, 3, 1, 1).to(DEVICE)
                std_t = torch.tensor(STD).view(1, 3, 1, 1).to(DEVICE)
                
                raw_denorm = (raw_images * std_t) + mean_t
                gen_denorm = (gen_images * std_t) + mean_t
                ref_denorm = (ref_images * std_t) + mean_t
                
                comparaison = torch.cat((raw_denorm[:1], gen_denorm[:1], ref_denorm[:1]), dim=3)
                save_image(comparaison, f"test_gan/gan_epoque_{epoch+1}.png")

        # Sauvegarde des modèles
        torch.save(generator.state_dict(), "generator_final.pth")
        torch.save(discriminator.state_dict(), "discriminator_final.pth")

if __name__ == "__main__":
    train_gan()