import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import Underwaterdataset
from model import Unet
from config import DEVICE, MEAN, STD, CHEMIN_RAW, CHEMIN_REF, BATCH_SIZE
from metrics import calculate_psnr 
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def tensor_to_numpy(tensor):
    """
    Convertit un tenseur PyTorch normalisé en une image Numpy classique (0 à 255)
    pour pouvoir calculer le SSIM.
    """
    # 1. Dénormalisation
    mean_t = torch.tensor(MEAN).view(1, 3, 1, 1).to(tensor.device)
    std_t = torch.tensor(STD).view(1, 3, 1, 1).to(tensor.device)
    img = (tensor * std_t) + mean_t
    img = torch.clamp(img, 0, 1)
    
    # 2. Conversion format [Hauteur, Largeur, Couleurs]
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)

def evaluate_model():
    print(f"Début de l'évaluation des métriques (PSNR/SSIM) sur {DEVICE}...")
    
    # 1. Chargement du modèle V2
    model = Unet().to(DEVICE)
    try:
        model.load_state_dict(torch.load("generator_final.pth", map_location=DEVICE, weights_only=True))
        print("Poids du modèle chargés avec succès.")
    except:
        print("Erreur : Fichier de poids generator_final.pth introuvable.")
        return
        
    model.eval() # Mode évaluation
    
    # 2. Préparation des données
    dataset = Underwaterdataset(CHEMIN_RAW, CHEMIN_REF)
    loader = DataLoader(dataset, batch_size=1, shuffle=False) # Batch size = 1 pour itération fine
    
    total_psnr = 0.0
    total_ssim = 0.0
    
    with torch.no_grad(): # Désactivation d'Autograd pour accélérer le processus
        for raw_img, ref_img in tqdm(loader, desc="Analyse des images"):
            raw_img = raw_img.to(DEVICE)
            ref_img = ref_img.to(DEVICE)
            
            # Prédiction de l'IA
            prediction = model(raw_img)
            
            # Calcul du PSNR (en utilisant la fonction de ton metrics.py)
            current_psnr = calculate_psnr(prediction, ref_img).item()
            total_psnr += current_psnr
            
            # Calcul du SSIM (Nécessite de repasser en format classique Numpy)
            pred_np = tensor_to_numpy(prediction)
            ref_np = tensor_to_numpy(ref_img)
            
            # SSIM sur 3 canaux (RGB), plage de valeurs 0-255
            current_ssim = ssim(pred_np, ref_np, channel_axis=2, data_range=255)
            total_ssim += current_ssim

    avg_psnr = total_psnr / len(loader)
    avg_ssim = total_ssim / len(loader)
    
    print("\n" + "="*40)
    print("RÉSULTATS DE L'ÉVALUATION :")
    print(f"Images évaluées : {len(dataset)}")
    print(f"PSNR Moyen : {avg_psnr:.2f} dB")
    print(f"SSIM Moyen : {avg_ssim:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate_model()