import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import sys
import os

from core.model import Unet
from core.config import DEVICE, MEAN, STD

def corriger_image(chemin_image_floue, chemin_sauvegarde="image_corrigee.png"):
    print(f"Chargement du modèle sur {DEVICE}...")
    
    # 1. Chargement et préparation du modèle en mode évaluation
    model = Unet().to(DEVICE)
    chemin_poids = "generator_final.pth"
    
    if not os.path.exists(chemin_poids):
        print(f"Erreur : Le fichier de poids '{chemin_poids}' est introuvable.")
        return
        
    # Chargement des poids entraînés
    model.load_state_dict(torch.load(chemin_poids, map_location=DEVICE, weights_only=True))
    model.eval() # Désactivation du calcul des gradients pour l'inférence
    
    # 2. On prépare l'image d'entrée
    print(f"Traitement de l'image : {chemin_image_floue}")
    try:
        img_pil = Image.open(chemin_image_floue).convert("RGB")
    except Exception as e:
        print(f"Erreur d'ouverture de l'image : {e}")
        return

    # Application des mêmes transformations (Resize, Normalize) que lors de l'entraînement
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    # Ajout d'une dimension batch : [C, H, W] -> [1, C, H, W]
    image_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    
    # 3. Inférence
    print("Exécution de l'inférence...")
    with torch.no_grad(): # Désactivation d'Autograd pour réduire la consommation mémoire
        prediction = model(image_tensor)
        
    # 4. Dénormalisation et Sauvegarde
    mean_tensor = torch.tensor(MEAN).view(1, 3, 1, 1).to(DEVICE)
    std_tensor = torch.tensor(STD).view(1, 3, 1, 1).to(DEVICE)
    prediction_denorm = (prediction * std_tensor) + mean_tensor
    
    # Concaténation de l'image originale et corrigée pour comparaison visuelle
    image_originale_denorm = (image_tensor * std_tensor) + mean_tensor
    comparaison = torch.cat((image_originale_denorm[0], prediction_denorm[0]), dim=2) # dim=2 pour coller en largeur sans le batch
    
    save_image(comparaison, chemin_sauvegarde)
    print(f"Succès : Image restaurée sauvegardée vers {chemin_sauvegarde}")

if __name__ == "__main__":
    # Point d'entrée CLI pour tester une image
    if len(sys.argv) > 1:
        image_a_tester = sys.argv[1]
        corriger_image(image_a_tester)
    else:
        print("Usage : python inference.py <chemin_vers_ton_image.jpg>")
        print("Exemple de commande d'inférence :")
        # Fallback : test sur une image par défaut si aucun argument n'est fourni
        image_secours = "/content/uieb_data/raw/raw-890/10_img_.png"
        if os.path.exists(image_secours):
            corriger_image(image_secours)
