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

def corriger_image(chemin_image_floue, chemin_sauvegarde=None):
    if chemin_sauvegarde is None:
        os.makedirs("results", exist_ok=True)
        nom_base = os.path.splitext(os.path.basename(chemin_image_floue))[0]
        chemin_sauvegarde = os.path.join("results", f"image_corrigee_V2.png")
        
    print(f"Chargement du modèle sur {DEVICE}...")
    
    # 1. Chargement et préparation du modèle en mode évaluation
    # 1. Chargement du modèle
    model = Unet().to(DEVICE)
    chemin_poids = "weights/unet_gan_v2.pth"
    
    if not os.path.exists(chemin_poids):
        print(f"Erreur : Le fichier de poids '{chemin_poids}' est introuvable.")
        return
        
    # Chargement des poids entraînés
    model.load_state_dict(torch.load(chemin_poids, map_location=DEVICE, weights_only=True))
    # On empêche PyTorch de calculer les gradients (ça économise beaucoup de RAM pour la prédiction)
    model.eval()
    
    # 2. On prépare l'image d'entrée
    print(f"Traitement de l'image : {chemin_image_floue}")
    try:
        img_pil = Image.open(chemin_image_floue).convert("RGB")
    except Exception as e:
        print(f"Erreur d'ouverture de l'image : {e}")
        return

    # 3. On applique les mêmes transformations que pendant l'entraînement de la V2
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    # On ajoute la dimension batch pour que le modèle l'accepte
    image_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    
    # 4. Le modèle fait sa magie
    with torch.no_grad(): # Pas besoin de gradients ici non plus
        prediction = model(image_tensor)
        
    # 5. On remet les pixels à leur échelle normale pour sauvegarder l'image
    mean_tensor = torch.tensor(MEAN).view(1, 3, 1, 1).to(DEVICE)
    std_tensor = torch.tensor(STD).view(1, 3, 1, 1).to(DEVICE)
    prediction_denorm = (prediction * std_tensor) + mean_tensor
    
    # On colle l'image d'origine et la nouvelle côte à côte pour bien voir la différence
    image_originale_denorm = (image_tensor * std_tensor) + mean_tensor
    comparaison = torch.cat((image_originale_denorm[0], prediction_denorm[0]), dim=2)
    
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
        image_secours = "tests/test1.jpeg"
        if os.path.exists(image_secours):
            corriger_image(image_secours)
        else:
            print(f"L'image {image_secours} n'existe pas dans ce dossier.")
