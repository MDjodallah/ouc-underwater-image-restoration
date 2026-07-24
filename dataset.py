import os # pour utiliser les fichier (importation, modification et recuperation de chemin)
from PIL import Image # pour ouvrir les images et les convertir en format RGB (3 canaux)
import torch
from torch.utils.data import Dataset # pour créer notre propre dataset (classe qui hérite de Dataset)
from typing import Optional, Callable # pour typer les arguments de notre classe
import torchvision.transforms as transforms # pour transformer les images
from config import MEAN, STD, CHEMIN_RAW, CHEMIN_REF # Centralisation depuis config.py

class Underwaterdataset(Dataset):
    
    # On définit le constructeur de notre classe 
    def __init__(self, raw_dir: str, ref_dir: str, transform: Optional[Callable] = None) -> None:
        self.raw_dir = raw_dir 
        self.ref_dir = ref_dir
        liste_image = os.listdir(raw_dir)
        self.image_names = [image for image in liste_image if image.endswith(('.png', '.jpg', '.jpeg'))]
        
        # On définit la transformation par défaut des images
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)), # 1. Redimensionnement
                transforms.ToTensor(),         # 2. Conversion en Tenseur (Pixels de 0 à 1) - DOIT ÊTRE AVANT NORMALIZE
                transforms.Normalize(mean=MEAN, std=STD) # 3. Normalisation (Centrage autour de 0)
            ])
        else:
            self.transform = transform
        
    def __len__(self):
        return len(self.image_names)
        
    def __getitem__(self, idx):
        # 1. On récupère le nom du fichier commun aux deux dossiers
        img_name = self.image_names[idx]
        
        # 2. On construit les chemins complets pour les deux versions (X et Y)
        raw_path = os.path.join(self.raw_dir, img_name)
        ref_path = os.path.join(self.ref_dir, img_name)
        
        # 3. On ouvre les deux images (conversion RGB obligatoire pour l'uniformité)
        raw_img = Image.open(raw_path).convert("RGB")
        ref_img = Image.open(ref_path).convert("RGB")
        
        # 4. On transforme les deux images en tenseurs
        raw_tensor = self.transform(raw_img)
        ref_tensor = self.transform(ref_img)
        
        # 5. On renvoie le couple : le modèle recevra (X, Y)
        return raw_tensor, ref_tensor
        
if __name__ == "__main__":
    try:
        # On crée notre "usine" avec les chemins de config.py
        mon_dataset = Underwaterdataset(CHEMIN_RAW, CHEMIN_REF)
        
        print("✅ Création du dataset avec succès !")
        print(f"📊 Nombre d'images trouvées : {len(mon_dataset)}")
        
        if len(mon_dataset) > 0:
            # On demande la toute première image (index 0)
            X, Y = mon_dataset[0]
            
            print("\n🔍 Test :")
            print(f"- Forme du tenseur RAW (X) : {X.shape}")
            print(f"- Forme du tenseur REF (Y) : {Y.shape}")
            print(f"- Extrait de la liste des images : {mon_dataset.image_names[:3]}...")
            print("\n🎉 Les tailles doivent être (ex: [3, 256, 256]), si oui le code fonctionne !")
            
    except Exception as e:
        print(f"❌ Une erreur s'est produite : {e}")
        print("Vérifie que les chemins vers les images sont corrects dans config.py.")