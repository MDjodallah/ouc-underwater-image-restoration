import os 
from PIL import Image 
import torch
from torch.utils.data import Dataset 
from typing import Optional, Callable 
import torchvision.transforms as transforms 
from core.config import MEAN, STD, CHEMIN_RAW, CHEMIN_REF 

class Underwaterdataset(Dataset):
    
    # Initialisation de la classe de Dataset personnalisé
    def __init__(self, raw_dir: str, ref_dir: str, transform: Optional[Callable] = None) -> None:
        self.raw_dir = raw_dir 
        self.ref_dir = ref_dir
        liste_image = os.listdir(raw_dir)
        self.image_names = [image for image in liste_image if image.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Pipeline de transformation par défaut (Redimensionnement, Tensorisation, Normalisation)
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
        # 1. Extraction du nom de fichier commun
        img_name = self.image_names[idx]
        
        # 2. Construction des chemins absolus (Entrée dégradée X et Référence Y)
        raw_path = os.path.join(self.raw_dir, img_name)
        ref_path = os.path.join(self.ref_dir, img_name)
        
        # 3. Chargement et standardisation en RGB (3 canaux)
        raw_img = Image.open(raw_path).convert("RGB")
        ref_img = Image.open(ref_path).convert("RGB")
        
        # 4. Application du pipeline de transformation PyTorch
        raw_tensor = self.transform(raw_img)
        ref_tensor = self.transform(ref_img)
        
        # 5. Renvoi du tuple pour l'entraînement (Input, Target)
        return raw_tensor, ref_tensor
        
if __name__ == "__main__":
    try:
        # Instanciation du dataset de test
        mon_dataset = Underwaterdataset(CHEMIN_RAW, CHEMIN_REF)
        
        print("Création du dataset avec succès !")
        print(f"Nombre d'images trouvées : {len(mon_dataset)}")
        
        if len(mon_dataset) > 0:
            # Extraction du premier élément
            X, Y = mon_dataset[0]
            
            print("\nTest de structure :")
            print(f"- Forme du tenseur RAW (X) : {X.shape}")
            print(f"- Forme du tenseur REF (Y) : {Y.shape}")
            print(f"- Extrait de la liste des images : {mon_dataset.image_names[:3]}...")
            print("\nValidation réussie : Les dimensions des tenseurs sont correctes.")
            
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
        print("Vérifie que les chemins vers les images sont corrects dans config.py.")