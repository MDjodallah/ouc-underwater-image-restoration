import torch
from torch.utils.data import DataLoader
# Import strict de la classe pour éviter l'exécution du bloc de test de dataset.py
from core.dataset import Underwaterdataset
from tqdm import tqdm

def calculate_mean_std(dataset, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    mean_sum = torch.zeros(3)
    squared_mean_sum = torch.zeros(3)
    num_batches = 0
    
    print(f"Analyse des tenseurs en cours ({len(dataset)} images)...")
    for batch in tqdm(loader, desc="Calcul des statistiques"):
        raw_images, _ = batch 
        batch_mean = torch.mean(raw_images, dim=(0,2,3))
        batch_squared_mean = torch.mean(raw_images ** 2, dim=(0,2,3))
        
        mean_sum += batch_mean
        squared_mean_sum += batch_squared_mean
        num_batches += 1
        
    global_mean = mean_sum / num_batches
    global_squared_mean = squared_mean_sum / num_batches
    
    global_variance = global_squared_mean - (global_mean ** 2)
    global_std = torch.sqrt(global_variance)
    
    return global_mean, global_std

if __name__ == "__main__":
    from core.config import CHEMIN_RAW, CHEMIN_REF
    import torchvision.transforms as transforms
    
    try:
        print(f"Lecture des images depuis : {CHEMIN_RAW}")
        # Important : La normalisation est désactivée ici pour calculer les statistiques brutes réelles du dataset
        transform_sans_norm = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        dataset = Underwaterdataset(CHEMIN_RAW, CHEMIN_REF, transform=transform_sans_norm)
        
        print(f"Calcul en cours sur {len(dataset)} images...")
        mean, std = calculate_mean_std(dataset)
        
        print("\nSTATISTIQUES CALCULEES DU DATASET :")
        print(f"MEAN = [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]")
        print(f"STD = [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")