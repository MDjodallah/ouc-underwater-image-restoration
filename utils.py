import torch
from dataset import Underwaterdataset
from torch.utils.data import DataLoader
def calculate_mean_std(dataset, batch_size=32):
    # Le DataLoader va piocher dans ton dataset et te donner des batchs de 32 images
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # On initialise des tenseurs de taille 3 (pour R, G, B) à zéro
    mean_sum = torch.zeros(3)
    squared_mean_sum = torch.zeros(3)
    num_batches = 0
    
    for batch in loader:
        # Rappel : ton dataset renvoie (raw_tensor, ref_tensor)
        # On va calculer les stats uniquement sur les images dégradées (raw)
        raw_images, _ = batch 
        
        # raw_images a la dimension [B, C, H, W]
        
        # 1. Calcule la moyenne de ce batch pour les 3 canaux (utilise la bonne dimension !)
        batch_mean = torch.mean(raw_images, dim=(0,2,3)) # COMPLÈTE ICI
        
        # 2. Calcule la moyenne des CARRÉS de ce batch pour les 3 canaux
        # Indice : tu dois d'abord élever raw_images au carré
        batch_squared_mean = torch.mean(raw_images ** 2, dim=(0,2,3)) # COMPLÈTE ICI
        
        # On accumule
        mean_sum += batch_mean
        squared_mean_sum += batch_squared_mean
        num_batches += 1
        
    # On divise par le nombre total de batchs pour avoir la moyenne globale
    global_mean = mean_sum / num_batches
    global_squared_mean = squared_mean_sum / num_batches
    
    # 3. Calcule la variance globale puis l'écart-type (std)
    # Rappel : Variance = moyenne(X²) - moyenne(X)²
    # L'écart-type est la racine carrée (torch.sqrt) de la variance
    global_variance = global_squared_mean - global_mean ** 2 # COMPLÈTE ICI
    global_std = torch.sqrt(global_variance)
    
    return global_mean, global_std

if __name__ == "__main__":
    try:
        CHEMIN_RAW = "/Users/moutassim/Documents/Cours/OUC/Underwater/raw"
        CHEMIN_REF = "/Users/moutassim/Documents/Cours/OUC/Underwater/ref"
        # Utilisation :
        dataset = Underwaterdataset(CHEMIN_RAW, CHEMIN_REF)
        mean, std = calculate_mean_std(dataset)
        print("\nTest :")
        print("Mean:", mean)
        print("Std:", std)
    except Exception as e:
        print(f"❌ Une erreur s'est produite : {e}")