import torch

def calculate_psnr(pred, target, max_val=1.0):
    """
    Calcule le PSNR (Peak Signal-to-Noise Ratio) entre deux lots d'images (batchs).
    Plus le PSNR est élevé, plus l'image prédite est proche de la référence.
    
    Arguments:
    - pred : Les images générées par notre U-Net
    - target : Les images de référence (nettes)
    - max_val : La valeur maximale possible d'un pixel (1.0 car on utilise ToTensor)
    """
    # 1. On calcule l'Erreur Quadratique Moyenne (MSE)
    # C'est la moyenne des (prédiction - réalité) au carré
    mse = torch.mean((pred - target) ** 2)
    
    # Si le MSE est 0, les images sont parfaitement identiques, le PSNR est infini.
    if mse == 0:
        return torch.tensor(100.0).to(pred.device)
    
    # 2. Formule officielle du PSNR : 10 * log10( MAX^2 / MSE )
    # Équivalent mathématique : 20 * log10( MAX ) - 10 * log10( MSE )
    psnr = 20 * torch.log10(torch.tensor(max_val).to(pred.device)) - 10 * torch.log10(mse)
    
    return psnr