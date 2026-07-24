import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Underwaterdataset
from model import Unet
from torchvision.utils import save_image 

# 1. Configuration (C'est ici qu'on mettra notre config.py plus tard)
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 100

# 2. Préparation du terrain (MISE À JOUR POUR LE MAC M4 !)
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps') # <-- C'est ici qu'on active le GPU de ton Mac M4 !
else:
    device = torch.device('cpu')
dataset = Underwaterdataset("raw/", "ref/")
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 3. Initialisation des ingrédients
model = Unet().to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"🚀 Entraînement lancé sur {device}...")

for _ in range(EPOCHS):
    for batch_idx, (input_images, target_images) in enumerate(train_loader):
        # Déplacement des données sur le bon device
        input_images = input_images.to(device)
        target_images = target_images.to(device)

        # Étape 1 : On efface les gradients du pas précédent
        optimizer.zero_grad()

        # Étape 2 : On fait passer les images d'entrée dans le modèle
        output_images = model(input_images)

        # Étape 3 : On calcule la perte entre la sortie et la cible
        loss = criterion(output_images, target_images)

        # Étape 4 : On calcule les gradients
        loss.backward()

        # Étape 5 : On met à jour les poids du modèle
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch [{_+1}/{EPOCHS}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # --- OBSERVABILITÉ (Voir, c'est croire !) ---
    # On sauvegarde l'image générée à la fin de chaque époque pour voir l'évolution
    if _ % 10 == 0:  # On ne sauvegarde pas à chaque époque pour ne pas saturer le disque
        save_image(output_images, f"test/prediction_epoque_{_+1}.png")
### Ton travail pour la suite
# Maintenant, il manque **la boucle d'entraînement**. C'est le cœur battant du programme. En PyTorch, une boucle d'entraînement suit toujours ce rituel sacré (le "Train Step") :

# 1.  **`optimizer.zero_grad()`** : On efface les traces du calcul précédent (sinon ça s'accumule et ça bug).
# 2.  **`output = model(input)`** : Le modèle prédit une image.
# 3.  **`loss = criterion(output, target)`** : Le juge compare la prédiction avec la réalité.
# 4.  **`loss.backward()`** : C'est la magie ! PyTorch calcule automatiquement de combien chaque poids du réseau doit changer pour réduire l'erreur.
# 5.  **`optimizer.step()`** : Le coach applique les changements aux poids.

# **Ton défi :** 
# Essaye d'écrire une boucle `for` qui parcourt tes `epochs` (ex: 10 tours) et à l'intérieur, une boucle `for` qui parcourt ton `train_loader`. 

# Auras-tu besoin d'aide pour imbriquer les deux boucles `for`, ou est-ce que tu te sens de tenter l'écriture de cette boucle ? (Indice : une boucle pour les époques, une boucle pour les batchs).