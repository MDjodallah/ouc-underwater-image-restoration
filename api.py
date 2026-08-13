from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import math
import gc
from model import Unet
from config import DEVICE, MEAN, STD

# --- OPTIMISATION EXTRÊME DE MÉMOIRE POUR RENDER (512 Mo RAM) ---
# Limiter PyTorch à un seul thread CPU pour éviter l'explosion de RAM
torch.set_num_threads(1)

app = FastAPI(title="OUC Underwater Enhancement API")

# Configuration CORS pour autoriser l'application web frontend à communiquer avec l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Accepte toutes les requêtes (pratique pour le dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement du modèle en mémoire au démarrage du serveur
print("Chargement du modèle Générateur (U-Net V2)...")
model = Unet().to(DEVICE)
# Fichier contenant les poids entraînés du modèle
chemin_poids = "generator_final.pth" 
try:
    model.load_state_dict(torch.load(chemin_poids, map_location=DEVICE, weights_only=True))
    model.eval()
    print("Modèle chargé et prêt pour l'inférence !")
except FileNotFoundError:
    print(f"Avertissement : Fichier de poids introuvable ({chemin_poids}). Le modèle renverra des tenseurs vides.")

def get_multiple_of_16(val):
    """Arrondit une valeur au multiple de 16 supérieur le plus proche."""
    return int(math.ceil(val / 16.0)) * 16

# Endpoint de Ping (Utilisé par UptimeRobot pour empêcher le serveur de s'endormir)
@app.get("/")
async def ping():
    return {"status": "online", "message": "DeepLens API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Reçoit l'image du navigateur web, la passe dans le modèle, et renvoie l'image corrigée.
    """
    # 1. Lecture de l'image envoyée par le navigateur
    image_data = await file.read()
    image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # --- Gestion intelligente de la résolution ---
    # L'objectif est de traiter l'image en haute qualité sans saturer la mémoire (RAM) du serveur.
    # On fixe une limite (ex: 800px). Si l'image est plus grande, on la réduit proportionnellement
    # tout en conservant son ratio (format) d'origine.
    w, h = image_pil.size
    max_size = 800
    if max(w, h) > max_size:
        ratio = max_size / float(max(w, h))
        w, h = int(w * ratio), int(h * ratio)
        
    # Contrainte architecturale : Le modèle U-Net effectue 4 opérations de division par 2 
    # (poolings) dans son encodeur (2^4 = 16). Les dimensions de l'image d'entrée 
    # doivent donc obligatoirement être des multiples de 16 pour pouvoir être reconstruites.
    new_w = get_multiple_of_16(w)
    new_h = get_multiple_of_16(h)
    
    # 2. Préparation pour le modèle
    transform = transforms.Compose([
        transforms.Resize((new_h, new_w)), # Redimensionnement aux multiples de 16
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    img_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
    
    # 3. Inférence du modèle
    with torch.no_grad():
        prediction = model(img_tensor)
        
    # 4. Dénormalisation
    # Le modèle renvoie un tenseur avec des valeurs mathématiques potentiellement négatives (dues à la normalisation).
    # On doit annuler cette normalisation (multiplier par l'écart-type et additionner la moyenne)
    # pour retrouver des vraies valeurs de couleurs comprises entre 0 et 1.
    mean_t = torch.tensor(MEAN).view(1, 3, 1, 1).to(DEVICE)
    std_t = torch.tensor(STD).view(1, 3, 1, 1).to(DEVICE)
    prediction_denorm = (prediction * std_t) + mean_t
    prediction_denorm = torch.clamp(prediction_denorm, 0, 1)
    
    # 5. Conversion du Tenseur vers un fichier Image JPEG en mémoire
    img_array = prediction_denorm.squeeze(0).cpu()
    img_pil_result = transforms.ToPILImage()(img_array)
    
    buf = io.BytesIO()
    img_pil_result.save(buf, format="JPEG")
    buf.seek(0)
    
    # 6. Renvoi de l'image traitée au client
    response = Response(content=buf.getvalue(), media_type="image/jpeg")
    
    # --- NETTOYAGE FORCÉ DE LA MÉMOIRE ---
    # Supprimer les tenseurs lourds manuellement pour éviter un crash OOM
    del img_tensor, prediction, prediction_denorm, img_array, img_pil_result
    gc.collect()
    
    return response

# Commande pour lancer le serveur local :
# uvicorn api:app --reload