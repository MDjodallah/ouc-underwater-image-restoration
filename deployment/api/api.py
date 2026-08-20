from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import math
import gc
import numpy as np
import torch
import onnxruntime as ort
import os

from core.config import ONNX_WEIGHTS_PATH, ONNX_FP16_WEIGHTS_PATH
from core.model import Unet

app = FastAPI(title="OUC Underwater API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- VARIABLES GLOBALES ---
# On garde les modèles en mémoire pour éviter de les recharger à chaque requête
loaded_onnx_sessions = {}
loaded_pt_models = {}

ONNX_PATHS = {
    "v1": "weights/unet_V1.onnx",
    "v1_fast": "weights/unet_v1_fp16.onnx",
    "v2": "weights/unet_gan_v2.onnx",
    "v2_fast": "weights/unet_v2_fp16.onnx",
    "v3": "weights/generator_final.onnx",
    "v3_fast": ONNX_FP16_WEIGHTS_PATH
}

PT_PATHS = {
    "v1": "weights/unet_V1.pth",
    "v1_fast": "weights/unet_V1.pth",
    "v2": "weights/unet_gan_V2.pth",
    "v2_fast": "weights/unet_gan_V2.pth",
    "v3": "weights/generator_final.pth",
    "v3_fast": "weights/generator_final.pth"
}

def get_onnx_session(version: str):
    if version in loaded_onnx_sessions:
        return loaded_onnx_sessions[version]
        
    path = ONNX_PATHS.get(version, "weights/unet_gan_v2.onnx")
    if not os.path.exists(path):
        print(f"[WARN] ONNX {path} introuvable, fallback vers weights/unet_gan_v2.onnx")
        path = "weights/unet_gan_v2.onnx"
        
    try:
        session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        loaded_onnx_sessions[version] = session
        return session
    except Exception as e:
        print(f"[ERROR] ONNX loading failed for {version}: {e}")
        return None

def get_pt_model(version: str):
    if version in loaded_pt_models:
        return loaded_pt_models[version]
    
    path = PT_PATHS.get(version, "weights/unet_v2_gan_890.pth")
    if not os.path.exists(path):
        print(f"[WARN] PyTorch {path} introuvable, fallback vers weights/unet_v2_gan_890.pth")
        path = "weights/unet_v2_gan_890.pth"
        
    try:
        model = Unet()
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()
        # Optimisation mémoire : on n'a pas besoin de calculer les gradients pour l'inférence
        for param in model.parameters():
            param.requires_grad = False
        loaded_pt_models[version] = model
        return model
    except Exception as e:
        print(f"[ERROR] PyTorch loading failed for {version}: {e}")
        return None

def get_multiple_of_16(val):
    return int(math.ceil(val / 16.0)) * 16

def get_weight_mask(h, w):
    y = torch.linspace(-1, 1, h)
    x = torch.linspace(-1, 1, w)
    weight_y = torch.cos(y * math.pi / 2.0)
    weight_x = torch.cos(x * math.pi / 2.0)
    weight_2d = torch.outer(weight_y, weight_x)
    # Add small epsilon to avoid division by zero at strict boundaries
    return weight_2d.unsqueeze(0).unsqueeze(0) + 1e-5

def process_by_patches_onnx(img_tensor, ort_session, input_name, patch_size=512, overlap=64):
    _, _, H, W = img_tensor.shape
    stride = patch_size - overlap
    
    out_tensor = torch.zeros_like(img_tensor)
    count_tensor = torch.zeros((1, 1, H, W), dtype=torch.float32)
    
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1 = min(H, y + patch_size)
            x1 = min(W, x + patch_size)
            y0 = max(0, y1 - patch_size)
            x0 = max(0, x1 - patch_size)
            
            patch = img_tensor[:, :, y0:y1, x0:x1]
            weight = get_weight_mask(y1-y0, x1-x0)
            
            patch_np = patch.numpy()
            out_patch_np = ort_session.run(None, {input_name: patch_np})[0]
            out_patch = torch.from_numpy(out_patch_np)
            
            out_tensor[:, :, y0:y1, x0:x1] += out_patch * weight
            count_tensor[:, :, y0:y1, x0:x1] += weight
            
    return out_tensor / count_tensor


def process_by_patches_pt(img_tensor, model, patch_size=512, overlap=64):
    _, _, H, W = img_tensor.shape
    stride = patch_size - overlap
    
    out_tensor = torch.zeros_like(img_tensor)
    count_tensor = torch.zeros((1, 1, H, W), dtype=torch.float32)
    
    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y1 = min(H, y + patch_size)
                x1 = min(W, x + patch_size)
                y0 = max(0, y1 - patch_size)
                x0 = max(0, x1 - patch_size)
                
                patch = img_tensor[:, :, y0:y1, x0:x1]
                weight = get_weight_mask(y1-y0, x1-x0).to(patch.device)
                
                out_patch = model(patch)
                
                out_tensor[:, :, y0:y1, x0:x1] += out_patch * weight
                count_tensor[:, :, y0:y1, x0:x1] += weight
            
    return out_tensor / count_tensor

@app.get("/health")
def read_root():
    return {"status": "online", "message": "DeepLens Hybrid API is running!"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...), 
    model_version: str = Form("v2"),
    use_tiles: str = Form("false"),
    use_onnx: str = Form("false")
):
    use_tiles = use_tiles.lower() == "true"
    use_onnx = use_onnx.lower() == "true"
    
    import time
    start_time = time.time()
    print(f"\n[{'ONNX' if use_onnx else 'PyTorch'}] Inférence lancée - Modèle: {model_version.upper()} - Tuiles: {'OUI' if use_tiles else 'NON'}")
    
    # 1. Sélection et Chargement du Modèle
    if use_onnx:
        ort_session = get_onnx_session(model_version)
        if ort_session is None:
            return Response(content=b"Erreur: Impossible de charger le modele ONNX", status_code=500)
        input_name = ort_session.get_inputs()[0].name
    else:
        pt_model = get_pt_model(model_version)
        if pt_model is None:
            return Response(content=b"PyTorch Model not loaded", status_code=500)
    
    # 2. Configuration des Moyennes et Ecarts-types selon la version du modèle
    if model_version in ["v3", "v3_fast"]:
        current_mean = [0.237, 0.476, 0.464]
        current_std = [0.197, 0.239, 0.239]
    else:
        current_mean = [0.269, 0.491, 0.496]
        current_std = [0.213, 0.197, 0.216]
        
    # 3. Lecture de l'image envoyée par l'utilisateur
    image_data = await file.read()
    image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
    w, h = image_pil.size
    
    # On force des dimensions multiples de 16 pour éviter les erreurs avec le U-Net
    new_w = get_multiple_of_16(w)
    new_h = get_multiple_of_16(h)
    
    # 4. Pré-traitement de l'image
    if not use_tiles:
        # Traitement classique (on limite la taille à 800px max pour préserver la mémoire)
        max_dim = 800
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_w = get_multiple_of_16(w * scale)
            new_h = get_multiple_of_16(h * scale)
            
    img_resized = image_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    
    mean_np = np.array(current_mean, dtype=np.float32).reshape(3, 1, 1)
    std_np = np.array(current_std, dtype=np.float32).reshape(3, 1, 1)
    img_array = (img_array - mean_np) / std_np
    
    img_batch = np.expand_dims(img_array, axis=0) # [1, 3, H, W]
    
    if model_version.endswith('_fast'):
        img_batch = img_batch.astype(np.float16)
        
    # 5. Lancement de la prédiction (Inférence)
    if use_onnx:
        if use_tiles:
            img_tensor = torch.from_numpy(img_batch)
            out_tensor = process_by_patches_onnx(img_tensor, ort_session, input_name)
            prediction = out_tensor.numpy()
        else:
            outputs = ort_session.run(None, {input_name: img_batch})
            prediction = outputs[0]
    else:
        img_tensor = torch.from_numpy(img_batch)
        if use_tiles:
            out_tensor = process_by_patches_pt(img_tensor, pt_model)
        else:
            with torch.no_grad():
                out_tensor = pt_model(img_tensor)
        prediction = out_tensor.numpy()
    
    # 6. Post-traitement et Envoi de l'image
    prediction = prediction.astype(np.float32)
    prediction = (prediction * std_np) + mean_np
    prediction = np.clip(prediction, 0.0, 1.0)
    
    prediction_img = np.transpose(prediction[0], (1, 2, 0))
    prediction_img = (prediction_img * 255.0).astype(np.uint8)
    
    final_pil = Image.fromarray(prediction_img)
    # On retaille à la dimension originale exacte
    final_pil = final_pil.resize((w, h), Image.Resampling.BILINEAR)
    
    buf = io.BytesIO()
    final_pil.save(buf, format="PNG")
    buf.seek(0)
    
    elapsed_time = time.time() - start_time
    print(f"[SUCCÈS] Inférence terminée en {elapsed_time:.2f} secondes.")
    
    # Nettoyage de la RAM pour éviter que le serveur crash (très important)
    del img_batch, img_array, prediction, prediction_img, final_pil
    if 'img_tensor' in locals():
        del img_tensor
    if 'out_tensor' in locals():
        del out_tensor
    gc.collect()
    
    return Response(content=buf.getvalue(), media_type="image/png")

# Serve the static frontend (this MUST be at the end so it doesn't override /predict)
app.mount("/", StaticFiles(directory="deployment/frontend", html=True), name="frontend")