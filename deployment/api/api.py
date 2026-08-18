from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import io
import math
import gc
from core.model import Unet
from core.config import DEVICE, MEAN, STD

# --- MEMORY OPTIMIZATION FOR LIMITED ENVIRONMENTS ---
# Restrict PyTorch to a single CPU thread to prevent OOM errors
torch.set_num_threads(1)

app = FastAPI(title="OUC Underwater Enhancement API")

# CORS Configuration to allow cross-origin requests from the frontend client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Accept all requests for development purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model weights into memory at server startup
print("[INFO] Initializing Generator Model (U-Net V2)...")
model = Unet().to(DEVICE)
# Pre-trained model weights file
# TIP: To test another version, simply change the filename below (e.g., "weights/unet_v2_nogan_890.pth")
chemin_poids = "weights/unet_v3_gan_12k.pth"
try:
    model.load_state_dict(torch.load(chemin_poids, map_location=DEVICE, weights_only=True))
    model.eval()
    print("[SUCCESS] Model loaded and ready for inference.")
except FileNotFoundError:
    print(f"[WARNING] Weights file not found ({chemin_poids}). Model will return blank tensors.")

def get_multiple_of_16(val):
    """Rounds a value to the nearest upper multiple of 16."""
    return int(math.ceil(val / 16.0)) * 16

# Health check endpoint (used by UptimeRobot to prevent server sleep)
@app.get("/")
async def ping():
    return {"status": "online", "message": "DeepLens API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image from the client, processes it through the model, and returns the restored image.
    """
    # 1. Read the image payload
    image_data = await file.read()
    image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Original image might be ultra-high resolution (e.g., 4K)
    w, h = image_pil.size
    
    # IMPORTANT: The model was trained on 256x256 patches.
    # Processing 4K images directly exceeds the receptive field, causing hallucinations or identity mapping.
    # Downscaling is required before inference, followed by upscaling.
    max_size = 800
    if max(w, h) > max_size:
        ratio = max_size / float(max(w, h))
        new_w = get_multiple_of_16(w * ratio)
        new_h = get_multiple_of_16(h * ratio)
    else:
        new_w = get_multiple_of_16(w)
        new_h = get_multiple_of_16(h)
    
    # 2. Pre-processing pipeline
    transform = transforms.Compose([
        transforms.Resize((new_h, new_w)), # Adjust to multiple of 16 for U-Net architecture
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    img_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
    
    # 3. Model Inference
    with torch.no_grad():
        prediction = model(img_tensor)
        
    # 4. Denormalization
    mean_t = torch.tensor(MEAN).view(1, 3, 1, 1).to(DEVICE)
    std_t = torch.tensor(STD).view(1, 3, 1, 1).to(DEVICE)
    prediction_denorm = (prediction * std_t) + mean_t
    prediction_denorm = torch.clamp(prediction_denorm, 0, 1)
    
    # 5. Direct conversion from HD Tensor to JPEG
    img_array = prediction_denorm.squeeze(0).cpu()
    img_pil_result = transforms.ToPILImage()(img_array)
    
    # Restore exact original dimensions (compensating for the multiple of 16 adjustment)
    img_pil_result = img_pil_result.resize((w, h), Image.Resampling.LANCZOS)
    
    buf = io.BytesIO()
    img_pil_result.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    
    # 6. Return the processed image stream to the client
    response = Response(content=buf.getvalue(), media_type="image/jpeg")
    
    # --- FORCED MEMORY CLEANUP ---
    # Manually delete heavy tensors to prevent OOM crashes on limited RAM instances
    del img_tensor, prediction, prediction_denorm, img_array, img_pil_result
    gc.collect()
    
    return response

# Command to launch the local server:
# uvicorn api:app --reload