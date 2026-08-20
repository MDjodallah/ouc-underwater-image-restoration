<div align="center">
  
# 🌊 DeepLens : Underwater Image Restoration

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Status](https://img.shields.io/badge/Status-Active-success)](#)

*An advanced Deep Learning solution to mathematically reverse underwater color distortion and scattering, built for my 4th-year research internship at the **Ocean University of China (OUC)**.*

</div>

---

## 📖 Overview

Underwater photography suffers from severe degradation. Water absorbs light (particularly the red spectrum) and scatters it, resulting in images that are often green/blue, low-contrast, and blurry. 

If we want to use AI for marine biology, autonomous underwater vehicles (AUVs), or simply to admire the ocean floor, we first need to restore the true colors and sharpness of these images. **DeepLens** uses an image-to-image translation deep neural network to learn the physical properties of water distortion and mathematically reverse it.

---

## 🧠 Architecture: U-Net + PatchGAN

The core of this project relies on a hybrid architecture combining a **U-Net** Generator and a **PatchGAN** Discriminator (cGAN approach).

### 1. The Generator (U-Net with Spatial Attention)
The U-Net acts as an encoder-decoder. It downsamples the blurry image to extract deep features, and upsamples it back to a high-resolution clear image. 
- **Skip Connections**: We use skip connections to bypass the bottleneck, allowing the model to preserve fine details (like coral textures or fish scales) perfectly.
- **Spatial Attention**: Added in V2, it allows the network to focus on important objects rather than the uniform blue background.

### 2. The Discriminator (PatchGAN)
Instead of classifying the entire image as "real" or "fake", the PatchGAN discriminator looks at small patches (e.g., 70x70 pixels) of the image. This forces the Generator to produce highly realistic high-frequency details (sharpness) and eliminates the blurry artifacts common in standard L1/L2 loss models.

---

## 📊 Dataset & Training

The model is trained on a massive hybrid dataset combining two of the most popular underwater image datasets:
1. **UIEB** (Underwater Image Enhancement Benchmark)
2. **EUVP** (Enhancing Underwater Visual Perception)

By fusing these datasets, the model learns from over **12,000 paired images**, covering various water types (greenish, bluish, murky, and clear).

---

## 📂 Project Structure

A clean, modular, and professional architecture separating the core logic from deployment scripts.

```text
.
├── dataset/                # 🗃️ Training Data (Images)
│   ├── raw/                # Distorted underwater images
│   └── ref/                # Clear reference images
│
├── core/                   # 🧠 Core Deep Learning Engine
│   ├── model.py            # Generator Architecture (U-Net)
│   ├── discriminator.py    # Discriminator Architecture (PatchGAN)
│   ├── dataset.py          # PyTorch Dataloaders
│   ├── metrics.py          # PSNR & SSIM Evaluation
│   ├── utils.py            # Normalization helpers
│   └── config.py           # Hyperparameters & Hardware Detection
│
├── weights/                # 🗄️ Model Checkpoints & Versions
│   ├── unet_V1.onnx        # Baseline U-Net (ONNX)
│   ├── unet_gan_v2.onnx    # V2 PatchGAN (ONNX)
│   └── generator_final.pth # V3 PatchGAN Massif (PyTorch)
│
├── scripts/                # 🛠️ Execution Scripts
│   ├── training/           # Training (U-Net, GAN)
│   ├── evaluation/         # Inference & Metrics
│   ├── preprocessing/      # Dataset preparation
│   └── export/             # ONNX export
│
├── deployment/             # 🚀 Production Deployment
│   ├── api/                # FastAPI Server (PyTorch & ONNX)
│   └── frontend/           # Web Interface
│
├── tests/                  # 🧪 Automated Unit Tests (Pytest)
├── Dockerfile              # 🐳 Docker Deployment Configuration
├── docker-compose.yml      # 🐳 Docker Compose Orchestration
├── .env.example            # 🔐 Environment Variables Template
├── README.md               # Documentation
└── requirements.txt        # Python dependencies
```

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/MDjodallah/ouc-underwater-image-restoration.git
cd ouc-underwater-image-restoration
pip install -r requirements.txt
```

### 2. Run the Local Backend (FastAPI)
The backend loads the model weights from the `weights/` directory. It supports three model versions (V1, V2, V3) and two inference engines (PyTorch, ONNX).
```bash
uvicorn deployment.api.api:app --reload
```
*The API will start at `http://127.0.0.1:8000/`*

### 3. Test with the UI
Simply open the `deployment/frontend/index.html` file in your favorite web browser (Chrome, Safari, Firefox). 
Upload an underwater image, and the frontend will automatically send it to the FastAPI backend for restoration.

---

## 📈 Evaluation

| Metric | U-Net V1 (Baseline) | U-Net V2 (Attention + Bilinear) | U-Net V3 (Final GAN) |
|--------|---------------------|---------------------|-------------------------------|
| **PSNR** | 6.12 dB | 4.85 dB | **12.16 dB** |
| **SSIM** | 0.5553 | 0.4964 | **0.7255** |

- **Why V1 scores mathematically higher than V2**: V1 is a simple baseline that focuses strictly on pixel-to-pixel reconstruction. V2 introduces Spatial Attention and Bilinear Upsampling. While Bilinear upsampling avoids the "checkerboard artifacts" of V1 and often produces visually smoother and more natural images (especially on simple clear waters), it inherently blurs high-frequency details. This blurriness is heavily penalized by strict mathematical metrics like PSNR and SSIM, which demand pixel-perfect sharpness, resulting in mathematically lower scores for V2 despite its visual improvements in some contexts.
- **The Power of GAN (V3)**: The massive jump to 12.16 dB / 0.7255 SSIM proves the discriminator successfully forces the generator to restore the lost high-frequency textures and true color distribution without the blurriness of V2, making it an incredibly robust all-terrain model for extreme underwater environments.

---

## ⚡ ONNX Runtime Optimization

To ensure real-time performance and overcome PyTorch inference overhead in production, we export all our PyTorch checkpoints to **ONNX (Open Neural Network Exchange)** with dynamic axes support. This allows the backend to perform HD image restoration on standard CPUs in seconds, enabling cost-effective cloud deployments.

---

<div align="center">
  <b>4th Year Internship</b> — Ocean University of China (OUC) · Qingdao, China · 2026<br>
  <b>Author:</b> Moutassim Djodalah annour · <b>Supervisor:</b> Prof. Cao Jingchao
</div>
