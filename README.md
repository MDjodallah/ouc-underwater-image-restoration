# Underwater Image Restoration — OUC Research Internship

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-In_Progress-orange)

A deep learning research project conducted at the **Ocean University of China (OUC, 985 institution)** as part of a research internship. The goal is to restore degraded underwater images using a custom **U-Net convolutional neural network**, trained on paired raw/reference image datasets captured in real aquatic conditions.

---

## 🔬 Research Context

Underwater images suffer from severe degradation caused by light absorption and scattering in water. The **red channel** is the most affected, resulting in characteristic color casts (green/blue dominance), reduced contrast, and blurriness. Restoring these images is a prerequisite for any downstream marine computer vision task (e.g., object detection, species classification, habitat mapping).

This project is positioned at the intersection of **image-to-image translation** and **low-level computer vision**, drawing from the latest literature in the field (see [`references.md`](./references.md)).

---

## 🏗 Architecture

The restoration model is a **U-Net**, an encoder-decoder architecture with skip connections, chosen for its ability to preserve spatial detail while learning high-level degradation patterns.

```
Input (degraded)       Output (restored)
[3, 256, 256]    →  U-Net  →  [3, 256, 256]

Encoder (Downsampling)       Decoder (Upsampling)
  DoubleConv → 64ch            ConvTranspose → + skip1
  DoubleConv → 128ch           ConvTranspose → + skip2
  DoubleConv → 256ch           ConvTranspose → + skip3
  DoubleConv → 512ch  (Bottleneck)
                               Final Conv → 3ch (RGB)
```

Each `DoubleConv` block: `Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU`

Skip connections concatenate encoder feature maps with the corresponding decoder stage, allowing the model to recover fine-grained details lost during downsampling.

---

## 📂 Project Structure

```
.
├── model.py           # U-Net architecture definition (PyTorch)
├── dataset.py         # Custom Dataset class (paired raw/ref loading + normalization)
├── train.py           # Training loop (Adam optimizer, L1 loss, MPS/CUDA/CPU support)
├── utils.py           # Dataset statistics computation (mean/std per channel)
├── config.example.py  # Configuration template (paths, hyperparameters, normalization stats)
├── config.py          # Local configuration (gitignored — contains absolute paths)
├── references.md      # State-of-the-art literature reviewed for this project
├── raw/               # Input: degraded underwater images (gitignored)
└── ref/               # Target: reference/clean underwater images (gitignored)
```

> **Note:** Raw image data is gitignored to avoid committing large binary files. The dataset must be placed locally according to `config.example.py`.

---

## 🚀 Getting Started

### 1. Clone and set up the environment

```bash
git clone https://github.com/mdjodallah/ouc-underwater-image-restoration.git
cd ouc-underwater-image-restoration

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install torch torchvision pillow
```

### 2. Configure paths

Copy the example config and fill in your local dataset paths:

```bash
cp config.example.py config.py
```

Edit `config.py`:

```python
CHEMIN_RAW = "path/to/your/raw/images"   # Degraded underwater images
CHEMIN_REF = "path/to/your/reference/images"  # Clean reference images
```

### 3. (Optional) Compute dataset statistics

If using a new dataset, recompute the normalization statistics before training:

```bash
python utils.py
```

Update the `MEAN` and `STD` values in `config.py` with the output.

### 4. Train the model

```bash
python train.py
```

Training runs for `EPOCHS` iterations (configurable in `config.py`). The script automatically selects the best available device:
- Apple Silicon → **MPS**
- NVIDIA GPU → **CUDA**
- Fallback → **CPU**

---

## ⚙️ Key Design Choices

| Choice | Rationale |
|---|---|
| **U-Net** | Strong baseline for image-to-image translation; skip connections preserve spatial resolution |
| **L1 Loss** | Pixel-wise MAE; more robust to outliers than MSE and avoids over-smoothed outputs |
| **Channel normalization** | Per-channel mean/std computed on the raw dataset; accounts for the red-channel deficit typical of underwater imagery |
| **256×256 resize** | Standardizes heterogeneous input sizes; balances detail vs. training speed |
| **MPS support** | Enables native GPU acceleration on Apple Silicon during local development |

---

## 📚 References

See [`references.md`](./references.md) for a curated review of the state-of-the-art literature this project builds upon, including:
- Survey of deep learning methods for underwater enhancement (Cong et al., 2024)
- UniUIR all-in-one restoration framework (Zhang et al., 2025)
- Semi-supervised contrastive learning for paired-data scarcity (Huang et al., 2023)

---

**Research Internship** — Ocean University of China (OUC) · Qingdao, China · 2026  
**Supervisor:** Ocean University of China, College of Computer Science and Technology
