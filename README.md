# Underwater Image Restoration — OUC Internship Project

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-In_Progress-orange)

This repository contains the code for my 4th-year research internship project at the **Ocean University of China (OUC)**. 
The goal of this project is to restore underwater images using Deep Learning. I built and trained a custom **U-Net convolutional neural network** to correct the colors and remove the blur caused by water.

---

## The Problem

When you take photos underwater, the water absorbs light (especially red colors) and scatters it. This makes the images look green/blue, blurry, and low contrast. 
If we want to use AI to detect fish or map the ocean floor, we first need to restore the true colors of these images.

This project uses image-to-image translation techniques to mathematically reverse these physical distortions.

---

## Architecture

I chose a **U-Net** architecture. It is an encoder-decoder model with skip connections, which is excellent at keeping the shape of objects (like corals or fish) intact while we change the colors.

```
Input (blurry)             Output (clear)
[3, 256, 256]    →  U-Net  →  [3, 256, 256]

Encoder (Downsampling)       Decoder (Upsampling)
  DoubleConv → 64ch            ConvTranspose → + skip1
  DoubleConv → 128ch           ConvTranspose → + skip2
  DoubleConv → 256ch           ConvTranspose → + skip3
  DoubleConv → 512ch  (Bottleneck)
                               Final Conv → 3ch (RGB)
```

The skip connections pass the detailed outlines directly from the encoder to the decoder, preventing the image from becoming too blurry during the process.

---

## Project Structure

```
.
├── model.py           # The PyTorch U-Net code
├── dataset.py         # Code to load and prepare the images
├── train.py           # The training loop (loss, optimizer, etc.)
├── utils.py           # Helper scripts (like calculating mean/std)
├── config.example.py  # Template for your local paths
├── config.py          # Your actual config file (gitignored)
├── references.md      # Scientific papers I read for this project
├── raw/               # The blurry input images (gitignored)
└── ref/               # The clean reference images (gitignored)
```

Note: The actual dataset images are not uploaded to GitHub because they are too heavy.

---

## Evaluation & Analysis

## 1. Les Résultats Obtenus
Après avoir entraîné notre modèle **U-Net V2 (avec les blocs d'attention)** sur notre dataset, voici les scores que nous avons mesurés :
- **PSNR (Netteté et Couleurs)** : 12.51 dB
- **SSIM (Formes et Géométrie)** : 0.93

Ces résultats sont intéressants : le SSIM est excellent, mais le PSNR reste assez bas par rapport aux standards habituels. Voici pourquoi.

## 2. Le SSIM (0.93) : De très bonnes formes
Le score SSIM compare la structure globale de l'image. 
Un score de **0.93** prouve que le modèle fait un travail remarquable pour conserver les formes (les contours des poissons, le relief des rochers, etc.). Grâce à l'Attention Spatiale que nous avons ajoutée dans la V2, le réseau comprend ce qui est important dans l'image et ne déforme pas les objets.

## 3. Le PSNR (12.51 dB) : Le problème des couleurs
Le PSNR mesure l'écart exact de couleur pixel par pixel entre notre résultat et l'image parfaite. 
Sous l'eau, il manque énormément de lumière rouge. Notre modèle a du mal à deviner et recréer les couleurs exactes, ce qui fait chuter le score PSNR, pour deux raisons :

1. **La fonction d'erreur (Loss) :** Si le modèle génère une image très nette mais avec un bleu légèrement différent de la réalité, le calcul mathématique du PSNR va fortement pénaliser le modèle, même si l'image nous paraît très belle à l'œil nu.
2. **Le manque de données :** Notre dataset actuel est trop petit. Le modèle n'a pas vu assez d'exemples de fonds marins différents pour apprendre à corriger parfaitement les couleurs dans toutes les situations.

## 4. Le problème de la "grille" (Effet Damier)
En zoomant sur certaines images générées, on remarque parfois un léger motif de grille (le "Checkerboard Artifact"). 
C'est un problème classique lié aux couches de "Déconvolution" utilisées pour agrandir l'image dans le réseau. Quand les pixels se chevauchent mal lors du calcul, cela crée cette grille invisible qui fait aussi baisser notre score PSNR.

## 5. Conclusion et passage à la Phase 3
Ces scores prouvent que notre architecture U-Net V2 est très solide pour comprendre la géométrie sous-marine (bon SSIM), mais qu'elle a besoin d'aide pour les couleurs (faible PSNR).

Pour régler l'effet de grille et booster les couleurs, nous devons passer à l'étape supérieure. C'est l'objectif de la **Phase 3 : l'entraînement massif**. Nous allons utiliser un nouveau dataset géant (EUVP) de plus de 11 000 images pour donner au modèle l'expérience qui lui manque.


---

## How to use it

### 1. Installation

```bash
git clone https://github.com/mdjodallah/ouc-underwater-image-restoration.git
cd ouc-underwater-image-restoration

python -m venv venv
source venv/bin/activate       # Windows: venv\Scriptsctivate
pip install torch torchvision pillow
```

### 2. Configuration

Copy the config template and add your own image folder paths:

```bash
cp config.example.py config.py
```

Edit `config.py`:
```python
CHEMIN_RAW = "path/to/your/raw/images"
CHEMIN_REF = "path/to/your/reference/images"
```

### 3. Training

Run the training script:

```bash
python train.py
```

The script will automatically use your GPU if you have one (it supports Apple Silicon MPS and NVIDIA CUDA).

---

## Technical Choices

- **U-Net**: Very stable and great at preserving image shapes.
- **L1 Loss**: I used L1 (Mean Absolute Error) instead of MSE because it keeps the images sharper and avoids a "smudged" look.
- **Normalization**: I normalize the images per color channel to help the AI handle the lack of red light underwater.
- **256x256 resizing**: A good balance to train the model relatively fast without losing too much detail.

---

**4th Year Internship** — Ocean University of China (OUC) · Qingdao, China · 2026  
**Author:** Moutassim Djodalah annour
**Supervisor:** Prof. Cao Jingchao
