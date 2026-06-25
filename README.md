# Nuclei Instance Segmentation — Data Science Bowl 2018

> **Assignment project** for the 2018 Kaggle Data Science Bowl challenge:  
> *"Find the nuclei in divergent images to advance medical discovery"*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-DSB%202018-20BEFF)](https://www.kaggle.com/c/data-science-bowl-2018)

---

## Table of Contents

- [Overview](#overview)
- [Dataset Analysis & Key Challenges](#dataset-analysis--key-challenges)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Setup](#setup)
- [Reference Repositories](#reference-repositories)

---

## Overview

The goal is to build a robust **nucleus instance segmentation** pipeline that works across diverse microscopy modalities — from fluorescence to H&E-stained histology slides. The project is structured in two parts:

- **Part A — Augmentation & Diffusion**: Address the small dataset problem via classical augmentation and diffusion-model-based synthetic data generation.
- **Part B — Segmentation Models**: Progressively build and compare segmentation architectures (CNN → U-Net → Transformer → Pretrained backbone).

---

## Dataset Analysis & Key Challenges

The EDA notebook (`notebooks/01_EDA.ipynb`) reveals four critical issues that drive every design decision in this project.

### 1. 🔴 High Risk of Overfitting — 670 Training Images

Deep learning models are data-hungry. 670 training images is a remarkably small dataset for segmentation.

**Takeaway:** This directly motivates **Part A2** (Classical Augmentation) and **Part A3** (Diffusion Models). Feeding just 670 raw images to a U-Net will cause it to memorize the training set and fail at inference.

---

### 2. 🟠 Inconsistent Tensor Shapes — 9 Unique Image Sizes

The dataset spans sizes from compact `256×256` squares to large `1040×1388` rectangles. Standard CNNs and Transformers require uniform tensor shapes for batched processing.

**Takeaway:** Raw images cannot be loaded directly into a PyTorch `DataLoader`. A preprocessing step must resize (e.g., to `256×256`) or crop all images and masks to a consistent dimension before training.

---

### 3. 🟡 Modality Clash — Grayscale vs. Color

The dataset contains a severe domain shift *within itself*:

| Modality | Share | Description |
|---|---|---|
| Dark grayscale | **~84%** | Fluorescence microscopy (1-channel) |
| Bright color | **~16%** | Histology stained slides (3-channel) |

**Takeaway:** Mixing 1-channel and 3-channel inputs without normalization will crash most model input layers. Solutions: convert all images to grayscale, or replicate the grayscale channel to get 3-channel RGB so the network sees a uniform structure.

---

### 4. 🔵 Severe Class Imbalance — 1 : 6.2 Pixel Ratio

Over **86%** of all pixels are empty background. Only **~13.8%** belong to an actual nucleus.

**Takeaway:** A model that outputs a completely black mask achieves 86% pixel accuracy while being completely useless. Evaluation must use **spatial overlap metrics**:

- **IoU (Intersection over Union)**
- **Dice Coefficient**

These metrics only reward the model when it correctly localises actual nuclei, making them the correct choice for Part B4 evaluation.

---

## Project Structure

```
Nuclei/
├── data/
│   └── raw/
│       ├── train/          # 670 training samples (image + instance masks)
│       └── test/           # Test samples (image only)
├── notebooks/
│   └── 01_EDA.ipynb        # Exploratory Data Analysis
├── outputs/
│   ├── train_metadata.csv  # Per-image stats (size, channels, mask coverage)
│   ├── test_metadata.csv
│   ├── eda_samples.png
│   ├── eda_image_sizes.png
│   ├── eda_class_imbalance.png
│   ├── eda_diversity.png
│   └── eda_statistics.png
├── src/
│   ├── models/             # Segmentation model implementations
│   └── diffusion/          # Diffusion model for data augmentation
├── requirements.txt
└── README.md
```

> **Note:** `data/raw/` is excluded from version control (195 MB). Download the dataset from [Kaggle](https://www.kaggle.com/c/data-science-bowl-2018/data) and place it under `data/raw/`.

---

## Roadmap

### Part A — Augmentation & Diffusion

| Step | Description | Key Tool |
|---|---|---|
| **A1** | Exploratory Data Analysis | `notebooks/01_EDA.ipynb` ✅ |
| **A2** | Classical augmentation pipeline | [`albumentations`](https://github.com/albumentations-team/albumentations) |
| **A3** | Unconditional DDPM for nuclei | [`NuDiff`](https://arxiv.org/abs/2303.09664) / [HuggingFace annotated diffusion](https://huggingface.co/blog/annotated-diffusion) |
| **A4** | Conditioned diffusion for mask-guided synthesis | [`Polyp-DDPM`](https://arxiv.org/abs/2306.07579) |

### Part B — Segmentation Models

| Step | Description | Key Tool / Reference |
|---|---|---|
| **B1** | Baseline CNN | [kamalkraj/DATA-SCIENCE-BOWL-2018](https://github.com/kamalkraj/DATA-SCIENCE-BOWL-2018) |
| **B2** | U-Net | [philferriere/tf-dsb-18](https://github.com/philferriere/tf-dsb-18) |
| **B3** | Transformer (SegFormer / Swin-UNet) | [SegFormer](https://github.com/NVlabs/SegFormer) |
| **B4** | Evaluation with IoU & Dice | — |
| **Bonus** | Fine-tune pretrained U-Net | [selimsef/dsb2018_topcoders](https://github.com/selimsef/dsb2018_topcoders) (1st place) |
| **Bonus** | ResNet101 backbone (Mask-RCNN) | [mirzaevinom/data_science_bowl_2018](https://github.com/mirzaevinom/data_science_bowl_2018) (5th place) |

---

## Setup

```bash
# Clone the repository
git clone https://github.com/Kostantinoskanell/nuclei-segmentation-dsb2018.git
cd nuclei-segmentation-dsb2018

# Install dependencies
pip install -r requirements.txt

# Download the dataset from Kaggle
# https://www.kaggle.com/c/data-science-bowl-2018/data
# Place train/ and test/ inside data/raw/
```

### Launch the EDA notebook

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

---

## Reference Repositories

### Part A — Augmentation + Diffusion

| Reference | Use |
|---|---|
| [albumentations](https://github.com/albumentations-team/albumentations) | flip, rotate, elastic, stain normalization (A2) |
| [NuDiff](https://arxiv.org/abs/2303.09664) | First diffusion-based augmentation method specifically for nuclei segmentation — unconditional diffusion → conditioned synthesis (A3–A4) |
| [HuggingFace Annotated Diffusion](https://huggingface.co/blog/annotated-diffusion) | Step-by-step DDPM from scratch in PyTorch (A3–A4) |
| [Polyp-DDPM](https://arxiv.org/abs/2306.07579) | Conditioning diffusion on segmentation masks for synthetic medical image generation (A3–A4) |

### Part B — Segmentation Models

| Reference | Use |
|---|---|
| [kamalkraj/DATA-SCIENCE-BOWL-2018](https://github.com/kamalkraj/DATA-SCIENCE-BOWL-2018) | Baseline U-Net inspired CNN, resize 256×256, ~0.302 leaderboard score (B1) |
| [philferriere/tf-dsb-18](https://github.com/philferriere/tf-dsb-18) | Dual U-Net with contour prediction and cyclic LR (B2) |
| [SegFormer](https://github.com/NVlabs/SegFormer) / [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet) | SOTA transformer architectures for medical segmentation (B3) |
| [selimsef/dsb2018_topcoders](https://github.com/selimsef/dsb2018_topcoders) | **1st place solution** — Generic U-Net TensorFlow implementation (Bonus) |
| [mirzaevinom/data_science_bowl_2018](https://github.com/mirzaevinom/data_science_bowl_2018) | **5th place solution** — Mask-RCNN with ResNet101 backbone and Adam optimizer (Bonus) |

---

## License

This project is for academic/educational purposes. Dataset © Kaggle / respective owners.
