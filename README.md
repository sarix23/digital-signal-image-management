# Gesture AI Suite — GAN for Sign Language + Video Gesture Recognition

This repository contains two complementary deep learning projects focused on **gesture recognition**:

1. **Sign Language MNIST with GAN / cGAN**  
   Generative models used to synthesize and augment sign language images.

2. **Video Gesture Recognition**  
   Sequence-based models for classifying gestures from short video clips.

Both projects are implemented entirely in **Jupyter notebooks**, with reproducible pipelines for **data preprocessing, training, evaluation, and visualization**.

---

# Project 1 — GAN for Sign Language MNIST

## Overview
This module implements:

- an **unconditional GAN** for generating 28×28 grayscale sign language images  
- a **conditional GAN (cGAN)** that generates images conditioned on class labels

The main goal is to explore **synthetic data generation** and **data augmentation** for gesture recognition.

Main notebook:
- `GAN.ipynb`

## Key Components

Inside the notebook:

- Data preprocessing → `pre_processing`
- GAN training → `train_gan`
- cGAN training → `train_cgan`
- FID evaluation → `calculate_fid`, `calculate_fid_cgan`
- Model architectures → `generator`, `discriminator`, `generator_model_c`

Additional tools:

- `GAN/demo_GAN.py` → Streamlit demo for dataset visualization and image generation
- `GAN/archive/GAN.ipynb` → archived versions

## Evaluation (FID)

The project evaluates generation quality using **Fréchet Inception Distance (FID)**:

$$
d_F(\mathcal{N}(\mu,\Sigma),\mathcal{N}(\mu',\Sigma'))^2 = \|\mu-\mu'\|_2^2 + \operatorname{tr}\big(\Sigma + \Sigma' - 2(\Sigma\Sigma')^{1/2}\big)
$$

FID is computed using **InceptionV3 features** on resized RGB images.

## Running the GAN module

```bash
jupyter lab
````

Open `GAN.ipynb` and run all cells.

To run the demo:

```bash
streamlit run GAN/demo_GAN.py
```

Saved models are typically stored in:

* `GAN_B256/`
* `cGAN_B128/`

---

# Project 2 — Video Gesture Recognition

## Overview

This project performs classification of **5 gestures**:

* Thumbs up
* Thumbs down
* Left swipe
* Right swipe
* Stop

Each sample is a **video sequence of 30 frames**.

Main notebook:

* `Gesture_Recognition.ipynb`

## Data Pipeline

Raw structure:

```
jesture/train/
jesture/val/
```

After preprocessing:

```
jesture/training_set_resized/
jesture/test_set_resized/
```

Generated artifacts:

* `X_train.npy`, `y_train.npy`
* `X_test.npy`, `y_test.npy`
* `X_train_augmented.npy`, `y_train_augmented.npy`

## Models

Two approaches are implemented:

### 1. LRCN (Conv + LSTM)

* TimeDistributed Conv2D layers
* LSTM temporal modeling
* Accuracy ≈ **75%**
* Suffers from overfitting

### 2. Transfer Learning (Best model)

* **MobileNetV2 backbone**
* TimeDistributed feature extraction
* LSTM for temporal dynamics
* Accuracy ≈ **93%**

## Preprocessing and Augmentation

* Frame resizing (64×64 or 224×224)
* Normalization to [0,1]
* Augmentation:

  * brightness
  * contrast
  * zoom
  * JPEG compression
  * central crop
* No horizontal flip (to preserve gesture direction)

## Training and Evaluation

Training pipeline includes:

* Early stopping
* Dropout + L2 regularization
* Confusion matrix and learning curves

---

# Installation

### Create environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install numpy pandas matplotlib seaborn pillow scipy scikit-learn tensorflow
```

Optional:

```bash
pip install opencv-python regex streamlit
```

---

# Quick Start

### Run GAN project

```bash
jupyter lab
```

Open `GAN.ipynb` and execute all cells.

### Run Video Gesture Recognition

Open `Gesture_Recognition.ipynb` and run sequentially:

1. Data organization
2. Frame resizing
3. Array creation
4. Augmentation (optional)
5. Model training (LRCN or MobileNetV2+LSTM)
6. Evaluation

---

# Reproducibility Notes

* Set random seeds for **NumPy and TensorFlow**
* Use **GPU acceleration** when available
* Monitor both **generator/discriminator losses** for GAN stability
* Video preprocessing can require **significant disk space**

---

# Results Summary

| Task                             | Model              | Performance       |
| -------------------------------- | ------------------ | ----------------- |
| Sign Language (image generation) | GAN / cGAN         | evaluated via FID |
| Video gesture recognition        | LRCN               | ~75% accuracy     |
| Video gesture recognition        | MobileNetV2 + LSTM | ~93% accuracy     |

---

# References

* Sign Language MNIST (Kaggle)
* Gesture Recognition Dataset (Kaggle)
* MobileNetV2 (Keras Applications)
* LRCN (CNN + LSTM for video)
* FID — Heusel et al.

---

# Notes

This repository is fully **notebook-based** and designed for **experimentation and research** on gesture recognition and generative modeling.
