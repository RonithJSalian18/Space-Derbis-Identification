# 🛰️ Space Debris Identification System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-FF6F00.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

A production-grade Deep Learning & Computer Vision system designed to classify orbital imagery into **Space Debris** vs. **Non-Debris (Active Satellites & Spacecraft)**. Built with TensorFlow/Keras and OpenCV, featuring dynamic GPU memory management, automated image deduplication, zero-g spatial augmentations, and modular model backbones.

---

## 📌 Problem Statement & Vision

Space debris poses a critical threat to orbital infrastructure and satellite constellations. Automated detection requires models that:

1. **Preserve Fine Geometric Features:** Detect small orbital fragment textures alongside structured metallic solar panels at $224\times224$ resolution.
2. **Prevent Overfitting:** Utilize **Global Average Pooling**, L2 Regularization, and **Label Smoothing** to avoid memorizing black space background artifacts.
3. **Handle Zero-Gravity Spatial Variance:** Apply 180° rotation invariance and multi-axis flips to account for arbitrary orbital orientations.

---

## 📂 Project Architecture

```text
Space-Debris-Identification/
├── configs/                  # Global hyperparameter settings & path configuration
│   ├── __init__.py
│   └── config.py
├── saved_models/             # Export directory for trained model artifacts (.h5)
├── plots/                    # Automated evaluation plots (Confusion matrix, ROC, Learning curves)
├── src/                      # Core Modular Package
│   ├── data/                 # Dataset loader, cleaning (MD5/pHash), and preprocessing
│   │   ├── loader.py
│   │   └── preprocessing.py
│   ├── models/               # Model Factory & Architecture Definitions
│   │   ├── builder.py        # Dynamic Model Factory
│   │   ├── cnn.py            # Custom 4-stage CNN with Global Average Pooling
│   │   ├── mobilenet.py      # MobileNetV2 Transfer Learning
│   │   ├── resnet.py         # ResNet50 Transfer Learning
│   │   └── efficientnet.py   # EfficientNetB0 Transfer Learning
│   ├── evaluation/           # Evaluation metrics & learning curve visualization
│   │   └── metrics.py
│   ├── inference/            # Prediction engine
│   │   └── predictor.py
│   └── utils/                # GPU detection & memory growth setup
│       └── gpu.py
├── train.py                  # Unified CLI entrypoint for model training
├── predict.py                # Unified CLI entrypoint for single/batch inference
├── Dockerfile                # Production Docker container configuration
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🧠 Supported Model Architectures

| Model Name         | Type              | Key Characteristics                              | Recommended Use Case                         |
| :----------------- | :---------------- | :----------------------------------------------- | :------------------------------------------- |
| **Custom CNN**     | Custom Deep CNN   | 4-Stage Conv2D + Global Average Pooling + L2 Reg | Fast training, lightweight, zero overfitting |
| **MobileNetV2**    | Transfer Learning | Pretrained ImageNet backbone + Native scaling    | Edge deployment / Real-time inference        |
| **ResNet50**       | Transfer Learning | Residual connections + Deep feature extraction   | High complexity pattern recognition          |
| **EfficientNetB0** | Transfer Learning | Compound scaling architecture                    | Maximum accuracy-to-parameter efficiency     |

---

## ⚙️ Data Quality & Preprocessing Pipeline

- **Image Corruption Pruning:** Scans image headers and removes unreadable or degraded image files.
- **Perceptual Hashing (pHash):** Uses `imagehash.phash()` to prune visually identical frames, preventing data leakage between train/validation/test splits.
- **Space-Domain Augmentation:**
  - `rotation_range = 180` (Full 180° rotation invariance)
  - `horizontal_flip` & `vertical_flip` (Zero-g spatial invariance)
  - `brightness_range = [0.7, 1.3]` (Orbital solar lighting variations)
- **Label Smoothing Regularization:** Applies `BinaryCrossentropy(label_smoothing=0.1)` to prevent overconfident target memorization.

---

## 🚀 Quick Start Guide

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/RonithJSalian18/Space-Derbis-Identification.git
cd Space-Derbis-Identification

# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Model Training

Train any supported model architecture via `train.py`:

```bash
# Train Custom CNN (Default, 224x224 resolution)
python train.py --model cnn --epochs 30 --batch-size 16

# Train MobileNetV2 Transfer Learning
python train.py --model mobilenet --epochs 20

# Train ResNet50 Transfer Learning
python train.py --model resnet --epochs 20

# Train EfficientNetB0
python train.py --model efficientnet --epochs 20
```

> **Artifact Outputs:** Best model weights are automatically saved to `saved_models/<model_name>_debris.h5`.

---

### 3. Running Predictions / Inference

Classify a single space image using `predict.py`:

```bash
python predict.py --image "path/to/space_image.jpg" --model "saved_models/cnn_debris.h5" --type cnn
```

**Sample Output:**

```text
==================================================
📸 INFERENCE RESULT
==================================================
File Path:       path/to/space_image.jpg
Prediction:      Debris
Confidence:      98.64%
Debris Prob:     0.9864
Non-Debris Prob: 0.0136
==================================================
```

---

## 📊 Evaluation & Visualization Outputs

After training, metrics and charts are automatically saved in `plots/<model_name>/`:

- `learning_curves.png`: 4-Panel Loss, Accuracy, Precision, and Recall history curves.
- `confusion_matrix.png`: Heatmap showing true vs. predicted classifications.
- `roc_curve.png`: Receiver Operating Characteristic curve with AUC score.
- `precision_recall_curve.png`: Precision-Recall curve highlighting debris safety recall.

---

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t space-debris-detector .

# Run GPU-accelerated container
docker run --gpus all space-debris-detector
```

---
