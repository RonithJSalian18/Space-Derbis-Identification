# 🛰️ Space Debris Identification System — Advanced Orbital Computer Vision

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-FF6F00.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

A production-grade Deep Learning & Computer Vision pipeline designed to classify orbital space domain imagery into **Space Debris** (fragments, rocket bodies) vs. **Non-Debris (Active Satellites & Spacecraft)**. Engineered with zero-I/O bottleneck caching, Group-Based trajectory splitting, perceptual hash deduplication, harsh space-domain augmentations, architecture-specific tensor routing, and Zero-Trust Grad-CAM visual auditing.

---

## 📌 Technical Highlights & Engineering Decisions

### 1. Trajectory-Based Group Splitting (Curing Data Leakage)
- **The Problem**: Standard random train/test splitting on sequential video imagery causes extreme data leakage because consecutive frames of the same rotating spacecraft scatter across splits, artificially inflating evaluation metrics (1.00 AUC).
- **The Solution**: Implemented `split_dataset_by_trajectory()` in [`src/data/loader.py`](file:///D:/Space-VS/Space-Derbis-Identification/src/data/loader.py) using `GroupShuffleSplit` from `scikit-learn`. Grouping data strictly by object sequence ID guarantees a **70/15/15 Train/Val/Test** split where no trajectory in Train exists in Val or Test.

### 2. Perceptual Hash Deduplication (Solving the "Clone Problem")
- **The Problem**: Synthetic space datasets contain near-identical "frozen" frames that reduce batch variance and cause memory over-fitting.
- **The Solution**: Standalone parallel deduplicator [`scripts/deduplicate_cache.py`](file:///D:/Space-VS/Space-Derbis-Identification/scripts/deduplicate_cache.py) using `imagehash` (pHash) and PIL with `ProcessPoolExecutor`. Filters consecutive frames within classes where Hamming distance $\le 2$ and exports cleaned manifests (`cleaned_manifest_train.csv`).

### 3. Harsh Space-Domain Augmentations (Domain Randomization)
- **The Problem**: Overfitting to synthetic rendering engines or atmospheric light assumptions.
- **The Solution**: Graph-compatible TensorFlow operations in [`src/data/augmentation.py`](file:///D:/Space-VS/Space-Derbis-Identification/src/data/augmentation.py):
  - **Extreme Solar Glare**: Simulates un-attenuated direct sunlight blooms in orbital vacuum.
  - **Sensor Noise**: Injects Salt-and-Pepper radiation noise simulating cosmic ray strikes on space sensors.
  - **Photometric Jitter**: Aggressive randomized contrast and brightness shifts for extreme orbital light variations.
  - **Zero-G Invariance**: Full 180° rotation and dual-axis spatial flips.

### 4. Architecture-Specific Tensor Normalization
- **The Problem**: Feeding identically normalized data `[0, 1]` to transfer learning backbones that expect different scale inputs causes silent performance degradation.
- **The Solution**: Dedicated preprocessing router `apply_architecture_preprocessing()` in [`src/data/preprocessing.py`](file:///D:/Space-VS/Space-Derbis-Identification/src/data/preprocessing.py):
  - **ResNet**: ImageNet BGR conversion & mean subtraction (`resnet.preprocess_input`).
  - **MobileNetV2**: Pixel scaling to `[-1.0, 1.0]` (`mobilenet_v2.preprocess_input`).
  - **EfficientNetB0**: Raw `[0.0, 255.0]` tensor (native internal rescaling layers).
  - **Custom CNN**: `[0.0, 1.0]` float32 normalization.

### 5. Zero-Trust Grad-CAM Audit & Threshold Optimization
- **Grad-CAM Audit**: Automated feature visualization engine [`src/evaluation/gradcam.py`](file:///D:/Space-VS/Space-Derbis-Identification/src/evaluation/gradcam.py) that dynamically discovers the final convolutional layer of any model and exports visual overlays with intensity colorbars to `plots/gradcam_audit/`.
- **Threshold Optimization**: Optimizes binary decision thresholds on Precision-Recall curves to mitigate 10:1 class imbalance.

---

## 🧠 Supported Model Architectures

| Architecture | Input Format | Preprocessing Pipeline | Primary Strengths |
| :--- | :--- | :--- | :--- |
| **Custom CNN** | `(224, 224, 1)` Grayscale | Normalized `[0, 1]` | 4-Stage Conv2D, L2 Reg, GAP, 50% Dropout. Fast & lightweight. |
| **MobileNetV2** | `(224, 224, 3)` RGB | Scaled `[-1, 1]` | Inverted residual depthwise blocks for low-latency space edge devices. |
| **ResNet50** | `(224, 224, 3)` RGB | BGR + Mean Subtraction | Deep residual connections for complex pattern recognition. |
| **EfficientNetB0** | `(224, 224, 3)` RGB | Raw `[0, 255]` | Compound scaling for optimal accuracy-to-parameter ratio. |

---

## 📂 System File Architecture

```text
Space-Debris-Identification/
├── configs/                  # Global hyperparameter settings & YAML config parser
│   ├── base_config.yaml
│   └── config.py
├── saved_models/             # Export directory for trained model weights (.h5)
├── plots/                    # Evaluation plots & Grad-CAM audit outputs
│   ├── cnn/                  # Learning curves, confusion matrix, ROC, PR curves
│   └── gradcam_audit/        # Zero-trust Grad-CAM heatmap overlays with colorbars
├── scripts/                  # Offline Caching & Deduplication Engines
│   ├── cache_dataset.py      # Precomputes 224x224 zero-g square padded crops
│   └── deduplicate_cache.py  # Parallel pHash perceptual deduplicator
├── src/                      # Core System Package
│   ├── data/                 # Data loading, group splitting & augmentation
│   │   ├── loader.py         # GroupShuffleSplit trajectory parser
│   │   ├── preprocessing.py  # Architecture-specific tensor router & Sequence generator
│   │   └── augmentation.py   # Solar glare, sensor noise & photometric jitter
│   ├── models/               # Factory Pattern Model Architecture Builder
│   │   ├── factory.py        # ModelFactory pattern registry
│   │   ├── cnn.py            # Deep Custom CNN builder
│   │   ├── mobilenet.py      # MobileNetV2 builder
│   │   ├── resnet.py         # ResNet50 builder
│   │   └── efficientnet_builder.py # EfficientNetB0 builder
│   ├── evaluation/           # Metrics calculation & Grad-CAM visual auditor
│   │   ├── metrics.py        # Evaluation pipeline & PR-threshold optimization
│   │   └── gradcam.py        # Zero-trust Grad-CAM audit engine
│   ├── inference/            # Prediction engine wrapper
│   │   └── predictor.py      # DebrisPredictor inference handler
│   └── utils/                # GPU memory growth initialization
│       └── gpu.py
├── train.py                  # CLI orchestrator for Phase 1 & Phase 2 training
├── predict.py                # Unified CLI entrypoint for inference
├── Dockerfile                # GPU container configuration
└── README.md
```

---

## 🚀 Execution Guide

### 1. Offline Dataset Caching & Perceptual Deduplication

```powershell
# 1. Precompute 224x224 padded crops from raw SPARK-2022 dataset
python scripts/cache_dataset.py --spark-dir SPARK-2022 --target-dir SPARK-2022-Preprocessed

# 2. Run pHash perceptual deduplication across all splits (Hamming distance <= 2)
python scripts/deduplicate_cache.py --cache-dir SPARK-2022-Preprocessed --threshold 2
```

### 2. Model Training

```powershell
# Train Custom CNN (Default 224x224 resolution)
python train.py --model cnn --epochs 25 --batch-size 32

# Train Transfer Learning Models (EfficientNet / MobileNet / ResNet)
python train.py --model efficientnet --epochs 20
```

### 3. Inference & Grad-CAM Visual Audit

```powershell
# Single Image Inference
python predict.py --image "sample_debris/img022768.jpg" --model "saved_models/cnn_spark_debris.h5" --type cnn

# Run Zero-Trust Grad-CAM Visual Audit
python -m src.evaluation.gradcam --model saved_models/cnn_spark_debris.h5 --image "sample_debris/img022768.jpg" --model-type cnn --color-mode grayscale
```
