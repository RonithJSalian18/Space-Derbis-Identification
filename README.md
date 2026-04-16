# 🚀 Space Debris Detection using CNN

A deep learning project that classifies images as **Space Debris** or **Non-Debris** using a Convolutional Neural Network (CNN) built with TensorFlow and OpenCV.

---

## 📌 Project Overview

This project uses computer vision and deep learning to automatically detect space debris from images. It includes:

- Image preprocessing (grayscale + normalization)
- CNN model for binary classification
- Data augmentation for generalization
- Train / Validation / Test split (70/15/15)
- Model evaluation with real metrics
- Prediction system for new images

---

## 🧠 Model Architecture

- Conv2D → BatchNormalization → MaxPooling
- Conv2D → BatchNormalization → MaxPooling
- Conv2D → BatchNormalization → MaxPooling
- Flatten → Dense(128) → Dropout(0.6)
- Output Layer (Sigmoid)

---

## 📂 Dataset Structure

```id="ds2"
dataset/
│
├── debris/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
│
└── non_debris/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

- 📦 Size: ~1.5GB
- 🖼️ Images: ~10,000
- 🏷️ Classes:
  - `debris` → 0
  - `non_debris` → 1

---

## ⚙️ Environment Setup

### 1. Create Virtual Environment

```bash id="env11"
python -m venv tf-gpu-env
```

### 2. Activate Environment

#### Windows:

```bash id="env22"
tf-gpu-env\Scripts\activate
```

#### Linux/Mac:

```bash id="env33"
source tf-gpu-env/bin/activate
```

---

### 3. Install Dependencies

```bash id="env44"
pip install tensorflow==2.10.1 numpy<2 opencv-python matplotlib scikit-learn
```

---

## 🖥️ GPU Setup (Optional)

Ensure:

- NVIDIA GPU
- CUDA + cuDNN installed

Check GPU:

```python id="gpu2"
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## ▶️ How to Run

### Step 1: Add Dataset ZIP

```id="run11"
dataset.zip
```

### Step 2: Run Script

```bash id="run22"
python cnn.py
```

---

## 🔄 Pipeline

1. Extract dataset
2. Load & preprocess images
3. Shuffle dataset
4. Split into train / validation / test
5. Apply data augmentation
6. Train CNN
7. Evaluate on test set
8. Save model

---

## 📊 Final Results

```id="res11"
Accuracy: 88%

Class-wise Performance:

Debris:
Precision: 0.81
Recall:    1.00
F1-score:  0.89

Non-Debris:
Precision: 1.00
Recall:    0.76
F1-score:  0.86
```

---

## 🧠 Model Interpretation

- Detects all debris (Recall = 1.00)
- Occasionally misclassifies non-debris as debris
- Designed to prioritize safety (high recall)

---

## 🔍 Prediction Code

```python id="pred11"
from tensorflow.keras.models import load_model
import cv2, numpy as np

model = load_model("debris_model.h5")

def predict(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128,128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.expand_dims(img, axis=(0,-1))

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        print("Non-Debris")
    else:
        print("Debris")
```

---

## 📁 Output Files

- `debris_model.h5` → trained model
- Training logs
- Evaluation metrics

---
