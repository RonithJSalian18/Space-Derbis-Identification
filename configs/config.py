"""
Configuration settings for Space Debris Identification project.
"""
import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
DATASET_ZIP_PATH = os.path.join(BASE_DIR, "dataset.zip")
DATASET_EXTRACT_DIR = os.path.join(BASE_DIR, "dataset")

# Classes Mapping
CLASS_MAPPING = {
    "debris": 0,
    "non_debris": 1
}
CLASS_NAMES = ["Debris", "Non-Debris"]

# Default Hyperparameters (Optimized for 224x224 Space Debris Images)
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4
LABEL_SMOOTHING = 0.1
SEED = 42

# Ensure saved models directory exists
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
