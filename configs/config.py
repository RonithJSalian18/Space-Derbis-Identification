"""
Configuration settings and Loader for Space Debris Identification project.
"""
import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List

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

# Default Hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001
LABEL_SMOOTHING = 0.1
SEED = 42

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)


@dataclass
class AppConfig:
    experiment_name: str = "space_debris_identification"
    seed: int = 42
    data: Dict[str, Any] = field(default_factory=lambda: {
        "dataset_zip_path": DATASET_ZIP_PATH,
        "extract_dir": DATASET_EXTRACT_DIR,
        "image_size": (224, 224),
        "batch_size": 32,
        "keep_duplicates": False,
        "class_mapping": CLASS_MAPPING,
        "class_names": CLASS_NAMES,
    })
    training: Dict[str, Any] = field(default_factory=lambda: {
        "epochs": 30,
        "learning_rate": 0.0001,
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "label_smoothing": 0.1,
        "use_class_weights": True,
        "clipnorm": 1.0,
    })
    checkpoint: Dict[str, Any] = field(default_factory=lambda: {
        "saved_models_dir": SAVED_MODELS_DIR,
        "log_dir": "plots/logs",
    })
    models: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load_from_yaml(cls, yaml_path: str = None) -> "AppConfig":
        if yaml_path is None:
            yaml_path = os.path.join(BASE_DIR, "configs", "base_config.yaml")

        if not os.path.exists(yaml_path):
            return cls()

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        if "data" in data and "image_size" in data["data"]:
            data["data"]["image_size"] = tuple(data["data"]["image_size"])

        return cls(
            experiment_name=data.get("experiment_name", "space_debris_identification"),
            seed=data.get("seed", 42),
            data=data.get("data", {}),
            training=data.get("training", {}),
            checkpoint=data.get("checkpoint", {}),
            models=data.get("models", {}),
        )
