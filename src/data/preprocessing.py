"""
Preprocessing & Architecture-Specific Tensor Normalization for Space Debris Identification.

Implements strict model-specific tensor preprocessing router:
- ResNet ('resnet', 'resnet50'): ImageNet mean subtraction & BGR conversion via tf.keras.applications.resnet.preprocess_input.
- MobileNet ('mobilenet', 'mobilenetv2'): Input scaled strictly to [-1, 1] via tf.keras.applications.mobilenet_v2.preprocess_input.
- EfficientNet ('efficientnet', 'efficientnetb0'): Raw [0, 255] float32 tensor (EfficientNet has native internal scaling).
- Custom CNN ('cnn'): Scaled to [0, 1] float32.

Includes reflection padding (cv2.BORDER_REFLECT_101) to eliminate artificial border artifacts.
"""

import os
import cv2
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from configs import IMAGE_SIZE


def apply_architecture_preprocessing(img: np.ndarray, model_type: str = "cnn") -> np.ndarray:
    """
    Applies strict model-specific normalization routing to an image array [0, 255] uint8/float32:
    - 'resnet' / 'resnet50': Converts RGB to BGR and applies ImageNet mean subtraction via resnet.preprocess_input.
    - 'mobilenet' / 'mobilenetv2': Scales pixels to range [-1.0, 1.0] via mobilenet_v2.preprocess_input.
    - 'efficientnet' / 'efficientnetb0': Retains raw [0.0, 255.0] float32 tensor for native internal scaling.
    - 'cnn': Scales pixels to range [0.0, 1.0] (image / 255.0).

    Args:
        img (np.ndarray): Image array with shape (H, W, C) in RGB or Grayscale format (values 0-255).
        model_type (str): Target model architecture ('resnet50', 'mobilenetv2', 'efficientnetb0', 'cnn').

    Returns:
        np.ndarray: Preprocessed float32 tensor matching target model input scale.
    """
    arch = (model_type or "cnn").lower().replace("_", "").replace("-", "")
    img_float = img.astype(np.float32)

    if "resnet" in arch:
        from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
        return resnet_preprocess(img_float)

    elif "mobilenet" in arch:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
        return mobilenet_preprocess(img_float)

    elif "efficientnet" in arch or "effinet" in arch:
        # EfficientNet has internal rescaling; return raw [0, 255] float32 tensor
        return img_float

    else:
        # Standard [0.0, 1.0] scaling for custom CNN
        return img_float / 255.0


def center_crop_and_resize(img: np.ndarray, target_size: tuple = IMAGE_SIZE) -> np.ndarray:
    """
    Pads the image to a square using reflection padding (cv2.BORDER_REFLECT_101)
    and resizes to target_size using cv2.INTER_AREA, eliminating zero-padding artifacts.
    """
    h, w = img.shape[:2]
    max_dim = max(h, w)

    pad_top = (max_dim - h) // 2
    pad_bottom = max_dim - h - pad_top
    pad_left = (max_dim - w) // 2
    pad_right = max_dim - w - pad_left

    square_img = cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
    )

    resized_img = cv2.resize(square_img, target_size, interpolation=cv2.INTER_AREA)
    return resized_img


def crop_bbox_and_pad_square(img: np.ndarray, bbox: list, target_size: tuple = IMAGE_SIZE) -> np.ndarray:
    """
    Bounding-Box Cropping Pipeline for SPARK-2022 dataset caching:
    1. Crops raw high-res image exactly to the [xmin, ymin, xmax, ymax] bounding box.
    2. Applies reflection padding (cv2.BORDER_REFLECT_101) to convert rectangular crop into a square,
       eliminating artificial constant black border frame artifacts that corrupt Grad-CAM heatmaps.
    3. Resizes square container to target_size (224x224).
    """
    if bbox is None or len(bbox) != 4:
        return center_crop_and_resize(img, target_size=target_size)

    h_img, w_img = img.shape[:2]
    xmin, ymin, xmax, ymax = bbox

    xmin = max(0, min(int(xmin), w_img - 1))
    ymin = max(0, min(int(ymin), h_img - 1))
    xmax = max(xmin + 1, min(int(xmax), w_img))
    ymax = max(ymin + 1, min(int(ymax), h_img))

    cropped = img[ymin:ymax, xmin:xmax]
    h_crop, w_crop = cropped.shape[:2]
    if h_crop == 0 or w_crop == 0:
        return center_crop_and_resize(img, target_size=target_size)

    max_dim = max(h_crop, w_crop)
    pad_top = (max_dim - h_crop) // 2
    pad_bottom = max_dim - h_crop - pad_top
    pad_left = (max_dim - w_crop) // 2
    pad_right = max_dim - w_crop - pad_left

    # Apply reflection padding to eliminate constant black border frame artifacts
    square_img = cv2.copyMakeBorder(
        cropped, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_REFLECT_101
    )

    resized_img = cv2.resize(square_img, target_size, interpolation=cv2.INTER_AREA)
    return resized_img


def preprocess_image(
    path: str,
    bbox: list = None,
    zip_path: str = None,
    zip_filename: str = None,
    target_size: tuple = IMAGE_SIZE,
    color_mode: str = "grayscale",
    model_type: str = "cnn"
) -> np.ndarray:
    """
    Preprocesses a single image file or zip archive entry with architecture-specific normalization.
    """
    img = None
    if path and os.path.exists(path):
        img = cv2.imread(path)

    if img is None and zip_path and os.path.exists(zip_path) and zip_filename:
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                img_bytes = z.read(zip_filename)
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            img = None

    if img is None:
        return None

    img = crop_bbox_and_pad_square(img, bbox=bbox, target_size=target_size)

    if color_mode == "grayscale":
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        img = np.expand_dims(img, axis=-1)
    else:
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img_tensor = apply_architecture_preprocessing(img, model_type=model_type)
    return img_tensor


class SparkDataGenerator(tf.keras.utils.Sequence):
    """
    Production High-Speed Keras Sequence Generator with Architecture-Specific Preprocessing Router.

    Removes universal [0, 1] scaling and applies exact normalization for ResNet50, MobileNetV2,
    EfficientNetB0, and Custom CNN directly inside __getitem__.
    """

    def __init__(
        self,
        records: list,
        batch_size: int = 32,
        target_size: tuple = IMAGE_SIZE,
        color_mode: str = "grayscale",
        model_type: str = "cnn",
        shuffle: bool = True
    ):
        """
        Args:
            records (list): List of dict records containing 'path' (or 'cached_path') and 'label'.
            batch_size (int): Number of samples per batch (default: 32).
            target_size (tuple): Image resolution (default: (224, 224)).
            color_mode (str): Output color format ('grayscale' or 'rgb').
            model_type (str): Target model architecture ('resnet50', 'mobilenetv2', 'efficientnetb0', 'cnn').
            shuffle (bool): Shuffle records at epoch end (default: True).
        """
        self.records = list(records)
        self.batch_size = batch_size
        self.target_size = target_size
        self.color_mode = color_mode
        self.model_type = model_type
        self.shuffle = shuffle
        self.indices = np.arange(len(self.records))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.records) / float(self.batch_size)))

    def __getitem__(self, index: int) -> tuple:
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        batch_records = [self.records[k] for k in batch_indices]
        num_samples = len(batch_records)

        channels = 1 if self.color_mode == "grayscale" else 3
        X_batch = np.zeros((num_samples, self.target_size[0], self.target_size[1], channels), dtype=np.float32)
        y_batch = np.zeros((num_samples,), dtype=np.int32)

        valid_idx = 0
        for record in batch_records:
            if isinstance(record, dict):
                path = record.get("path") or record.get("cached_path")
                label = record.get("label")
            else:
                path, label = record[0], record[1]

            if not path or not os.path.exists(path):
                continue

            img = cv2.imread(path)
            if img is None:
                continue

            if self.color_mode == "grayscale":
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                elif len(img.shape) == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                img = np.expand_dims(img, axis=-1)
            else:
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            processed_tensor = apply_architecture_preprocessing(img, model_type=self.model_type)
            X_batch[valid_idx] = processed_tensor
            y_batch[valid_idx] = label
            valid_idx += 1

        if valid_idx < num_samples:
            X_batch = X_batch[:valid_idx]
            y_batch = y_batch[:valid_idx]

        return X_batch, y_batch

    def on_epoch_end(self):
        import gc
        gc.collect()
        if self.shuffle:
            np.random.shuffle(self.indices)


def load_dataset_in_memory(records: list, target_size: tuple = IMAGE_SIZE, color_mode: str = "grayscale", model_type: str = "cnn") -> tuple:
    """
    Helper function for loading subset dataset directly into RAM memory arrays.
    """
    generator = SparkDataGenerator(records, batch_size=len(records) or 1, target_size=target_size, color_mode=color_mode, model_type=model_type, shuffle=False)
    if len(generator) == 0:
        channels = 1 if color_mode == "grayscale" else 3
        return np.zeros((0, target_size[0], target_size[1], channels), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return generator[0]


def get_data_generators(color_mode: str = "grayscale") -> tuple:
    """
    Returns spatial zero-g ImageDataGenerators for training and validation with reflection padding.
    """
    train_datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'  # Use reflection fill to eliminate border frame artifacts
    )
    val_datagen = ImageDataGenerator()
    return train_datagen, val_datagen
