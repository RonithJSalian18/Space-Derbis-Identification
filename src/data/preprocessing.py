"""
Preprocessing & Optimized Batch Generator Pipeline for Space Debris Identification.

Includes offline zero-g square padding utilities for initial caching as well as
an ultra-fast SparkDataGenerator (tf.keras.utils.Sequence) designed to load 
preprocessed 224x224 images directly without runtime I/O or cropping bottlenecks.
"""

import os
import cv2
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from configs import IMAGE_SIZE


def center_crop_and_resize(img: np.ndarray, target_size: tuple = IMAGE_SIZE) -> np.ndarray:
    """
    Fallback method: Crops a square from the dead center of the image and resizes to target_size.
    """
    h, w = img.shape[:2]
    min_dim = min(h, w)

    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2

    cropped_img = img[start_y:start_y + min_dim, start_x:start_x + min_dim]
    resized_img = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_AREA)
    return resized_img


def crop_bbox_and_pad_square(img: np.ndarray, bbox: list, target_size: tuple = IMAGE_SIZE) -> np.ndarray:
    """
    Bounding-Box Cropping Pipeline for SPARK-2022 dataset caching:
    1. Crops raw high-res image (1024x1024) exactly to the [xmin, ymin, xmax, ymax] bounding box.
    2. Applies zero-g padding (black constant fill) to convert the rectangular crop into 
       a perfect square without aspect-ratio distortion leakage.
    3. Resizes the square container to target_size (224x224).

    Args:
        img (np.ndarray): Raw input image array from cv2.imread or cv2.imdecode (1024x1024).
        bbox (list): Bounding box coordinates [xmin, ymin, xmax, ymax].
        target_size (tuple): Desired output dimensions (height, width).

    Returns:
        np.ndarray: Square-padded and resized image array.
    """
    if bbox is None or len(bbox) != 4:
        return center_crop_and_resize(img, target_size=target_size)

    h_img, w_img = img.shape[:2]
    xmin, ymin, xmax, ymax = bbox

    # Clamp bounding box coordinates safely within image boundaries
    xmin = max(0, min(int(xmin), w_img - 1))
    ymin = max(0, min(int(ymin), h_img - 1))
    xmax = max(xmin + 1, min(int(xmax), w_img))
    ymax = max(ymin + 1, min(int(ymax), h_img))

    # Crop raw high-resolution image exactly to the bounding box
    cropped = img[ymin:ymax, xmin:xmax]

    h_crop, w_crop = cropped.shape[:2]
    if h_crop == 0 or w_crop == 0:
        return center_crop_and_resize(img, target_size=target_size)

    # Calculate zero-g padding to make the crop a perfect square
    max_dim = max(h_crop, w_crop)
    pad_top = (max_dim - h_crop) // 2
    pad_bottom = max_dim - h_crop - pad_top
    pad_left = (max_dim - w_crop) // 2
    pad_right = max_dim - w_crop - pad_left

    # Apply constant black border fill (0 for space background)
    if len(cropped.shape) == 3:
        square_img = cv2.copyMakeBorder(
            cropped, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    else:
        square_img = cv2.copyMakeBorder(
            cropped, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0
        )

    # Resize square padded crop to 224x224
    resized_img = cv2.resize(square_img, target_size, interpolation=cv2.INTER_AREA)
    return resized_img


def preprocess_image(
    path: str,
    bbox: list = None,
    zip_path: str = None,
    zip_filename: str = None,
    target_size: tuple = IMAGE_SIZE,
    color_mode: str = "grayscale"
) -> np.ndarray:
    """
    Preprocesses a single image for training or inference.
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
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
    else:
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img.astype(np.float32) / 255.0

    return img


class SparkDataGenerator(tf.keras.utils.Sequence):
    """
    Production High-Speed Keras Sequence Generator for Offline Preprocessed Datasets.

    Loads pre-cropped 224x224 images directly from disk in mini-batches, converting them
    to normalized float32 tensors on-the-fly without runtime cropping or resizing overhead.
    """

    def __init__(
        self,
        records: list,
        batch_size: int = 32,
        target_size: tuple = IMAGE_SIZE,
        color_mode: str = "grayscale",
        shuffle: bool = True
    ):
        """
        Args:
            records (list): List of dict records containing 'path' (or 'cached_path') and 'label':
                            [{'path': str, 'label': int}, ...]
            batch_size (int): Number of samples per batch (default: 32).
            target_size (tuple): Image resolution (default: (224, 224)).
            color_mode (str): Output color format ('grayscale' or 'rgb').
            shuffle (bool): Shuffle records at epoch end (default: True).
        """
        self.records = list(records)
        self.batch_size = batch_size
        self.target_size = target_size
        self.color_mode = color_mode
        self.shuffle = shuffle
        self.indices = np.arange(len(self.records))
        self.on_epoch_end()

    def __len__(self) -> int:
        """Denotes the total number of batches per epoch."""
        return int(np.ceil(len(self.records) / float(self.batch_size)))

    def __getitem__(self, index: int) -> tuple:
        """
        Generates one mini-batch of preprocessed image tensors directly from cached disk files.

        Args:
            index (int): Batch index.

        Returns:
            tuple: (X_batch, y_batch) float32 image batch and int32 target label array.
        """
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        batch_records = [self.records[k] for k in batch_indices]

        X_list, y_list = [], []

        for record in batch_records:
            if isinstance(record, dict):
                path = record.get("path") or record.get("cached_path")
                label = record.get("label")
            else:
                path, label = record[0], record[1]

            if not path or not os.path.exists(path):
                continue

            # Direct fast read of preprocessed 224x224 image
            img = cv2.imread(path)
            if img is None:
                continue

            if self.color_mode == "grayscale":
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                elif len(img.shape) == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=-1)  # (H, W, 1)
            else:  # RGB mode
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = img.astype(np.float32) / 255.0

            X_list.append(img)
            y_list.append(label)

        if len(X_list) == 0:
            channels = 1 if self.color_mode == "grayscale" else 3
            return (
                np.zeros((0, self.target_size[0], self.target_size[1], channels), dtype=np.float32),
                np.zeros((0,), dtype=np.int32)
            )

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)

    def on_epoch_end(self):
        """Shuffles record indices at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


def load_dataset_in_memory(records: list, target_size: tuple = IMAGE_SIZE, color_mode: str = "grayscale") -> tuple:
    """
    Legacy helper function for small subset loading into memory.
    """
    generator = SparkDataGenerator(records, batch_size=len(records) or 1, target_size=target_size, color_mode=color_mode, shuffle=False)
    if len(generator) == 0:
        channels = 1 if color_mode == "grayscale" else 3
        return np.zeros((0, target_size[0], target_size[1], channels), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return generator[0]


def get_data_generators(color_mode: str = "grayscale") -> tuple:
    """
    Returns spatial zero-g ImageDataGenerators for training and validation.
    """
    train_datagen = ImageDataGenerator(
        rotation_range=180,       # Full 180-degree zero-g rotation invariance
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,      # Zero-g spatial flip
        vertical_flip=True,        # Zero-g spatial flip
        fill_mode='constant',
        cval=0.0                  # Black space background fill
    )

    val_datagen = ImageDataGenerator()

    return train_datagen, val_datagen
