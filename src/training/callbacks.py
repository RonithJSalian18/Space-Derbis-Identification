import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard


def get_callbacks(
    save_path: str,
    log_dir: str = "plots/logs",
    patience_early_stopping: int = 7,
    patience_reduce_lr: int = 3
) -> list:
    """
    Construct training callbacks including ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, and TensorBoard.
    Uses save_weights_only=True to prevent Keras HDF5 JSON serialization bugs with pretrained models.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience_early_stopping,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience_reduce_lr,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            save_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=0,
            write_graph=False
        )
    ]
    return callbacks
