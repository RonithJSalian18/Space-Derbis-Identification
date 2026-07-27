"""
Unified Command Line Training Script for Space Debris Identification.

Usage examples:
    python train.py --model cnn --epochs 30 --batch-size 32
    python train.py --model resnet --epochs 20
    python train.py --model mobilenet
"""
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from configs import SEED, BATCH_SIZE, EPOCHS, LEARNING_RATE, SAVED_MODELS_DIR, DATASET_ZIP_PATH, DATASET_EXTRACT_DIR
from src.utils import setup_gpu
from src.data import get_cleaned_dataset, load_dataset_in_memory, get_data_generators
from src.models import get_model
from src.evaluation import evaluate_and_plot, plot_learning_curves


def main():
    parser = argparse.ArgumentParser(description="Train Space Debris Identification Models")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "mobilenet", "resnet", "efficientnet"],
                        help="Model architecture to train (default: cnn)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Number of training epochs (default: {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--zip", type=str, default=DATASET_ZIP_PATH, help="Path to dataset.zip file")
    parser.add_argument("--extract-to", type=str, default=DATASET_EXTRACT_DIR, help="Path to extract dataset")
    parser.add_argument("--keep-duplicates", action="store_true", help="Skip duplicate image removal step")

    args = parser.parse_args()

    # Set random seeds & activate GPU
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    setup_gpu()

    print("==================================================")
    print(f"[+] STARTING TRAINING PIPELINE: Architecture = {args.model.upper()}")
    print("==================================================")

    # 1. Load and clean image paths
    paths, labels = get_cleaned_dataset(
        base_dir=args.extract_to,
        remove_duplicates=not args.keep_duplicates
    )

    if len(paths) == 0:
        raise ValueError("[-] No image files found in dataset! Please ensure dataset is placed correctly.")

    # 2. Build model architecture & determine color mode
    model, color_mode = get_model(architecture_name=args.model, learning_rate=args.lr)
    model.summary()

    # 3. Load dataset images into memory (Resized to 224x224 and normalized by 255.0)
    print(f"[+] Preprocessing images into memory (Color mode: {color_mode})...")
    X, y = load_dataset_in_memory(paths, labels, color_mode=color_mode)

    # 4. Train / Validation / Test split (70% train, 15% val, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

    print(f"[+] Dataset split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 5. Class Balancing: Calculate class weights automatically
    classes_arr = np.unique(y_train)
    class_weights_vals = compute_class_weight(
        class_weight='balanced',
        classes=classes_arr,
        y=y_train
    )
    class_weight_dict = dict(zip(classes_arr, class_weights_vals))
    print(f"[+] Computed Class Weights: {class_weight_dict}")

    # 6. Callbacks: EarlyStopping & ReduceLROnPlateau for learning rate stability
    save_path = os.path.join(SAVED_MODELS_DIR, f"{args.model}_debris.h5")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    # 7. Data Augmentation & Batching (Batch Size = 32)
    train_datagen, val_datagen = get_data_generators(color_mode=color_mode)
    train_gen = train_datagen.flow(X_train, y_train, batch_size=args.batch_size, seed=SEED)

    # 8. Model Training Loop
    print(f"\n[+] Training {args.model.upper()} for {args.epochs} epochs (Batch Size={args.batch_size}, LR={args.lr})...")
    history = model.fit(
        train_gen,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    print(f"\n[+] Model trained successfully and saved to: {save_path}")

    # 9. Evaluate on Test Set & Plot Learning Curves
    plot_dir = os.path.join("plots", args.model)
    plot_learning_curves(history, save_dir=plot_dir)
    evaluate_and_plot(model, X_test, y_test, save_dir=plot_dir)


if __name__ == "__main__":
    main()
