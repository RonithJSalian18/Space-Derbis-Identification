"""
Production-Ready Command Line Training Script for Space Debris Identification.

Includes two-phase training (Warmup -> Fine-Tuning) and duplicate purging safeguards
to prevent mode collapse, inverted ROC curves, and pitch-black image hash over-purging.

Usage examples:
    python train.py --model efficientnet --epochs 30
    python train.py --model cnn --epochs 25
    python train.py --model mobilenet --epochs 30
"""
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from configs import (
    AppConfig, SEED, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    SAVED_MODELS_DIR, DATASET_EXTRACT_DIR
)
from src.utils import setup_gpu
from src.data import get_cleaned_dataset, load_dataset_in_memory, get_data_generators
from src.models import ModelFactory, unfreeze_efficientnet
from src.training import get_callbacks
from src.evaluation import evaluate_and_plot, plot_learning_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train Space Debris Identification Models")
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet",
        choices=["cnn", "custom_cnn", "mobilenet", "resnet", "efficientnet"],
        help="Model architecture to train (default: efficientnet)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Total training epochs across both phases (default: {EPOCHS})"
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Number of Phase 1 warmup epochs for classification head (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Alias for learning rate (sets Phase 1 warmup learning rate)"
    )
    parser.add_argument(
        "--lr-phase1",
        type=float,
        default=1e-3,
        help="Learning rate for Phase 1 head warmup (default: 1e-3)"
    )
    parser.add_argument(
        "--lr-phase2",
        type=float,
        default=1e-4,
        help="Learning rate for Phase 2 backbone fine-tuning (default: 1e-4)"
    )
    parser.add_argument(
        "--extract-to",
        type=str,
        default=DATASET_EXTRACT_DIR,
        help="Path to extract dataset"
    )
    parser.add_argument(
        "--purge-duplicates",
        action="store_true",
        help="Explicitly enable pHash duplicate removal"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.lr is not None:
        args.lr_phase1 = args.lr

    # 1. Load Centralized Configuration
    config = AppConfig.load_from_yaml(args.config)
    seed = config.seed or SEED

    # Determine duplicate purging policy
    remove_duplicates = args.purge_duplicates

    # Random Seed & GPU Setup
    np.random.seed(seed)
    tf.random.set_seed(seed)
    setup_gpu()

    print("==================================================")
    print(f"[+] STARTING TWO-PHASE PIPELINE: Architecture = {args.model.upper()}")
    print(f"[+] Total Epochs: {args.epochs} | Warmup Epochs: {args.warmup_epochs} | Batch Size: {args.batch_size}")
    print(f"[+] Duplicate Purging Active: {remove_duplicates}")
    print("==================================================")

    # 2. Load and Clean Dataset
    paths, labels = get_cleaned_dataset(
        base_dir=args.extract_to,
        remove_duplicates=remove_duplicates
    )

    if len(paths) == 0:
        raise ValueError("[-] No image files found in dataset! Please check dataset extraction directory.")

    # 3. Build Model via Factory (Initial State: Base Model Completely Frozen)
    model, color_mode = ModelFactory.create_model(
        architecture_name=args.model,
        learning_rate=args.lr_phase1,
        label_smoothing=config.training.get("label_smoothing", 0.0),
        config=config.models.get(args.model, {})
    )
    print("\n[+] Initial Architecture Summary (Base Frozen):")
    model.summary()

    # 4. Preprocess Images into Memory
    print(f"\n[+] Preprocessing {len(paths)} images into memory (Color mode: {color_mode})...")
    X, y = load_dataset_in_memory(paths, labels, color_mode=color_mode)

    # 5. Train / Validation / Test Split (70% train, 15% val, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp)

    print(f"[+] Dataset split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 6. Class Weight Calculation
    classes_arr = np.unique(y_train)
    class_weights_vals = compute_class_weight(
        class_weight='balanced',
        classes=classes_arr,
        y=y_train
    )
    class_weight_dict = {int(c): float(w) for c, w in zip(classes_arr, class_weights_vals)}
    print(f"[+] Computed Class Weights: {class_weight_dict}")

    # 7. Setup Data Generators & Callbacks
    save_models_dir = config.checkpoint.get("saved_models_dir", SAVED_MODELS_DIR)
    save_path = os.path.join(save_models_dir, f"{args.model}_debris.h5")
    log_dir = os.path.join(config.checkpoint.get("log_dir", "plots/logs"), args.model)
    callbacks = get_callbacks(save_path=save_path, log_dir=log_dir)

    train_datagen, val_datagen = get_data_generators(color_mode=color_mode)
    train_gen = train_datagen.flow(X_train, y_train, batch_size=args.batch_size, seed=seed)
    val_gen = val_datagen.flow(X_val, y_val, batch_size=args.batch_size, shuffle=False)

    # -------------------------------------------------------------------------
    # PHASE 1: Transfer Learning Warmup (Train Classification Head Only)
    # -------------------------------------------------------------------------
    warmup_epochs = min(args.warmup_epochs, args.epochs)
    if warmup_epochs > 0:
        print("\n==================================================")
        print(f"[+] PHASE 1: Feature Extraction Warmup ({warmup_epochs} Epochs, LR={args.lr_phase1})")
        print("==================================================")

        # Recompile with Phase 1 Warmup Learning Rate
        model.compile(
            optimizer=Adam(learning_rate=args.lr_phase1, clipnorm=1.0),
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=config.training.get("label_smoothing", 0.0)),
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
        )

        history_phase1 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=warmup_epochs,
            class_weight=class_weight_dict,
            verbose=1
        )
        print("[+] Phase 1 Warmup Complete! Classification head initialized.")

    # -------------------------------------------------------------------------
    # PHASE 2: Fine-Tuning (Unfreeze Top Backbone Layers & Train Remaining Epochs)
    # -------------------------------------------------------------------------
    remaining_epochs = max(0, args.epochs - warmup_epochs)
    if remaining_epochs > 0:
        print("\n==================================================")
        print(f"[+] PHASE 2: Fine-Tuning Backbone ({remaining_epochs} Epochs, LR={args.lr_phase2})")
        print("==================================================")

        # Unfreeze top layers for Transfer Learning architectures
        if args.model in ["efficientnet", "mobilenet", "resnet"]:
            unfreeze_efficientnet(model, fine_tune_at=30)
            print("[+] Unfroze top 30 backbone layers (BatchNormalization locked in inference mode).")

        # Recompile with Fine-Tuning Learning Rate
        model.compile(
            optimizer=Adam(learning_rate=args.lr_phase2, clipnorm=1.0),
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=config.training.get("label_smoothing", 0.0)),
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
        )

        history_phase2 = model.fit(
            train_gen,
            validation_data=val_gen,
            initial_epoch=warmup_epochs,
            epochs=args.epochs,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        if warmup_epochs > 0:
            for key in history_phase1.history.keys():
                history_phase1.history[key].extend(history_phase2.history[key])
            history = history_phase1
        else:
            history = history_phase2
    else:
        history = history_phase1

    print(f"\n[+] Model trained successfully and weights saved to: {save_path}")

    # 8. Evaluation & Learning Curve Plotting
    plot_dir = os.path.join("plots", args.model)
    plot_learning_curves(history, save_dir=plot_dir)
    
    # Load best checkpoint weights before evaluation
    if os.path.exists(save_path):
        model.load_weights(save_path)
        print(f"[+] Restored best weights from {save_path} for final test evaluation.")

    evaluate_and_plot(model, X_test, y_test, save_dir=plot_dir)


if __name__ == "__main__":
    main()
