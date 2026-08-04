"""
Production-Ready Command Line Training Script for Space Debris Identification.

Uses offline-preprocessed cached 224x224 dataset images (SPARK-2022-Preprocessed) 
and Keras Custom Sequence Generators (SparkDataGenerator) for zero-I/O bottleneck, 
RAM-efficient GPU training across 110,000 images.

Usage examples:
    python train.py --model cnn --epochs 25
    python train.py --model efficientnet --epochs 30 --max-samples 5000
    python train.py --model mobilenet --epochs 20
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

from configs import (
    AppConfig, SEED, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    SAVED_MODELS_DIR
)
from src.utils import setup_gpu
from src.data import (
    load_cached_records,
    load_spark_split,
    get_cleaned_dataset,
    SparkDataGenerator
)
from src.models import ModelFactory, unfreeze_efficientnet
from src.training import get_callbacks
from src.evaluation import evaluate_and_plot, plot_learning_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train Space Debris Identification Models on SPARK-2022")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "custom_cnn", "mobilenet", "resnet", "efficientnet"],
        help="Model architecture to train (default: cnn)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="SPARK-2022-Preprocessed",
        help="Path to offline preprocessed dataset directory (default: SPARK-2022-Preprocessed)"
    )
    parser.add_argument(
        "--spark-dir",
        type=str,
        default="SPARK-2022",
        help="Path to root raw SPARK-2022 dataset directory (default: SPARK-2022)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Total training epochs across Phase 1 & Phase 2 (default: {EPOCHS})"
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
        "--max-samples",
        type=int,
        default=None,
        help="Optional max sample limit per split for rapid prototyping/benchmarking"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from existing saved model weights if available"
    )
    parser.add_argument(
        "--resume-weights",
        type=str,
        default=None,
        help="Optional explicit path to model weights file (.h5) to resume from"
    )
    parser.add_argument(
        "--initial-epoch",
        type=int,
        default=0,
        help="Epoch index to start/resume training from (default: 0)"
    )

    return parser.parse_args()


def load_dataset_records(split: str, cache_dir: str, spark_dir: str) -> list:
    """
    Attempts to load pre-cached 224x224 records from cache_dir.
    If the offline cache does not exist, falls back gracefully to raw SPARK-2022 metadata.
    """
    try:
        records = load_cached_records(split=split, cache_dir=cache_dir)
        print(f"[+] Using cached 224x224 dataset from: {os.path.abspath(cache_dir)} ({split})")
        return records
    except FileNotFoundError as e:
        print(f"[!] Warning: Offline cache not found ({e}). Falling back to raw SPARK-2022 parsing.")
        print(f"[💡 TIP] Run 'python scripts/cache_dataset.py' once to build offline cache for 10x faster training!")
        return load_spark_split(split=split, spark_dir=spark_dir)


def main():
    args = parse_args()

    # 1. Load Centralized Configuration & Environment Setup
    config = AppConfig.load_from_yaml(args.config)
    seed = config.seed or SEED

    np.random.seed(seed)
    tf.random.set_seed(seed)
    setup_gpu()

    print("==================================================")
    print(f"[+] STARTING SPARK-2022 TRAINING PIPELINE | Architecture: {args.model.upper()}")
    print(f"[+] Total Epochs: {args.epochs} | Warmup Epochs: {args.warmup_epochs} | Batch Size: {args.batch_size}")
    print(f"[+] Dataset Cache Path: {args.cache_dir}")
    print("==================================================")

    # 2. Ingest Dataset Records across Train, Val, and Test Splits
    train_records = load_dataset_records("train", cache_dir=args.cache_dir, spark_dir=args.spark_dir)
    val_records = load_dataset_records("val", cache_dir=args.cache_dir, spark_dir=args.spark_dir)
    test_records = load_dataset_records("test", cache_dir=args.cache_dir, spark_dir=args.spark_dir)

    # Optional sample limiting for fast prototyping
    if args.max_samples is not None and args.max_samples > 0:
        print(f"[+] Sampling max {args.max_samples} records per split for rapid prototyping...")
        debris_train = [r for r in train_records if r["label"] == 0]
        non_debris_train = [r for r in train_records if r["label"] == 1]
        n_deb = min(len(debris_train), max(1, args.max_samples // 10))
        n_non_deb = min(len(non_debris_train), args.max_samples - n_deb)
        train_records = debris_train[:n_deb] + non_debris_train[:n_non_deb]

        debris_val = [r for r in val_records if r["label"] == 0]
        non_debris_val = [r for r in val_records if r["label"] == 1]
        val_records = debris_val[:min(len(debris_val), 5)] + non_debris_val[:min(len(non_debris_val), 15)]

        debris_test = [r for r in test_records if r["label"] == 0]
        non_debris_test = [r for r in test_records if r["label"] == 1]
        test_records = debris_test[:min(len(debris_test), 5)] + non_debris_test[:min(len(non_debris_test), 15)]

    # 3. Instantiate Architecture via Factory Pattern
    model, color_mode = ModelFactory.create_model(
        architecture_name=args.model,
        learning_rate=args.lr_phase1,
        label_smoothing=config.training.get("label_smoothing", 0.0),
        config=config.models.get(args.model, {})
    )
    print(f"\n[+] Architecture '{args.model.upper()}' compiled (Color mode: {color_mode}):")
    model.summary()

    # 4. Instantiate High-Speed Keras Sequence Data Generators
    print(f"\n[+] Initializing SparkDataGenerators...")
    train_gen = SparkDataGenerator(
        train_records,
        batch_size=args.batch_size,
        color_mode=color_mode,
        model_type=args.model,
        shuffle=True
    )
    val_gen = SparkDataGenerator(
        val_records,
        batch_size=args.batch_size,
        color_mode=color_mode,
        model_type=args.model,
        shuffle=False
    )
    test_gen = SparkDataGenerator(
        test_records,
        batch_size=args.batch_size,
        color_mode=color_mode,
        model_type=args.model,
        shuffle=False
    )
    print(f"[+] Generator Batches per Epoch: Train={len(train_gen)}, Val={len(val_gen)}, Test={len(test_gen)}")

    # 5. Dynamically Calculate Class Weights directly from record labels (10:1 Imbalance Mitigation)
    y_train_labels = [record['label'] for record in train_records]
    classes_arr = np.unique(y_train_labels)
    class_weights_vals = compute_class_weight(
        class_weight='balanced',
        classes=classes_arr,
        y=y_train_labels
    )
    class_weight_dict = {0: 1.0, 1: 1.0}
    for c, w in zip(classes_arr, class_weights_vals):
        class_weight_dict[int(c)] = float(w)

    print(f"[+] Dynamically Computed Class Weights (10:1 Imbalance Mitigation): {class_weight_dict}")

    # 6. Setup Callbacks
    save_models_dir = config.checkpoint.get("saved_models_dir", SAVED_MODELS_DIR)
    os.makedirs(save_models_dir, exist_ok=True)
    save_path = os.path.join(save_models_dir, f"{args.model}_spark_debris.h5")
    log_dir = os.path.join(config.checkpoint.get("log_dir", "plots/logs"), args.model)
    callbacks = get_callbacks(save_path=save_path, log_dir=log_dir)

    # Check for resuming from existing checkpoint
    resume_path = args.resume_weights or save_path
    if (args.resume or args.resume_weights is not None) and os.path.exists(resume_path):
        try:
            model.load_weights(resume_path)
            print(f"[+] RESUME SUCCESS: Loaded previous weights from '{resume_path}'.")
        except Exception as e:
            print(f"[!] Warning: Could not load weights from '{resume_path}': {e}")

    # -------------------------------------------------------------------------
    # PHASE 1: Feature Extraction Warmup (Train Classification Head Only)
    # -------------------------------------------------------------------------
    warmup_epochs = min(args.warmup_epochs, args.epochs)
    if warmup_epochs > 0:
        print("\n==================================================")
        print(f"[+] PHASE 1: Feature Extraction Warmup ({warmup_epochs} Epochs, LR={args.lr_phase1})")
        print("==================================================")

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
    # PHASE 2: Fine-Tuning Backbone (Unfreeze Top Layers & Train Remaining Epochs)
    # -------------------------------------------------------------------------
    remaining_epochs = max(0, args.epochs - warmup_epochs)
    if remaining_epochs > 0:
        print("\n==================================================")
        print(f"[+] PHASE 2: Fine-Tuning Backbone ({remaining_epochs} Epochs, LR={args.lr_phase2})")
        print("==================================================")

        if args.model in ["efficientnet", "mobilenet", "resnet"]:
            unfreeze_efficientnet(model, fine_tune_at=30)
            print("[+] Unfroze top 30 backbone layers (BatchNormalization locked in inference mode).")

        model.compile(
            optimizer=Adam(learning_rate=args.lr_phase2, clipnorm=1.0),
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=config.training.get("label_smoothing", 0.0)),
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
        )

        start_epoch = max(warmup_epochs, args.initial_epoch)
        history_phase2 = model.fit(
            train_gen,
            validation_data=val_gen,
            initial_epoch=start_epoch,
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

    print(f"\n[+] Training complete! Best weights saved to: {save_path}")

    # 7. Evaluation & Metric Visualizations
    plot_dir = os.path.join("plots", args.model)
    plot_learning_curves(history, save_dir=plot_dir)

    if os.path.exists(save_path):
        model.load_weights(save_path)
        print(f"[+] Loaded best model checkpoint weights from {save_path} for final test evaluation.")

    y_test_labels = np.array([r['label'] for r in test_records], dtype=np.int32)
    evaluate_and_plot(model, test_gen, y_test_labels, save_dir=plot_dir)


if __name__ == "__main__":
    main()
