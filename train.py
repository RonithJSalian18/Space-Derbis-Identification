"""
Production-Ready Command Line Training Script for Space Debris Identification.

Uses offline-preprocessed cached 224x224 dataset images (SPARK-2022-Preprocessed) 
and Keras Custom Sequence Generators (SparkDataGenerator) for zero-I/O bottleneck, 
RAM-efficient GPU training across 110,000 images.

Usage examples:
    python train.py --model resnet --epochs 25
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
from src.models import (
    ModelFactory,
    unfreeze_backbone,
    unfreeze_resnet,
    unfreeze_efficientnet,
    unfreeze_mobilenet
)
from src.training import get_callbacks
from src.evaluation import evaluate_and_plot, plot_learning_curves, find_optimal_threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Train Space Debris Identification Models on SPARK-2022")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet",
        choices=["cnn", "custom_cnn", "mobilenet", "resnet", "efficientnet"],
        help="Model architecture to train (default: resnet)"
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
        default=None,
        help="Learning rate for Phase 1 head warmup (default from config: 5e-4)"
    )
    parser.add_argument(
        "--lr-phase2",
        type=float,
        default=None,
        help="Learning rate for Phase 2 backbone fine-tuning (default from config: 2e-5)"
    )
    parser.add_argument(
        "--loss",
        type=str,
        default=None,
        choices=["binary_crossentropy", "focal", "binary_focal_crossentropy"],
        help="Loss function type (default from config: binary_crossentropy)"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=None,
        help="Label smoothing factor (default from config: 0.05)"
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
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable space-domain augmentations (for ablation study comparisons)"
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


def get_loss_function(loss_name: str = "binary_crossentropy", label_smoothing: float = 0.05):
    """Instantiates stabilized Loss function with label smoothing."""
    loss_lower = (loss_name or "binary_crossentropy").lower()
    if "focal" in loss_lower:
        if hasattr(tf.keras.losses, "BinaryFocalCrossentropy"):
            return tf.keras.losses.BinaryFocalCrossentropy(
                gamma=2.0,
                label_smoothing=label_smoothing,
                from_logits=False
            )
    return tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)


def unfreeze_model_backbone(model: tf.keras.Model, arch_name: str, config_dict: dict = None):
    """Directs model backbone to architecture-specific block-aware unfreezing via unified unfreeze_backbone."""
    config_dict = config_dict or {}
    detected = unfreeze_backbone(model, arch_name=arch_name, **config_dict)
    print(f"[+] Unfroze {detected or arch_name} backbone (BatchNormalization locked in inference mode, config={config_dict}).")


def main():
    args = parse_args()

    # 1. Load Centralized Configuration & Environment Setup
    config = AppConfig.load_from_yaml(args.config)
    seed = config.seed or SEED

    np.random.seed(seed)
    tf.random.set_seed(seed)
    setup_gpu()

    # Determine hyperparameters with CLI overrides
    lr_phase1 = args.lr_phase1 if args.lr_phase1 is not None else float(config.training.get("lr_phase1", 5e-4))
    lr_phase2 = args.lr_phase2 if args.lr_phase2 is not None else float(config.training.get("lr_phase2", 2e-5))
    loss_name = args.loss or config.training.get("loss", "binary_crossentropy")
    label_smoothing = args.label_smoothing if args.label_smoothing is not None else float(config.training.get("label_smoothing", 0.05))
    clipnorm = float(config.training.get("clipnorm", 1.0))

    print("==================================================")
    print(f"[+] STARTING SPARK-2022 TRAINING PIPELINE | Architecture: {args.model.upper()}")
    print(f"[+] Total Epochs: {args.epochs} | Warmup Epochs: {args.warmup_epochs} | Batch Size: {args.batch_size}")
    print(f"[+] Phase 1 LR: {lr_phase1} | Phase 2 LR: {lr_phase2} | Loss: {loss_name} (Smoothing: {label_smoothing})")
    print(f"[+] Dataset Cache Path: {args.cache_dir}")
    print("==================================================")

    # 2. Ingest Dataset Records across Train, Val, and Test Splits
    train_records = load_dataset_records("train", cache_dir=args.cache_dir, spark_dir=args.spark_dir)
    val_records = load_dataset_records("val", cache_dir=args.cache_dir, spark_dir=args.spark_dir)
    test_records = load_dataset_records("test", cache_dir=args.cache_dir, spark_dir=args.spark_dir)

    # Optional sample limiting for fast prototyping
    if args.max_samples is not None and args.max_samples > 0:
        print(f"[+] Sampling max {args.max_samples} records per split using randomized sampling (seed={seed})...")
        rng = np.random.default_rng(seed)
        debris_train = [r for r in train_records if r["label"] == 0]
        non_debris_train = [r for r in train_records if r["label"] == 1]
        n_deb = min(len(debris_train), max(1, args.max_samples // 10))
        n_non_deb = min(len(non_debris_train), args.max_samples - n_deb)
        deb_idx = rng.choice(len(debris_train), size=n_deb, replace=False) if len(debris_train) >= n_deb else range(len(debris_train))
        non_deb_idx = rng.choice(len(non_debris_train), size=n_non_deb, replace=False) if len(non_debris_train) >= n_non_deb else range(len(non_debris_train))
        train_records = [debris_train[i] for i in deb_idx] + [non_debris_train[i] for i in non_deb_idx]
        rng.shuffle(train_records)

        debris_val = [r for r in val_records if r["label"] == 0]
        non_debris_val = [r for r in val_records if r["label"] == 1]
        val_records = debris_val[:min(len(debris_val), 10)] + non_debris_val[:min(len(non_debris_val), 50)]

        debris_test = [r for r in test_records if r["label"] == 0]
        non_debris_test = [r for r in test_records if r["label"] == 1]
        test_records = debris_test[:min(len(debris_test), 10)] + non_debris_test[:min(len(non_debris_test), 50)]

    # 3. Instantiate Architecture via Factory Pattern
    model_cfg = config.models.get(args.model, {}).copy()
    model_cfg["loss"] = loss_name

    model, color_mode = ModelFactory.create_model(
        architecture_name=args.model,
        learning_rate=lr_phase1,
        label_smoothing=label_smoothing,
        config=model_cfg
    )
    print(f"\n[+] Architecture '{args.model.upper()}' compiled (Color mode: {color_mode}):")
    model.summary()

    # 4. Instantiate High-Speed Keras Sequence Data Generators
    augment_train = not args.no_augment
    print(f"\n[+] Initializing SparkDataGenerators (Space-Domain Augmentation: {augment_train})...")
    train_gen = SparkDataGenerator(
        train_records,
        batch_size=args.batch_size,
        color_mode=color_mode,
        model_type=args.model,
        shuffle=True,
        augment=augment_train
    )
    val_gen = SparkDataGenerator(
        val_records,
        batch_size=args.batch_size,
        color_mode=color_mode,
        model_type=args.model,
        shuffle=False,
        augment=False
    )
    test_gen = SparkDataGenerator(
        test_records,
        batch_size=args.batch_size,
        color_mode=color_mode,
        model_type=args.model,
        shuffle=False,
        augment=False
    )
    print(f"[+] Generator Batches per Epoch: Train={len(train_gen)}, Val={len(val_gen)}, Test={len(test_gen)}")

    # 5. Dynamically Calculate Class Weights directly from verified generator labels
    y_train_labels = train_gen.labels
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
    monitor_metric = config.training.get("monitor", "val_loss")
    callbacks = get_callbacks(
        save_path=save_path,
        log_dir=log_dir,
        patience_early_stopping=int(config.training.get("patience_early_stopping", 7)),
        patience_reduce_lr=int(config.training.get("patience_reduce_lr", 3)),
        monitor=monitor_metric
    )

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
    loss_fn = get_loss_function(loss_name=loss_name, label_smoothing=label_smoothing)

    if warmup_epochs > 0:
        print("\n==================================================")
        print(f"[+] PHASE 1: Feature Extraction Warmup ({warmup_epochs} Epochs, LR={lr_phase1})")
        print("==================================================")

        model.compile(
            optimizer=Adam(learning_rate=lr_phase1, clipnorm=clipnorm),
            loss=loss_fn,
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
                tf.keras.metrics.AUC(name='roc_auc', curve='ROC')
            ]
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
    # PHASE 2: Fine-Tuning Backbone (Unfreeze Top Blocks & Train Remaining Epochs)
    # -------------------------------------------------------------------------
    remaining_epochs = max(0, args.epochs - warmup_epochs)
    if remaining_epochs > 0:
        print("\n==================================================")
        print(f"[+] PHASE 2: Fine-Tuning Backbone ({remaining_epochs} Epochs, LR={lr_phase2})")
        print("==================================================")

        if args.model in ["efficientnet", "mobilenet", "resnet"]:
            unfreeze_model_backbone(model, args.model, model_cfg)

        model.compile(
            optimizer=Adam(learning_rate=lr_phase2, clipnorm=clipnorm),
            loss=loss_fn,
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
                tf.keras.metrics.AUC(name='roc_auc', curve='ROC')
            ]
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
        print(f"[+] Loaded best model checkpoint weights from {save_path} for final evaluation.")

    # STEP A: Tune optimal decision threshold strictly on VALIDATION split (no test leakage)
    print("\n[+] Calibrating decision threshold on VALIDATION split...")
    val_pred_probs = model.predict(val_gen).ravel()
    y_val_labels = val_gen.labels
    min_val_len = min(len(val_pred_probs), len(y_val_labels))
    val_pred_probs = val_pred_probs[:min_val_len]
    y_val_labels = y_val_labels[:min_val_len]

    optimal_threshold, best_val_f1, val_stats = find_optimal_threshold(y_val_labels, val_pred_probs, metric="f1")
    print(f"[+] Validation Calibration Complete -> Optimal Threshold: {optimal_threshold:.4f} (Val F1: {best_val_f1:.4f})")
    print(f"    (Val Precision: {val_stats.get('val_precision', 0):.4f} | Val Recall: {val_stats.get('val_recall', 0):.4f})")

    # STEP B: Evaluate on held-out TEST split using frozen validation threshold
    y_test_labels = test_gen.labels
    evaluate_and_plot(
        model=model,
        X_test=test_gen,
        y_test=y_test_labels,
        threshold=optimal_threshold,
        save_dir=plot_dir
    )


if __name__ == "__main__":
    main()

