"""
Diagnostic Script for Test B (Grouped Zero-Leakage Split Evaluation)
and Test C (Grad-CAM Heatmap Visualizer) for Space Debris Identification.

Usage:
    python test_grouped_split.py --model saved_models/efficientnet_debris.h5 --type efficientnet
"""
import os
import argparse
import numpy as np
from PIL import Image
import imagehash
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf

from configs import SEED, CLASS_NAMES
from src.data import get_cleaned_dataset, load_dataset_in_memory
from src.models import ModelFactory
from src.evaluation import evaluate_and_plot
from src.evaluation.gradcam import save_gradcam_visualizations


def compute_perceptual_cluster_groups(paths, threshold=8):
    """
    Group images into visual cluster IDs based on perceptual hashing (pHash).
    Images with pHash distance <= threshold belong to the same sequence/object cluster.
    """
    print(f"[+] Computing perceptual hash visual clusters across {len(paths)} images (Threshold={threshold})...")
    hashes = []
    groups = []
    current_group_id = 0

    for path in paths:
        try:
            h = imagehash.phash(Image.open(path))
        except Exception:
            h = None

        if h is None:
            groups.append(current_group_id)
            current_group_id += 1
            continue

        matched_group = None
        for prev_h, prev_group in hashes:
            if abs(h - prev_h) <= threshold:
                matched_group = prev_group
                break

        if matched_group is not None:
            groups.append(matched_group)
        else:
            groups.append(current_group_id)
            hashes.append((h, current_group_id))
            current_group_id += 1

    unique_groups = len(set(groups))
    print(f"[+] Formed {unique_groups} unique visual sequence groups across {len(paths)} images.")
    return np.array(groups)


def run_test_b_and_c(model_path, arch_type="efficientnet", seed=SEED):
    print("==================================================")
    print(f"[+] STARTING TEST B (Grouped Split) & TEST C (Grad-CAM Inspection)")
    print(f"[+] Model Checkpoint: {model_path} | Architecture: {arch_type.upper()}")
    print("==================================================")

    # 1. Load Dataset Paths & Labels
    paths, labels = get_cleaned_dataset(remove_duplicates=False)
    color_mode = "grayscale" if arch_type == "cnn" else "rgb"

    # 2. Compute Visual Sequence Clusters for Grouped Split (TEST B)
    groups = compute_perceptual_cluster_groups(paths, threshold=8)

    # 3. Perform Grouped Train/Test Split (80% Train, 20% Strict Test)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, test_idx = next(gss.split(paths, labels, groups))

    paths_train, labels_train = paths[train_idx], labels[train_idx]
    paths_test, labels_test = paths[test_idx], labels[test_idx]

    print(f"\n[+] TEST B Grouped Split Complete:")
    print(f"   |-- Train Set: {len(paths_train)} images across {len(set(groups[train_idx]))} groups")
    print(f"   +-- Test Set:  {len(paths_test)} images across {len(set(groups[test_idx]))} groups (ZERO frame leakage!)")

    # 4. Load Test Images into Memory
    print(f"\n[+] Loading {len(paths_test)} held-out Grouped Test images into memory...")
    X_test, y_test = load_dataset_in_memory(paths_test, labels_test, color_mode=color_mode)

    # 5. Load Model Checkpoint
    model, _ = ModelFactory.create_model(arch_type)
    if arch_type in ["efficientnet", "mobilenet", "resnet"]:
        from src.models import unfreeze_efficientnet
        unfreeze_efficientnet(model, 30)

    if os.path.exists(model_path):
        model.load_weights(model_path)
        print(f"[+] Successfully loaded trained weights from: {model_path}")
    else:
        print(f"[-] Warning: Checkpoint '{model_path}' not found. Evaluating uncompiled model.")

    # 6. RUN TEST B: Evaluate Model on Strict Zero-Leakage Grouped Test Set
    print("\n==================================================")
    print("[+] TEST B RESULTS: Zero-Leakage Grouped Split Evaluation")
    print("==================================================")
    plot_dir = os.path.join("plots", "grouped_split", arch_type)
    results = evaluate_and_plot(model, X_test, y_test, class_names=CLASS_NAMES, save_dir=plot_dir)

    # 7. RUN TEST C: Grad-CAM Heatmap Inspection
    print("\n==================================================")
    print("[+] TEST C RESULTS: Grad-CAM Heatmap Inspection")
    print("==================================================")
    gradcam_dir = os.path.join("plots", "gradcam", arch_type)
    saved_gradcam_files = save_gradcam_visualizations(
        model=model,
        X_test=X_test,
        y_test=y_test,
        class_names=CLASS_NAMES,
        save_dir=gradcam_dir,
        num_samples=6
    )

    print("\n==================================================")
    print("[+] DIAGNOSTIC SUITE COMPLETE!")
    print(f"   |-- Grouped Test Accuracy: {np.mean((model.predict(X_test).ravel() > 0.5) == y_test):.4f}")
    print(f"   |-- Grouped Evaluation Plots: {os.path.abspath(plot_dir)}")
    print(f"   +-- Grad-CAM Heatmaps:        {os.path.abspath(gradcam_dir)}")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test B & Test C Evaluation Suite")
    parser.add_argument("--model", type=str, default="saved_models/efficientnet_debris.h5", help="Path to model weights")
    parser.add_argument("--type", type=str, default="efficientnet", choices=["cnn", "mobilenet", "resnet", "efficientnet"], help="Architecture type")
    args = parser.parse_args()

    run_test_b_and_c(args.model, args.type)
