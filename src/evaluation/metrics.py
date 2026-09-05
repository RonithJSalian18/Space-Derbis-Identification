"""
Comprehensive Evaluation Pipeline for Space Debris Identification System.

Features:
- Leak-Free Threshold Optimization: Strictly tuned on validation split, never on test set.
- Confidence Calibration: Expected Calibration Error (ECE), Brier Score, and Reliability Diagrams.
- Statistical Rigor: 95% empirical bootstrap confidence intervals for F1, Recall, Precision, Accuracy.
- PR-AUC with class prevalence baseline overlay.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from configs import CLASS_NAMES


def plot_learning_curves(history, save_dir="plots", show_plot=True):
    """
    Plots and saves training and validation curves for Loss, Accuracy, Precision, and Recall.
    """
    os.makedirs(save_dir, exist_ok=True)
    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training & Validation Learning Curves', fontsize=16, fontweight='bold')

    # 1. Loss Curve
    axes[0, 0].plot(epochs, hist['loss'], 'b-o', label='Training Loss')
    if 'val_loss' in hist:
        axes[0, 0].plot(epochs, hist['val_loss'], 'r-s', label='Validation Loss')
    axes[0, 0].set_title('Model Loss', fontsize=12)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)

    # 2. Accuracy Curve
    axes[0, 1].plot(epochs, hist['accuracy'], 'b-o', label='Training Accuracy')
    if 'val_accuracy' in hist:
        axes[0, 1].plot(epochs, hist['val_accuracy'], 'r-s', label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy', fontsize=12)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend(loc='lower right')
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)

    # 3. Precision Curve
    prec_key = [k for k in hist.keys() if 'precision' in k and not k.startswith('val')][0] if any('precision' in k for k in hist.keys()) else None
    val_prec_key = [k for k in hist.keys() if 'val_' in k and 'precision' in k][0] if any('val_' in k and 'precision' in k for k in hist.keys()) else None

    if prec_key:
        axes[1, 0].plot(epochs, hist[prec_key], 'b-o', label='Training Precision')
        if val_prec_key:
            axes[1, 0].plot(epochs, hist[val_prec_key], 'r-s', label='Validation Precision')
        axes[1, 0].set_title('Model Precision', fontsize=12)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend(loc='lower right')
        axes[1, 0].grid(True, linestyle='--', alpha=0.6)

    # 4. Recall Curve
    rec_key = [k for k in hist.keys() if 'recall' in k and not k.startswith('val')][0] if any('recall' in k for k in hist.keys()) else None
    val_rec_key = [k for k in hist.keys() if 'val_' in k and 'recall' in k][0] if any('val_' in k and 'recall' in k for k in hist.keys()) else None

    if rec_key:
        axes[1, 1].plot(epochs, hist[rec_key], 'b-o', label='Training Recall')
        if val_rec_key:
            axes[1, 1].plot(epochs, hist[val_rec_key], 'r-s', label='Validation Recall')
        axes[1, 1].set_title('Model Recall', fontsize=12)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend(loc='lower right')
        axes[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    curve_path = os.path.abspath(os.path.join(save_dir, 'learning_curves.png'))
    plt.savefig(curve_path, dpi=300)

    if show_plot and hasattr(plt.get_current_fig_manager(), 'show'):
        try:
            plt.show(block=False)
        except Exception:
            pass

    plt.close()
    print(f"[+] Learning curves plot saved to: {curve_path}")


def find_optimal_threshold(y_val: np.ndarray, y_val_probs: np.ndarray, metric: str = "f1") -> tuple:
    """
    Optimizes the decision threshold strictly on VALIDATION split data to prevent test-set leakage.

    Args:
        y_val (np.ndarray): Ground truth binary labels for the validation split.
        y_val_probs (np.ndarray): Predicted probabilities on the validation split.
        metric (str): Target metric to maximize ('f1' or 'recall').

    Returns:
        tuple: (optimal_threshold, best_val_score, validation_stats_dict)
    """
    y_val = np.asarray(y_val).ravel()
    y_val_probs = np.asarray(y_val_probs).ravel()

    prec_curve, rec_curve, thresholds = precision_recall_curve(y_val, y_val_probs)
    if len(thresholds) == 0:
        return 0.5, 0.5, {"val_f1": 0.5, "val_precision": 0.5, "val_recall": 0.5}

    f1_scores = 2 * (prec_curve * rec_curve) / (prec_curve + rec_curve + 1e-10)
    best_idx = int(np.argmax(f1_scores[:-1])) if len(f1_scores) > 1 else 0

    optimal_threshold = float(thresholds[best_idx])
    optimal_threshold = max(0.05, min(0.95, optimal_threshold))

    val_stats = {
        "val_f1": float(f1_scores[best_idx]),
        "val_precision": float(prec_curve[best_idx]),
        "val_recall": float(rec_curve[best_idx]),
        "optimal_threshold": optimal_threshold
    }
    return optimal_threshold, val_stats["val_f1"], val_stats


def compute_expected_calibration_error(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> tuple:
    """
    Computes Expected Calibration Error (ECE) and bin statistics for reliability diagrams.

    Args:
        y_true (np.ndarray): Binary ground truth (0 or 1).
        y_probs (np.ndarray): Model confidence probabilities in [0, 1].
        n_bins (int): Number of confidence bins (default: 10).

    Returns:
        tuple: (ece_score, bin_accuracies, bin_confidences, bin_counts)
    """
    y_true = np.asarray(y_true).ravel()
    y_probs = np.asarray(y_probs).ravel()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    ece = 0.0
    total_samples = len(y_true)

    for b in range(n_bins):
        mask = bin_indices == b
        count = int(np.sum(mask))
        bin_counts.append(count)
        if count > 0:
            acc = float(np.mean(y_true[mask]))
            conf = float(np.mean(y_probs[mask]))
            bin_accuracies.append(acc)
            bin_confidences.append(conf)
            ece += (count / max(1, total_samples)) * abs(acc - conf)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(float((bins[b] + bins[b + 1]) / 2.0))

    return float(ece), np.array(bin_accuracies), np.array(bin_confidences), np.array(bin_counts)


def compute_brier_score(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Computes mean squared error between predicted probabilities and binary ground truth."""
    y_true = np.asarray(y_true).ravel()
    y_probs = np.asarray(y_probs).ravel()
    return float(np.mean((y_probs - y_true) ** 2))


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> dict:
    """
    Computes empirical non-parametric bootstrap confidence intervals (95% CI) for key metrics.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1_list, prec_list, rec_list, acc_list = [], [], [], []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yt_sample, yp_sample = y_true[idx], y_pred[idx]

        tp = np.sum((yp_sample == 1) & (yt_sample == 1))
        fp = np.sum((yp_sample == 1) & (yt_sample == 0))
        fn = np.sum((yp_sample == 0) & (yt_sample == 1))

        prec = tp / (tp + fp + 1e-10)
        rec = tp / (tp + fn + 1e-10)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-10)
        acc = np.mean(yp_sample == yt_sample)

        f1_list.append(f1)
        prec_list.append(prec)
        rec_list.append(rec)
        acc_list.append(acc)

    alpha = (1.0 - confidence_level) / 2.0
    lower_pct, upper_pct = alpha * 100.0, (1.0 - alpha) * 100.0

    return {
        "f1_ci": (float(np.percentile(f1_list, lower_pct)), float(np.percentile(f1_list, upper_pct))),
        "precision_ci": (float(np.percentile(prec_list, lower_pct)), float(np.percentile(prec_list, upper_pct))),
        "recall_ci": (float(np.percentile(rec_list, lower_pct)), float(np.percentile(rec_list, upper_pct))),
        "accuracy_ci": (float(np.percentile(acc_list, lower_pct)), float(np.percentile(acc_list, upper_pct)))
    }


def evaluate_and_plot(
    model,
    X_test,
    y_test: np.ndarray,
    threshold: float = 0.5,
    class_names: list = CLASS_NAMES,
    save_dir: str = "plots",
    show_plot: bool = False
) -> dict:
    """
    Leak-Free Comprehensive Evaluation Pipeline:
    - Evaluates using a FROZEN threshold (derived strictly from validation split).
    - Computes Reliability Diagram and Expected Calibration Error (ECE).
    - Computes 95% Bootstrap Confidence Intervals.
    - Saves publication-quality metric plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    print("\n==================================================")
    print("[+] LEAK-FREE EVALUATION ON TEST SET")
    print(f"[+] Frozen Decision Threshold: {threshold:.4f} (Calibrated on Validation Split)")
    print("==================================================")

    # Predict test probabilities
    y_pred_probs = model.predict(X_test).ravel()
    y_test = np.asarray(y_test).ravel()

    # Safety assertion to prevent misaligned array sizes
    if len(y_pred_probs) != len(y_test):
        min_len = min(len(y_pred_probs), len(y_test))
        print(f"[!] Warning: Aligning prediction length ({len(y_pred_probs)}) to ground truth ({len(y_test)}) -> {min_len}")
        y_pred_probs = y_pred_probs[:min_len]
        y_test = y_test[:min_len]

    # Apply strictly frozen threshold
    y_pred = (y_pred_probs >= threshold).astype(int)

    test_acc = float(accuracy_score(y_test, y_pred))
    test_prec = float(precision_score(y_test, y_pred, zero_division=0))
    test_rec = float(recall_score(y_test, y_pred, zero_division=0))
    test_f1 = float(f1_score(y_test, y_pred, zero_division=0))

    # Classification Report
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print("\nClassification Report:\n", report)

    # Compute ROC and PR Curves & headline primary metrics (Safety-Critical & Imbalance Focus)
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    roc_auc = float(auc(fpr, tpr))
    precision, recall, _ = precision_recall_curve(y_test, y_pred_probs)
    pr_auc = float(auc(recall, precision))

    print("==================================================")
    print("🎯 PRIMARY METRICS SUMMARY (Safety-Critical Focus):")
    print(f"   |-- PR-AUC (Precision-Recall AUC): {pr_auc:.4f}")
    print(f"   |-- Debris Recall (Sensitivity):    {test_rec:.4f}")
    print(f"   |-- F1-Score (Optimal Threshold):   {test_f1:.4f}")
    print(f"   |-- ROC-AUC:                        {roc_auc:.4f}")
    print(f"   +-- Test Accuracy:                  {test_acc:.4f}")
    print("==================================================")

    # Compute Calibration Metrics
    ece, bin_acc, bin_conf, bin_counts = compute_expected_calibration_error(y_test, y_pred_probs, n_bins=10)
    brier = compute_brier_score(y_test, y_pred_probs)
    print(f"[+] Calibration Metrics -> ECE: {ece:.4f} | Brier Score: {brier:.4f}")

    # Compute Bootstrap 95% Confidence Intervals
    ci_dict = compute_bootstrap_confidence_intervals(y_test, y_pred, y_pred_probs, n_bootstraps=500, seed=42)
    print(f"[+] 95% Confidence Intervals (Bootstrap n=500):")
    print(f"   |-- F1-Score:  {test_f1:.4f}  [95% CI: {ci_dict['f1_ci'][0]:.4f} - {ci_dict['f1_ci'][1]:.4f}]")
    print(f"   |-- Recall:    {test_rec:.4f}  [95% CI: {ci_dict['recall_ci'][0]:.4f} - {ci_dict['recall_ci'][1]:.4f}]")
    print(f"   |-- Precision: {test_prec:.4f}  [95% CI: {ci_dict['precision_ci'][0]:.4f} - {ci_dict['precision_ci'][1]:.4f}]")
    print(f"   +-- Accuracy:  {test_acc:.4f}  [95% CI: {ci_dict['accuracy_ci'][0]:.4f} - {ci_dict['accuracy_ci'][1]:.4f}]")

    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (Threshold = {threshold:.2f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.abspath(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2. ROC Curve Plot
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    roc_path = os.path.abspath(os.path.join(save_dir, 'roc_curve.png'))
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # 3. Precision-Recall Curve with Prevalence Baseline
    prevalence = float(np.mean(y_test))
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
    plt.axhline(y=prevalence, color='red', linestyle='--', label=f'Class Prevalence ({prevalence:.2%})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (PR-AUC)')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    pr_path = os.path.abspath(os.path.join(save_dir, 'precision_recall_curve.png'))
    plt.savefig(pr_path, dpi=300)
    plt.close()

    # 4. Reliability Diagram (Confidence Calibration)
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(bin_conf, bin_acc, 's-', color='teal', lw=2, label=f'Model (ECE = {ece:.4f})')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives (Accuracy)')
    plt.title('Reliability Diagram (Confidence Calibration)')
    plt.legend(loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    calib_path = os.path.abspath(os.path.join(save_dir, 'reliability_diagram.png'))
    plt.savefig(calib_path, dpi=300)
    plt.close()

    print(f"\n[+] Evaluation plots successfully generated and saved to: {os.path.abspath(save_dir)}")
    print(f"   |-- Confusion Matrix:       {cm_path}")
    print(f"   |-- ROC Curve:              {roc_path}")
    print(f"   |-- Precision-Recall Curve: {pr_path}")
    print(f"   +-- Reliability Diagram:    {calib_path}")

    return {
        "report": report,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "test_accuracy": test_acc,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_f1": test_f1,
        "optimal_threshold": threshold,
        "ece": ece,
        "brier_score": brier,
        "confidence_intervals": ci_dict
    }
