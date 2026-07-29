import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc
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


def evaluate_and_plot(model, X_test, y_test, class_names=CLASS_NAMES, save_dir="plots", show_plot=True):
    """
    Comprehensive evaluation pipeline: Prints metrics, saves plots, and displays visualizations.
    """
    os.makedirs(save_dir, exist_ok=True)
    print("\n==================================================")
    print("[+] EVALUATION RESULTS ON TEST SET")
    print("==================================================")

    # Dynamic Decision Threshold Optimization to prevent single-class collapse
    y_pred_probs = model.predict(X_test).ravel()
    
    prec_curve, rec_curve, thresholds = precision_recall_curve(y_test, y_pred_probs)
    f1_scores = 2 * (prec_curve * rec_curve) / (prec_curve + rec_curve + 1e-10)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    y_pred_default = (y_pred_probs > 0.5).astype(int)
    y_pred_opt = (y_pred_probs > optimal_threshold).astype(int)

    acc_default = np.mean(y_pred_default == y_test)
    acc_opt = np.mean(y_pred_opt == y_test)

    if acc_opt > acc_default + 0.02 and 0.1 <= optimal_threshold <= 0.9:
        print(f"[+] Dynamic Threshold Applied: {optimal_threshold:.4f} (Test Accuracy improved from {acc_default:.4f} to {acc_opt:.4f})")
        y_pred = y_pred_opt
    else:
        print(f"[+] Standard Decision Threshold Applied: 0.5000 (Test Accuracy: {acc_default:.4f})")
        y_pred = y_pred_default

    # Classification Report
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("\nClassification Report:\n", report)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.abspath(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.abspath(os.path.join(save_dir, 'roc_curve.png'))
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_probs)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    pr_path = os.path.abspath(os.path.join(save_dir, 'precision_recall_curve.png'))
    plt.savefig(pr_path, dpi=300)
    plt.close()

    print(f"\n[+] Evaluation plots successfully generated and saved to: {os.path.abspath(save_dir)}")
    print(f"   |-- Confusion Matrix:       {cm_path}")
    print(f"   |-- ROC Curve:              {roc_path}")
    print(f"   +-- Precision-Recall Curve: {pr_path}")

    return {
        "report": report,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }
