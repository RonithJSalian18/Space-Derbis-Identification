import numpy as np
import pytest
from src.evaluation.metrics import (
    find_optimal_threshold,
    compute_expected_calibration_error,
    compute_brier_score,
    compute_bootstrap_confidence_intervals
)

def test_find_optimal_threshold():
    # Synthetic ground truth and predictions with clear separation
    y_val = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # Debris = 0, Non-Debris = 1
    # Perfect probability model
    val_probs = np.array([0.05, 0.10, 0.12, 0.20, 0.25, 0.75, 0.80, 0.85, 0.90, 0.95])

    thresh, best_f1, stats = find_optimal_threshold(y_val, val_probs, metric="f1")
    assert 0.25 < thresh <= 0.75, f"Expected threshold separating clusters, got {thresh}"
    assert best_f1 == pytest.approx(1.0, abs=1e-4), f"Expected perfect F1 of 1.0, got {best_f1}"

def test_expected_calibration_error():
    # Perfectly calibrated model
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([0.0, 0.0, 1.0, 1.0])
    ece, _, _, _ = compute_expected_calibration_error(y_true, y_probs, n_bins=5)
    assert ece == pytest.approx(0.0, abs=1e-5), f"Perfect calibration should have ECE 0.0, got {ece}"

    # Severely overconfident miscalibrated model
    y_true_bad = np.array([0, 0, 0, 0])
    y_probs_bad = np.array([0.9, 0.95, 0.99, 0.92])
    ece_bad, _, _, _ = compute_expected_calibration_error(y_true_bad, y_probs_bad, n_bins=5)
    assert ece_bad > 0.8, f"Expected high ECE for overconfident errors, got {ece_bad}"

def test_brier_score():
    y_true = np.array([1, 0])
    y_probs = np.array([1.0, 0.0])
    assert compute_brier_score(y_true, y_probs) == 0.0

    y_probs_worst = np.array([0.0, 1.0])
    assert compute_brier_score(y_true, y_probs_worst) == 1.0

def test_bootstrap_confidence_intervals():
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
    y_probs = np.array([0.1, 0.2, 0.1, 0.3, 0.6, 0.8, 0.9, 0.85, 0.7, 0.4])

    cis = compute_bootstrap_confidence_intervals(y_true, y_pred, y_probs, n_bootstraps=100, seed=42)
    assert "f1_ci" in cis
    assert "recall_ci" in cis
    assert "precision_ci" in cis
    assert "accuracy_ci" in cis

    f1_low, f1_high = cis["f1_ci"]
    assert 0.0 <= f1_low <= f1_high <= 1.0
