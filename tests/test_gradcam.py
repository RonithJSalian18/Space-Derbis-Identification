import numpy as np
import pytest
from src.evaluation.gradcam import (
    compute_pointing_game_accuracy,
    compute_cam_energy_inside_bbox,
    compute_cam_bbox_iou
)

def test_pointing_game_hit_and_miss():
    heatmap = np.zeros((224, 224), dtype=np.float32)
    # Peak at (100, 100)
    heatmap[100, 100] = 1.0

    # Bounding box containing the peak: [xmin, ymin, xmax, ymax]
    bbox_hit = [80, 80, 120, 120]
    bbox_miss = [10, 10, 50, 50]

    assert compute_pointing_game_accuracy(heatmap, bbox_hit) == 1
    assert compute_pointing_game_accuracy(heatmap, bbox_miss) == 0

def test_cam_energy_inside_bbox():
    heatmap = np.zeros((224, 224), dtype=np.float32)
    # Put all energy inside bbox
    heatmap[50:100, 50:100] = 1.0
    bbox = [50, 50, 100, 100]

    energy = compute_cam_energy_inside_bbox(heatmap, bbox)
    assert energy == pytest.approx(1.0, abs=1e-3)

    # Put half outside
    heatmap[150:200, 150:200] = 1.0
    energy_half = compute_cam_energy_inside_bbox(heatmap, bbox)
    assert energy_half == pytest.approx(0.5, abs=1e-2)

def test_cam_bbox_iou():
    heatmap = np.zeros((224, 224), dtype=np.float32)
    heatmap[50:100, 50:100] = 0.8
    bbox = [50, 50, 100, 100]

    iou = compute_cam_bbox_iou(heatmap, bbox, threshold_ratio=0.5)
    assert iou == pytest.approx(1.0, abs=1e-2)
