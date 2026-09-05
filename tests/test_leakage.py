import pytest
from src.data.loader import extract_trajectory_id, split_dataset_by_trajectory

def test_extract_trajectory_id():
    r1 = {"path": "SPARK-2022/train/cheops/img_0012.jpg", "class_name": "cheops"}
    r2 = {"path": "SPARK-2022/train/cheops/img_0015.jpg", "class_name": "cheops"}
    r3 = {"path": "SPARK-2022/train/cheops/img_0150.jpg", "class_name": "cheops"}

    t1 = extract_trajectory_id(r1)
    t2 = extract_trajectory_id(r2)
    t3 = extract_trajectory_id(r3)

    assert t1 == t2, "Frames within the same hundred-range sequence should share trajectory ID"
    assert t1 != t3, "Frames across different sequences must have distinct trajectory IDs"

def test_no_split_leakage():
    # Synthetic records across 10 distinct trajectories
    synthetic_records = []
    for traj in range(10):
        for frame in range(10):
            synthetic_records.append({
                "path": f"fake_dir/sat_{traj:02d}_{frame:04d}.jpg",
                "label": 0 if traj < 3 else 1,
                "class_name": "debris" if traj < 3 else "satellite",
                "trajectory_id": f"traj_{traj}"
            })

    train_recs, val_recs, test_recs = split_dataset_by_trajectory(
        synthetic_records,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )

    train_groups = set(r["trajectory_id"] for r in train_recs)
    val_groups = set(r["trajectory_id"] for r in val_recs)
    test_groups = set(r["trajectory_id"] for r in test_recs)

    # Core Scientific Assertions
    assert len(train_groups & val_groups) == 0, f"Leakage between Train & Val: {train_groups & val_groups}"
    assert len(train_groups & test_groups) == 0, f"Leakage between Train & Test: {train_groups & test_groups}"
    assert len(val_groups & test_groups) == 0, f"Leakage between Val & Test: {val_groups & test_groups}"
    assert len(train_recs) + len(val_recs) + len(test_recs) == len(synthetic_records), "Total sample count preserved"
