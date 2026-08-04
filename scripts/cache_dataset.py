"""
Offline Dataset Caching Engine for Space Debris Identification System.

Refactored to enforce strict leak-free ordering:
1. Loads full 110,000 SPARK-2022 dataset records across train, val, and test splits.
2. GroupKFold / Trajectory grouping & Perceptual Hashing (pHash) deduplication executed FIRST.
3. Groups are split into 70/15/15 Train/Val/Test subsets BEFORE caching to eliminate data leakage.
4. Precomputes bounding box crops with reflection padding (cv2.BORDER_REFLECT_101) to eliminate Grad-CAM artifacts.
5. Saves 224x224 images to SPARK-2022-Preprocessed/ and outputs lightweight cached_{split}.csv manifests.

Usage:
    python scripts/cache_dataset.py --spark-dir SPARK-2022 --target-dir SPARK-2022-Preprocessed
"""

import os
import sys
import time
import zipfile
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root directory to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import get_cleaned_dataset, split_dataset_by_trajectory
from src.data.preprocessing import crop_bbox_and_pad_square


def cache_records_subset(records: list, split_name: str, target_dir: str = "SPARK-2022-Preprocessed") -> str:
    """
    Caches preprocessed 224x224 images for a deduplicated, trajectory-grouped split subset.

    Args:
        records (list): List of dict records belonging to this specific split.
        split_name (str): Split identifier ('train', 'val', or 'test').
        target_dir (str): Root destination directory for preprocessed images.

    Returns:
        str: Path to written CSV manifest.
    """
    abs_target_dir = os.path.abspath(target_dir)
    split_target_folder = os.path.join(abs_target_dir, split_name)
    labels_target_folder = os.path.join(abs_target_dir, "labels")

    os.makedirs(split_target_folder, exist_ok=True)
    os.makedirs(labels_target_folder, exist_ok=True)

    print(f"\n==================================================")
    print(f"[+] CACHING PREPROCESSED SUBSET: Split = {split_name.upper()} ({len(records)} records)")
    print(f"[+] Target Folder: {split_target_folder}")
    print(f"==================================================")

    cached_rows = []
    zip_handles = {}
    start_time = time.time()

    try:
        for record in tqdm(records, desc=f"Caching {split_name.capitalize()} Split", unit="img"):
            path = record.get("path")
            label = record.get("label")
            bbox = record.get("bbox")
            zip_path = record.get("zip_path")
            zip_filename = record.get("zip_filename")
            filename = os.path.basename(path)

            out_image_path = os.path.join(split_target_folder, filename)
            rel_cached_path = os.path.join(split_name, filename)

            if not os.path.exists(out_image_path):
                img = None
                if path and os.path.exists(path):
                    img = cv2.imread(path)

                if img is None and zip_path and os.path.exists(zip_path) and zip_filename:
                    if zip_path not in zip_handles:
                        zip_handles[zip_path] = zipfile.ZipFile(zip_path, 'r')
                    try:
                        img_bytes = zip_handles[zip_path].read(zip_filename)
                        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    except Exception:
                        img = None

                if img is not None:
                    # Apply reflection padding (cv2.BORDER_REFLECT_101) to eliminate Grad-CAM border artifacts
                    processed_img = crop_bbox_and_pad_square(img, bbox=bbox, target_size=(224, 224))
                    cv2.imwrite(out_image_path, processed_img)

            cached_rows.append({
                "cached_path": rel_cached_path,
                "label": label
            })
    finally:
        for zh in zip_handles.values():
            try:
                zh.close()
            except Exception:
                pass

    elapsed = time.time() - start_time

    csv_manifest_path = os.path.join(labels_target_folder, f"cached_{split_name}.csv")
    df_cached = pd.DataFrame(cached_rows)
    df_cached.to_csv(csv_manifest_path, index=False)

    print(f"[+] Completed caching split '{split_name.upper()}' in {elapsed:.2f}s!")
    print(f"   |-- Images Saved:   {len(cached_rows)} -> {split_target_folder}")
    print(f"   +-- Manifest Saved: {csv_manifest_path}")

    return csv_manifest_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Precompute and Cache 224x224 Images for SPARK-2022 with Grouping & Reflection Padding")
    parser.add_argument("--spark-dir", type=str, default="SPARK-2022", help="Source dataset root directory")
    parser.add_argument("--target-dir", type=str, default="SPARK-2022-Preprocessed", help="Target cache directory")
    args = parser.parse_args()

    total_start = time.time()
    print("==================================================")
    print("[+] SPARK-2022 LEAK-FREE OFFLINE DATASET CACHING ENGINE")
    print(f"Source Directory: {args.spark_dir}")
    print(f"Target Directory: {args.target_dir}")
    print("==================================================")

    # STEP 1: Ingest full raw 110,000 records from SPARK-2022
    print("\n[+] STEP 1: Ingesting entire 110,000 raw SPARK-2022 dataset...")
    raw_all_records = get_cleaned_dataset(spark_dir=args.spark_dir)

    # STEP 2: Trajectory grouping & deduplication executed FIRST before splitting & caching
    print("\n[+] STEP 2: Grouping trajectory sequences (GroupShuffleSplit) FIRST to prevent frame leakage...")
    train_recs, val_recs, test_recs = split_dataset_by_trajectory(
        raw_all_records,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )

    # STEP 3: Cache preprocessed 224x224 images with reflection padding into train, val, and test subsets
    print("\n[+] STEP 3: Precomputing 224x224 images with reflection padding (cv2.BORDER_REFLECT_101)...")
    cache_records_subset(train_recs, split_name="train", target_dir=args.target_dir)
    cache_records_subset(val_recs, split_name="val", target_dir=args.target_dir)
    cache_records_subset(test_recs, split_name="test", target_dir=args.target_dir)

    total_elapsed = time.time() - total_start
    print("\n==================================================")
    print(f"[+] LEAK-FREE CACHING COMPLETED SUCCESSFULLY in {total_elapsed / 60.0:.2f} minutes!")
    print(f"Cached Dataset Root: {os.path.abspath(args.target_dir)}")
    print("==================================================")


if __name__ == "__main__":
    main()
