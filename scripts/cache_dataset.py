"""
Offline Dataset Caching Script for Space Debris Identification System.

Precomputes bounding box cropping, zero-g square padding, and 224x224 resizing 
once offline for the entire SPARK-2022 dataset (110,000 images across train, val, test).
Saves preprocessed 224x224 images and lightweight CSV annotation manifests to 
eliminate runtime disk I/O bottlenecks and GPU starvation.

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

from src.data.loader import load_spark_split
from src.data.preprocessing import crop_bbox_and_pad_square


def cache_split(split_name: str, src_dir: str = "SPARK-2022", target_dir: str = "SPARK-2022-Preprocessed") -> str:
    """
    Offline preprocessing pipeline for a single dataset split ('train', 'val', or 'test').

    Args:
        split_name (str): Split name ('train', 'val', or 'test').
        src_dir (str): Root directory of raw SPARK-2022 dataset.
        target_dir (str): Destination directory for preprocessed 224x224 images.

    Returns:
        str: Absolute path to generated lightweight CSV manifest file.
    """
    abs_target_dir = os.path.abspath(target_dir)
    split_target_folder = os.path.join(abs_target_dir, split_name)
    labels_target_folder = os.path.join(abs_target_dir, "labels")

    os.makedirs(split_target_folder, exist_ok=True)
    os.makedirs(labels_target_folder, exist_ok=True)

    print(f"\n==================================================")
    print(f"[+] STARTING OFFLINE PREPROCESSING: Split = {split_name.upper()}")
    print(f"[+] Output Image Folder: {split_target_folder}")
    print(f"==================================================")

    # 1. Load raw record metadata (path, bbox, label, zip_path, zip_filename)
    records = load_spark_split(split=split_name, spark_dir=src_dir)
    print(f"[+] Loaded {len(records)} raw records for '{split_name}' split.")

    cached_rows = []
    zip_handles = {}

    start_time = time.time()

    try:
        # Progress bar iterator using tqdm
        for record in tqdm(records, desc=f"Caching {split_name.capitalize()} Split", unit="img"):
            path = record.get("path")
            label = record.get("label")
            bbox = record.get("bbox")
            zip_path = record.get("zip_path")
            zip_filename = record.get("zip_filename")
            filename = os.path.basename(path)

            out_image_path = os.path.join(split_target_folder, filename)
            rel_cached_path = os.path.join(split_name, filename)

            # Check if image is already cached to allow resume capability
            if not os.path.exists(out_image_path):
                img = None
                # 1. Attempt reading from unzipped disk file
                if path and os.path.exists(path):
                    img = cv2.imread(path)

                # 2. Fallback to direct in-memory zip decoding if reading from archive
                if img is None and zip_path and os.path.exists(zip_path) and zip_filename:
                    if zip_path not in zip_handles:
                        zip_handles[zip_path] = zipfile.ZipFile(zip_path, 'r')
                    try:
                        img_bytes = zip_handles[zip_path].read(zip_filename)
                        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    except Exception:
                        img = None

                if img is not None:
                    # 3. Apply exact bbox crop, zero-g square padding, and 224x224 resize
                    processed_img = crop_bbox_and_pad_square(img, bbox=bbox, target_size=(224, 224))
                    # 4. Save preprocessed 224x224 image to disk
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

    # Save lightweight CSV manifest
    csv_manifest_path = os.path.join(labels_target_folder, f"cached_{split_name}.csv")
    df_cached = pd.DataFrame(cached_rows)
    df_cached.to_csv(csv_manifest_path, index=False)

    print(f"\n[+] Completed caching split '{split_name.upper()}' in {elapsed:.2f}s!")
    print(f"   |-- Images Saved:     {len(cached_rows)} -> {split_target_folder}")
    print(f"   +-- Manifest Saved:   {csv_manifest_path}")

    return csv_manifest_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Precompute and Cache 224x224 Images for SPARK-2022")
    parser.add_argument("--spark-dir", type=str, default="SPARK-2022", help="Source dataset root directory")
    parser.add_argument("--target-dir", type=str, default="SPARK-2022-Preprocessed", help="Target cache directory")
    args = parser.parse_args()

    total_start = time.time()
    print("==================================================")
    print("[+] SPARK-2022 OFFLINE DATASET CACHING ENGINE")
    print(f"Source Directory: {args.spark_dir}")
    print(f"Target Directory: {args.target_dir}")
    print("==================================================")

    for split in ["train", "val", "test"]:
        cache_split(split_name=split, src_dir=args.spark_dir, target_dir=args.target_dir)

    total_elapsed = time.time() - total_start
    print("\n==================================================")
    print(f"[+] ALL DATASET SPLITS CACHED SUCCESSFULLY in {total_elapsed / 60.0:.2f} minutes!")
    print(f"Cached Dataset Root: {os.path.abspath(args.target_dir)}")
    print("==================================================")


if __name__ == "__main__":
    main()
