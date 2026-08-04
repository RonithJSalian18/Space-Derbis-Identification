"""
Perceptual Deduplication Script for Space Debris Identification System.

Computes Perceptual Hash (pHash) using imagehash & PIL to filter near-duplicate 
consecutive frames (Hamming distance <= threshold) from offline cached datasets.
Uses multiprocessing for high-speed computation across large image datasets.

Usage:
    python scripts/deduplicate_cache.py --cache-dir SPARK-2022-Preprocessed --threshold 2 --output cleaned_manifest.csv
"""

import os
import sys
import argparse
import time
import pandas as pd
import numpy as np
from PIL import Image
import imagehash
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def compute_phash_single(args_tuple):
    """
    Worker function to compute perceptual hash (pHash) for a single image file path.
    """
    cached_path, full_path, label = args_tuple
    try:
        if not os.path.exists(full_path):
            return cached_path, None, label, "file_not_found"
        
        with Image.open(full_path) as img:
            ph = imagehash.phash(img)
            return cached_path, ph, label, None
    except Exception as e:
        return cached_path, None, label, str(e)


def process_deduplication(manifest_csv: str, cache_dir: str, threshold: int = 2, max_workers: int = None, output_csv: str = None) -> str:
    """
    Reads dataset manifest CSV, computes pHash in parallel, filters near-duplicates
    (Hamming distance <= threshold), and saves output_csv.
    """
    print(f"\n==================================================")
    print(f"[+] STARTING PERCEPTUAL DEDUPLICATION (pHash)")
    print(f"Manifest CSV: {manifest_csv}")
    print(f"Cache Directory: {cache_dir}")
    print(f"Hamming Distance Threshold: <= {threshold}")
    print(f"==================================================")

    df = pd.read_csv(manifest_csv)
    abs_cache_dir = os.path.abspath(cache_dir)

    task_args = []
    for idx, row in df.iterrows():
        cached_path = str(row["cached_path"]).strip()
        label = int(row["label"])
        
        if os.path.isabs(cached_path):
            full_path = cached_path
        else:
            full_path = os.path.abspath(os.path.join(abs_cache_dir, cached_path))
            
        task_args.append((cached_path, full_path, label))

    print(f"[+] Computing pHash across {len(task_args)} image records in parallel...")
    start_time = time.time()

    workers = max_workers or min(32, os.cpu_count() or 4)
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(
            executor.map(compute_phash_single, task_args, chunksize=100),
            total=len(task_args),
            desc="Computing pHash",
            unit="img"
        ))

    hash_time = time.time() - start_time
    print(f"[+] pHash computation finished in {hash_time:.2f}s using {workers} worker processes.")

    # Group by label to perform intra-class deduplication
    by_class = {}
    valid_count = 0
    for cached_path, ph, label, err in results:
        if ph is not None:
            valid_count += 1
            by_class.setdefault(label, []).append((cached_path, ph, label))

    print(f"[+] Filtering duplicates (Hamming Distance <= {threshold}) within classes...")

    cleaned_records = []
    duplicate_count = 0

    for label, class_records in by_class.items():
        distinct_hashes = []
        for cached_path, ph, lbl in class_records:
            is_dup = False
            for prev_path, prev_hash in reversed(distinct_hashes[-50:]):
                distance = ph - prev_hash  # Hamming distance
                if distance <= threshold:
                    is_dup = True
                    duplicate_count += 1
                    break

            if not is_dup:
                distinct_hashes.append((cached_path, ph))
                cleaned_records.append({"cached_path": cached_path, "label": lbl})

    if output_csv is None:
        output_csv = os.path.join(os.path.dirname(manifest_csv), "cleaned_manifest.csv")

    df_cleaned = pd.DataFrame(cleaned_records)
    df_cleaned.to_csv(output_csv, index=False)

    print(f"\n==================================================")
    print(f"[+] DEDUPLICATION COMPLETE!")
    print(f"   |-- Initial Records:    {len(df)}")
    print(f"   |-- Duplicates Removed: {duplicate_count} ({(duplicate_count/max(1, len(df)))*100:.2f}%)")
    print(f"   |-- Retained Distinct:  {len(cleaned_records)}")
    print(f"   +-- Cleaned Manifest:   {os.path.abspath(output_csv)}")
    print(f"==================================================")

    return output_csv


def deduplicate_all_splits(cache_dir: str = "SPARK-2022-Preprocessed", threshold: int = 2, max_workers: int = None):
    abs_cache_dir = os.path.abspath(cache_dir)
    labels_folder = os.path.join(abs_cache_dir, "labels")

    for split in ["train", "val", "test"]:
        candidates = [
            os.path.join(labels_folder, f"cached_{split}.csv"),
            os.path.join(abs_cache_dir, f"cached_{split}.csv")
        ]
        manifest_path = None
        for cand in candidates:
            if os.path.exists(cand):
                manifest_path = cand
                break

        if manifest_path:
            out_folder = labels_folder if os.path.exists(labels_folder) else abs_cache_dir
            out_csv = os.path.join(out_folder, f"cleaned_manifest_{split}.csv")
            process_deduplication(manifest_csv=manifest_path, cache_dir=cache_dir, threshold=threshold, max_workers=max_workers, output_csv=out_csv)


def main():
    parser = argparse.ArgumentParser(description="Perceptual Deduplication (pHash) for SPARK-2022 Cache")
    parser.add_argument("--cache-dir", type=str, default="SPARK-2022-Preprocessed", help="Path to preprocessed cache folder")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest CSV (defaults to cached_train.csv inside cache-dir)")
    parser.add_argument("--threshold", type=int, default=2, help="Hamming distance threshold (default: <= 2)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes")

    args = parser.parse_args()

    manifest_path = args.manifest
    if manifest_path is None:
        deduplicate_all_splits(cache_dir=args.cache_dir, threshold=args.threshold, max_workers=args.workers)
    else:
        if not os.path.exists(manifest_path):
            print(f"[-] Manifest CSV not found at '{manifest_path}'.")
            sys.exit(1)
        process_deduplication(manifest_csv=manifest_path, cache_dir=args.cache_dir, threshold=args.threshold, max_workers=args.workers)


if __name__ == "__main__":
    main()
