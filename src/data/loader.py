"""
SPARK-2022 Dataset Loader & Annotations Parser for Space Debris Identification.

Production-grade module for managing SPARK-2022 dataset archives (train.zip, val.zip, test.zip),
parsing CSV annotations with pandas, mapping 11 raw categories into a binary classification 
problem (Debris vs. Non-Debris), and loading offline preprocessed cached datasets.
"""

import os
import ast
import zipfile
import hashlib
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import imagehash

# Base SPARK-2022 directory defaults
DEFAULT_SPARK_DIR = "SPARK-2022"
DEFAULT_CACHE_DIR = "SPARK-2022-Preprocessed"

# SPARK Class Binary Mapping Specification:
# "debris" -> 0 (Debris)
# All 10 active spacecraft/satellite categories -> 1 (Non-Debris)
SPARK_CLASS_MAPPING = {
    "debris": 0,
    "smart_1": 1,
    "cheops": 1,
    "lisa_pathfinder": 1,
    "proba_3_ocs": 1,
    "proba_3_csc": 1,
    "soho": 1,
    "earth_observation_sat_1": 1,
    "proba_2": 1,
    "xmm_newton": 1,
    "double_star": 1,
}

CLASS_NAMES = ["Debris", "Non-Debris"]


def load_cached_records(split: str = "train", cache_dir: str = DEFAULT_CACHE_DIR) -> list:
    """
    Loads preprocessed 224x224 dataset records from the offline cache directory.

    Args:
        split (str): Split name ('train', 'val', or 'test').
        cache_dir (str): Path to root SPARK-2022-Preprocessed directory.

    Returns:
        list: List of dictionaries formatted as:
              [{"path": str, "label": int}, ...]
    """
    abs_cache_dir = os.path.abspath(cache_dir)
    
    # Support both labels/cached_{split}.csv and cached_{split}.csv
    candidate_csvs = [
        os.path.join(abs_cache_dir, "labels", f"cached_{split}.csv"),
        os.path.join(abs_cache_dir, f"cached_{split}.csv")
    ]

    csv_path = None
    for cand in candidate_csvs:
        if os.path.exists(cand):
            csv_path = cand
            break

    if csv_path is None:
        raise FileNotFoundError(
            f"[-] Cached manifest for split '{split}' not found in '{abs_cache_dir}'.\n"
            f"Please run the offline caching script first:\n"
            f"    python scripts/cache_dataset.py --spark-dir {DEFAULT_SPARK_DIR} --target-dir {cache_dir}"
        )

    print(f"[+] Loading cached dataset manifest: {csv_path}")
    df = pd.read_csv(csv_path)

    records = []
    for idx, row in df.iterrows():
        rel_or_abs_path = str(row["cached_path"]).strip()
        label = int(row["label"])

        if os.path.isabs(rel_or_abs_path):
            img_path = rel_or_abs_path
        else:
            img_path = os.path.abspath(os.path.join(abs_cache_dir, rel_or_abs_path))

        records.append({
            "path": img_path,
            "label": label
        })

    debris_count = sum(1 for r in records if r["label"] == 0)
    non_debris_count = sum(1 for r in records if r["label"] == 1)
    print(f"[+] Loaded {len(records)} cached records for '{split}' "
          f"(Debris [0]: {debris_count}, Non-Debris [1]: {non_debris_count}).")

    return records


def extract_spark_dataset(spark_dir: str = DEFAULT_SPARK_DIR) -> str:
    """
    Checks for train.zip, val.zip, and test.zip in the SPARK-2022/ directory.
    """
    abs_spark_dir = os.path.abspath(spark_dir)
    if not os.path.exists(abs_spark_dir):
        raise FileNotFoundError(f"[-] SPARK-2022 root directory not found at: {abs_spark_dir}")

    zip_splits = ["train", "val", "test"]
    for split in zip_splits:
        zip_path = os.path.join(abs_spark_dir, f"{split}.zip")
        extracted_folder = os.path.join(abs_spark_dir, split)

        already_extracted = (
            os.path.exists(extracted_folder) 
            and os.path.isdir(extracted_folder) 
            and len(os.listdir(extracted_folder)) > 0
        )

        if not already_extracted and os.path.exists(zip_path):
            try:
                print(f"[+] Attempting disk extraction of '{split}.zip' into '{abs_spark_dir}'...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(abs_spark_dir)
                print(f"[+] Extraction of '{split}.zip' completed successfully.")
            except Exception as e:
                print(f"[!] Note: Skipping full disk extraction of '{split}.zip' ({e}). "
                      f"Using direct in-memory zip decoding mode.")
        elif already_extracted:
            print(f"[+] Verified split directory '{split}/' is extracted ({len(os.listdir(extracted_folder))} files).")

    return abs_spark_dir


def parse_spark_csv(csv_path: str, img_dir: str, split_name: str = "train", spark_dir: str = DEFAULT_SPARK_DIR) -> list:
    """
    Parses a raw SPARK annotation CSV (filename, class, bbox) using pandas.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[-] Annotation CSV not found at: {csv_path}")

    print(f"[+] Reading annotation CSV via pandas: {csv_path}")
    df = pd.read_csv(csv_path)

    abs_spark_dir = os.path.abspath(spark_dir)
    zip_path = os.path.join(abs_spark_dir, f"{split_name}.zip")

    records = []
    skipped_count = 0

    for idx, row in df.iterrows():
        filename = str(row["filename"]).strip()
        raw_class = str(row["class"]).strip().lower()
        bbox_str = str(row["bbox"]).strip()

        label = 0 if raw_class == "debris" else 1

        try:
            bbox = list(ast.literal_eval(bbox_str))
            if len(bbox) != 4:
                raise ValueError(f"Invalid bbox length {len(bbox)}")
        except (ValueError, SyntaxError):
            skipped_count += 1
            continue

        img_path = os.path.join(img_dir, filename)
        zip_filename = f"{split_name}/{filename}"

        records.append({
            "path": img_path,
            "zip_path": zip_path,
            "zip_filename": zip_filename,
            "label": label,
            "bbox": bbox,
            "class_name": raw_class
        })

    if skipped_count > 0:
        print(f"[-] Skipped {skipped_count} rows with corrupt/unparseable bounding boxes.")

    debris_count = sum(1 for r in records if r["label"] == 0)
    non_debris_count = sum(1 for r in records if r["label"] == 1)
    print(f"[+] Successfully loaded {len(records)} records from {os.path.basename(csv_path)} "
          f"(Debris [0]: {debris_count}, Non-Debris [1]: {non_debris_count}).")

    return records


def load_spark_split(split: str = "train", spark_dir: str = DEFAULT_SPARK_DIR) -> list:
    """
    Loads raw records for a specific SPARK-2022 dataset split ("train", "val", or "test").
    """
    abs_spark_dir = extract_spark_dataset(spark_dir=spark_dir)
    csv_path = os.path.join(abs_spark_dir, "labels", f"{split}.csv")
    img_dir = os.path.join(abs_spark_dir, split)

    return parse_spark_csv(csv_path=csv_path, img_dir=img_dir, split_name=split, spark_dir=abs_spark_dir)


def get_cleaned_dataset(spark_dir: str = DEFAULT_SPARK_DIR, split: str = None, remove_duplicates: bool = False) -> list:
    """
    Main entrypoint for dataset ingestion.
    """
    abs_spark_dir = extract_spark_dataset(spark_dir=spark_dir)

    if split is not None:
        records = load_spark_split(split=split, spark_dir=abs_spark_dir)
    else:
        records = []
        for s in ["train", "val", "test"]:
            records.extend(load_spark_split(split=s, spark_dir=abs_spark_dir))

    print(f"[+] Total SPARK-2022 records loaded: {len(records)}")
    return records


# Backward compatibility aliases
extract_dataset = extract_spark_dataset
find_dataset_path = lambda base_dir=DEFAULT_SPARK_DIR: extract_spark_dataset(spark_dir=base_dir)
load_raw_paths = lambda dataset_dir=None: get_cleaned_dataset(spark_dir=dataset_dir or DEFAULT_SPARK_DIR)
