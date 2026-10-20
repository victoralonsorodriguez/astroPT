#!/usr/bin/env python3
"""
Hugging Face Arrow Dataset Metadata Repair Utility.

Reconstructs missing or corrupted 'dataset_info.json' and 'state.json' metadata files
for Apache Arrow shards to ensure compatibility with datasets.load_from_disk().
"""

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)
logger = logging.getLogger("repair_arrow_metadata")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct missing dataset_info.json and state.json for Arrow dataset splits."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing Arrow split partitions (e.g., train_0, test_0) or path to a split directory."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Dataset split names to inspect (default: train test)."
    )
    return parser.parse_args()


def repair_partition_metadata(folder_path: str, split_name: str) -> bool:
    """
    Inspects arrow files in a directory and writes dataset_info.json and state.json.
    """
    arrow_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(folder_path, "*.arrow"))])
    if not arrow_files:
        logger.warning(f"No .arrow files found in partition directory: {folder_path}")
        return False

    info = {
        "builder_name": "arrow",
        "config_name": "default",
        "version": {"version_str": "0.0.0", "major": 0, "minor": 0, "patch": 0},
        "splits": {
            split_name: {
                "name": split_name,
                "num_bytes": 0,
                "num_examples": 0,
                "dataset_name": "arrow"
            }
        },
    }

    state = {
        "_data_files": [{"filename": f} for f in arrow_files],
        "_fingerprint": "manual_fix_" + folder_path.replace("/", "_")[-50:],
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": split_name
    }

    info_path = os.path.join(folder_path, "dataset_info.json")
    state_path = os.path.join(folder_path, "state.json")

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    logger.info(f"Successfully generated metadata for '{split_name}' in {folder_path} ({len(arrow_files)} shards)")
    return True


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_dir).resolve()

    if not data_path.exists():
        logger.error(f"Provided path does not exist: {data_path}")
        sys.exit(1)

    # Check if data_path is directly a partition folder with .arrow files
    direct_arrow_files = list(data_path.glob("*.arrow"))
    if direct_arrow_files:
        split_guess = "train" if "train" in data_path.name.lower() else "test"
        repair_partition_metadata(str(data_path), split_guess)
        return

    # Otherwise, scan for split subfolders matching {split}_*
    repaired_count = 0
    for split in args.splits:
        matching_dirs = sorted(data_path.glob(f"{split}_*"))
        for split_dir in matching_dirs:
            if split_dir.is_dir():
                if repair_partition_metadata(str(split_dir), split):
                    repaired_count += 1

    if repaired_count == 0:
        logger.warning(f"No split directories matching '{args.splits}' found in {data_path}")
    else:
        logger.info(f"Metadata repair finished. Repaired {repaired_count} partition(s).")


if __name__ == "__main__":
    main()
