#!/usr/bin/env python3
"""
Stream conversion of raw Euclid-DESI Pickle batches into Apache Arrow datasets.

This script iterates through chunked Pickle batches, extracts image stamps (VIS, NISP)
and 1D spectra (flux, wave, ivar, mask), normalizes data structures for memory contiguity,
and streams records into Hugging Face Apache Arrow format with 500MB max sharding.
"""

import argparse
import gc
import glob
import logging
import os
import pickle
import sys
from typing import Any, Iterator, List, Optional
import numpy as np
from datasets import Array2D, Dataset, Features, Sequence, Value

# Configure standard production logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)
logger = logging.getLogger("pickle_to_arrow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Euclid-DESI raw pickle files into Hugging Face Arrow datasets."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data",
        help="Directory containing raw batch_<split>_*.pkl files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow",
        help="Target directory for processed Arrow dataset partitions."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to process (default: train test)."
    )
    parser.add_argument(
        "--save_size",
        type=int,
        default=50,
        help="Number of pickle batch files to process before saving an Arrow part (default: 50)."
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="500MB",
        help="Maximum shard size per Arrow file (default: 500MB)."
    )
    return parser.parse_args()


def clean_array(
    arr: Optional[np.ndarray | list[Any]], 
    dtype: type = np.float32
) -> Optional[np.ndarray]:
    """
    Prepares, normalizes, and optimizes an array for efficient storage in Apache Arrow.

    Performs:
    1. Null handling (None safety).
    2. Data type conversion to target dtype.
    3. Memory reorganization (C-contiguity) for zero-copy operations.
    """
    if arr is None:
        return None

    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    return np.ascontiguousarray(arr.astype(dtype))


def generator_logic(
    split_name: str,
    file_list: List[str]
) -> Iterator[dict[str, Any]]:
    """
    Stream-processes chunked Pickle files to yield single galaxy examples for Arrow conversion.

    Uses an out-of-core generator pattern to control resident memory,
    cleaning arrays and extracting visual and spectroscopic records.
    """
    for i, filepath in enumerate(file_list):
        if i % 10 == 0 or i == len(file_list) - 1:
            logger.info(f"[{split_name}] Reading file {i + 1}/{len(file_list)}: {os.path.basename(filepath)}")
        
        try:
            with open(filepath, "rb") as f:
                batch_data = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to read file {filepath}: {e}")
            continue
    
        if i == 0 and len(batch_data) > 0:
            sample_keys = list(batch_data[0].keys())
            logger.debug(f"Sample dictionary keys in pickle batch: {sample_keys}")

        for item in batch_data:
            try:
                targetid = item.get("targetid", -1)
                
                # Visual channels
                vis_raw = item.get("vis_image") if item.get("vis_image") is not None else item.get("VIS_image")
                vis = clean_array(vis_raw)
                
                def get_nisp(key_lower: str, key_upper: str) -> Optional[np.ndarray]:
                    val = item.get(key_lower)
                    if val is None: 
                        val = item.get(key_upper)
                    return clean_array(val)
                    
                nisp_y = get_nisp("nisp_y_image", "NISP_Y_image")
                nisp_j = get_nisp("nisp_j_image", "NISP_J_image")
                nisp_h = get_nisp("nisp_h_image", "NISP_H_image")
                
                # Visual inspection stamp validation: VIS is mandatory
                if vis is None:
                    continue

                # Spectroscopy channels
                spec_data = item.get("spectrum")
                flux = None
                wave = None
                ivar = None
                mask = None
                
                if spec_data is not None and isinstance(spec_data, dict):
                    flux = clean_array(spec_data.get("flux"))
                    wave = clean_array(spec_data.get("wavelength"))
                    ivar = clean_array(spec_data.get("ivar"))
                    mask = clean_array(spec_data.get("mask"), dtype=np.uint32)

                yield {
                    "targetid": targetid,
                    "healpix": item.get("healpix", -1),
                    "redshift": item.get("redshift", -1.0),
                    
                    # Images
                    "image_vis": vis,
                    "image_nisp_y": nisp_y,
                    "image_nisp_j": nisp_j,
                    "image_nisp_h": nisp_h,
                    
                    # Spectra
                    "spectrum_flux": flux,
                    "spectrum_wave": wave,
                    "spectrum_ivar": ivar,
                    "spectrum_mask": mask
                }

            except Exception as e:
                logger.debug(f"Skipping corrupt target {targetid}: {e}")
                continue
        
        del batch_data
        gc.collect()


def get_dataset_features() -> Features:
    """Returns the explicit PyArrow schema specification for dataset optimization."""
    return Features({
        "targetid": Value("int64"),
        "healpix": Value("int64"),
        "redshift": Value("float32"),
        
        # 2D Scientific Images
        "image_vis": Array2D(shape=(224, 224), dtype="float32"),
        "image_nisp_y": Array2D(shape=(224, 224), dtype="float32"),
        "image_nisp_j": Array2D(shape=(224, 224), dtype="float32"),
        "image_nisp_h": Array2D(shape=(224, 224), dtype="float32"),
        
        # 1D Spectra Sequences
        "spectrum_flux": Sequence(Value("float32")),
        "spectrum_wave": Sequence(Value("float32")),
        "spectrum_ivar": Sequence(Value("float32")),
        "spectrum_mask": Sequence(Value("uint32")),
    })


def main() -> None:
    args = parse_args()
    features = get_dataset_features()
    os.makedirs(args.output_dir, exist_ok=True)

    for split in args.splits:
        logger.info(f"Starting Pickle to Arrow conversion for split: {split}")
        pattern = os.path.join(args.input_dir, f"batch_{split}_*.pkl")
        all_files = sorted(glob.glob(pattern))
        all_files = [f for f in all_files if "_info" not in f]
        
        if not all_files:
            logger.warning(f"No pickle batches found for split '{split}' in {args.input_dir}")
            continue
        
        logger.info(f"Found {len(all_files)} files. Processing in chunks of {args.save_size}...")
        total_chunks = (len(all_files) + args.save_size - 1) // args.save_size
        
        for i in range(0, len(all_files), args.save_size):
            chunk_idx = i // args.save_size
            chunk_files = all_files[i : i + args.save_size]
            part_dir = os.path.join(args.output_dir, f"{split}_{chunk_idx}")
            
            if os.path.exists(part_dir):
                logger.info(f"Chunk {chunk_idx + 1}/{total_chunks} already exists at {part_dir}. Skipping.")
                continue
            
            logger.info(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_files)} pickle files)...")
            
            ds = Dataset.from_generator(
                generator_logic, 
                gen_kwargs={"split_name": split, "file_list": chunk_files},
                features=features 
            )
            
            ds.save_to_disk(part_dir, max_shard_size=args.max_shard_size)
            del ds
            gc.collect()

    logger.info("Conversion completed successfully.")


if __name__ == "__main__":
    main()
