#!/usr/bin/env python3
"""
AstroPT-MM: Target Catalog Cross-Reference & Object Presence Checker
=====================================================================
Author: Victor Alonso Rodriguez
Date: October 2026
Description:
    Verifies whether a list of astronomical targets (provided via FITS, CSV, or Parquet)
    exists in the AstroPT-MM training and testing Arrow splits and the filtered metadata catalog.
    Supports multi-column ID queries (e.g. dual systems, cross-survey matching).

Usage:
    python scripts/dataset/dataset_object_checker.py \\
        --query_table /path/to/candidates.csv \\
        --id_cols targetid ID \\
        --metadata_fits /path/to/catalog_FILTERED.fits \\
        --data_dir /path/to/processed_data_arrow_interpolated
"""

import argparse
import sys
import os
import glob
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from astropy.table import Table
from datasets import load_from_disk

# Configure logging
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-Checker")

def parse_args():
    parser = argparse.ArgumentParser(description="Check overlap of target candidates within the AstroPT dataset.")
    parser.add_argument(
        "--query_table", type=str, required=True,
        help="Path to the input catalog file (FITS, CSV, or Parquet) containing candidate IDs."
    )
    parser.add_argument(
        "--id_cols", type=str, nargs="+", required=True,
        help="List of column names in the query table representing the IDs to look up (e.g. id_DESI_EDR_1 id_DESI_EDR_2)"
    )
    parser.add_argument(
        "--metadata_fits", type=str,
        default="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1_FILTERED.fits",
        help="Path to the Euclid metadata catalog FITS file."
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow",
        help="Directory containing the processed Arrow dataset splits."
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/dataset_logs",
        help="Directory where output reports will be saved."
    )
    return parser.parse_args()

def load_query_table(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        logger.error(f"Input file not found at: {file_path}")
        sys.exit(1)
        
    ext = path.suffix.lower()
    logger.info(f"Loading query table: {path.name}...")
    try:
        if ext in ['.fits', '.fit']:
            t = Table.read(path)
            return t.to_pandas()
        elif ext == '.csv':
            return pd.read_csv(path)
        elif ext == '.parquet':
            return pd.read_parquet(path)
        else:
            logger.error(f"Unsupported file format: {ext}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        sys.exit(1)

def main():
    args = parse_args()
    
    # 1. Load the input candidates
    df_query = load_query_table(args.query_table)
    logger.info(f"Query table loaded successfully. Rows: {len(df_query)}")
    
    # Collect all search IDs from the specified columns
    search_ids = set()
    id_map = {} # Maps targetid -> list of query table indices/source keys
    
    for col in args.id_cols:
        if col not in df_query.columns:
            logger.warning(f"ID Column '{col}' not found in query table columns: {list(df_query.columns)}")
            continue
        
        # Clean IDs, converting to int64
        valid_ids = df_query[col].dropna()
        for idx, val in valid_ids.items():
            try:
                tid = int(val)
                search_ids.add(tid)
                if tid not in id_map:
                    id_map[tid] = []
                id_map[tid].append((idx, col))
            except (ValueError, TypeError):
                continue
                
    if not search_ids:
        logger.error("No valid numeric IDs could be extracted from the specified columns.")
        sys.exit(1)
        
    logger.info(f"Extracted {len(search_ids)} unique TargetIDs to check.")
    
    # 2. Check metadata FITS catalog presence
    metadata_matches = {}
    if os.path.exists(args.metadata_fits):
        logger.info(f"Checking metadata presence in {args.metadata_fits}...")
        try:
            meta_table = Table.read(args.metadata_fits)
            meta_df = meta_table.to_pandas()
            target_id_col = 'TARGETID' if 'TARGETID' in meta_df.columns else 'targetid'
            
            # Index meta df for faster lookup
            meta_df_indexed = meta_df.drop_duplicates(subset=[target_id_col]).set_index(target_id_col)
            
            for tid in search_ids:
                if tid in meta_df_indexed.index:
                    meta_row = meta_df_indexed.loc[tid]
                    metadata_matches[tid] = {
                        "RA": float(meta_row.get("RA", np.nan)),
                        "DEC": float(meta_row.get("DEC", np.nan)),
                        "Z": float(meta_row.get("Z", np.nan)),
                        "SPECTYPE": str(meta_row.get("SPECTYPE", "N/A")),
                    }
            logger.info(f"Matched {len(metadata_matches)} / {len(search_ids)} TargetIDs in metadata catalog.")
        except Exception as e:
            logger.error(f"Error checking metadata FITS catalog: {e}")
    else:
        logger.warning(f"Euclid metadata catalog not found at {args.metadata_fits}, skipping catalog check.")

    # 3. Check physical dataset splits presence
    dataset_matches = {}
    if os.path.exists(args.data_dir):
        logger.info(f"Checking physical presence in Arrow splits under {args.data_dir}...")
        remaining_ids = set(search_ids)
        
        for split in ["train", "test"]:
            if not remaining_ids:
                break
            arrow_pattern = os.path.join(args.data_dir, f"{split}_*")
            arrow_folders = sorted(glob.glob(arrow_pattern))
            
            for path in arrow_folders:
                if not remaining_ids:
                    break
                try:
                    ds = load_from_disk(path)
                    targetids_in_partition = np.array(ds["targetid"], dtype=np.int64)
                    
                    for tid in list(remaining_ids):
                        row_idx = np.where(targetids_in_partition == tid)[0]
                        if len(row_idx) > 0:
                            dataset_matches[tid] = {
                                "split": split,
                                "partition": os.path.basename(path)
                            }
                            remaining_ids.remove(tid)
                except Exception as e:
                    logger.error(f"Error checking partition {path}: {e}")
                    
        logger.info(f"Found {len(dataset_matches)} / {len(search_ids)} TargetIDs physically in Arrow splits.")
    else:
        logger.warning(f"Arrow data directory not found at {args.data_dir}, skipping dataset check.")

    # 4. Generate final report
    report_rows = []
    for tid in sorted(search_ids):
        sources = id_map[tid]
        meta_info = metadata_matches.get(tid, {})
        ds_info = dataset_matches.get(tid, {})
        
        for query_idx, col_name in sources:
            row_dict = {
                "query_row_idx": query_idx,
                "queried_id_column": col_name,
                "targetid": tid,
                "in_metadata": tid in metadata_matches,
                "ra": meta_info.get("RA", np.nan),
                "dec": meta_info.get("DEC", np.nan),
                "z": meta_info.get("Z", np.nan),
                "spectype": meta_info.get("SPECTYPE", "N/A"),
                "in_dataset": tid in dataset_matches,
                "dataset_split": ds_info.get("split", "N/A"),
                "dataset_partition": ds_info.get("partition", "N/A"),
            }
            report_rows.append(row_dict)
            
    df_report = pd.DataFrame(report_rows)
    
    # Save reports
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = Path(args.query_table).stem
    
    csv_report_path = Path(args.output_dir) / f"{base_name}_overlap_report.csv"
    md_report_path = Path(args.output_dir) / f"{base_name}_overlap_report.md"
    
    df_report.to_csv(csv_report_path, index=False)
    
    # Generate markdown table for visual report
    with open(md_report_path, 'w') as f:
        f.write(f"# AstroPT Overlap Report: {base_name}\n\n")
        f.write(f"**Query File:** `{args.query_table}`  \n")
        f.write(f"**Metadata Catalog:** `{args.metadata_fits}`  \n")
        f.write(f"**Arrow Dataset Dir:** `{args.data_dir}`  \n\n")
        f.write("## Summary Statistics\n")
        f.write(f"- Total Queried TargetIDs: {len(search_ids)}  \n")
        f.write(f"- Overlapping in Metadata Catalog: {len(metadata_matches)} / {len(search_ids)} ({len(metadata_matches)/len(search_ids):.1%})  \n")
        f.write(f"- Physically present in Dataset Splits: {len(dataset_matches)} / {len(search_ids)} ({len(dataset_matches)/len(search_ids):.1%})  \n\n")
        f.write("## Overlap Matches List\n")
        f.write(df_report.to_markdown(index=False))
        
    logger.info(f"Overlap CSV report saved to: {csv_report_path}")
    logger.info(f"Overlap Markdown report saved to: {md_report_path}")
    
    print("\n--------------------------------------------------------")
    print(f"  Overlap Analysis Completed for {base_name}")
    print("--------------------------------------------------------")
    print(f"  Metadata overlap:  {len(metadata_matches)} / {len(search_ids)}")
    print(f"  Dataset overlap:   {len(dataset_matches)} / {len(search_ids)}")
    print(f"  Reports saved in:  {args.output_dir}")
    print("--------------------------------------------------------")

if __name__ == "__main__":
    main()
