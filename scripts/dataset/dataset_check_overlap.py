#!/usr/bin/env python3
"""
AstroPT-MM: Astrometric Coordinate Cross-Match & Lens Overlap Utility
=====================================================================
Author: Victor Alonso Rodriguez
Date: October 2026
Description:
    Performs on-sky angular coordinate cross-matching using Astropy SkyCoord
    between external target catalogs (e.g. gravitational lens candidates) and 
    the Euclid DR1 AstroPT survey metadata catalog within a configurable search radius.

Usage:
    python scripts/dataset/dataset_check_overlap.py \\
        --lens_csv /path/to/lens_catalog.csv \\
        --metadata_fits /path/to/catalog_FILTERED.fits \\
        --radius_arcsec 1.0 \\
        --output_csv /path/to/matches.csv
"""

import argparse
import logging
import sys
from pathlib import Path
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)
logger = logging.getLogger("dataset_check_overlap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check spatial overlap between external target candidates and Euclid DR1 metadata."
    )
    parser.add_argument(
        "--lens_csv",
        type=str,
        required=True,
        help="Path to external query catalog CSV containing coordinates (e.g., lens candidates)."
    )
    parser.add_argument(
        "--metadata_fits",
        type=str,
        required=True,
        help="Path to survey metadata catalog (FITS format)."
    )
    parser.add_argument(
        "--radius_arcsec",
        type=float,
        default=1.0,
        help="Matching radius tolerance in arcseconds (default: 1.0)."
    )
    parser.add_argument(
        "--downloaded_csv",
        type=str,
        default=None,
        help="Optional path to downloaded results CSV to verify subset presence."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to save the matched overlap table (CSV)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info(f"Loading query catalog from: {args.lens_csv}")
    lens_df = pd.read_csv(args.lens_csv)
    logger.info(f"Loaded {len(lens_df)} query targets.")

    logger.info(f"Loading Euclid metadata catalog from: {args.metadata_fits}")
    euclid_table = Table.read(args.metadata_fits)
    euclid_df = euclid_table.to_pandas()
    logger.info(f"Loaded {len(euclid_df)} survey sources.")

    # Clean up column names for Euclid coordinates
    ra_col = next((c for c in euclid_df.columns if c.upper() == "RA"), None)
    dec_col = next((c for c in euclid_df.columns if c.upper() == "DEC"), None)
    id_col = next((c for c in euclid_df.columns if c.upper() in ["TARGETID", "ID", "OBJECT_ID"]), None)

    if ra_col is None or dec_col is None:
        logger.error("Could not find RA/DEC coordinate columns in survey catalog.")
        sys.exit(1)

    logger.info(f"Survey coordinates identified: RA={ra_col}, DEC={dec_col} (ID column: {id_col})")

    # Build SkyCoords
    lens_ra_col = "right_ascension" if "right_ascension" in lens_df.columns else "ra"
    lens_dec_col = "declination" if "declination" in lens_df.columns else "dec"
    lens_id_col = "id_str" if "id_str" in lens_df.columns else ("targetid" if "targetid" in lens_df.columns else lens_df.columns[0])

    lens_coords = SkyCoord(
        ra=lens_df[lens_ra_col].values * u.deg,
        dec=lens_df[lens_dec_col].values * u.deg
    )
    euclid_coords = SkyCoord(
        ra=euclid_df[ra_col].values * u.deg,
        dec=euclid_df[dec_col].values * u.deg
    )

    # Coordinate matching
    logger.info("Executing astrometric spatial cross-match...")
    idx, d2d, _ = lens_coords.match_to_catalog_sky(euclid_coords)

    # Filter by radius
    matches_mask = d2d < (args.radius_arcsec * u.arcsec)
    num_matches = int(matches_mask.sum())

    pct = (num_matches / len(lens_df)) * 100 if len(lens_df) > 0 else 0.0
    logger.info(f"Matched {num_matches}/{len(lens_df)} sources ({pct:.2f}%) within {args.radius_arcsec} arcsec.")

    matched_indices = idx[matches_mask]
    lens_matched_rows = lens_df[matches_mask].copy()
    euclid_matched_rows = euclid_df.iloc[matched_indices].copy()

    results = []
    for l_idx, e_idx in zip(lens_matched_rows.index, matched_indices):
        lens_row = lens_df.loc[l_idx]
        euclid_row = euclid_df.iloc[e_idx]

        results.append({
            "query_id": lens_row[lens_id_col],
            "query_ra": lens_row[lens_ra_col],
            "query_dec": lens_row[lens_dec_col],
            "euclid_id": euclid_row[id_col] if id_col else "N/A",
            "euclid_ra": euclid_row[ra_col],
            "euclid_dec": euclid_row[dec_col],
            "separation_arcsec": float(d2d[l_idx].to(u.arcsec).value)
        })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        if args.downloaded_csv:
            downloaded_df = pd.read_csv(args.downloaded_csv)
            down_id_col = "id_str" if "id_str" in downloaded_df.columns else downloaded_df.columns[0]
            downloaded_ids = set(downloaded_df[down_id_col].dropna().values)
            overlap_ids = set(results_df["query_id"].values)
            intersection = overlap_ids.intersection(downloaded_ids)
            logger.info(f"Verified downloaded intersection: {len(intersection)}/{len(results_df)} targets present.")

        if args.output_csv:
            Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(args.output_csv, index=False)
            logger.info(f"Saved cross-match catalog to: {args.output_csv}")
    else:
        logger.warning("No overlapping sources found within the specified tolerance.")


if __name__ == "__main__":
    main()
