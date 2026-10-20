#!/usr/bin/env python3
import argparse
import sys
import pandas as pd
from pathlib import Path
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

def parse_args():
    parser = argparse.ArgumentParser(description="Check overlap between lens candidates and Euclid dataset")
    parser.add_argument("--lens_csv", type=str, required=True,
                        help="Path to q1_discovery_engine_lens_catalog.csv")
    parser.add_argument("--metadata_fits", type=str, required=True,
                        help="Path to Euclid catalog (FITS)")
    parser.add_argument("--radius_arcsec", type=float, default=1.0,
                        help="Matching radius in arcseconds (default: 1.0)")
    parser.add_argument("--downloaded_csv", type=str, default=None,
                        help="Path to result.csv to check downloaded status of matches")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to save the matching overlap results")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading lens catalog from: {args.lens_csv}")
    lens_df = pd.read_csv(args.lens_csv)
    print(f"Loaded {len(lens_df)} lens candidates.")
    
    print(f"Loading Euclid metadata catalog from: {args.metadata_fits}")
    euclid_table = Table.read(args.metadata_fits)
    euclid_df = euclid_table.to_pandas()
    print(f"Loaded {len(euclid_df)} Euclid sources.")
    
    # Clean up column names for Euclid coordinates
    ra_col = next((c for c in euclid_df.columns if c.upper() == 'RA'), None)
    dec_col = next((c for c in euclid_df.columns if c.upper() == 'DEC'), None)
    id_col = next((c for c in euclid_df.columns if c.upper() in ['TARGETID', 'ID']), None)
    
    if ra_col is None or dec_col is None:
        raise ValueError("Could not find RA/DEC columns in Euclid catalog.")
        
    print(f"Using Euclid coordinates: {ra_col}, {dec_col}")
    if id_col:
        print(f"Using Euclid ID column: {id_col}")
        
    # Build SkyCoords
    lens_coords = SkyCoord(
        ra=lens_df['right_ascension'].values * u.deg,
        dec=lens_df['declination'].values * u.deg
    )
    
    euclid_coords = SkyCoord(
        ra=euclid_df[ra_col].values * u.deg,
        dec=euclid_df[dec_col].values * u.deg
    )
    
    # Coordinate matching
    print("Matching catalogs...")
    idx, d2d, d3d = lens_coords.match_to_catalog_sky(euclid_coords)
    
    # Filter by radius
    matches_mask = d2d < (args.radius_arcsec * u.arcsec)
    num_matches = matches_mask.sum()
    
    print(f"\nResults (radius = {args.radius_arcsec} arcsec):")
    print(f"--------------------------------------------------")
    print(f"Total overlapping lenses: {num_matches} / {len(lens_df)} ({num_matches/len(lens_df):.2%})")
    
    # Build match dataframe
    matched_indices = idx[matches_mask]
    lens_matched_rows = lens_df[matches_mask].copy()
    
    euclid_matched_rows = euclid_df.iloc[matched_indices].copy()
    
    # Merge matches
    results = []
    for i, (l_idx, e_idx) in enumerate(zip(lens_matched_rows.index, matched_indices)):
        lens_row = lens_df.loc[l_idx]
        euclid_row = euclid_df.iloc[e_idx]
        
        res_dict = {
            'lens_id': lens_row['id_str'],
            'lens_ra': lens_row['right_ascension'],
            'lens_dec': lens_row['declination'],
            'euclid_id': euclid_row[id_col] if id_col else 'N/A',
            'euclid_ra': euclid_row[ra_col],
            'euclid_dec': euclid_row[dec_col],
            'separation_arcsec': d2d[l_idx].to(u.arcsec).value
        }
        results.append(res_dict)
        
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        print("\nFirst 10 overlapping matches:")
        print(results_df.head(10).to_string(index=False))
        
        if args.downloaded_csv:
            print(f"\nChecking downloaded status against: {args.downloaded_csv}")
            downloaded_df = pd.read_csv(args.downloaded_csv)
            downloaded_ids = set(downloaded_df['id_str'].dropna().values)
            overlap_ids = set(results_df['lens_id'].values)
            intersection = overlap_ids.intersection(downloaded_ids)
            print(f"Lenses downloaded that are ALSO in the Euclid dataset: {len(intersection)} / {len(results_df)}")
            if len(intersection) > 0:
                print(f"Downloaded and overlapping lens IDs: {sorted(list(intersection))}")
            else:
                print("None of the 29 overlapping matches are in the downloaded result.csv subset.")
        
        if args.output_csv:
            results_df.to_csv(args.output_csv, index=False)
            print(f"\nSaved overlap results to: {args.output_csv}")
    else:
        print("\nNo matching overlapping lenses found.")

if __name__ == "__main__":
    main()
