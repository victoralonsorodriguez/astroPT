#!/usr/bin/env python3
"""
AstroPT Appendix A: k-NN Similarity Search Mosaic
===================================================
Generates a 2x4 grid showing a query galaxy and its top-1 neighbors
from the Visual, Spectral, and Joint latent spaces.
Top row: Images
Bottom row: Spectra (Square layout)
"""

import os
import sys
import glob
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("appendix_a_mosaic")

# LaTeX font settings
plt.rcParams['text.latex.preamble'] = r'''
            \usepackage{siunitx}
            \usepackage{amsmath} 
            \usepackage{amssymb}
            \usepackage{times}
            \usepackage{bm}
            '''
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def make_rgb_lupton(image_stack: np.ndarray, Q: float = 12.0, stretch: float = 0.5, m: float = 0.0) -> np.ndarray:
    intensity = np.mean(image_stack, axis=0) - m
    intensity = np.maximum(intensity, 1e-10)
    f_I = np.arcsinh(Q * stretch * intensity) / Q
    scale_factor = f_I / intensity
    rgb_out = image_stack * scale_factor[np.newaxis, :, :]
    max_rgb = np.percentile(rgb_out, 99.5)
    if max_rgb > 0:
        rgb_out = rgb_out / max_rgb
    return np.clip(rgb_out, 0, 1).transpose(1, 2, 0)

def make_lupton_composite(vis, y_band, j_band, h_band, Q=12.0, stretch=0.5, rgb_weights=(1.2, 1.3, 1.0)):
    vis_c = np.nan_to_num(vis).astype(np.float32)
    y_c   = np.nan_to_num(y_band).astype(np.float32)
    j_c   = np.nan_to_num(j_band).astype(np.float32)
    h_c   = np.nan_to_num(h_band).astype(np.float32)
    
    has_h = h_c.size > 0 and np.any(h_c > 0)
    raw_stack = np.stack([vis_c, h_c, j_c, y_c], axis=0) if has_h else np.stack([vis_c, j_c, y_c], axis=0)
    
    bg_val = np.percentile(raw_stack, 50, axis=(1, 2), keepdims=True)
    raw_bg = raw_stack - bg_val
    
    normed_channels = []
    for c in range(raw_bg.shape[0]):
        v_max = np.percentile(np.abs(raw_bg[c]), 99.5)
        normed_channels.append(np.clip(raw_bg[c] / max(v_max, 1.0), 0, 100))
    raw_norm = np.stack(normed_channels)
    
    w_r, w_g, w_b = rgb_weights
    if has_h:
        r, g, b = raw_norm[1] * w_r, ((raw_norm[2] + raw_norm[3]) / 2.0) * w_g, raw_norm[0] * w_b
    else:
        r, g, b = raw_norm[1] * w_r, raw_norm[2] * w_g, raw_norm[0] * w_b
        
    return make_rgb_lupton(np.stack([r, g, b], axis=0), Q=Q, stretch=stretch)

def extract_raw_rgb(raw_record: dict) -> np.ndarray:
    vis_data = np.array(raw_record['image_vis'], dtype=np.float32)
    y_data   = np.array(raw_record['image_nisp_y'], dtype=np.float32)
    j_data   = np.array(raw_record['image_nisp_j'], dtype=np.float32)
    h_val = raw_record.get('image_nisp_h', None)
    h_data = np.array(h_val, dtype=np.float32) if h_val is not None else np.zeros_like(vis_data)
    return make_lupton_composite(vis_data, y_data, j_data, h_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_id", type=int, required=True, help="TargetID for the Query")
    parser.add_argument("--euclid_id", type=int, required=True, help="TargetID for the Visual k-NN")
    parser.add_argument("--desi_id", type=int, required=True, help="TargetID for the Spectral k-NN")
    parser.add_argument("--joint_id", type=int, required=True, help="TargetID for the Joint k-NN")
    parser.add_argument("--data_dir", type=str, default="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow")
    parser.add_argument("--metadata_path", type=str, default="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1_FILTERED.fits", help="Path to FITS catalog metadata")
    parser.add_argument("--output_dir", type=str, default="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/dataset_logs/appendix")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load FITS catalog metadata if available
    fits_indexed = None
    if os.path.exists(args.metadata_path):
        logger.info(f"Loading FITS metadata catalog from {args.metadata_path}...")
        try:
            from astropy.table import Table
            fits_table = Table.read(args.metadata_path)
            fits_df = fits_table.to_pandas()
            for col in fits_df.columns:
                if fits_df[col].dtype == object:
                    try:
                        sample = fits_df[col].dropna().iloc[0]
                        if isinstance(sample, bytes):
                            fits_df[col] = fits_df[col].str.decode('utf-8')
                    except (IndexError, AttributeError):
                        pass
            target_id_col = 'TARGETID' if 'TARGETID' in fits_df.columns else 'targetid'
            fits_indexed = fits_df.drop_duplicates(subset=[target_id_col]).set_index(target_id_col)
            logger.info("Successfully loaded and indexed FITS metadata catalog.")
        except Exception as e:
            logger.error(f"Error loading FITS metadata catalog: {e}")
    
    target_ids = [args.query_id, args.euclid_id, args.desi_id, args.joint_id]
    records = {tid: None for tid in target_ids}
    
    logger.info(f"Looking for TargetIDs {target_ids} in {args.data_dir}...")
    
    # Efficiently load records in a single pass (or early exit)
    for split in ["train", "test"]:
        for path in sorted(glob.glob(os.path.join(args.data_dir, f"{split}_*"))):
            try:
                ds = load_from_disk(path)
                tids = np.array(ds["targetid"], dtype=np.int64)
                for tid in target_ids:
                    if records[tid] is None:
                        idx = np.where(tids == tid)[0]
                        if len(idx) > 0:
                            records[tid] = ds[int(idx[0])]
                            logger.info(f"Found TargetID {tid}")
            except Exception:
                continue
            # Break if all found
            if all(v is not None for v in records.values()):
                break
        if all(v is not None for v in records.values()):
            break
            
    # Check if any were not found
    missing = [tid for tid, rec in records.items() if rec is None]
    if missing:
        logger.error(f"Could not find the following TargetIDs: {missing}")
        # Proceed with what we have (blank plots for missing ones)
    
    fig = plt.figure(figsize=(16, 9), dpi=200)
    fig.patch.set_facecolor('white')
    
    col_titles = [r"\textbf{Query}", r"\textbf{Euclid k-NN \#1}", r"\textbf{DESI k-NN \#1}", r"\textbf{Joint k-NN \#1}"]
    
    # 4 columns, 2 rows, perfectly adjacent square blocks
    plot_w = 0.24
    plot_h = 0.24 * (16.0 / 9.0) # Matches physical square aspect ratio
    start_x = (1.0 - 4 * plot_w) / 2
    start_y = 0.04
    
    for col, tid in enumerate(target_ids):
        record = records[tid]
        
        x_pos = start_x + col * plot_w
        y_bottom = start_y
        y_top = start_y + plot_h
        
        ax_img = fig.add_axes([x_pos, y_top, plot_w, plot_h])
        ax_spec = fig.add_axes([x_pos, y_bottom, plot_w, plot_h])
        
        if record is None:
            ax_img.axis('off')
            ax_spec.axis('off')
            ax_img.set_title(f"{col_titles[col]}\n[NOT FOUND: {tid}]", fontsize=16)
            continue
            
        # Data Extraction
        rgb = extract_raw_rgb(record)
        spec_flux = np.array(record['spectrum_flux']).flatten()
        spec_wave = np.array(record['spectrum_wave']).flatten()
        
        valid_idx = np.where(spec_flux != 0)[0]
        if len(valid_idx) > 0:
            spec_flux = spec_flux[valid_idx[0]:valid_idx[-1]+1]
            spec_wave = spec_wave[valid_idx[0]:valid_idx[-1]+1]
            
        # --- Top Row: Image ---
        ax_img.imshow(rgb, origin='upper')
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        euclid_id = record.get('euclid_id', None)
        if hasattr(euclid_id, 'item'):
            euclid_id = euclid_id.item()
        
        # Fallback to FITS catalog metadata if loaded
        if euclid_id is None and fits_indexed is not None and tid in fits_indexed.index:
            meta = fits_indexed.loc[tid]
            def _safe_fits(key):
                v = meta.get(key, None)
                return None if (v is None or (isinstance(v, float) and pd.isna(v))) else v
            euclid_id = _safe_fits('EUCLID_ID')
            if euclid_id is None:
                euclid_id = _safe_fits('euclid_id')
                
        if euclid_id is None:
            euclid_id = record.get('ID_Euclid', None)
        if euclid_id is None:
            euclid_id = record.get('ID', None)
        euclid_id_str = f"{euclid_id}" if euclid_id is not None else "N/A"
        
        euclid_label = rf"\textbf{{Euclid ID}}: {euclid_id_str}"
        desi_label = rf"\textbf{{DESI ID}}: {tid}"
        ax_img.set_title(f"{col_titles[col]}\n{euclid_label}\n{desi_label}", fontsize=14, pad=15)
        # Custom outer black border, inner white dividers
        for side, spine in ax_img.spines.items():
            spine.set_linewidth(2.0)
            if side in ['top']:
                spine.set_edgecolor('black')
            elif side in ['bottom']:
                spine.set_edgecolor('white')
            elif side == 'left':
                spine.set_edgecolor('black' if col == 0 else 'white')
            elif side == 'right':
                spine.set_edgecolor('black' if col == len(target_ids) - 1 else 'white')
            
        # --- Bottom Row: Spectrum ---
        ax_spec.plot(spec_wave, spec_flux, color='black', lw=0.8)
        ax_spec.set_xlim(spec_wave.min(), spec_wave.max())
        ax_spec.set_yticks([]) # Hiding y-ticks for cleaner look
        ax_spec.tick_params(axis='x', labelsize=12)
        ax_spec.set_facecolor('white')
        # No set_box_aspect needed because add_axes creates mathematically perfect squares
        
        for side, spine in ax_spec.spines.items():
            spine.set_linewidth(2.0)
            if side in ['bottom']:
                spine.set_edgecolor('black')
            elif side in ['top']:
                spine.set_edgecolor('black')
            elif side == 'left':
                spine.set_edgecolor('black' if col == 0 else 'white')
            elif side == 'right':
                spine.set_edgecolor('black' if col == len(target_ids) - 1 else 'white')

    save_path = os.path.join(args.output_dir, f"similarity_dashboard_{args.query_id}.png")
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved Similarity Dashboard to {save_path}")

if __name__ == "__main__":
    main()
