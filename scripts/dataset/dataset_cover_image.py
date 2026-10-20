#!/usr/bin/env python3
"""
AstroPT Cover Image Generator
================================
Extracts a single high-resolution galaxy cutout for the thesis cover,
saving it with absolute zero margins or axes.
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
logger = logging.getLogger("cover_image")

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
    parser.add_argument("--target_id", type=int, required=True, help="TargetID of the galaxy to plot")
    parser.add_argument("--data_dir", type=str, default="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow")
    parser.add_argument("--output_dir", type=str, default="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/dataset_logs")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    target_id = args.target_id
    record = None
    
    logger.info(f"Looking for TargetID {target_id} in {args.data_dir}...")
    
    for split in ["train", "test"]:
        for path in sorted(glob.glob(os.path.join(args.data_dir, f"{split}_*"))):
            try:
                ds = load_from_disk(path)
                tids = np.array(ds["targetid"], dtype=np.int64)
                idx = np.where(tids == target_id)[0]
                if len(idx) > 0:
                    record = ds[int(idx[0])]
                    logger.info(f"Found TargetID {target_id}")
                    break
            except Exception:
                continue
        if record is not None:
            break
            
    if record is None:
        logger.error(f"Could not find TargetID: {target_id}")
        return
        
    rgb = extract_raw_rgb(record)
    
    # We want ONLY the image, no axes, no margins. Ultra-high DPI.
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(rgb, origin='upper', interpolation='lanczos')
    
    # Save as PNG
    save_path_png = os.path.join(args.output_dir, f"cover_galaxy_{target_id}.png")
    plt.savefig(save_path_png, bbox_inches='tight', pad_inches=0, transparent=True)
    logger.info(f"Saved high-resolution PNG cover image to {save_path_png}")
    
    # Save as PDF
    save_path_pdf = os.path.join(args.output_dir, f"cover_galaxy_{target_id}.pdf")
    plt.savefig(save_path_pdf, bbox_inches='tight', pad_inches=0, transparent=True)
    logger.info(f"Saved high-resolution PDF cover image to {save_path_pdf}")

if __name__ == "__main__":
    main()
