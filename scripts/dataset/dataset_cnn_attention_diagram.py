#!/usr/bin/env python3
"""
AstroPT CNN Local Processing Diagram Generator
==============================================
Generates a Matplotlib diagram showing the CNN receptive field concept 
(a 3x3 filter mapping to a single output pixel) replicating the requested visual style.
"""

import os
import sys
import glob
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datasets import load_from_disk
import scipy.ndimage as ndimage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cnn_diagram")

# LaTeX font settings
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', weight='bold')

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

def draw_grid(ax, img_size, n_cells, color='white', alpha=0.5, lw=1.0):
    step = img_size / n_cells
    for i in range(1, n_cells):
        ax.axhline(i * step, color=color, alpha=alpha, lw=lw)
        ax.axvline(i * step, color=color, alpha=alpha, lw=lw)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_id", type=int, default=39633516366922547)
    parser.add_argument("--data_dir", type=str, default="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow")
    parser.add_argument("--output_dir", type=str, default="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/dataset_logs/cnn_diagrams")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Looking for TargetID {args.target_id} in {args.data_dir}...")
    record = None
    for split in ["train", "test"]:
        for path in sorted(glob.glob(os.path.join(args.data_dir, f"{split}_*"))):
            try:
                ds = load_from_disk(path)
                tids = np.array(ds["targetid"], dtype=np.int64)
                idx = np.where(tids == args.target_id)[0]
                if len(idx) > 0:
                    record = ds[int(idx[0])]
                    break
            except Exception:
                continue
        if record: break
        
    if not record:
        logger.error(f"TargetID {args.target_id} not found.")
        sys.exit(1)

    rgb = extract_raw_rgb(record)
    
    # ---------------------------------------------------------
    # Create the Diagram
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(7, 10), dpi=200)
    fig.patch.set_facecolor('white') # Changed to pure white
    
    # --- Top Axes (Input Image) ---
    ax_top = fig.add_axes([0.15, 0.45, 0.7, 0.5])
    ax_top.imshow(rgb, origin='upper')
    ax_top.axis('off')
    
    # 16x16 Grid (White over the dark galaxy image)
    n_top = 16
    step_top = 224 / n_top
    draw_grid(ax_top, 224, n_top, color='white', alpha=0.4, lw=0.8)
    
    # 3x3 Filter Highlight
    # Position: top-left area
    col, row = 3, 3
    f_size = 3
    x0, y0 = col * step_top, row * step_top
    
    # Fill color green
    rect_top = patches.Rectangle((x0, y0), f_size * step_top, f_size * step_top, 
                                 linewidth=0, facecolor='#a4ff4a', alpha=0.6)
    # Border green
    rect_top_edge = patches.Rectangle((x0, y0), f_size * step_top, f_size * step_top, 
                                 linewidth=3, edgecolor='#a4ff4a', facecolor='none')
    ax_top.add_patch(rect_top)
    ax_top.add_patch(rect_top_edge)
    
    # Internal grid lines for the 3x3 filter
    for i in range(1, f_size):
        ax_top.plot([x0 + i*step_top, x0 + i*step_top], [y0, y0 + f_size*step_top], color='#1a1a1a', lw=1.5, alpha=0.8)
        ax_top.plot([x0, x0 + f_size*step_top], [y0 + i*step_top, y0 + i*step_top], color='#1a1a1a', lw=1.5, alpha=0.8)

    # Callout text for "FILTRO 3x3"
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1.2, alpha=0.9)
    ax_top.text(x0 + f_size*step_top + 30, y0 + (f_size/2)*step_top, 
                r"\textbf{3$\times$3 FILTER}", color="black", fontsize=16, 
                ha="left", va="center", bbox=bbox_props)
    
    # Connecting line to callout
    ax_top.plot([x0 + f_size*step_top, x0 + f_size*step_top + 30], 
                [y0 + (f_size/2)*step_top, y0 + (f_size/2)*step_top], color='black', lw=1.5)

    # --- Bottom Axes (Output Feature Map) ---
    # Downscaled / blurred view to represent the feature map
    rgb_blur = ndimage.gaussian_filter(rgb, sigma=(2, 2, 0))
    ax_bottom = fig.add_axes([0.35, 0.18, 0.3, 0.25])
    ax_bottom.imshow(rgb_blur, origin='upper')
    ax_bottom.axis('off')
    
    # 5x5 Grid for output
    n_bottom = 5
    step_bottom = 224 / n_bottom
    draw_grid(ax_bottom, 224, n_bottom, color='white', alpha=0.5, lw=1.0)
    
    # Highlight corresponding single pixel (Shifted to 1,1)
    b_col, b_row = 1, 1
    bx0, by0 = b_col * step_bottom, b_row * step_bottom
    
    rect_bot = patches.Rectangle((bx0, by0), step_bottom, step_bottom, 
                                 linewidth=0, facecolor='#a4ff4a', alpha=0.8)
    rect_bot_edge = patches.Rectangle((bx0, by0), step_bottom, step_bottom, 
                                 linewidth=2, edgecolor='#a4ff4a', facecolor='none')
    ax_bottom.add_patch(rect_bot)
    ax_bottom.add_patch(rect_bot_edge)
    
    # Text "PÍXEL DE SALIDA"
    #fig.text(0.5, 0.06, r"\textbf{P\'IXEL DE SALIDA}", color="black", fontsize=16, ha="center", va="top")

    # --- Connecting Arrows (from Top to Bottom) ---
    # Points in the top filter (let's use the center of each of the 3 columns)
    start_points = [
        (x0 + 0.5*step_top, y0 + 1.5*step_top),
        (x0 + 1.5*step_top, y0 + 1.5*step_top),
        (x0 + 2.5*step_top, y0 + 1.5*step_top)
    ]
    
    # Destination point in the bottom cell (top edge, center)
    dest_point = (bx0 + 0.5*step_bottom, by0)
    
    for pt in start_points:
        con = patches.ConnectionPatch(
            xyA=pt, xyB=dest_point,
            coordsA='data', coordsB='data',
            axesA=ax_top, axesB=ax_bottom,
            color='#ff3333', linewidth=3.5, alpha=0.8
        )
        fig.add_artist(con)
        
    # Add a big arrowhead at the destination using an invisible line
    center_start = (x0 + 1.5*step_top, y0 + 1.5*step_top)
    con_arrow = patches.ConnectionPatch(
        xyA=center_start, xyB=dest_point,
        coordsA='data', coordsB='data',
        axesA=ax_top, axesB=ax_bottom,
        color='#ff3333', linewidth=0,
        arrowstyle='-|>,head_width=1.0,head_length=1.2', mutation_scale=30
    )
    fig.add_artist(con_arrow)

    save_path = os.path.join(args.output_dir, f"cnn_receptive_field_{args.target_id}.png")
    plt.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    logger.info(f"Saved CNN Receptive Field diagram to {save_path}")
    
    # ---------------------------------------------------------
    # Create the Top-Only Diagram
    # ---------------------------------------------------------
    fig2 = plt.figure(figsize=(7, 7), dpi=200)
    fig2.patch.set_facecolor('white')
    
    ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
    ax2.imshow(rgb, origin='upper')
    ax2.axis('off')
    
    draw_grid(ax2, 224, n_top, color='white', alpha=0.4, lw=0.8)
    
    rect2 = patches.Rectangle((x0, y0), f_size * step_top, f_size * step_top, 
                                 linewidth=0, facecolor='#a4ff4a', alpha=0.6)
    rect2_edge = patches.Rectangle((x0, y0), f_size * step_top, f_size * step_top, 
                                 linewidth=3, edgecolor='#a4ff4a', facecolor='none')
    ax2.add_patch(rect2)
    ax2.add_patch(rect2_edge)
    
    for i in range(1, f_size):
        ax2.plot([x0 + i*step_top, x0 + i*step_top], [y0, y0 + f_size*step_top], color='#1a1a1a', lw=1.5, alpha=0.8)
        ax2.plot([x0, x0 + f_size*step_top], [y0 + i*step_top, y0 + i*step_top], color='#1a1a1a', lw=1.5, alpha=0.8)

    ax2.text(x0 + f_size*step_top + 30, y0 + (f_size/2)*step_top, 
                r"\textbf{3$\times$3 FILTER}", color="black", fontsize=16, 
                ha="left", va="center", bbox=bbox_props)
    
    ax2.plot([x0 + f_size*step_top, x0 + f_size*step_top + 30], 
                [y0 + (f_size/2)*step_top, y0 + (f_size/2)*step_top], color='black', lw=1.5)

    save_path_top = os.path.join(args.output_dir, f"cnn_receptive_field_{args.target_id}_top_only.png")
    plt.savefig(save_path_top, bbox_inches='tight', facecolor=fig2.get_facecolor())
    logger.info(f"Saved CNN Top-Only diagram to {save_path_top}")

if __name__ == "__main__":
    main()
