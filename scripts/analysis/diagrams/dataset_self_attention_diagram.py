#!/usr/bin/env python3
"""
AstroPT Self-Attention Diagram Generator
==============================================
Generates a Matplotlib diagram showing the self-attention mechanism 
(a query patch attending to other patches globally) on real Euclid galaxy data.
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
logger = logging.getLogger("self_attention_diagram")

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

def draw_grid(ax, img_size, n_cells, color='black', alpha=0.8, lw=2.0):
    step = img_size / n_cells
    for i in range(1, n_cells):
        ax.axhline(i * step, color=color, alpha=alpha, lw=lw)
        ax.axvline(i * step, color=color, alpha=alpha, lw=lw)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_id", type=int, default=39633516366922547)
    parser.add_argument("--data_dir", type=str, default="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow")
    parser.add_argument("--output_dir", type=str, default="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/dataset_logs/attention_diagrams")
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
    fig = plt.figure(figsize=(8, 8), dpi=200)
    fig.patch.set_facecolor('white') 
    
    # --- Main Axes (Input Image with Grid) ---
    ax = fig.add_axes([0.1, 0.15, 0.8, 0.8])
    
    # Use normal image brightness (same as CNN)
    ax.imshow(rgb, origin='upper')
    ax.axis('off')
    
    # 16x16 Grid of Patches
    n_patches = 16
    step = 224 / n_patches
    # Dibuja el grid
    draw_grid(ax, 224, n_patches, color='white', alpha=0.4, lw=0.8)
    
    # --- QUERY PATCH ---
    # Columna 8, Fila 8 (Centro del grid de 16x16)
    q_col, q_row = 8, 8
    qx0, qy0 = q_col * step, q_row * step
    
    # Green Highlight for the query patch (matching CNN)
    q_color = '#a4ff4a' 
    rect_q = patches.Rectangle((qx0, qy0), step, step, 
                                 linewidth=0, facecolor=q_color, alpha=1, zorder=9)
    rect_q_edge = patches.Rectangle((qx0, qy0), step, step, 
                                 linewidth=3, edgecolor=q_color, facecolor='none', zorder=10)
    ax.add_patch(rect_q)
    ax.add_patch(rect_q_edge)
    
    # Center of Query Patch
    q_cx = qx0 + step / 2
    q_cy = qy0 + step / 2

    # Callout text for "QUERY PATCH"
    # Lo colocamos un poco desplazado abajo a la derecha del recuadro para que se vea claro
    #bbox_props = dict(boxstyle="round,pad=0.5", fc="#333333", ec="none", alpha=0.9)
    #ax.text(qx0 + step, qy0 + step + 5, 
    #        r"\textbf{QUERY PATCH}", color="white", fontsize=14, 
    #        ha="left", va="top", bbox=bbox_props, zorder=12)
            
    # Línea blanca que une la caja de texto con el parche amarillo
    #ax.plot([q_cx, qx0 + step + 5], [q_cy, qy0 + step + 5], color='white', lw=1.5, zorder=11)

    # --- ATTENTION ARROWS ---
    # Como el parche origen ahora es (3,3), redistribuimos las dianas para que radien desde el centro
    # Hacia todas las direcciones (arriba, abajo, izquierda, derecha) adaptado a 16x16
    targets = [
        (2, 1, '#00E5FF', 4.0, 0.8, 0.0),    # Cyan, hacia arriba-izquierda
        (14, 2, '#FFA500', 3.5, 0.9, 0.1),   # Orange, hacia arriba-derecha
        (2, 13, '#FF69B4', 3.0, 0.8, -0.2),  # Pink, hacia abajo-izquierda
        (3, 8, '#00E5FF', 2.5, 0.7, 0.2),    # Cyan, izquierda curvo
        (13, 11, '#FF00FF', 3.5, 0.8, -0.3), # Magenta, derecha-abajo curvo
        (8, 3, '#FF4500', 2.0, 0.6, 0.0),    # Red/Orange, arriba recto
        (6, 11, '#FFA500', 2.5, 0.7, 0.0),   # Orange, abajo recto
        (11, 6, '#DEB887', 3.0, 0.8, 0.1),   # Burlywood/Tan, derecha-arriba
        (3, 6, '#FF69B4', 2.0, 0.6, 0.2),    # Pink, izquierda-arriba curvo
        (11, 13, '#FF00FF', 3.0, 0.7, -0.1), # Magenta, abajo-derecha
    ]
    
    for t_col, t_row, color, lw, alpha, rad in targets:
        # Center of target patch
        t_cx = t_col * step + step / 2
        t_cy = t_row * step + step / 2
        
        # Add some random jitter so arrows don't all point to the exact dead center
        t_cx += np.random.uniform(-step/4, step/4)
        t_cy += np.random.uniform(-step/4, step/4)
        
        # To make arrows originate from slightly different parts of the query patch
        q_cx_jitter = q_cx + np.random.uniform(-step/5, step/5)
        q_cy_jitter = q_cy + np.random.uniform(-step/5, step/5)
        
        style = f"arc3,rad={rad}" if rad != 0 else "arc3,rad=0.0"
        
        # Puntas de flecha muy discretas y pequeñas (mutation_scale=10, head_width=1.5)
        arrow = patches.FancyArrowPatch(
            (q_cx_jitter, q_cy_jitter), (t_cx, t_cy),
            connectionstyle=style,
            color=color,
            linewidth=lw,
            alpha=alpha,
            arrowstyle='-|>,head_width=1.5,head_length=2.0',
            mutation_scale=10,
            zorder=5
        )
        ax.add_patch(arrow)

    save_path = os.path.join(args.output_dir, f"self_attention_global_{args.target_id}.png")
    plt.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    logger.info(f"Saved Self-Attention diagram to {save_path}")

if __name__ == "__main__":
    main()
