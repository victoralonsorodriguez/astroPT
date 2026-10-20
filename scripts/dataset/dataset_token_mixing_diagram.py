#!/usr/bin/env python3
"""
AstroPT Token Mixing Diagram Generator
==============================================
Generates a methodological diagram showing the tokenization of the image and spectrum,
and the assembly of the interleaved mixed sequence.
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
logger = logging.getLogger("token_mixing_diagram")

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

def draw_grid(ax, img_size, n_cells, color='white', alpha=0.5, lw=1.0):
    step = img_size / n_cells
    for i in range(1, n_cells):
        ax.axhline(i * step, color=color, alpha=alpha, lw=lw)
        ax.axvline(i * step, color=color, alpha=alpha, lw=lw)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_id", type=int, default=39633516366922547)
    parser.add_argument("--data_dir", type=str, default="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow")
    parser.add_argument("--output_dir", type=str, default="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/dataset_logs/token_mixing")
    
    # Parámetros configurables del esquema
    parser.add_argument("--m_patches", type=int, default=8, help="Número de parches espaciales (m x m) en la imagen")
    parser.add_argument("--n_points", type=int, default=250, help="Número de puntos por token espectral")
    parser.add_argument("--l_vis", type=int, default=3, help="Bloque dinámico: número de tokens visuales seguidos")
    parser.add_argument("--l_spec", type=int, default=3, help="Bloque dinámico: número de tokens espectrales seguidos")
    parser.add_argument("--max_tokens_row", type=int, default=18, help="Número máximo de tokens por fila antes de plegar")
    parser.add_argument("--num_sequence_rows", type=int, default=4, help="Número de filas de secuencia a dibujar")
    
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

    # Carga de datos
    rgb = extract_raw_rgb(record)
    spec_flux = np.array(record['spectrum_flux']).flatten()
    spec_wave = np.array(record['spectrum_wave']).flatten()
    
    # Recorte para evitar zonas vacías en los bordes del espectro si los hay
    valid_idx = np.where(spec_flux != 0)[0]
    if len(valid_idx) > 0:
        spec_flux = spec_flux[valid_idx[0]:valid_idx[-1]+1]
        spec_wave = spec_wave[valid_idx[0]:valid_idx[-1]+1]
    
    # ---------------------------------------------------------
    # Preparación de Parches y Tokens (En espiral)
    # ---------------------------------------------------------
    img_size = rgb.shape[0] # asume imagen cuadrada (224)
    step_img = img_size // args.m_patches
    
    def get_spiral_indices(m):
        r, c = m // 2, m // 2
        if m % 2 == 0:
            r, c = m // 2, m // 2 - 1
        indices = []
        dr = [0, -1, 0, 1] # Right, Up, Left, Down (Anti-horario)
        dc = [1, 0, -1, 0]
        dir_idx = 0
        steps_in_dir = 1
        steps_taken = 0
        changes = 0
        # Expandir la búsqueda para garantizar que cubrimos todo
        while len(indices) < m * m:
            if 0 <= r < m and 0 <= c < m:
                if (r, c) not in indices:
                    indices.append((r, c))
            r += dr[dir_idx]
            c += dc[dir_idx]
            steps_taken += 1
            if steps_taken == steps_in_dir:
                steps_taken = 0
                dir_idx = (dir_idx + 1) % 4
                changes += 1
                if changes == 2:
                    changes = 0
                    steps_in_dir += 1
        return indices

    spiral_order = get_spiral_indices(args.m_patches)
    
    # Extraer tokens visuales siguiendo el orden espiral guardando el índice
    vis_tokens = []
    for i, (r, c) in enumerate(spiral_order):
        patch = rgb[r*step_img:(r+1)*step_img, c*step_img:(c+1)*step_img, :]
        vis_tokens.append((i, patch))
            
    # Extraer todos los tokens espectrales
    spec_tokens = []
    num_spec_tokens = len(spec_flux) // args.n_points
    for i in range(num_spec_tokens):
        flux_chunk = spec_flux[i*args.n_points:(i+1)*args.n_points]
        wave_chunk = spec_wave[i*args.n_points:(i+1)*args.n_points]
        spec_tokens.append((wave_chunk, flux_chunk))
        
    # Construir la secuencia mezclada real
    mixed_sequence = []
    vis_idx, spec_idx = 0, 0
    
    while vis_idx < len(vis_tokens) or spec_idx < len(spec_tokens):
        # Añadir bloque visual
        for _ in range(args.l_vis):
            if vis_idx < len(vis_tokens):
                mixed_sequence.append(('vis', vis_tokens[vis_idx]))
                vis_idx += 1
        # Añadir bloque espectral
        for _ in range(args.l_spec):
            if spec_idx < len(spec_tokens):
                mixed_sequence.append(('spec', spec_tokens[spec_idx]))
                spec_idx += 1

    # ---------------------------------------------------------
    # Crear el Diagrama
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(12, 10), dpi=200)
    fig.patch.set_facecolor('white') 
    
    # --- Top Right: Image ---
    ax_img = fig.add_axes([0.65, 0.60, 0.33, 0.33]) # Movido a la derecha
    ax_img.imshow(rgb, origin='upper')
    ax_img.axis('off')
    draw_grid(ax_img, img_size, args.m_patches, color='white', alpha=0.5, lw=1.2)
    
    # Escribir los números del orden en espiral en cada parche
    for i, (r, c) in enumerate(spiral_order):
        cx = c * step_img + step_img / 2
        cy = r * step_img + step_img / 2
        ax_img.text(cx, cy, str(i), color='white', fontsize=9, ha='center', va='center', fontweight='bold', alpha=0.9)
    
    # Texto de la imagen
    #ax_img.text(img_size/2, -10, rf"\textbf{{Image Patches}} ($m \times m$ Spiral)", 
    #            ha='center', va='bottom', fontsize=14)

    # --- Top Left: Spectrum ---
    ax_spec = fig.add_axes([0.06, 0.60, 0.55, 0.33]) # Espectro más ancho y alto, movido a la derecha
    ax_spec.plot(spec_wave, spec_flux, color='#000000', lw=0.75)
    ax_spec.set_xlim(spec_wave.min(), spec_wave.max())
    ax_spec.set_yticks([])
    ax_spec.tick_params(axis='x', labelsize=10)
    
    # Dibuja líneas verticales para ilustrar el particionado del espectro en tokens
    for i in range(1, num_spec_tokens):
        x_line = spec_wave[i * args.n_points]
        ax_spec.axvline(x_line, color='red', alpha=0.3, linestyle='--', lw=1.0)
        
    #ax_spec.text(0.5, 1.05, rf"\textbf{{Spectrum Intervals}} (size $n$)", 
    #             ha='center', va='bottom', transform=ax_spec.transAxes, fontsize=14)

    # --- Flechas ilustrativas ---
    # Flecha bajando desde el Espectro
    fig.patches.append(patches.FancyArrowPatch(
        (0.335, 0.57), (0.335, 0.47), transform=fig.transFigure,

        color='gray', arrowstyle='-|>,head_width=4.0,head_length=5.0', lw=2.0
    ))
    
    # Flecha bajando desde la Imagen
    fig.patches.append(patches.FancyArrowPatch(
        (0.815, 0.57), (0.815, 0.47), transform=fig.transFigure,
        color='gray', arrowstyle='-|>,head_width=4.0,head_length=5.0', lw=2.0
    ))

    # Texto central de aplanado
    fig.text(0.56, 0.53, r"\textbf{Token Mixing Final Sequence}", ha='center', va='center', fontsize=16, 
             bbox=dict(boxstyle="round,pad=0.5", fc="#f0f0f0", ec="gray", lw=1.0))

    # --- Secuencia Unificada Abajo ---
    seq_y_start = 0.40 # Empezamos un poco más abajo
    token_w = 0.04
    token_h = token_w * (fig.get_figwidth() / fig.get_figheight()) # Para que sean cuadrados perfectos en la figura
    spacing_x = 0.005
    spacing_y = 0.03
    
    total_drawn = 0
    max_to_draw = args.max_tokens_row * args.num_sequence_rows
    
    # Título de la secuencia
    #fig.text(0.05, seq_y_start + token_h + 0.02, 
    #         rf"\textbf{{Unified Sequence}} (Interleaved chunks of $l_{{vis}}={args.l_vis}$ and $l_{{spec}}={args.l_spec}$ tokens)", 
    #         fontsize=14, ha='left')

    for row in range(args.num_sequence_rows):
        if total_drawn >= len(mixed_sequence): break
        
        y_pos = seq_y_start - row * (token_h + spacing_y)
        
        for col in range(args.max_tokens_row):
            if total_drawn >= len(mixed_sequence): break
            
            x_pos = 0.05 + col * (token_w + spacing_x)
            
            # Crear los ejes para este token específico
            ax_token = fig.add_axes([x_pos, y_pos, token_w, token_h])
            
            token_type, token_data = mixed_sequence[total_drawn]
            
            if token_type == 'vis':
                # Token Visual: Muestra el recorte de la imagen y su número
                patch_idx, patch_data = token_data
                ax_token.imshow(patch_data, origin='upper')
                ax_token.text(patch_data.shape[1]/2, patch_data.shape[0]/2, str(patch_idx), 
                              color='white', fontsize=11, ha='center', va='center', fontweight='bold', alpha=0.9)
                # Borde sutil del color modal
                for spine in ax_token.spines.values():
                    spine.set_edgecolor('#000000') # Verde para visual
                    spine.set_linewidth(1.0)
                ax_token.set_xticks([])
                ax_token.set_yticks([])
                
            elif token_type == 'spec':
                # Token Espectral: Muestra la minigráfica
                wave_c, flux_c = token_data
                ax_token.plot(wave_c, flux_c, color='#000000', lw=0.75)
                ax_token.set_xlim(wave_c.min(), wave_c.max())
                ax_token.set_facecolor('#f4f9fd') # Fondo ligerísimamente azulado
                # Borde sutil del color modal
                for spine in ax_token.spines.values():
                    spine.set_edgecolor('#000000') # Azul para espectro
                    spine.set_linewidth(1.0)
                ax_token.set_xticks([])
                ax_token.set_yticks([])
            
            total_drawn += 1

    # Leyenda visual modal
    #fig.text(0.05, seq_y_start - (args.num_sequence_rows) * (token_h + spacing_y), 
    #         r"\textbf{$\square$ Visual Token} \quad \textbf{$\blacksquare$ Spectral Token}", 
    #         color="black", fontsize=12, ha='left')
             
    # Rectángulos de la leyenda pintados a mano para ser precisos
    # Verde
    #fig.patches.append(patches.Rectangle(
    #    (0.045, seq_y_start - (args.num_sequence_rows) * (token_h + spacing_y) - 0.005), 
    #    0.015, 0.015, transform=fig.transFigure, fill=False, ec='#2ca02c', lw=2.0
    #))
    # Azul
    #fig.patches.append(patches.Rectangle(
    #    (0.18, seq_y_start - (args.num_sequence_rows) * (token_h + spacing_y) - 0.005), 
    #    0.015, 0.015, transform=fig.transFigure, fill=True, fc='#f4f9fd', ec='#1f77b4', lw=2.0
    #))


    save_path = os.path.join(args.output_dir, f"token_mixing_{args.target_id}.png")
    plt.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    logger.info(f"Saved Token Mixing diagram to {save_path}")

if __name__ == "__main__":
    main()
