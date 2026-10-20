#!/usr/bin/env python3
"""
AstroPT Multimodal Similarity Search Tool
==================================================
Description:
    Takes a Query TargetID and retrieves the Top K nearest neighbors in three 
    independent latent spaces:
    1. Image space (EuclidImage) - morphological matches
    2. Spectrum space (DESISpectrum) - spectral/physical matches
    3. Joint space (joint) - morpho-spectroscopic matches
    
    Generates an premium multi-page PDF report demonstrating AstroPT's 
    representation learning capabilities.
"""

import argparse
import sys
import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import fields
from contextlib import nullcontext

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from astropy.table import Table
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from astropt.dataloader_multimodal import MultimodalDatasetArrow
from astropt.training_utils import create_dataloaders
from astropt.config import TrainingConfig
from astropt.model_utils import load_local_model

# --- Configure logging ---
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-SimilaritySearch")

# Configure Matplotlib Headless Mode immediately to prevent GUI thread issues in SLURM
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Check LaTeX availability and configure rcParams
usetex_val = True
try:
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rc('text', usetex=True)
except Exception as e:
    logger.warning(f"LaTeX is not available, falling back to plain text formatting: {e}")
    plt.rc('text', usetex=False)
    usetex_val = False

def clean_latex(text: str) -> str:
    """Removes LaTeX command structures if LaTeX rendering is disabled."""
    if usetex_val:
        return text
    import re
    cleaned = text
    cleaned = re.sub(r'\\textbf\{([^}]*)\}', r'\1', cleaned)
    cleaned = re.sub(r'\\text\{([^}]*)\}', r'\1', cleaned)
    cleaned = re.sub(r'\\bm\{([^}]*)\}', r'\1', cleaned)
    cleaned = re.sub(r'\\alpha', 'alpha', cleaned)
    cleaned = re.sub(r'\\beta', 'beta', cleaned)
    cleaned = re.sub(r'\\sigma', 'sigma', cleaned)
    cleaned = re.sub(r'\\theta', 'theta', cleaned)
    cleaned = re.sub(r'\\cos', 'cos', cleaned)
    cleaned = re.sub(r'\\circ', ' deg', cleaned)
    cleaned = re.sub(r'\\AA', 'A', cleaned)
    cleaned = re.sub(r'\\$', '', cleaned)
    cleaned = cleaned.replace(r'\#', '#').replace(r'\_', '_').replace(r'\^', '^')
    return cleaned


class CombinedDatasetWrapper:
    def __init__(self, train_ds, val_ds):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.train_ids = np.array(train_ds.ds['targetid'], dtype=np.int64) if len(train_ds) > 0 else np.array([], dtype=np.int64)
        self.val_ids = np.array(val_ds.ds['targetid'], dtype=np.int64) if len(val_ds) > 0 else np.array([], dtype=np.int64)
        self.combined_ids = np.concatenate([self.train_ids, self.val_ids])
        
    @property
    def ds(self):
        return self
        
    def __getitem__(self, key):
        if key == 'targetid':
            return self.combined_ids
        idx = int(key)
        n_train = len(self.train_ids)
        if idx < n_train:
            return self.train_ds.ds[idx]
        else:
            return self.val_ds.ds[idx - n_train]


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="AstroPT Multimodal Similarity Search")
    
    parser.add_argument("--embeddings_dir", type=str, required=True, 
                        help="Directory containing pre-computed embeddings (.npy)")
    parser.add_argument("--ckpt_path", type=str, required=True, 
                        help="Path to the training checkpoint (.pt)")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Root folder of raw Arrow processed data")
    parser.add_argument("--metadata_path", type=str, required=True, 
                        help="Path to FITS metadata catalog")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Saving directory for similarity search dashboards")
    
    parser.add_argument("--query_id", type=int, default=None, 
                        help="TargetID of the query galaxy")
    parser.add_argument("--query_fits", type=str, default=None,
                        help="Path to a custom FITS file or directory of FITS files to query")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Optional CSV filename to save similarity search results")
    parser.add_argument("--k", type=int, default=5, 
                        help="Number of nearest neighbors to retrieve")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for dataloader")
    
    args = parser.parse_args()
    if args.query_id is None and args.query_fits is None:
        parser.error("Either --query_id or --query_fits must be provided.")
    return args


# ---------------------------------------------------------------------------
# Helper functions for RGB and Spectral plotting (LaTeX safe)
# ---------------------------------------------------------------------------

MAIN_LINES = {
    r"Ly$\alpha$": 1216.0, r"C IV": 1549.0, "C III": 1908.7, r"Mg II": 2798.0, r"[O II]": 3727.3, r"[Ne III]": 3868.7,
    r"Ca K": 3933.7, r"Ca H": 3968.5, r"H$\delta$": 4102.0, r"H$\gamma$": 4341.0,
    r"H$\beta$": 4861.0, r"[O III]": 4959.0, r"[O III]": 5007.0, r"Mg I": 5175.0,
    r"Na D": 5890.0, r"[N II]": 6548.0, r"H$\alpha$": 6563.0, r"[N II]": 6583.5, r"[S II]": 6730.8
}

def plot_spectral_lines(ax, min_wl, max_wl, z):
    """Annotates spectral lines with alternating heights."""
    if z is None or pd.isna(z):
        return
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    pos_high = y_min + (y_range * 0.95)
    pos_low  = y_min + (y_range * 0.25)
    
    sorted_lines = sorted(MAIN_LINES.items(), key=lambda x: x[1])
    counter = 0
    
    for name, rest_wave in sorted_lines:
        obs_wave = rest_wave * (1 + z)
        if min_wl < obs_wave < max_wl:
            y_pos = pos_high if counter % 2 == 0 else pos_low
            ax.axvline(obs_wave, color='royalblue', linestyle='--', alpha=0.6, lw=1)
            ax.text(obs_wave, y_pos, rf"\textbf{{{name}}}", rotation=90, 
                    color='royalblue', va='top', ha='right', fontsize=10, alpha=1, fontweight='bold')
            counter += 1

def make_rgb_lupton(image_stack: np.ndarray, Q: float = 12.0, stretch: float = 0.5, m: float = 0.0) -> np.ndarray:
    """Lupton et al. (2004) asinh scaling."""
    intensity = np.mean(image_stack, axis=0) - m
    intensity = np.maximum(intensity, 1e-10)

    f_I = np.arcsinh(Q * stretch * intensity) / Q
    scale_factor = f_I / intensity

    rgb_out = image_stack * scale_factor[np.newaxis, :, :]
    max_rgb = np.percentile(rgb_out, 99.5)
    if max_rgb > 0:
        rgb_out = rgb_out / max_rgb

    rgb_out = np.clip(rgb_out, 0, 1)
    return rgb_out.transpose(1, 2, 0)


def make_lupton_composite(
    vis: np.ndarray,
    y_band: np.ndarray,
    j_band: np.ndarray,
    h_band: np.ndarray,
    Q: float = 12.0,
    stretch: float = 0.5,
    rgb_weights: tuple = (1.2, 1.3, 1.0),
) -> np.ndarray:
    """
    Lupton 2004 composite for 4 Euclid bands (VIS/Y/J/H):
      R = H        * w_r
      G = (J+Y)/2  * w_g
      B = VIS      * w_b
    Falls back to 3-band if H is unavailable.
    """
    vis_c = np.nan_to_num(vis,    nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    y_c   = np.nan_to_num(y_band, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    j_c   = np.nan_to_num(j_band, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    h_c   = np.nan_to_num(h_band, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    has_h = h_c.size > 0 and np.any(h_c > 0)

    if has_h:
        raw_stack = np.stack([vis_c, h_c, j_c, y_c], axis=0)
    else:
        raw_stack = np.stack([vis_c, j_c, y_c], axis=0)

    bg_val = np.percentile(raw_stack, 50, axis=(1, 2), keepdims=True)
    raw_bg = raw_stack - bg_val

    normed_channels = []
    for c in range(raw_bg.shape[0]):
        v_max = np.percentile(np.abs(raw_bg[c]), 99.5)
        if v_max <= 0:
            v_max = 1.0
        normed_channels.append(np.clip(raw_bg[c] / v_max, 0, 100))
    raw_norm = np.stack(normed_channels)

    w_r, w_g, w_b = rgb_weights

    if has_h:
        vis_ch, h_ch, j_ch, y_ch = raw_norm[0], raw_norm[1], raw_norm[2], raw_norm[3]
        r = h_ch * w_r
        g = ((j_ch + y_ch) / 2.0) * w_g
        b = vis_ch * w_b
    else:
        vis_ch, j_ch, y_ch = raw_norm[0], raw_norm[1], raw_norm[2]
        r = j_ch * w_r
        g = y_ch * w_g
        b = vis_ch * w_b

    rgb_input = np.stack([r, g, b], axis=0)
    return make_rgb_lupton(rgb_input, Q=Q, stretch=stretch)


def extract_raw_rgb(raw_record: dict) -> Optional[np.ndarray]:
    """Safely extract and reconstruct VIS+NISP false color RGB."""
    def get_raw(k):
        try:
            val = raw_record[k]
            return np.array(val if val is not None else [], dtype=np.float32)
        except (KeyError, TypeError):
            return np.array([], dtype=np.float32)

    vis = get_raw('image_vis')
    y = get_raw('image_nisp_y')
    j = get_raw('image_nisp_j')
    
    if vis.size == 0:
        return None
        
    try:
        h_val = raw_record.get('image_nisp_h', None)
        h = np.array(h_val, dtype=np.float32) if h_val is not None else np.zeros_like(vis)
    except Exception:
        h = np.zeros_like(vis)
        
    return make_lupton_composite(vis, y, j, h)


def plot_spectrum_into_axes(ax_blue, ax_red, raw_record: Optional[dict], tid: Optional[int], z_val: Optional[float] = None):
    """Plots DESI spectrum split in Blue and Red channels into the provided axes."""
    if raw_record is not None and raw_record.get('spectrum_flux') is not None:
        spec_gt = np.array(raw_record['spectrum_flux']).flatten()
        wave_ang = np.array(raw_record['spectrum_wave']).flatten()
        
        # Blue Channel
        ax_blue.plot(wave_ang, spec_gt, 'k-', lw=1, alpha=0.8)
        ax_blue.set_xlim(wave_ang.min(), (wave_ang.min() + wave_ang.max()) / 2)
        ax_blue.set_title(r"\textbf{Spectrum (Blue Channel)}", fontsize=11, fontweight='bold')
        ax_blue.set_ylabel(r"Flux", fontsize=9)
        if z_val is not None:
            plot_spectral_lines(ax_blue, wave_ang.min(), (wave_ang.min() + wave_ang.max()) / 2, z_val)
        
        # Red Channel
        ax_red.plot(wave_ang, spec_gt, 'k-', lw=1, alpha=0.8)
        ax_red.set_xlim((wave_ang.min() + wave_ang.max()) / 2, wave_ang.max())
        ax_red.set_title(r"\textbf{Spectrum (Red Channel)}", fontsize=11, fontweight='bold')
        ax_red.set_ylabel(r"Flux", fontsize=9)
        ax_red.set_xlabel(r"Observed Wavelength [\AA]", fontsize=10, fontweight='bold')
        if z_val is not None:
            plot_spectral_lines(ax_red, (wave_ang.min() + wave_ang.max()) / 2, wave_ang.max(), z_val)
    else:
        ax_blue.text(0.5, 0.5, "DESISpectrum not found / available", ha='center', va='center', fontsize=12)
        ax_blue.axis('off')
        ax_red.axis('off')


def extract_rgb_from_fits_file(fits_path: Path) -> Optional[np.ndarray]:
    from astropy.io import fits
    try:
        with fits.open(fits_path) as hdul:
            data = None
            for hdu in hdul:
                if hdu.data is not None:
                    data = hdu.data
                    break
        if data is None:
            logger.error(f"FITS file {fits_path} contains no image data in any HDU.")
            return None
        if data.ndim == 2:
            data = np.stack([data, data, data, data], axis=0)
        elif data.ndim != 3:
            logger.error(f"FITS data in {fits_path} has invalid dimensions: {data.shape} (must be 2D or 3D)")
            return None
            
        c, h, w = data.shape
        if h > 224 or w > 224:
            sh = (h - 224) // 2
            sw = (w - 224) // 2
            data = data[:, sh:sh+224, sw:sw+224]
            
        bg_val = np.percentile(data, 50, axis=(1,2), keepdims=True)
        raw_bg = data - bg_val
        
        vis_ch = raw_bg[0]
        j_ch = raw_bg[2]
        y_ch = raw_bg[3]
        
        return make_hsl_composite(vis_ch, y_ch, j_ch)
    except Exception as e:
        logger.error(f"Error extracting RGB from fits {fits_path}: {e}")
        return None


def get_embedding_from_fits(fits_path: Path, model: nn.Module, device: torch.device, training_config: Any = None) -> Optional[torch.Tensor]:
    from astropy.io import fits
    from astropt.processors.image import EuclidImageProcessor
    try:
        with fits.open(fits_path) as hdul:
            data = None
            for hdu in hdul:
                if hdu.data is not None:
                    data = hdu.data
                    break
        if data is None:
            logger.error(f"FITS file {fits_path} contains no image data in any HDU for embedding.")
            return None
        if data.ndim == 2:
            data = np.stack([data, data, data, data], axis=0)
        elif data.ndim != 3:
            logger.error(f"FITS data in {fits_path} has invalid dimensions: {data.shape} for embedding.")
            return None
            
        if data.shape[0] == 4:
            data = data[[0, 3, 2, 1]]
            
        c, h, w = data.shape
        if h > 224 or w > 224:
            sh = (h - 224) // 2
            sw = (w - 224) // 2
            data = data[:, sh:sh+224, sw:sw+224]
            
        cropped_tensor = torch.from_numpy(data.copy()).to(torch.float32)
        p_size = 8
        c, h, w = cropped_tensor.shape
        patch_galaxy = cropped_tensor.reshape(c, h // p_size, p_size, w // p_size, p_size)
        patch_galaxy = patch_galaxy.permute(1, 3, 2, 4, 0)
        patch_galaxy = patch_galaxy.reshape((h // p_size) * (w // p_size), p_size * p_size * c)
        
        # Apply normalization according to config
        norm_type = "z_score"
        norm_scaler = 1.0
        norm_const = 1.0
        if training_config is not None:
            norm_type = getattr(training_config, 'images_norm_type', 'z_score')
            norm_scaler = getattr(training_config, 'images_norm_scaler', 1.0)
            norm_const = getattr(training_config, 'images_norm_const', 1.0)

        if norm_type == "z_score":
            std, mean = torch.std_mean(patch_galaxy, dim=1, keepdim=True)
            patch_galaxy = (patch_galaxy - mean) / (std + 1e-8)
        elif norm_type == "asinh":
            patch_galaxy = torch.asinh(patch_galaxy / norm_scaler) / norm_const
        elif norm_type == "constant":
            patch_galaxy = patch_galaxy / norm_const
        
        processor = EuclidImageProcessor(spiral=True)
        patch_galaxy = processor._spiralise(patch_galaxy)
        
        positions = torch.arange(patch_galaxy.shape[0], dtype=torch.long)
        
        model_dtype = next(model.parameters()).dtype
        flux_batch = patch_galaxy.unsqueeze(0).to(device=device, dtype=model_dtype)
        pos_batch = positions.unsqueeze(0).to(device=device, dtype=torch.long)
        
        inputs = {"EuclidImage": flux_batch, "EuclidImage_positions": pos_batch}
        
        ctx = torch.amp.autocast(device_type="cuda", dtype=model_dtype) if (device.type == "cuda" and model_dtype == torch.bfloat16) else nullcontext()
        with torch.no_grad(), ctx:
            emb = model.get_embeddings(inputs, draw_from_centre=False)
            
        pooled_emb = emb["EuclidImage"].mean(dim=1).squeeze(0)
        return pooled_emb
    except Exception as e:
        logger.error(f"Error extracting embedding from fits {fits_path}: {e}")
        return None


def extract_and_search_fits(fits_file: Path, model: nn.Module, device: torch.device, norm_X: np.ndarray, ids: np.ndarray, ds: Any, ds_ids: np.ndarray, fits_indexed: pd.DataFrame, save_dir: Path, args: argparse.Namespace, training_config: Any = None) -> Dict[str, Any]:
    query_name = fits_file.stem
    logger.info(f"Processing FITS Query: {fits_file.name}")
    
    query_rgb = extract_rgb_from_fits_file(fits_file)
    query_emb = get_embedding_from_fits(fits_file, model, device, training_config)
    if query_emb is None:
        logger.warning(f"Could not extract embedding for {fits_file.name}, skipping.")
        return {}
        
    query_emb_np = query_emb.cpu().numpy()
    norm_query = query_emb_np / (np.linalg.norm(query_emb_np) + 1e-8)
    sims = np.dot(norm_X, norm_query)
    
    top_indices = np.argsort(sims)[::-1][:args.k]
    top_ids = ids[top_indices]
    top_sims = sims[top_indices]
    
    logger.info(f"Top {args.k} Neighbors for FITS Query {fits_file.name}:")
    for idx, (tid, sim) in enumerate(zip(top_ids, top_sims)):
        logger.info(f"  Rank #{idx+1}: TargetID={tid} | Cosine Similarity={sim:.4%}")
        
    pdf_path = save_dir / f"similarity_search_{query_name}.pdf"
    logger.info(f"Generating PDF Similarity Search Report at: {pdf_path}")
    
    def get_morphological_meta_str(tid):
        if tid in fits_indexed.index:
            meta = fits_indexed.loc[tid]
            ra_val = meta.get('RA', None)
            dec_val = meta.get('DEC', None)
            coords_str = f"${ra_val:.6f}^\\circ, {dec_val:.6f}^\\circ$" if (ra_val is not None and not pd.isna(ra_val)) else "N/A"
            z_val = meta.get('Z', None)
            z_str = f"{z_val:.4f}" if (z_val is not None and not pd.isna(z_val)) else "N/A"
            logmstar_val = meta.get('LOGMSTAR', None)
            mstar = f"$10^{{{logmstar_val:.2f}}} \\text{{ M}}_\\odot$" if (logmstar_val is not None and not pd.isna(logmstar_val)) else "N/A"
            logsfr_val = meta.get('LOGSFR', None)
            sfr = f"${10**logsfr_val:.3f} \\text{{ M}}_\\odot/\\text{{yr}}$" if (logsfr_val is not None and not pd.isna(logsfr_val)) else "N/A"
            spectype = str(meta.get('SPECTYPE', 'N/A'))
            sersic_n_val = meta.get('sersic_sersic_vis_index', None)
            n_str = f"{sersic_n_val:.2f}" if (sersic_n_val is not None and not pd.isna(sersic_n_val)) else "N/A"
            sersic_re_val = meta.get('sersic_sersic_vis_radius', None)
            re_str = f"{sersic_re_val:.3f}''" if (sersic_re_val is not None and not pd.isna(sersic_re_val)) else "N/A"
            ba_val = meta.get('sersic_sersic_vis_axis_ratio', None)
            ba_str = f"{ba_val:.3f}" if (ba_val is not None and not pd.isna(ba_val)) else "N/A"
            vis_flux_val = meta.get('flux_vis_1fwhm_aper', None)
            vis_flux_str = f"{vis_flux_val:.2f}" if (vis_flux_val is not None and not pd.isna(vis_flux_val)) else "N/A"
            meta_str = (
                f"\\textbf{{Coords}}: {coords_str}\n"
                f"\\textbf{{Z}}: {z_str}\n"
                f"\\textbf{{$M_*$}}: {mstar}\n"
                f"\\textbf{{SFR}}: {sfr}\n"
                f"\\textbf{{Spectype}}: {spectype}\n"
                f"\\textbf{{VIS Flux}}: {vis_flux_str}\n"
                f"\\textbf{{Sersic n}}: {n_str}\n"
                f"\\textbf{{Radius $R_{{eff}}$}}: {re_str}\n"
                f"\\textbf{{Axis Ratio}}: {ba_str}"
            )
            return meta_str.replace("_", "\\_"), z_val
        return f"\\textbf{{ID}}: {tid}\n(No metadata)", None

    def get_spectroscopic_meta_str(tid):
        if tid in fits_indexed.index:
            meta = fits_indexed.loc[tid]
            ha_f = meta.get('HALPHA_FLUX', None)
            hb_f = meta.get('HBETA_FLUX', None)
            oiii_f = meta.get('OIII_5007_FLUX', None)
            oii_f = meta.get('OII_3726_FLUX', None)
            ha_ew = meta.get('HALPHA_EW', None)
            hb_ew = meta.get('HBETA_EW', None)
            oii_ew = meta.get('OII_3726_EW', None)
            ha_sig = meta.get('HALPHA_SIGMA', None)
            snr_r = meta.get('SNR_SPEC_R', None)
            snr_z = meta.get('SNR_SPEC_Z', None)
            def fmt_flux(v): return f"{v:.2f}" if (v is not None and not pd.isna(v)) else "N/A"
            def fmt_ew(v): return f"{v:.2f} \\AA" if (v is not None and not pd.isna(v)) else "N/A"
            def fmt_val(v, unit=""): return f"{v:.1f}{unit}" if (v is not None and not pd.isna(v)) else "N/A"
            stats_text = (
                f"\\textbf{{Spectral Line Fluxes}}:\n"
                f"  - H$\\alpha$ Flux: {fmt_flux(ha_f)}\n"
                f"  - H$\\beta$ Flux: {fmt_flux(hb_f)}\n"
                f"  - [O III] 5007 Flux: {fmt_flux(oiii_f)}\n"
                f"  - [O II] 3726 Flux: {fmt_flux(oii_f)}\n\n"
                f"\\textbf{{Equivalent Widths}}:\n"
                f"  - H$\\alpha$ EW: {fmt_ew(ha_ew)}\n"
                f"  - H$\\beta$ EW: {fmt_ew(hb_ew)}\n"
                f"  - [O II] 3726 EW: {fmt_ew(oii_ew)}\n\n"
                f"\\textbf{{Spectra Quality}}:\n"
                f"  - H$\\alpha$ Width $\\sigma$: {fmt_val(ha_sig, ' km/s')}\n"
                f"  - SNR Spec R: {fmt_val(snr_r)} | SNR Spec Z: {fmt_val(snr_z)}"
            )
            return stats_text.replace("_", "\\_")
        return "\\textbf{Spectra data missing}"

    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(pdf_path) as pdf:
        # PAGE 1: Query Profile
        fig = plt.figure(figsize=(20, 14), dpi=150)
        gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], width_ratios=[1, 1.1, 0.9], hspace=0.4, wspace=0.3)
        ax_meta = fig.add_subplot(gs[0, 0])
        ax_img = fig.add_subplot(gs[0, 1])
        ax_stats = fig.add_subplot(gs[0, 2])
        ax_spec_blue = fig.add_subplot(gs[1, :])
        ax_spec_red = fig.add_subplot(gs[2, :])
        
        ax_meta.axis('off')
        query_meta = (
            f"\\textbf{{Query FITS}}: {fits_file.name}\n"
            f"\\textbf{{Bands}}: VIS, Y, J, H\n"
            f"\\textbf{{Dimensions}}: 224x224 (cropped)"
        )
        ax_meta.text(0.5, 0.95, clean_latex(query_meta), transform=ax_meta.transAxes, fontsize=14, linespacing=1.8, ha='center', va='top', bbox=dict(boxstyle="round,pad=0.8", fc="ivory", alpha=0.95, ec="darkgrey", lw=1.0))
        ax_meta.set_title(clean_latex(r"\textbf{Physical and Morphological properties}"), fontsize=16, fontweight='bold', color='navy', pad=15)
        
        if query_rgb is not None:
            ax_img.imshow(query_rgb, origin='lower')
        else:
            ax_img.text(0.5, 0.5, "EuclidImage missing", ha='center', va='center')
        ax_img.axis('off')
        
        ax_stats.axis('off')
        ax_stats.text(0.5, 0.95, clean_latex("\\textbf{Spectroscopic properties}:\nNot available for custom FITS query."), transform=ax_stats.transAxes, fontsize=14, linespacing=1.6, ha='center', va='top', bbox=dict(boxstyle="round,pad=0.8", fc="aliceblue", alpha=0.95, ec="steelblue", lw=1.0))
        ax_stats.set_title(clean_latex(r"\textbf{Spectroscopic properties}"), fontsize=16, fontweight='bold', color='steelblue', pad=15)
        
        ax_spec_blue.text(0.5, 0.5, "Spectrum not available for custom query", ha='center', va='center', fontsize=14)
        ax_spec_blue.axis('off')
        ax_spec_red.axis('off')
        
        plt.suptitle(clean_latex(rf"\textbf{{AstroPT similarity search: Custom Query ({fits_file.name})}}"), fontsize=22, fontweight='bold', y=0.98)
        pdf.savefig()
        plt.close()
        
        # PAGE 2: Morphological Grid
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=150)
        fig.subplots_adjust(hspace=0.3, wspace=0.2)
        ax_q = axes[0, 0]
        if query_rgb is not None:
            ax_q.imshow(query_rgb, origin='lower')
        ax_q.axis('off')
        ax_q.patch.set_edgecolor('navy')
        ax_q.patch.set_linewidth(3)
        
        for idx, tid in enumerate(top_ids[:5]):
            row_ax = (idx + 1) // 3
            col_ax = (idx + 1) % 3
            ax_n = axes[row_ax, col_ax]
            n_matches = np.where(ds_ids == tid)[0]
            n_rgb = None
            if len(n_matches) > 0:
                n_rec = ds.ds[int(n_matches[0])]
                n_rgb = extract_raw_rgb(n_rec)
            if n_rgb is not None:
                ax_n.imshow(n_rgb, origin='lower')
            else:
                ax_n.text(0.5, 0.5, f"Image not found | ID: {tid}", ha='center', va='center')
            ax_n.axis('off')
            
        plt.suptitle(clean_latex(r"\textbf{Euclid Morphological Neighbors}"), fontsize=22, fontweight='bold', y=0.98)
        pdf.savefig()
        plt.close()
        
        # PAGES 3-7: Detailed profiles
        for idx, (tid, sim) in enumerate(zip(top_ids, top_sims)):
            n_matches = np.where(ds_ids == tid)[0]
            if len(n_matches) == 0:
                continue
            n_rec = ds.ds[int(n_matches[0])]
            n_rgb = extract_raw_rgb(n_rec)
            n_meta, n_z = get_morphological_meta_str(tid)
            n_spec_meta = get_spectroscopic_meta_str(tid)
            
            fig = plt.figure(figsize=(20, 14), dpi=150)
            gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], width_ratios=[1, 1.1, 0.9], hspace=0.4, wspace=0.3)
            ax_meta = fig.add_subplot(gs[0, 0])
            ax_img = fig.add_subplot(gs[0, 1])
            ax_stats = fig.add_subplot(gs[0, 2])
            ax_spec_blue = fig.add_subplot(gs[1, :])
            ax_spec_red = fig.add_subplot(gs[2, :])
            
            ax_meta.axis('off')
            ax_meta.text(0.5, 0.95, clean_latex(n_meta), transform=ax_meta.transAxes, fontsize=14, linespacing=1.8, ha='center', va='top', bbox=dict(boxstyle="round,pad=0.8", fc="ivory", alpha=0.95, ec="darkgrey", lw=1.0))
            ax_meta.set_title(clean_latex(r"\textbf{Physical and Morphological properties}"), fontsize=16, fontweight='bold', color='navy', pad=15)
            
            if n_rgb is not None:
                ax_img.imshow(n_rgb, origin='lower')
            else:
                ax_img.text(0.5, 0.5, "EuclidImage missing", ha='center', va='center')
            ax_img.axis('off')
            
            ax_stats.axis('off')
            ax_stats.text(0.5, 0.95, clean_latex(n_spec_meta), transform=ax_stats.transAxes, fontsize=14, linespacing=1.6, ha='center', va='top', bbox=dict(boxstyle="round,pad=0.8", fc="aliceblue", alpha=0.95, ec="steelblue", lw=1.0))
            ax_stats.set_title(clean_latex(r"\textbf{Spectroscopic properties}"), fontsize=16, fontweight='bold', color='steelblue', pad=15)
            
            plot_spectrum_into_axes(ax_spec_blue, ax_spec_red, n_rec, tid, n_z)
            ax_spec_blue.set_title(clean_latex(rf"\textbf{{Spectrum (Blue Channel)}}"), fontsize=11, fontweight='bold')
            
            plt.suptitle(clean_latex(rf"\textbf{{Euclid Morphological Rank \#{idx+1} ({sim*100:.2f}\%): ID: {tid}}}"), fontsize=22, fontweight='bold', y=0.98)
            pdf.savefig()
            plt.close()
            
    row_dict = {"query_fits": fits_file.name}
    for idx, (tid, sim) in enumerate(zip(top_ids, top_sims)):
        row_dict[f"neighbor_{idx+1}_id"] = tid
        row_dict[f"neighbor_{idx+1}_similarity"] = sim
    return row_dict


# ---------------------------------------------------------------------------
# Search Math Engine
# ---------------------------------------------------------------------------

def compute_similarity(query_idx: int, X: np.ndarray, ids: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Computes cosine similarity between query vec and all matrix rows."""
    query_vec = X[query_idx]
    
    # L2 normalize
    norm_X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    norm_query = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    
    # Cosine similarities
    sims = np.dot(norm_X, norm_query)
    
    # Exclude query itself by setting similarity to -2.0
    sims[query_idx] = -2.0
    
    # Get top k indices in descending order
    top_indices = np.argsort(sims)[::-1][:k]
    return ids[top_indices], sims[top_indices]


# ---------------------------------------------------------------------------
# Main similarity pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    
    embeddings_dir = Path(args.embeddings_dir)
    save_dir = Path(args.output_dir) if args.output_dir else embeddings_dir / "similarity"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load active embeddings
    logger.info("Loading pre-computed embeddings...")
    emb_files = {f.stem: f for f in embeddings_dir.glob("*.npy")}
    
    if "ids" not in emb_files:
        raise FileNotFoundError(f"ids.npy not found in {embeddings_dir}")
        
    ids = np.load(embeddings_dir / "ids.npy").astype(np.int64)
    
    # If FITS mode is requested
    if args.query_fits is not None:
        logger.info("\nInitializing active AstroPT Datasets and Model via checkpoint for FITS mode...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, config, registry, raw_config_dict = load_local_model(Path(args.ckpt_path), device)
        
        raw_config_dict['data_dir'] = args.data_dir
        raw_config_dict['batch_size'] = args.batch_size
        
        valid_keys = {f.name for f in fields(TrainingConfig)}
        clean_config_dict = {k: v for k, v in raw_config_dict.items() if k in valid_keys}
        training_config = TrainingConfig(**clean_config_dict)
        
        train_loader, val_loader, _ = create_dataloaders(training_config, ddp=False)
        ds = CombinedDatasetWrapper(train_loader.dataset, val_loader.dataset)
        ds_ids = np.array(ds.ds['targetid'])
        
        # Read FITS metadata catalog
        logger.info(f"Reading FITS metadata catalog from {args.metadata_path}...")
        fits_table = Table.read(args.metadata_path)
        fits_df = fits_table.to_pandas()
        
        # Decode string bytes
        for col in fits_df.columns:
            if fits_df[col].dtype == object and isinstance(fits_df[col].iloc[0], bytes):
                try: fits_df[col] = fits_df[col].str.decode('utf-8')
                except: pass
                
        target_id_col = 'TARGETID' if 'TARGETID' in fits_df.columns else 'targetid'
        fits_indexed = fits_df.drop_duplicates(subset=[target_id_col]).set_index(target_id_col)
        
        # Load image embeddings only since custom fits files only have images
        fallbacks = ["EuclidImage_pooled", "EuclidImage_phase2", "EuclidImage_phase1", "EuclidImage", "images"]
        found_file = None
        for fallback in fallbacks:
            if fallback in emb_files:
                found_file = emb_files[fallback]
                break
        if found_file is None:
            raise FileNotFoundError(f"Could not find EuclidImage embedding file in {embeddings_dir}")
            
        logger.info(f"Loading 'EuclidImage' space from: {found_file.name}")
        X = np.load(found_file)
        if X.ndim == 3:
            X = X.mean(axis=1)
            
        # L2 normalize X for similarity calculations
        norm_X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        
        query_path = Path(args.query_fits)
        if query_path.is_dir():
            fits_files = sorted(list(query_path.glob("**/*.fits")))
        else:
            fits_files = [query_path]
            
        if not fits_files:
            logger.error(f"No FITS files found at path {args.query_fits}")
            sys.exit(1)
            
        logger.info(f"Found {len(fits_files)} FITS files to query.")
        
        results_list = []
        from tqdm import tqdm
        for fits_file in tqdm(fits_files, desc="Processing FITS queries"):
            row_dict = extract_and_search_fits(fits_file, model, device, norm_X, ids, ds, ds_ids, fits_indexed, save_dir, args)
            if row_dict:
                results_list.append(row_dict)
                
        # Write CSV if requested
        if args.output_csv:
            csv_path = Path(args.output_csv)
            pd.DataFrame(results_list).to_csv(csv_path, index=False)
            logger.info(f"Saved similarity search results to: {csv_path}")
            
        sys.exit(0)
        
    # Find query index
    query_id = args.query_id
    matches = np.where(ids == query_id)[0]
    if len(matches) == 0:
        logger.error(f"Query TargetID {query_id} not found in the latent database!")
        # Suggest some IDs
        logger.info(f"Available sample IDs in dataset: {list(ids[:10])}...")
        sys.exit(1)
    query_idx = int(matches[0])
    logger.info(f"Resolved Query TargetID {query_id} at index {query_idx}")
    
    # Compute similarity for Image, Spectrum, and Joint spaces
    spaces = {}
    filename_mappings = {
        "EuclidImage": ["EuclidImage_pooled", "EuclidImage_phase2", "EuclidImage_phase1", "EuclidImage", "images"],
        "DESISpectrum": ["DESISpectrum_pooled", "DESISpectrum_phase2", "DESISpectrum_phase1", "DESISpectrum", "spectra"],
        "joint": ["joint_mean", "joint_concat", "joint"]
    }
    
    for space_name, fallbacks in filename_mappings.items():
        found_file = None
        for fallback in fallbacks:
            if fallback in emb_files:
                found_file = emb_files[fallback]
                break
                
        if found_file is None:
            logger.warning(f"Embedding space file for '{space_name}' not found (tried: {fallbacks})! Skipping similarity computation for it.")
            continue
            
        logger.info(f"Loading '{space_name}' space from: {found_file.name}")
        X = np.load(found_file)
        if X.ndim == 3:
            X = X.mean(axis=1) # Mean pooling sequence tokens
            
        top_ids, top_sims = compute_similarity(query_idx, X, ids, k=args.k)
        spaces[space_name] = {"ids": top_ids, "sims": top_sims}
        
        logger.info(f"\nTop {args.k} Neighbors in '{space_name}' Space:")
        for idx, (tid, sim) in enumerate(zip(top_ids, top_sims)):
            logger.info(f"  Rank #{idx+1}: TargetID={tid} | Cosine Similarity={sim:.4%}")

    # 2. Setup Dataloader and metadata
    logger.info("\nInitializing active AstroPT Datasets via checkpoint...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, registry, raw_config_dict = load_local_model(Path(args.ckpt_path), device)
    
    raw_config_dict['data_dir'] = args.data_dir
    raw_config_dict['batch_size'] = args.batch_size
    
    valid_keys = {f.name for f in fields(TrainingConfig)}
    clean_config_dict = {k: v for k, v in raw_config_dict.items() if k in valid_keys}
    training_config = TrainingConfig(**clean_config_dict)
    
    train_loader, val_loader, _ = create_dataloaders(training_config, ddp=False)
    ds = CombinedDatasetWrapper(train_loader.dataset, val_loader.dataset)
    
    # Read FITS metadata catalog
    logger.info(f"Reading FITS metadata catalog from {args.metadata_path}...")
    fits_table = Table.read(args.metadata_path)
    fits_df = fits_table.to_pandas()
    
    # Decode string bytes
    for col in fits_df.columns:
        if fits_df[col].dtype == object and isinstance(fits_df[col].iloc[0], bytes):
            try: fits_df[col] = fits_df[col].str.decode('utf-8')
            except: pass
            
    target_id_col = 'TARGETID' if 'TARGETID' in fits_df.columns else 'targetid'
    fits_indexed = fits_df.drop_duplicates(subset=[target_id_col]).set_index(target_id_col)
    
    def find_col_val(df_row, patterns):
        for pat in patterns:
            for col in df_row.index:
                if pat.lower() in col.lower():
                    val = df_row[col]
                    if val is not None and not pd.isna(val):
                        return float(val)
        return None

    def get_morphological_meta_str(tid):
        if tid in fits_indexed.index:
            meta = fits_indexed.loc[tid]
            
            # Coordinates
            ra_val = meta.get('RA', None)
            dec_val = meta.get('DEC', None)
            coords_str = f"${ra_val:.6f}^\\circ, {dec_val:.6f}^\\circ$" if (ra_val is not None and not pd.isna(ra_val)) else "N/A"
            
            # Redshift
            z_val = meta.get('Z', None)
            z_str = f"{z_val:.4f}" if (z_val is not None and not pd.isna(z_val)) else "N/A"
            
            # Stellar Mass
            logmstar_val = meta.get('LOGMSTAR', None)
            mstar = f"$10^{{{logmstar_val:.2f}}} \\text{{ M}}_\\odot$" if (logmstar_val is not None and not pd.isna(logmstar_val)) else "N/A"
            
            # Star Formation Rate
            logsfr_val = meta.get('LOGSFR', None)
            sfr = f"${10**logsfr_val:.3f} \\text{{ M}}_\\odot/\\text{{yr}}$" if (logsfr_val is not None and not pd.isna(logsfr_val)) else "N/A"
            
            # Spectype
            spectype = str(meta.get('SPECTYPE', 'N/A'))
            
            # Euclid Sersic Index
            sersic_n_val = meta.get('sersic_sersic_vis_index', None)
            n_str = f"{sersic_n_val:.2f}" if (sersic_n_val is not None and not pd.isna(sersic_n_val)) else "N/A"
            
            # Euclid Effective Radius
            sersic_re_val = meta.get('sersic_sersic_vis_radius', None)
            re_str = f"{sersic_re_val:.3f}''" if (sersic_re_val is not None and not pd.isna(sersic_re_val)) else "N/A"
            
            # Axis Ratio
            ba_val = meta.get('sersic_sersic_vis_axis_ratio', None)
            ba_str = f"{ba_val:.3f}" if (ba_val is not None and not pd.isna(ba_val)) else "N/A"
            
            # VIS Aperture Flux
            vis_flux_val = meta.get('flux_vis_1fwhm_aper', None)
            vis_flux_str = f"{vis_flux_val:.2f}" if (vis_flux_val is not None and not pd.isna(vis_flux_val)) else "N/A"
            
            meta_str = (
                f"\\textbf{{Coords}}: {coords_str}\n"
                f"\\textbf{{Z}}: {z_str}\n"
                f"\\textbf{{$M_*$}}: {mstar}\n"
                f"\\textbf{{SFR}}: {sfr}\n"
                f"\\textbf{{Spectype}}: {spectype}\n"
                f"\\textbf{{VIS Flux}}: {vis_flux_str}\n"
                f"\\textbf{{Sersic n}}: {n_str}\n"
                f"\\textbf{{Radius $R_{{eff}}$}}: {re_str}\n"
                f"\\textbf{{Axis Ratio}}: {ba_str}"
            )
            return meta_str.replace("_", "\\_"), z_val
        return f"\\textbf{{ID}}: {tid}\n(No metadata)", None

    def get_spectroscopic_meta_str(tid):
        if tid in fits_indexed.index:
            meta = fits_indexed.loc[tid]
            
            ha_f = meta.get('HALPHA_FLUX', None)
            hb_f = meta.get('HBETA_FLUX', None)
            oiii_f = meta.get('OIII_5007_FLUX', None)
            oii_f = meta.get('OII_3726_FLUX', None)
            
            ha_ew = meta.get('HALPHA_EW', None)
            hb_ew = meta.get('HBETA_EW', None)
            oii_ew = meta.get('OII_3726_EW', None)
            
            ha_sig = meta.get('HALPHA_SIGMA', None)
            snr_r = meta.get('SNR_SPEC_R', None)
            snr_z = meta.get('SNR_SPEC_Z', None)
            
            def fmt_flux(v): return f"{v:.2f}" if (v is not None and not pd.isna(v)) else "N/A"
            def fmt_ew(v): return f"{v:.2f} \\AA" if (v is not None and not pd.isna(v)) else "N/A"
            def fmt_val(v, unit=""): return f"{v:.1f}{unit}" if (v is not None and not pd.isna(v)) else "N/A"
            
            stats_text = (
                f"\\textbf{{Spectral Line Fluxes}}:\n"
                f"  - H$\\alpha$ Flux: {fmt_flux(ha_f)}\n"
                f"  - H$\\beta$ Flux: {fmt_flux(hb_f)}\n"
                f"  - [O III] 5007 Flux: {fmt_flux(oiii_f)}\n"
                f"  - [O II] 3726 Flux: {fmt_flux(oii_f)}\n\n"
                f"\\textbf{{Equivalent Widths}}:\n"
                f"  - H$\\alpha$ EW: {fmt_ew(ha_ew)}\n"
                f"  - H$\\beta$ EW: {fmt_ew(hb_ew)}\n"
                f"  - [O II] 3726 EW: {fmt_ew(oii_ew)}\n\n"
                f"\\textbf{{Spectra Quality}}:\n"
                f"  - H$\\alpha$ Width $\\sigma$: {fmt_val(ha_sig, ' km/s')}\n"
                f"  - SNR Spec R: {fmt_val(snr_r)} | SNR Spec Z: {fmt_val(snr_z)}"
            )
            return stats_text.replace("_", "\\_")
        return "\\textbf{Spectra data missing}"

    # --- Helper to retrieve metadata block ---
    def get_meta_str(tid):
        if tid in fits_indexed.index:
            meta = fits_indexed.loc[tid]
            z_val = meta.get('Z', meta.get('z', None))
            z_str = f"{z_val:.4f}" if (z_val is not None and not pd.isna(z_val)) else "N/A"
            logmstar_val = meta.get('LOGMSTAR', meta.get('logmstar', None))
            mstar = f"$10^{{{logmstar_val:.2f}}} \\text{{ M}}_{{\\odot}}$" if (logmstar_val is not None and not pd.isna(logmstar_val)) else "N/A"
            logsfr_val = meta.get('LOGSFR', meta.get('logsfr', None))
            sfr = f"${10**logsfr_val:.3f} \\text{{ M}}_{{\\odot}}/\\text{{yr}}$" if (logsfr_val is not None and not pd.isna(logsfr_val)) else "N/A"
            spectype = str(meta.get('SPECTYPE', meta.get('spectype', 'N/A')))
            
            meta_str = (
                f"\\textbf{{Z}}: {z_str}\n"
                f"\\textbf{{$M_*$}}: {mstar}\n"
                f"\\textbf{{SFR}}: {sfr}\n"
                f"\\textbf{{Spectype}}: {spectype}"
            )
            return meta_str.replace("_", "\\_"), z_val
        return f"\\textbf{{ID}}: {tid}\n(No metadata)", None

    # Retrieve query record
    ds_ids = np.array(ds.ds['targetid'])
    query_matches = np.where(ds_ids == query_id)[0]
    if len(query_matches) == 0:
        logger.error(f"Query TargetID {query_id} not found in the Arrow raw catalog!")
        sys.exit(1)
    query_record = ds.ds[int(query_matches[0])]
    query_rgb = extract_raw_rgb(query_record)
    query_meta, query_z = get_morphological_meta_str(query_id)
    query_spec_meta = get_spectroscopic_meta_str(query_id)
    
    # ---------------------------------------------------------------------------
    # Plot Generation (Premium Dashboard PDF)
    # ---------------------------------------------------------------------------
    matplotlib_use_agg = True
    import matplotlib
    if matplotlib_use_agg:
        matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages
    
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', weight='bold')
    
    pdf_path = save_dir / f"similarity_search_{query_id}.pdf"
    logger.info(f"Generating PDF Similarity Search Report at: {pdf_path}")
    
    with PdfPages(pdf_path) as pdf:
        
        # ==========================================
        # PAGE 1: QUERY GALAXY PROFILE
        # ==========================================
        fig = plt.figure(figsize=(20, 14), dpi=150)
        gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], width_ratios=[1, 1.1, 0.9], hspace=0.4, wspace=0.3)
        
        ax_meta = fig.add_subplot(gs[0, 0])
        ax_img = fig.add_subplot(gs[0, 1])
        ax_stats = fig.add_subplot(gs[0, 2])
        ax_spec_blue = fig.add_subplot(gs[1, :])
        ax_spec_red = fig.add_subplot(gs[2, :])
        
        # Meta
        ax_meta.axis('off')
        ax_meta.text(
            0.5, 0.95, clean_latex(query_meta), 
            transform=ax_meta.transAxes, 
            fontsize=14,
            linespacing=1.8,
            horizontalalignment='center',
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.8", fc="ivory", alpha=0.95, ec="darkgrey", lw=1.0)
        )
        ax_meta.set_title(clean_latex(r"\textbf{Physical and Morphological properties}"), fontsize=16, fontweight='bold', color='navy', pad=15)
        
        # Image
        if query_rgb is not None:
            ax_img.imshow(query_rgb, origin='lower')
            # Title is removed per user request
        else:
            ax_img.text(0.5, 0.5, "EuclidImage missing", ha='center', va='center')
        ax_img.axis('off')
        
        # Stats info box (Spectroscopic properties)
        ax_stats.axis('off')
        ax_stats.text(
            0.5, 0.95, clean_latex(query_spec_meta), 
            transform=ax_stats.transAxes, 
            fontsize=14,
            linespacing=1.6,
            horizontalalignment='center',
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.8", fc="aliceblue", alpha=0.95, ec="steelblue", lw=1.0)
        )
        ax_stats.set_title(clean_latex(r"\textbf{Spectroscopic properties}"), fontsize=16, fontweight='bold', color='steelblue', pad=15)
        
        # Spectrum
        plot_spectrum_into_axes(ax_spec_blue, ax_spec_red, query_record, query_id, query_z)
        ax_spec_blue.set_title(clean_latex(rf"\textbf{{Query Spectrum (Blue Channel)}}"), fontsize=11, fontweight='bold')
        
        plt.suptitle(clean_latex(rf"\textbf{{AstroPT similarity search: ID: {query_id}}}"), fontsize=22, fontweight='bold', y=0.98)
        pdf.savefig()
        plt.close()
        
        # ==========================================
        # PAGE 2: EUCLID MORPHOLOGICAL NEIGHBORS (SUMMARY GRID)
        # ==========================================
        if "EuclidImage" in spaces:
            logger.info("Plotting Euclid Morphological Neighbors summary grid...")
            fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=150)
            fig.subplots_adjust(hspace=0.3, wspace=0.2)
            
            # Query Image (Top Left)
            ax_q = axes[0, 0]
            if query_rgb is not None:
                ax_q.imshow(query_rgb, origin='lower')
            # Title is completely removed per user request
            ax_q.axis('off')
            ax_q.patch.set_edgecolor('navy')
            ax_q.patch.set_linewidth(3)
            
            for idx, (tid, sim) in enumerate(zip(spaces["EuclidImage"]["ids"][:5], spaces["EuclidImage"]["sims"][:5])):
                row_ax = (idx + 1) // 3
                col_ax = (idx + 1) % 3
                ax_n = axes[row_ax, col_ax]
                
                n_matches = np.where(ds_ids == tid)[0]
                n_rgb = None
                if len(n_matches) > 0:
                    n_rec = ds.ds[int(n_matches[0])]
                    n_rgb = extract_raw_rgb(n_rec)
                    
                if n_rgb is not None:
                    ax_n.imshow(n_rgb, origin='lower')
                else:
                    ax_n.text(0.5, 0.5, f"Image not found | ID: {tid}", ha='center', va='center')
                # Title is completely removed per user request
                ax_n.axis('off')
                
            plt.suptitle(clean_latex(r"\textbf{Euclid Morphological Neighbors}"), fontsize=22, fontweight='bold', y=0.98)
            pdf.savefig()
            plt.close()
            
            # ==========================================
            # PAGES 3-7: DETAILED PROFILE FOR EACH NEIGHBOR
            # ==========================================
            for idx, (tid, sim) in enumerate(zip(spaces["EuclidImage"]["ids"], spaces["EuclidImage"]["sims"])):
                logger.info(f"Plotting detailed profile page for Euclid Morphological Rank #{idx+1} (ID: {tid})...")
                
                n_matches = np.where(ds_ids == tid)[0]
                if len(n_matches) == 0:
                    logger.warning(f"Neighbor TargetID {tid} not found in Arrow datasets, skipping detailed profile.")
                    continue
                    
                n_rec = ds.ds[int(n_matches[0])]
                n_rgb = extract_raw_rgb(n_rec)
                n_meta, n_z = get_morphological_meta_str(tid)
                n_spec_meta = get_spectroscopic_meta_str(tid)
                
                # Render Page
                fig = plt.figure(figsize=(20, 14), dpi=150)
                gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], width_ratios=[1, 1.1, 0.9], hspace=0.4, wspace=0.3)
                
                ax_meta = fig.add_subplot(gs[0, 0])
                ax_img = fig.add_subplot(gs[0, 1])
                ax_stats = fig.add_subplot(gs[0, 2])
                ax_spec_blue = fig.add_subplot(gs[1, :])
                ax_spec_red = fig.add_subplot(gs[2, :])
                
                # Meta box
                ax_meta.axis('off')
                ax_meta.text(
                    0.5, 0.95, clean_latex(n_meta), 
                    transform=ax_meta.transAxes, 
                    fontsize=14,
                    linespacing=1.8,
                    horizontalalignment='center',
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.8", fc="ivory", alpha=0.95, ec="darkgrey", lw=1.0)
                )
                ax_meta.set_title(clean_latex(r"\textbf{Physical and Morphological properties}"), fontsize=16, fontweight='bold', color='navy', pad=15)
                
                # Image
                if n_rgb is not None:
                    ax_img.imshow(n_rgb, origin='lower')
                    # No title above the neighbor image per user request
                else:
                    ax_img.text(0.5, 0.5, "EuclidImage missing", ha='center', va='center')
                ax_img.axis('off')
                
                # Stats box (Spectroscopic properties)
                ax_stats.axis('off')
                ax_stats.text(
                    0.5, 0.95, clean_latex(n_spec_meta), 
                    transform=ax_stats.transAxes, 
                    fontsize=14,
                    linespacing=1.6,
                    horizontalalignment='center',
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.8", fc="aliceblue", alpha=0.95, ec="steelblue", lw=1.0)
                )
                ax_stats.set_title(clean_latex(r"\textbf{Spectroscopic properties}"), fontsize=16, fontweight='bold', color='steelblue', pad=15)
                
                # Spectrum
                plot_spectrum_into_axes(ax_spec_blue, ax_spec_red, n_rec, tid, n_z)
                ax_spec_blue.set_title(clean_latex(rf"\textbf{{Spectrum (Blue Channel)}}"), fontsize=11, fontweight='bold')
                
                # Page super title
                plt.suptitle(clean_latex(rf"\textbf{{Euclid Morphological Rank \#{idx+1} ({sim*100:.2f}\%): ID: {tid}}}"), fontsize=22, fontweight='bold', y=0.98)
                pdf.savefig()
                plt.close()
                
        # ==========================================
        # PAGE 8: DESI SPECTROSCOPIC NEIGHBORS (SUMMARY GRID/STACK)
        # ==========================================
        if "DESISpectrum" in spaces:
            logger.info("Plotting DESI Spectroscopic Neighbors summary grid...")
            fig, axes = plt.subplots(6, 1, figsize=(20, 16), dpi=150, sharex=True)
            fig.subplots_adjust(hspace=0.55, left=0.08, right=0.95, top=0.92, bottom=0.06)
            
            # Subplot 0: Query Spectrum
            ax_q = axes[0]
            if query_record.get('spectrum_flux') is not None:
                spec_gt = np.array(query_record['spectrum_flux']).flatten()
                wave_ang = np.array(query_record['spectrum_wave']).flatten()
                ax_q.plot(wave_ang, spec_gt, 'r-', lw=1, alpha=0.8)
                ax_q.set_xlim(wave_ang.min(), wave_ang.max())
                ax_q.set_title(clean_latex(rf"\textbf{{Query (ID: {query_id})}}"), fontsize=11, fontweight='bold')
                if query_z is not None:
                    for name, rest_wave in MAIN_LINES.items():
                        obs_wave = rest_wave * (1 + query_z)
                        if wave_ang.min() < obs_wave < wave_ang.max():
                            ax_q.axvline(obs_wave, color='royalblue', linestyle='--', alpha=0.3, lw=0.7)
            else:
                ax_q.text(0.5, 0.5, "Query Spectrum not found", ha='center', va='center', fontsize=12)
            
            ax_q.set_ylabel(r"Flux", fontsize=9)
            
            # Subplots 1-5: Spectroscopic Neighbors
            for idx, (tid, sim) in enumerate(zip(spaces["DESISpectrum"]["ids"][:5], spaces["DESISpectrum"]["sims"][:5])):
                ax_n = axes[idx + 1]
                n_matches = np.where(ds_ids == tid)[0]
                
                n_rec = None
                n_z = None
                if len(n_matches) > 0:
                    n_rec = ds.ds[int(n_matches[0])]
                    if tid in fits_indexed.index:
                        n_z = fits_indexed.loc[tid].get('Z', None)
                
                if n_rec is not None and n_rec.get('spectrum_flux') is not None:
                    spec_gt = np.array(n_rec['spectrum_flux']).flatten()
                    wave_ang = np.array(n_rec['spectrum_wave']).flatten()
                    ax_n.plot(wave_ang, spec_gt, 'k-', lw=1, alpha=0.8)
                    ax_n.set_xlim(wave_ang.min(), wave_ang.max())
                    ax_n.set_title(clean_latex(rf"\textbf{{Rank \#{idx+1} ({sim*100:.2f}\%) | ID: {tid}}}"), fontsize=11, fontweight='bold')
                    if n_z is not None and not pd.isna(n_z):
                        for name, rest_wave in MAIN_LINES.items():
                            obs_wave = rest_wave * (1 + n_z)
                            if wave_ang.min() < obs_wave < wave_ang.max():
                                ax_n.axvline(obs_wave, color='royalblue', linestyle='--', alpha=0.3, lw=0.7)
                else:
                    ax_n.text(0.5, 0.5, f"Spectrum not found | ID: {tid}", ha='center', va='center', fontsize=12)
                
                ax_n.set_ylabel(r"Flux", fontsize=9)
            
            axes[-1].set_xlabel(clean_latex(r"Observed Wavelength [\AA]"), fontsize=12, fontweight='bold')
            
            plt.suptitle(clean_latex(r"\textbf{DESI Spectroscopic Neighbors}"), fontsize=22, fontweight='bold', y=0.98)
            pdf.savefig()
            plt.close()
            
            # ==========================================
            # PAGES 9-13: DETAILED PROFILE FOR EACH SPECTROSCOPIC NEIGHBOR
            # ==========================================
            for idx, (tid, sim) in enumerate(zip(spaces["DESISpectrum"]["ids"], spaces["DESISpectrum"]["sims"])):
                logger.info(f"Plotting detailed profile page for DESI Spectroscopic Rank #{idx+1} (ID: {tid})...")
                
                n_matches = np.where(ds_ids == tid)[0]
                if len(n_matches) == 0:
                    logger.warning(f"Neighbor TargetID {tid} not found in Arrow datasets, skipping detailed profile.")
                    continue
                    
                n_rec = ds.ds[int(n_matches[0])]
                n_rgb = extract_raw_rgb(n_rec)
                n_meta, n_z = get_morphological_meta_str(tid)
                n_spec_meta = get_spectroscopic_meta_str(tid)
                
                # Render Page
                fig = plt.figure(figsize=(20, 14), dpi=150)
                gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], width_ratios=[1, 1.1, 0.9], hspace=0.4, wspace=0.3)
                
                ax_meta = fig.add_subplot(gs[0, 0])
                ax_img = fig.add_subplot(gs[0, 1])
                ax_stats = fig.add_subplot(gs[0, 2])
                ax_spec_blue = fig.add_subplot(gs[1, :])
                ax_spec_red = fig.add_subplot(gs[2, :])
                
                # Meta box
                ax_meta.axis('off')
                ax_meta.text(
                    0.5, 0.95, clean_latex(n_meta), 
                    transform=ax_meta.transAxes, 
                    fontsize=14,
                    linespacing=1.8,
                    horizontalalignment='center',
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.8", fc="ivory", alpha=0.95, ec="darkgrey", lw=1.0)
                )
                ax_meta.set_title(clean_latex(r"\textbf{Physical and Morphological properties}"), fontsize=16, fontweight='bold', color='navy', pad=15)
                
                # Image
                if n_rgb is not None:
                    ax_img.imshow(n_rgb, origin='lower')
                else:
                    ax_img.text(0.5, 0.5, "EuclidImage missing", ha='center', va='center')
                ax_img.axis('off')
                
                # Stats box (Spectroscopic properties)
                ax_stats.axis('off')
                ax_stats.text(
                    0.5, 0.95, clean_latex(n_spec_meta), 
                    transform=ax_stats.transAxes, 
                    fontsize=14,
                    linespacing=1.6,
                    horizontalalignment='center',
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.8", fc="aliceblue", alpha=0.95, ec="steelblue", lw=1.0)
                )
                ax_stats.set_title(clean_latex(r"\textbf{Spectroscopic properties}"), fontsize=16, fontweight='bold', color='steelblue', pad=15)
                
                # Spectrum
                plot_spectrum_into_axes(ax_spec_blue, ax_spec_red, n_rec, tid, n_z)
                ax_spec_blue.set_title(clean_latex(rf"\textbf{{Spectrum (Blue Channel)}}"), fontsize=11, fontweight='bold')
                
                # Page super title
                plt.suptitle(clean_latex(rf"\textbf{{DESI Spectroscopic Rank \#{idx+1} ({sim*100:.2f}\%): ID: {tid}}}"), fontsize=22, fontweight='bold', y=0.98)
                pdf.savefig()
                plt.close()
                
        # ==========================================
        # PAGES 14-18: DETAILED PROFILE FOR EACH JOINT NEIGHBOR
        # ==========================================
        if "joint" in spaces:
            for idx, (tid, sim) in enumerate(zip(spaces["joint"]["ids"], spaces["joint"]["sims"])):
                logger.info(f"Plotting detailed profile page for Joint Rank #{idx+1} (ID: {tid})...")
                
                n_matches = np.where(ds_ids == tid)[0]
                if len(n_matches) == 0:
                    logger.warning(f"Neighbor TargetID {tid} not found in Arrow datasets, skipping detailed profile.")
                    continue
                    
                n_rec = ds.ds[int(n_matches[0])]
                n_rgb = extract_raw_rgb(n_rec)
                n_meta, n_z = get_morphological_meta_str(tid)
                n_spec_meta = get_spectroscopic_meta_str(tid)
                
                # Render Page
                fig = plt.figure(figsize=(20, 14), dpi=150)
                gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], width_ratios=[1, 1.1, 0.9], hspace=0.4, wspace=0.3)
                
                ax_meta = fig.add_subplot(gs[0, 0])
                ax_img = fig.add_subplot(gs[0, 1])
                ax_stats = fig.add_subplot(gs[0, 2])
                ax_spec_blue = fig.add_subplot(gs[1, :])
                ax_spec_red = fig.add_subplot(gs[2, :])
                
                # Meta box
                ax_meta.axis('off')
                ax_meta.text(
                    0.5, 0.95, clean_latex(n_meta), 
                    transform=ax_meta.transAxes, 
                    fontsize=14,
                    linespacing=1.8,
                    horizontalalignment='center',
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.8", fc="ivory", alpha=0.95, ec="darkgrey", lw=1.0)
                )
                ax_meta.set_title(clean_latex(r"\textbf{Physical and Morphological properties}"), fontsize=16, fontweight='bold', color='navy', pad=15)
                
                # Image
                if n_rgb is not None:
                    ax_img.imshow(n_rgb, origin='lower')
                else:
                    ax_img.text(0.5, 0.5, "EuclidImage missing", ha='center', va='center')
                ax_img.axis('off')
                
                # Stats box (Spectroscopic properties)
                ax_stats.axis('off')
                ax_stats.text(
                    0.5, 0.95, clean_latex(n_spec_meta), 
                    transform=ax_stats.transAxes, 
                    fontsize=14,
                    linespacing=1.6,
                    horizontalalignment='center',
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.8", fc="aliceblue", alpha=0.95, ec="steelblue", lw=1.0)
                )
                ax_stats.set_title(clean_latex(r"\textbf{Spectroscopic properties}"), fontsize=16, fontweight='bold', color='steelblue', pad=15)
                
                # Spectrum
                plot_spectrum_into_axes(ax_spec_blue, ax_spec_red, n_rec, tid, n_z)
                ax_spec_blue.set_title(clean_latex(rf"\textbf{{Spectrum (Blue Channel)}}"), fontsize=11, fontweight='bold')
                
                # Page super title
                plt.suptitle(clean_latex(rf"\textbf{{Joint Rank \#{idx+1} ({sim*100:.2f}\%): ID: {tid}}}"), fontsize=22, fontweight='bold', y=0.98)
                pdf.savefig()
                plt.close()

                
    logger.info(f"Done. Multimodal similarity search complete. Saved dashboard PDF to {pdf_path}")


if __name__ == "__main__":
    main()
