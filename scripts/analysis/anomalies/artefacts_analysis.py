"""
AstroPT Embedding-Based Dataset Artifact Detection Tool.

This script uses pre-computed AstroPT latent embeddings to identify observational
artifacts (sensor failures, satellite trails, coordinate misalignments, corrupt data).

Strategy:
1. Compute the Unified Weirdness Index (UWI) using embedding-space anomaly detectors.
2. For top-N UWI outliers, compute a Pixel Artifact Score (PAS) from raw image data
   to distinguish real rare galaxies from instrumental artifacts.
3. Classify outliers using combined UWI + PAS + anchor matching.
4. Output a catalog of flagged artifacts and a PDF dashboard for visual validation.

The PAS uses pixel-level indicators (dead/hot pixels, inter-channel correlation,
background asymmetry, zero/NaN fraction, edge sharpness) that directly measure
data quality issues invisible to embedding-space methods.

Author: Victor Alonso Rodriguez
Date: June 2026
"""

import argparse
import sys
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import fields

import numpy as np
import pandas as pd
from astropy.table import Table
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.cluster import DBSCAN

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
logger = logging.getLogger("AstroPT-ArtifactDetector")

def find_id_index(ids_array: np.ndarray, target_id: float) -> Optional[int]:
    """Finds the index of a TargetID in the IDs array, handling float64 precision issues."""
    # Try exact match first
    matches = np.where(ids_array.astype(int) == int(target_id))[0]
    if len(matches) > 0:
        return int(matches[0])
    
    # Try float64 distance matching
    diffs = np.abs(ids_array.astype(np.float64) - float(target_id))
    min_idx = np.argmin(diffs)
    if diffs[min_idx] < 16.0:
        return int(min_idx)
        
    return None

def find_arrow_index(ds_ids: np.ndarray, target_id: float) -> Optional[int]:
    """Finds the index of a TargetID in the Arrow dataset, handling float64 rounding in target_id."""
    # Try exact match
    matches = np.where(ds_ids == int(target_id))[0]
    if len(matches) > 0:
        return int(matches[0])
    
    # Try float64 distance matching
    diffs = np.abs(ds_ids.astype(np.float64) - float(target_id))
    min_idx = np.argmin(diffs)
    if diffs[min_idx] < 16.0:
        return int(min_idx)
        
    return None

MAIN_LINES = {
    r"Ly$\alpha$": 1216.0, r"C IV": 1549.0, "C III": 1908.7, r"Mg II": 2798.0, r"[O II]": 3727.3, r"[Ne III]": 3868.7,
    r"Ca K": 3933.7, r"Ca H": 3968.5, r"H$\delta$": 4102.0, r"H$\gamma$": 4341.0,
    r"H$\beta$": 4861.0, r"[O III]": 4959.0, r"[O III]": 5007.0, r"Mg I": 5175.0,
    r"Na D": 5890.0, r"[N II]": 6548.0, r"H$\alpha$": 6563.0, r"[N II]": 6583.5, r"[S II]": 6730.8
}

def plot_spectral_lines(ax, min_wl, max_wl, z):
    """Annotates spectral lines with alternating heights."""
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
                    color='royalblue', va='top', ha='right', fontsize=12, alpha=1, fontweight='bold')
            counter += 1

def get_morphological_meta_str(tid, fits_indexed) -> Tuple[str, Optional[float]]:
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
        
        NL = "\n"
        meta_str = (
            f"\\textbf{{Coords}}: {coords_str}{NL}"
            f"\\textbf{{Z}}: {z_str}{NL}"
            f"\\textbf{{$M_*$}}: {mstar}{NL}"
            f"\\textbf{{SFR}}: {sfr}{NL}"
            f"\\textbf{{Spectype}}: {spectype}{NL}"
            f"\\textbf{{VIS Flux}}: {vis_flux_str}{NL}"
            f"\\textbf{{Sersic n}}: {n_str}{NL}"
            f"\\textbf{{Radius $R_{{eff}}$}}: {re_str}{NL}"
            f"\\textbf{{Axis Ratio}}: {ba_str}"
        )
        return meta_str.replace("_", "\\_"), z_val
    NL = "\n"
    return f"\\textbf{{ID}}: {tid}{NL}(No metadata)", None

def get_spectroscopic_meta_str(tid, fits_indexed) -> str:
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
        
        NL = "\n"
        stats_text = (
            f"\\textbf{{Spectral Line Fluxes}}:{NL}"
            f"  - H$\\alpha$ Flux: {fmt_flux(ha_f)}{NL}"
            f"  - H$\\beta$ Flux: {fmt_flux(hb_f)}{NL}"
            f"  - [O III] 5007 Flux: {fmt_flux(oiii_f)}{NL}"
            f"  - [O II] 3726 Flux: {fmt_flux(oii_f)}{NL}{NL}"
            f"\\textbf{{Equivalent Widths}}:{NL}"
            f"  - H$\\alpha$ EW: {fmt_ew(ha_ew)}{NL}"
            f"  - H$\\beta$ EW: {fmt_ew(hb_ew)}{NL}"
            f"  - [O II] 3726 EW: {fmt_ew(oii_ew)}{NL}{NL}"
            f"\\textbf{{Spectra Quality}}:{NL}"
            f"  - H$\\alpha$ Width $\\sigma$: {fmt_val(ha_sig, ' km/s')}{NL}"
            f"  - SNR Spec R: {fmt_val(snr_r)} | SNR Spec Z: {fmt_val(snr_z)}"
        )
        return stats_text.replace("_", "\\_")
    return "\\textbf{Spectra data missing}"

def parse_args():
    parser = argparse.ArgumentParser(description="AstroPT Embedding-Based Artifact Detector")
    parser.add_argument("--embeddings_dir", type=str, required=True,
                        help="Directory containing pre-computed embeddings (.npy)")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to the training checkpoint (.pt)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root folder of raw Arrow processed data")
    parser.add_argument("--metadata_path", type=str, required=True,
                        help="Path to FITS metadata catalog")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Saving directory for catalogs and plots")

    parser.add_argument("--n_candidates", type=int, default=1000,
                        help="Number of top UWI outliers to audit with pixel-level scoring")
    parser.add_argument("--n_plot", type=int, default=30,
                        help="Number of top artifacts to plot in the PDF dashboard")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for dataloader")
    parser.add_argument("--base_modality", type=str, default="EuclidImage",
                        help="Base embedding modality for outlier detection")
    parser.add_argument("--contamination", type=float, default=0.01,
                        help="Expected fraction of outliers for IF/LOF algorithms")
    parser.add_argument("--knn_neighbors", type=int, default=20,
                        help="Neighbors count for KNN and LOF models")
    parser.add_argument("--anchor_ids", type=int, nargs="+", default=None,
                        help="Optional TargetIDs of reference anchors for few-shot cosine similarity detection")
    parser.add_argument("--similarity_threshold", type=float, default=0.88,
                        help="Cosine similarity threshold for matching reference anchors")
    
    # PAS weight args for manual tuning
    parser.add_argument("--pas_w_dead_hot", type=float, default=0.20,
                        help="PAS weight for dead/hot pixel fraction")
    parser.add_argument("--pas_w_channel_corr", type=float, default=0.30,
                        help="PAS weight for inter-channel decorrelation (strongest artifact discriminator)")
    parser.add_argument("--pas_w_bg_asymmetry", type=float, default=0.20,
                        help="PAS weight for background asymmetry")
    parser.add_argument("--pas_w_zero_nan", type=float, default=0.15,
                        help="PAS weight for zero/NaN fraction")
    parser.add_argument("--pas_w_edge", type=float, default=0.15,
                        help="PAS weight for edge sharpness (Sobel)")
    parser.add_argument("--pas_threshold", type=float, default=0.40,
                        help="PAS threshold to flag as artifact (0-1). Higher = stricter")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pixel Artifact Score (PAS) — pixel-level corruption indicators
# ---------------------------------------------------------------------------

def _extract_raw_channels(raw_record) -> Optional[np.ndarray]:
    """Extract raw VIS/Y/J/H channels as (4, H, W) array from Arrow record."""
    try:
        def get_ch(k):
            val = raw_record.get(k)
            return np.array(val if val is not None else [], dtype=np.float32)

        vis = get_ch('image_vis')
        if vis.size == 0:
            return None
            
        y = get_ch('image_nisp_y')
        j = get_ch('image_nisp_j')
        h = get_ch('image_nisp_h')
        
        h = h if h.size > 0 else np.zeros_like(vis)
        j = j if j.size > 0 else np.zeros_like(vis)
        y = y if y.size > 0 else np.zeros_like(vis)
        
        return np.stack([vis, h, j, y], axis=0)  # (4, H, W)
    except Exception:
        return None


def compute_dead_hot_fraction(channels: np.ndarray) -> float:
    """Fraction of pixels that are dead (<=0) or isolated hot (>local_median + 10*local_mad).
    
    Uses per-channel local sigma clipping rather than a global percentile, so that
    galaxy clusters (many extended bright sources) are NOT confused with hot pixels.
    A genuine hot pixel is an isolated point far above the LOCAL background noise,
    not just any bright astronomical source.
    """
    total = channels.size
    n_dead = 0
    n_hot = 0
    
    for ch in channels:  # iterate each band separately
        flat = ch.flatten()
        # Dead: strictly zero or negative (sensor not reading)
        n_dead += int(np.sum(flat <= 0))
        
        # Hot: isolated points > median + 10 * MAD (local noise estimate)
        positive = flat[flat > 0]
        if positive.size < 50:
            continue
        med = np.median(positive)
        mad = np.median(np.abs(positive - med))
        if mad < 1e-12:  # uniform field, no isolated hot pixels possible
            continue
        hot_threshold = med + 10.0 * mad  # ~10-sigma above local background
        n_hot += int(np.sum(flat > hot_threshold))
    
    return float(n_dead + n_hot) / max(total, 1)


def compute_inter_channel_correlation(channels: np.ndarray) -> float:
    """Mean Pearson correlation between channel pairs. Real sources have high correlation.
    Returns value in [0, 1]. Low correlation suggests single-channel artifact."""
    n_ch = channels.shape[0]
    if n_ch < 2:
        return 1.0
    
    # Flatten each channel
    flat = channels.reshape(n_ch, -1)
    
    # Count valid channels (not all-zero)
    valid_mask = np.std(flat, axis=1) > 1e-10
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) < 2:
        return 0.0  # Can't compute meaningful correlation
    
    corrs = []
    for i in range(len(valid_indices)):
        for j in range(i + 1, len(valid_indices)):
            ci = valid_indices[i]
            cj = valid_indices[j]
            r = np.corrcoef(flat[ci], flat[cj])[0, 1]
            if not np.isnan(r):
                corrs.append(abs(r))
    
    return float(np.mean(corrs)) if corrs else 0.0


def compute_background_asymmetry(channels: np.ndarray) -> float:
    """Background RMS asymmetry across quadrants. High = sensor gradient / misalignment."""
    # Use VIS channel (index 0) as primary
    img = channels[0]
    h, w = img.shape
    mh, mw = h // 2, w // 2
    
    quadrants = [
        img[:mh, :mw],   # top-left
        img[:mh, mw:],   # top-right
        img[mh:, :mw],   # bottom-left
        img[mh:, mw:],   # bottom-right
    ]
    
    # Compute background RMS in each quadrant (using median-subtracted values)
    rms_values = []
    for q in quadrants:
        bg = q - np.median(q)
        rms_values.append(np.sqrt(np.mean(bg**2)))
    
    rms_values = np.array(rms_values)
    mean_rms = np.mean(rms_values)
    
    if mean_rms < 1e-10:
        return 0.0
    
    # Coefficient of variation of the quadrant RMS values
    return float(np.std(rms_values) / mean_rms)


def compute_zero_nan_fraction(channels: np.ndarray) -> float:
    """Fraction of values that are exactly 0 or NaN."""
    total = channels.size
    n_zero = np.sum(channels == 0)
    n_nan = np.sum(np.isnan(channels))
    return float(n_zero + n_nan) / total


def compute_edge_sharpness(channels: np.ndarray) -> float:
    """Normalized max Sobel gradient. Satellite trails and sensor boundaries cause sharp edges."""
    # Use VIS channel
    img = channels[0].astype(np.float64)
    
    # Simple Sobel-like gradient (avoid scipy dependency)
    # Horizontal gradient
    gx = np.abs(np.diff(img, axis=1))
    # Vertical gradient  
    gy = np.abs(np.diff(img, axis=0))
    
    # P99.5 of gradient magnitude (robust to a few bright pixels)
    gx_p99 = np.percentile(gx, 99.5) if gx.size > 0 else 0
    gy_p99 = np.percentile(gy, 99.5) if gy.size > 0 else 0
    max_grad = max(gx_p99, gy_p99)
    
    # Normalize by the image's own dynamic range
    img_range = np.percentile(img, 99.5) - np.percentile(img, 0.5)
    if img_range < 1e-10:
        return 0.0
    
    return float(min(max_grad / img_range, 1.0))


def compute_pixel_artifact_score(
    raw_record,
    w_dead_hot: float = 0.20,
    w_channel_corr: float = 0.25,
    w_bg_asymmetry: float = 0.20,
    w_zero_nan: float = 0.20,
    w_edge: float = 0.15,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the Pixel Artifact Score (PAS) for a single raw Arrow record.
    
    Returns:
        pas: Combined score in [0, 1]. Higher = more likely artifact.
        components: Dict of individual metric values.
    """
    channels = _extract_raw_channels(raw_record)
    if channels is None:
        return 0.0, {"dead_hot": 0, "channel_corr": 0, "bg_asymmetry": 0, "zero_nan": 0, "edge_sharp": 0}
    
    dead_hot = compute_dead_hot_fraction(channels)
    channel_corr = compute_inter_channel_correlation(channels)
    bg_asym = compute_background_asymmetry(channels)
    zero_nan = compute_zero_nan_fraction(channels)
    edge_sharp = compute_edge_sharpness(channels)
    
    # Combine: channel_corr inverted (low correlation = suspicious)
    # Clamp bg_asym and edge_sharp to [0, 1]
    bg_asym_norm = min(bg_asym, 1.0)
    
    pas = (
        w_dead_hot * dead_hot +
        w_channel_corr * (1.0 - channel_corr) +
        w_bg_asymmetry * bg_asym_norm +
        w_zero_nan * zero_nan +
        w_edge * edge_sharp
    )
    
    components = {
        "dead_hot": round(dead_hot, 4),
        "channel_corr": round(channel_corr, 4),
        "bg_asymmetry": round(bg_asym_norm, 4),
        "zero_nan": round(zero_nan, 4),
        "edge_sharp": round(edge_sharp, 4),
    }
    
    return float(min(pas, 1.0)), components


# ---------------------------------------------------------------------------
# Anomaly Scoring Engine (Lightweight, embedding-only)
# ---------------------------------------------------------------------------

def compute_uwi_scores(
    embeddings_dir: Path,
    base_modality: str,
    contamination: float,
    knn_neighbors: int,
    allowed_ids_filter: set = None,
) -> pd.DataFrame:
    """Compute Unified Weirdness Index using IsolationForest, LOF, SVDD, Mahalanobis, KNN."""
    logger.info("Loading pre-computed embeddings...")

    ids = np.load(embeddings_dir / "ids.npy")

    keep_mask = None
    if allowed_ids_filter is not None:
        keep_mask = np.isin(ids.astype(int), list(allowed_ids_filter))
        num_kept = np.sum(keep_mask)
        logger.info(f"Filtering embeddings: keeping {num_kept} / {len(ids)} sources.")
        if num_kept == 0:
            logger.warning("Zero sources match filters! Disabling filter.")
            keep_mask = None

    if keep_mask is not None:
        ids = ids[keep_mask]

    df_scores = pd.DataFrame({"TargetID": ids})

    X_base = np.load(embeddings_dir / f"{base_modality}.npy")
    if keep_mask is not None:
        X_base = X_base[keep_mask]
    if X_base.ndim == 3:
        X_base = X_base.mean(axis=1)

    # --- Isolation Forest ---
    logger.info("Running Isolation Forest...")
    clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    clf.fit(X_base)
    df_scores["score_iforest"] = -clf.decision_function(X_base)

    # --- Local Outlier Factor ---
    logger.info(f"Running Local Outlier Factor (k={knn_neighbors})...")
    clf = LocalOutlierFactor(n_neighbors=knn_neighbors, contamination=contamination, novelty=True, n_jobs=-1)
    clf.fit(X_base)
    df_scores["score_lof"] = -clf.decision_function(X_base)

    # --- Deep SVDD / Centroid Distance ---
    logger.info("Running Deep SVDD Hypersphere distance...")
    c = np.median(X_base, axis=0)
    df_scores["score_svdd"] = np.linalg.norm(X_base - c, axis=1)**2

    # --- Mahalanobis Distance ---
    logger.info("Running Mahalanobis Distance (Ledoit-Wolf)...")
    try:
        cov = LedoitWolf().fit(X_base)
        df_scores["score_mahalanobis"] = cov.mahalanobis(X_base)
    except Exception as e:
        logger.error(f"Mahalanobis failed: {e}")

    # --- KNN Distance ---
    logger.info(f"Running KNN Outlier detector (k={knn_neighbors})...")
    neigh = NearestNeighbors(n_neighbors=knn_neighbors, n_jobs=-1)
    neigh.fit(X_base)
    distances, _ = neigh.kneighbors(X_base)
    df_scores["score_knn"] = distances.mean(axis=1)

    # --- Unified Weirdness Index ---
    score_cols = [c for c in df_scores.columns if c.startswith("score_")]
    logger.info("Standardizing scores and calculating Unified Weirdness Index (UWI)...")
    for col in score_cols:
        raw_vals = df_scores[col].values
        ranks = np.argsort(np.argsort(raw_vals)) / (len(raw_vals) - 1.0)
        df_scores[f"percentile_{col.replace('score_', '')}"] = ranks

    percentile_cols = [c for c in df_scores.columns if c.startswith("percentile_")]
    df_scores["Unified_Weirdness_Index"] = df_scores[percentile_cols].mean(axis=1)

    df_scores = df_scores.sort_values(by="Unified_Weirdness_Index", ascending=False).reset_index(drop=True)

    return df_scores


# ---------------------------------------------------------------------------
# RGB Extraction Helper
# ---------------------------------------------------------------------------

def make_rgb_lupton(image_tensor: np.ndarray, Q: float = 12.0, stretch: float = 0.5) -> np.ndarray:
    """Lupton et al. (2004) algorithm implementation for false-color RGB."""
    I = np.mean(image_tensor, axis=0)
    I = np.maximum(I, 1e-10)
    f_I = np.arcsinh(Q * stretch * I) / Q
    scale_factor = f_I / I
    rgb_out = image_tensor * scale_factor[np.newaxis, :, :]
    max_rgb = np.percentile(rgb_out, 99.5)
    if max_rgb > 0:
        rgb_out = rgb_out / max_rgb
    rgb_out = np.clip(rgb_out, 0, 1)
    return rgb_out.transpose(1, 2, 0)

def extract_raw_rgb(raw_record) -> np.ndarray:
    """Safely extracts channels and returns false-color RGB."""
    try:
        def get_ch(k):
            val = raw_record.get(k)
            return np.array(val if val is not None else [], dtype=np.float32)

        vis = get_ch('image_vis')
        y = get_ch('image_nisp_y')
        j = get_ch('image_nisp_j')
        h = get_ch('image_nisp_h')

        if vis.size > 0:
            h = h if h.size > 0 else np.zeros_like(vis)
            j = j if j.size > 0 else np.zeros_like(vis)
            y = y if y.size > 0 else np.zeros_like(vis)

            raw_stack = np.stack([vis, h, j, y], axis=0)
            bg_val = np.percentile(raw_stack, 50, axis=(1,2), keepdims=True)
            raw_bg = raw_stack - bg_val

            raw_rgb_stack = []
            for c in range(raw_bg.shape[0]):
                v_max = np.percentile(np.abs(raw_bg[c]), 99.5)
                if v_max <= 0: v_max = 1.0
                raw_rgb_stack.append(np.clip(raw_bg[c] / v_max, 0, 100))
            raw_norm = np.stack(raw_rgb_stack)

            RGB_WEIGHTS = [1.2, 1.3, 1.0]
            r = raw_norm[1] * RGB_WEIGHTS[0]
            g = ((raw_norm[2] + raw_norm[3]) / 2.0) * RGB_WEIGHTS[1]
            b = raw_norm[0] * RGB_WEIGHTS[2]

            return make_rgb_lupton(np.stack([r, g, b], axis=0), Q=12.0, stretch=0.5)
    except Exception as e:
        logger.error(f"Error extracting RGB image: {e}")
    return None


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    save_dir = Path(args.output_dir) if args.output_dir else embeddings_dir / "anomalies"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load checkpoint config to reconstruct dataloader
    logger.info("Initializing active AstroPT Datasets via config checkpoint...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, registry, raw_config_dict = load_local_model(Path(args.ckpt_path), device)

    raw_config_dict['data_dir'] = args.data_dir
    raw_config_dict['metadata_path'] = args.metadata_path
    raw_config_dict['batch_size'] = args.batch_size

    valid_keys = {f.name for f in fields(TrainingConfig)}
    clean_config_dict = {k: v for k, v in raw_config_dict.items() if k in valid_keys}
    training_config = TrainingConfig(**clean_config_dict)

    _, loader, _ = create_dataloaders(training_config, ddp=False)
    ds = loader.dataset

    # 2. Read FITS metadata catalog
    logger.info(f"Reading FITS metadata catalog from {args.metadata_path}...")
    fits_table = Table.read(args.metadata_path)
    fits_df = fits_table.to_pandas()

    for col in fits_df.columns:
        if fits_df[col].dtype == object and isinstance(fits_df[col].iloc[0], bytes):
            try: fits_df[col] = fits_df[col].str.decode('utf-8')
            except: pass

    # 3. Dynamic dataset filtering from model configuration
    allowed_ids = None
    applied_filters = raw_config_dict.get('applied_filters', getattr(config, 'applied_filters', None))
    if applied_filters:
        logger.info(f"Detected applied dataset filters: {applied_filters}")
        # Evaluate filters on FITS catalog
        id_col = 'TARGETID' if 'TARGETID' in fits_df.columns else 'targetid'
        mask = np.ones(len(fits_df), dtype=bool)
        for expr in applied_filters:
            py_expr = expr.replace('&&', '&').replace('||', '|')
            found_words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr)
            eval_ns = {}
            for word in found_words:
                if word in fits_df.columns:
                    eval_ns[word] = fits_df[word].values
            try:
                expr_mask = eval(py_expr, {"np": np}, eval_ns)
                mask &= np.array(expr_mask, dtype=bool)
            except Exception as e:
                logger.error(f"Error evaluating filter '{expr}': {e}")
        allowed_ids = set(fits_df.loc[mask, id_col].values.astype(int))
        logger.info(f"FITS filter matched {len(allowed_ids)} / {len(fits_df)} records.")

    # 4. Compute UWI from pre-computed embeddings
    base_modality = args.base_modality
    if not (embeddings_dir / f"{base_modality}.npy").exists():
        # Fallback: try to find any available modality
        available = [f.stem for f in embeddings_dir.glob("*.npy") if f.stem != "ids"]
        if not available:
            raise FileNotFoundError(f"No embedding .npy files found in {embeddings_dir}")
        base_modality = available[0]
        logger.warning(f"Requested modality '{args.base_modality}' not found. Using '{base_modality}' instead.")

    df_img = compute_uwi_scores(
        embeddings_dir=embeddings_dir,
        base_modality=base_modality,
        contamination=args.contamination,
        knn_neighbors=args.knn_neighbors,
        allowed_ids_filter=allowed_ids
    )

    # Resolve spectrum modality name dynamically, case-insensitive and matching any variant
    spec_modality = None
    for f in embeddings_dir.glob("*.npy"):
        stem_lower = f.stem.lower()
        if "spec" in stem_lower or "specturm" in stem_lower:
            # Avoid picking up images or joint by accident, and exclude the ids file
            if "image" not in stem_lower and "joint" not in stem_lower and f.stem != "ids":
                spec_modality = f.stem
                logger.info(f"Dynamically detected spectrum embedding: '{spec_modality}'")
                break

    df_spec = pd.DataFrame()
    if spec_modality is not None:
        df_spec = compute_uwi_scores(
            embeddings_dir=embeddings_dir,
            base_modality=spec_modality,
            contamination=args.contamination,
            knn_neighbors=args.knn_neighbors,
            allowed_ids_filter=allowed_ids
        )

    # Align indices of embeddings correctly using the unsorted active IDs list
    ids = np.load(embeddings_dir / "ids.npy")
    if allowed_ids is not None:
        allowed_floats = np.array(list(allowed_ids), dtype=np.float64)
        keep_mask = np.isin(ids.astype(np.float64), allowed_floats)
        active_ids_unsorted = ids[keep_mask]
        X_img_raw = np.load(embeddings_dir / f"{base_modality}.npy")[keep_mask]
    else:
        active_ids_unsorted = ids
        X_img_raw = np.load(embeddings_dir / f"{base_modality}.npy")

    # ---------------------------------------------------------------------------
    # 5. Prepare embedding-space features (UWI + Cross-Modal + Anchor matching)
    # ---------------------------------------------------------------------------
    logger.info("Computing embedding-space features (UWI, Cross-Modal, Anchors)...")
    
    df_img_clean = df_img[["TargetID", "Unified_Weirdness_Index"]].copy()
    df_img_clean.rename(columns={"Unified_Weirdness_Index": "UWI_Image"}, inplace=True)
    
    if df_spec is not None and not df_spec.empty:
        df_spec_clean = df_spec[["TargetID", "Unified_Weirdness_Index"]].copy()
        df_spec_clean.rename(columns={"Unified_Weirdness_Index": "UWI_Spectrum"}, inplace=True)
        df_merged = pd.merge(df_img_clean, df_spec_clean, on="TargetID", how="left")
        df_merged['UWI_Spectrum'] = df_merged['UWI_Spectrum'].fillna(df_merged['UWI_Spectrum'].median())
    else:
        df_merged = df_img_clean.copy()
        df_merged['UWI_Spectrum'] = 0.5
        logger.warning("No Spectrum embeddings provided. Cross-Modal Contradiction will be disabled.")
    
    df_merged['Cross_Modal_Contradiction'] = df_merged['UWI_Image'] - df_merged['UWI_Spectrum']
    
    ids_int = active_ids_unsorted.astype(int)
    
    # Prepare normalized embeddings for anchor matching
    X_base = X_img_raw.mean(axis=1) if X_img_raw.ndim == 3 else X_img_raw
    X_norm = X_base / np.maximum(np.linalg.norm(X_base, axis=1, keepdims=True), 1e-10)
    
    n_samples = len(ids_int)
    anchor_sims = np.zeros(n_samples)
    matched_anchors = np.full(n_samples, "None", dtype=object)
    
    # Anchor matching
    if args.anchor_ids:
        logger.info(f"Computing Anchor Matching with {len(args.anchor_ids)} anchors (threshold={args.similarity_threshold})...")
        anchor_embeds = []
        resolved_anchor_ids = []
        for aid in args.anchor_ids:
            idx = find_id_index(active_ids_unsorted, aid)
            if idx is not None:
                anchor_embeds.append(X_norm[idx])
                resolved_anchor_ids.append(aid)
            else:
                logger.warning(f"Anchor TargetID {aid} not found in the active dataset.")
        
        if anchor_embeds:
            anchor_matrix = np.stack(anchor_embeds)
            cos_sims = np.dot(X_norm, anchor_matrix.T)
            anchor_sims = cos_sims.max(axis=1)
            best_anchor_indices = cos_sims.argmax(axis=1)
            matched_anchors = np.array([resolved_anchor_ids[i] for i in best_anchor_indices])
    
    df_anchors = pd.DataFrame({"TargetID": ids_int, "Anchor_Similarity": anchor_sims, "Matched_Anchor": matched_anchors})
    df_merged = pd.merge(df_merged, df_anchors, on="TargetID", how="left")
    
    df_merged.rename(columns={"UWI_Image": "Unified_Weirdness_Index"}, inplace=True)
    df_merged = df_merged.sort_values(by="Unified_Weirdness_Index", ascending=False).reset_index(drop=True)

    # ---------------------------------------------------------------------------
    # 6. Pixel Artifact Score (PAS) — audit top-N UWI candidates with raw data
    # ---------------------------------------------------------------------------
    ds_ids = np.array(ds.ds['targetid'])
    
    n_candidates = min(args.n_candidates, len(df_merged))
    candidate_tids = df_merged.head(n_candidates)['TargetID'].values
    
    logger.info(f"Computing Pixel Artifact Score (PAS) for top {n_candidates} UWI candidates...")
    logger.info(f"  PAS weights: dead_hot={args.pas_w_dead_hot}, channel_corr={args.pas_w_channel_corr}, "
                f"bg_asym={args.pas_w_bg_asymmetry}, zero_nan={args.pas_w_zero_nan}, edge={args.pas_w_edge}")
    logger.info(f"  PAS threshold: {args.pas_threshold}")
    
    pas_records = []
    for i, tid in enumerate(candidate_tids):
        tid_int = int(tid)
        idx = find_arrow_index(ds_ids, tid_int)
        
        if idx is None:
            pas_records.append({
                "TargetID": tid_int, "PAS": 0.0,
                "PAS_dead_hot": 0.0, "PAS_channel_corr": 0.0,
                "PAS_bg_asymmetry": 0.0, "PAS_zero_nan": 0.0, "PAS_edge_sharp": 0.0
            })
            continue
        
        raw_record = ds.ds[idx]
        pas, comps = compute_pixel_artifact_score(
            raw_record,
            w_dead_hot=args.pas_w_dead_hot,
            w_channel_corr=args.pas_w_channel_corr,
            w_bg_asymmetry=args.pas_w_bg_asymmetry,
            w_zero_nan=args.pas_w_zero_nan,
            w_edge=args.pas_w_edge,
        )
        
        pas_records.append({
            "TargetID": tid_int, "PAS": round(pas, 4),
            "PAS_dead_hot": comps["dead_hot"], "PAS_channel_corr": comps["channel_corr"],
            "PAS_bg_asymmetry": comps["bg_asymmetry"], "PAS_zero_nan": comps["zero_nan"],
            "PAS_edge_sharp": comps["edge_sharp"]
        })
        
        if (i + 1) % 100 == 0:
            logger.info(f"  PAS computed for {i + 1} / {n_candidates} candidates...")
    
    df_pas = pd.DataFrame(pas_records)
    df_merged = pd.merge(df_merged, df_pas, on="TargetID", how="left")
    
    # Fill PAS for candidates not in top-N with 0
    for col in ["PAS", "PAS_dead_hot", "PAS_channel_corr", "PAS_bg_asymmetry", "PAS_zero_nan", "PAS_edge_sharp"]:
        df_merged[col] = df_merged[col].fillna(0.0)
    
    # ---------------------------------------------------------------------------
    # 7. Combined Artifact Classification (PAS + Anchor + UWI)
    # ---------------------------------------------------------------------------
    logger.info("Classifying artifacts using combined PAS + Anchor + UWI scoring...")
    
    # IMPORTANT: only classify the top-n_candidates rows (those with PAS computed).
    # Rows beyond n_candidates have PAS=0 (unfilled) and must NOT be flagged by
    # anchor matching, which would otherwise silently scan all 27k rows.
    is_artifact = np.full(len(df_merged), False)
    artifact_reasons = np.full(len(df_merged), "Clean", dtype=object)
    
    classify_limit = min(n_candidates, len(df_merged))
    for i in range(classify_limit):
        row = df_merged.iloc[i]
        reasons = []
        
        pas_val = row['PAS']
        anchor_sim = row['Anchor_Similarity']
        
        # Criterion 1: High PAS (pixel-level corruption evidence)
        if pas_val >= args.pas_threshold:
            reasons.append(f"PixelArtifact(PAS={pas_val:.3f})")
        
        # Criterion 2: Anchor match (close neighbor of a known artifact in embedding space)
        # Requires PAS >= 0.20 as a minimum sanity check so that clean objects that happen
        # to be embedding-neighbors of known artifacts are not flagged.
        if anchor_sim >= args.similarity_threshold and pas_val >= 0.20:
            if not reasons:  # avoid duplicate label if already PixelArtifact
                reasons.append(f"AnchorMatch(sim={anchor_sim:.3f}, PAS={pas_val:.3f})")
        
        if reasons:
            is_artifact[i] = True
            artifact_reasons[i] = " | ".join(reasons)
    
    df_merged['Is_Artifact'] = is_artifact
    df_merged['Artifact_Reasons'] = artifact_reasons
    
    n_artifacts_found = int(np.sum(is_artifact))
    logger.info(f"Artifact detection complete: found {n_artifacts_found} artifacts out of {classify_limit} candidates audited.")
    
    # PAS distribution summary
    pas_vals = df_merged.head(n_candidates)['PAS'].values
    logger.info(f"  PAS distribution (top {n_candidates}): "
                f"min={pas_vals.min():.4f}, median={np.median(pas_vals):.4f}, "
                f"P95={np.percentile(pas_vals, 95):.4f}, max={pas_vals.max():.4f}")

    # 8. Save full catalog
    csv_path = save_dir / "dataset_corruption_audit_ML.csv"
    df_merged.to_csv(csv_path, index=False)
    logger.info(f"Saved complete artifact audit catalog to {csv_path}")

    # 9. Extract top artifacts for PDF report
    df_artifacts = df_merged[df_merged["Is_Artifact"] == True]
    n_plot = min(args.n_plot, len(df_artifacts))
    df_top_artifacts = df_artifacts.head(n_plot).copy()

    logger.info(f"Generating PDF report for top {n_plot} flagged artifacts...")

    # 10. Generate visual validation PDF
    pdf_path = save_dir / "top_flagged_corrupt_galaxies.pdf"

    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{xcolor}'
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', weight='bold')

    if n_plot == 0:
        logger.warning("No artifacts found in the audited candidates. Skipping PDF generation.")
    else:
        from matplotlib.backends.backend_pdf import PdfPages
        
        # Index FITS catalog by target ID for quick morphological metadata lookup
        target_id_col = 'TARGETID' if 'TARGETID' in fits_df.columns else 'targetid'
        fits_indexed = fits_df.drop_duplicates(subset=[target_id_col]).set_index(target_id_col)
        
        logger.info(f"Writing visual validation sheets to multi-page PDF: {pdf_path}")
        with PdfPages(pdf_path) as pdf:
            for r in range(n_plot):
                row = df_top_artifacts.iloc[r]
                tid = int(row.TargetID)
                uwi = row.Unified_Weirdness_Index
                pas_val = row.PAS
                cross_modal = row.Cross_Modal_Contradiction

                # Find record index in Arrow dataset
                idx = find_arrow_index(ds_ids, tid)
                if idx is None:
                    logger.warning(f"ID {tid} not found in raw dataset. Skipping from dashboard.")
                    continue
                    
                sample = ds[idx]
                raw_record = ds.ds[idx]
                
                # Fetch metadata strings
                meta_str, z_val = get_morphological_meta_str(tid, fits_indexed)
                spec_meta_str = get_spectroscopic_meta_str(tid, fits_indexed)
                
                # Create a 4-panel figure matching anomalies_analysis.py layout
                fig = plt.figure(figsize=(22, 14), dpi=150)
                
                gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], width_ratios=[1, 1.1, 0.9], hspace=0.4, wspace=0.3)
                
                ax_meta = fig.add_subplot(gs[0, 0])
                ax_img = fig.add_subplot(gs[0, 1])
                ax_stats = fig.add_subplot(gs[0, 2])
                ax_spec_blue = fig.add_subplot(gs[1, :])
                ax_spec_red = fig.add_subplot(gs[2, :])
                
                # Render Euclid Image
                ax_img.axis('off')
                rgb = extract_raw_rgb(raw_record)
                if rgb is not None:
                    ax_img.imshow(rgb, origin='lower')
                else:
                    ax_img.text(0.5, 0.5, "EuclidImage missing", ha='center', va='center', fontsize=12)
                
                # Classification & Spectroscopic data panel (no "Artifact ML Flags:" label)
                ax_stats.axis('off')
                NL = "\n"
                
                # Build PAS component details
                pas_detail = (
                    f"  - PAS: {pas_val:.4f}{NL}"
                    f"  - Cross-Modal: {cross_modal:.4f}{NL}"
                    f"  - Dead/Hot: {row.PAS_dead_hot:.4f}{NL}"
                    f"  - Ch. Corr: {row.PAS_channel_corr:.4f}{NL}"
                    f"  - BG Asym: {row.PAS_bg_asymmetry:.4f}{NL}"
                    f"  - Zero/NaN: {row.PAS_zero_nan:.4f}{NL}"
                    f"  - Edge Sharp: {row.PAS_edge_sharp:.4f}{NL}{NL}"
                    f"{spec_meta_str}"
                )
                ax_stats.text(
                    0.5, 0.95, pas_detail,
                    transform=ax_stats.transAxes,
                    fontsize=14,
                    linespacing=1.6,
                    horizontalalignment='center',
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.8", fc="mistyrose", alpha=0.95, ec="red", lw=1.5)
                )
                ax_stats.set_title(r"\textbf{Artifact classification \& spectra}", fontsize=16, fontweight='bold', color='darkred', pad=15)
                
                # Physical and Morphological properties card in ax_meta
                ax_meta.axis('off')
                ax_meta.text(
                    0.5, 0.95, meta_str, 
                    transform=ax_meta.transAxes, 
                    fontsize=14,
                    linespacing=1.8,
                    horizontalalignment='center',
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.8", fc="ivory", alpha=0.95, ec="darkgrey", lw=1.0)
                )
                ax_meta.set_title(r"\textbf{Physical and Morphological properties}", fontsize=16, fontweight='bold', color='navy', pad=15)
                
                # DESI Spectrum split into Blue and Red Channels
                has_spectra = False
                try:
                    if raw_record.get('spectrum_flux') is not None:
                        spec_gt = np.array(raw_record['spectrum_flux']).flatten()
                        wave_ang = np.array(raw_record['spectrum_wave']).flatten()
                        
                        w_min, w_max = wave_ang.min(), wave_ang.max()
                        w_mid = (w_min + w_max) / 2
                        
                        # Plot Blue Channel
                        ax_spec_blue.plot(wave_ang, spec_gt, 'k-', lw=1, alpha=0.8)
                        ax_spec_blue.set_xlim(w_min, w_mid)
                        ax_spec_blue.set_title(r"\textbf{Spectrum (Blue Channel)}", fontsize=11, fontweight='bold')
                        ax_spec_blue.set_ylabel(r"Flux", fontsize=12)
                        plot_spectral_lines(ax_spec_blue, w_min, w_mid, z_val)
                        
                        # Plot Red Channel
                        ax_spec_red.plot(wave_ang, spec_gt, 'k-', lw=1, alpha=0.8)
                        ax_spec_red.set_xlim(w_mid, w_max)
                        ax_spec_red.set_title(r"\textbf{Spectrum (Red Channel)}", fontsize=11, fontweight='bold')
                        ax_spec_red.set_ylabel(r"Flux", fontsize=12)
                        ax_spec_red.set_xlabel(r"Observed Wavelength [\AA]", fontsize=13, fontweight='bold')
                        plot_spectral_lines(ax_spec_red, w_mid, w_max, z_val)
                        
                        has_spectra = True
                    else:
                        ax_spec_blue.text(0.5, 0.5, "DESISpectrum not in raw database", ha='center', va='center', fontsize=12)
                        ax_spec_red.axis('off')
                except Exception as e:
                    logger.error(f"Error rendering DESI spectrum for artifact sheet: {e}")
                    ax_spec_blue.text(0.5, 0.5, "DESISpectrum rendering failed", ha='center', va='center', fontsize=12)
                    ax_spec_red.axis('off')
                    
                plt.suptitle(rf"\textbf{{Artifact (UWI: {uwi:.4f}, PAS: {pas_val:.4f}): ID {tid}}}", fontsize=22, fontweight='bold', color='darkred', y=0.98)
                pdf.savefig()
                plt.close()

    logger.info(f"Successfully saved artifact detection results to {save_dir}")
    logger.info("Dataset artifact detection completed successfully.")

if __name__ == "__main__":
    main()
