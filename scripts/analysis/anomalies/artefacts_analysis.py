"""
AstroPT Embedding-Based Dataset Artifact Detection Tool.

This script uses pre-computed AstroPT latent embeddings to identify observational
artifacts (sensor failures, satellite trails, coordinate misalignments, corrupt data).

Strategy:
1. Compute the Unified Weirdness Index (UWI) using embedding-space anomaly detectors.
2. Audit the top N UWI outliers by inspecting raw pixel/spectral data.
3. Classify each outlier as a camera/sensor artifact or a physical anomaly.
4. Output a catalog of flagged artifacts and a PDF dashboard for visual validation.

This approach is orders of magnitude faster than per-sample model inference because
it leverages pre-computed embedding matrices instead of running the full model.

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
                        help="Number of top UWI outliers to audit for artifact classification")
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
    parser.add_argument("--similarity_threshold", type=float, default=0.85,
                        help="Cosine similarity threshold for matching reference anchors")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Artifact Classification Engine
# ---------------------------------------------------------------------------

def check_if_artifact(raw_record: dict) -> Tuple[bool, List[str], float, float]:
    """
    Inspects raw pixel and spectral data from an Arrow record to determine
    whether an outlier is a camera/sensor artifact or a genuine physical anomaly.

    Returns:
        (is_artifact, reasons, zero_fraction, edge_score)
    """
    reasons = []
    zero_fraction = 0.0
    edge_score = 0.0

    def get_ch(k):
        val = raw_record.get(k)
        if val is None:
            return np.array([], dtype=np.float32)
        return np.array(val, dtype=np.float32)

    vis = get_ch('image_vis')
    y = get_ch('image_nisp_y')
    j = get_ch('image_nisp_j')
    h = get_ch('image_nisp_h')

    # --- Image-based artifact checks ---
    bands = {'VIS': vis, 'Y': y, 'J': j, 'H': h}

    for band_name, band_data in bands.items():
        if band_data.size == 0:
            reasons.append(f"Missing {band_name} band")
            continue

        # Flat band detection (sensor failure / dead channel)
        if np.std(band_data) < 1e-4:
            reasons.append(f"Flat {band_name} band (std < 1e-4)")

    # Zero-pixel fraction in VIS (primary band)
    if vis.size > 0:
        zero_fraction = float(np.mean(vis <= 1e-7))
        if zero_fraction > 0.05:
            reasons.append(f"High zero-pixel fraction in VIS ({zero_fraction:.1%})")

        # Laplacian edge score for sharp unphysical gradients
        if vis.ndim == 2:
            from scipy.signal import convolve2d
            lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
            laplacian = convolve2d(vis, lap_kernel, mode='same')
            edge_score = float(np.std(laplacian))
            if edge_score > 15.0:
                reasons.append(f"Sharp edges detected (Laplacian std={edge_score:.1f})")

    # --- Spectrum-based artifact checks ---
    spec_flux = raw_record.get('spectrum_flux')
    if spec_flux is not None:
        spec = np.array(spec_flux, dtype=np.float32).flatten()
        if spec.size > 0:
            if np.std(spec) < 1e-4:
                reasons.append("Flat spectrum (std < 1e-4)")
            if np.all(spec <= 0):
                reasons.append("All-negative/zero spectrum")

    is_artifact = len(reasons) > 0
    return is_artifact, reasons, zero_fraction, edge_score


def classify_artifacts_by_embeddings(
    df_scores: pd.DataFrame,
    X_base: np.ndarray,
    active_ids_unsorted: np.ndarray,
    anchor_ids: Optional[List[int]] = None,
    threshold: float = 0.85,
    eps: float = 0.15,
    min_samples: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classifies outliers using purely embedding-space logic (Method 1 or Method 2).
    
    Returns:
        is_artifact_col: boolean array indicating if each source is an artifact
        artifact_reasons_col: string array explaining why
    """
    n_samples = len(df_scores)
    is_artifact_col = np.full(n_samples, False)
    artifact_reasons_col = np.full(n_samples, "Clean", dtype=object)

    ids_int = active_ids_unsorted.astype(int)
    
    # Normalize embeddings to unit L2 length
    X_norm = X_base / np.maximum(np.linalg.norm(X_base, axis=1, keepdims=True), 1e-10)

    if anchor_ids:
        logger.info(f"Using Method 2: Few-Shot Cosine Similarity with {len(anchor_ids)} anchors...")
        anchor_embeds = []
        resolved_anchor_ids = []
        for aid in anchor_ids:
            idx_list = np.where(ids_int == int(aid))[0]
            if len(idx_list) > 0:
                anchor_embeds.append(X_norm[idx_list[0]])
                resolved_anchor_ids.append(aid)
            else:
                logger.warning(f"Anchor TargetID {aid} not found in the dataset embeddings.")
        
        if not anchor_embeds:
            logger.error("None of the specified anchor IDs were found in the dataset! Falling back to Method 1 (DBSCAN).")
            anchor_ids = None
        else:
            anchor_matrix = np.stack(anchor_embeds)
            cos_sims = np.dot(X_norm, anchor_matrix.T)
            max_sims = cos_sims.max(axis=1)
            best_anchor_indices = cos_sims.argmax(axis=1)
            
            for idx in range(n_samples):
                tid = int(df_scores.iloc[idx]["TargetID"])
                matches = np.where(ids_int == tid)[0]
                if len(matches) == 0:
                    continue
                m_idx = matches[0]
                sim = max_sims[m_idx]
                if sim >= threshold:
                    matched_aid = resolved_anchor_ids[best_anchor_indices[m_idx]]
                    is_artifact_col[idx] = True
                    artifact_reasons_col[idx] = f"Anchor Similarity: matched {matched_aid} (sim={sim:.3f})"
            return is_artifact_col, artifact_reasons_col

    if not anchor_ids:
        logger.info("Using Method 1: DBSCAN Outlier Clustering...")
        candidate_tids = df_scores["TargetID"].values
        
        candidate_indices = []
        valid_candidate_tids = []
        for tid in candidate_tids:
            matches = np.where(ids_int == int(tid))[0]
            if len(matches) > 0:
                candidate_indices.append(matches[0])
                valid_candidate_tids.append(tid)
        
        candidate_indices = np.array(candidate_indices)
        
        if len(candidate_indices) > 0:
            X_cand = X_norm[candidate_indices]
            db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            labels = db.fit_predict(X_cand)
            
            artifact_tids_set = set()
            tid_to_label = {}
            for tid, label in zip(valid_candidate_tids, labels):
                if label >= 0:
                    artifact_tids_set.add(tid)
                    tid_to_label[tid] = label
            
            for idx in range(n_samples):
                tid = int(df_scores.iloc[idx]["TargetID"])
                if tid in artifact_tids_set:
                    is_artifact_col[idx] = True
                    artifact_reasons_col[idx] = f"Clustered Artifact: DBSCAN cluster {tid_to_label[tid]}"
                else:
                    if tid in valid_candidate_tids:
                        artifact_reasons_col[idx] = "Clean (DBSCAN Outlier/Noise)"
                    else:
                        artifact_reasons_col[idx] = "Not Audited (Below threshold)"
                        
    return is_artifact_col, artifact_reasons_col


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

    df_scores = compute_uwi_scores(
        embeddings_dir=embeddings_dir,
        base_modality=base_modality,
        contamination=args.contamination,
        knn_neighbors=args.knn_neighbors,
        allowed_ids_filter=allowed_ids
    )

    # Align indices of embeddings correctly using the unsorted active IDs list
    ids = np.load(embeddings_dir / "ids.npy")
    if allowed_ids is not None:
        active_ids_unsorted = ids[np.isin(ids.astype(int), list(allowed_ids))]
    else:
        active_ids_unsorted = ids

    X_base = np.load(embeddings_dir / f"{base_modality}.npy")
    if allowed_ids is not None:
        X_base = X_base[np.isin(ids.astype(int), list(allowed_ids))]
    if X_base.ndim == 3:
        X_base = X_base.mean(axis=1)

    # 5. Pure Embedding-Based Artifact Detection
    is_artifact_col, artifact_reasons_col = classify_artifacts_by_embeddings(
        df_scores=df_scores,
        X_base=X_base,
        active_ids_unsorted=active_ids_unsorted,
        anchor_ids=args.anchor_ids,
        threshold=args.similarity_threshold,
        eps=0.15,
        min_samples=3
    )

    df_scores["Is_Artifact"] = is_artifact_col
    df_scores["Artifact_Reasons"] = artifact_reasons_col
    df_scores["Zero_Pixel_Fraction"] = np.nan
    df_scores["Sharp_Edge_Score"] = np.nan

    n_artifacts_found = int(np.sum(is_artifact_col))
    logger.info(f"Artifact detection complete: found {n_artifacts_found} artifacts.")

    # 6. Save full catalog
    csv_path = save_dir / "dataset_corruption_audit.csv"
    df_scores.to_csv(csv_path, index=False)
    logger.info(f"Saved complete artifact audit catalog to {csv_path}")

    # 7. Extract top artifacts for PDF report
    df_artifacts = df_scores[df_scores["Is_Artifact"] == True]
    n_plot = min(args.n_plot, len(df_artifacts))
    df_top_artifacts = df_artifacts.head(n_plot).copy()
    
    # Audit spatial priors ONLY for the top artifacts selected for plotting to save time
    logger.info(f"Auditing spatial priors for top {n_plot} plotted artifacts...")
    ds_ids = np.array(ds.ds['targetid'])
    zero_fracs = []
    edge_scores = []
    
    for ridx in range(n_plot):
        tid = int(df_top_artifacts.iloc[ridx]["TargetID"])
        matches = np.where(ds_ids == tid)[0]
        z_frac, e_score = 0.0, 0.0
        if len(matches) > 0:
            try:
                raw_record = ds.ds[int(matches[0])]
                _, _, z_frac, e_score = check_if_artifact(raw_record)
            except Exception as e:
                logger.warning(f"Error computing spatial priors for top artifact {tid}: {e}")
        zero_fracs.append(z_frac)
        edge_scores.append(e_score)
        
    df_top_artifacts["Zero_Pixel_Fraction"] = zero_fracs
    df_top_artifacts["Sharp_Edge_Score"] = edge_scores

    logger.info(f"Generating PDF report for top {n_plot} flagged artifacts...")

    # 8. Generate visual validation PDF
    pdf_path = save_dir / "top_flagged_corrupt_galaxies.pdf"

    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{xcolor}'
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', weight='bold')

    if n_plot == 0:
        logger.warning("No artifacts found in the audited candidates. Skipping PDF generation.")
    else:
        fig, axes = plt.subplots(n_plot, 1, figsize=(10, 4 * n_plot), dpi=150)
        fig.subplots_adjust(hspace=0.45, top=0.92, bottom=0.04)

        if n_plot == 1:
            axes = [axes]

        for r in range(n_plot):
            row = df_top_artifacts.iloc[r]
            tid = int(row.TargetID)
            uwi = row.Unified_Weirdness_Index
            z_frac = row.Zero_Pixel_Fraction
            edge_val = row.Sharp_Edge_Score
            art_reasons = row.Artifact_Reasons

            ax = axes[r]
            ax.axis('off')

            matches = np.where(ds_ids == tid)[0]
            rgb = None
            if len(matches) > 0:
                raw_record = ds.ds[int(matches[0])]
                rgb = extract_raw_rgb(raw_record)

            sub_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=ax.get_subplotspec(), width_ratios=[1, 1], wspace=0.1)

            ax_img = fig.add_subplot(sub_gs[0, 0])
            ax_txt = fig.add_subplot(sub_gs[0, 1])

            ax_img.axis('off')
            if rgb is not None:
                ax_img.imshow(rgb, origin='lower')
            else:
                ax_img.text(0.5, 0.5, "EuclidImage missing", ha='center', va='center', fontsize=12)

            ax_txt.axis('off')
            # Escape underscores for LaTeX
            safe_reasons = str(art_reasons).replace("_", "\\_")
            meta_text = (
                rf"\textbf{{TargetID}}: {tid}" + "\n"
                rf"\textbf{{UWI Score}}: {uwi:.4f}" + "\n\n"
                rf"\textbf{{Zero Pixel Fraction}}: {z_frac:.2%}" + "\n"
                rf"\textbf{{Laplacian Edge Score}}: {edge_val:.1f}" + "\n\n"
                rf"\textbf{{Artifact Reasons}}:" + "\n"
                f"{safe_reasons}"
            )

            ax_txt.text(
                0.05, 0.95, meta_text,
                transform=ax_txt.transAxes,
                fontsize=12,
                linespacing=1.6,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.7", fc="mistyrose", alpha=0.95, ec="red", lw=1.5)
            )

            ax.set_title(rf"\textbf{{Rank \#{r+1} - Artifact (UWI: {uwi:.4f})}}", fontsize=14, fontweight='bold', color='darkred', pad=10)

        plt.suptitle(r"\textbf{AstroPT Embedding-Based Artifact Detection (Top Flags)}", fontsize=18, fontweight='bold', y=0.97)
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close()

    logger.info(f"Successfully saved artifact detection results to {save_dir}")
    logger.info("Dataset artifact detection completed successfully.")

if __name__ == "__main__":
    main()
