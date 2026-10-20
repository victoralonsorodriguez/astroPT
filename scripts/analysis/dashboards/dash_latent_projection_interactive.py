import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import logging
import sys
import warnings
import umap
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning

st.set_page_config(layout="wide", page_title="AstroPT 3D Latent Projections")

# Logger Configuration
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-Latent-Interactive")
warnings.simplefilter('ignore', category=AstropyWarning)

# Constants & Palettes
UMAP_PARAMS = {
    "n_neighbors": 30,
    "min_dist": 0.1,
    "n_components": 3,  # 3D!
    "metric": "cosine",
    "random_state": 61,
    "verbose": False,
    "n_jobs": -1
}

SURVEY_ORDER = ['main', 'sv1', 'sv3', 'special']
CAT_PALETTE = {
    "GALAXY": "#006eff", 
    "QSO": "#fbd500", 
    "DR1_R1": "#1b9e77", 
    "DR1_R2": "#e7298a", 
    "Other": "#666666"
}

# Color configurations matching the static dashboard properties
def log_transform(x): 
    return np.log10(np.where(x <= 0, 1e-5, x))

COLOR_PROPERTIES = {
    "Redshift (z)": {"col": "Z", "cat": False, "trans": None},
    "Stellar Mass (log M*)": {"col": "LOGMSTAR", "cat": False, "trans": None},
    "SFR (log SFR)": {"col": "LOGSFR", "cat": False, "trans": None},
    "Color (g-r)": {"col": "GR", "cat": False, "trans": None},
    "Spectral Type": {"col": "SPECTYPE", "cat": True, "trans": None},
    "Data Set Release": {"col": "data_set_release", "cat": True, "trans": None},
    "Total Flux (log)": {"col": "flux_detection_total", "cat": False, "trans": log_transform},
    "Halpha EW (log)": {"col": "HALPHA_EW", "cat": False, "trans": log_transform},
    "Halpha Flux (log)": {"col": "HALPHA_FLUX", "cat": False, "trans": log_transform},
    "Sersic VIS Index": {"col": "sersic_sersic_vis_index", "cat": False, "trans": None},
    "Spiral Arms Prob": {"col": "has_spiral_arms_yes", "cat": False, "trans": None},
    "Gini Coefficient": {"col": "gini", "cat": False, "trans": None}
}

@st.cache_data(show_spinner="Loading embeddings and catalog data...")
def load_data(emb_dir: str, metadata_path: str):
    emb_dir = Path(emb_dir)
    metadata_path = Path(metadata_path)
    
    data_dict = {}
    id_candidates = ['targetid.npy', 'ids.npy', 'target_ids.npy', 'object_ids.npy']
    id_path = next((emb_dir / c for c in id_candidates if (emb_dir / c).exists()), None)
    if not id_path: 
        raise FileNotFoundError(f"Could not find Target IDs in {emb_dir}")
    data_dict['targetid'] = np.load(id_path)

    excluded = set(id_candidates) | {'metadata.npy', 'indices.npy'}
    for f in emb_dir.glob("*.npy"):
        if f.name not in excluded and not f.name.startswith("cache_umap"):
            data_dict[f.stem] = np.load(f, mmap_mode='r')

    # Load Metadata Table
    catalog = Table.read(metadata_path)
    df = catalog.to_pandas()
    
    for col in df.columns:
        if df[col].dtype == object and isinstance(df[col].iloc[0], bytes):
            try: 
                df[col] = df[col].str.decode('utf-8')
            except: 
                pass
    if 'SURVEY' in df.columns: 
        df['SURVEY'] = df['SURVEY'].str.lower().str.strip()
        
    return data_dict, df

def align_data(embeddings_dict, catalog_df):
    emb_ids = embeddings_dict['targetid']
    target_col = 'TARGETID' if 'TARGETID' in catalog_df.columns else 'targetid'
    
    catalog_df[target_col] = catalog_df[target_col].astype('int64')
    catalog_indexed = catalog_df.drop_duplicates(subset=[target_col]).set_index(target_col)
    
    common_ids = np.intersect1d(emb_ids, catalog_indexed.index.values)
    matched_catalog = catalog_indexed.loc[common_ids].reset_index()
    
    id_to_idx = {id_val: i for i, id_val in enumerate(emb_ids)}
    valid_indices = np.array([id_to_idx[uid] for uid in matched_catalog[target_col].values])
    
    return matched_catalog, valid_indices

def get_umap_3d(emb_dir: Path, mod: str, emb_data: np.ndarray, valid_idx: np.ndarray) -> np.ndarray:
    """Precompute or load 3D UMAP coordinates from local cache."""
    cache_path = emb_dir / f"cache_umap_3d_{mod}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    
    # Calculate UMAP
    logger.info(f"Computing 3D UMAP manifold for {mod} (using all aligned indices)...")
    reducer = umap.UMAP(**UMAP_PARAMS)
    umap_coords = reducer.fit_transform(emb_data[valid_idx])
    np.save(cache_path, umap_coords)
    return umap_coords

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive 3D UMAP Dashboard")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to .fits catalog")
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing training weights")
    parser.add_argument("--emb_dir", type=str, required=True, help="Directory containing .npy embedding files")
    return parser.parse_known_args()

def main():
    # Parse CLI args
    args, unknown = parse_args()
    emb_dir = Path(args.emb_dir)
    
    st.title("🔭 AstroPT Interactive 3D Latent Projections")
    st.write("A modality-agnostic web app to compare multi-dimensional embeddings via UMAP-3D.")
    
    # Load and Align Data
    try:
        raw_data, catalog_df = load_data(str(args.emb_dir), str(args.metadata_path))
        df_aligned, valid_idx = align_data(raw_data, catalog_df)
    except Exception as e:
        st.error(f"Error loading or aligning dataset files: {e}")
        return

    modalities = [k for k in raw_data.keys() if k != 'targetid']
    if not modalities:
        st.error("No modalities detected in the embeddings directory.")
        return

    # Sidebar controls
    st.sidebar.header("🎛️ Control Panel")
    
    # Subsampling slider to maintain UI responsiveness
    max_pts = len(df_aligned)
    num_points = st.sidebar.slider(
        "Max Points to Render", 
        min_value=1000, 
        max_value=min(max_pts, 50000), 
        value=min(max_pts, 15000),
        step=500,
        help="Reducing points ensures fluid 3D interactions in the web browser."
    )
    
    # Subsample data deterministically
    np.random.seed(42)
    if max_pts > num_points:
        subsample_idx = np.random.choice(max_pts, num_points, replace=False)
        subsample_idx = np.sort(subsample_idx)
        plot_df = df_aligned.iloc[subsample_idx].reset_index(drop=True)
        plot_valid_idx = valid_idx[subsample_idx]
    else:
        plot_df = df_aligned.copy()
        plot_valid_idx = valid_idx.copy()

    # Pre-compute/Load 3D UMAPs for all modalities (cached)
    with st.spinner("Preparing UMAP-3D manifolds..."):
        umap_storage = {}
        for mod in modalities:
            # We pass the full valid_idx to keep cache consistent, then slice for plotting
            full_coords = get_umap_3d(emb_dir, mod, raw_data[mod], valid_idx)
            if max_pts > num_points:
                umap_storage[mod] = full_coords[subsample_idx]
            else:
                umap_storage[mod] = full_coords

    # Color Mapping Selector
    st.sidebar.subheader("🎨 Colormap Options")
    color_label = st.sidebar.selectbox("Color points by:", options=list(COLOR_PROPERTIES.keys()), index=0)
    color_cfg = COLOR_PROPERTIES[color_label]
    
    # Prepare color column
    color_col = color_cfg["col"]
    if color_col not in plot_df.columns:
        st.sidebar.warning(f"Column {color_col} not found in metadata. Defaulting to Redshift.")
        color_label = "Redshift (z)"
        color_cfg = COLOR_PROPERTIES[color_label]
        color_col = color_cfg["col"]

    # Handle transformation if any (like log)
    c_vals = plot_df[color_col].values
    if not color_cfg["cat"]:
        c_vals = c_vals.astype(float)
        if color_cfg["trans"]:
            c_vals = color_cfg["trans"](c_vals)
        # Clean infs/nans
        c_vals = np.where(np.isnan(c_vals), np.nanmedian(c_vals), c_vals)
        c_vals = np.where(np.isinf(c_vals), np.nanmedian(c_vals), c_vals)
        plot_df['color_val'] = c_vals
    else:
        plot_df['color_val'] = c_vals.astype(str)

    # Filter by survey if needed
    available_surveys = sorted(plot_df['SURVEY'].dropna().unique().tolist())
    selected_surveys = st.sidebar.multiselect("Filter by Survey:", options=available_surveys, default=available_surveys)
    survey_mask = plot_df['SURVEY'].isin(selected_surveys)

    # Layout comparison modes
    st.sidebar.markdown("---")
    st.sidebar.subheader("🖥️ View Mode")
    compare_mode = st.sidebar.checkbox("Compare two Modalities Side-by-Side", value=True)

    # Theme selection for background
    bg_theme = st.sidebar.selectbox("Background theme:", ["Dark", "Light"], index=0)
    bg_color = "rgb(15, 15, 15)" if bg_theme == "Dark" else "rgb(255, 255, 255)"
    text_color = "white" if bg_theme == "Dark" else "black"

    # Define plotting function helper
    def render_3d_plot(coords, df_subset, mod_title):
        # Filter coords
        coords_masked = coords[survey_mask]
        df_masked = df_subset[survey_mask]
        
        if len(df_masked) == 0:
            return go.Figure().update_layout(title="No data matches filters.")

        # Build hover text containing multiple properties
        hover_texts = []
        for _, row in df_masked.iterrows():
            z_val = f"{row['Z']:.4f}" if 'Z' in row else 'N/A'
            mstar = f"{row['LOGMSTAR']:.2f}" if 'LOGMSTAR' in row else 'N/A'
            spec = row['SPECTYPE'] if 'SPECTYPE' in row else 'N/A'
            hover_texts.append(
                f"<b>ID:</b> {row['TARGETID']}<br>"
                f"<b>Redshift (z):</b> {z_val}<br>"
                f"<b>log M*:</b> {mstar}<br>"
                f"<b>Spectype:</b> {spec}<br>"
                f"<b>Survey:</b> {row['SURVEY'].upper()}"
            )

        if color_cfg["cat"]:
            # Grouped Scatter3D to support discrete categorical legend colors
            fig = go.Figure()
            unique_cats = sorted(df_masked['color_val'].unique())
            for cat in unique_cats:
                cat_mask = df_masked['color_val'] == cat
                cat_coords = coords_masked[cat_mask]
                cat_indices = np.where(cat_mask)[0]
                
                cat_hovers = [hover_texts[i] for i in cat_indices]
                
                fig.add_trace(go.Scatter3d(
                    x=cat_coords[:, 0],
                    y=cat_coords[:, 1],
                    z=cat_coords[:, 2],
                    mode='markers',
                    name=cat,
                    marker=dict(
                        size=3,
                        color=CAT_PALETTE.get(cat, '#888888'),
                        opacity=0.8
                    ),
                    text=cat_hovers,
                    hoverinfo='text'
                ))
        else:
            # Continuous Scatter3D
            fig = go.Figure(data=[go.Scatter3d(
                x=coords_masked[:, 0],
                y=coords_masked[:, 1],
                z=coords_masked[:, 2],
                mode='markers',
                marker=dict(
                    size=2.5,
                    color=df_masked['color_val'],
                    colorscale='turbo' if bg_theme == "Light" else 'Viridis',
                    colorbar=dict(title=color_label, thickness=15),
                    opacity=0.75
                ),
                text=hover_texts,
                hoverinfo='text'
            )])

        # Clear axes ticks and labels for clean look
        fig.update_layout(
            title=dict(text=f"<b>{mod_title.upper()} Space</b>", font=dict(size=18, color=text_color)),
            scene=dict(
                xaxis=dict(backgroundcolor=bg_color, gridcolor='rgba(128,128,128,0.2)', showbackground=False, showticklabels=False, title=''),
                yaxis=dict(backgroundcolor=bg_color, gridcolor='rgba(128,128,128,0.2)', showbackground=False, showticklabels=False, title=''),
                zaxis=dict(backgroundcolor=bg_color, gridcolor='rgba(128,128,128,0.2)', showbackground=False, showticklabels=False, title=''),
            ),
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            margin=dict(l=0, r=0, b=0, t=50),
            height=700
        )
        return fig

    # Render plots based on View Mode
    if compare_mode:
        col1, col2 = st.columns(2)
        
        with col1:
            mod1 = st.selectbox("Select Left Modality:", options=modalities, index=0, key="left_mod")
            fig1 = render_3d_plot(umap_storage[mod1], plot_df, mod1)
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            mod2 = st.selectbox("Select Right Modality:", options=modalities, index=min(1, len(modalities)-1), key="right_mod")
            fig2 = render_3d_plot(umap_storage[mod2], plot_df, mod2)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        mod_selected = st.selectbox("Select Modality:", options=modalities, index=0, key="single_mod")
        fig = render_3d_plot(umap_storage[mod_selected], plot_df, mod_selected)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
