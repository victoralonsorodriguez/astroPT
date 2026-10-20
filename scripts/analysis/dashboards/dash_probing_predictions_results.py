import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Classification targets to skip/warn (plots only apply to regression)
CLASSIFICATION_TARGETS = ['SPECTYPE', 'data_set_release']
FLUX_TARGETS = ['flux_detection_total', 'HALPHA_EW', 'HALPHA_FLUX', 'NII_6584_FLUX', 'OIII_5007_FLUX', 'HBETA_FLUX']

# LaTeX Label mapping for publication-quality plots
LATEX_TARGET_MAP = {
    'Z': r'$z$',
    'LOGMSTAR': r'$\bm{\log(M_*)}$',
    'LOGSFR': r'$\log(\mathrm{SFR})$',
    'GR': r'$g - r$',
    'flux_detection_total': r'$F_{\mathrm{total}}$',
    'HALPHA_EW': r'$\mathrm{EW}(\mathrm{H}\alpha)$',
    'HALPHA_FLUX': r'$F(\mathrm{H}\alpha)$',
    'NII_6584_FLUX': r'$F([\mathrm{N\,\mathrm{II}}]\,\lambda 6584)$',
    'OIII_5007_FLUX': r'$F([\mathrm{O\,\mathrm{III}}]\,\lambda 5007)$',
    'OIII_5007_SIGMA': r'$\sigma([\mathrm{O\,\mathrm{III}}]\,\lambda 5007)$',
    'HBETA_FLUX': r'$\bm{F(\mathrm{H}\beta)}$',
    'smooth_or_featured_smooth': r'Smoothness frac.',
    'smoothness': r'Smoothness',
    'gini': r'Gini',
    'sersic_sersic_vis_radius': r'$\bm{R_{\mathrm{Sersic}}}$',
    'sersic_sersic_vis_index': r'$\bm{n_{\mathrm{Sersic}}}$',
    'sersic_sersic_vis_axis_ratio': r'$b/a$',
    'has_spiral_arms_yes': r'Spiral arms',
    'phz_pp_median_redshift': r'$z_{\mathrm{phot}}$',
    'phz_pp_median_stellarmass': r'$\bm{\log(M_{*,\mathrm{phot}})}$'
}

# Plotting Global Configuration (matching LaTeX style of other TFM plots)
plt.rcParams['text.latex.preamble'] = r'''
            \usepackage{siunitx}
            \usepackage{amsmath} 
            \usepackage{times}
            \usepackage{bm}
            \sisetup{
            detect-family,
            separate-uncertainty=true,
            output-decimal-marker={.},
            exponent-product=\cdot,
            inter-unit-product=\cdot,
            }
            \DeclareSIUnit{\cts}{cts}
            '''
plt.rc('text', usetex=True)
plt.rc('font', family='serif') 

plt.rcParams.update({
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'font.size': 16,          
    'axes.labelsize': 22,     
    'axes.titlesize': 26,     
    'xtick.labelsize': 20,   
    'ytick.labelsize': 20,
    'legend.fontsize': 18,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titlesize': 28,
    'figure.titleweight': 'bold',
})

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-ready probing diagnostic plots.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing prediction NPZ files.")
    parser.add_argument("--targets", nargs="+", type=str, required=True, help="Regression targets to plot (e.g. Z HALPHA_FLUX).")
    parser.add_argument("--modes", nargs="+", type=str, required=True, help="Modalities/embeddings to plot (e.g. EuclidImage DESISpectrum joint).")
    parser.add_argument("--probes", nargs="+", type=str, default=["MLP"], help="Probes to plot (e.g. MLP LP KNN).")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory where generated plots will be saved.")
    parser.add_argument("--subsample", type=int, default=10000, help="Subsample limit for visualization speed.")
    parser.add_argument("--disable_asinh", action="store_true", help="Disable asinh scaling for fluxes.")
    return parser.parse_args()

def parse_npz_filename(filename: str) -> tuple:
    """Parse target, mode, and probe from prediction filename."""
    KNOWN_PROBES = {'knn', 'lp', 'mlp'}
    KNOWN_MODES = {
        'desispectrum', 'euclidimage', 'cls', 'joint',
        'desispectrum_phase1', 'desispectrum_phase2',
        'euclidimage_phase1', 'euclidimage_phase2',
        'cls_phase1', 'cls_phase2',
        'joint_phase1', 'joint_phase2',
        'images', 'spectra',
        'images_phase1', 'images_phase2',
        'spectra_phase1', 'spectra_phase2',
        'desispectrum_pooled', 'euclidimage_pooled',
        'joint_mean', 'joint_concat'
    }
    
    stem = Path(filename).stem
    if not stem.startswith('preds_'):
        return None, None, None
        
    body = stem[len('preds_'):]
    parts = body.split('_')
    if len(parts) < 3:
        return None, None, None
        
    probe = parts[-1]
    if probe.lower() not in KNOWN_PROBES:
        return None, None, None
        
    remaining = parts[:-1]
    mode = None
    target_parts = None
    
    # Try 2-token mode (e.g. DESISpectrum_pooled, joint_mean)
    if len(remaining) >= 3:
        candidate_mode = remaining[-2] + '_' + remaining[-1]
        if candidate_mode.lower() in KNOWN_MODES:
            mode = candidate_mode
            target_parts = remaining[:-2]
            
    # Try 1-token mode (e.g. joint, cls)
    if mode is None and len(remaining) >= 2:
        candidate_mode = remaining[-1]
        if candidate_mode.lower() in KNOWN_MODES:
            mode = candidate_mode
            target_parts = remaining[:-1]
            
    if mode is None or not target_parts:
        return None, None, None
        
    target = '_'.join(target_parts)
    return target, mode, probe

def matches_list(value: str, req_list: list) -> bool:
    v_low = value.lower()
    for req in req_list:
        r_low = req.lower()
        if r_low == v_low or r_low in v_low:
            return True
    return False

def get_label(target: str, is_scaled: bool) -> str:
    base = LATEX_TARGET_MAP.get(target, target.replace('_', r'\_'))
    if is_scaled and target in FLUX_TARGETS:
        if base.startswith('$') and base.endswith('$'):
            inner = base[1:-1]
            return rf'$\bm{{\mathrm{{asinh}}}}({inner})$'
        else:
            return rf'$\bm{{\mathrm{{asinh}}}}({{\mathrm{{{base}}}}})$'
    return base

def apply_scaling(vals: np.ndarray, target: str) -> np.ndarray:
    if target in FLUX_TARGETS:
        return np.arcsinh(vals)
    return vals

def generate_diagnostic_plot(
    npz_path: Path,
    target: str,
    mode: str,
    probe: str,
    save_dir: Path,
    subsample_limit: int,
    enable_asinh: bool,
    results_df: pd.DataFrame = None
) -> None:
    print(f"[INFO] Processing: {npz_path.name}")
    try:
        data = np.load(npz_path)
        preds = data['preds']
        true_vals = data['true_vals']
    except Exception as e:
        print(f"[ERROR] Failed to load data from {npz_path.name}: {e}")
        return
        
    # Filter non-finite elements
    mask = np.isfinite(preds) & np.isfinite(true_vals)
    preds = preds[mask]
    true_vals = true_vals[mask]
    
    if len(preds) == 0:
        print(f"[WARNING] No valid finite prediction data found in {npz_path.name}")
        return
        
    # Calculate global metrics on full dataset
    ss_res = np.sum((true_vals - preds) ** 2)
    ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean((true_vals - preds) ** 2))
    bias = np.median(preds - true_vals)
    
    # Scale values
    is_scaled = enable_asinh and (target in FLUX_TARGETS)
    plot_true = apply_scaling(true_vals, target) if is_scaled else true_vals
    plot_pred = apply_scaling(preds, target) if is_scaled else preds
    
    # Specific outlier filtering to prevent axis compression
    if target == 'HBETA_FLUX':
        outlier_mask = (plot_true < 8.0) & (plot_pred < 6.0)
        plot_true = plot_true[outlier_mask]
        plot_pred = plot_pred[outlier_mask]
        
    
    # Subsample for rendering performance
    if len(plot_true) > subsample_limit:
        idx = np.random.choice(len(plot_true), subsample_limit, replace=False)
        plot_true_sub = plot_true[idx]
        plot_pred_sub = plot_pred[idx]
    else:
        plot_true_sub = plot_true
        plot_pred_sub = plot_pred
        
    # Calculate residuals
    plot_res_sub = plot_pred_sub - plot_true_sub
    
    # Fast 2D Density Estimation using 2D Histogram
    counts, xedges, yedges = np.histogram2d(plot_true_sub, plot_pred_sub, bins=100)
    ix = np.clip(np.digitize(plot_true_sub, xedges) - 1, 0, len(xedges) - 2)
    iy = np.clip(np.digitize(plot_pred_sub, yedges) - 1, 0, len(yedges) - 2)
    density = counts[ix, iy]
    
    # Sort points by density to draw high density points on top
    sort_idx = np.argsort(density)
    plot_true_sub = plot_true_sub[sort_idx]
    plot_pred_sub = plot_pred_sub[sort_idx]
    plot_res_sub = plot_res_sub[sort_idx]
    density = density[sort_idx]
    
    # Render layout: Top (True vs Pred), Bottom (Residuals vs True)
    # Using a narrower, vertically elongated shape (6.0 x 10.0) for 1/3 page layouts
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(6.0, 10.0),
        sharex=True,
        gridspec_kw={'height_ratios': [1.2, 0.8]}
    )
    
    # Put grid lines behind points
    ax_top.set_axisbelow(True)
    ax_top.grid(linestyle=':', alpha=0.6)
    ax_bot.set_axisbelow(True)
    ax_bot.grid(linestyle=':', alpha=0.6)
    
    # Top scatter
    ax_top.scatter(
        plot_true_sub, plot_pred_sub,
        c=density, cmap='turbo', s=6, alpha=0.6, rasterized=True
    )
    
    # Bottom scatter
    ax_bot.scatter(
        plot_true_sub, plot_res_sub,
        c=density, cmap='turbo', s=6, alpha=0.6, rasterized=True
    )
    
    # Diagonal y=x reference line
    min_val = min(plot_true.min(), plot_pred.min())
    max_val = max(plot_true.max(), plot_pred.max())
    margin = (max_val - min_val) * 0.05
    min_lim = min_val - margin
    max_lim = max_val + margin
    
    ax_top.plot([min_lim, max_lim], [min_lim, max_lim], color='red', linestyle='--', linewidth=1.5, label='y=x')
    
    # Fit OLS line on scaled values
    try:
        m, c = np.polyfit(plot_true, plot_pred, 1)
        x_line = np.array([min_lim, max_lim])
        y_line = m * x_line + c
        ax_top.plot(x_line, y_line, color='darkorange', linewidth=2.5, label=f'OLS Fit (slope={m:.3f})')
    except Exception as e:
        print(f"[WARNING] OLS fit failed on Top panel: {e}")
        
    ax_top.set_xlim(min_lim, max_lim)
    ax_top.set_ylim(min_lim, max_lim)
    
    # Bottom panel: zero line
    ax_bot.axhline(0, color='red', linestyle='--', linewidth=1.5)
    
    # Bottom panel: OLS of residuals
    try:
        m_res, c_res = np.polyfit(plot_true, plot_pred - plot_true, 1)
        y_line_res = m_res * x_line + c_res
        ax_bot.plot(x_line, y_line_res, color='darkorange', linewidth=2.5)
    except Exception as e:
        print(f"[WARNING] OLS fit failed on Bottom panel: {e}")
        
    # Residual axis limits
    res_min, res_max = plot_res_sub.min(), plot_res_sub.max()
    res_margin = max(0.1, (res_max - res_min) * 0.1)
    ax_bot.set_ylim(res_min - res_margin, res_max + res_margin)
    
    # Labels & Styling
    target_label = get_label(target, is_scaled)
    ax_top.set_ylabel(rf"\textbf{{Predicted {target_label}}}", fontweight='bold', fontsize=20)
    ax_bot.set_ylabel(rf"\textbf{{Pred - True}}", fontweight='bold', fontsize=20)
    ax_bot.set_xlabel(rf"\textbf{{True {target_label}}}", fontweight='bold', fontsize=20)
    
    # Try looking up mean metrics in CSV
    r2_val, rmse_val, bias_val = None, None, None
    if results_df is not None:
        sub = results_df[
            (results_df['Target'].astype(str).str.lower() == target.lower()) &
            (results_df['Modality'].astype(str).str.lower() == mode.lower()) &
            (results_df['Probe'].astype(str).str.lower() == probe.lower())
        ]
        if not sub.empty:
            r2_val = sub.iloc[0]['R2']
            rmse_val = sub.iloc[0]['RMSE']
            bias_val = sub.iloc[0]['Bias']
            print(f"[INFO] Using CSV metrics for {target}/{mode}/{probe}: R2={r2_val:.4f}, RMSE={rmse_val:.4f}, Bias={bias_val:+.4f}")
            
    final_r2 = r2_val if r2_val is not None else r2
    final_rmse = rmse_val if rmse_val is not None else rmse
    final_bias = bias_val if bias_val is not None else bias
    
    # Metric Annotation box inside top axes
    metric_text = rf"\begin{{tabular}}{{ll}} " \
                  rf"$R^2$ & {final_r2:.4f} \\ " \
                  rf"$\mathrm{{RMSE}}$ & {final_rmse:.4f} \\ " \
                  rf"$\mathrm{{Bias}}$ & {final_bias:+.4f} " \
                  rf"\end{{tabular}}"
    ax_top.text(
        0.05, 0.95, metric_text, transform=ax_top.transAxes,
        verticalalignment='top', horizontalalignment='left',
        fontsize=18,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
    )
    
    ax_top.legend(loc='lower right', frameon=True, fontsize=18)
    
    # Enforce 4 spines visible
    for ax in [ax_top, ax_bot]:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        
    title_mode = mode.replace('_', r'\_')
    #ax_top.set_title(rf"\textbf{{{target_label} Predictions ({title_mode} | {probe.upper()})}}", fontweight='bold', pad=15, fontsize=18)
    
    plt.tight_layout()
    output_path = save_dir / f"preds_diagnostic_{target}_{mode}_{probe}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved diagnostic plot to: {output_path}")

def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if not pred_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {pred_dir}")
        
    # Load downstream results CSV for accurate metrics lookup
    csv_path = pred_dir.parent / "downstream_results.csv"
    results_df = None
    if csv_path.is_file():
        try:
            results_df = pd.read_csv(csv_path)
            print(f"[INFO] Loaded downstream metrics from: {csv_path}")
        except Exception as e:
            print(f"[WARNING] Could not load metrics CSV: {e}")
            
    # Scan prediction npz files
    npz_files = list(pred_dir.glob("preds_*.npz"))
    if not npz_files:
        print(f"[WARNING] No prediction `.npz` files found in {pred_dir}")
        return
        
    processed_count = 0
    for f in npz_files:
        target, mode, probe = parse_npz_filename(f.name)
        if target is None:
            continue
            
        # Skip classification targets since they don't have standard residuals
        if target in CLASSIFICATION_TARGETS:
            print(f"[INFO] Skipping classification target: {target} (file: {f.name})")
            continue
            
        # Verify if matches the filters
        if matches_list(target, args.targets) and matches_list(mode, args.modes) and matches_list(probe, args.probes):
            generate_diagnostic_plot(
                npz_path=f,
                target=target,
                mode=mode,
                probe=probe,
                save_dir=save_dir,
                subsample_limit=args.subsample,
                enable_asinh=not args.disable_asinh,
                results_df=results_df
            )
            processed_count += 1
            
    print(f"[INFO] Finished! Total plots generated: {processed_count}")

if __name__ == "__main__":
    main()
