import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Ordered list of regression targets (excluding classification ones)
TARGET_ORDER = [
    'Z', 'LOGMSTAR', 'LOGSFR', 
    'HALPHA_EW', 'HALPHA_FLUX', 'NII_6584_FLUX', 'OIII_5007_FLUX','OIII_5007_SIGMA', 'HBETA_FLUX',
    'flux_detection_total',
    'smoothness', 'gini',
    'sersic_sersic_vis_radius', 'sersic_sersic_vis_index', 'sersic_sersic_vis_axis_ratio',
    'has_spiral_arms_yes'
]

# LaTeX Label mapping for publication-quality plots
LATEX_TARGET_MAP = {
    'Z': r'$\bm{z}$',
    'LOGMSTAR': r'$\bm{\log(M_*)}$',
    'LOGSFR': r'$\bm{\log(\mathrm{SFR})}$',
    'flux_detection_total': r'$\bm{F_{\mathrm{total}}}$',
    'HALPHA_EW': r'$\bm{\mathrm{EW}(\mathrm{H}\alpha)}$',
    'HALPHA_FLUX': r'$\bm{F(\mathrm{H}\alpha)}$',
    'NII_6584_FLUX': r'$\bm{F([\mathrm{N\,II}]\,\lambda 6584)}$',
    'OIII_5007_FLUX': r'$\bm{F([\mathrm{O\,III}]\,\lambda 5007)}$',
    'OIII_5007_SIGMA': r'$\bm{\sigma{[\mathrm{O\,III}]}}$',
    'HBETA_FLUX': r'$\bm{F(\mathrm{H}\beta)}$',
    'smoothness': r'\textbf{Smoothness}',
    'gini': r'\textbf{Gini}',
    'sersic_sersic_vis_radius': r'$\bm{R_{\mathrm{eff}}}$',
    'sersic_sersic_vis_index': r'$\bm{n_{\mathrm{Sersic}}}$',
    'sersic_sersic_vis_axis_ratio': r'$\bm{b/a}$',
    'has_spiral_arms_yes': r'\textbf{Spiral arms}',
}

# Plotting Global Configuration (matching LaTeX style of dash_internal_reconstruction_samples.py)
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
    'axes.labelsize': 30,     
    'axes.titlesize': 30,     
    'xtick.labelsize': 25,   
    'ytick.labelsize': 30,
    'legend.fontsize': 20,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titlesize': 26,
    'figure.titleweight': 'bold',
})

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate R2 difference comparison bar plots.")
    parser.add_argument("--hybrid_csv", type=str, required=True, help="Path to main run results CSV (Run A).")
    parser.add_argument("--compare_csv", type=str, default=None, help="Path to comparison run results CSV (Run B).")
    parser.add_argument("--supervised_spectra_csv", type=str, default=None, help="Path to supervised spectra results CSV.")
    parser.add_argument("--supervised_images_csv", type=str, default=None, help="Path to supervised images results CSV.")
    parser.add_argument("--unimodal_spectra_csv", type=str, default=None, help="Path to unimodal spectra results CSV.")
    parser.add_argument("--unimodal_images_csv", type=str, default=None, help="Path to unimodal images results CSV.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory where generated plots will be saved.")
    return parser.parse_args()

def is_spectra(modality_str) -> bool:
    m = str(modality_str).lower()
    return 'spec' in m

def is_images(modality_str) -> bool:
    m = str(modality_str).lower()
    return 'image' in m or 'img' in m

def is_joint(modality_str) -> bool:
    m = str(modality_str).lower()
    return 'joint' in m

def get_best_r2_info(df: pd.DataFrame, target: str, modality_type: str):
    """Finds the maximum R2 value and its corresponding configuration label."""
    subset = df[(df['Target'].astype(str) == target) & (df['Task'].astype(str).str.lower() == 'regression')]
    if subset.empty:
        return None, None
    
    # Filter to only consider MLP probers for representation models, and keep supervised baselines
    subset = subset[subset['Probe'].str.upper().isin(['MLP', 'RESNET18', 'TRANSFORMER_BASELINE'])]
    if subset.empty:
        return None, None
    
    if modality_type == 'spectra':
        mask = subset['Modality'].apply(is_spectra)
    elif modality_type == 'images':
        mask = subset['Modality'].apply(is_images)
    elif modality_type == 'joint':
        mask = subset['Modality'].apply(is_joint)
    else:
        return None, None
        
    matching = subset[mask].copy()
    if matching.empty:
        return None, None
        
    matching['R2_num'] = pd.to_numeric(matching['R2'], errors='coerce')
    matching = matching.dropna(subset=['R2_num'])
    if matching.empty:
        return None, None
        
    # Get the row with the maximum R2
    best_row = matching.loc[matching['R2_num'].idxmax()]
    max_val = best_row['R2_num']
    
    modality = str(best_row['Modality'])
    probe = str(best_row['Probe'])
    
    # Extract joint type if joint
    joint_type = ""
    if modality_type == 'joint':
        if 'concat' in modality.lower():
            joint_type = "Concat"
        else:
            joint_type = "Mean"
            
    # Extract phase if present
    phase = ""
    if 'phase1' in modality.lower():
        phase = "P1"
    elif 'phase2' in modality.lower():
        phase = "P2"
        
    # Extract probe
    probe_label = ""
    if probe.upper() in ['KNN', 'MLP', 'LP']:
        probe_label = probe.upper()
    elif 'transformer' not in probe.lower() and 'resnet' not in probe.lower():
        probe_label = probe
        
    # Construct label suffix
    parts = []
    if joint_type:
        parts.append(joint_type)
    if phase:
        parts.append(phase)
    if probe_label:
        parts.append(probe_label)
    label = " - ".join(parts)
        
    return max_val, label

def generate_diff_plot(hybrid_df: pd.DataFrame, 
                       ref_spectra_df: pd.DataFrame, 
                       ref_images_df: pd.DataFrame, 
                       supervised_spectra_df: pd.DataFrame,
                       supervised_images_df: pd.DataFrame,
                       output_path: Path, 
                       title: str, 
                       ylabel: str,
                       ref_type: str = "supervised") -> None:
    targets = []
    diff_list = []
    colors = []
    ref_modalities = []
    
    spectra_targets = []
    images_targets = []
    
    for target in TARGET_ORDER:
        # Determine the primary reference modality dynamically based on whichever SUPERVISED reference performed best
        sup_spec, _ = get_best_r2_info(supervised_spectra_df, target, 'spectra') if supervised_spectra_df is not None else (None, None)
        sup_img, _ = get_best_r2_info(supervised_images_df, target, 'images') if supervised_images_df is not None else (None, None)
        
        if target in ['LOGSFR', 'LOGMSTAR']:
            primary_ref = 'spectra'
        elif sup_spec is not None and sup_img is not None:
            if sup_spec >= sup_img:
                primary_ref = 'spectra'
            else:
                primary_ref = 'images'
        elif sup_spec is not None:
            primary_ref = 'spectra'
        elif sup_img is not None:
            primary_ref = 'images'
        else:
            primary_ref = 'spectra' # default fallback
            
        if primary_ref == 'spectra':
            spectra_targets.append((target, primary_ref))
        else:
            images_targets.append((target, primary_ref))
            
    grouped_targets_with_ref = spectra_targets + images_targets
    
    for target, primary_ref in grouped_targets_with_ref:
        best_hybrid_joint, _ = get_best_r2_info(hybrid_df, target, 'joint')
            
        # Get the corresponding reference baseline value from the CURRENT reference df (supervised or unimodal)
        if primary_ref == 'spectra':
            ref_primary_val, _ = get_best_r2_info(ref_spectra_df, target, 'spectra') if ref_spectra_df is not None else (None, None)
        else:
            ref_primary_val, _ = get_best_r2_info(ref_images_df, target, 'images') if ref_images_df is not None else (None, None)
            
        if best_hybrid_joint is not None and ref_primary_val is not None:
            targets.append(target)
            diff = best_hybrid_joint - ref_primary_val
            diff_list.append(diff)
            ref_modalities.append(primary_ref)
            
            # Yellow if reference is images, blue if spectra
            if primary_ref == 'images':
                colors.append('#e1be30') # Yellow/Gold
            else:
                colors.append('#1f77b4') # Blue
                
    if not targets:
        print(f"[WARNING] No targets to plot for: {title}")
        return
        
    N = len(targets)
    
    # Wide single horizontal line plot for 18 targets
    fig, ax = plt.subplots(figsize=(18, 7.5))
    
    # Ensure grid lines are drawn behind the bars
    ax.set_axisbelow(True)
    
    x = np.arange(N)
    bars = ax.bar(x, diff_list, color=colors, edgecolor='black', width=0.6, alpha=0.9)
    
    # Horizontal line at 0
    ax.axhline(0, color='#333333', linewidth=1.5, linestyle='--')
    
    # Set y-limits with a margin
    ymin, ymax = min(diff_list), max(diff_list)
    margin = max(0.05, (ymax - ymin) * 0.25)
    ylim_min = min(ymin - margin, -0.05)
    ylim_max = max(ymax + margin, 0.05)
    ax.set_ylim(ylim_min, ylim_max)
    
    # Add values on top/below each bar (config labels removed per user request)
    for idx, (bar, val) in enumerate(zip(bars, diff_list)):
        if val >= 0:
            va_val = 'bottom'
            offset_val = (ylim_max - ylim_min) * 0.015
            y_pos_val = val + offset_val
        else:
            va_val = 'top'
            offset_val = (ylim_max - ylim_min) * 0.015
            y_pos_val = val - offset_val
            
        text = f"\\textbf{{{val:+.3f}}}"
        ax.text(idx, y_pos_val, text, ha='center', va=va_val, rotation=45, fontsize=17, color='#111111')
        
    # Set labels
    ax.set_ylabel(r'$\bm{\Delta R^2}$', fontsize=28)
    #ax.set_title(title, fontweight='bold', pad=20, fontsize=21)
    
    # Custom LaTeX X-axis labels
    x_labels = [LATEX_TARGET_MAP.get(t, r'\textbf{' + t.replace('_', r'\_') + '}') for t in targets]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=22)
    
    # Grid configuration (grid goes below bars because of set_axisbelow(True) above)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Show all 4 spines (borders)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    # Legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor='#1f77b4', alpha=0.9, edgecolor='black'),
        Patch(facecolor='#e1be30', alpha=0.9, edgecolor='black')
    ]
    if ref_type == "supervised":
        labels = [r'\textbf{Spectra Supervised Baseline}', r'\textbf{Images Supervised Baselines}']
    else:
        labels = [r'\textbf{Spectra SSL Unimodal}', r'\textbf{Images SSL Unimodal}']
    ax.legend(handles, labels, loc='upper right', frameon=True, fontsize=22)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved plot: {output_path}")

def generate_comparison_plot(run_a_df: pd.DataFrame, 
                             run_b_df: pd.DataFrame, 
                             output_path: Path, 
                             title: str, 
                             ylabel: str) -> None:
    targets = []
    diff_list = []
    colors = []
    
    for target in TARGET_ORDER:
        best_a_joint, _ = get_best_r2_info(run_a_df, target, 'joint')
        best_b_joint, _ = get_best_r2_info(run_b_df, target, 'joint')
        
        if best_a_joint is not None and best_b_joint is not None:
            targets.append(target)
            diff = best_a_joint - best_b_joint
            diff_list.append(diff)
            colors.append('#59a14f') # Sage green
            
    if not targets:
        print(f"[WARNING] No targets to plot for: {title}")
        return
        
    N = len(targets)
    fig, ax = plt.subplots(figsize=(18, 7.5))
    
    ax.set_axisbelow(True)
    
    x = np.arange(N)
    bars = ax.bar(x, diff_list, color=colors, edgecolor='black', width=0.6, alpha=0.9)
    
    # Horizontal reference line at 0
    ax.axhline(0, color='#333333', linewidth=1.5, linestyle='--')
    
    # Set y-limits with a margin
    ymin, ymax = min(diff_list), max(diff_list)
    margin = max(0.05, (ymax - ymin) * 0.25)
    ylim_min = min(ymin - margin, -0.05)
    ylim_max = max(ymax + margin, 0.05)
    ax.set_ylim(ylim_min, ylim_max)
    
    # Add values on top/below each bar (config labels removed)
    for idx, (bar, val) in enumerate(zip(bars, diff_list)):
        if val >= 0:
            va_val = 'bottom'
            offset_val = (ylim_max - ylim_min) * 0.015
            y_pos_val = val + offset_val
        else:
            va_val = 'top'
            offset_val = (ylim_max - ylim_min) * 0.015
            y_pos_val = val - offset_val
            
        text = f"\\textbf{{{val:+.3f}}}"
        ax.text(idx, y_pos_val, text, ha='center', va=va_val, rotation=45, fontsize=17, color='#111111')
        
    # Set labels and title
    ax.set_ylabel(r'$\bm{\Delta R^2}$', fontsize=28)
    #ax.set_title(title, fontweight='bold', pad=20, fontsize=21)
    
    # Custom LaTeX X-axis labels
    x_labels = [LATEX_TARGET_MAP.get(t, r'\textbf{' + t.replace('_', r'\_') + '}') for t in targets]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=22)
    
    # Grid configuration
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Show all 4 spines
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved plot: {output_path}")

def get_run_name(csv_path: str, default_name: str) -> str:
    import json
    path = Path(csv_path)
    try:
        run_dir = path.parents[3]
        name = run_dir.name
    except Exception:
        return default_name
        
    config_path = run_dir / "weights" / "config.json"
    if config_path.is_file():
        try:
            with open(config_path, 'r') as f:
                name = json.load(f).get("train_name", name)
        except Exception:
            pass
    return name

def main() -> None:
    args = parse_args()
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if args.compare_csv:
        # Comparison mode: Run A vs Run B
        run_a_df = pd.read_csv(args.hybrid_csv).dropna(subset=['Target', 'Task'])
        run_b_df = pd.read_csv(args.compare_csv).dropna(subset=['Target', 'Task'])
        
        run_a_name = get_run_name(args.hybrid_csv, "Run A")
        run_b_name = get_run_name(args.compare_csv, "Run B")
        
        def sanitize_filename(name: str) -> str:
            cleaned = re.sub(r'[\s/\\?%*:|"<>\[\]\(\)+]+', '_', name)
            return re.sub(r'_+', '_', cleaned).strip('_')
            
        run_a_clean = sanitize_filename(run_a_name)
        run_b_clean = sanitize_filename(run_b_name)
        filename = f"r2_diff_{run_a_clean}_vs_{run_b_clean}.png"
            
        title = f"AstroPT Comparison: ({run_a_name}) vs ({run_b_name})"
        ylabel = f"R2 Difference: ({run_a_name}) vs ({run_b_name})"
        
        generate_comparison_plot(
            run_a_df=run_a_df,
            run_b_df=run_b_df,
            output_path=save_dir / filename,
            title=title,
            ylabel=ylabel
        )
    else:
        # Standard mode: SSL vs Baselines (require other CSVs)
        missing = []
        for name in ["supervised_spectra_csv", "supervised_images_csv", "unimodal_spectra_csv", "unimodal_images_csv"]:
            if getattr(args, name) is None:
                missing.append(f"--{name}")
        if missing:
            raise ValueError(f"Standard mode requires the following arguments: {', '.join(missing)}")
            
        hybrid_df = pd.read_csv(args.hybrid_csv).dropna(subset=['Target', 'Task'])
        supervised_spectra_df = pd.read_csv(args.supervised_spectra_csv).dropna(subset=['Target', 'Task'])
        supervised_images_df = pd.read_csv(args.supervised_images_csv).dropna(subset=['Target', 'Task'])
        unimodal_spectra_df = pd.read_csv(args.unimodal_spectra_csv).dropna(subset=['Target', 'Task'])
        unimodal_images_df = pd.read_csv(args.unimodal_images_csv).dropna(subset=['Target', 'Task'])
        
        run_name = get_run_name(args.hybrid_csv, "AstroPT Hybrid (100M)")
        
        # Figure 1: SSL vs Supervised
        generate_diff_plot(
            hybrid_df=hybrid_df,
            ref_spectra_df=supervised_spectra_df,
            ref_images_df=supervised_images_df,
            supervised_spectra_df=supervised_spectra_df,
            supervised_images_df=supervised_images_df,
            output_path=save_dir / "r2_diff_supervised.png",
            title=f"{run_name}: R2 Difference vs Supervised Baselines",
            ylabel=f"R2 Difference ({run_name} - Supervised)",
            ref_type="supervised"
        )
        
        # Figure 2: SSL vs Unimodal
        generate_diff_plot(
            hybrid_df=hybrid_df,
            ref_spectra_df=unimodal_spectra_df,
            ref_images_df=unimodal_images_df,
            supervised_spectra_df=supervised_spectra_df,
            supervised_images_df=supervised_images_df,
            output_path=save_dir / "r2_diff_unimodal.png",
            title=f"{run_name}: R2 Difference vs Unimodal SSL Baselines",
            ylabel=f"R2 Difference ({run_name} - Unimodal)",
            ref_type="unimodal"
        )

if __name__ == "__main__":
    main()
