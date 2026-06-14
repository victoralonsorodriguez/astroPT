import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Ordered list of regression targets (excluding classification ones)
TARGET_ORDER = [
    'Z', 'LOGMSTAR', 'LOGSFR', 'GR', 'flux_detection_total',
    'HALPHA_EW', 'HALPHA_FLUX', 'NII_6584_FLUX', 'OIII_5007_FLUX','OIII_5007_SIGMA', 'HBETA_FLUX',
    'smooth_or_featured_smooth', 'smoothness', 'gini',
    'sersic_sersic_vis_radius', 'sersic_sersic_vis_index', 'sersic_sersic_vis_axis_ratio',
    'has_spiral_arms_yes'
]

# Primary reference baselines are determined dynamically based on best unimodal performance.

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
                       output_path: Path, 
                       title: str, 
                       ylabel: str) -> None:
    targets = []
    diff_spectra_list = []
    diff_images_list = []
    diff_joint_list = []
    
    hybrid_spec_labels = []
    hybrid_img_labels = []
    hybrid_joint_labels = []
    primary_refs_detected = []
    
    for target in TARGET_ORDER:
        best_hybrid_spec, spec_lbl = get_best_r2_info(hybrid_df, target, 'spectra')
        best_hybrid_img, img_lbl = get_best_r2_info(hybrid_df, target, 'images')
        best_hybrid_joint, joint_lbl = get_best_r2_info(hybrid_df, target, 'joint')
        
        ref_spec, _ = get_best_r2_info(ref_spectra_df, target, 'spectra') if ref_spectra_df is not None else (None, None)
        ref_img, _ = get_best_r2_info(ref_images_df, target, 'images') if ref_images_df is not None else (None, None)
        
        # Determine the primary reference modality dynamically based on whichever reference performed best
        if ref_spec is not None and ref_img is not None:
            if ref_spec >= ref_img:
                primary_ref = 'spectra'
                ref_primary_val = ref_spec
            else:
                primary_ref = 'images'
                ref_primary_val = ref_img
        elif ref_spec is not None:
            primary_ref = 'spectra'
            ref_primary_val = ref_spec
        elif ref_img is not None:
            primary_ref = 'images'
            ref_primary_val = ref_img
        else:
            primary_ref = 'spectra'
            ref_primary_val = None
        
        if best_hybrid_spec is not None or best_hybrid_img is not None or best_hybrid_joint is not None:
            targets.append(target)
            
            diff_spec = (best_hybrid_spec - ref_spec) if (best_hybrid_spec is not None and ref_spec is not None) else None
            diff_img = (best_hybrid_img - ref_img) if (best_hybrid_img is not None and ref_img is not None) else None
            diff_joint = (best_hybrid_joint - ref_primary_val) if (best_hybrid_joint is not None and ref_primary_val is not None) else None
            
            diff_spectra_list.append(diff_spec)
            diff_images_list.append(diff_img)
            diff_joint_list.append(diff_joint)
            
            hybrid_spec_labels.append(spec_lbl)
            hybrid_img_labels.append(img_lbl)
            hybrid_joint_labels.append(joint_lbl)
            primary_refs_detected.append(primary_ref)
            
    if not targets:
        print(f"[WARNING] No targets to plot for: {title}")
        return
        
    N = len(targets)
    
    # Configure 2-row layout dynamically to be as balanced as possible
    if N > 8:
        n_cols = (N + 1) // 2
        n_rows = 2
    else:
        n_cols = N
        n_rows = 1
        
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 6.2 * n_rows), sharey=True)
    
    # Flatten axes list for unified 1D index mapping
    if n_rows > 1:
        axes_flat = axes.flatten()
    elif n_cols > 1:
        axes_flat = list(axes)
    else:
        axes_flat = [axes]
        
    # Find global min/max for y-axis limits to scale correctly
    all_vals = [v for v in diff_spectra_list + diff_images_list + diff_joint_list if v is not None]
    if all_vals:
        ymin = min(all_vals)
        ymax = max(all_vals)
        margin = max(0.05, (ymax - ymin) * 0.25)
        ylim_min = ymin - margin
        ylim_max = ymax + margin
    else:
        ylim_min, ylim_max = -0.2, 0.2
        
    ylim_min = min(ylim_min, -0.05)
    ylim_max = max(ylim_max, 0.05)
    
    # Premium colors: Blue (Spectra), Orange (Images), Green (Joint)
    color_spec = '#4e79a7'  # Slate Blue
    color_img = '#f28e2b'   # Coral Orange
    color_joint = '#59a14f' # Sage Green
    
    for idx, target in enumerate(targets):
        ax = axes_flat[idx]
        diff_spec = diff_spectra_list[idx]
        diff_img = diff_images_list[idx]
        diff_jnt = diff_joint_list[idx]
        
        spec_lbl = hybrid_spec_labels[idx] or ""
        img_lbl = hybrid_img_labels[idx] or ""
        jnt_lbl = hybrid_joint_labels[idx] or ""
        
        # Horizontal baseline line
        ax.axhline(0, color='#333333', linewidth=1.2, linestyle='--')
        
        bars = []
        # Draw Spectra bar (x = 0)
        if diff_spec is not None:
            ax.bar(0, diff_spec, color=color_spec, edgecolor='black', width=0.5, alpha=0.85)
            bars.append((0, diff_spec))
        else:
            ax.text(0, 0.0, 'N/A', ha='center', va='center', color='gray', fontsize=9.5, style='italic')
            
        # Draw Images bar (x = 1.4)
        if diff_img is not None:
            ax.bar(1.4, diff_img, color=color_img, edgecolor='black', width=0.5, alpha=0.85)
            bars.append((1.4, diff_img))
        else:
            ax.text(1.4, 0.0, 'N/A', ha='center', va='center', color='gray', fontsize=9.5, style='italic')
            
        # Draw Joint bar (x = 2.8)
        if diff_jnt is not None:
            ax.bar(2.8, diff_jnt, color=color_joint, edgecolor='black', width=0.5, alpha=0.85)
            bars.append((2.8, diff_jnt))
        else:
            ax.text(2.8, 0.0, 'N/A', ha='center', va='center', color='gray', fontsize=9.5, style='italic')
            
        # Add value labels above or below each bar
        for x_pos, val in bars:
            va = 'bottom' if val >= 0 else 'top'
            offset = (ylim_max - ylim_min) * 0.025
            text_y = val + offset if val >= 0 else val - offset
            ax.text(x_pos, text_y, f"{val:+.3f}", ha='center', va=va, fontsize=8.5, fontweight='bold', color='#222222')
            
        # Subplot customization
        ax.set_title(target, fontsize=12, fontweight='bold', pad=12)
        ax.set_xlim(-0.7, 3.5)
        ax.set_ylim(ylim_min, ylim_max)
        ax.set_xticks([0, 1.4, 2.8])
        
        # Stacked label strings without parentheses to reduce text width
        lbl_spec = f"Spectra\n{spec_lbl}" if spec_lbl else "Spectra"
        lbl_img = f"Images\n{img_lbl}" if img_lbl else "Images"
        
        # Resolve reference target to append (e.g., vs Spec or vs Img)
        primary_ref = primary_refs_detected[idx]
        vs_suffix = "vs Spec" if primary_ref == 'spectra' else "vs Img"
        lbl_jnt = f"Joint\n{jnt_lbl}\n{vs_suffix}" if jnt_lbl else f"Joint\n{vs_suffix}"
        
        ax.set_xticklabels([lbl_spec, lbl_img, lbl_jnt], fontsize=8.5, fontweight='medium')
        
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Only show y-axis marks and labels on the first column of each row
        if idx % n_cols > 0:
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', left=False)
        else:
            ax.tick_params(axis='y', labelsize=10.5)
            
    # Hide any unused axes in the grid
    for k in range(N, len(axes_flat)):
        axes_flat[k].set_axis_off()
        
    fig.supylabel(ylabel, fontsize=14, fontweight='bold', x=0.01)
    plt.suptitle(title, fontsize=17, fontweight='bold', y=0.98)
    
    # Legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=color_spec, alpha=0.85, edgecolor='black'),
        Patch(facecolor=color_img, alpha=0.85, edgecolor='black'),
        Patch(facecolor=color_joint, alpha=0.85, edgecolor='black')
    ]
    labels = ['Spectra (DESI)', 'Images (Euclid)', 'Joint (vs. Primary Modality Baseline)']
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=12, frameon=True, bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout(rect=[0.03, 0.08, 1, 0.94])
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
    diff_spectra_list = []
    diff_images_list = []
    diff_joint_list = []
    
    run_a_spec_labels = []
    run_a_img_labels = []
    run_a_joint_labels = []
    
    run_b_spec_labels = []
    run_b_img_labels = []
    run_b_joint_labels = []
    
    for target in TARGET_ORDER:
        best_a_spec, spec_lbl_a = get_best_r2_info(run_a_df, target, 'spectra')
        best_a_img, img_lbl_a = get_best_r2_info(run_a_df, target, 'images')
        best_a_joint, joint_lbl_a = get_best_r2_info(run_a_df, target, 'joint')
        
        best_b_spec, spec_lbl_b = get_best_r2_info(run_b_df, target, 'spectra')
        best_b_img, img_lbl_b = get_best_r2_info(run_b_df, target, 'images')
        best_b_joint, joint_lbl_b = get_best_r2_info(run_b_df, target, 'joint')
        
        if (best_a_spec is not None or best_a_img is not None or best_a_joint is not None) and \
           (best_b_spec is not None or best_b_img is not None or best_b_joint is not None):
            
            targets.append(target)
            
            diff_spec = (best_a_spec - best_b_spec) if (best_a_spec is not None and best_b_spec is not None) else None
            diff_img = (best_a_img - best_b_img) if (best_a_img is not None and best_b_img is not None) else None
            diff_joint = (best_a_joint - best_b_joint) if (best_a_joint is not None and best_b_joint is not None) else None
            
            diff_spectra_list.append(diff_spec)
            diff_images_list.append(diff_img)
            diff_joint_list.append(diff_joint)
            
            run_a_spec_labels.append(spec_lbl_a)
            run_a_img_labels.append(img_lbl_a)
            run_a_joint_labels.append(joint_lbl_a)
            
            run_b_spec_labels.append(spec_lbl_b)
            run_b_img_labels.append(img_lbl_b)
            run_b_joint_labels.append(joint_lbl_b)
            
    if not targets:
        print(f"[WARNING] No targets to plot for: {title}")
        return
        
    N = len(targets)
    
    # Configure 2-row layout dynamically to be as balanced as possible
    if N > 8:
        n_cols = (N + 1) // 2
        n_rows = 2
    else:
        n_cols = N
        n_rows = 1
        
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 6.2 * n_rows), sharey=True)
    
    # Flatten axes list for unified 1D index mapping
    if n_rows > 1:
        axes_flat = axes.flatten()
    elif n_cols > 1:
        axes_flat = list(axes)
    else:
        axes_flat = [axes]
        
    # Find global min/max for y-axis limits to scale correctly
    all_vals = [v for v in diff_spectra_list + diff_images_list + diff_joint_list if v is not None]
    if all_vals:
        ymin = min(all_vals)
        ymax = max(all_vals)
        margin = max(0.05, (ymax - ymin) * 0.25)
        ylim_min = ymin - margin
        ylim_max = ymax + margin
    else:
        ylim_min, ylim_max = -0.2, 0.2
        
    ylim_min = min(ylim_min, -0.05)
    ylim_max = max(ylim_max, 0.05)
    
    # Premium colors: Blue (Spectra), Orange (Images), Green (Joint)
    color_spec = '#4e79a7'  # Slate Blue
    color_img = '#f28e2b'   # Coral Orange
    color_joint = '#59a14f' # Sage Green
    
    for idx, target in enumerate(targets):
        ax = axes_flat[idx]
        diff_spec = diff_spectra_list[idx]
        diff_img = diff_images_list[idx]
        diff_jnt = diff_joint_list[idx]
        
        spec_lbl_a = run_a_spec_labels[idx] or ""
        img_lbl_a = run_a_img_labels[idx] or ""
        jnt_lbl_a = run_a_joint_labels[idx] or ""
        
        spec_lbl_b = run_b_spec_labels[idx] or ""
        img_lbl_b = run_b_img_labels[idx] or ""
        jnt_lbl_b = run_b_joint_labels[idx] or ""
        
        # Horizontal baseline line
        ax.axhline(0, color='#333333', linewidth=1.2, linestyle='--')
        
        bars = []
        # Draw Spectra bar (x = 0)
        if diff_spec is not None:
            ax.bar(0, diff_spec, color=color_spec, edgecolor='black', width=0.5, alpha=0.85)
            bars.append((0, diff_spec))
        else:
            ax.text(0, 0.0, 'N/A', ha='center', va='center', color='gray', fontsize=9.5, style='italic')
            
        # Draw Images bar (x = 1.4)
        if diff_img is not None:
            ax.bar(1.4, diff_img, color=color_img, edgecolor='black', width=0.5, alpha=0.85)
            bars.append((1.4, diff_img))
        else:
            ax.text(1.4, 0.0, 'N/A', ha='center', va='center', color='gray', fontsize=9.5, style='italic')
            
        # Draw Joint bar (x = 2.8)
        if diff_jnt is not None:
            ax.bar(2.8, diff_jnt, color=color_joint, edgecolor='black', width=0.5, alpha=0.85)
            bars.append((2.8, diff_jnt))
        else:
            ax.text(2.8, 0.0, 'N/A', ha='center', va='center', color='gray', fontsize=9.5, style='italic')
            
        # Add value labels above or below each bar
        for x_pos, val in bars:
            va = 'bottom' if val >= 0 else 'top'
            offset = (ylim_max - ylim_min) * 0.025
            text_y = val + offset if val >= 0 else val - offset
            ax.text(x_pos, text_y, f"{val:+.3f}", ha='center', va=va, fontsize=8.5, fontweight='bold', color='#222222')
            
        # Subplot customization
        ax.set_title(target, fontsize=12, fontweight='bold', pad=12)
        ax.set_xlim(-0.7, 3.5)
        ax.set_ylim(ylim_min, ylim_max)
        ax.set_xticks([0, 1.4, 2.8])
        
        # Stacked label strings without parentheses to reduce text width
        lbl_spec = f"Spectra\n{spec_lbl_a}\nvs {spec_lbl_b}" if (spec_lbl_a or spec_lbl_b) else "Spectra"
        lbl_img = f"Images\n{img_lbl_a}\nvs {img_lbl_b}" if (img_lbl_a or img_lbl_b) else "Images"
        lbl_jnt = f"Joint\n{jnt_lbl_a}\nvs {jnt_lbl_b}" if (jnt_lbl_a or jnt_lbl_b) else "Joint"
        
        ax.set_xticklabels([lbl_spec, lbl_img, lbl_jnt], fontsize=8.5, fontweight='medium')
        
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Only show y-axis marks and labels on the first column of each row
        if idx % n_cols > 0:
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', left=False)
        else:
            ax.tick_params(axis='y', labelsize=10.5)
            
    # Hide any unused axes in the grid
    for k in range(N, len(axes_flat)):
        axes_flat[k].set_axis_off()
        
    fig.supylabel(ylabel, fontsize=14, fontweight='bold', x=0.01)
    plt.suptitle(title, fontsize=17, fontweight='bold', y=0.98)
    
    # Legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=color_spec, alpha=0.85, edgecolor='black'),
        Patch(facecolor=color_img, alpha=0.85, edgecolor='black'),
        Patch(facecolor=color_joint, alpha=0.85, edgecolor='black')
    ]
    labels = ['Spectra (DESI)', 'Images (Euclid)', 'Joint (Multimodal)']
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=12, frameon=True, bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout(rect=[0.03, 0.08, 1, 0.94])
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
            
        title = f"AstroPT Comparison: ({run_a_name}) vs ({run_b_name})"
        ylabel = f"R2 Difference: ({run_a_name}) vs ({run_b_name})"
        
        generate_comparison_plot(
            run_a_df=run_a_df,
            run_b_df=run_b_df,
            output_path=save_dir / "r2_diff_compare_runs.png",
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
            output_path=save_dir / "r2_diff_supervised.png",
            title=f"{run_name}: R2 Difference vs Supervised Baselines",
            ylabel=f"R2 Difference ({run_name} - Supervised)"
        )
        
        # Figure 2: SSL vs Unimodal
        generate_diff_plot(
            hybrid_df=hybrid_df,
            ref_spectra_df=unimodal_spectra_df,
            ref_images_df=unimodal_images_df,
            output_path=save_dir / "r2_diff_unimodal.png",
            title=f"{run_name}: R2 Difference vs Unimodal SSL Baselines",
            ylabel=f"R2 Difference ({run_name} - Unimodal)"
        )

if __name__ == "__main__":
    main()
