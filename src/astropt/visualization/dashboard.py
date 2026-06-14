"""
AstroPT Visualization — Dashboard Layouts.

Provides two reusable high-level layout functions that compose
images, metadata cards, and spectrum panels into full figures:

    reconstruction_dashboard — 3-row layout:
        Row 0:  [GT Image]  [Predicted Image]  [Residual Map]
        Row 1:  [Spectrum Blue Channel]   (full width)
        Row 2:  [Spectrum Red Channel]    (full width)

    profile_dashboard — 3-row layout:
        Row 0:  [Morphological Metadata]  [Galaxy Image]  [Spectroscopic Stats]
        Row 1:  [Spectrum Blue Channel]   (full width)
        Row 2:  [Spectrum Red Channel]    (full width)

Both functions return a ``plt.Figure`` — the caller decides whether to
save it as PNG, inject it into a ``PdfPages``, or display it.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Optional, List, Dict, Any

from astropt.visualization.spectra import plot_spectrum_panels
from astropt.visualization.metadata import render_metadata_card


# ──────────────────────────────────────────────────────────────────────
# Layout 1: Reconstruction Dashboard (dash_internal / dash_zero_shot)
# ──────────────────────────────────────────────────────────────────────

def reconstruction_dashboard(
    target_id: int,
    z_val: float,
    train_name: str,
    # --- Image panels ---
    rgb_gt: Optional[np.ndarray] = None,
    rgb_pred: Optional[np.ndarray] = None,
    res_map: Optional[np.ndarray] = None,
    has_image: bool = True,
    gt_title: str = "Real (Log Scale)",
    pred_title: str = "Reconstructed (Log Scale)",
    res_title: str = "Residuals (Physical)",
    # --- Spectrum panels ---
    wave_ang: Optional[np.ndarray] = None,
    spectrum_curves: Optional[List[Dict[str, Any]]] = None,
    has_spectra: bool = True,
    wl_range: Optional[tuple] = None,
    blue_title: str = "Spectrum (Blue Channel)",
    red_title: str = "Spectrum (Red Channel)",
    spectrum_xlabel: str = r"Wavelength [\AA]",
    legend_loc: str = "lower left",
    # --- Dashboard chrome ---
    dashboard_title: str = "AstroPT Reconstruction",
    figsize: tuple = (20, 14),
    res_percentile: float = 98,
) -> plt.Figure:
    """Creates a reconstruction dashboard and returns the Figure."""
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], wspace=0.1, hspace=0.3)

    fig.suptitle(
        rf"\textbf{{{dashboard_title} | ID: {target_id} | z={z_val:.3f}}}"
        + f"\n[{train_name}]",
        fontsize=22, y=0.96,
    )

    # ── Row 0: Images ──
    if has_image and rgb_gt is not None and rgb_pred is not None:
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(rgb_gt, origin='lower')
        ax1.set_title(rf"\textbf{{{gt_title}}}")
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(rgb_pred, origin='lower')
        ax2.set_title(rf"\textbf{{{pred_title}}}")
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[0, 2])
        vlim = np.percentile(np.abs(res_map), res_percentile) if res_map is not None else 1.0
        if vlim <= 0:
            vlim = 1.0
        im = ax3.imshow(res_map, origin='lower', cmap='seismic', vmin=-vlim, vmax=vlim)
        ax3.set_title(rf"\textbf{{{res_title}}}")
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="Flux Diff")

    # ── Rows 1-2: Spectra ──
    if has_spectra and wave_ang is not None and spectrum_curves:
        ax_blue = fig.add_subplot(gs[1, :])
        ax_red  = fig.add_subplot(gs[2, :])
        plot_spectrum_panels(
            ax_blue, ax_red, wave_ang, spectrum_curves,
            z=z_val, wl_range=wl_range,
            blue_title=blue_title,
            red_title=red_title,
            xlabel=spectrum_xlabel,
            legend_loc=legend_loc,
        )

    return fig


# ──────────────────────────────────────────────────────────────────────
# Layout 2: Profile Dashboard (similarity / anomalies / diffusion)
# ──────────────────────────────────────────────────────────────────────

def profile_dashboard(
    suptitle: str,
    # --- Metadata cards ---
    morphological_meta: Optional[str] = None,
    spectroscopic_meta: Optional[str] = None,
    meta_fontsize: int = 14,
    # --- Galaxy image ---
    rgb_image: Optional[np.ndarray] = None,
    # --- Spectrum panels ---
    wave_ang: Optional[np.ndarray] = None,
    spectrum_curves: Optional[List[Dict[str, Any]]] = None,
    z: Optional[float] = None,
    blue_title: str = "Spectrum (Blue Channel)",
    red_title: str = "Spectrum (Red Channel)",
    xlabel: str = r"Observed Wavelength [\AA]",
    show_legend: bool = True,
    legend_loc: str = "lower left",
    # --- Style ---
    figsize: tuple = (20, 14),
    dpi: int = 150,
) -> plt.Figure:
    """Creates a profile dashboard and returns the Figure."""
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(
        3, 3,
        height_ratios=[2, 1, 1],
        width_ratios=[1, 1.1, 0.9],
        hspace=0.4, wspace=0.3,
    )

    ax_meta = fig.add_subplot(gs[0, 0])
    ax_img  = fig.add_subplot(gs[0, 1])
    ax_stats = fig.add_subplot(gs[0, 2])
    ax_spec_blue = fig.add_subplot(gs[1, :])
    ax_spec_red  = fig.add_subplot(gs[2, :])

    # ── Metadata cards ──
    if morphological_meta is not None:
        render_metadata_card(ax_meta, morphological_meta, style='morphological', fontsize=meta_fontsize)
    else:
        ax_meta.axis('off')

    if spectroscopic_meta is not None:
        render_metadata_card(ax_stats, spectroscopic_meta, style='spectroscopic', fontsize=meta_fontsize)
    else:
        ax_stats.axis('off')

    # ── Galaxy image ──
    if rgb_image is not None:
        ax_img.imshow(rgb_image, origin='lower')
    else:
        ax_img.text(0.5, 0.5, "EuclidImage missing", ha='center', va='center', fontsize=12)
    ax_img.axis('off')

    # ── Spectrum panels ──
    if wave_ang is not None and spectrum_curves:
        plot_spectrum_panels(
            ax_spec_blue, ax_spec_red, wave_ang, spectrum_curves,
            z=z,
            blue_title=blue_title,
            red_title=red_title,
            xlabel=xlabel,
            show_legend=show_legend,
            legend_loc=legend_loc,
        )
    else:
        ax_spec_blue.text(0.5, 0.5, "DESISpectrum not found", ha='center', va='center', fontsize=12)
        ax_spec_red.axis('off')

    plt.suptitle(suptitle, fontsize=22, fontweight='bold', y=0.98)

    return fig
