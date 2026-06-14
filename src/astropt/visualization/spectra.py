"""
AstroPT Visualization — Spectral Line Annotation & Spectrum Panels.

Functions:
    plot_spectral_lines  — Annotates spectral lines on a single axis with
                           redshift-corrected positions and alternating label heights.
    plot_spectrum_panels — Plots one or more spectrum curves split into Blue / Red
                           channel panels across two provided axes.
"""

import numpy as np
from typing import Optional, List, Dict, Any

from astropt.visualization.config import get_config, get_spectral_lines


def plot_spectral_lines(
    ax,
    min_wl: float,
    max_wl: float,
    z: float,
    color: Optional[str] = None,
    linestyle: Optional[str] = None,
    alpha: Optional[float] = None,
    lw: Optional[float] = None,
    fontsize: Optional[int] = None,
    font_alpha: Optional[float] = None,
):
    """
    Annotates spectral lines on *ax* for all rest-frame lines whose
    redshifted wavelength falls within [min_wl, max_wl].

    All style parameters default to values in ``visualization_defaults.json``
    under the ``spectral_line_annotation`` section.
    """
    if z is None:
        return
    try:
        import pandas as pd
        if pd.isna(z):
            return
    except (ImportError, TypeError, ValueError):
        pass

    cfg = get_config().get("spectral_line_annotation", {})
    color      = color      if color      is not None else cfg.get("color", "royalblue")
    linestyle  = linestyle  if linestyle  is not None else cfg.get("linestyle", "--")
    alpha      = alpha      if alpha      is not None else cfg.get("line_alpha", 0.6)
    lw         = lw         if lw         is not None else cfg.get("line_width", 1)
    fontsize   = fontsize   if fontsize   is not None else cfg.get("fontsize", 12)
    font_alpha = font_alpha if font_alpha is not None else cfg.get("font_alpha", 1.0)

    spectral_lines = get_spectral_lines()

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    pos_high = y_min + (y_range * 0.95)
    pos_low  = y_min + (y_range * 0.25)

    sorted_lines = sorted(spectral_lines, key=lambda x: x[1])
    counter = 0

    for name, rest_wave in sorted_lines:
        obs_wave = rest_wave * (1 + z)
        if min_wl < obs_wave < max_wl:
            y_pos = pos_high if counter % 2 == 0 else pos_low
            ax.axvline(obs_wave, color=color, linestyle=linestyle, alpha=alpha, lw=lw)
            ax.text(
                obs_wave, y_pos,
                rf"\textbf{{{name}}}",
                rotation=90,
                color=color,
                va='top', ha='right',
                fontsize=fontsize,
                alpha=font_alpha,
                fontweight='bold',
            )
            counter += 1


def plot_spectrum_panels(
    ax_blue,
    ax_red,
    wave_ang: np.ndarray,
    curves: List[Dict[str, Any]],
    z: Optional[float] = None,
    wl_range: Optional[tuple] = None,
    blue_title: str = "Spectrum (Blue Channel)",
    red_title: str = "Spectrum (Red Channel)",
    xlabel: str = r"Wavelength [\AA]",
    show_legend: bool = True,
    legend_loc: str = "lower left",
):
    """
    Plots spectrum curves split into Blue and Red channels on two axes.

    Args:
        ax_blue, ax_red: matplotlib Axes for the two panels.
        wave_ang: Wavelength array in Angstroms.
        curves: List of dicts, each with keys:
            - "data":      np.ndarray of flux values (required)
            - "color":     str  (default "k")
            - "lw":        float (default 1)
            - "alpha":     float (default 0.8)
            - "label":     str or None (default None)
            - "linestyle": str  (default "-")
        z: Redshift for spectral line annotation. None disables annotations.
        wl_range: (min, max) tuple to zoom. None uses full range.
        blue_title, red_title: Panel titles.
        xlabel: Label for the x-axis (bottom panel only).
        show_legend: Whether to display a legend on the blue panel.
        legend_loc: Legend location string.
    """
    if wl_range:
        w_min, w_max = wl_range
    else:
        w_min, w_max = float(wave_ang.min()), float(wave_ang.max())
    w_mid = (w_min + w_max) / 2

    for panel_idx, (ax, start, end, title) in enumerate([
        (ax_blue, w_min, w_mid, blue_title),
        (ax_red,  w_mid, w_max, red_title),
    ]):
        for curve in curves:
            ax.plot(
                wave_ang,
                curve["data"],
                color=curve.get("color", "k"),
                lw=curve.get("lw", 1),
                alpha=curve.get("alpha", 0.8),
                label=curve.get("label", None),
                linestyle=curve.get("linestyle", "-"),
            )
        ax.set_xlim(start, end)
        ax.set_title(rf"\textbf{{{title}}}")
        ax.set_ylabel(r"Flux")
        if panel_idx == 1:
            ax.set_xlabel(xlabel)
        if panel_idx == 0 and show_legend:
            # Only add legend if any curve has a label
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(loc=legend_loc)
        if z is not None:
            plot_spectral_lines(ax, start, end, z)
