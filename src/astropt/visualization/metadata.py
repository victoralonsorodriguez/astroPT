"""
AstroPT Visualization — FITS Metadata Card Generators.

Generates LaTeX-formatted metadata strings from a FITS-indexed DataFrame
for morphological and spectroscopic galaxy properties, and renders them
as styled text boxes on matplotlib axes.
"""

from typing import Optional, Tuple


def get_morphological_meta_str(tid: int, fits_indexed) -> Tuple[str, Optional[float]]:
    """
    Generates a LaTeX-formatted metadata string for morphological/physical properties.

    Args:
        tid: Target ID to look up.
        fits_indexed: pandas DataFrame indexed by target ID.

    Returns:
        Tuple of (formatted_string, z_value_or_None).
    """
    import pandas as pd

    if fits_indexed is not None and tid in fits_indexed.index:
        meta = fits_indexed.loc[tid]

        # Coordinates
        ra_val = meta.get('RA', None)
        dec_val = meta.get('DEC', None)
        coords_str = (f"${ra_val:.6f}^\\circ, {dec_val:.6f}^\\circ$"
                      if (ra_val is not None and not pd.isna(ra_val)) else "N/A")

        # Redshift
        z_val = meta.get('Z', meta.get('z', None))
        z_str = f"{z_val:.4f}" if (z_val is not None and not pd.isna(z_val)) else "N/A"

        # Stellar Mass
        logmstar_val = meta.get('LOGMSTAR', meta.get('logmstar', None))
        mstar = (f"$10^{{{logmstar_val:.2f}}} \\text{{ M}}_\\odot$"
                 if (logmstar_val is not None and not pd.isna(logmstar_val)) else "N/A")

        # Star Formation Rate
        logsfr_val = meta.get('LOGSFR', meta.get('logsfr', None))
        sfr = (f"${10**logsfr_val:.3f} \\text{{ M}}_\\odot/\\text{{yr}}$"
               if (logsfr_val is not None and not pd.isna(logsfr_val)) else "N/A")

        # Spectype
        spectype = str(meta.get('SPECTYPE', meta.get('spectype', 'N/A')))

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


def get_spectroscopic_meta_str(tid: int, fits_indexed) -> str:
    """
    Generates a LaTeX-formatted metadata string for spectroscopic properties.

    Args:
        tid: Target ID to look up.
        fits_indexed: pandas DataFrame indexed by target ID.

    Returns:
        Formatted string.
    """
    import pandas as pd

    if fits_indexed is not None and tid in fits_indexed.index:
        meta = fits_indexed.loc[tid]

        ha_f  = meta.get('HALPHA_FLUX', None)
        hb_f  = meta.get('HBETA_FLUX', None)
        oiii_f = meta.get('OIII_5007_FLUX', None)
        oii_f  = meta.get('OII_3726_FLUX', None)

        ha_ew  = meta.get('HALPHA_EW', None)
        hb_ew  = meta.get('HBETA_EW', None)
        oii_ew = meta.get('OII_3726_EW', None)

        ha_sig = meta.get('HALPHA_SIGMA', None)
        snr_r  = meta.get('SNR_SPEC_R', None)
        snr_z  = meta.get('SNR_SPEC_Z', None)

        def fmt_flux(v):
            return f"{v:.2f}" if (v is not None and not pd.isna(v)) else "N/A"

        def fmt_ew(v):
            return f"{v:.2f} \\AA" if (v is not None and not pd.isna(v)) else "N/A"

        def fmt_val(v, unit=""):
            return f"{v:.1f}{unit}" if (v is not None and not pd.isna(v)) else "N/A"

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


def render_metadata_card(ax, text: str, style: str = 'morphological', fontsize: int = 14):
    """
    Renders a metadata text box with consistent visual styling.

    Args:
        ax: matplotlib Axes to render into.
        text: LaTeX-formatted metadata string.
        style: 'morphological' (ivory card) or 'spectroscopic' (aliceblue card).
        fontsize: Font size for the text.
    """
    if style == 'morphological':
        fc, ec, title_color = "ivory", "darkgrey", "navy"
        title = r"\textbf{Physical and Morphological properties}"
        linespacing = 1.8
    else:
        fc, ec, title_color = "aliceblue", "steelblue", "steelblue"
        title = r"\textbf{Spectroscopic properties}"
        linespacing = 1.6

    ax.axis('off')
    ax.text(
        0.5, 0.95, text,
        transform=ax.transAxes,
        fontsize=fontsize,
        linespacing=linespacing,
        horizontalalignment='center',
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.8", fc=fc, alpha=0.95, ec=ec, lw=1.0),
    )
    ax.set_title(title, fontsize=16, fontweight='bold', color=title_color, pad=15)
