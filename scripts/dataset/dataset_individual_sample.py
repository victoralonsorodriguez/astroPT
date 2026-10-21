#!/usr/bin/env python3
"""
AstroPT-MM: Target Diagnostic Visual Inspector & Spectroscopic Overlay
=======================================================================
Author: Victor Alonso Rodriguez
Date: October 2026
Description:
    Generates publication-quality galaxy diagnostic dashboards (Figure 4.1 in Master Thesis).
    Renders false-color Euclid VIS+NISP composite stamps using Lupton asinh scaling alongside
    redshift-corrected DESI blue and red channel spectra with 1D collision-free line overlays.
    Supports querying by TargetID, RA/DEC coordinates, or external ID strings.

Usage:
    python scripts/dataset/dataset_individual_sample.py \\
        --target_ids 39633411307996619 \\
        --data_dir /path/to/processed_data_arrow \\
        --metadata_path /path/to/catalog_FILTERED.fits \\
        --output_dir /path/to/dashboards
"""

import os
import sys
import glob
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.table import Table
from datasets import load_from_disk
from astropy.coordinates import SkyCoord
import astropy.units as u

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("astropt_individual_target_inspector")

plt.rcParams['text.latex.preamble'] = r'''
            \usepackage{siunitx}
            \usepackage{amsmath} 
            \usepackage{times}
            \usepackage{bm}
            '''
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    'figure.dpi': 150,
    'savefig.bbox': 'tight'
})

MAIN_LINES = {
    r"Ly$\alpha$": 1216.0, 
    r"C IV": 1549.0, 
    r"C III]": 1908.7,  
    r"Mg II": 2798.0,
    r"[O\,II]": 3727.3, 
    r"[Ne\,III]": 3868.7, 
    r"Ca K": 3933.7, 
    r"Ca H": 3968.5,
    r"H$\delta$": 4102.0, 
    r"H$\gamma$": 4341.0, 
    r"H$\beta$": 4861.0,
    r"[O\,III] 4959": 4959.0, 
    r"[O\,III] 5007": 5007.0, 
    r"Mg I": 5175.0, 
    r"Na D": 5890.0,
    r"[N\,II] 6548": 6548.0, 
    r"H$\alpha$": 6563.0, 
    r"[N\,II] 6584": 6583.5, 
    r"[S\,II]": 6730.8
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AstroPT Premium Target Inspector")
    parser.add_argument("--target_ids", "-t", type=int, nargs="+", default=None)
    parser.add_argument("--ra", type=float, nargs="+", default=None)
    parser.add_argument("--dec", type=float, nargs="+", default=None)
    parser.add_argument("--query_id_str", type=str, nargs="+", default=None)
    parser.add_argument(
        "--lens_csv", type=str,
        default="/home/valonso/iac18_mhuertas_shared/valonso/lens/q1_discovery_engine_lens_catalog.csv"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"
    )
    parser.add_argument(
        "--data_dir_interpolated", type=str,
        default="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated"
    )
    parser.add_argument(
        "--metadata_path", type=str,
        default="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1_FILTERED.fits"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str,
        default="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/dataset_logs/dataset_individual_samples"
    )

    args = parser.parse_args()
    if args.target_ids is None and args.ra is None and args.query_id_str is None:
        parser.error("Provide --target_ids, --ra/--dec, or --query_id_str.")
    if (args.ra is None) != (args.dec is None):
        parser.error("--ra and --dec must be provided together.")
    if args.ra is not None and args.dec is not None and len(args.ra) != len(args.dec):
        parser.error("Number of RA and DEC values must match.")
    return args


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


def extract_raw_rgb(raw_record: dict) -> np.ndarray:
    vis_data = np.array(raw_record['image_vis'],    dtype=np.float32)
    y_data   = np.array(raw_record['image_nisp_y'], dtype=np.float32)
    j_data   = np.array(raw_record['image_nisp_j'], dtype=np.float32)

    try:
        h_val = raw_record.get('image_nisp_h', None)
        h_data = np.array(h_val, dtype=np.float32) if h_val is not None else np.zeros_like(vis_data)
    except (TypeError, ValueError):
        h_data = np.zeros_like(vis_data)

    tid = raw_record.get('targetid', 'Unknown')
    logger.info(f"TargetID Statistics for {tid}:")

    bands = [("VIS", vis_data), ("NISP Y", y_data), ("NISP J", j_data), ("NISP H", h_data)]
    for name, band in bands:
        clean = band[np.isfinite(band)]
        if len(clean) > 0 and np.any(band > 0):
            logger.info(
                f"  {name:<8} -> shape: {band.shape}, "
                f"min: {np.nanmin(band):.4f}, max: {np.nanmax(band):.4f}, mean: {np.nanmean(band):.4f}"
            )
        else:
            logger.info(f"  {name:<8} -> empty/all zeros")

    return make_lupton_composite(vis_data, y_data, j_data, h_data)


def plot_spectral_lines(ax, min_wl: float, max_wl: float, z: float,
                        wave_arr: np.ndarray = None, flux_arr: np.ndarray = None) -> None:
    if z is None or pd.isna(z):
        return

    trans = ax.get_xaxis_transform()

    lines_in_range = []
    for name, rest_wave in sorted(MAIN_LINES.items(), key=lambda x: x[1]):
        obs_wave = rest_wave * (1 + z)
        if min_wl < obs_wave < max_wl:
            lines_in_range.append((name, obs_wave))

    if not lines_in_range:
        return

    wl_span = max_wl - min_wl
    MIN_DX = 0.022 * wl_span  # Minimum horizontal separation in wavelength units

    # 1D relaxation layout to prevent label overlap horizontally
    text_x = [wave for name, wave in lines_in_range]
    for _ in range(50):
        for i in range(len(text_x) - 1):
            dist = text_x[i+1] - text_x[i]
            if dist < MIN_DX:
                overlap = MIN_DX - dist
                text_x[i] -= overlap / 2.0
                text_x[i+1] += overlap / 2.0

    for i, (name, obs_wave) in enumerate(lines_in_range):
        # Draw vertical line across the entire spectrum channel
        ax.axvline(obs_wave, color='crimson', linestyle='--', alpha=1.0, lw=1.2)

        # Draw a small connecting line from the spectrum top to the label position
        t_x = text_x[i]
        ax.plot([obs_wave, t_x], [1.0, 1.02], transform=trans,
                color='crimson', lw=1.0, alpha=0.6, clip_on=False)

        # Draw the label at a fixed vertical level, shifted horizontally to prevent overlap
        ax.text(t_x, 1.03, rf"\textbf{{{name}}}",
                transform=trans, rotation=45, va='bottom', ha='left',
                fontsize=16, color='crimson', fontweight='bold', clip_on=False)


def plot_spectrum_into_axes(fig, ax_blue, ax_red, raw_record: dict, tid: int, z_val: float) -> None:
    if raw_record.get('spectrum_flux') is not None:
        spec_gt  = np.array(raw_record['spectrum_flux']).flatten()
        wave_ang = np.array(raw_record['spectrum_wave']).flatten()

        w_min, w_max = wave_ang.min(), wave_ang.max()
        w_mid = (w_min + w_max) / 2

        ax_blue.plot(wave_ang, spec_gt, 'k-', lw=1.2, alpha=0.8)
        ax_blue.set_xlim(w_min, w_mid)
        ax_blue.tick_params(axis='both', which='major', labelsize=18)
        if z_val is not None:
            plot_spectral_lines(ax_blue, w_min, w_mid, z_val, wave_ang, spec_gt)

        ax_red.plot(wave_ang, spec_gt, 'k-', lw=1.2, alpha=0.8)
        ax_red.set_xlim(w_mid, w_max)
        ax_red.set_xlabel(r"\textbf{Observed Wavelength} [\AA]", fontsize=20, fontweight='bold')
        ax_red.tick_params(axis='both', which='major', labelsize=18)
        if z_val is not None:
            plot_spectral_lines(ax_red, w_mid, w_max, z_val, wave_ang, spec_gt)
    else:
        ax_blue.text(0.5, 0.5, "DESISpectrum not found", ha='center', va='center', fontsize=12)
        ax_red.axis('off')


def get_morphological_meta_str(tid: int, fits_indexed: pd.DataFrame, query_record: dict = None):
    if tid not in fits_indexed.index:
        return rf"\textbf{{ID}}: {tid}" + "\n(No metadata)", None

    meta = fits_indexed.loc[tid]

    def _safe(key):
        v = meta.get(key, None)
        return None if (v is None or (isinstance(v, float) and pd.isna(v))) else v

    euclid_id = None
    if query_record is not None:
        euclid_id = query_record.get('euclid_id', None)
        if hasattr(euclid_id, 'item'):  # In case it's a scalar array or tensor
            euclid_id = euclid_id.item()
            
    if euclid_id is None:
        euclid_id = _safe('EUCLID_ID')
    if euclid_id is None:
        euclid_id = _safe('euclid_id')

    object_id_str = f"{euclid_id}" if euclid_id is not None else "N/A"
    
    ra_val  = _safe('RA')
    dec_val = _safe('DEC')
    coords_str = (
        rf"${ra_val:.6f}^\circ, {dec_val:.6f}^\circ$"
        if ra_val is not None and dec_val is not None else "N/A"
    )

    z_val    = _safe('Z')
    z_str    = f"{z_val:.4f}" if z_val is not None else "N/A"

    logmstar = _safe('LOGMSTAR')
    mstar    = rf"$10^{{{logmstar:.2f}}} \text{{ M}}_\odot$" if logmstar is not None else "N/A"

    logsfr   = _safe('LOGSFR')
    sfr      = rf"${10**logsfr:.3f} \text{{ M}}_\odot/\text{{yr}}$" if logsfr is not None else "N/A"

    spectype  = str(meta.get('SPECTYPE', 'N/A'))
    sersic_n  = _safe('sersic_sersic_vis_index')
    sersic_re = _safe('sersic_sersic_vis_radius')
    ba        = _safe('sersic_sersic_vis_axis_ratio')
    vis_flux  = _safe('flux_vis_1fwhm_aper')

    meta_str = (
        rf"\textbf{{DESI TARGETID}}: {tid}" + "\n"
        rf"\textbf{{Euclid ID}}: {object_id_str}" + "\n"
        rf"\textbf{{Coords}}: {coords_str}" + "\n"
        rf"$\bm{{z}}$: ${z_str}$" + "\n"
        rf"$\bm{{M_{{*}}}}$: {mstar}" + "\n" 
        rf"$\bm{{\mathrm{{SFR}}}}$: {sfr}" + "\n"
        rf"\textbf{{Spectype}}: {spectype}" + "\n"
        rf"$\bm{{F(\mathrm{{VIS}})}}$: {f'${vis_flux:.2f}$' if vis_flux is not None else 'N/A'}" + "\n"
        rf"$\bm{{n_{{\mathrm{{Sersic}}}}}}$: {f'${sersic_n:.2f}$' if sersic_n is not None else 'N/A'}" + "\n"
        rf"$\bm{{R_{{\mathrm{{eff}}}}}}$: {f'${sersic_re:.3f}$' if sersic_re is not None else 'N/A'}" + "\n"
        rf"$\bm{{b/a}}$: {f'${ba:.3f}$' if ba is not None else 'N/A'}"
    )
    return meta_str, z_val


def get_spectroscopic_meta_str(tid: int, fits_indexed: pd.DataFrame) -> str:
    if tid not in fits_indexed.index:
        return r"\textbf{Spectra data missing}"

    meta = fits_indexed.loc[tid]

    def _safe(key):
        v = meta.get(key, None)
        return None if (v is None or (isinstance(v, float) and pd.isna(v))) else v

    def fmt_flux(v): return f"{v:.2f}"  if v is not None else "N/A"
    def fmt_ew(v):   return rf"{v:.2f} \text{{\AA}}" if v is not None else "N/A"
    def fmt_val(v, unit=""): return f"{v:.1f}{unit}" if v is not None else "N/A"

    return (
        r"\textbf{Spectral Line Fluxes}:" + "\n"
        rf"  - $\bm{{F(\mathrm{{H}}\alpha)}}$: ${fmt_flux(_safe('HALPHA_FLUX'))}$" + "\n"
        rf"  - $\bm{{F(\mathrm{{H}}\beta)}}$: ${fmt_flux(_safe('HBETA_FLUX'))}$" + "\n"
        rf"  - $\bm{{F([\mathrm{{O\,III}}]\,\lambda 5007)}}$: ${fmt_flux(_safe('OIII_5007_FLUX'))}$" + "\n"
        rf"  - $\bm{{F([\mathrm{{O\,II}}]\,\lambda 3726)}}$: ${fmt_flux(_safe('OII_3726_FLUX'))}$" + "\n\n"
        r"\textbf{Equivalent Widths}:" + "\n"
        rf"  - $\bm{{\mathrm{{EW}}(\mathrm{{H}}\alpha)}}$: ${fmt_ew(_safe('HALPHA_EW'))}$" + "\n"
        rf"  - $\bm{{\mathrm{{EW}}(\mathrm{{H}}\beta)}}$: ${fmt_ew(_safe('HBETA_EW'))}$" + "\n"
        rf"  - $\bm{{\mathrm{{EW}}([\mathrm{{O\,II}}]\,\lambda 3726)}}$: ${fmt_ew(_safe('OII_3726_EW'))}$"
    ).replace("_", "\\_")

def render_dashboard(
    tid: int,
    data: dict,
    fits_indexed: pd.DataFrame,
    output_dir: str,
    suffix: str
) -> None:
    logger.info(f"Rendering dashboard for TargetID {tid}{suffix}...")

    query_record = data["record"]
    query_rgb    = extract_raw_rgb(query_record)
    query_meta, query_z = get_morphological_meta_str(tid, fits_indexed, query_record)
    query_spec_meta     = get_spectroscopic_meta_str(tid, fits_indexed)

    fig = plt.figure(figsize=(20, 12), dpi=150)
    gs  = gridspec.GridSpec(
        3, 3,
        height_ratios=[2.4, 1.0, 1.0],
        width_ratios=[0.9, 1.4, 0.9],
        hspace=0.45,
        wspace=0.10
    )

    ax_meta      = fig.add_subplot(gs[0, 0])
    ax_img       = fig.add_subplot(gs[0, 1])
    ax_stats     = fig.add_subplot(gs[0, 2])
    ax_spec_blue = fig.add_subplot(gs[1, :])
    ax_spec_red  = fig.add_subplot(gs[2, :], sharey=ax_spec_blue)

    ax_meta.axis('off')
    ax_meta.text(
        0.5, 0.95, query_meta,
        transform=ax_meta.transAxes, fontsize=18, linespacing=2.0,
        horizontalalignment='center', verticalalignment='top',
        bbox=dict(boxstyle="round,pad=1.0", fc="ivory", alpha=0.95, ec="darkgrey", lw=1.0)
    )

    if query_rgb is not None:
        ax_img.imshow(query_rgb, origin='lower')
    else:
        ax_img.text(0.5, 0.5, "EuclidImage missing", ha='center', va='center')
    ax_img.axis('off')

    ax_stats.axis('off')
    ax_stats.text(
        0.5, 0.95, query_spec_meta,
        transform=ax_stats.transAxes, fontsize=18, linespacing=1.8,
        horizontalalignment='center', verticalalignment='top',
        bbox=dict(boxstyle="round,pad=1.0", fc="aliceblue", alpha=0.95, ec="steelblue", lw=1.0)
    )

    plot_spectrum_into_axes(fig, ax_spec_blue, ax_spec_red, query_record, tid, query_z)

    y_label_cgs = r"\textbf{Flux} [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ \text{\AA}$^{-1}$]"
    blue_pos = ax_spec_blue.get_position()
    red_pos  = ax_spec_red.get_position()
    y_mid = (blue_pos.y0 + red_pos.y1) / 2.0
    fig.text(0.075, y_mid, y_label_cgs, fontsize=20, fontweight='bold',
             rotation=90, va='center', ha='center')

    plt.subplots_adjust(top=0.98, bottom=0.08, left=0.12, right=0.96)

    save_path = os.path.join(output_dir, f"target_diagnostic_{tid}{suffix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Dashboard saved to: {save_path}")


def load_targets_from_arrow(
    data_dir: str,
    resolved_tids: list
) -> dict:
    targets_data = {}
    remaining_ids = set(resolved_tids)

    for split in ["train", "test"]:
        if not remaining_ids:
            break
        for path in sorted(glob.glob(os.path.join(data_dir, f"{split}_*"))):
            if not remaining_ids:
                break
            try:
                ds = load_from_disk(path)
                targetids_in_partition = np.array(ds["targetid"], dtype=np.int64)
                for tid in list(remaining_ids):
                    row_idx = np.where(targetids_in_partition == tid)[0]
                    if len(row_idx) > 0:
                        row = ds[int(row_idx[0])]
                        targets_data[tid] = {
                            "split": split,
                            "partition": os.path.basename(path),
                            "targetid": tid,
                            "redshift": row.get("redshift", 0.0),
                            "record": row
                        }
                        remaining_ids.remove(tid)
            except Exception as e:
                logger.error(f"Error reading partition {path}: {e}")

    for tid in remaining_ids:
        logger.warning(f"TargetID {tid} not found in {data_dir}.")

    return targets_data


def resolve_target_ids(args: argparse.Namespace, fits_indexed: pd.DataFrame) -> list:
    resolved_tids = list(args.target_ids) if args.target_ids else []

    ra_col  = next((c for c in fits_indexed.columns if c.upper() == 'RA'), None)
    dec_col = next((c for c in fits_indexed.columns if c.upper() == 'DEC'), None)
    catalog_coords = None

    if (args.ra is not None or args.query_id_str is not None) and ra_col and dec_col:
        catalog_coords = SkyCoord(
            ra=fits_indexed[ra_col].values * u.deg,
            dec=fits_indexed[dec_col].values * u.deg
        )

    if args.ra is not None and args.dec is not None:
        if not ra_col or not dec_col:
            logger.error("RA/DEC columns not found in FITS catalog.")
            sys.exit(1)
        for q_ra, q_dec in zip(args.ra, args.dec):
            query_coord = SkyCoord(ra=q_ra * u.deg, dec=q_dec * u.deg)
            idx, d2d, _ = query_coord.match_to_catalog_sky(catalog_coords)
            if d2d[0] < 2.0 * u.arcsec:
                matched_tid = fits_indexed.index[int(idx)]
                logger.info(
                    f"Matched ({q_ra}, {q_dec}) -> TargetID {matched_tid} "
                    f"(sep: {d2d[0].to(u.arcsec).value:.4f}\")"
                )
                resolved_tids.append(int(matched_tid))
            else:
                logger.error(
                    f"No match for ({q_ra}, {q_dec}) within 2\" "
                    f"(nearest: {d2d[0].to(u.arcsec).value:.2f}\")"
                )

    if args.query_id_str is not None:
        if not os.path.exists(args.lens_csv):
            logger.error(f"Lens catalog not found: {args.lens_csv}")
            sys.exit(1)
        if not ra_col or not dec_col:
            logger.error("RA/DEC columns not found in FITS catalog.")
            sys.exit(1)
        lens_df = pd.read_csv(args.lens_csv)
        for query_id in args.query_id_str:
            row = lens_df[lens_df['id_str'] == query_id]
            if len(row) == 0:
                logger.error(f"Lens ID {query_id} not found in {args.lens_csv}")
                continue
            q_ra  = float(row.iloc[0]['right_ascension'])
            q_dec = float(row.iloc[0]['declination'])
            logger.info(f"Resolved Lens ID {query_id} -> RA={q_ra}, DEC={q_dec}")
            query_coord = SkyCoord(ra=q_ra * u.deg, dec=q_dec * u.deg)
            idx, d2d, _ = query_coord.match_to_catalog_sky(catalog_coords)
            if d2d[0] < 2.0 * u.arcsec:
                matched_tid = fits_indexed.index[int(idx)]
                logger.info(f"Matched Lens ID {query_id} -> TargetID {matched_tid}")
                resolved_tids.append(int(matched_tid))
            else:
                logger.error(
                    f"No match for Lens ID {query_id} within 2\" "
                    f"(nearest: {d2d[0].to(u.arcsec).value:.2f}\")"
                )

    seen = set()
    return [x for x in resolved_tids if not (x in seen or seen.add(x))]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Reading FITS catalog from {args.metadata_path}...")
    fits_table = Table.read(args.metadata_path)
    fits_df    = fits_table.to_pandas()

    for col in fits_df.columns:
        if fits_df[col].dtype == object:
            try:
                sample = fits_df[col].dropna().iloc[0]
                if isinstance(sample, bytes):
                    fits_df[col] = fits_df[col].str.decode('utf-8')
            except (IndexError, AttributeError):
                pass

    target_id_col = 'TARGETID' if 'TARGETID' in fits_df.columns else 'targetid'
    fits_indexed  = fits_df.drop_duplicates(subset=[target_id_col]).set_index(target_id_col)

    resolved_tids = resolve_target_ids(args, fits_indexed)
    if not resolved_tids:
        logger.error("No valid target IDs resolved. Exiting.")
        sys.exit(1)

    runs = [
        (args.data_dir,              ""),
        (args.data_dir_interpolated, "_interpolated"),
    ]

    for current_data_dir, suffix in runs:
        if not os.path.exists(current_data_dir):
            logger.warning(f"Data directory not found, skipping: {current_data_dir}")
            continue

        logger.info(f"Processing: {current_data_dir} (suffix: '{suffix}')")
        targets_data = load_targets_from_arrow(current_data_dir, resolved_tids)

        if not targets_data:
            logger.warning(f"No records loaded from {current_data_dir}.")
            continue

        for tid, data in targets_data.items():
            render_dashboard(tid, data, fits_indexed, args.output_dir, suffix)


if __name__ == "__main__":
    main()