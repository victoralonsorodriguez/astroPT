"""
AstroPT Visualization — Lupton RGB Image Composition.

Implements the asinh color-preserving scaling from:
    Lupton, Gunn, & Szalay (2004) 'Preparing Red-Green-Blue Images from CCD Data'

Functions:
    make_rgb_lupton  — Core asinh transfer on a (C, H, W) tensor.
    extract_raw_rgb  — Full pipeline: Arrow record → background subtract → channel
                       normalize → weighted RGB → Lupton false-color (H, W, 3).
"""

import numpy as np
from typing import Optional, List

from astropt.visualization.config import get_config


def make_rgb_lupton(
    image_tensor: np.ndarray,
    Q: Optional[float] = None,
    stretch: Optional[float] = None,
    m: float = 0.0,
) -> np.ndarray:
    """
    Lupton et al. (2004) asinh color-preserving RGB scaling.

    Args:
        image_tensor: (C, H, W) array — C=3 for RGB.
        Q: Softening parameter (controls linear-to-log transition).
        stretch: Linear scale factor.
        m: Minimum value (background) to subtract.

    Returns:
        (H, W, 3) array clipped to [0, 1].
    """
    cfg = get_config().get("rgb", {})
    if Q is None:
        Q = cfg.get("lupton_Q", 12.0)
    if stretch is None:
        stretch = cfg.get("lupton_stretch", 0.5)

    # Compute intensity and transfer function
    I = np.mean(image_tensor, axis=0)
    I = I - m
    I = np.maximum(I, 1e-10)

    f_I = np.arcsinh(Q * stretch * I) / Q
    scale_factor = f_I / I

    # Preserve color ratios
    rgb_out = image_tensor * scale_factor[np.newaxis, :, :]

    # Final normalisation
    max_rgb = np.percentile(rgb_out, 99.5)
    if max_rgb > 0:
        rgb_out = rgb_out / max_rgb

    rgb_out = np.clip(rgb_out, 0, 1)
    return rgb_out.transpose(1, 2, 0)   # (H, W, C)


def extract_raw_rgb(
    raw_record: dict,
    rgb_weights: Optional[List[float]] = None,
    Q: Optional[float] = None,
    stretch: Optional[float] = None,
) -> Optional[np.ndarray]:
    """
    Builds a false-color RGB image from a raw Arrow record containing
    Euclid VIS + NISP (H, J, Y) bands.

    Pipeline:
        1. Extract VIS, H, J, Y channels (zeros fallback for missing bands).
        2. Per-channel background subtraction (median).
        3. Per-channel percentile normalisation.
        4. Weighted channel mapping:  R = H·w0,  G = (J+Y)/2·w1,  B = VIS·w2
        5. Lupton asinh scaling.

    Returns:
        (H, W, 3) RGB array clipped to [0, 1], or None if VIS is missing.
    """
    cfg = get_config().get("rgb", {})
    if rgb_weights is None:
        rgb_weights = cfg.get("weights", [1.2, 1.3, 1.0])
    if Q is None:
        Q = cfg.get("lupton_Q", 12.0)
    if stretch is None:
        stretch = cfg.get("lupton_stretch", 0.5)
    bg_percentile = cfg.get("background_percentile", 50)
    clip_percentile = cfg.get("channel_clip_percentile", 99.5)

    # --- Safely extract channels ---
    def _get_raw(k):
        try:
            val = raw_record[k]
            return np.array(val if val is not None else [], dtype=np.float32)
        except (KeyError, TypeError):
            return np.array([], dtype=np.float32)

    vis = _get_raw('image_vis')
    y   = _get_raw('image_nisp_y')
    j   = _get_raw('image_nisp_j')
    h   = _get_raw('image_nisp_h')

    if vis.size == 0:
        return None

    h = h if h.size > 0 else np.zeros_like(vis)
    j = j if j.size > 0 else np.zeros_like(vis)
    y = y if y.size > 0 else np.zeros_like(vis)

    raw_stack = np.stack([vis, h, j, y], axis=0)            # (4, H, W)

    # Background subtraction
    bg_val = np.percentile(raw_stack, bg_percentile, axis=(1, 2), keepdims=True)
    raw_bg = raw_stack - bg_val

    # Per-channel normalisation
    raw_rgb_stack = []
    for c in range(raw_bg.shape[0]):
        v_max = np.percentile(np.abs(raw_bg[c]), clip_percentile)
        if v_max <= 0:
            v_max = 1.0
        r_ch = np.clip(raw_bg[c] / v_max, 0, 100)
        raw_rgb_stack.append(r_ch)
    raw_norm = np.stack(raw_rgb_stack)

    # Weighted channel mapping: R=H, G=(J+Y)/2, B=VIS
    vis_ch, h_ch, j_ch, y_ch = raw_norm[0], raw_norm[1], raw_norm[2], raw_norm[3]
    r = h_ch * rgb_weights[0]
    g = ((j_ch + y_ch) / 2.0) * rgb_weights[1]
    b = vis_ch * rgb_weights[2]

    rgb_input = np.stack([r, g, b], axis=0)
    return make_rgb_lupton(rgb_input, Q=Q, stretch=stretch)
