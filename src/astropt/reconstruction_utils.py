"""
AstroPT — Image Reconstruction Utilities.

Shared helpers for reconstructing full-resolution images from the
flattened patch sequences produced by the AstroPT autoregressive model.

Functions:
    get_spiral_indices                      — Spiral → Raster index mapping.
    reconstruct_image_from_patches          — Un-patchify a (seq_len, patch_dim) array
                                              back to (C, H, W).
    denormalize                             — Reverse normalization to physical units.
    rebuild_full_sequence_from_teacher_forcing — Re-align model outputs with input lengths
                                              under teacher-forcing or causal shift.
"""

import logging
import numpy as np
import torch
from typing import Optional, Any

logger = logging.getLogger("AstroPT")


def get_spiral_indices(side_len: int) -> np.ndarray:
    """
    Generates indices to map Spiral → Raster order (inverse mapping).

    AstroPT images are serialized in a center-outwards spiral.
    This function returns an index array that un-does the spiral
    so patches can be laid out in standard raster (row-major) order.
    """
    layout = np.arange(side_len * side_len).reshape(side_len, side_len)
    spiral_indices = []

    while layout.size > 0:
        spiral_indices.append(layout[0])            # Top
        layout = layout[1:]
        if layout.size == 0:
            break
        spiral_indices.append(layout[:, -1])        # Right
        layout = layout[:, :-1]
        if layout.size == 0:
            break
        spiral_indices.append(layout[-1][::-1])     # Bottom
        layout = layout[:-1]
        if layout.size == 0:
            break
        spiral_indices.append(layout[:, 0][::-1])   # Left
        layout = layout[:, 1:]

    spiral_order = np.concatenate(spiral_indices)

    # Invert mapping: Spiral → Raster
    flat_indices = np.empty(side_len * side_len, dtype=int)
    flat_indices[spiral_order] = np.arange(side_len * side_len)
    final_indices = (side_len * side_len - 1) - flat_indices

    return final_indices


def reconstruct_image_from_patches(
    patch_sequence: np.ndarray,
    mod_config: Optional[Any] = None,
    apply_antispiral: bool = True,
) -> np.ndarray:
    """
    Reconstructs a (C, H, W) image from a flattened patch sequence.

    Args:
        patch_sequence: Shape (seq_len, patch_dim).
        mod_config: Modality config object with ``patch_size`` and ``input_size``.
                    If None, assumes 4 channels and infers patch size.
        apply_antispiral: If True, reorders patches from spiral to raster order.

    Returns:
        (C, H, W) numpy array.
    """
    seq_len, patch_dim = patch_sequence.shape

    if mod_config:
        p_size = mod_config.patch_size
        channels = mod_config.input_size // (p_size ** 2)
    else:
        channels = 4
        p_size = int(np.sqrt(patch_dim // channels))

    grid_side = int(np.sqrt(seq_len))
    if grid_side * grid_side != seq_len:
        logger.error(
            f"Cannot reconstruct: Sequence length {seq_len} is not a perfect square."
        )
        return np.zeros((channels, grid_side * p_size, grid_side * p_size))

    # Anti-Spiral (only if data was generated in spiral order)
    if apply_antispiral:
        spiral_indices = get_spiral_indices(grid_side)
        raster_patches = patch_sequence[spiral_indices]
    else:
        raster_patches = patch_sequence

    # Un-Patchify: (grid, grid, p, p, C) → (C, H, W)
    grid = raster_patches.reshape(grid_side, grid_side, p_size, p_size, channels)
    grid = grid.transpose(4, 0, 2, 1, 3)   # (C, Grid_H, P_H, Grid_W, P_W)
    image = grid.reshape(channels, grid_side * p_size, grid_side * p_size)

    return image


def denormalize(data: np.ndarray, method: str, scaler: float, const: float) -> np.ndarray:
    """
    Reverts normalization to return physical units.

    Args:
        data: Normalized data from the model.
        method: 'constant', 'asinh', or 'z_score'.
        scaler: Scale factor (used in asinh branch).
        const: Normalization constant (or softening parameter 'a').
    """
    if method == "asinh":
        # x_norm = asinh(x_phys / a) / C  →  x_phys = a * sinh(data * C)
        return scaler * np.sinh(data * const)

    elif method == "constant":
        # x_norm = x_phys / const  →  x_phys = x_norm * const
        return data * const

    elif method == "z_score":
        # Cannot reverse without mean/std; return as-is
        return data

    return data


def rebuild_full_sequence_from_teacher_forcing(
    input_tokens: torch.Tensor,
    pred_tokens: torch.Tensor,
    mode_name: str,
) -> torch.Tensor:
    """
    Rebuilds a full-length predicted sequence from model outputs under
    ``process_modes()``.

    ``process_modes()`` + model forward can produce three valid length relations:
      - ``pred_len == input_len + 1``: prediction is already a full sequence (no prepend)
      - ``pred_len == input_len``    : prepend the first input token
    """
    # Ensure input and pred have same ndim
    if input_tokens.ndim < pred_tokens.ndim:
        input_tokens = input_tokens.unsqueeze(-1)
    elif pred_tokens.ndim < input_tokens.ndim:
        pred_tokens = pred_tokens.unsqueeze(-1)

    input_len = int(input_tokens.shape[1])
    pred_len  = int(pred_tokens.shape[1])

    if pred_len == input_len:
        first_token = input_tokens[:, 0:1, ...]
        return torch.cat([first_token, pred_tokens[:, 1:, ...]], dim=1)

    if pred_len >= input_len - 1:
        diff = input_len - pred_len
        if diff > 0:
            start_token = input_tokens[:, 0:diff, ...]
            return torch.cat([start_token, pred_tokens], dim=1)
        else:
            return pred_tokens[:, :input_len, ...]

    # Fallback for Token Mixing (Interleaved Sparse Outputs)
    logger.warning(
        f"Unexpected lengths: {mode_name} input={input_len}, pred={pred_len}. Padding with zeros."
    )

    full_shape = list(pred_tokens.shape)
    full_shape[1] = input_len
    padded = torch.zeros(tuple(full_shape), device=pred_tokens.device, dtype=pred_tokens.dtype)

    actual_len = min(input_len, pred_len)
    padded[:, :actual_len, ...] = pred_tokens[:, :actual_len, ...]
    return padded
