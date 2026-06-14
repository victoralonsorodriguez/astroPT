"""
AstroPT Visualization Toolkit.

Centralized plotting utilities shared across all analysis scripts.
Edit ``visualization_defaults.json`` (or set ASTROPT_VIS_CONFIG env var)
to change spectral lines, matplotlib settings, or RGB parameters
without touching Python code.

Usage::

    from astropt.visualization import setup_matplotlib, reconstruction_dashboard
    setup_matplotlib()  # applies global rcParams from JSON config
    fig = reconstruction_dashboard(...)
    fig.savefig("output.png", dpi=300, bbox_inches='tight')
"""

from astropt.visualization.config import (
    setup_matplotlib,
    load_config,
    get_config,
    get_spectral_lines,
)
from astropt.visualization.rgb import make_rgb_lupton, extract_raw_rgb
from astropt.visualization.spectra import plot_spectral_lines, plot_spectrum_panels
from astropt.visualization.metadata import (
    get_morphological_meta_str,
    get_spectroscopic_meta_str,
    render_metadata_card,
)
from astropt.visualization.dashboard import (
    reconstruction_dashboard,
    profile_dashboard,
)

__all__ = [
    "setup_matplotlib",
    "load_config",
    "get_config",
    "get_spectral_lines",
    "make_rgb_lupton",
    "extract_raw_rgb",
    "plot_spectral_lines",
    "plot_spectrum_panels",
    "get_morphological_meta_str",
    "get_spectroscopic_meta_str",
    "render_metadata_card",
    "reconstruction_dashboard",
    "profile_dashboard",
]
