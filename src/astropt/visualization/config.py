"""
AstroPT Visualization — Configuration Loader.

Loads visualization defaults from a JSON file. Users can override settings by:
  1. Passing a custom path to load_config() / setup_matplotlib()
  2. Setting the ASTROPT_VIS_CONFIG environment variable to point to a custom JSON file

This avoids modifying Python source code for routine changes (spectral lines, fonts, etc.).
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "visualization_defaults.json"
_loaded_config: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads visualization config from JSON. Caches result globally."""
    global _loaded_config

    if config_path:
        path = Path(config_path)
    else:
        env_path = os.environ.get("ASTROPT_VIS_CONFIG")
        path = Path(env_path) if env_path else _DEFAULT_CONFIG_PATH

    with open(path, "r") as f:
        _loaded_config = json.load(f)
    return _loaded_config


def get_config() -> Dict[str, Any]:
    """Returns the currently loaded config, loading defaults if needed."""
    global _loaded_config
    if _loaded_config is None:
        return load_config()
    return _loaded_config


def get_spectral_lines() -> List[Tuple[str, float]]:
    """Returns spectral lines as a list of (name, wavelength_angstrom) tuples."""
    cfg = get_config()
    return [(line["name"], line["wavelength"]) for line in cfg.get("spectral_lines", [])]


def setup_matplotlib(config_path: Optional[str] = None, **overrides):
    """
    Applies global matplotlib configuration from the JSON config.

    Args:
        config_path: Optional path to a custom JSON config file.
        **overrides: Key-value pairs that override any matplotlib settings
                     from the JSON config (e.g., font_size=18).
    """
    import matplotlib.pyplot as plt

    cfg = load_config(config_path) if config_path else get_config()
    mpl_cfg = dict(cfg.get("matplotlib", {}))
    mpl_cfg.update(overrides)

    # LaTeX preamble (scientific units, bold math, etc.)
    plt.rcParams['text.latex.preamble'] = r'''
                \usepackage{siunitx}
                \usepackage{bm}
                \usepackage{amsmath} 
                \sisetup{
                detect-family,
                separate-uncertainty=true,
                output-decimal-marker={.},
                exponent-product=\cdot,
                inter-unit-product=\cdot,
                }
                \DeclareSIUnit{\cts}{cts}
                '''

    if mpl_cfg.get("usetex", True):
        plt.rc('text', usetex=True)
    plt.rc('font',
           family=mpl_cfg.get("font_family", "serif"),
           weight=mpl_cfg.get("font_weight", "bold"))

    plt.rcParams.update({
        'axes.grid':          mpl_cfg.get("axes_grid", True),
        'grid.alpha':         mpl_cfg.get("grid_alpha", 0.3),
        'lines.linewidth':    mpl_cfg.get("lines_linewidth", 2),
        'font.size':          mpl_cfg.get("font_size", 14),
        'axes.labelsize':     mpl_cfg.get("axes_labelsize", 17),
        'axes.titlesize':     mpl_cfg.get("axes_titlesize", 19),
        'xtick.labelsize':    mpl_cfg.get("xtick_labelsize", 15),
        'ytick.labelsize':    mpl_cfg.get("ytick_labelsize", 15),
        'legend.fontsize':    mpl_cfg.get("legend_fontsize", 14),
        'axes.labelweight':   mpl_cfg.get("axes_labelweight", "bold"),
        'axes.titleweight':   mpl_cfg.get("axes_titleweight", "bold"),
        'figure.titlesize':   mpl_cfg.get("figure_titlesize", 22),
        'figure.titleweight': mpl_cfg.get("figure_titleweight", "bold"),
    })
