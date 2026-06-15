"""Matplotlib styling and decibel metrics shared by the figure scripts.

Provides the single-column IEEE figure style, the manuscript's colorblind-safe
curve palette, and the decibel-domain error metrics used throughout (RMS and
maximum of ``10*log10(sigma)`` differences). Importing this module changes
nothing until :func:`set_ieee_style` is called.
"""
from __future__ import annotations

import numpy as np
import matplotlib

# Okabe-Ito colorblind-safe palette mapped to the manuscript's curve roles.
COLOR_COMPLETION = "#0072B2"   # blue         : global-pole completion
COLOR_TRUNCATION = "#D55E00"   # vermillion   : plain truncation
COLOR_FULL = "#000000"         # black        : full FDTD reference
COLOR_MIE = "#009E73"          # bluish-green : exact Mie
COLOR_FLOOR = "#000000"        # black        : solver-vs-Mie floor


def set_ieee_style() -> None:
    """Configure Matplotlib for single-column IEEE figures (Times, 6-7 pt, TrueType)."""
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.5,
        "lines.linewidth": 1.0, "lines.markersize": 3.4, "axes.linewidth": 0.6,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "xtick.direction": "in", "ytick.direction": "in", "axes.labelpad": 2.0,
        "legend.frameon": False, "legend.handlelength": 1.6,
        "legend.handletextpad": 0.4, "legend.labelspacing": 0.3,
        "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def to_db(sigma) -> np.ndarray:
    """Convert linear RCS to decibels (``10*log10``), floored at 1e-30 to avoid -inf."""
    return 10.0 * np.log10(np.maximum(np.asarray(sigma, float), 1e-30))


def rms_db(a, b) -> float:
    """Root-mean-square of the decibel difference between two spectra."""
    return float(np.sqrt(np.mean((to_db(a) - to_db(b)) ** 2)))


def max_abs_db(a, b) -> float:
    """Maximum absolute decibel difference between two spectra."""
    return float(np.max(np.abs(to_db(a) - to_db(b))))
