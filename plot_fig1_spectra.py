#!/usr/bin/env python3
"""Manuscript Figure 1: wideband backscatter RCS overlay and per-frequency error.

For the eps_r = 25 sphere and the finite cylinder, plots the full-FDTD,
75 %-completed, and plain-truncated backscatter spectra (plus exact Mie for the
sphere), with the per-frequency decibel error beneath each panel. All inputs are
cached spectra produced by the ``run_*`` data scripts; no FDTD solver is invoked,
so this figure rebuilds in seconds. The band-RMS and maximum decibel errors
quoted in the text are printed to stdout.

Inputs
------
paperB_fig3_data.npz
    Sphere spectra (linear RCS [m^2]): ``f`` [Hz], ``full``, ``ext_4000``
    (completed at the 75 % cut), ``trunc_4000`` (truncated), ``mie`` (exact).
paperB_multibody_validation_spectra.npz
    Cylinder spectra: ``cylinder_f_Hz``, ``cylinder_full``,
    ``cylinder_completed``, ``cylinder_truncated``.

Outputs
-------
``<out>.pdf`` and ``<out>.png`` -- the 2x2 figure (overlays on top, errors below).

Usage
-----
    python plot_fig1_spectra.py --out results/fig1_spectra.pdf
    python plot_fig1_spectra.py --data-dir . --out results/fig1_spectra.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import BAND_HZ
from plotting_utils import (set_ieee_style, to_db, rms_db, max_abs_db,
                            COLOR_COMPLETION, COLOR_TRUNCATION, COLOR_FULL, COLOR_MIE)

SPHERE_NPZ = "paperB_fig3_data.npz"
CYLINDER_NPZ = "paperB_multibody_validation_spectra.npz"


def _in_band(freq_hz: np.ndarray) -> np.ndarray:
    """Boolean mask selecting FFT bins inside the manuscript evaluation band."""
    return (freq_hz >= BAND_HZ[0]) & (freq_hz <= BAND_HZ[1])


def _print_error_table(s_full, s_comp, s_trunc, c_full, c_comp, c_trunc) -> None:
    """Print the band-RMS and max decibel errors cited in the manuscript."""
    print("SPHERE   completed vs full: RMS %.2f dB, max %.2f dB"
          % (rms_db(s_comp, s_full), max_abs_db(s_comp, s_full)))
    print("SPHERE   truncated vs full: RMS %.2f dB, max %.2f dB"
          % (rms_db(s_trunc, s_full), max_abs_db(s_trunc, s_full)))
    print("CYLINDER completed vs full: RMS %.2f dB, max %.2f dB"
          % (rms_db(c_comp, c_full), max_abs_db(c_comp, c_full)))
    print("CYLINDER truncated vs full: RMS %.2f dB, max %.2f dB"
          % (rms_db(c_trunc, c_full), max_abs_db(c_trunc, c_full)))


def build_figure(data_dir: Path):
    """Load the cached spectra, print the error table, and return the Matplotlib figure."""
    sphere = np.load(data_dir / SPHERE_NPZ)
    cyl = np.load(data_dir / CYLINDER_NPZ)

    fs = sphere["f"]
    bs = _in_band(fs)
    s_full, s_comp, s_trunc, s_mie = (sphere["full"][bs], sphere["ext_4000"][bs],
                                      sphere["trunc_4000"][bs], sphere["mie"][bs])
    f_sphere = fs[bs] / 1e9

    fc = cyl["cylinder_f_Hz"]
    bc = _in_band(fc)
    c_full, c_comp, c_trunc = (cyl["cylinder_full"][bc], cyl["cylinder_completed"][bc],
                               cyl["cylinder_truncated"][bc])
    f_cyl = fc[bc] / 1e9

    _print_error_table(s_full, s_comp, s_trunc, c_full, c_comp, c_trunc)

    set_ieee_style()
    fig, ax = plt.subplots(2, 2, figsize=(3.5, 2.9), sharex="col",
                           gridspec_kw={"height_ratios": [2.0, 1.0],
                                        "hspace": 0.12, "wspace": 0.34})

    # Top row: spectral overlays.
    a = ax[0, 0]
    a.plot(f_sphere, to_db(s_full), color=COLOR_FULL, label="full FDTD")
    a.plot(f_sphere, to_db(s_comp), color=COLOR_COMPLETION, label="completed (75\\% cut)")
    a.plot(f_sphere, to_db(s_trunc), color=COLOR_TRUNCATION, ls="--", lw=0.9, label="truncated")
    a.plot(f_sphere, to_db(s_mie), color=COLOR_MIE, ls=":", lw=0.9, label="exact Mie")
    a.set_ylabel(r"$\sigma$ (dBsm)")
    a.set_title(r"(a) sphere, $\varepsilon_r=25$")
    a.legend(loc="lower right", ncol=1)

    b = ax[0, 1]
    b.plot(f_cyl, to_db(c_full), color=COLOR_FULL, label="full FDTD")
    b.plot(f_cyl, to_db(c_comp), color=COLOR_COMPLETION, label="completed (75\\% cut)")
    b.plot(f_cyl, to_db(c_trunc), color=COLOR_TRUNCATION, ls="--", lw=0.9, label="truncated")
    b.set_title(r"(b) cylinder, $Q\approx33$")
    b.legend(loc="lower right", ncol=1)

    # Bottom row: per-frequency decibel error relative to the full run.
    a2 = ax[1, 0]
    a2.axhline(0, color="0.6", lw=0.5)
    a2.plot(f_sphere, to_db(s_comp) - to_db(s_full), color=COLOR_COMPLETION)
    a2.plot(f_sphere, to_db(s_trunc) - to_db(s_full), color=COLOR_TRUNCATION, ls="--", lw=0.9)
    a2.set_ylabel(r"$\Delta\sigma$ (dB)")
    a2.set_xlabel("frequency (GHz)")

    b2 = ax[1, 1]
    b2.axhline(0, color="0.6", lw=0.5)
    b2.plot(f_cyl, to_db(c_comp) - to_db(c_full), color=COLOR_COMPLETION)
    b2.plot(f_cyl, to_db(c_trunc) - to_db(c_full), color=COLOR_TRUNCATION, ls="--", lw=0.9)
    b2.set_xlabel("frequency (GHz)")

    for row in ax:
        for axis in row:
            axis.margins(x=0.02)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Manuscript Fig. 1: spectra and per-frequency error.")
    parser.add_argument("--data-dir", default=".", type=Path,
                        help="Directory holding the cached .npz spectra (default: repo root).")
    parser.add_argument("--out", default="results/fig1_spectra.pdf", type=Path,
                        help="Output PDF path (a .png twin is written alongside).")
    args = parser.parse_args()

    fig = build_figure(args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {args.out} and {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
