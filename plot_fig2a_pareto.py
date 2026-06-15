#!/usr/bin/env python3
"""Manuscript Figure 2(a): early-stop speed-accuracy (Pareto) curve.

Backscatter-RCS RMS error versus the percentage of FDTD time steps retained for
the eps_r = 25 sphere -- plain truncation versus global-pole completion, both
relative to the full 16000-step run, with the solver-versus-Mie accuracy floor.
Plots only measured data read from the cached outputs of ``run_cutoff_sweep.py``;
no FDTD solver is invoked.

Inputs
------
paperB_cutoff_sweep.csv
    One row per cut, with columns ``cut_steps``, ``extrap_vs_full_RMS_dB``
    (completion error) and ``trunc_vs_full_RMS_dB`` (truncation error).
paperB_cutoff_sweep_spectra.npz (optional)
    Provides ``f_Hz``, ``full_FDTD``, ``mie`` for the solver-versus-Mie floor.

Outputs
-------
``<out>.pdf`` and ``<out>.png`` -- one IEEE-column figure. The floor value is
printed to stdout.

Usage
-----
    python plot_fig2a_pareto.py --out results/fig2a_pareto.pdf
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotting_utils import (set_ieee_style, to_db,
                            COLOR_COMPLETION, COLOR_TRUNCATION, COLOR_FLOOR)


def load_error_curve(csv_path: Path, n_full: int):
    """Read the error-versus-cut table.

    Parameters
    ----------
    csv_path : Path
        CSV written by ``run_cutoff_sweep.py``.
    n_full : int
        Full-run length in time steps, used to convert ``cut_steps`` to a
        retained-percentage abscissa.

    Returns
    -------
    retained_pct, completion_err_db, truncation_err_db : np.ndarray
        Sorted by ascending retained percentage.
    """
    retained_pct, completion_err, truncation_err = [], [], []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                cut = float(row["cut_steps"])
                completion = float(row["extrap_vs_full_RMS_dB"])
                truncation = float(row["trunc_vs_full_RMS_dB"])
            except (KeyError, ValueError):
                continue
            retained_pct.append(100.0 * cut / n_full)
            completion_err.append(completion)
            truncation_err.append(truncation)
    if not retained_pct:
        raise RuntimeError(f"No usable rows in {csv_path}.")
    order = np.argsort(retained_pct)
    return (np.array(retained_pct)[order], np.array(completion_err)[order],
            np.array(truncation_err)[order])


def solver_vs_mie_floor(npz_path: Path):
    """Return the band-RMS full-FDTD-versus-exact-Mie floor, or None if unavailable."""
    if not npz_path.exists():
        print(f"WARNING: {npz_path} not found; drawing the curve without the floor line.")
        return None
    data = np.load(npz_path)
    if not {"f_Hz", "mie", "full_FDTD"}.issubset(data.files):
        return None
    return float(np.sqrt(np.mean((to_db(data["full_FDTD"]) - to_db(data["mie"])) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Manuscript Fig. 2(a): early-stop Pareto curve.")
    parser.add_argument("--csv", default="paperB_cutoff_sweep.csv", type=Path)
    parser.add_argument("--npz", default="paperB_cutoff_sweep_spectra.npz", type=Path)
    parser.add_argument("--n-full", default=16000, type=int)
    parser.add_argument("--out", default="results/fig2a_pareto.pdf", type=Path)
    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(f"ERROR: {args.csv} not found. Run run_cutoff_sweep.py first.")

    retained_pct, completion_err, truncation_err = load_error_curve(args.csv, args.n_full)
    floor = solver_vs_mie_floor(args.npz)

    set_ieee_style()
    fig, ax = plt.subplots(figsize=(1.75, 1.45))
    ax.plot(retained_pct, truncation_err, color=COLOR_TRUNCATION, ls="--", marker="s",
            label="Truncation")
    ax.plot(retained_pct, completion_err, color=COLOR_COMPLETION, ls="-", marker="o",
            label="Completion")
    if floor is not None:
        ax.axhline(floor, color=COLOR_FLOOR, ls=":", lw=1.0,
                   label=f"Solver-vs-Mie floor ({floor:.2f} dB)")
        y_max = max(float(np.max(truncation_err)), floor) + 0.5
    else:
        y_max = float(np.max(truncation_err)) + 0.5

    ax.set_xlabel("Retained time steps (%)")
    ax.set_ylabel("RCS RMS error (dB)")
    ax.set_ylim(0.0, y_max)
    ax.margins(x=0.04)
    ax.legend(loc="lower left", handlelength=1.6, borderaxespad=0.3)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {args.out} and {args.out.with_suffix('.png')}")
    if floor is not None:
        print(f"solver-vs-Mie floor = {floor:.3f} dB")


if __name__ == "__main__":
    main()
