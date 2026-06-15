#!/usr/bin/env python3
"""Dispatcher that regenerates the manuscript figures by invoking the plot scripts.

Each figure has its own ``plot_*`` script (kept separate for readability); this
wrapper runs a selection of them and writes the outputs to ``results/``. Figures
1 and 2(a) rebuild in seconds from cached data; figures 2(b)-5 invoke the FDTD
solver and take minutes each (Numba required).

Usage
-----
    python generate_manuscript_figures.py --all
    python generate_manuscript_figures.py --cache-only          # Figs 1, 2(a)
    python generate_manuscript_figures.py --figures 1 3 5
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# figure id -> (plot script, output filename, runs_fdtd)
FIGURES = {
    "1":  ("plot_fig1_spectra.py",        "fig1_spectra.pdf",        True),
    "2a": ("plot_fig2a_pareto.py",        "fig2a_pareto.pdf",        True),
    "2b": ("plot_fig2b_convergence.py",   "fig2b_convergence.pdf",   True),
    "3":  ("plot_fig3_bistatic.py",       "fig3_bistatic.pdf",       True),
    "4":  ("plot_fig4_cylinder_poles.py", "fig4_cylinder_poles.pdf", True),
    "5":  ("plot_fig5_pole_scatter.py",   "fig5_pole_scatter.pdf",   True),
}
RESULTS_DIR = Path("results")


def run_figure(fig_id: str) -> int:
    """Invoke one plot script, writing its output into ``results/``. Returns the exit code."""
    script, out_name, runs_fdtd = FIGURES[fig_id]
    tag = "FDTD" if runs_fdtd else "plot"
    print(f"[Fig {fig_id}] {script} ({tag}) -> {RESULTS_DIR / out_name}", flush=True)
    return subprocess.call([sys.executable, script, "--out", str(RESULTS_DIR / out_name)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate manuscript figures.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Run all figures.")
    group.add_argument("--cache-only", action="store_true",
                       help="Run only the plot-from-data figures (1, 2a); their "
                            "Step-1 data files must already exist.")
    group.add_argument("--figures", nargs="+", metavar="ID", choices=list(FIGURES),
                       help="Run a subset, e.g. --figures 1 3 5.")
    args = parser.parse_args()

    if args.cache_only:
        selected = [k for k, v in FIGURES.items() if not v[2]]
    elif args.figures:
        selected = args.figures
    else:  # default and --all
        selected = list(FIGURES)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    failures = [fid for fid in selected if run_figure(fid) != 0]
    if failures:
        sys.exit(f"FAILED figures: {', '.join(failures)}")
    print(f"Done: {', '.join(selected)}")


if __name__ == "__main__":
    main()
