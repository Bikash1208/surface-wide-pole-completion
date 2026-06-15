#!/usr/bin/env python3
"""
plot_fig2b_convergence.py  --  AWPL Fig. (4a): grid convergence.

Sweeps the FDTD grid for the eps_r=25 sphere and, at each resolution, reports
  - solver-vs-Mie floor          (full FDTD vs exact Mie),
  - completion-vs-full           (75% early stop, completed vs full run),
  - completion-vs-Mie            (completed vs exact Mie),
all RMS over the band. Steps and windows scale ~1/dx so each grid covers the
same physical decay window. The figure shows whether the completion error stays
acceptable as the discretization floor shrinks (reviewer 4a).

WARNING: fine grids are expensive (grid ~ (1/dx)^3, steps ~ 1/dx). Default
sweep [2.0, 1.0, 0.75] mm is tractable; add 0.5 only if you have the time.

Run on the solver machine (Numba/SciPy).
    python plot_fig2b_convergence.py
    python plot_fig2b_convergence.py --dx 2.0,1.0,0.75,0.5 --out Fig/fig_awpl_convergence.pdf
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import awpl_pipeline as A

EPS_R, RADIUS_MM = 25.0, 24.0
N0, DX0 = 16000, 2.0          # reference steps at dx0
T0_0, WFIT_0 = 600, 2400      # reference window at dx0
REMOVE = 0.75                 # early-stop fraction


def parse_floats(s):
    return tuple(float(x) for x in s.split(",") if x.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=parse_floats, default=parse_floats("2.0,1.0,0.75"),
                    help="grid spacings (mm), comma-separated")
    ap.add_argument("--out", default="fig_awpl_convergence.pdf")
    args = ap.parse_args()

    rows = []
    for dx in args.dx:
        scale = DX0 / dx
        n_full = int(round(N0 * scale))
        t0 = int(round(T0_0 * scale))
        w_fit = int(round(WFIT_0 * scale))
        cut = int(round((1.0 - REMOVE) * n_full))
        if cut <= t0 + w_fit:
            print(f"dx={dx}: cut {cut} below fit window end {t0+w_fit}; skipping.")
            continue
        print(f"\n=== dx={dx} mm  (n_full={n_full}, cut={cut}, window=[{t0},{t0+w_fit}]) ===",
              flush=True)
        try:
            dt, rec, wf, meta = A.run_sphere(EPS_R, dx, n_full, RADIUS_MM)
        except Exception as exc:
            print(f"  FDTD failed at dx={dx}: {exc}")
            continue
        print(f"  grid {meta['grid']}, solver {meta['solver_s']:.1f} s", flush=True)
        fsel, sel = A.band_bins(dt)
        full, Yall = A.stack_record(rec)
        mie = A.mie_backscatter_band(fsel, EPS_R, RADIUS_MM * 1e-3)
        sig_full = A.backscatter_spectrum(rec, full, dt, wf, fsel, sel, n_full)
        floor = A.rms_db(sig_full, mie)

        z, R, _, _ = A.global_pole_fit(Yall, t0=t0, w_fit=w_fit)
        comp = A.complete_series(full, z, R, cut, n_full, t0=t0)
        sig_comp = A.backscatter_spectrum(rec, comp, dt, wf, fsel, sel, n_full)
        e_cf = A.rms_db(sig_comp, sig_full)
        e_cm = A.rms_db(sig_comp, mie)
        print(f"  floor(full-Mie)={floor:.2f} dB | comp-full={e_cf:.2f} dB | comp-Mie={e_cm:.2f} dB")
        rows.append((dx, meta["ng"], floor, e_cf, e_cm))

    if not rows:
        sys.exit("No grids completed.")

    dxs = np.array([r[0] for r in rows])
    floor = np.array([r[2] for r in rows])
    e_cf = np.array([r[3] for r in rows])
    e_cm = np.array([r[4] for r in rows])

    A.set_ieee_style()
    fig, ax = plt.subplots(figsize=(A.FIG_W, A.FIG_H))
    ax.plot(dxs, floor, "s--", color=A.C_FLOOR, label="Full FDTD vs Mie (floor)")
    ax.plot(dxs, e_cm, "o-", color=A.C_COMP, label="Completed vs Mie")
    ax.plot(dxs, e_cf, "^-", color=A.C_TRUNC, label="Completed vs full run")
    ax.set_xlabel("Grid spacing $\\Delta$ (mm)")
    ax.set_ylabel("RCS RMS error (dB)")
    ax.set_xlim(max(dxs) * 1.05, min(dxs) * 0.9)   # finer to the right
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best", handlelength=2.0)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); fig.savefig(out.with_suffix(".png")); plt.close(fig)

    print("\n dx(mm)  grid   floor  comp-full  comp-Mie")
    for r in rows:
        print(f" {r[0]:5.2f}  {r[1]:4d}  {r[2]:6.2f}  {r[3]:8.2f}  {r[4]:8.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
