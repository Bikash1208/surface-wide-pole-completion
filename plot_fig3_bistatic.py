#!/usr/bin/env python3
"""
plot_fig3_bistatic.py  --  AWPL Fig. (6): bistatic / multi-angle RCS.

Completion is checked over the full bistatic angular pattern, not just
backscatter. For the eps_r=25 sphere it computes sigma(theta) at the
electric-dipole resonance for the full run, the 75%-completed run, and exact
Mie, and reports the angular RMS error (completed vs full, completed vs Mie).
This answers reviewer 6: "RCS recovered" must rest on more than one direction.

Run on the solver machine (Numba/SciPy).
    python plot_fig3_bistatic.py
    python plot_fig3_bistatic.py --f-GHz 1.71 --out Fig/fig_awpl_bistatic.pdf
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import awpl_pipeline as A

EPS_R, RADIUS_MM = 25.0, 24.0
N_FULL, CUT, T0, W_FIT = 16000, 4000, 600, 2400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f-GHz", type=float, default=1.71, help="evaluation frequency (resonance)")
    ap.add_argument("--f2-GHz", type=float, default=2.2, help="off-resonance frequency")
    ap.add_argument("--n-theta", type=int, default=73, help="theta samples 0..180 deg")
    ap.add_argument("--out", default="fig_awpl_bistatic.pdf")
    args = ap.parse_args()

    thetas = np.linspace(np.deg2rad(2.0), np.deg2rad(178.0), args.n_theta)
    deg = np.rad2deg(thetas)

    dt, rec, wf, meta = A.run_sphere(EPS_R, 2.0, N_FULL, RADIUS_MM)
    print(f"grid {meta['grid']}, solver {meta['solver_s']:.1f} s", flush=True)
    full, Yall = A.stack_record(rec)
    z, R, _, _ = A.global_pole_fit(Yall, t0=T0, w_fit=W_FIT)
    comp = A.complete_series(full, z, R, CUT, N_FULL, t0=T0)

    A.set_ieee_style()
    fig, axes = plt.subplots(2, 1, figsize=(A.FIG_W, A.FIG_H2), sharex=True)
    panels = [(args.f_GHz * 1e9, "(a) resonance"), (args.f2_GHz * 1e9, "(b) off resonance")]
    for ax, (f0, tag) in zip(axes, panels):
        fbin, sig_full = A.bistatic_pattern(rec, full, dt, wf, f0, thetas, N_FULL)
        _, sig_comp = A.bistatic_pattern(rec, comp, dt, wf, f0, thetas, N_FULL)
        mie = A.mie_bistatic(thetas, EPS_R, RADIUS_MM * 1e-3, fbin, pol="E")
        e_cf = A.rms_db(sig_comp, sig_full); e_fm = A.rms_db(sig_full, mie)
        print(f"{tag}: f={fbin/1e9:.4f} GHz | completed-vs-full {e_cf:.2f} dB | "
              f"full-vs-Mie {e_fm:.2f} dB")
        ax.plot(deg, A.db10(mie), color=A.C_MIE, ls=(0, (1, 1)), lw=1.0, label="Exact Mie")
        ax.plot(deg, A.db10(sig_full), color=A.C_FULL, ls="-", lw=1.0, label="Full FDTD")
        ax.plot(deg, A.db10(sig_comp), color=A.C_COMP, ls="--", lw=1.1,
                label="Completed (75% removed)")
        ax.set_xlim(0, 180); ax.grid(True, ls=":", lw=0.4, alpha=0.5)
        ax.set_ylabel("RCS (dBsm)")
        ax.set_title(f"{tag}: $f={fbin/1e9:.3f}$ GHz, comp-vs-full {e_cf:.2f} dB",
                     fontsize=6.0)
    axes[1].set_xlabel("Scattering angle $\\theta$ (deg)")
    axes[0].legend(loc="lower center", ncol=1, fontsize=5.0)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out); fig.savefig(out.with_suffix(".png")); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
