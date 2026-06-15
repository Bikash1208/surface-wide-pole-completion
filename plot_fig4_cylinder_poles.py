#!/usr/bin/env python3
"""
plot_fig4_cylinder_poles.py  --  AWPL Fig. (4c): cylinder pole spectrum.

Extracts the global poles of the finite dielectric cylinder and plots the
integrated residue energy E_m versus frequency (a stem plot), plus the (f, Q)
candidate plane colored by E_m. This shows whether a single high-Q pole
dominates the late-time tail -- the assumption behind the cylinder's 0.12 dB
completion (reviewer 4c). The energy-selected pole and its share of the total
in-band energy are reported.

Run on the solver machine (Numba/SciPy).
    python plot_fig4_cylinder_poles.py
    python plot_fig4_cylinder_poles.py --n-full 16000 --out Fig/fig_awpl_cylinder_poles.pdf
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import awpl_pipeline as A

T0, W_FIT = 600, 2400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-full", type=int, default=16000)
    ap.add_argument("--out", default="fig_awpl_cylinder_poles.pdf")
    args = ap.parse_args()

    dt, rec, wf, meta = A.run_cylinder(25.0, 2.0, args.n_full, 16.0, 40.0)
    print(f"cylinder grid {meta['grid']}, solver {meta['solver_s']:.1f} s", flush=True)
    _, Yall = A.stack_record(rec)
    z, R, _, _ = A.global_pole_fit(Yall, t0=T0, w_fit=W_FIT)
    f, q, amp, E = A.candidate_table(z, R, dt)

    band = [m for m in range(z.size) if A.BAND[0] < f[m] < A.BAND[1] and np.abs(z[m]) < 1.0]
    if not band:
        raise SystemExit("No in-band poles found for the cylinder.")
    band = np.array(band)
    Eb = E[band]
    order = band[np.argsort(-Eb)]
    sel = order[0]
    share = E[sel] / np.sum(E[band])
    print(f"dominant pole: f={f[sel]/1e9:.4f} GHz, Q={q[sel]:.1f}, "
          f"energy share in band = {100*share:.1f}%")
    if order.size > 1:
        s2 = order[1]
        print(f"second pole:   f={f[s2]/1e9:.4f} GHz, Q={q[s2]:.1f}, "
              f"ratio E1/E2 = {E[sel]/max(E[s2],1e-30):.1f}")

    A.set_ieee_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(A.FIG_W, A.FIG_H2))

    # (a) energy stem vs frequency
    fb = f[band] / 1e9
    En = E[band] / E[sel]
    ax1.stem(fb, En, linefmt="-", markerfmt="o", basefmt=" ")
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Integrated energy $E_m$ (norm.)")
    ax1.set_xlim(A.BAND[0] / 1e9, A.BAND[1] / 1e9)
    ax1.grid(True, ls=":", lw=0.4, alpha=0.5)
    ax1.set_title(f"(a) top pole: {100*share:.0f}% of band energy", fontsize=6.0)

    # (b) (f,Q) plane colored by E_m
    sc = ax2.scatter(f[band] / 1e9, q[band], c=np.log10(E[band] / E[band].max()),
                     cmap="viridis", s=24, edgecolors="k", linewidths=0.3)
    ax2.scatter([f[sel] / 1e9], [q[sel]], s=80, marker="*",
                facecolors="none", edgecolors=A.C_TRUNC, linewidths=1.2)
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("Quality factor $Q$")
    ax2.set_xlim(A.BAND[0] / 1e9, A.BAND[1] / 1e9)
    ax2.grid(True, ls=":", lw=0.4, alpha=0.5)
    cb = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cb.set_label("$\\log_{10}(E_m/E_{\\max})$", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    ax2.set_title("(b) candidate plane", fontsize=6.5)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out); fig.savefig(out.with_suffix(".png")); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
