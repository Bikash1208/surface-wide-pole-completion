#!/usr/bin/env python3
"""
plot_fig5_pole_scatter.py  --  AWPL Fig. 2 (per-node vs global pole).

Pole estimates in the frequency-quality (f, Q) plane for the eps_r = 25 sphere:
  - gray cloud : poles from INDEPENDENT per-signal matrix-pencil fits,
  - filled marker : the surface-wide GLOBAL pole estimate,
  - star : the exact Mie pole,
  - crosses : the amplitude-selected and max-Q candidates (the artifacts the
              energy rule rejects).

The figure makes the novelty visual: per-signal fitting is not only slow, it is
fragile -- the estimates scatter -- while one surface-wide fit collapses them to
a single estimate near the exact Mie pole.

REQUIRES the local FDTD modules (fdtd_engine, tfsf, geometry,
spherical_recorder, fdtd_extrapolate) and compute_exact_mie_poles (for the exact pole).
Run from the folder that contains them. Numba/SciPy are needed, so run on the
machine that runs the solver (not in a no-Numba sandbox).

Per-signal MPM over all 6912 signals is the slow baseline by design; for the
figure a random subset (default 600) gives the same scatter at a fraction of
the cost. Use --n-sub 6912 to fit every signal.

Usage:
    python plot_fig5_pole_scatter.py
    python plot_fig5_pole_scatter.py --n-sub 800 --out Fig/fig_awpl_polescatter.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# geometry / processing -- identical to run_cutoff_sweep.py / compute_exact_mie_poles.py
EPS_R = 25.0
RC = 12; D = 2e-3; NPML = 8; BUFFER = 4; HY_GAP = 6; TF_GAP = 3
T0 = 600; WIN = 2400; ORDER = 28; NTH, NPH = 24, 48
BAND = (1.2e9, 2.6e9)
Q_MIN = 3.0

C_CLOUD  = "#9A9A9A"   # gray        : per-signal estimates
C_GLOBAL = "#0072B2"   # blue        : global-pole estimate
C_MIE    = "#000000"   # black star  : exact Mie pole
C_ART    = "#D55E00"   # vermillion  : amplitude / max-Q artifacts


def set_ieee_style() -> None:
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 6.5,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.0,
        "lines.linewidth": 1.1, "axes.linewidth": 0.6,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2, "axes.labelpad": 2.0,
        "xtick.direction": "in", "ytick.direction": "in",
        "legend.frameon": False, "savefig.dpi": 600,
        "legend.handlelength": 1.4, "legend.handletextpad": 0.4,
        "legend.labelspacing": 0.25,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def fq_of(z: np.ndarray, dt: float):
    s = np.log(z) / dt
    f = s.imag / (2.0 * np.pi)
    q = np.abs(s.imag) / (2.0 * np.abs(s.real) + 1e-30)
    return f, q


def run_record():
    """Full sphere FDTD; return the stacked window Y (W x 6K) and dt."""
    from fdtd_engine import FDTD
    from tfsf import PlaneWaveBox, gaussian_modulated
    from geometry import sphere_mask
    from spherical_recorder import SphericalHuygensRecorder

    half = RC + HY_GAP + BUFFER + NPML
    ng = 2 * half + (2 * half) % 2
    c = ng // 2
    center = np.array([c, c, c]) * D
    lo, hi = c - RC - TF_GAP, c + RC + TF_GAP
    rsph = (RC + HY_GAP) * D
    comps = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

    sim = FDTD(ng, ng, ng, D, D, D, npml=NPML)
    sim.set_material(sphere_mask(sim, center, RC * D), eps_r=EPS_R)
    sim.finalize_material()
    wf = gaussian_modulated(1.8e9, 1.4 * 1.8e9, sim.dt)
    src = PlaneWaveBox(sim, lo, hi, lo, hi, lo, hi, wf)
    n_steps = T0 + WIN + 200
    rec = SphericalHuygensRecorder(sim, center, rsph, NTH, NPH, n_steps=n_steps)
    for n in range(n_steps):
        sim.update_H(); src.correct_H(sim, n)
        sim.update_E(); src.correct_E(sim, n)
        rec.record(sim, n)
    Y = np.concatenate([np.asarray(rec.data[cc], float)[T0:T0 + WIN] for cc in comps], axis=1)
    return Y, float(sim.dt)


def per_signal_cloud(Y: np.ndarray, dt: float, n_sub: int, seed: int = 0):
    """One in-band pole per signal (its strongest), from independent fits."""
    from fdtd_extrapolate import mpm_poles
    rng = np.random.default_rng(seed)
    cols = np.arange(Y.shape[1])
    if n_sub < Y.shape[1]:
        cols = rng.choice(cols, size=n_sub, replace=False)
    fpts, qpts = [], []
    for k in cols:
        y = Y[:, k]
        if np.linalg.norm(y) == 0:
            continue
        try:
            z, R = mpm_poles(y, ORDER)
        except Exception:
            continue
        f, q = fq_of(z, dt)
        amp = np.abs(R)
        cand = [m for m in range(z.size)
                if BAND[0] < f[m] < BAND[1] and np.abs(z[m]) < 1.0 and q[m] >= Q_MIN]
        if not cand:
            continue
        m = max(cand, key=lambda j: amp[j])
        fpts.append(f[m] / 1e9); qpts.append(q[m])
    return np.array(fpts), np.array(qpts)


def global_estimate(Y: np.ndarray, dt: float):
    """Global pole set + the energy-selected pole and the two artifact picks."""
    from fdtd_extrapolate import mpm_poles
    U, S, _ = np.linalg.svd(Y, full_matrices=False)
    z, _ = mpm_poles(U[:, 0] * S[0], ORDER)
    V = z[None, :] ** np.arange(Y.shape[0])[:, None]
    R = np.linalg.pinv(V) @ Y                      # M x 6K
    f, q = fq_of(z, dt)
    amp = np.linalg.norm(R, axis=1)
    E = amp ** 2 / np.maximum(1.0 - np.abs(z) ** 2, 1e-12)
    band = [m for m in range(z.size)
            if BAND[0] < f[m] < BAND[1] and np.abs(z[m]) < 1.0]
    if not band:
        raise RuntimeError("No global candidate in band.")
    m_energy = max([m for m in band if q[m] >= Q_MIN] or band, key=lambda j: E[j])
    m_amp = max(band, key=lambda j: amp[j])
    m_q = max(band, key=lambda j: q[j])
    return (f[m_energy] / 1e9, q[m_energy]), (f[m_amp] / 1e9, q[m_amp]), (f[m_q] / 1e9, q[m_q])


def exact_mie_a1():
    """Exact electric-dipole (a1) Mie pole of the eps_r=25, 24 mm sphere."""
    import compute_exact_mie_poles as MP
    fa, _fb = MP.exact_table()[25.0]
    return fa.real / 1e9, abs(fa.real) / (2.0 * abs(fa.imag))


def main() -> None:
    ap = argparse.ArgumentParser(description="AWPL Fig. 2: per-node vs global pole scatter.")
    ap.add_argument("--n-sub", type=int, default=600, help="Per-signal fits for the cloud (<=6912).")
    ap.add_argument("--out", default="fig_awpl_polescatter.pdf")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        Y, dt = run_record()
    except Exception as exc:
        sys.exit(f"ERROR: FDTD run failed ({exc}). Run on the machine with the solver/Numba.")

    print(f"per-signal fits on {args.n_sub} of {Y.shape[1]} signals ...", flush=True)
    fc, qc = per_signal_cloud(Y, dt, args.n_sub, seed=args.seed)
    (fg, qg), (fa, qa), (fq_, qq) = global_estimate(Y, dt)
    fmie, qmie = exact_mie_a1()
    print(f"global f={fg:.4f} GHz Q={qg:.1f} | Mie f={fmie:.4f} GHz Q={qmie:.1f} "
          f"| cloud n={fc.size}")

    set_ieee_style()
    fig, ax = plt.subplots(figsize=(1.75, 1.45))
    ax.scatter(fc, qc, s=5, c=C_CLOUD, alpha=0.45, edgecolors="none",
               label="Per-signal fits", zorder=1)
    ax.scatter([fa], [qa], s=55, marker="x", c=C_ART, lw=1.4,
               label="Amplitude pick", zorder=3)
    ax.scatter([fq_], [qq], s=55, marker="+", c=C_ART, lw=1.4,
               label="Max-$Q$ pick", zorder=3)
    ax.scatter([fg], [qg], s=40, marker="o", facecolors=C_GLOBAL,
               edgecolors="white", lw=0.6, label="Global pole", zorder=4)
    ax.scatter([fmie], [qmie], s=90, marker="*", facecolors=C_MIE,
               edgecolors="white", lw=0.4, label="Exact Mie", zorder=5)

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Quality factor $Q$")
    ax.set_xlim(BAND[0] / 1e9, BAND[1] / 1e9)
    ax.set_yscale("log")          # log Q so the full scatter and the Q~15 collapse both show
    ax.legend(loc="upper right", handlelength=1.4, ncol=1, fontsize=6)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
