#!/usr/bin/env python3
"""
plot_variance_supplement.py  --  AWPL (W2): uncertainty on the completion error.

One sphere run, then the 75%-completion error is measured under two kinds of
variation, to replace the point estimates with ranges/error bars:
  (i)  projection choice -- the leading PC plus K random projection vectors a
       (Eq. 2 says any generic a works; this quantifies how much the choice matters);
  (ii) parameter spread -- pencil order in {24,28,32} x window in {2400,3000}.
Reports mean +/- std over projections and min/max over parameters, and writes a
bar-with-error-bar figure.

Run on the solver machine (Numba/SciPy + local modules).
    python plot_variance_supplement.py --k-rand 12 --out Fig/fig_awpl_variance.pdf
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
N_FULL, CUT, T0 = 16000, 4000, 600


def fit_complete_error(full, Yall, rec, dt, wf, fsel, sel, sig_full, a, W, order):
    from fdtd_extrapolate import mpm_poles
    Yw = Yall[T0:T0 + W]
    q = Yw @ a
    nq = np.linalg.norm(q)
    if nq == 0 or not np.isfinite(nq):
        return np.nan
    z, _ = mpm_poles(q, order)
    z = z[np.isfinite(z) & (np.abs(z) > 0) & (np.abs(z) < 0.9999995)]
    if z.size < 2:
        return np.nan
    V = z[None, :] ** np.arange(W)[:, None]
    R = np.linalg.pinv(V) @ Yw
    comp = A.complete_series(full, z, R, CUT, N_FULL, t0=T0)
    sig = A.backscatter_spectrum(rec, comp, dt, wf, fsel, sel, N_FULL)
    return A.rms_db(sig, sig_full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-rand", type=int, default=12)
    ap.add_argument("--W", type=int, default=2400)
    ap.add_argument("--order", type=int, default=28)
    ap.add_argument("--out", default="fig_awpl_variance.pdf")
    args = ap.parse_args()

    dt, rec, wf, meta = A.run_sphere(EPS_R, 2.0, N_FULL, RADIUS_MM)
    print(f"grid {meta['grid']}, solver {meta['solver_s']:.1f} s", flush=True)
    fsel, sel = A.band_bins(dt)
    full, Yall = A.stack_record(rec)
    sig_full = A.backscatter_spectrum(rec, full, dt, wf, fsel, sel, N_FULL)
    K6 = Yall.shape[1]

    # (i) projection-choice variance: leading PC + K random projections
    rng = np.random.default_rng(0)
    a_lead = rng.normal(size=K6); a_lead /= np.linalg.norm(a_lead)
    Yw = Yall[T0:T0 + args.W]
    for _ in range(12):
        u = Yw @ a_lead; a_lead = Yw.T @ u; a_lead /= np.linalg.norm(a_lead)
    errs = [fit_complete_error(full, Yall, rec, dt, wf, fsel, sel, sig_full,
                               a_lead, args.W, args.order)]
    for k in range(args.k_rand):
        a = rng.normal(size=K6); a /= np.linalg.norm(a)
        errs.append(fit_complete_error(full, Yall, rec, dt, wf, fsel, sel, sig_full,
                                       a, args.W, args.order))
    errs = np.array(errs, float); errs = errs[np.isfinite(errs)]
    print(f"\nprojection-choice completion error: {errs.mean():.2f} +/- {errs.std():.2f} dB "
          f"(min {errs.min():.2f}, max {errs.max():.2f}, n={errs.size})")

    # (ii) parameter spread
    par = []
    for W in (2400, 3000):
        for order in (24, 28, 32):
            e = fit_complete_error(full, Yall, rec, dt, wf, fsel, sel, sig_full,
                                   a_lead, W, order)
            par.append((W, order, e)); print(f"  W={W}, M={order}: {e:.2f} dB")
    pe = np.array([p[2] for p in par], float); pe = pe[np.isfinite(pe)]
    print(f"parameter spread: {pe.min():.2f}--{pe.max():.2f} dB")

    A.set_ieee_style()
    fig, ax = plt.subplots(figsize=(A.FIG_W, A.FIG_H))
    ax.bar([0], [errs.mean()], 0.5, yerr=[errs.std()], capsize=4, color=A.C_COMP,
           label="projection choice")
    ax.bar([1], [pe.mean()], 0.5, yerr=[[pe.mean()-pe.min()], [pe.max()-pe.mean()]],
           capsize=4, color=A.C_TRUNC, label="param. (M, W)")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["projection", "M, W"])
    ax.set_ylabel("Completion error (dB)")
    ax.set_ylim(bottom=0.0); ax.legend(loc="upper left", fontsize=6)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out); fig.savefig(out.with_suffix(".png")); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
