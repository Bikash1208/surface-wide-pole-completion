#!/usr/bin/env python3
"""
plot_sensitivity_supplement.py  --  AWPL Fig. (4d/4e): parameter sensitivity.

One sphere FDTD run, then the global-pole completion is repeated while sweeping
each free parameter in turn (the others held at the nominal values):
  - pencil order M,
  - fit-window length W,
  - window start t0,
  - minimum quality factor Q_min.
For each setting it reports the 75%-early-stop completion error (RMS vs full)
and the selected-pole frequency, so a reader can see how fragile the 1.26 dB
number is (reviewer 4d/4e). Cheap: one FDTD run plus many MPM fits.

Run on the solver machine (Numba/SciPy).
    python plot_sensitivity_supplement.py
    python plot_sensitivity_supplement.py --out Fig/fig_awpl_sensitivity.pdf
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
N_FULL, CUT = 16000, 4000          # 75% removed
NOM = dict(M=28, W=2400, t0=600, qmin=3.0)
SWEEP = dict(
    M=[16, 20, 24, 28, 32, 40],
    W=[1200, 1800, 2400, 3000, 3600],
    t0=[400, 600, 800, 1000, 1200],
    qmin=[1.0, 2.0, 3.0, 5.0, 8.0],
)
LABEL = {"M": "Pencil order $M$", "W": "Window length $W$",
         "t0": "Window start $n_1$", "qmin": "$Q_{\\min}$"}


def completion_error(full, Yall, rec, dt, wf, fsel, sel, sig_full, M, W, t0, qmin):
    if CUT <= t0 + W:
        return np.nan, np.nan
    z, R, _, _ = A.global_pole_fit(Yall, t0=t0, w_fit=W, order=M)
    info = A.energy_select(z, R, dt, q_min=qmin)
    fsel_pole = info["f"][info["sel"]] / 1e9 if info else np.nan
    comp = A.complete_series(full, z, R, CUT, N_FULL, t0=t0)
    sig = A.backscatter_spectrum(rec, comp, dt, wf, fsel, sel, N_FULL)
    return A.rms_db(sig, sig_full), fsel_pole


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig_awpl_sensitivity.pdf")
    args = ap.parse_args()

    dt, rec, wf, meta = A.run_sphere(EPS_R, 2.0, N_FULL, RADIUS_MM)
    print(f"grid {meta['grid']}, solver {meta['solver_s']:.1f} s", flush=True)
    fsel, sel = A.band_bins(dt)
    full, Yall = A.stack_record(rec)
    sig_full = A.backscatter_spectrum(rec, full, dt, wf, fsel, sel, N_FULL)

    results = {}
    for key, vals in SWEEP.items():
        errs, poles = [], []
        for v in vals:
            p = dict(NOM)
            p[key] = v
            e, fp = completion_error(full, Yall, rec, dt, wf, fsel, sel, sig_full,
                                     int(p["M"]), int(p["W"]), int(p["t0"]), float(p["qmin"]))
            errs.append(e); poles.append(fp)
            print(f"  {key}={v}: comp-vs-full {e:.2f} dB, pole {fp:.4f} GHz", flush=True)
        results[key] = (vals, errs, poles)

    A.set_ieee_style()
    fig, axes = plt.subplots(4, 1, figsize=(A.FIG_W, 4.8))
    for ax, key in zip(axes.ravel(), ["M", "W", "t0", "qmin"]):
        vals, errs, _ = results[key]
        ax.plot(vals, errs, "o-", color=A.C_COMP)
        ax.axvline(NOM[key], color=A.C_FLOOR, ls=":", lw=0.8)   # nominal value
        ax.set_xlabel(LABEL[key])
        ax.set_ylabel("Completion error (dB)")
        ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); fig.savefig(out.with_suffix(".png")); plt.close(fig)

    print("\nnominal M=28, W=2400, t0=600, Qmin=3 (dotted line in each panel)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
