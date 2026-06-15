#!/usr/bin/env python3
"""
run_spectrum_data.py

One eps_r=25 sphere run (N_TOTAL steps, full decay), then everything Phase 1.1 / 1.3 /
5.2 of the improvement plan needs:

  FIG. 3 (four curves, wideband backscatter vs frequency):
     Mie analytic | full FDTD record | truncated record (no extrapolation) |
     truncated + global-pole extrapolation -- for CUTS = 3000, 4000, 6000 steps.
     Saves paperB_fig3_data.npz + draft figures/paperB_fig3_draft.png, prints the
     RMS-error table (full-vs-Mie, extrap-vs-full, extrap-vs-Mie, trunc-vs-full,
     timestep saving) per cut -> the "safe early-stop region" sentence.

  TABLE I (timing, 3 repeats, machine spec printed):
     global path  : PC1 (power iteration) + one MPM + one Vandermonde residue GEMM
     per-node path: MPM per signal, timed on a 64-signal subset and scaled linearly
                    to all 6K signals (DISCLOSED as such -- full per-node timing at
                    this record size would take ~an hour and adds nothing).

Conventions: backscatter through the project's validated ntff_rcs.FarField; incident
normalization from the source waveform spectrum on the same FFT grid; energy-rule pole
band per Sec. II-C.
"""
import time
import platform
import numpy as np

EPS_R = 25.0
RC = 12; d_ = 2e-3; npml = 8; buffer = 4; HY_GAP = 6; tf_gap = 3
N_TOTAL = 16000
T0 = 600
CUTS = (3000, 4000, 6000)
ORDER = 28
W_FIT = 2400                    # fixed pole-fit window (validated extraction window)
NTH, NPH = 24, 48
BAND = (1.2e9, 2.6e9)
NFFT = 65536
N_SUB = 64                      # per-node timing subset (scaled, disclosed)
N_REP = 3

half = RC + HY_GAP + buffer + npml; NG = 2 * half + (2 * half) % 2; cc = NG // 2
center = np.array([cc, cc, cc]) * d_
lo, hi = cc - RC - tf_gap, cc + RC + tf_gap
Rsph = (RC + HY_GAP) * d_
COMPS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

def run_full():
    from fdtd_engine import FDTD
    from tfsf import PlaneWaveBox, gaussian_modulated
    from geometry import sphere_mask
    from spherical_recorder import SphericalHuygensRecorder
    sim = FDTD(NG, NG, NG, d_, d_, d_, npml=npml)
    sim.set_material(sphere_mask(sim, center, RC * d_), eps_r=EPS_R)
    sim.finalize_material()
    wf = gaussian_modulated(1.8e9, 1.4 * 1.8e9, sim.dt)
    src = PlaneWaveBox(sim, lo, hi, lo, hi, lo, hi, wf)
    rec = SphericalHuygensRecorder(sim, center, Rsph, NTH, NPH, n_steps=N_TOTAL)
    t0 = time.time()
    for n in range(N_TOTAL):
        sim.update_H(); src.correct_H(sim, n); sim.update_E(); src.correct_E(sim, n)
        rec.record(sim, n)
    t_solver = time.time() - t0
    print(f"FDTD {N_TOTAL} steps in {t_solver:.0f}s")
    return sim.dt, rec, wf, t_solver

def band_bins(dt):
    f = np.fft.rfftfreq(NFFT, dt)
    sel = np.where((f >= BAND[0]) & (f <= BAND[1]))[0]
    return f[sel], sel

def backscatter_spectrum(rec, series_dict, dt, wf, fsel, sel):
    """sigma_back(f) over the band from a (possibly spliced) time-series dict."""
    from ntff_rcs import FarField
    ph = {c: np.fft.rfft(series_dict[c], n=NFFT, axis=0)[sel].astype(np.complex64)
          for c in COMPS}                                     # (Nf, K) each
    wfs = np.array([wf(n) for n in range(N_TOTAL)])
    Einc = np.abs(np.fft.rfft(wfs, n=NFFT))[sel]
    out = np.zeros(len(fsel))
    for i, f0 in enumerate(fsel):
        ff = FarField(rec.x, rec.y, rec.z, rec.nx, rec.ny, rec.nz, rec.dS,
                      ph["Ex"][i], ph["Ey"][i], ph["Ez"][i],
                      ph["Hx"][i], ph["Hy"][i], ph["Hz"][i])
        out[i] = ff.backscatter(f0, Einc[i])
    return out

def global_pole_fit(Y, dt, win_lo, win_hi, timeit=False):
    """PC1 by power iteration + one MPM + one residue GEMM. Returns z, R, (t_pole, t_res)."""
    from fdtd_extrapolate import mpm_poles
    # Pole estimation needs only a short, fixed window; it must NOT grow with the
    # cut (a long-window Hankel SVD costs ~W^2.5 and destroys the speedup).
    win_hi = min(win_hi, win_lo + W_FIT)
    Yw = Y[win_lo:win_hi]
    t0 = time.time()
    v = np.random.default_rng(0).normal(size=Yw.shape[1])
    for _ in range(12):
        u = Yw @ v; v = Yw.T @ u; v /= np.linalg.norm(v)
    pc1 = Yw @ v
    z, _ = mpm_poles(pc1, ORDER)
    t_pole = time.time() - t0
    t0 = time.time()
    Vinv = np.linalg.pinv(z[None, :] ** np.arange(win_hi - win_lo)[:, None])
    R = Vinv @ Yw
    t_res = time.time() - t0
    return z, R, (t_pole, t_res)

def main():
    print("MACHINE:", platform.platform())
    print("        ", platform.processor() or "(cpu model: fill manually)",
          "| python", platform.python_version(), "| numpy", np.__version__)
    dt, rec, wf, t_solver = run_full()
    full = {c: np.asarray(rec.data[c], np.float32) for c in COMPS}
    Yall = np.concatenate([full[c] for c in COMPS], axis=1).astype(np.float64)  # (N, 6K)
    K6 = Yall.shape[1]
    fsel, sel = band_bins(dt)
    print(f"band bins: {len(fsel)} ({fsel[0]/1e9:.2f}-{fsel[-1]/1e9:.2f} GHz), 6K={K6}")

    # ---- reference curves ----
    from mie_sphere import mie_backscatter
    mie = np.array([mie_backscatter(EPS_R, RC * d_, f0) for f0 in fsel])
    sig_full = backscatter_spectrum(rec, full, dt, wf, fsel, sel)
    dB = lambda x: 10 * np.log10(np.maximum(x, 1e-30))
    rms = lambda a, b: np.sqrt(np.mean((dB(a) - dB(b)) ** 2))
    print(f"\nfull-FDTD vs Mie RMS over band: {rms(sig_full, mie):.2f} dB  (solver floor)")

    # ---- cuts: truncated and extrapolated ----
    results = {}
    data = dict(f=fsel, mie=mie, full=sig_full, t_solver=t_solver)
    for cut in CUTS:
        z, R, _ = global_pole_fit(Yall, dt, T0, cut)
        n_ext = np.arange(cut, N_TOTAL)
        ext = {}; trunc = {}
        base = (z[None, :] ** (n_ext - T0)[:, None])          # (Next, M)
        col = 0
        for c in COMPS:
            Kc = full[c].shape[1]
            tail = (base @ R[:, col:col + Kc]).real.astype(np.float32)
            s = full[c].copy(); s[cut:] = tail; ext[c] = s
            s2 = full[c].copy(); s2[cut:] = 0.0; trunc[c] = s2
            col += Kc
        sig_ext = backscatter_spectrum(rec, ext, dt, wf, fsel, sel)
        sig_tr = backscatter_spectrum(rec, trunc, dt, wf, fsel, sel)
        results[cut] = (rms(sig_ext, sig_full), rms(sig_ext, mie), rms(sig_tr, sig_full))
        data[f"ext_{cut}"] = sig_ext; data[f"trunc_{cut}"] = sig_tr
        print(f"cut={cut:5d} ({100*(1-cut/N_TOTAL):.0f}% steps saved): "
              f"extrap-vs-full {results[cut][0]:.2f} dB | extrap-vs-Mie {results[cut][1]:.2f} dB"
              f" | trunc-vs-full {results[cut][2]:.2f} dB")
    np.savez("paperB_fig3_data.npz", **data)

    # ---- TABLE I timing (3 repeats) ----
    from fdtd_extrapolate import mpm_poles
    cut = 4000
    tg_pole, tg_res, tn = [], [], []
    for rep in range(N_REP):
        _, _, (tp, tr) = global_pole_fit(Yall, dt, T0, cut)
        tg_pole.append(tp); tg_res.append(tr)
        idx = np.random.default_rng(rep).choice(K6, N_SUB, replace=False)
        wfit = min(cut, T0 + W_FIT)        # same fixed window as the global fit (fair)
        t0 = time.time()
        for j in idx:
            mpm_poles(Yall[T0:wfit, j], ORDER)
        tn.append((time.time() - t0) * K6 / N_SUB)
    print(f"\nTABLE I (pole-fit window [{T0}:{T0+W_FIT}], W={W_FIT}, M={ORDER}, 6K={K6}, {N_REP} repeats):")
    print(f"  global pole estimation : {np.mean(tg_pole):7.2f} +- {np.std(tg_pole):.2f} s")
    print(f"  global residue solve   : {np.mean(tg_res):7.2f} +- {np.std(tg_res):.2f} s")
    print(f"  global TOTAL           : {np.mean(tg_pole)+np.mean(tg_res):7.2f} s")
    print(f"  per-node MPM (scaled from {N_SUB} signals -- DISCLOSE in paper): "
          f"{np.mean(tn):7.1f} +- {np.std(tn):.1f} s")
    print(f"  speedup                : {np.mean(tn)/(np.mean(tg_pole)+np.mean(tg_res)):.0f}x")

    # ---- draft Fig. 3 ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, os
    cut = 4000
    plt.figure(figsize=(7, 4))
    plt.plot(fsel / 1e9, dB(mie), "k-", lw=2, label="Mie (exact)")
    plt.plot(fsel / 1e9, dB(sig_full), "-", color="tab:blue", lw=1.2,
             label=f"full FDTD ({N_TOTAL} steps)")
    plt.plot(fsel / 1e9, dB(data[f"trunc_{cut}"]), ":", color="tab:gray", lw=1.2,
             label=f"truncated at {cut} (no extrapolation)")
    plt.plot(fsel / 1e9, dB(data[f"ext_{cut}"]), "--", color="tab:red", lw=1.4,
             label="truncated + global-pole extrapolation")
    plt.xlabel("frequency (GHz)"); plt.ylabel("backscatter RCS (dBsm)")
    plt.legend(fontsize=8); plt.grid(True, ls=":", alpha=0.5); plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/paperB_fig3_draft.png", dpi=200)
    print("\nsaved paperB_fig3_data.npz + figures/paperB_fig3_draft.png")

if __name__ == "__main__":
    main()
