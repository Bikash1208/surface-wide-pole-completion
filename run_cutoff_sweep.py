#!/usr/bin/env python3
"""
run_cutoff_sweep.py

Cutoff sweep for the early-stopped FDTD + global-pole completion claim.

Default experiment:
    - solid dielectric sphere, eps_r=25
    - one full 16000-step FDTD run retained as the reference record
    - sweep Ncut = 4000, 3500, 3000, 2500, 2000, 1500
    - for each cutoff:
        * fit one global pole set on the window [T0, Ncut]
        * extrapolate the missing tail to N_FULL
        * compute backscatter RCS
        * report RMS dB errors vs full FDTD and Mie
        * report estimated speedup vs plain full FDTD
        * report selected high-energy pole frequency and Q

Timing convention:
    Full FDTD loop time is measured directly.
    Cut FDTD time is estimated by step fraction unless --measure-cut-times is used.
    Global-pole post-processing time is measured for each cutoff.

Run:
    python run_cutoff_sweep.py

Optional actual cut-loop timings:
    python run_cutoff_sweep.py --measure-cut-times

Outputs:
    paperB_cutoff_sweep.csv
    paperB_cutoff_sweep.png
    paperB_cutoff_sweep.json
    paperB_cutoff_sweep_spectra.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ---------------- constants matching the current Paper B scripts ----------------
EPS_R = 25.0
RC = 12
D = 2e-3
NPML = 8
BUFFER = 4
HY_GAP = 6
TF_GAP = 3
N_FULL = 16000
T0 = 600
ORDER = 28
W_FIT = 2400                    # fixed pole-fit window (must match paperB_fig3_timing)
NTH, NPH = 24, 48
BAND = (1.2e9, 2.6e9)
NFFT = 65536
COMPS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
DEFAULT_CUTS = (4000, 3500, 3000, 2500, 2000, 1500)


def parse_cuts(text: str) -> Tuple[int, ...]:
    cuts = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not cuts:
        raise ValueError("At least one cutoff is required.")
    for cut in cuts:
        if cut <= T0 + 2 * ORDER:
            raise ValueError(f"cut={cut} is too close to T0={T0} for ORDER={ORDER}.")
        if cut >= N_FULL:
            raise ValueError(f"cut={cut} must be smaller than N_FULL={N_FULL}.")
    return cuts


def grid_for_sphere() -> Tuple[int, int, np.ndarray, int, int, float]:
    half = RC + HY_GAP + BUFFER + NPML
    ng = 2 * half + (2 * half) % 2
    cc = ng // 2
    center = np.array([cc, cc, cc]) * D
    lo, hi = cc - RC - TF_GAP, cc + RC + TF_GAP
    rsph = (RC + HY_GAP) * D
    return ng, cc, center, lo, hi, rsph


def run_solid_sphere(n_steps: int, keep_record: bool = False):
    """Run the solid-sphere FDTD case. Setup is excluded from the timer."""
    from fdtd_engine import FDTD
    from geometry import sphere_mask
    from spherical_recorder import SphericalHuygensRecorder
    from tfsf import PlaneWaveBox, gaussian_modulated

    ng, cc, center, lo, hi, rsph = grid_for_sphere()

    sim = FDTD(ng, ng, ng, D, D, D, npml=NPML)
    sim.set_material(sphere_mask(sim, center, RC * D), eps_r=EPS_R)
    sim.finalize_material()

    wf = gaussian_modulated(1.8e9, 1.4 * 1.8e9, sim.dt)
    src = PlaneWaveBox(sim, lo, hi, lo, hi, lo, hi, wf)
    rec = SphericalHuygensRecorder(sim, center, rsph, NTH, NPH, n_steps=n_steps)

    t0 = time.perf_counter()
    for n in range(n_steps):
        sim.update_H()
        src.correct_H(sim, n)
        sim.update_E()
        src.correct_E(sim, n)
        rec.record(sim, n)
    elapsed = time.perf_counter() - t0

    if keep_record:
        return sim.dt, rec, wf, elapsed, f"{ng}^3"
    return elapsed, f"{ng}^3"


def band_bins(dt: float):
    f = np.fft.rfftfreq(NFFT, dt)
    sel = np.where((f >= BAND[0]) & (f <= BAND[1]))[0]
    return f[sel], sel


def db10(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x), 1e-30))


def rms_db(a, b) -> float:
    return float(np.sqrt(np.mean((db10(a) - db10(b)) ** 2)))


def backscatter_spectrum(rec, series_dict: Dict[str, np.ndarray], dt, wf, fsel, sel):
    """Backscatter RCS spectrum over BAND from a time-series dictionary."""
    from ntff_rcs import FarField

    ph = {
        c: np.fft.rfft(series_dict[c], n=NFFT, axis=0)[sel].astype(np.complex64)
        for c in COMPS
    }
    wfs = np.array([wf(n) for n in range(N_FULL)])
    einc = np.abs(np.fft.rfft(wfs, n=NFFT))[sel]

    out = np.zeros(len(fsel), dtype=float)
    for i, f0 in enumerate(fsel):
        ff = FarField(
            rec.x,
            rec.y,
            rec.z,
            rec.nx,
            rec.ny,
            rec.nz,
            rec.dS,
            ph["Ex"][i],
            ph["Ey"][i],
            ph["Ez"][i],
            ph["Hx"][i],
            ph["Hy"][i],
            ph["Hz"][i],
        )
        out[i] = ff.backscatter(float(f0), float(einc[i]))
    return out


def global_pole_fit(Y: np.ndarray, win_lo: int, win_hi: int):
    """PC1 + one MPM + one global residue solve."""
    from fdtd_extrapolate import mpm_poles

    # Pole estimation uses a short FIXED window; it must not grow with the cut
    # (long-window Hankel SVD costs ~W^2.5 and destroys the workflow speedup).
    win_hi = min(win_hi, win_lo + W_FIT)
    Yw = Y[win_lo:win_hi]
    W = Yw.shape[0]

    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    v = rng.normal(size=Yw.shape[1])
    v /= np.linalg.norm(v)
    for _ in range(12):
        u = Yw @ v
        v = Yw.T @ u
        v /= np.linalg.norm(v)
    pc1 = Yw @ v
    z, _ = mpm_poles(pc1, ORDER)
    pole_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    Vinv = np.linalg.pinv(z[None, :] ** np.arange(W)[:, None])
    R = Vinv @ Yw
    residue_time = time.perf_counter() - t0

    return z, R, pole_time, residue_time


def selected_energy_pole(z: np.ndarray, R: np.ndarray, dt: float):
    """Return the strongest modal residue-energy candidate in BAND."""
    s = np.log(z) / dt
    f_hz = s.imag / (2.0 * np.pi)
    q = np.abs(s.imag) / (2.0 * np.abs(s.real) + 1e-30)
    amp = np.linalg.norm(R, axis=1)
    score = amp**2 / np.maximum(1.0 - np.abs(z) ** 2, 1e-12)

    cand = [
        m
        for m in range(len(z))
        if BAND[0] < f_hz[m] < BAND[1] and np.abs(z[m]) < 1.0 and q[m] >= 3.0
    ]
    if not cand:
        return None
    m = max(cand, key=lambda k: score[k])
    return {
        "f_GHz": float(f_hz[m] / 1e9),
        "Q": float(q[m]),
        "abs_z": float(np.abs(z[m])),
        "score": float(score[m]),
    }


def make_completed_and_truncated(full: Dict[str, np.ndarray], z: np.ndarray, R: np.ndarray, cut: int):
    """Build completed and zero-truncated records of length N_FULL."""
    n_ext = np.arange(cut, N_FULL)
    base = z[None, :] ** (n_ext - T0)[:, None]

    completed = {}
    truncated = {}
    col = 0
    for c in COMPS:
        kc = full[c].shape[1]
        tail = (base @ R[:, col : col + kc]).real.astype(np.float32)

        s = full[c].copy()
        s[cut:] = tail
        completed[c] = s

        s0 = full[c].copy()
        s0[cut:] = 0.0
        truncated[c] = s0

        col += kc
    return completed, truncated


def write_csv(rows: List[dict], path: Path) -> None:
    fields = [
        "cut_steps",
        "steps_saved_percent",
        "window_length",
        "cut_solver_s",
        "global_pole_estimation_s",
        "global_residue_solve_s",
        "global_total_s",
        "proposed_total_s",
        "speedup_vs_plain_full_FDTD",
        "extrap_vs_full_RMS_dB",
        "extrap_vs_Mie_RMS_dB",
        "trunc_vs_full_RMS_dB",
        "selected_pole_f_GHz",
        "selected_pole_Q",
        "selected_pole_abs_z",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def plot_rows(rows: List[dict], path: Path) -> None:
    cuts = np.array([r["cut_steps"] for r in rows])
    speedup = np.array([r["speedup_vs_plain_full_FDTD"] for r in rows])
    err_full = np.array([r["extrap_vs_full_RMS_dB"] for r in rows])
    err_mie = np.array([r["extrap_vs_Mie_RMS_dB"] for r in rows])
    trunc = np.array([r["trunc_vs_full_RMS_dB"] for r in rows])

    # Error plot
    fig, ax1 = plt.subplots(figsize=(6.2, 3.8))
    ax1.plot(cuts, err_full, "o-", label="completed vs full FDTD")
    ax1.plot(cuts, err_mie, "s--", label="completed vs Mie")
    ax1.plot(cuts, trunc, "^:", label="zero-truncated vs full FDTD")
    ax1.set_xlabel("cutoff step")
    ax1.set_ylabel("RMS error (dB)")
    ax1.grid(True, alpha=0.35)
    ax1.invert_xaxis()

    ax2 = ax1.twinx()
    ax2.plot(cuts, speedup, "d-.", label="speedup vs full FDTD")
    ax2.set_ylabel("speedup vs full FDTD")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Sweep early-stop cutoff for solid-sphere FDTD completion.")
    p.add_argument(
        "--cuts",
        default=",".join(str(c) for c in DEFAULT_CUTS),
        help="Comma-separated cutoff steps. Default: 4000,3500,3000,2500,2000,1500",
    )
    p.add_argument(
        "--measure-cut-times",
        action="store_true",
        help="Run separate cut-length FDTD simulations to measure cut solver time. Otherwise use full_s*cut/N_FULL.",
    )
    p.add_argument("--out-prefix", default="paperB_cutoff_sweep", help="Output prefix.")
    p.add_argument(
        "--skip-spectra-npz",
        action="store_true",
        help="Do not save all spectra. CSV and PNG are still written.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cuts = parse_cuts(args.cuts)
    prefix = Path(args.out_prefix)

    print("Machine:")
    print(" ", platform.platform())
    print(" ", platform.processor() or "(processor not reported)")
    print(" ", "Python", platform.python_version(), "| NumPy", np.__version__)
    print("Experiment: solid sphere, eps_r=25, full reference = 16000 steps")
    print("Cuts:", cuts)
    print("Timing convention: FDTD update loop + recorder only; setup excluded.\n")

    print(f"=== full FDTD reference: {N_FULL} steps ===", flush=True)
    dt, rec, wf, full_solver_s, grid = run_solid_sphere(N_FULL, keep_record=True)
    print(f"full solver: {full_solver_s:.3f} s, grid {grid}", flush=True)

    full = {c: np.asarray(rec.data[c], np.float32) for c in COMPS}
    Yall = np.concatenate([full[c] for c in COMPS], axis=1).astype(np.float64)
    K6 = Yall.shape[1]
    fsel, sel = band_bins(dt)
    print(f"6K = {K6}, band bins = {len(fsel)} ({fsel[0]/1e9:.2f}-{fsel[-1]/1e9:.2f} GHz)")

    from mie_sphere import mie_backscatter

    print("Computing Mie and full-FDTD spectra...", flush=True)
    mie = np.array([mie_backscatter(EPS_R, RC * D, float(f0)) for f0 in fsel])
    sig_full = backscatter_spectrum(rec, full, dt, wf, fsel, sel)
    solver_floor = rms_db(sig_full, mie)
    print(f"full-FDTD vs Mie RMS over band: {solver_floor:.3f} dB")

    rows = []
    spectra = {"f_Hz": fsel, "mie": mie, "full_FDTD": sig_full}

    for cut in cuts:
        print(f"\n=== cutoff {cut} steps ({100*(1-cut/N_FULL):.1f}% steps saved) ===", flush=True)
        if args.measure_cut_times:
            cut_solver_s, _ = run_solid_sphere(cut, keep_record=False)
            cut_time_note = "measured"
        else:
            cut_solver_s = full_solver_s * cut / N_FULL
            cut_time_note = "estimated from step fraction"
        print(f"cut solver time: {cut_solver_s:.3f} s ({cut_time_note})", flush=True)

        z, R, pole_s, residue_s = global_pole_fit(Yall, T0, cut)
        global_s = pole_s + residue_s
        print(f"global-pole post: {global_s:.3f} s = {pole_s:.3f} pole + {residue_s:.3f} residue")

        selected = selected_energy_pole(z, R, dt)
        if selected is None:
            print("selected pole: none in band/Q/stability filter")
            fGHz = qsel = absz = np.nan
        else:
            fGHz = selected["f_GHz"]
            qsel = selected["Q"]
            absz = selected["abs_z"]
            print(f"selected pole: f={fGHz:.4f} GHz, Q={qsel:.1f}, |z|={absz:.6f}")

        completed, truncated = make_completed_and_truncated(full, z, R, cut)
        sig_ext = backscatter_spectrum(rec, completed, dt, wf, fsel, sel)
        sig_trunc = backscatter_spectrum(rec, truncated, dt, wf, fsel, sel)

        e_ext_full = rms_db(sig_ext, sig_full)
        e_ext_mie = rms_db(sig_ext, mie)
        e_trunc_full = rms_db(sig_trunc, sig_full)
        proposed_total = cut_solver_s + global_s
        speedup = full_solver_s / proposed_total

        print(
            f"RMS: completed-vs-full {e_ext_full:.3f} dB | "
            f"completed-vs-Mie {e_ext_mie:.3f} dB | "
            f"truncated-vs-full {e_trunc_full:.3f} dB"
        )
        print(f"plain full FDTD {full_solver_s:.3f} s -> proposed {proposed_total:.3f} s: {speedup:.2f}x")

        rows.append(
            {
                "cut_steps": cut,
                "steps_saved_percent": 100.0 * (1.0 - cut / N_FULL),
                "window_length": cut - T0,
                "cut_solver_s": cut_solver_s,
                "global_pole_estimation_s": pole_s,
                "global_residue_solve_s": residue_s,
                "global_total_s": global_s,
                "proposed_total_s": proposed_total,
                "speedup_vs_plain_full_FDTD": speedup,
                "extrap_vs_full_RMS_dB": e_ext_full,
                "extrap_vs_Mie_RMS_dB": e_ext_mie,
                "trunc_vs_full_RMS_dB": e_trunc_full,
                "selected_pole_f_GHz": fGHz,
                "selected_pole_Q": qsel,
                "selected_pole_abs_z": absz,
            }
        )
        spectra[f"completed_{cut}"] = sig_ext
        spectra[f"truncated_{cut}"] = sig_trunc

        # Let temporary large arrays be released before next cutoff.
        del completed, truncated, sig_ext, sig_trunc

    csv_path = prefix.with_suffix(".csv")
    png_path = prefix.with_suffix(".png")
    json_path = prefix.with_suffix(".json")
    npz_path = Path(str(prefix) + "_spectra.npz")

    write_csv(rows, csv_path)
    plot_rows(rows, png_path)

    meta = {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "timing_convention": "FDTD update loop plus recorder only; setup excluded.",
        "constants": {
            "EPS_R": EPS_R,
            "RC_cells": RC,
            "D_m": D,
            "N_FULL": N_FULL,
            "T0": T0,
            "ORDER": ORDER,
            "NTH": NTH,
            "NPH": NPH,
            "BAND_Hz": BAND,
            "NFFT": NFFT,
            "cuts": cuts,
            "cut_solver_times": "measured" if args.measure_cut_times else "estimated from full_s*cut/N_FULL",
            "full_solver_s": full_solver_s,
            "solver_floor_full_vs_Mie_RMS_dB": solver_floor,
        },
    }
    json_path.write_text(json.dumps(meta, indent=2))

    if not args.skip_spectra_npz:
        np.savez(npz_path, **spectra)

    print("\n==================== CUTOFF SWEEP SUMMARY ====================")
    print("cut   saved%  proposed(s)  speedup  ext-full(dB)  ext-Mie(dB)  trunc-full(dB)  pole(GHz)   Q")
    for r in rows:
        print(
            f"{r['cut_steps']:4d} "
            f"{r['steps_saved_percent']:7.1f} "
            f"{r['proposed_total_s']:11.3f} "
            f"{r['speedup_vs_plain_full_FDTD']:8.2f} "
            f"{r['extrap_vs_full_RMS_dB']:12.3f} "
            f"{r['extrap_vs_Mie_RMS_dB']:11.3f} "
            f"{r['trunc_vs_full_RMS_dB']:14.3f} "
            f"{r['selected_pole_f_GHz']:9.4f} "
            f"{r['selected_pole_Q']:7.1f}"
        )

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {png_path}")
    print(f"  {json_path}")
    if not args.skip_spectra_npz:
        print(f"  {npz_path}")

    print("\nDecision rule suggestion:")
    print("  Pick the smallest cutoff for which completed-vs-full error remains below the")
    print("  full-FDTD-vs-Mie solver floor, or below the error threshold you want to defend.")


if __name__ == "__main__":
    main()
