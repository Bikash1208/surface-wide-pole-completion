#!/usr/bin/env python3
"""
run_multibody_transfer.py

Validate the application claim beyond the Mie-checkable solid sphere:
    early-stopped FDTD + surface-wide global-pole completion for hollow shell,
    ellipsoid, and cylinder Huygens-surface records.

Default experiment:
    bodies: hollow_shell, ellipsoid, cylinder
    full reference: 16000 steps
    cutoff: 3000 steps
    band: 1.20--2.60 GHz
    timing: FDTD update loop + recorder only; setup excluded

For each body, the script measures:
    * full FDTD solver time
    * cut FDTD solver time, measured by a separate cut-length run unless --estimate-cut-time is used
    * global-pole post-processing time on the full record truncated at cutoff
    * completed-vs-full RCS RMS error
    * zero-truncated-vs-full RCS RMS error
    * selected dominant pole frequency and Q from the cutoff record
    * speedup versus plain full FDTD

For hollow_shell only, the script also computes a stratified-Mie reference and reports:
    * full-FDTD-vs-Mie RMS solver floor
    * completed-vs-Mie RMS error

Run:
    python run_multibody_transfer.py

Useful options:
    python run_multibody_transfer.py --cut 3000
    python run_multibody_transfer.py --bodies hollow_shell,ellipsoid,cylinder
    python run_multibody_transfer.py --estimate-cut-time
    python run_multibody_transfer.py --out-prefix paperB_multibody_validation_3000

Outputs:
    paperB_multibody_validation.csv
    paperB_multibody_validation.png
    paperB_multibody_validation.json
    paperB_multibody_validation_spectra.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Make local FDTD modules importable when this script is run from the same folder.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

C0 = 299792458.0
EPS_R = 25.0
NPML = 8
D0 = 2e-3
N_FULL = 16000
T0 = 600
ORDER = 28
NTH, NPH = 24, 48
NFFT = 65536
BAND = (1.20e9, 2.60e9)
COMPS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

DEFAULT_BODIES = ("hollow_shell", "ellipsoid", "cylinder")

BODY_DEFS = {
    "hollow_shell": {
        "kind": "shell",
        "outer_radius": 24e-3,
        "core_radius": 12e-3,
        "eps_shell": EPS_R,
        "eps_core": 1.0,
    },
    "ellipsoid": {
        "kind": "ellipsoid",
        "semi": np.array([14, 11, 8], dtype=float) * 2e-3,
        "eps": EPS_R,
    },
    "cylinder": {
        "kind": "cylinder",
        "radius": 16e-3,
        "height": 40e-3,
        "eps": EPS_R,
    },
}


def parse_bodies(text: str) -> Tuple[str, ...]:
    bodies = tuple(x.strip() for x in text.split(",") if x.strip())
    if not bodies:
        raise ValueError("At least one body is required.")
    bad = [b for b in bodies if b not in BODY_DEFS]
    if bad:
        raise ValueError(f"Unknown body/bodies {bad}. Available: {sorted(BODY_DEFS)}")
    return bodies


def db10(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x), 1e-30))


def rms_db(a, b) -> float:
    return float(np.sqrt(np.mean((db10(a) - db10(b)) ** 2)))


def band_bins(dt: float):
    f = np.fft.rfftfreq(NFFT, dt)
    sel = np.where((f >= BAND[0]) & (f <= BAND[1]))[0]
    return f[sel], sel


def body_extent(body: dict) -> float:
    if body["kind"] == "shell":
        return float(body["outer_radius"])
    if body["kind"] == "ellipsoid":
        return float(np.max(body["semi"]))
    if body["kind"] == "cylinder":
        return float(max(body["radius"], body["height"] / 2.0))
    raise ValueError(body["kind"])


def build_grid(body: dict, d: float):
    # Match the geometry scripts: Huygens gap 6 cells, buffer 4 cells, TFSF gap 3 cells.
    am = int(round(body_extent(body) / d))
    hy = int(round(6 * D0 / d))
    buf = int(round(4 * D0 / d))
    tg = int(round(3 * D0 / d))
    half = am + hy + buf + NPML
    ng = 2 * half + (2 * half) % 2
    cc = ng // 2
    center = np.array([cc, cc, cc], dtype=float) * d
    lo, hi = cc - am - tg, cc + am + tg
    rsph = (am + hy) * d
    return ng, cc, center, lo, hi, rsph


def material_mask(body: dict, sim, center: np.ndarray, d: float):
    if body["kind"] == "shell":
        from geometry import sphere_mask
        outer = sphere_mask(sim, center, float(body["outer_radius"]))
        core = sphere_mask(sim, center, float(body["core_radius"]))
        return outer & ~core

    from geometry import grid_coords
    x, y, z = grid_coords(sim)
    X, Y, Z = np.meshgrid(x - center[0], y - center[1], z - center[2], indexing="ij")

    if body["kind"] == "ellipsoid":
        s = body["semi"]
        return (X / s[0]) ** 2 + (Y / s[1]) ** 2 + (Z / s[2]) ** 2 <= 1.0

    if body["kind"] == "cylinder":
        r = float(body["radius"])
        h = float(body["height"])
        return (X**2 + Y**2 <= r**2) & (np.abs(Z) <= h / 2.0)

    raise ValueError(body["kind"])


def run_body(name: str, n_steps: int, keep_record: bool = True):
    from fdtd_engine import FDTD
    from spherical_recorder import SphericalHuygensRecorder
    from tfsf import PlaneWaveBox, gaussian_modulated

    body = BODY_DEFS[name]
    d = D0
    ng, cc, center, lo, hi, rsph = build_grid(body, d)

    sim = FDTD(ng, ng, ng, d, d, d, npml=NPML)
    mask = material_mask(body, sim, center, d)
    eps = body.get("eps", body.get("eps_shell", EPS_R))
    sim.set_material(mask, eps_r=float(eps))
    sim.finalize_material()

    # The shell script used 1.9 GHz; the sphere cutoff script used 1.8 GHz.
    # Use 1.9 GHz for all non-spherical/general bodies to match phase-4 scripts.
    wf = gaussian_modulated(1.9e9, 1.4 * 1.9e9, sim.dt)
    src = PlaneWaveBox(sim, lo, hi, lo, hi, lo, hi, wf)
    rec = SphericalHuygensRecorder(sim, center, rsph, NTH, NPH, n_steps=n_steps)

    t0 = time.perf_counter()
    for n in range(n_steps):
        sim.update_H(); src.correct_H(sim, n)
        sim.update_E(); src.correct_E(sim, n)
        rec.record(sim, n)
    elapsed = time.perf_counter() - t0

    grid = f"{ng}^3"
    if keep_record:
        return sim.dt, rec, wf, elapsed, grid
    return elapsed, grid


def backscatter_spectrum(rec, series_dict: Dict[str, np.ndarray], dt, wf, fsel, sel):
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
            rec.x, rec.y, rec.z, rec.nx, rec.ny, rec.nz, rec.dS,
            ph["Ex"][i], ph["Ey"][i], ph["Ez"][i],
            ph["Hx"][i], ph["Hy"][i], ph["Hz"][i],
        )
        out[i] = ff.backscatter(float(f0), float(einc[i]))
    return out


def global_pole_fit(Y: np.ndarray, cut: int):
    from fdtd_extrapolate import mpm_poles

    Yw = Y[T0:cut]
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
    pole_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    Vinv = np.linalg.pinv(z[None, :] ** np.arange(W)[:, None])
    R = Vinv @ Yw
    residue_s = time.perf_counter() - t0

    return z, R, pole_s, residue_s


def selected_energy_pole(z: np.ndarray, R: np.ndarray, dt: float):
    s = np.log(z) / dt
    f_hz = s.imag / (2.0 * np.pi)
    q = np.abs(s.imag) / (2.0 * np.abs(s.real) + 1e-30)
    amp = np.linalg.norm(R, axis=1)
    score = amp**2 / np.maximum(1.0 - np.abs(z) ** 2, 1e-12)

    cand = [
        m for m in range(len(z))
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
    base = z[None, :] ** (np.arange(cut, N_FULL) - T0)[:, None]
    completed = {}
    truncated = {}
    col = 0
    for c in COMPS:
        kc = full[c].shape[1]
        tail = (base @ R[:, col:col + kc]).real.astype(np.float32)
        s = full[c].copy(); s[cut:] = tail; completed[c] = s
        s0 = full[c].copy(); s0[cut:] = 0.0; truncated[c] = s0
        col += kc
    return completed, truncated


# ---------- Stratified-Mie reference for hollow shell ----------
# Same formulation as paperB_phase4_coated.py, copied here so this script is standalone.
from scipy.special import jv as _jv, hankel1 as _hankel1

# Spherical Bessel/Hankel + Riccati-Bessel derivative (complex-argument capable),
# inlined from the former vsh_pairing helpers so this package has no external dep.
def _sph_jl(l, z):
    z = np.asarray(z, dtype=complex)
    return np.sqrt(np.pi / (2.0 * z)) * _jv(l + 0.5, z)
def _sph_hl1(l, z):
    z = np.asarray(z, dtype=complex)
    return np.sqrt(np.pi / (2.0 * z)) * _hankel1(l + 0.5, z)
def _riccati_d(l, z, kind="j"):
    f = _sph_jl if kind == "j" else _sph_hl1
    return z * f(l - 1, z) - l * f(l, z)


def _shell_available() -> bool:
    return True


def shell_coated_ab(l: int, f: complex):
    body = BODY_DEFS["hollow_shell"]
    a_core = body["core_radius"]
    a_out = body["outer_radius"]
    eps_core = body["eps_core"]
    eps_shell = body["eps_shell"]

    def sph_yl(ll, z):
        return (_sph_hl1(ll, z) - _sph_jl(ll, z)) / 1j
    def psi(ll, z): return z * _sph_jl(ll, z)
    def chi(ll, z): return -z * sph_yl(ll, z)
    def xi(ll, z): return z * _sph_hl1(ll, z)
    def dpsi(ll, z): return _riccati_d(ll, z, "j")
    def dxi(ll, z): return _riccati_d(ll, z, "h")
    def dchi(ll, z):
        return -(z * sph_yl(ll - 1, z) - ll * sph_yl(ll, z))

    k = 2 * np.pi * f / C0
    m1 = np.sqrt(eps_core + 0j)
    m2 = np.sqrt(eps_shell + 0j)
    x = k * a_core
    y = k * a_out
    A = ((m2 * psi(l, m2 * x) * dpsi(l, m1 * x) - m1 * dpsi(l, m2 * x) * psi(l, m1 * x)) /
         (m2 * chi(l, m2 * x) * dpsi(l, m1 * x) - m1 * dchi(l, m2 * x) * psi(l, m1 * x)))
    B = ((m2 * psi(l, m1 * x) * dpsi(l, m2 * x) - m1 * psi(l, m2 * x) * dpsi(l, m1 * x)) /
         (m2 * dchi(l, m2 * x) * psi(l, m1 * x) - m1 * dpsi(l, m1 * x) * chi(l, m2 * x)))
    pa = psi(l, m2 * y) - A * chi(l, m2 * y)
    dpa = dpsi(l, m2 * y) - A * dchi(l, m2 * y)
    pb = psi(l, m2 * y) - B * chi(l, m2 * y)
    dpb = dpsi(l, m2 * y) - B * dchi(l, m2 * y)
    Na = psi(l, y) * dpa - m2 * dpsi(l, y) * pa
    Da = xi(l, y) * dpa - m2 * dxi(l, y) * pa
    Nb = m2 * psi(l, y) * dpb - dpsi(l, y) * pb
    Db = m2 * xi(l, y) * dpb - dxi(l, y) * pb
    return Na / Da, Nb / Db


def shell_mie_backscatter(fsel: np.ndarray, lmax: int = 8):
    out = []
    for f0 in fsel:
        k = 2 * np.pi * float(f0) / C0
        Sb = 0.0 + 0.0j
        for l in range(1, lmax + 1):
            al, bl = shell_coated_ab(l, float(f0))
            Sb += (2 * l + 1) * ((-1) ** l) * (al - bl)
        out.append(np.pi / k**2 * abs(Sb) ** 2)
    return np.asarray(out, dtype=float)


def write_csv(rows: List[dict], path: Path):
    fields = [
        "body", "grid", "cut_steps", "steps_saved_percent", "full_solver_s", "cut_solver_s",
        "global_pole_estimation_s", "global_residue_solve_s", "global_total_s",
        "proposed_total_s", "speedup_vs_plain_full_FDTD",
        "completed_vs_full_RMS_dB", "truncated_vs_full_RMS_dB",
        "full_vs_exact_Mie_RMS_dB", "completed_vs_exact_Mie_RMS_dB",
        "selected_pole_f_GHz", "selected_pole_Q", "selected_pole_abs_z",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def plot_rows(rows: List[dict], path: Path):
    bodies = [r["body"].replace("_", " ") for r in rows]
    x = np.arange(len(rows))
    width = 0.35
    comp = np.array([r["completed_vs_full_RMS_dB"] for r in rows], dtype=float)
    trunc = np.array([r["truncated_vs_full_RMS_dB"] for r in rows], dtype=float)
    speed = np.array([r["speedup_vs_plain_full_FDTD"] for r in rows], dtype=float)

    fig, ax1 = plt.subplots(figsize=(6.8, 3.9))
    ax1.bar(x - width/2, comp, width, label="completed vs full FDTD")
    ax1.bar(x + width/2, trunc, width, label="zero-truncated vs full FDTD")
    ax1.set_ylabel("RMS error (dB)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(bodies, rotation=15, ha="right")
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, speed, "o-", label="speedup vs full FDTD")
    ax2.set_ylabel("speedup vs full FDTD")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Validate early-stopped global-pole completion for non-sphere geometries.")
    p.add_argument("--bodies", default=",".join(DEFAULT_BODIES), help="Comma-separated bodies: hollow_shell,ellipsoid,cylinder")
    p.add_argument("--cut", type=int, default=3000, help="Cutoff step. Default: 3000")
    p.add_argument("--estimate-cut-time", action="store_true", help="Estimate cut time from full_s*cut/N_FULL instead of measuring it.")
    p.add_argument("--out-prefix", default="paperB_multibody_validation", help="Output prefix.")
    p.add_argument("--skip-spectra-npz", action="store_true", help="Do not save spectra NPZ.")
    return p.parse_args()


def main():
    args = parse_args()
    bodies = parse_bodies(args.bodies)
    cut = int(args.cut)
    if cut <= T0 + 2 * ORDER:
        raise ValueError(f"cut={cut} is too close to T0={T0} for ORDER={ORDER}.")
    if cut >= N_FULL:
        raise ValueError(f"cut={cut} must be smaller than N_FULL={N_FULL}.")

    prefix = Path(args.out_prefix)
    print("Machine:")
    print(" ", platform.platform())
    print(" ", platform.processor() or "(processor not reported)")
    print(" ", "Python", platform.python_version(), "| NumPy", np.__version__)
    print(f"Bodies: {bodies}")
    print(f"Full reference: {N_FULL} steps | cutoff: {cut} steps ({100*(1-cut/N_FULL):.1f}% steps saved)")
    print("Timing convention: FDTD update loop + recorder only; setup excluded.")
    print(f"Band: {BAND[0]/1e9:.2f}-{BAND[1]/1e9:.2f} GHz\n")

    rows = []
    spectra = {}

    for name in bodies:
        print(f"\n=== {name.replace('_', ' ').upper()} ===", flush=True)
        dt, rec, wf, full_s, grid = run_body(name, N_FULL, keep_record=True)
        print(f"full solver: {full_s:.3f} s, grid {grid}", flush=True)

        if args.estimate_cut_time:
            cut_s = full_s * cut / N_FULL
            cut_note = "estimated from step fraction"
        else:
            cut_s, _ = run_body(name, cut, keep_record=False)
            cut_note = "measured"
        print(f"cut solver: {cut_s:.3f} s ({cut_note})", flush=True)

        full = {c: np.asarray(rec.data[c], np.float32) for c in COMPS}
        Yall = np.concatenate([full[c] for c in COMPS], axis=1).astype(np.float64)
        fsel, sel = band_bins(dt)
        print(f"6K = {Yall.shape[1]}, band bins = {len(fsel)}")

        print("global-pole completion...", flush=True)
        z, R, pole_s, residue_s = global_pole_fit(Yall, cut)
        global_s = pole_s + residue_s
        selected = selected_energy_pole(z, R, dt)
        if selected is None:
            fGHz = qsel = absz = math.nan
            print("selected pole: none in band/Q/stability filter")
        else:
            fGHz = selected["f_GHz"]
            qsel = selected["Q"]
            absz = selected["abs_z"]
            print(f"selected pole: f={fGHz:.4f} GHz, Q={qsel:.1f}, |z|={absz:.6f}")
        print(f"global post: {global_s:.3f} s = {pole_s:.3f} pole + {residue_s:.3f} residue")

        completed, truncated = make_completed_and_truncated(full, z, R, cut)
        print("computing spectra...", flush=True)
        sig_full = backscatter_spectrum(rec, full, dt, wf, fsel, sel)
        sig_comp = backscatter_spectrum(rec, completed, dt, wf, fsel, sel)
        sig_trunc = backscatter_spectrum(rec, truncated, dt, wf, fsel, sel)

        e_comp_full = rms_db(sig_comp, sig_full)
        e_trunc_full = rms_db(sig_trunc, sig_full)
        proposed = cut_s + global_s
        speedup = full_s / proposed

        row = {
            "body": name,
            "grid": grid,
            "cut_steps": cut,
            "steps_saved_percent": 100.0 * (1.0 - cut / N_FULL),
            "full_solver_s": full_s,
            "cut_solver_s": cut_s,
            "global_pole_estimation_s": pole_s,
            "global_residue_solve_s": residue_s,
            "global_total_s": global_s,
            "proposed_total_s": proposed,
            "speedup_vs_plain_full_FDTD": speedup,
            "completed_vs_full_RMS_dB": e_comp_full,
            "truncated_vs_full_RMS_dB": e_trunc_full,
            "selected_pole_f_GHz": fGHz,
            "selected_pole_Q": qsel,
            "selected_pole_abs_z": absz,
        }

        if name == "hollow_shell" and _shell_available():
            print("computing stratified-Mie shell reference...", flush=True)
            mie = shell_mie_backscatter(fsel)
            row["full_vs_exact_Mie_RMS_dB"] = rms_db(sig_full, mie)
            row["completed_vs_exact_Mie_RMS_dB"] = rms_db(sig_comp, mie)
            spectra[f"{name}_mie"] = mie
            print(
                f"RMS: completed-vs-full {e_comp_full:.3f} dB | trunc-vs-full {e_trunc_full:.3f} dB | "
                f"full-vs-Mie {row['full_vs_exact_Mie_RMS_dB']:.3f} dB | comp-vs-Mie {row['completed_vs_exact_Mie_RMS_dB']:.3f} dB"
            )
        else:
            print(f"RMS: completed-vs-full {e_comp_full:.3f} dB | trunc-vs-full {e_trunc_full:.3f} dB")
        print(f"plain full FDTD {full_s:.3f} s -> proposed {proposed:.3f} s: {speedup:.2f}x")

        spectra[f"{name}_f_Hz"] = fsel
        spectra[f"{name}_full"] = sig_full
        spectra[f"{name}_completed"] = sig_comp
        spectra[f"{name}_truncated"] = sig_trunc
        rows.append(row)

        del full, Yall, completed, truncated, sig_full, sig_comp, sig_trunc

    csv_path = prefix.with_suffix(".csv")
    png_path = prefix.with_suffix(".png")
    json_path = prefix.with_suffix(".json")
    npz_path = Path(str(prefix) + "_spectra.npz")

    write_csv(rows, csv_path)
    plot_rows(rows, png_path)
    json_path.write_text(json.dumps({
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "timing_convention": "FDTD update loop plus recorder only; setup excluded.",
        "constants": {
            "N_FULL": N_FULL,
            "T0": T0,
            "ORDER": ORDER,
            "NTH": NTH,
            "NPH": NPH,
            "NFFT": NFFT,
            "BAND_Hz": BAND,
            "cut_steps": cut,
            "bodies": bodies,
            "cut_solver_times": "estimated" if args.estimate_cut_time else "measured",
        },
    }, indent=2))
    if not args.skip_spectra_npz:
        np.savez(npz_path, **spectra)

    print("\n==================== MULTIBODY VALIDATION SUMMARY ====================")
    print("body           full(s)  cut(s)  post(s)  prop(s) speedup  comp-full  trunc-full  pole(GHz)   Q")
    for r in rows:
        print(
            f"{r['body']:13s} "
            f"{r['full_solver_s']:7.3f} "
            f"{r['cut_solver_s']:7.3f} "
            f"{r['global_total_s']:7.3f} "
            f"{r['proposed_total_s']:7.3f} "
            f"{r['speedup_vs_plain_full_FDTD']:7.2f} "
            f"{r['completed_vs_full_RMS_dB']:10.3f} "
            f"{r['truncated_vs_full_RMS_dB']:10.3f} "
            f"{r['selected_pole_f_GHz']:9.4f} "
            f"{r['selected_pole_Q']:6.1f}"
        )

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {png_path}")
    print(f"  {json_path}")
    if not args.skip_spectra_npz:
        print(f"  {npz_path}")

    print("\nPaper-use decision:")
    print("  If completed-vs-full is consistently smaller than zero-truncated-vs-full, then")
    print("  the global-pole tail is adding real information beyond simply stopping early.")
    print("  Use the table to support the non-spherical FDTD-only application claim.")


if __name__ == "__main__":
    main()
