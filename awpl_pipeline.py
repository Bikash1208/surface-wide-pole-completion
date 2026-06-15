#!/usr/bin/env python3
"""
awpl_pipeline.py  --  shared engine for the five AWPL revision figures.

Reuses the validated FDTD/MPM/NTFF pipeline (fdtd_engine, tfsf, geometry,
spherical_recorder, fdtd_extrapolate, ntff_rcs, mie_sphere). The physics
snippets here are copied from run_cutoff_sweep.py and
paperB_phase4b_geometries.py so the figures use exactly the validated code.

Requires Numba/SciPy and the local modules -> run on the solver machine.
"""
from __future__ import annotations

import time
import numpy as np

C0 = 299792458.0
COMPS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
NPML = 8
NTH, NPH = 24, 48
NFFT = 65536
BAND = (1.2e9, 2.6e9)
ORDER = 28
T0 = 600          # window start at the reference grid (dx = 2 mm)
W_FIT = 2400      # fixed pole-fit window at the reference grid
Q_MIN = 3.0

# physical absorber/gap budget (mm) -- kept fixed so finer grids cover the
# same physical domain (cells scale as 1/dx, like paperB_phase4b_geometries.py)
HY_GAP_MM, BUF_MM, TF_GAP_MM = 12.0, 8.0, 6.0


# ----------------------------------------------------------------------------- style
FIG_W = 1.75          # target figure width (in) -- half an IEEE column
FIG_H = 1.45          # single-panel height (in)
FIG_H2 = 2.85         # two-panel (stacked) height (in)


def set_ieee_style():
    """Legible at 1.75 in width: fonts/markers sized for a half-column figure."""
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 6.5,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.5,
        "lines.linewidth": 1.1, "lines.markersize": 3.4, "axes.linewidth": 0.6,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "xtick.direction": "in", "ytick.direction": "in",
        "axes.labelpad": 2.0, "legend.frameon": False,
        "legend.handlelength": 1.6, "legend.handletextpad": 0.4,
        "legend.labelspacing": 0.3, "legend.borderaxespad": 0.3,
        "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


# Okabe-Ito colorblind-safe palette
C_COMP, C_TRUNC, C_FULL, C_MIE = "#0072B2", "#D55E00", "#000000", "#009E73"
C_FLOOR, C_CLOUD, C_ART = "#000000", "#9A9A9A", "#D55E00"


def db10(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x, float), 1e-30))


def rms_db(a, b):
    return float(np.sqrt(np.mean((db10(a) - db10(b)) ** 2)))


def band_bins(dt, nfft=NFFT, band=BAND):
    f = np.fft.rfftfreq(nfft, dt)
    sel = np.where((f >= band[0]) & (f <= band[1]))[0]
    if sel.size == 0:
        raise ValueError("No FFT bins in band.")
    return f[sel], sel


# ----------------------------------------------------------------------------- runs
def run_sphere(eps_r, dx_mm, n_steps, radius_mm=24.0, src_GHz=1.8):
    """Solid dielectric sphere; returns (dt, rec, wf, meta). Mirrors cutoff_sweep."""
    from fdtd_engine import FDTD
    from tfsf import PlaneWaveBox, gaussian_modulated
    from geometry import sphere_mask
    from spherical_recorder import SphericalHuygensRecorder

    d = dx_mm * 1e-3
    rc = int(round(radius_mm * 1e-3 / d))
    hy = int(round(HY_GAP_MM * 1e-3 / d))
    buf = int(round(BUF_MM * 1e-3 / d))
    tg = int(round(TF_GAP_MM * 1e-3 / d))
    half = rc + hy + buf + NPML
    ng = 2 * half + (2 * half) % 2
    cc = ng // 2
    center = np.array([cc, cc, cc], float) * d
    lo, hi = cc - rc - tg, cc + rc + tg
    rsph = (rc + hy) * d

    sim = FDTD(ng, ng, ng, d, d, d, npml=NPML)
    sim.set_material(sphere_mask(sim, center, rc * d), eps_r=float(eps_r))
    sim.finalize_material()
    wf = gaussian_modulated(src_GHz * 1e9, 1.4 * src_GHz * 1e9, sim.dt)
    src = PlaneWaveBox(sim, lo, hi, lo, hi, lo, hi, wf)
    rec = SphericalHuygensRecorder(sim, center, rsph, NTH, NPH, n_steps=n_steps)
    t0 = time.perf_counter()
    for n in range(n_steps):
        sim.update_H(); src.correct_H(sim, n)
        sim.update_E(); src.correct_E(sim, n)
        rec.record(sim, n)
    solver_s = time.perf_counter() - t0
    meta = {"grid": f"{ng}^3", "ng": ng, "dx_mm": dx_mm, "rc": rc,
            "radius_mm": radius_mm, "solver_s": solver_s, "n_steps": n_steps,
            "cells_per_radius": rc}
    return float(sim.dt), rec, wf, meta


def run_cylinder(eps_r, dx_mm, n_steps, radius_mm=16.0, height_mm=40.0, src_GHz=1.9):
    """Finite dielectric cylinder (axis z); returns (dt, rec, wf, meta).
    Mask + grid from paperB_phase4b_geometries.py."""
    from fdtd_engine import FDTD
    from tfsf import PlaneWaveBox, gaussian_modulated
    from geometry import grid_coords
    from spherical_recorder import SphericalHuygensRecorder

    d = dx_mm * 1e-3
    r, h = radius_mm * 1e-3, height_mm * 1e-3
    am = int(round(max(r, h / 2) / d))
    hy = int(round(HY_GAP_MM * 1e-3 / d))
    buf = int(round(BUF_MM * 1e-3 / d))
    tg = int(round(TF_GAP_MM * 1e-3 / d))
    half = am + hy + buf + NPML
    ng = 2 * half + (2 * half) % 2
    cc = ng // 2
    center = np.array([cc, cc, cc], float) * d

    sim = FDTD(ng, ng, ng, d, d, d, npml=NPML)
    x, y, z = grid_coords(sim)
    X, Y, Z = np.meshgrid(x - center[0], y - center[1], z - center[2], indexing="ij")
    mask = (X ** 2 + Y ** 2 <= r ** 2) & (np.abs(Z) <= h / 2)
    sim.set_material(mask, eps_r=float(eps_r))
    sim.finalize_material()
    wf = gaussian_modulated(src_GHz * 1e9, 1.4 * src_GHz * 1e9, sim.dt)
    src = PlaneWaveBox(sim, cc - am - tg, cc + am + tg, cc - am - tg, cc + am + tg,
                       cc - am - tg, cc + am + tg, wf)
    rec = SphericalHuygensRecorder(sim, center, (am + hy) * d, NTH, NPH, n_steps=n_steps)
    t0 = time.perf_counter()
    for n in range(n_steps):
        sim.update_H(); src.correct_H(sim, n)
        sim.update_E(); src.correct_E(sim, n)
        rec.record(sim, n)
    solver_s = time.perf_counter() - t0
    meta = {"grid": f"{ng}^3", "ng": ng, "dx_mm": dx_mm, "solver_s": solver_s,
            "n_steps": n_steps}
    return float(sim.dt), rec, wf, meta


# ----------------------------------------------------------------------------- modal
def stack_record(rec):
    full = {c: np.asarray(rec.data[c], np.float32) for c in COMPS}
    Yall = np.concatenate([full[c] for c in COMPS], axis=1).astype(np.float64)
    return full, Yall


def global_pole_fit(Yall, t0=T0, w_fit=W_FIT, order=ORDER, timeit=True):
    """Leading-PC projection + one MPM + one batched residue solve over the
    fixed window [t0, t0+w_fit]. Returns z, R (M x 6K), t_pole, t_res."""
    from fdtd_extrapolate import mpm_poles
    hi = t0 + w_fit
    Yw = Yall[t0:hi]
    tp = time.perf_counter()
    v = np.random.default_rng(0).normal(size=Yw.shape[1])
    v /= np.linalg.norm(v)
    for _ in range(12):
        u = Yw @ v
        v = Yw.T @ u
        v /= np.linalg.norm(v)
    pc1 = Yw @ v
    z, _ = mpm_poles(pc1, order)
    t_pole = time.perf_counter() - tp
    tr = time.perf_counter()
    Vinv = np.linalg.pinv(z[None, :] ** np.arange(w_fit)[:, None])
    R = Vinv @ Yw
    t_res = time.perf_counter() - tr
    return z, R, t_pole, t_res


def candidate_table(z, R, dt):
    """Per-pole f (GHz), Q, residue amplitude, integrated energy E_m."""
    s = np.log(z) / dt
    f = s.imag / (2.0 * np.pi)
    q = np.abs(s.imag) / (2.0 * np.abs(s.real) + 1e-30)
    amp = np.linalg.norm(R, axis=1)
    E = amp ** 2 / np.maximum(1.0 - np.abs(z) ** 2, 1e-12)
    return f, q, amp, E


def energy_select(z, R, dt, band=BAND, q_min=Q_MIN):
    """Return dict with the energy-selected, amplitude, and max-Q picks (indices)."""
    f, q, amp, E = candidate_table(z, R, dt)
    band_idx = [m for m in range(z.size) if band[0] < f[m] < band[1] and np.abs(z[m]) < 1.0]
    if not band_idx:
        return None
    qual = [m for m in band_idx if q[m] >= q_min] or band_idx
    sel = max(qual, key=lambda m: E[m])
    m_amp = max(band_idx, key=lambda m: amp[m])
    m_q = max(band_idx, key=lambda m: q[m])
    return {"f": f, "q": q, "amp": amp, "E": E, "band": band_idx,
            "sel": sel, "amp_pick": m_amp, "q_pick": m_q}


def complete_series(full, z, R, cut, n_total, t0=T0):
    """Replace [cut, n_total) of each signal by the common-pole model."""
    n_ext = np.arange(cut, n_total)
    base = z[None, :] ** (n_ext - t0)[:, None]   # (Next, M)
    comp = {}
    col = 0
    for c in COMPS:
        arr = np.asarray(full[c], np.float64).copy()
        kc = arr.shape[1]
        arr[cut:n_total, :] = (base @ R[:, col:col + kc]).real
        comp[c] = arr.astype(np.float32)
        col += kc
    return comp


def truncate_series(full, cut, n_total):
    out = {}
    for c in COMPS:
        arr = np.asarray(full[c], np.float32).copy()
        arr[cut:n_total, :] = 0.0
        out[c] = arr
    return out


# ----------------------------------------------------------------------------- RCS
def _einc_band(wf, n_full, sel, nfft=NFFT):
    wfs = np.array([wf(n) for n in range(n_full)], dtype=np.float64)
    return np.abs(np.fft.rfft(wfs, n=nfft))[sel]


def backscatter_spectrum(rec, series, dt, wf, fsel, sel, n_full):
    """Monostatic backscatter RCS over the band (m^2)."""
    from ntff_rcs import FarField
    ph = {c: np.fft.rfft(np.asarray(series[c], np.float32), n=NFFT, axis=0)[sel].astype(np.complex64)
          for c in COMPS}
    einc = _einc_band(wf, n_full, sel)
    out = np.zeros(len(fsel))
    for i, f0 in enumerate(fsel):
        inc = float(einc[i])
        if inc <= 0 or not np.isfinite(inc):
            out[i] = np.nan
            continue
        ff = FarField(rec.x, rec.y, rec.z, rec.nx, rec.ny, rec.nz, rec.dS,
                      ph["Ex"][i], ph["Ey"][i], ph["Ez"][i],
                      ph["Hx"][i], ph["Hy"][i], ph["Hz"][i])
        out[i] = ff.backscatter(float(f0), inc)
    out = np.where(np.isfinite(out), out, 1e-30)
    return out


def bistatic_pattern(rec, series, dt, wf, f0, thetas, n_full):
    """Bistatic RCS sigma(theta) at a single frequency f0, phi=0 (E-plane)."""
    from ntff_rcs import FarField
    f = np.fft.rfftfreq(NFFT, dt)
    i0 = int(np.argmin(np.abs(f - f0)))
    ph = {c: np.fft.rfft(np.asarray(series[c], np.float32), n=NFFT, axis=0)[i0]
          for c in COMPS}
    wfs = np.array([wf(n) for n in range(n_full)], dtype=np.float64)
    inc = float(np.abs(np.fft.rfft(wfs, n=NFFT))[i0])
    ff = FarField(rec.x, rec.y, rec.z, rec.nx, rec.ny, rec.nz, rec.dS,
                  ph["Ex"], ph["Ey"], ph["Ez"], ph["Hx"], ph["Hy"], ph["Hz"])
    th = np.asarray(thetas, float)
    phi = np.zeros_like(th)                      # phi must broadcast with theta
    return float(f[i0]), np.asarray(ff.rcs(th, phi, float(f[i0]), inc), float)


def mie_backscatter_band(fsel, eps_r, a_radius):
    from mie_sphere import mie_backscatter
    return np.asarray([mie_backscatter(eps_r, a_radius, float(f0)) for f0 in fsel], float)


def mie_bistatic(thetas, eps_r, a_radius, f0, pol="E"):
    from mie_sphere import mie_rcs
    return np.asarray(mie_rcs(np.asarray(thetas, float), eps_r, a_radius, float(f0), pol=pol), float)
