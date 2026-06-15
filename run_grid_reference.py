#!/usr/bin/env python3
"""
run_grid_reference.py -- completed-vs-reference for the non-analytic bodies
(addresses reviewer 4.3 / 4.8: separate completed-vs-coarse-full from
completed-vs-reference).

The ellipsoid and cylinder have no closed-form solution, so the honest reference
is a 2x grid-refined FULL run (1 mm vs the 2 mm production grid). This script:

  1. loads the cached COARSE spectra that produced Table I
     (paperB_multibody_validation_spectra.npz: *_full, *_completed at cut=4000),
     and self-checks that completed-vs-coarse-full reproduces Table I
     (ellipsoid 1.43 dB, cylinder 0.12 dB) -- proof the pipelines match;
  2. runs each body on a 2x grid (d = 1 mm, N = 32000 steps = same physical
     time) with the IDENTICAL geometry/source/NTFF code from
     run_multibody_transfer.py;
  3. reports, over 1.20-2.60 GHz:
        full(coarse)     vs 2x-grid full   == the discretization floor
        completed(coarse) vs 2x-grid full  == completed-vs-reference (the number
                                              the table needs)

Cost: the 2x grid is ~16x the coarse run (8x cells, 2x steps). Expect a few
minutes per body on the production machine. Output -> paperB_2xgrid_reference.npz
plus a printed table block to paste back.
"""
from __future__ import annotations
import sys
import time
import platform
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_multibody_transfer as MB   # reuse the EXACT Table-I pipeline

D_FINE = 1e-3            # 2x refinement of the 2 mm production grid
N_FINE = 32000          # 2x steps -> same physical duration as 16000 @ 2 mm
BODIES = ("ellipsoid", "cylinder")
COARSE_NPZ = "paperB_multibody_validation_spectra.npz"
OUT_NPZ = "paperB_2xgrid_reference.npz"


def run_full(name: str, d: float, n_steps: int):
    """Full FDTD run at grid spacing d -- mirrors MB.run_body but parametrised."""
    from fdtd_engine import FDTD
    from spherical_recorder import SphericalHuygensRecorder
    from tfsf import PlaneWaveBox, gaussian_modulated

    body = MB.BODY_DEFS[name]
    ng, cc, center, lo, hi, rsph = MB.build_grid(body, d)
    sim = FDTD(ng, ng, ng, d, d, d, npml=MB.NPML)
    sim.set_material(MB.material_mask(body, sim, center, d),
                     eps_r=float(body.get("eps", body.get("eps_shell", MB.EPS_R))))
    sim.finalize_material()
    wf = gaussian_modulated(1.9e9, 1.4 * 1.9e9, sim.dt)   # 1.9 GHz, as in MB.run_body
    src = PlaneWaveBox(sim, lo, hi, lo, hi, lo, hi, wf)
    rec = SphericalHuygensRecorder(sim, center, rsph, MB.NTH, MB.NPH, n_steps=n_steps)
    t0 = time.perf_counter()
    for n in range(n_steps):
        sim.update_H(); src.correct_H(sim, n)
        sim.update_E(); src.correct_E(sim, n)
        rec.record(sim, n)
    return float(sim.dt), rec, wf, f"{ng}^3", time.perf_counter() - t0


def spectrum(rec, dt, wf, n_steps):
    """Backscatter sigma(f) over BAND -- mirrors MB.backscatter_spectrum, but the
    incident normalisation uses this run's own n_steps and dt."""
    from ntff_rcs import FarField
    fsel, sel = MB.band_bins(dt)
    ph = {c: np.fft.rfft(np.asarray(rec.data[c], np.float32), n=MB.NFFT, axis=0)[sel].astype(np.complex64)
          for c in MB.COMPS}
    wfs = np.array([wf(n) for n in range(n_steps)])
    einc = np.abs(np.fft.rfft(wfs, n=MB.NFFT))[sel]
    out = np.zeros(len(fsel))
    for i, f0 in enumerate(fsel):
        ff = FarField(rec.x, rec.y, rec.z, rec.nx, rec.ny, rec.nz, rec.dS,
                      ph["Ex"][i], ph["Ey"][i], ph["Ez"][i],
                      ph["Hx"][i], ph["Hy"][i], ph["Hz"][i])
        out[i] = ff.backscatter(float(f0), float(einc[i]))
    return fsel, out


def main():
    print("Machine:", platform.platform(), "| NumPy", np.__version__)
    coarse = np.load(COARSE_NPZ)
    band = MB.BAND
    save = {}
    print("\n%-10s %-22s %-22s %-22s" % ("body", "completed-vs-full(chk)",
                                         "full-vs-2x(floor)", "completed-vs-2x(REF)"))
    rows = []
    for name in BODIES:
        fco = coarse[f"{name}_f_Hz"]
        full_co = coarse[f"{name}_full"]
        comp_co = coarse[f"{name}_completed"]
        mco = (fco >= band[0]) & (fco <= band[1])
        chk = MB.rms_db(comp_co[mco], full_co[mco])   # must match Table I

        print(f"  [{name}] running 2x grid (d={D_FINE*1e3:.1f} mm, {N_FINE} steps)...", flush=True)
        dt_f, rec_f, wf_f, grid_f, secs = run_full(name, D_FINE, N_FINE)
        ff, sig_f = spectrum(rec_f, dt_f, wf_f, N_FINE)
        print(f"     2x grid {grid_f}, {secs:.0f} s", flush=True)

        # interpolate the 2x-grid spectrum onto the coarse band frequencies (linear sigma)
        lo, hi = ff.min(), ff.max()
        m = mco & (fco >= lo) & (fco <= hi)
        sig_f_on_co = np.interp(fco, ff, sig_f)
        floor = MB.rms_db(full_co[m], sig_f_on_co[m])
        ref = MB.rms_db(comp_co[m], sig_f_on_co[m])

        print("%-10s %-22.3f %-22.3f %-22.3f" % (name, chk, floor, ref))
        rows.append((name, chk, floor, ref))
        save[f"{name}_f_2x"] = ff
        save[f"{name}_full_2x"] = sig_f
        save[f"{name}_full_2x_on_coarse"] = sig_f_on_co
    np.savez(OUT_NPZ, **save)

    print("\n==================== PASTE-BACK BLOCK ====================")
    print("(self-check 'completed-vs-full' must equal Table I: ellipsoid 1.43, cylinder 0.12)")
    for name, chk, floor, ref in rows:
        print(f"  {name:10s}: completed-vs-full={chk:.2f}  full-vs-2x(floor)={floor:.2f}  "
              f"completed-vs-2x(REF)={ref:.2f}  dB")
    print(f"\nSaved {OUT_NPZ}")


if __name__ == "__main__":
    main()
