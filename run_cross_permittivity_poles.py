#!/usr/bin/env python3
"""
run_cross_permittivity_poles.py  --  RUN THIS YOURSELF (no sandbox).

Track A, v3. v2 fixed mode assignment (VSH dipole lock works, l=1 fraction ~0.9). The
remaining failure was NOISY DAMPING: the dipole's Re(s)/Q jagged (Q=4.1 outlier at
eps=16, dQ% ~72). That is extraction noise from reading Re(s) off the global multi-pole
fit over a short window.

FIX (two parts):
  1. longer ringdown window so the decay is actually resolved (better Re(s)).
  2. ISOLATE the dipole: project the recorded fields onto the dipole mode shape -> a clean
     single damped exponential -> refit ONE pole for a robust (f, Q).

If the dipole trajectory (esp. Q) now smooths out and LOO tightens, the ROM is validated.
"""
import numpy as np
from scipy.signal import find_peaks
from fdtd_engine import FDTD, C0
from tfsf import PlaneWaveBox, gaussian_modulated
from geometry import sphere_mask
from fdtd_extrapolate import mpm_poles
from spherical_recorder import SphericalHuygensRecorder
from mie_sphere import mie_backscatter
try:
    from scipy.special import sph_harm_y
    def Ylm(m, l, az, pol):
        y = np.asarray(sph_harm_y(l, m, pol, az))
        while y.ndim > np.ndim(pol): y = y[0]
        return y
except ImportError:
    from scipy.special import sph_harm
    def Ylm(m, l, az, pol): return sph_harm(m, l, az, pol)

EPS_LIST = [12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 25.0]
RC = 12; d_ = 2e-3; npml = 8; buffer = 4; HY_GAP = 6; tf_gap = 3
N_CUT = 2600; WIN = 1500; ORDER = 24; NTH, NPH = 24, 48; LMAX = 6   # longer window
half = RC + HY_GAP + buffer + npml; N = 2*half; N += N % 2; c = N//2
center = np.array([c, c, c])*d_; lo, hi = c-RC-tf_gap, c+RC+tf_gap
radius = RC*d_; Rsph = (RC+HY_GAP)*d_
COMPS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def resonance_ka(eps_r):
    kas = np.linspace(0.45, 1.40, 400)
    mb = np.array([mie_backscatter(eps_r, radius, k*C0/(2*np.pi*radius)) for k in kas])
    pk, pr = find_peaks(np.log(mb+1e-30), prominence=0.4)
    return float(kas[pk[np.argmax(pr["prominences"])]]) if len(pk) else 0.7


def refit_single_pole(yc, dt, s_guess):
    """Clean single-pole (alpha,omega) from an isolated damped exponential yc[t]."""
    # low-order MPM on the projected signal, pick pole nearest the guess
    z, _ = mpm_poles(yc, 6)
    s_all = np.log(z) / dt
    j = np.argmin(np.abs(s_all - s_guess))
    return s_all[j]


def extract_dipole(eps_r):
    ka = resonance_ka(eps_r); f0 = ka*C0/(2*np.pi*radius)
    sim = FDTD(N, N, N, d_, d_, d_, npml=npml)
    sim.set_material(sphere_mask(sim, center, radius), eps_r=eps_r); sim.finalize_material()
    src = PlaneWaveBox(sim, lo, hi, lo, hi, lo, hi, gaussian_modulated(f0, 1.0*f0, sim.dt))
    rec = SphericalHuygensRecorder(sim, center, Rsph, NTH, NPH, n_steps=N_CUT)
    for n in range(N_CUT):
        sim.update_H(); src.correct_H(sim, n); sim.update_E(); src.correct_E(sim, n)
        rec.record(sim, n)
    Yraw = {cc: np.asarray(rec.data[cc][N_CUT-WIN:N_CUT], float) for cc in COMPS}
    U, S, _ = np.linalg.svd(np.concatenate([Yraw[cc] for cc in COMPS], axis=1), full_matrices=False)
    z, _ = mpm_poles(U[:, 0]*S[0], ORDER); s = np.log(z)/sim.dt
    Vinv = np.linalg.pinv(z[None, :] ** np.arange(WIN)[:, None])
    R = np.stack([Vinv @ Yraw[cc] for cc in COMPS], axis=-1)
    e = np.linalg.norm(R.reshape(z.size, -1), axis=1)

    dirs = np.stack([rec.nx, rec.ny, rec.nz], axis=1)
    th = np.arccos(np.clip(rec.nz, -1, 1)); ph = np.arctan2(rec.ny, rec.nx) % (2*np.pi)
    dOm = rec.dS / Rsph**2
    LM = [(l, mm) for l in range(LMAX+1) for mm in range(-l, l+1)]
    Y0 = np.stack([Ylm(mm, l, ph, th) for (l, mm) in LM], axis=1)
    phys = [m for m in range(z.size) if 0.05*f0 < s[m].imag/(2*np.pi) <= 3*f0
            and abs(z[m]) < 1.0 and e[m] > 0.05*e.max()]
    best, best_frac = None, -1.0
    for m in phys:
        Er = np.einsum("kj,kj->k", R[m, :, 0:3], dirs)
        Hr = np.einsum("kj,kj->k", R[m, :, 3:6], dirs)
        aE = (Er[:, None]*np.conj(Y0)*dOm[:, None]).sum(0)
        aH = (Hr[:, None]*np.conj(Y0)*dOm[:, None]).sum(0)
        El = np.zeros(LMAX+1)
        for (l, mm), ae, ah in zip(LM, aE, aH):
            El[l] += abs(ae)**2 + abs(ah)**2
        frac1 = El[1] / (El.sum() + 1e-30)
        if frac1 > best_frac:
            best_frac, best = frac1, m

    # RAW global-fit dipole pole. The longer window already stabilises Re(s); the projected
    # single-pole refit was MIS-PICKING a spurious near-undamped sibling (eps18 Q=42.2 vs
    # raw 8.3 / envelope 10.5 -- see rom_diag_eps18). Frequency from the raw fit is precise.
    # Optional robust cross-check: envelope alpha.
    F = R[best]
    proj = np.zeros(WIN, complex)
    for ci, cc in enumerate(COMPS):
        proj += Yraw[cc] @ np.conj(F[:, ci])
    a = np.abs(proj); t = np.arange(a.size) * sim.dt; m0 = a > 0.05 * a.max()
    slope = np.linalg.lstsq(np.vstack([t[m0], np.ones(m0.sum())]).T,
                            np.log(a[m0] + 1e-30), rcond=None)[0][0]
    alpha_env = -slope
    sd = s[best]                                          # raw dipole pole (use this)
    Q_raw = abs(sd.imag) / (2 * abs(sd.real))
    Q_env = abs(sd.imag) / (2 * alpha_env) if alpha_env > 0 else np.inf
    return sd, Q_raw, best_frac, Q_env


def main():
    print("v3: VSH dipole lock + mode-projected single-pole refit (longer window)...", flush=True)
    eps = np.array(EPS_LIST); traj = []
    for er in eps:
        sd, Q, frac, Q_env = extract_dipole(er)
        traj.append(sd)
        print(f"  eps_r={er:5.1f}: dipole f={sd.imag/(2*np.pi)/1e9:6.3f}GHz "
              f"Q_raw={Q:5.1f}  Q_env={Q_env:5.1f}  (l=1 fraction {frac:.2f})", flush=True)
    traj = np.array(traj)

    print("\n========  LEAVE-ONE-OUT (robust dipole)  ========")
    print(" held eps |  f_dir   f_pred |  Q_dir  Q_pred |  df%   dQ%")
    df, dQ = [], []
    for i in range(1, len(eps)-1):
        keep = [k for k in range(len(eps)) if k != i]; deg = min(2, len(keep)-1)
        pr = np.polyfit(eps[keep], traj[keep].real, deg)
        pi = np.polyfit(eps[keep], traj[keep].imag, deg)
        sp = np.polyval(pr, eps[i]) + 1j*np.polyval(pi, eps[i]); sd = traj[i]
        fd, fp = sd.imag/(2*np.pi)/1e9, sp.imag/(2*np.pi)/1e9
        Qd, Qp = abs(sd.imag)/(2*abs(sd.real)), abs(sp.imag)/(2*abs(sp.real))
        df.append(100*abs(fp-fd)/fd); dQ.append(100*abs(Qp-Qd)/Qd)
        print(f"  {eps[i]:6.1f}  | {fd:6.3f} {fp:6.3f} | {Qd:6.1f} {Qp:6.1f} | "
              f"{df[-1]:5.1f} {dQ[-1]:5.1f}")
    print(f"\nmean df% = {np.mean(df):.1f}, mean dQ% = {np.mean(dQ):.1f}")
    print("GO if df%<~2 and dQ%<~15 with a smooth Q(eps) -> ROM validated, build the")
    print("residue/RCS reconstruction next. Still noisy Q -> damping is genuinely hard to")
    print("extract here; ROM scope shrinks to frequency-only (still a usable design tool).")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, os
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(eps, traj.imag/(2*np.pi)/1e9, "o-"); ax[0].set_xlabel("eps_r")
    ax[0].set_ylabel("dipole f (GHz)"); ax[0].set_title("dipole frequency vs eps_r (v3)")
    ax[1].plot(eps, np.abs(traj.imag)/(2*np.abs(traj.real)), "s-", color="crimson")
    ax[1].set_xlabel("eps_r"); ax[1].set_ylabel("dipole Q"); ax[1].set_title("dipole Q vs eps_r (v3)")
    for a in ax: a.grid(True, ls=":", alpha=0.5)
    os.makedirs("figures", exist_ok=True); fig.tight_layout()
    fig.savefig("figures/rom_dipole_v3.png", dpi=200, bbox_inches="tight")
    print("\nsaved figures/rom_dipole_v3.png")


if __name__ == "__main__":
    main()
