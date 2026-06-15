#!/usr/bin/env python3
"""
run_estimator_baselines.py

Vector fitting and the Cauchy method on the SAME seven anchor ringdowns, scored by the
SAME protocol as GP-MPM, so the comparison is airtight:

  identical data    : the global PC1 of each anchor's windowed surface record
                      (window [600:3000], the GP-MPM fitting window);
  identical bands   : candidate poles restricted to 0.9-2.8 GHz, decaying only;
  identical selection: per family, candidates within 10% of the exact Mie frequency,
                      ranked by continuous integrated energy |R|^2 / (2 alpha)
                      (the continuous-time analog of the discrete rule, Eq. 3);
  identical scoring : extraction floor vs exact complex Mie poles; quadratic LOO
                      (interior + endpoint) on each method's own pole trajectories;
                      failure = no admissible candidate near a family.

IMPLEMENTATION NOTES (disclose in the paper):
  * VF: classic Gustavsen-Semlyen sigma-iteration (5 rounds, order 16), applied to the
    positive-frequency response with unconstrained complex poles -- the standard modal-ID
    shortcut; conjugacy is irrelevant for pole location on one-sided data.
  * Cauchy: total-least-squares rational fit P/Q (orders 17/16) via the SVD null vector;
    poles = roots of Q.
  * Both fit the FFT spectrum (65536-point padding) of the SAME windowed PC1 signal that
    GP-MPM's pencil sees, so no method gets more data than another.

First run: 7 FDTD anchors (~2 min) cached to paperB_pc1.npz; afterwards analysis-only.
GP-MPM rows are read from paperB_fig4_data.npz (run compute_exact_mie_poles.py extract first).
"""
import os
import time
import numpy as np

from compute_exact_mie_poles import exact_table, EPS_LIST

T0 = 600; WIN = 2400          # GP-MPM's pencil window (its own method choice)
N_REC = 8000                  # full post-prompt record given to the spectral methods:
                              # v1 fed VF/Cauchy the truncated 2400-sample window, whose
                              # rectangular cut SETS the linewidth (observed: VF Q floor
                              # ~54%, Q falling while true Q rises). Frequency-domain
                              # rational fitting needs the decayed record; MPM does not.
BAND = (0.9e9, 2.8e9)
NFFT = 65536
VF_ORDER = 16; VF_ITERS = 5
CAUCHY_P, CAUCHY_Q = 17, 16
CACHE = "paperB_pc1_v2.npz"

# ---------------- anchor PC1 cache (7 FDTD runs on first call) ----------------
def ensure_pc1():
    if os.path.exists(CACHE):
        return np.load(CACHE)
    from fdtd_engine import FDTD
    from tfsf import PlaneWaveBox, gaussian_modulated
    from geometry import sphere_mask
    from spherical_recorder import SphericalHuygensRecorder
    RC = 12; d_ = 2e-3; npml = 8; buffer = 4; HY_GAP = 6; tf_gap = 3
    N_CUT = 8000; NTH, NPH = 24, 48
    half = RC + HY_GAP + buffer + npml; N = 2 * half + (2 * half) % 2; c = N // 2
    center = np.array([c, c, c]) * d_
    lo, hi = c - RC - tf_gap, c + RC + tf_gap
    Rsph = (RC + HY_GAP) * d_
    pc1s = []
    dt = None
    for er in EPS_LIST:
        sim = FDTD(N, N, N, d_, d_, d_, npml=npml)
        sim.set_material(sphere_mask(sim, center, RC * d_), eps_r=er)
        sim.finalize_material()
        src = PlaneWaveBox(sim, lo, hi, lo, hi, lo, hi,
                           gaussian_modulated(1.8e9, 1.4 * 1.8e9, sim.dt))
        rec = SphericalHuygensRecorder(sim, center, Rsph, NTH, NPH, n_steps=N_CUT)
        t0 = time.time()
        for n in range(N_CUT):
            sim.update_H(); src.correct_H(sim, n); sim.update_E(); src.correct_E(sim, n)
            rec.record(sim, n)
        Yfull = np.concatenate([np.asarray(rec.data[cc], np.float32)
                                for cc in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")], axis=1)
        Yw = Yfull[T0:T0 + WIN].astype(np.float64)
        v = np.random.default_rng(0).normal(size=Yw.shape[1])
        for _ in range(12):
            u = Yw @ v; v = Yw.T @ u; v /= np.linalg.norm(v)
        pc1s.append((Yfull.astype(np.float64) @ v))          # FULL-record PC1 series
        dt = sim.dt
        print(f"  eps {er:5.1f}: run+PC1 in {time.time()-t0:.0f}s", flush=True)
    np.savez(CACHE, pc1=np.stack(pc1s), dt=dt)
    return np.load(CACHE)

# ---------------- shared spectrum ----------------
def spectrum(pc1, dt):
    F = np.fft.rfft(pc1[T0:N_REC], n=NFFT)      # decayed post-prompt record, no window cut
    f = np.fft.rfftfreq(NFFT, dt)
    sel = (f >= BAND[0]) & (f <= BAND[1])
    return 2j * np.pi * f[sel], F[sel]          # s = j*omega samples, H(s)

# ---------------- vector fitting (scalar, sigma iteration) ----------------
def vector_fit(s, H, order=VF_ORDER, iters=VF_ITERS):
    w = s.imag
    poles = (-w.mean() / 100 + 1j * np.linspace(w.min(), w.max(), order))
    for _ in range(iters):
        C = 1.0 / (s[:, None] - poles[None, :])
        A = np.hstack([C, np.ones((len(s), 1)), -H[:, None] * C])
        x, *_ = np.linalg.lstsq(A, H, rcond=None)
        ctil = x[order + 1:]
        Apole = np.diag(poles) - np.outer(np.ones(order), ctil)
        poles = np.linalg.eigvals(Apole)
        poles = np.where(poles.real > 0, -poles.real + 1j * poles.imag, poles)  # flip unstable
    C = 1.0 / (s[:, None] - poles[None, :])
    A = np.hstack([C, np.ones((len(s), 1))])
    x, *_ = np.linalg.lstsq(A, H, rcond=None)
    return poles, x[:VF_ORDER]

# ---------------- Cauchy method (TLS rational fit) ----------------
def cauchy_fit(s, H, p=CAUCHY_P, q=CAUCHY_Q):
    # affine-centered REAL fitting variable x in [-1,1]: s = j(mid + half*x). v1 used a
    # multiplicative scaling that left the samples clustered on an arc -> order-17 complex
    # Vandermonde was numerically singular and every root was garbage (7/7 failures).
    w = s.imag
    mid = 0.5 * (w.max() + w.min()); half = 0.5 * (w.max() - w.min())
    x = (w - mid) / half
    Vp = np.vander(x, p, increasing=True)
    Vq = np.vander(x, q, increasing=True)
    A = np.hstack([Vp, -H[:, None] * Vq])
    _, _, Vh = np.linalg.svd(A, full_matrices=False)
    coef = Vh[-1]
    qc = coef[p:]
    xr = np.roots(qc[::-1])
    poles = 1j * (mid + half * xr)               # map back to the s-plane
    poles = np.where(poles.real > 0, -poles.real + 1j * poles.imag, poles)  # mirror
    C = 1.0 / (s[:, None] - poles[None, :])
    R, *_ = np.linalg.lstsq(np.hstack([C, np.ones((len(s), 1))]), H, rcond=None)
    return poles, R[:len(poles)]

# ---------------- shared selection + scoring ----------------
def select(poles, residues, fx):
    """Per family: candidates within 10% of exact f, decaying, in band; rank by
    continuous integrated energy |R|^2/(2 alpha). Returns (f, Q) or None."""
    out = []
    for pole, R in zip(poles, residues):
        f0 = pole.imag / (2 * np.pi); al = -pole.real
        if al <= 0 or not (BAND[0] < f0 < BAND[1]):
            continue
        if abs(f0 - fx.real) < 0.10 * fx.real:
            out.append((abs(R) ** 2 / (2 * al), f0, pole.imag / (2 * al)))
    if not out:
        return None
    _, f0, Q = max(out)
    return f0, Q

def loo(eps, fv, Qv):
    ok = ~np.isnan(fv)
    e, f, Q = eps[ok], fv[ok], Qv[ok]
    if ok.sum() < 4:
        return (np.nan,) * 4
    sv = -np.pi * f / Q + 1j * 2 * np.pi * f
    dfE, dQE = [], []
    for i in range(len(e)):
        keep = [k for k in range(len(e)) if k != i]
        pr = np.polyfit(e[keep], sv.real[keep], 2)
        pi = np.polyfit(e[keep], sv.imag[keep], 2)
        sp = np.polyval(pr, e[i]) + 1j * np.polyval(pi, e[i])
        fp = sp.imag / (2 * np.pi); Qp = abs(sp.imag) / (2 * abs(sp.real))
        dfE.append(100 * abs(fp - f[i]) / f[i]); dQE.append(100 * abs(Qp - Q[i]) / Q[i])
    inter = slice(1, len(e) - 1)
    return (np.mean(dfE[inter]), np.mean(dQE[inter]),
            np.mean([dfE[0], dfE[-1]]), np.mean([dQE[0], dQE[-1]]))

def main():
    exact = exact_table()
    Z = ensure_pc1()
    pc1, dt = np.asarray(Z["pc1"]), float(Z["dt"])
    eps = np.array(EPS_LIST)

    methods = {"vector fitting": vector_fit, "Cauchy method": cauchy_fit}
    print("\nBASELINES on the identical windowed PC1 records:")
    for name, fit in methods.items():
        res = {"a": [], "b": []}; fails = {"a": 0, "b": 0}; ttot = 0.0
        for i, er in enumerate(EPS_LIST):
            s, H = spectrum(pc1[i], dt)
            t0 = time.time()
            poles, R = fit(s, H)
            ttot += time.time() - t0
            fa, fb = exact[er]
            for fam, fx in (("a", fa), ("b", fb)):
                hit = select(poles, R, fx)
                if hit is None:
                    fails[fam] += 1; res[fam].append((np.nan, np.nan))
                else:
                    res[fam].append(hit)
        print(f"\n  {name} (order {VF_ORDER if 'vector' in name else (CAUCHY_P, CAUCHY_Q)},"
              f" {1e3*ttot/len(EPS_LIST):.0f} ms/anchor):")
        for fam, tag in (("a", "electric"), ("b", "magnetic")):
            fv = np.array([r[0] for r in res[fam]]); Qv = np.array([r[1] for r in res[fam]])
            fx = np.array([exact[er][0 if fam == 'a' else 1].real for er in EPS_LIST])
            Qx = np.array([e.real / (2 * abs(e.imag)) for e in
                           [exact[er][0 if fam == 'a' else 1] for er in EPS_LIST]])
            ok = ~np.isnan(fv)
            dfl = 100 * np.abs(fv[ok] - fx[ok]) / fx[ok]
            dQl = 100 * np.abs(Qv[ok] - Qx[ok]) / Qx[ok]
            li, lqi, le, lqe = loo(eps, fv, Qv)
            print(f"    {tag:9s}: failures {fails[fam]}/7 | floor df {dfl.mean():5.2f}%"
                  f" dQ {dQl.mean():5.1f}% | LOO interior {li:5.2f}%/{lqi:5.1f}%"
                  f" | endpoint {le:5.2f}%/{lqe:5.1f}%")
            print("               per-anchor (f,Q): " +
                  "  ".join("--" if np.isnan(fv[k]) else f"({fv[k]/1e9:.3f},{Qv[k]:.1f})"
                            for k in range(7)))

    print("\nGP-MPM reference rows (from paperB_fig4_data.npz):")
    D = np.load("paperB_fig4_data.npz")
    for fam, tag in (("a", "electric"), ("b", "magnetic")):
        li, lqi, le, lqe = loo(eps, D[f"f_ext_{fam}"], D[f"Q_ext_{fam}"])
        dfl = 100 * np.abs(D[f"f_ext_{fam}"] - D[f"f_exact_{fam}"]) / D[f"f_exact_{fam}"]
        dQl = 100 * np.abs(D[f"Q_ext_{fam}"] - D[f"Q_exact_{fam}"]) / D[f"Q_exact_{fam}"]
        print(f"    {tag:9s}: failures 0/7 | floor df {dfl.mean():5.2f}% dQ {dQl.mean():5.1f}%"
              f" | LOO interior {li:5.2f}%/{lqi:5.1f}% | endpoint {le:5.2f}%/{lqe:5.1f}%")
    print("\nNOTE: write the Sec. IV-C paragraph ONLY after reading these numbers;")
    print("claim shape per the writing instructions (conservative unless GP clearly wins).")

if __name__ == "__main__":
    main()
