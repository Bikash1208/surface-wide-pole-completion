#!/usr/bin/env python3
"""Exact complex Mie dipole poles of the dielectric sphere (Table II / III reference).

Two modes:

``python compute_exact_mie_poles.py``
    Analytic only (seconds, no FDTD). Prints the exact complex poles of the
    electric (``a1``) and magnetic (``b1``) dipole families for the 24-mm sphere
    at the seven anchor permittivities -- roots of the Mie denominators in the
    complex-frequency plane, i.e. absolute ``(f, Q)`` references.

``python compute_exact_mie_poles.py extract``
    Additionally re-runs the seven anchor FDTD ringdowns (~15 s each), extracts
    both dipole families from the same records with the integrated-energy rule,
    matches them to the exact poles, prints the per-anchor extraction-error and
    selection-success tables, and writes ``paperB_fig4_data.npz`` plus a draft
    figure.

Conventions
-----------
Poles use the ``e^{-i w t}`` convention; ``f0 = Re f`` and ``Q = Re f / (2|Im f|)``.
Mie denominators (Bohren & Huffman), with ``x = (2*pi*f/c)*a`` complex and
``m = sqrt(eps_r)``:

    electric a_l : D_a(x) = m psi_l(mx) xi_l'(x) - xi_l(x) psi_l'(mx)
    magnetic b_l : D_b(x) =   psi_l(mx) xi_l'(x) - m xi_l(x) psi_l'(mx)

Roots are found by Newton iteration seeded from the real-axis modulus peak of
the corresponding Mie coefficient, then branch-tracked by continuation in eps_r.

Public API (imported by run_estimator_baselines.py): ``exact_table``,
``EPS_LIST``, ``A_SPH``, ``C0_``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import jv, hankel1

from constants import C0_M_PER_S as C0_

# Spherical Bessel/Hankel and Riccati-Bessel derivative, complex-argument capable.
# jv/hankel1 (not scipy.special.spherical_jn/yn) are required for complex z.
def _sph_jl(l, z):
    z = np.asarray(z, dtype=complex)
    return np.sqrt(np.pi / (2.0 * z)) * jv(l + 0.5, z)


def _sph_hl1(l, z):
    z = np.asarray(z, dtype=complex)
    return np.sqrt(np.pi / (2.0 * z)) * hankel1(l + 0.5, z)


def _riccati_d(l, z, kind="j"):
    """Derivative ``d/dz [z * z_l(z)] = z*z_{l-1}(z) - l*z_l(z)`` (kind 'j' or 'h')."""
    f = _sph_jl if kind == "j" else _sph_hl1
    return z * f(l - 1, z) - l * f(l, z)


A_SPH = 0.024                      # sphere radius [m] (RC=12 cells x 2 mm)
EPS_LIST = [12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 25.0]
BAND = (0.8e9, 3.0e9)              # wider pole-search band (not the analysis band)


def _psi(l, z):
    return z * _sph_jl(l, z)


def _xi(l, z):
    return z * _sph_hl1(l, z)


def _dpsi(l, z):
    return _riccati_d(l, z, "j")


def _dxi(l, z):
    return _riccati_d(l, z, "h")


def mie_denominator(eps_r, family):
    """Return the Mie denominator ``D(x)`` for the dipole family ('electric'|'magnetic')."""
    m = np.sqrt(eps_r + 0j)
    if family == "electric":
        return lambda x: m * _psi(1, m * x) * _dxi(1, x) - _xi(1, x) * _dpsi(1, m * x)
    return lambda x: _psi(1, m * x) * _dxi(1, x) - m * _xi(1, x) * _dpsi(1, m * x)


def fundamental_seed(eps_r, family):
    """Real-axis frequency [Hz] of the lowest prominent modulus peak (fundamental branch).

    A plain argmax can hop to the second resonance when its peak is marginally
    taller (observed for a1 at eps=25 and b1 at eps=18-22), so the lowest peak
    above half-maximum is selected instead.
    """
    fgrid = np.linspace(*BAND, 4000)
    x = (2 * np.pi * fgrid / C0_) * A_SPH
    m = np.sqrt(eps_r + 0j)
    if family == "electric":
        num = lambda xx: m * _psi(1, m * xx) * _dpsi(1, xx) - _psi(1, xx) * _dpsi(1, m * xx)
    else:
        num = lambda xx: _psi(1, m * xx) * _dpsi(1, xx) - m * _psi(1, xx) * _dpsi(1, m * xx)
    den = mie_denominator(eps_r, family)
    coef = np.abs(np.array([num(xx) / den(xx) for xx in x]))
    peaks = [i for i in range(1, len(coef) - 1)
             if coef[i] > coef[i - 1] and coef[i] > coef[i + 1] and coef[i] > 0.5 * coef.max()]
    return fgrid[min(peaks)] if peaks else fgrid[int(np.argmax(coef))]


def newton_pole(eps_r, family, f_seed):
    """Newton-iterate the complex pole of ``mie_denominator`` from a frequency seed [Hz]."""
    den = mie_denominator(eps_r, family)
    f = complex(f_seed) * (1 - 0.02j) if f_seed.imag == 0 else complex(f_seed)
    for _ in range(80):
        x = (2 * np.pi * f / C0_) * A_SPH
        d = den(x)
        h = 1e-7 * abs(f)
        dp = (den((2 * np.pi * (f + h) / C0_) * A_SPH) - d) / h
        step = d / dp
        f = f - step
        if abs(step) < 1e-3:
            break
    assert f.imag < 0, f"pole in wrong half-plane: {f}"
    return f


def exact_table():
    """Branch-tracked exact poles for every anchor permittivity.

    The first anchor is seeded from the fundamental real-axis peak; each later
    anchor is seeded by continuation from the previous root scaled by
    ``sqrt(eps_prev/eps)``. Poles move smoothly, so continuation cannot change
    branch (asserted via a 5 % drift bound).

    Returns
    -------
    dict
        ``{eps_r: (electric_pole, magnetic_pole)}`` of complex frequencies [Hz].
    """
    out = {er: [None, None] for er in EPS_LIST}
    print(f"EXACT complex Mie dipole poles (fundamental branches, continuation-tracked),"
          f" a = {A_SPH*1e3:.0f} mm sphere:")
    print("   eps_r |  electric a1: f0(GHz)    Q   |  magnetic b1: f0(GHz)    Q")
    for fi, family in enumerate(("electric", "magnetic")):
        root, er_prev = None, None
        for er in EPS_LIST:
            seed = (root * np.sqrt(er_prev / er)) if root is not None \
                else complex(fundamental_seed(er, family))
            root = newton_pole(er, family, seed)
            if er_prev is not None:
                drift = abs(root - seed) / abs(seed)
                assert drift < 0.05, \
                    f"{family} eps={er}: continuation drift {drift:.2f} -- branch jump?"
            out[er][fi] = root
            er_prev = er
    for er in EPS_LIST:
        fa, fb = out[er]
        print(f"   {er:5.1f} |     {fa.real/1e9:6.3f}   {fa.real/(2*abs(fa.imag)):7.1f}"
              f"   |     {fb.real/1e9:6.3f}   {fb.real/(2*abs(fb.imag)):7.1f}")
    return {er: tuple(v) for er, v in out.items()}


def extract_both_families(exact, out_npz="paperB_fig4_data.npz",
                          draft_png="figures/paperB_fig4_draft.png"):
    """Extract both dipole families from FDTD anchor ringdowns and score them.

    Re-runs the seven anchor ringdowns, fits the global poles from the leading
    SVD component of each windowed surface record, selects each family with the
    integrated-energy rule, and compares against ``exact``. Writes ``out_npz``
    (per-family exact/extracted f and Q arrays) and a draft figure.
    """
    import time
    from fdtd_engine import FDTD
    from tfsf import PlaneWaveBox, gaussian_modulated
    from geometry import sphere_mask
    from spherical_recorder import SphericalHuygensRecorder
    from fdtd_extrapolate import mpm_poles

    RC = 12; d_ = 2e-3; npml = 8; buffer = 4; HY_GAP = 6; tf_gap = 3
    # T0=600 opens the window just after the prompt: low-Q electric poles
    # (Q~5 at eps=12, decay ~350 steps) have vanished by step ~1100, so a later
    # window fits noise where the pole used to be (dQ up to 1.7e4% with T0=1100).
    N_CUT = 8000; T0 = 600; WIN = 2400; ORDER = 28; NTH, NPH = 24, 48
    half = RC + HY_GAP + buffer + npml
    N = 2 * half + (2 * half) % 2
    c = N // 2
    center = np.array([c, c, c]) * d_
    lo, hi = c - RC - tf_gap, c + RC + tf_gap
    Rsph = (RC + HY_GAP) * d_
    COMPS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

    rows = {}
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
        Y = {cc: np.asarray(rec.data[cc], float)[T0:T0 + WIN] for cc in COMPS}
        U, S, _ = np.linalg.svd(np.concatenate([Y[cc] for cc in COMPS], 1),
                                full_matrices=False)
        z, _ = mpm_poles(U[:, 0] * S[0], ORDER)
        s = np.log(z) / sim.dt
        Vinv = np.linalg.pinv(z[None, :] ** np.arange(WIN)[:, None])
        res = np.stack([Vinv @ Y[cc] for cc in COMPS], -1)
        amp = np.linalg.norm(res.reshape(z.size, -1), axis=1)
        eint = amp ** 2 / np.maximum(1 - np.abs(z) ** 2, 1e-12)
        fHz = s.imag / (2 * np.pi)
        Q = np.abs(s.imag) / (2 * np.abs(s.real) + 1e-30)
        cand = [m for m in range(z.size)
                if BAND[0] < fHz[m] < BAND[1] and abs(z[m]) < 1.0 and Q[m] >= 3.0]
        fa, fb = exact[er]

        def pick(fx):
            near = [m for m in cand if abs(fHz[m] - fx.real) < 0.10 * fx.real]
            return max(near, key=lambda m: eint[m]) if near else None

        ma, mb = pick(fa), pick(fb)
        # Unsupervised selection audit: naive rankings get the raw band candidates
        # (no Q filter -- that filter is part of the proposed rule). A candidate
        # within 5 % of an exact pole is classified as that family.
        cand_all = [m for m in range(z.size) if BAND[0] < fHz[m] < BAND[1] and abs(z[m]) < 1.0]

        def classify(m):
            if abs(fHz[m] - fa.real) < 0.05 * fa.real:
                return "a1"
            if abs(fHz[m] - fb.real) < 0.05 * fb.real:
                return "b1"
            return "artifact"

        def top2(metric, pool):
            return [classify(m) for m in sorted(pool, key=lambda m: -metric[m])[:2]]

        sel_audit = dict(amp=top2(amp, cand_all), q=top2(Q, cand_all), energy=top2(eint, cand))
        rows[er] = dict(
            exact_a=fa, exact_b=fb, sel=sel_audit,
            ext_a=(fHz[ma], Q[ma]) if ma is not None else None,
            ext_b=(fHz[mb], Q[mb]) if mb is not None else None)
        print(f"  eps {er:5.1f} done in {time.time()-t0:.0f}s", flush=True)

    print("\nEXTRACTED vs EXACT (per family: df%, dQ%):")
    print("   eps_r |  electric df%   dQ%  |  magnetic df%   dQ%")
    data = dict(eps=np.array(EPS_LIST))
    for fam in ("a", "b"):
        fe, Qe, fx, Qx = [], [], [], []
        for er in EPS_LIST:
            r = rows[er]
            ex = r[f"exact_{fam}"]
            fx.append(ex.real); Qx.append(ex.real / (2 * abs(ex.imag)))
            if r[f"ext_{fam}"] is None:
                fe.append(np.nan); Qe.append(np.nan)
            else:
                fe.append(r[f"ext_{fam}"][0]); Qe.append(r[f"ext_{fam}"][1])
        data[f"f_exact_{fam}"] = np.array(fx); data[f"Q_exact_{fam}"] = np.array(Qx)
        data[f"f_ext_{fam}"] = np.array(fe); data[f"Q_ext_{fam}"] = np.array(Qe)
    for er in EPS_LIST:
        r = rows[er]

        def errs(fam):
            ex = r[f"exact_{fam}"]; Qx = ex.real / (2 * abs(ex.imag))
            if r[f"ext_{fam}"] is None:
                return "  not found   "
            fE, QE = r[f"ext_{fam}"]
            return f"{100*abs(fE-ex.real)/ex.real:6.2f} {100*abs(QE-Qx)/Qx:6.1f}"

        print(f"   {er:5.1f} |   {errs('a')}     |   {errs('b')}")

    out_npz = Path(out_npz)
    np.savez(out_npz, **data)
    print(f"\nsaved {out_npz}")

    print("\nSELECTION SUCCESS (unsupervised top-2 per ranking; want {a1, b1}):")
    print("   eps_r |  amplitude        |  max-Q            |  integrated energy")
    wins = dict(amp=0, q=0, energy=0)
    for er in EPS_LIST:
        sel = rows[er]["sel"]
        for k in wins:
            if sorted(sel[k]) == ["a1", "b1"]:
                wins[k] += 1
        print(f"   {er:5.1f} |  {'+'.join(sel['amp']):16s} |  {'+'.join(sel['q']):16s} |  "
              f"{'+'.join(sel['energy'])}")
    print(f"   SCORE  |  amplitude {wins['amp']}/7  |  max-Q {wins['q']}/7  |  "
          f"energy {wins['energy']}/7")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for fam, col, lab in (("a", "tab:blue", "electric $a_1$"),
                          ("b", "tab:red", "magnetic $b_1$")):
        ax[0].plot(EPS_LIST, data[f"f_exact_{fam}"] / 1e9, "x--", color=col,
                   label=f"{lab} (exact Mie)")
        ax[0].plot(EPS_LIST, data[f"f_ext_{fam}"] / 1e9, "o", mfc="none", color=col,
                   label=f"{lab} (extracted)")
        ax[1].plot(EPS_LIST, data[f"Q_exact_{fam}"], "x--", color=col)
        ax[1].plot(EPS_LIST, data[f"Q_ext_{fam}"], "o", mfc="none", color=col)
    ax[0].set_xlabel(r"$\epsilon_r$"); ax[0].set_ylabel("f (GHz)"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel(r"$\epsilon_r$"); ax[1].set_ylabel("Q")
    for a in ax:
        a.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    draft_png = Path(draft_png)
    draft_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(draft_png, dpi=200)
    print(f"saved {draft_png}")


def main() -> None:
    exact = exact_table()
    if len(sys.argv) > 1 and sys.argv[1] == "extract":
        extract_both_families(exact)


if __name__ == "__main__":
    main()
