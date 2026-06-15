"""FDTD ringdown extrapolation (Group-D / D2 accelerator).

Idea (physics)
--------------
After the incident pulse ends, the scatterer rings at its own natural modes —
a sum of a few *damped sinusoids* (complex poles). That tail carries no new
physics, yet it forces FDTD to run many extra timesteps for high-Q objects.
So: run FDTD only through the driven burst + a slice of ringdown, fit the poles
from that early window with the Matrix-Pencil Method (MPM), and extrapolate the
rest analytically. No ML, no permittivity interpolation — a pure per-εr speedup.

Validated on this dataset: cutting at ~30% of the timesteps reproduces the
RCS-relevant DFT amplitude at 0.30 GHz to ~0.2-7% (median ~5%) across εr=3..30.

Typical use
-----------
1. Run FDTD only to `cut` timesteps (instead of the full `n_total`).
2. extrapolate_h5(in_truncated.h5, out_full.h5, n_total=1738, cut=520)
3. Feed out_full.h5 to the usual NTFF/RCS pipeline.

API
---
- mpm_poles(y, order)            -> (poles z, residues R)
- extrapolate_signal(y, n_total, cut, order) -> full length-n_total real signal
- extrapolate_h5(in_h5, out_h5, n_total, cut, order, comps)
"""
from __future__ import annotations
import os, logging
from typing import Optional, Sequence, Tuple
import numpy as np

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None

_COMPS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


# ----------------------------------------------------------------------------
# Matrix-Pencil pole estimation
# ----------------------------------------------------------------------------
def mpm_poles(y: np.ndarray, order: int = 24, pencil_frac: float = 1 / 3) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate complex poles z and residues R so that y[n] ≈ Σ R_i z_i^n.

    Matrix-Pencil Method: robust for sums of damped sinusoids. Poles with
    |z|>1 (unstable / would blow up on extrapolation) are reflected inside the
    unit circle, then residues are re-fit by least squares.
    """
    y = np.asarray(y, dtype=float).ravel()
    N = y.size
    L = max(2, int(N * pencil_frac))
    if N - L < 2:
        return np.zeros(0, complex), np.zeros(0, complex)
    H = np.stack([y[i:i + L + 1] for i in range(N - L)])     # Hankel (N-L, L+1)
    _, sv, Vh = np.linalg.svd(H, full_matrices=False)
    V = Vh.conj().T
    M = int(min(order, V.shape[1] - 1))
    if M < 1:
        return np.zeros(0, complex), np.zeros(0, complex)
    Vp = V[:, :M]
    z = np.linalg.eigvals(np.linalg.pinv(Vp[:-1]) @ Vp[1:])
    z = np.where(np.abs(z) > 1.0, 1.0 / np.conj(z), z)       # enforce stability
    n = np.arange(N)
    Z = z[None, :] ** n[:, None]
    R, *_ = np.linalg.lstsq(Z, y.astype(complex), rcond=None)
    return z, R


def _reconstruct(z: np.ndarray, R: np.ndarray, n0: int, n1: int) -> np.ndarray:
    """Real part of Σ R_i z_i^n for n in [n0, n1)."""
    if z.size == 0:
        return np.zeros(n1 - n0)
    n = np.arange(n0, n1)
    return (R[None, :] * (z[None, :] ** n[:, None])).sum(axis=1).real


def extrapolate_signal(y_cut: np.ndarray, n_total: int, cut: Optional[int] = None,
                       order: int = 24) -> np.ndarray:
    """Return a length-`n_total` signal: measured part kept, tail pole-extrapolated.

    y_cut : the early (truncated) FDTD samples for one node/component.
    Poles are fit on the ringdown only (from the energy peak to `cut`), so the
    driven-to-free transition doesn't bias the decay-rate estimate.
    """
    y_cut = np.asarray(y_cut, dtype=float).ravel()
    cut = int(cut or y_cut.size)
    cut = min(cut, y_cut.size)
    pk = int((y_cut[:cut] ** 2).argmax())
    z, R = mpm_poles(y_cut[pk:cut], order=order)
    out = np.empty(n_total)
    out[:cut] = y_cut[:cut]
    # reconstruct from the SAME pole-time origin (n measured from pk)
    out[cut:] = _reconstruct(z, R, cut - pk, n_total - pk)
    return out


# ----------------------------------------------------------------------------
# Whole-enclosure HDF5 extrapolation
# ----------------------------------------------------------------------------
def extrapolate_h5(in_h5: str, out_h5: str, n_total: int, cut: Optional[int] = None,
                   order: int = 24, comps: Sequence[str] = _COMPS,
                   dt: Optional[float] = None) -> str:
    """Extrapolate every node/component of a (truncated) FDTD HDF5 to n_total steps.

    Writes an output HDF5 with the same layout (t, Ex..Hz, coords, attrs) so the
    existing NTFF→RCS pipeline can consume it unchanged.
    """
    if h5py is None:
        raise RuntimeError("h5py is required for extrapolate_h5")
    with h5py.File(in_h5, "r") as h:
        data = {c: np.array(h[c]) for c in comps if c in h}
        t_in = np.array(h["t"]).ravel() if "t" in h else None
        extras = {k: np.array(h[k]) for k in ("plane_id", "x", "y", "z") if k in h}
        attrs = dict(h.attrs)
    Nt_in, K = next(iter(data.values())).shape
    cut = int(cut or Nt_in)
    if dt is None and t_in is not None and t_in.size >= 2:
        dt = float(np.median(np.diff(t_in)))
    logging.info("Extrapolating %s: %d nodes x %d comps, cut=%d -> n_total=%d (%.0f%% of steps used)",
                 os.path.basename(in_h5), K, len(data), cut, n_total, 100 * cut / n_total)

    out = {}
    for c, arr in data.items():
        full = np.empty((n_total, K), dtype=np.float64)
        for k in range(K):
            full[:, k] = extrapolate_signal(arr[:, k], n_total, cut=cut, order=order)
        out[c] = full

    t_full = (np.arange(n_total) * dt) if dt else np.arange(n_total, dtype=float)
    os.makedirs(os.path.dirname(os.path.abspath(out_h5)), exist_ok=True)
    with h5py.File(out_h5, "w") as h:
        for c, full in out.items():
            h.create_dataset(c, data=full)
        h.create_dataset("t", data=t_full.reshape(1, -1))
        for k, v in extras.items():
            h.create_dataset(k, data=v)
        for k, v in attrs.items():
            try:
                h.attrs[k] = v
            except Exception:
                pass
        h.attrs["extrapolated_from_cut"] = cut
        h.attrs["extrapolation_order"] = order
    logging.info("Wrote extrapolated fields -> %s", out_h5)
    return out_h5


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="FDTD ringdown extrapolation (Matrix-Pencil).")
    ap.add_argument("in_h5"); ap.add_argument("out_h5")
    ap.add_argument("--n_total", type=int, required=True)
    ap.add_argument("--cut", type=int, default=None)
    ap.add_argument("--order", type=int, default=24)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    extrapolate_h5(a.in_h5, a.out_h5, a.n_total, cut=a.cut, order=a.order)
