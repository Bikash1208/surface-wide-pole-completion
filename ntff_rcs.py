"""Near-to-far-field (NTFF) transform -> bistatic RCS, from Huygens-surface fields.

Given the scattered E/H time series recorded on a closed box (HuygensRecorder),
single-frequency phasors are formed by a DFT at f0, converted to equivalent
surface currents Js = n x H, Ms = -n x E, and radiated to the far field. The
radar cross section in direction (theta, phi) is

    sigma = (k0^2 / 4pi) * (|L_phi + eta*N_theta|^2 + |L_theta - eta*N_phi|^2)
                          / |E_inc|^2

with N = integral Js e^{j k0 rhat.r'} dS,  L = integral Ms e^{j k0 rhat.r'} dS.

theta is measured from +z (the incidence/forward direction): theta=180 deg is
monostatic backscatter, theta=0 is forward scatter.
"""
from __future__ import annotations

import numpy as np
from fdtd_engine import ETA0, C0


def _phasor(series: np.ndarray, f0: float, dt: float, axis: int = 0) -> np.ndarray:
    """DFT of a real time series at a single frequency f0 (no dt scaling)."""
    n = np.arange(series.shape[axis])
    w = np.exp(-1j * 2 * np.pi * f0 * n * dt)
    shape = [1] * series.ndim
    shape[axis] = -1
    return (series * w.reshape(shape)).sum(axis=axis)


class FarField:
    """Far-field / RCS evaluator built from a HuygensRecorder (or its HDF5)."""

    def __init__(self, x, y, z, nx, ny, nz, dS,
                 Ex_ph, Ey_ph, Ez_ph, Hx_ph, Hy_ph, Hz_ph):
        self.r = np.stack([x, y, z], axis=1)            # (K,3)
        self.dS = dS
        # surface equivalent currents  Js = n x H,  Ms = -n x E
        n = np.stack([nx, ny, nz], axis=1)
        E = np.stack([Ex_ph, Ey_ph, Ez_ph], axis=1)
        H = np.stack([Hx_ph, Hy_ph, Hz_ph], axis=1)
        self.Js = np.cross(n, H)
        self.Ms = -np.cross(n, E)

    # ---- constructors ----
    @classmethod
    def from_recorder(cls, rec, f0):
        dt = rec.sim.dt
        ph = {c: _phasor(rec.data[c], f0, dt) for c in rec.data}
        return cls(rec.x, rec.y, rec.z, rec.nx, rec.ny, rec.nz, rec.dS,
                   ph["Ex"], ph["Ey"], ph["Ez"], ph["Hx"], ph["Hy"], ph["Hz"])

    @classmethod
    def from_h5(cls, path, f0):
        import h5py
        with h5py.File(path, "r") as h:
            dt = float(h.attrs["dt"])
            d = {c: np.array(h[c]) for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
            x, y, z = np.array(h["x"]), np.array(h["y"]), np.array(h["z"])
            nx, ny, nz = np.array(h["nx"]), np.array(h["ny"]), np.array(h["nz"])
            dS = np.array(h["dS"])
        ph = {c: _phasor(d[c], f0, dt) for c in d}
        return cls(x, y, z, nx, ny, nz, dS,
                   ph["Ex"], ph["Ey"], ph["Ez"], ph["Hx"], ph["Hy"], ph["Hz"])

    # ------------------------------------------------------------------
    def rcs(self, theta, phi, f0, E_inc_amp):
        """Bistatic RCS (m^2) at directions theta, phi (radians, broadcastable).

        E_inc_amp : magnitude of the incident-field phasor at f0 (same DFT
        convention) -- use the measured incident series for self-calibration.
        """
        k0 = 2 * np.pi * f0 / C0
        theta = np.atleast_1d(np.asarray(theta, float))
        phi = np.atleast_1d(np.asarray(phi, float))
        out = np.zeros(theta.shape)
        x, y, z = self.r[:, 0], self.r[:, 1], self.r[:, 2]
        for idx in np.ndindex(theta.shape):
            th, ph = theta[idx], phi[idx]
            st, ct = np.sin(th), np.cos(th)
            sp, cp = np.sin(ph), np.cos(ph)
            rhat = np.array([st * cp, st * sp, ct])
            phase = np.exp(1j * k0 * (x * rhat[0] + y * rhat[1] + z * rhat[2])) * self.dS
            N = (self.Js * phase[:, None]).sum(axis=0)     # (3,)
            L = (self.Ms * phase[:, None]).sum(axis=0)
            N_th = N[0] * ct * cp + N[1] * ct * sp - N[2] * st
            N_ph = -N[0] * sp + N[1] * cp
            L_th = L[0] * ct * cp + L[1] * ct * sp - L[2] * st
            L_ph = -L[0] * sp + L[1] * cp
            amp2 = np.abs(L_ph + ETA0 * N_th) ** 2 + np.abs(L_th - ETA0 * N_ph) ** 2
            out[idx] = (k0 ** 2 / (4 * np.pi)) * amp2 / (E_inc_amp ** 2)
        return out if out.size > 1 else float(out.ravel()[0])

    def backscatter(self, f0, E_inc_amp):
        """Monostatic RCS toward the source (theta=180 deg)."""
        return self.rcs(np.pi, 0.0, f0, E_inc_amp)
