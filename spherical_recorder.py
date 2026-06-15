"""Spherical Huygens-surface recorder (Phase 4A).

Same physics as the cube HuygensRecorder, different sampling geometry: records
Ex..Hz on a sphere via trilinear interpolation from the Yee grid. A sphere maps to
itself under rotation, which is what makes rotation-synthesis of incidences
(Phase 4B) possible. Stores node coords, radial normals, spherical area weights dS,
and the six field components -- nothing new physically.

Yee staggering (matched to geometry.HuygensRecorder._colocate):
  Ex@(i+.5, j,   k  )  Ey@(i,   j+.5, k  )  Ez@(i,   j,   k+.5)
  Hx@(i,   j+.5, k+.5) Hy@(i+.5, j,   k+.5) Hz@(i+.5, j+.5, k  )
"""
from __future__ import annotations
import numpy as np

_OFF = {"Ex": (0.5, 0.0, 0.0), "Ey": (0.0, 0.5, 0.0), "Ez": (0.0, 0.0, 0.5),
        "Hx": (0.0, 0.5, 0.5), "Hy": (0.5, 0.0, 0.5), "Hz": (0.5, 0.5, 0.0)}
_COMPS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


class SphericalHuygensRecorder:
    def __init__(self, sim, center, R, n_theta, n_phi, n_steps, dtype=np.float32):
        self.sim = sim; self.Nt = int(n_steps); self.R = float(R)
        cx, cy, cz = center
        dth = np.pi / n_theta; dph = 2 * np.pi / n_phi
        th = (np.arange(n_theta) + 0.5) * dth          # avoid the poles
        ph = np.arange(n_phi) * dph
        TH, PH = np.meshgrid(th, ph, indexing="ij")
        TH = TH.ravel(); PH = PH.ravel()
        st, ct = np.sin(TH), np.cos(TH)
        ux, uy, uz = st * np.cos(PH), st * np.sin(PH), ct       # outward radial unit
        self.x = cx + R * ux; self.y = cy + R * uy; self.z = cz + R * uz
        self.nx, self.ny, self.nz = ux, uy, uz                  # radial normals
        self.dS = (R ** 2) * st * dth * dph                     # spherical area element
        self.K = self.x.size

        # precompute trilinear gather (flat indices + weights) per component
        Nx, Ny, Nz = sim.Nx, sim.Ny, sim.Nz
        self._idx = {}; self._w = {}
        gx, gy, gz = self.x / sim.dx, self.y / sim.dy, self.z / sim.dz
        for cmp in _COMPS:
            ox, oy, oz = _OFF[cmp]
            fx, fy, fz = gx - ox, gy - oy, gz - oz
            i0 = np.clip(np.floor(fx).astype(int), 0, Nx - 2)
            j0 = np.clip(np.floor(fy).astype(int), 0, Ny - 2)
            k0 = np.clip(np.floor(fz).astype(int), 0, Nz - 2)
            wx, wy, wz = fx - i0, fy - j0, fz - k0
            idx = np.empty((self.K, 8), np.int64); w = np.empty((self.K, 8))
            n = 0
            for di in (0, 1):
                for dj in (0, 1):
                    for dk in (0, 1):
                        idx[:, n] = ((i0 + di) * Ny + (j0 + dj)) * Nz + (k0 + dk)
                        w[:, n] = (wx if di else 1 - wx) * (wy if dj else 1 - wy) * (wz if dk else 1 - wz)
                        n += 1
            self._idx[cmp] = idx; self._w[cmp] = w

        self.data = {cmp: np.zeros((self.Nt, self.K), dtype=dtype) for cmp in _COMPS}
        self.t = np.arange(self.Nt) * sim.dt

    def _interp(self, cmp):
        A = getattr(self.sim, cmp).ravel()
        return (A[self._idx[cmp]] * self._w[cmp]).sum(axis=1)

    def record(self, sim, n: int):
        if n >= self.Nt: return
        for cmp in _COMPS:
            self.data[cmp][n] = self._interp(cmp)

    def far_field(self, f0):
        from ntff_rcs import FarField
        return FarField.from_recorder(self, f0)
