"""Geometry helpers and the Huygens-surface recorder for the FDTD engine.
Level 2: Standard NumPy (PyTorch dependencies removed).
"""
from __future__ import annotations
from typing import Optional
import numpy as np

def grid_coords(sim):
    x = np.arange(sim.Nx) * sim.dx
    y = np.arange(sim.Ny) * sim.dy
    z = np.arange(sim.Nz) * sim.dz
    return x, y, z

def sphere_mask(sim, center_m, radius_m) -> np.ndarray:
    x, y, z = grid_coords(sim)
    cx, cy, cz = center_m
    X, Y, Z = np.meshgrid(x - cx, y - cy, z - cz, indexing="ij")
    return (X ** 2 + Y ** 2 + Z ** 2) <= radius_m ** 2

def sphere_eps_smooth(sim, center_m, radius_m, eps_r, sub: int = 4) -> np.ndarray:
    x, y, z = grid_coords(sim)
    cx, cy, cz = center_m
    off = (np.arange(sub) + 0.5) / sub - 0.5
    ox = off * sim.dx; oy = off * sim.dy; oz = off * sim.dz
    frac = np.zeros((sim.Nx, sim.Ny, sim.Nz))
    r2 = radius_m ** 2
    for ax in ox:
        for ay in oy:
            for az in oz:
                X, Y, Z = np.meshgrid(x + ax - cx, y + ay - cy, z + az - cz, indexing="ij")
                frac += ((X ** 2 + Y ** 2 + Z ** 2) <= r2)
    frac /= sub ** 3
    return 1.0 + (eps_r - 1.0) * frac

def sphere_eps_subpixel(sim, center_m, radius_m, eps_r, sub: int = 4):
    """Anisotropic subpixel-averaged permittivity for a dielectric sphere.

    Returns (eps_x, eps_y, eps_z): a separate effective eps_r per E-component,
    sampled at that component's staggered Yee location. At a boundary cell the
    effective permittivity is the Meep-style tensor projection
        1/eps_eff,i = n_i^2 <1/eps> + (1 - n_i^2) / <eps>
    with n the radial (interface) normal, <eps> the volume-arithmetic mean and
    <1/eps> the volume mean of 1/eps. The component NORMAL to the interface sees
    the harmonic mean (correct for the discontinuous normal D = eps*E); tangential
    components see the arithmetic mean. Removes most high-contrast staircasing
    without a finer mesh.

    Use:  sim.set_material_subpixel(*sphere_eps_subpixel(sim, c, a, eps_r))
    """
    Nx, Ny, Nz = sim.Nx, sim.Ny, sim.Nz
    dx, dy, dz = sim.dx, sim.dy, sim.dz
    cx, cy, cz = center_m
    ein, eout = float(eps_r), 1.0
    off = (np.arange(sub) + 0.5) / sub - 0.5
    r2 = radius_m ** 2

    def comp(axis):
        xs = (np.arange(Nx) + (0.5 if axis == 0 else 0.0)) * dx - cx
        ys = (np.arange(Ny) + (0.5 if axis == 1 else 0.0)) * dy - cy
        zs = (np.arange(Nz) + (0.5 if axis == 2 else 0.0)) * dz - cz
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        f = np.zeros((Nx, Ny, Nz))
        for ax_ in off * dx:
            for ay_ in off * dy:
                for az_ in off * dz:
                    f += ((X + ax_) ** 2 + (Y + ay_) ** 2 + (Z + az_) ** 2) <= r2
        f /= sub ** 3
        rr = np.sqrt(X ** 2 + Y ** 2 + Z ** 2) + 1e-30
        n_ax = (X if axis == 0 else Y if axis == 1 else Z) / rr
        arith = f * ein + (1 - f) * eout
        inv = f / ein + (1 - f) / eout
        inv_eff = n_ax ** 2 * inv + (1 - n_ax ** 2) / arith
        return 1.0 / inv_eff

    return comp(0), comp(1), comp(2)


def sphere_inside(center_m, radius_m):
    cx, cy, cz = center_m; r2 = radius_m ** 2
    def inside(X, Y, Z): return (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= r2
    return inside

def conformal_fractions(sim, inside, sub: int = 8):
    Nx, Ny, Nz, dx, dy, dz = sim.Nx, sim.Ny, sim.Nz, sim.dx, sim.dy, sim.dz
    xi = np.arange(Nx); yj = np.arange(Ny); zk = np.arange(Nz)
    s = (np.arange(sub) + 0.5) / sub
    X = lambda off: ((xi[:, None, None] + off) * dx)
    Y = lambda off: ((yj[None, :, None] + off) * dy)
    Z = lambda off: ((zk[None, None, :] + off) * dz)
    x0, y0, z0 = X(0.0), Y(0.0), Z(0.0)
    
    ex = np.zeros((Nx, Ny, Nz)); ey = np.zeros_like(ex); ez = np.zeros_like(ex)
    for a in s:
        ex += ~inside(X(a), y0, z0)
        ey += ~inside(x0, Y(a), z0)
        ez += ~inside(x0, y0, Z(a))
    ex /= sub; ey /= sub; ez /= sub
    
    ax = np.zeros((Nx, Ny, Nz)); ay = np.zeros_like(ax); az = np.zeros_like(ax)
    for a in s:
        for b in s:
            ax += ~inside(x0, Y(a), Z(b))
            ay += ~inside(X(a), y0, Z(b))
            az += ~inside(X(a), Y(b), z0)
    ax /= sub * sub; ay /= sub * sub; az /= sub * sub
    return dict(ex=ex, ey=ey, ez=ez, ax=ax, ay=ay, az=az)

def box_mask(sim, lo_m, hi_m) -> np.ndarray:
    x, y, z = grid_coords(sim)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return ((X >= lo_m[0]) & (X <= hi_m[0]) & (Y >= lo_m[1]) & (Y <= hi_m[1]) & (Z >= lo_m[2]) & (Z <= hi_m[2]))

_FACES = {0: (0, +1), 1: (0, -1), 2: (1, +1), 3: (1, -1), 4: (2, +1), 5: (2, -1)}

class HuygensRecorder:
    def __init__(self, sim, i0, i1, j0, j1, k0, k1, n_steps, dtype=np.float32, mode="full", f0=None):
        self.sim = sim
        self.b = (int(i0), int(i1), int(j0), int(j1), int(k0), int(k1))
        self.Nt = int(n_steps)
        self.mode = mode
        self.f0 = f0

        pid, xs, ys, zs, nxs, nys, nzs, dS = [], [], [], [], [], [], [], []
        self._pts = []
        dx, dy, dz = sim.dx, sim.dy, sim.dz
        dA = {0: dy * dz, 1: dx * dz, 2: dx * dy}
        
        for fid, (axis, side) in _FACES.items():
            ii, jj, kk = self._face_indices(fid)
            for (i, j, k) in zip(ii.ravel(), jj.ravel(), kk.ravel()):
                self._pts.append((fid, i, j, k))
                pid.append(fid)
                xs.append(i * dx); ys.append(j * dy); zs.append(k * dz)
                n = [0.0, 0.0, 0.0]; n[axis] = float(side)
                nxs.append(n[0]); nys.append(n[1]); nzs.append(n[2])
                dS.append(dA[axis])
                
        self.plane_id = np.array(pid, dtype=np.int8)
        self.x = np.array(xs); self.y = np.array(ys); self.z = np.array(zs)
        self.nx = np.array(nxs); self.ny = np.array(nys); self.nz = np.array(nzs)
        self.dS = np.array(dS)
        self.K = len(self._pts)

        self._I = np.array([p[1] for p in self._pts])
        self._J = np.array([p[2] for p in self._pts])
        self._K = np.array([p[3] for p in self._pts])

        self._comps = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        
        if self.mode == "full":
            self.data = {c: np.zeros((self.Nt, self.K), dtype=dtype) for c in self._comps}
            self.t = np.arange(self.Nt) * sim.dt
        else:
            if self.f0 is None: raise ValueError("phasor mode needs f0")
            self.phasor = {c: np.zeros(self.K, dtype=complex) for c in self._comps}
            self._w = -1j * 2 * np.pi * self.f0 * sim.dt

    def _face_indices(self, fid):
        i0, i1, j0, j1, k0, k1 = self.b
        axis, side = _FACES[fid]
        if axis == 0:
            i = i1 if side > 0 else i0
            J, K = np.meshgrid(np.arange(j0, j1 + 1), np.arange(k0, k1 + 1), indexing="ij")
            return np.full_like(J, i), J, K
        if axis == 1:
            j = j1 if side > 0 else j0
            I, K = np.meshgrid(np.arange(i0, i1 + 1), np.arange(k0, k1 + 1), indexing="ij")
            return I, np.full_like(I, j), K
        k = k1 if side > 0 else k0
        I, J = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
        return I, J, np.full_like(I, k)

    def _colocate(self):
        s = self.sim
        I, J, K = self._I, self._J, self._K
        Ex = 0.5 * (s.Ex[I, J, K] + s.Ex[I - 1, J, K])
        Ey = 0.5 * (s.Ey[I, J, K] + s.Ey[I, J - 1, K])
        Ez = 0.5 * (s.Ez[I, J, K] + s.Ez[I, J, K - 1])
        Hx = 0.25 * (s.Hx[I, J, K] + s.Hx[I, J - 1, K] + s.Hx[I, J, K - 1] + s.Hx[I, J - 1, K - 1])
        Hy = 0.25 * (s.Hy[I, J, K] + s.Hy[I - 1, J, K] + s.Hy[I, J, K - 1] + s.Hy[I - 1, J, K - 1])
        Hz = 0.25 * (s.Hz[I, J, K] + s.Hz[I - 1, J, K] + s.Hz[I, J - 1, K] + s.Hz[I - 1, J - 1, K])
        return Ex, Ey, Ez, Hx, Hy, Hz

    def record(self, sim, n: int):
        if n >= self.Nt: return
        vals = dict(zip(self._comps, self._colocate()))
        if self.mode == "full":
            for c in self._comps: self.data[c][n] = vals[c]
        else:
            wn = np.exp(self._w * n)
            for c in self._comps: self.phasor[c] += vals[c] * wn

    def far_field(self):
        from ntff_rcs import FarField
        if self.mode != "phasor": raise RuntimeError("far_field() requires mode='phasor'")
        p = self.phasor
        return FarField(self.x, self.y, self.z, self.nx, self.ny, self.nz, self.dS,
                        p["Ex"], p["Ey"], p["Ez"], p["Hx"], p["Hy"], p["Hz"])

    def save(self, path: str, attrs: Optional[dict] = None):
        import h5py
        s = self.sim
        with h5py.File(path, "w") as h:
            for c, arr in self.data.items(): h.create_dataset(c, data=arr)
            h.create_dataset("t", data=self.t.reshape(1, -1))
            h.create_dataset("plane_id", data=self.plane_id)
            h.create_dataset("x", data=self.x); h.create_dataset("y", data=self.y)
            h.create_dataset("z", data=self.z)
            h.create_dataset("nx", data=self.nx); h.create_dataset("ny", data=self.ny)
            h.create_dataset("nz", data=self.nz)
            h.create_dataset("dS", data=self.dS)
            h.attrs.update(dict(dx=s.dx, dy=s.dy, dz=s.dz, dt=s.dt, Nt=self.Nt, npml=s.npml, K=self.K))
            if attrs:
                for k, v in attrs.items(): h.attrs[k] = v
        return path