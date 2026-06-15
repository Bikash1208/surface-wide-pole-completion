"""Geometry-agnostic 3D FDTD engine (Yee leapfrog + unsplit CPML).
Level 2: Numba CPU Multi-threading + Float32 Optimization.
"""
from __future__ import annotations
import os
os.environ["KMP_WARNINGS"] = "0"  # Suppress OpenMP deprecation warnings

from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
from numba import njit, prange

C0 = 299_792_458.0
MU0 = 4.0e-7 * np.pi
EPS0 = 8.854_187_8128e-12
ETA0 = np.sqrt(MU0 / EPS0)

# ----------------------------------------------------------------------------
# Numba JIT Accelerated Kernels
# ----------------------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def fast_update_H(Hx, Hy, Hz, Ex, Ey, Ez,
                  psi_Ezy, psi_Eyz, psi_Exz, psi_Ezx, psi_Eyx, psi_Exy,
                  bHx, cHx, kHx, bHy, cHy, kHy, bHz, cHz, kHz,
                  ch, dx, dy, dz, Nx, Ny, Nz,
                  ex, ey, ez, ax, ay, az):
    # --- Hx ---
    for i in prange(Nx):
        for j in range(Ny - 1):
            for k in range(Nz - 1):
                dEz_dy = (Ez[i, j+1, k] * ez[i, j+1, k] - Ez[i, j, k] * ez[i, j, k]) / dy
                dEy_dz = (Ey[i, j, k+1] * ey[i, j, k+1] - Ey[i, j, k] * ey[i, j, k]) / dz
                psi_Ezy[i, j, k] = bHy[j] * psi_Ezy[i, j, k] + cHy[j] * dEz_dy
                psi_Eyz[i, j, k] = bHz[k] * psi_Eyz[i, j, k] + cHz[k] * dEy_dz
                curl = (dEz_dy / kHy[j] + psi_Ezy[i, j, k]) - (dEy_dz / kHz[k] + psi_Eyz[i, j, k])
                Hx[i, j, k] -= (ch * curl) / ax[i, j, k]

    # --- Hy ---
    for i in prange(Nx - 1):
        for j in range(Ny):
            for k in range(Nz - 1):
                dEx_dz = (Ex[i, j, k+1] * ex[i, j, k+1] - Ex[i, j, k] * ex[i, j, k]) / dz
                dEz_dx = (Ez[i+1, j, k] * ez[i+1, j, k] - Ez[i, j, k] * ez[i, j, k]) / dx
                psi_Exz[i, j, k] = bHz[k] * psi_Exz[i, j, k] + cHz[k] * dEx_dz
                psi_Ezx[i, j, k] = bHx[i] * psi_Ezx[i, j, k] + cHx[i] * dEz_dx
                curl = (dEx_dz / kHz[k] + psi_Exz[i, j, k]) - (dEz_dx / kHx[i] + psi_Ezx[i, j, k])
                Hy[i, j, k] -= (ch * curl) / ay[i, j, k]

    # --- Hz ---
    for i in prange(Nx - 1):
        for j in range(Ny - 1):
            for k in range(Nz):
                dEy_dx = (Ey[i+1, j, k] * ey[i+1, j, k] - Ey[i, j, k] * ey[i, j, k]) / dx
                dEx_dy = (Ex[i, j+1, k] * ex[i, j+1, k] - Ex[i, j, k] * ex[i, j, k]) / dy
                psi_Eyx[i, j, k] = bHx[i] * psi_Eyx[i, j, k] + cHx[i] * dEy_dx
                psi_Exy[i, j, k] = bHy[j] * psi_Exy[i, j, k] + cHy[j] * dEx_dy
                curl = (dEy_dx / kHx[i] + psi_Eyx[i, j, k]) - (dEx_dy / kHy[j] + psi_Exy[i, j, k])
                Hz[i, j, k] -= (ch * curl) / az[i, j, k]

@njit(parallel=True, fastmath=True)
def fast_update_E(Ex, Ey, Ez, Hx, Hy, Hz,
                  psi_Hzy, psi_Hyz, psi_Hxz, psi_Hzx, psi_Hyx, psi_Hxy,
                  bEx, cEx, kEx, bEy, cEy, kEy, bEz, cEz, kEz,
                  C1x, C2x, C1y, C2y, C1z, C2z, dx, dy, dz, Nx, Ny, Nz):
    # Per-component update coefficients (C1x/C2x for Ex, etc.). For isotropic
    # media the three pairs are identical; anisotropic subpixel-averaged media
    # give each E-component its own effective permittivity.
    # --- Ex ---
    for i in prange(0, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                dHz_dy = (Hz[i, j, k] - Hz[i, j-1, k]) / dy
                dHy_dz = (Hy[i, j, k] - Hy[i, j, k-1]) / dz
                psi_Hzy[i, j, k] = bEy[j] * psi_Hzy[i, j, k] + cEy[j] * dHz_dy
                psi_Hyz[i, j, k] = bEz[k] * psi_Hyz[i, j, k] + cEz[k] * dHy_dz
                curl = (dHz_dy / kEy[j] + psi_Hzy[i, j, k]) - (dHy_dz / kEz[k] + psi_Hyz[i, j, k])
                Ex[i, j, k] = C1x[i, j, k] * (C2x[i, j, k] * Ex[i, j, k] + curl)

    # --- Ey ---
    for i in prange(1, Nx - 1):
        for j in range(0, Ny - 1):
            for k in range(1, Nz - 1):
                dHx_dz = (Hx[i, j, k] - Hx[i, j, k-1]) / dz
                dHz_dx = (Hz[i, j, k] - Hz[i-1, j, k]) / dx
                psi_Hxz[i, j, k] = bEz[k] * psi_Hxz[i, j, k] + cEz[k] * dHx_dz
                psi_Hzx[i, j, k] = bEx[i] * psi_Hzx[i, j, k] + cEx[i] * dHz_dx
                curl = (dHx_dz / kEz[k] + psi_Hxz[i, j, k]) - (dHz_dx / kEx[i] + psi_Hzx[i, j, k])
                Ey[i, j, k] = C1y[i, j, k] * (C2y[i, j, k] * Ey[i, j, k] + curl)

    # --- Ez ---
    for i in prange(1, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(0, Nz - 1):
                dHy_dx = (Hy[i, j, k] - Hy[i-1, j, k]) / dx
                dHx_dy = (Hx[i, j, k] - Hx[i, j-1, k]) / dy
                psi_Hyx[i, j, k] = bEx[i] * psi_Hyx[i, j, k] + cEx[i] * dHy_dx
                psi_Hxy[i, j, k] = bEy[j] * psi_Hxy[i, j, k] + cEy[j] * dHx_dy
                curl = (dHy_dx / kEx[i] + psi_Hyx[i, j, k]) - (dHx_dy / kEy[j] + psi_Hxy[i, j, k])
                Ez[i, j, k] = C1z[i, j, k] * (C2z[i, j, k] * Ez[i, j, k] + curl)

def _cpml_axis(N: int, d: float, dt: float, npml: int, m: float, kappa_max: float, alpha_max: float):
    sig_max = (m + 1) / (150 * np.pi * d)
    sigE, kE, aE = np.zeros(N), np.ones(N), np.zeros(N)
    sigH, kH, aH = np.zeros(N), np.ones(N), np.zeros(N)

    for i in range(1, npml + 1):
        xE, xH = (npml - i + 1) / npml, (npml - i + 0.5) / npml
        sigE[i - 1], kE[i - 1], aE[i - 1] = sig_max * xE ** m, 1 + (kappa_max - 1) * xE ** m, alpha_max * (1 - xE)
        sigH[i - 1], kH[i - 1], aH[i - 1] = sig_max * xH ** m, 1 + (kappa_max - 1) * xH ** m, alpha_max * (1 - xH)
        
        xE_R, xH_R = i / npml, (i - 0.5) / npml
        idxE = N - npml + i - 1            
        sigE[idxE], kE[idxE], aE[idxE] = sig_max * xE_R ** m, 1 + (kappa_max - 1) * xE_R ** m, alpha_max * (1 - xE_R)
        idxH = N - npml + i - 2            
        if idxH >= 0:
            sigH[idxH], kH[idxH], aH[idxH] = sig_max * xH_R ** m, 1 + (kappa_max - 1) * xH_R ** m, alpha_max * (1 - xH_R)

    bE = np.exp(-(sigE / kE + aE) * (dt / EPS0))
    cE = sigE / (kE * (sigE + kE * aE) + 1e-20) * (bE - 1)
    bH = np.exp(-(sigH / kH + aH) * (dt / EPS0))
    cH = sigH / (kH * (sigH + kH * aH) + 1e-20) * (bH - 1)
    
    return (bE.astype(np.float32), cE.astype(np.float32), kE.astype(np.float32), 
            bH.astype(np.float32), cH.astype(np.float32), kH.astype(np.float32))

@dataclass
class FDTD:
    Nx: int; Ny: int; Nz: int
    dx: float; dy: float; dz: float
    npml: int = 10; cfl: float = 0.9; m: float = 4.0; kappa_max: float = 7.0; alpha_max: float = 0.15
    dt: float = field(init=False)

    def __post_init__(self):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        self.dt = self.cfl / (C0 * np.sqrt((1 / self.dx) ** 2 + (1 / self.dy) ** 2 + (1 / self.dz) ** 2))

        z = lambda: np.zeros((Nx, Ny, Nz), dtype=np.float32)
        self.Ex, self.Ey, self.Ez = z(), z(), z()
        self.Hx, self.Hy, self.Hz = z(), z(), z()

        self.eps = np.full((Nx, Ny, Nz), EPS0, dtype=np.float32)
        self.sigma = z()
        self.pec = None
        self.conformal = None
        self._eps_comp = None        # optional (eps_x, eps_y, eps_z) for subpixel
        self._coeffs_ready = False
        
        self._ex, self._ey, self._ez = z() + 1.0, z() + 1.0, z() + 1.0
        self._ax, self._ay, self._az = z() + 1.0, z() + 1.0, z() + 1.0

        (self.bEx, self.cEx, self.kEx, self.bHx, self.cHx, self.kHx) = _cpml_axis(Nx, self.dx, self.dt, self.npml, self.m, self.kappa_max, self.alpha_max)
        (self.bEy, self.cEy, self.kEy, self.bHy, self.cHy, self.kHy) = _cpml_axis(Ny, self.dy, self.dt, self.npml, self.m, self.kappa_max, self.alpha_max)
        (self.bEz, self.cEz, self.kEz, self.bHz, self.cHz, self.kHz) = _cpml_axis(Nz, self.dz, self.dt, self.npml, self.m, self.kappa_max, self.alpha_max)

        self.psi_Ezy, self.psi_Eyz = z(), z()   
        self.psi_Exz, self.psi_Ezx = z(), z()   
        self.psi_Eyx, self.psi_Exy = z(), z()   
        self.psi_Hzy, self.psi_Hyz = z(), z()   
        self.psi_Hxz, self.psi_Hzx = z(), z()   
        self.psi_Hyx, self.psi_Hxy = z(), z()   

        self.energy_log: list[tuple[int, float]] = []

    def set_material(self, mask: np.ndarray, eps_r: float = 1.0, sigma: float = 0.0):
        self.eps[mask] = np.float32(EPS0 * eps_r)
        self.sigma[mask] = np.float32(sigma)
        self._coeffs_ready = False

    def set_material_array(self, eps_r: np.ndarray, sigma: Optional[np.ndarray] = None):
        self.eps = np.float32(EPS0 * np.asarray(eps_r))
        if sigma is not None: self.sigma = np.float32(sigma)
        self._eps_comp = None
        self._coeffs_ready = False

    def set_material_subpixel(self, eps_x, eps_y, eps_z):
        """Anisotropic subpixel-averaged permittivity: a separate effective
        eps_r for each E-component (arrays in units of eps_r, i.e. relative).
        Use with geometry.sphere_eps_subpixel to remove high-contrast
        staircasing without finer meshing."""
        self._eps_comp = (np.float32(EPS0 * np.asarray(eps_x)),
                          np.float32(EPS0 * np.asarray(eps_y)),
                          np.float32(EPS0 * np.asarray(eps_z)))
        # keep a scalar eps (mean) for energy diagnostics
        self.eps = np.float32((self._eps_comp[0] + self._eps_comp[1] + self._eps_comp[2]) / 3.0)
        self._coeffs_ready = False

    def set_pec(self, mask: np.ndarray):
        self.pec = np.asarray(mask, dtype=bool)

    def set_conformal_pec(self, fractions: dict, area_min: float = 0.5):
        self.conformal = dict(fractions)
        self._ex = self.conformal["ex"].astype(np.float32)
        self._ey = self.conformal["ey"].astype(np.float32)
        self._ez = self.conformal["ez"].astype(np.float32)
        self._ax = np.maximum(self.conformal["ax"], area_min).astype(np.float32)
        self._ay = np.maximum(self.conformal["ay"], area_min).astype(np.float32)
        self._az = np.maximum(self.conformal["az"], area_min).astype(np.float32)

    def finalize_material(self):
        def coeffs(eps):
            c1 = (1.0 / (eps / self.dt + self.sigma / 2.0)).astype(np.float32)
            c2 = (eps / self.dt - self.sigma / 2.0).astype(np.float32)
            return c1, c2
        if self._eps_comp is not None:
            self.C1x, self.C2x = coeffs(self._eps_comp[0])
            self.C1y, self.C2y = coeffs(self._eps_comp[1])
            self.C1z, self.C2z = coeffs(self._eps_comp[2])
        else:
            c1, c2 = coeffs(self.eps)
            self.C1x = self.C1y = self.C1z = c1
            self.C2x = self.C2y = self.C2z = c2
        # back-compat aliases
        self.C1, self.C2 = self.C1x, self.C2x
        self._coeffs_ready = True

    def _apply_pec(self):
        if self.pec is not None:
            self.Ex[self.pec] = 0.0
            self.Ey[self.pec] = 0.0
            self.Ez[self.pec] = 0.0

    def update_H(self):
        ch = np.float32(self.dt / MU0)
        fast_update_H(
            self.Hx, self.Hy, self.Hz, self.Ex, self.Ey, self.Ez,
            self.psi_Ezy, self.psi_Eyz, self.psi_Exz, self.psi_Ezx, self.psi_Eyx, self.psi_Exy,
            self.bHx, self.cHx, self.kHx, self.bHy, self.cHy, self.kHy, self.bHz, self.cHz, self.kHz,
            ch, np.float32(self.dx), np.float32(self.dy), np.float32(self.dz),
            self.Nx, self.Ny, self.Nz,
            self._ex, self._ey, self._ez, self._ax, self._ay, self._az
        )

    def update_E(self):
        if not self._coeffs_ready: self.finalize_material()
        fast_update_E(
            self.Ex, self.Ey, self.Ez, self.Hx, self.Hy, self.Hz,
            self.psi_Hzy, self.psi_Hyz, self.psi_Hxz, self.psi_Hzx, self.psi_Hyx, self.psi_Hxy,
            self.bEx, self.cEx, self.kEx, self.bEy, self.cEy, self.kEy, self.bEz, self.cEz, self.kEz,
            self.C1x, self.C2x, self.C1y, self.C2y, self.C1z, self.C2z,
            np.float32(self.dx), np.float32(self.dy), np.float32(self.dz),
            self.Nx, self.Ny, self.Nz
        )

    def total_energy(self) -> float:
        vol = self.dx * self.dy * self.dz
        uE = 0.5 * np.sum(self.eps * (self.Ex ** 2 + self.Ey ** 2 + self.Ez ** 2)) * vol
        uH = 0.5 * MU0 * np.sum(self.Hx ** 2 + self.Hy ** 2 + self.Hz ** 2) * vol
        return float(uE + uH)

    def run(self, n_steps: int, source_H=None, source_E=None, recorder=None, energy_every=0, progress_every=0):
        if not self._coeffs_ready: self.finalize_material()
        for n in range(n_steps):
            self.update_H()
            if source_H is not None: source_H(self, n)
            self.update_E()
            if source_E is not None: source_E(self, n)
            self._apply_pec()
            if recorder is not None: recorder.record(self, n)
            if energy_every and (n + 1) % energy_every == 0:
                self.energy_log.append((n + 1, self.total_energy()))
            if progress_every and (n + 1) % progress_every == 0:
                print(f"  step {n + 1}/{n_steps}  energy={self.total_energy():.3e} J")