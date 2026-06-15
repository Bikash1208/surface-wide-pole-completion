"""Total-field / scattered-field (TF-SF) plane-wave source for the FDTD engine.
Level 2: Numba CPU Multi-threading + Float32 Optimization.
"""
from __future__ import annotations
from typing import Callable
import numpy as np
from numba import njit, prange
from fdtd_engine import MU0, EPS0, C0

def gaussian_modulated(f0: float, fbw: float, dt: float) -> Callable[[int], float]:
    tau, t0 = 1.2 / fbw, 4.5 * (1.2 / fbw)
    def f(n: int) -> float:
        t = n * dt
        return float(np.exp(-((t - t0) / tau) ** 2) * np.sin(2 * np.pi * f0 * t))
    return f

@njit(fastmath=True)
def fast_step_aux_H(hy1, ex1, cH):
    for k in range(len(hy1) - 1):
        hy1[k] -= cH * (ex1[k+1] - ex1[k])

@njit(fastmath=True)
def fast_step_aux_E(ex1, hy1, cE, mur, h0, h1, hN, hNm1, ks, e_inc):
    e1, eNm1 = ex1[1], ex1[-2]
    for k in range(1, len(ex1) - 1):
        ex1[k] -= cE * (hy1[k] - hy1[k-1])
    ex1[ks] += e_inc
    ex1[0] = h1 + mur * (ex1[1] - h0)
    ex1[-1] = hNm1 + mur * (ex1[-2] - hN)
    return ex1[1], e1, ex1[-2], eNm1

@njit(parallel=True, fastmath=True)
def fast_correct_H(Hy, Hz, ia, ib, ja, jb, ka, kb, cz, cy, ex1, koff):
    ex_ka, ex_kb1 = ex1[ka - koff], ex1[kb + 1 - koff]
    for i in prange(ia, ib + 1):
        for j in range(ja, jb + 1):
            Hy[i, j, ka - 1] += cz * ex_ka
            Hy[i, j, kb]     -= cz * ex_kb1
    for i in prange(ia, ib + 1):
        for k in range(ka, kb + 1):
            ex_k = ex1[k - koff]
            Hz[i, ja - 1, k] -= cy * ex_k
            Hz[i, jb, k]     += cy * ex_k

@njit(parallel=True, fastmath=True)
def fast_correct_E(Ex, Ez, C1, ia, ib, ja, jb, ka, kb, dz, dx, hy1, koff):
    hy_kam1_dz, hy_kb_dz = hy1[ka - 1 - koff] / dz, hy1[kb - koff] / dz
    for i in prange(ia, ib + 1):
        for j in range(ja, jb + 1):
            Ex[i, j, ka]     += C1[i, j, ka] * hy_kam1_dz
            Ex[i, j, kb + 1] -= C1[i, j, kb + 1] * hy_kb_dz
    for j in prange(ja, jb + 1):
        for k in range(ka, kb + 1):
            hy_k_dx = hy1[k - koff] / dx
            Ez[ia, j, k]     -= C1[ia, j, k] * hy_k_dx
            Ez[ib + 1, j, k] += C1[ib + 1, j, k] * hy_k_dx


class PlaneWaveBox:
    def __init__(self, sim, ia, ib, ja, jb, ka, kb, waveform: Callable[[int], float], amplitude: float = 1.0):
        self.sim = sim
        self.ia, self.ib, self.ja, self.jb, self.ka, self.kb = int(ia), int(ib), int(ja), int(jb), int(ka), int(kb)
        self.wf, self.amp = waveform, float(amplitude)
        
        self.Nz1 = sim.Nz + 4
        self.ex1 = np.zeros(self.Nz1, dtype=np.float32)      
        self.hy1 = np.zeros(self.Nz1, dtype=np.float32)      
        self.ks = 1                        
        
        self.cE = np.float32(sim.dt / (EPS0 * sim.dz))
        self.cH = np.float32(sim.dt / (MU0 * sim.dz))
        self.mur = np.float32((C0 * sim.dt - sim.dz) / (C0 * sim.dt + sim.dz))
        self._h0 = self._h1 = self._hN = self._hNm1 = np.float32(0.0)
        self.koff = (self.ka - 1) - self.ks

    def correct_H(self, sim, n: int):
        fast_step_aux_H(self.hy1, self.ex1, self.cH)
        cz, cy = np.float32(sim.dt / (MU0 * sim.dz)), np.float32(sim.dt / (MU0 * sim.dy))
        fast_correct_H(sim.Hy, sim.Hz, self.ia, self.ib, self.ja, self.jb, self.ka, self.kb, cz, cy, self.ex1, self.koff)

    def correct_E(self, sim, n: int):
        e_inc = np.float32(self.amp * self.wf(n))
        self._h0, self._h1, self._hN, self._hNm1 = fast_step_aux_E(
            self.ex1, self.hy1, self.cE, self.mur, self._h0, self._h1, self._hN, self._hNm1, self.ks, e_inc
        )
        dx, dz = np.float32(sim.dx), np.float32(sim.dz)
        fast_correct_E(sim.Ex, sim.Ez, sim.C1, self.ia, self.ib, self.ja, self.jb, self.ka, self.kb, dz, dx, self.hy1, self.koff)

    def incident_ex(self, k_ref: int) -> float:
        return float(self.ex1[np.asarray(k_ref) - self.koff])