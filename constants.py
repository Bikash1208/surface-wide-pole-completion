"""Shared manuscript-wide constants for the AWPL reproduction code.

Centralizes invariants common to several scripts so they cannot drift apart.
Scientific parameters that legitimately differ between experiments (fit-window
length, pencil order, per-script cut step, the wider Mie pole-search band) are
kept local to those scripts and documented there.
"""
from __future__ import annotations

C0_M_PER_S = 299_792_458.0        # speed of light in vacuum [m/s]

# Manuscript backscatter-RMS evaluation band and zero-padded rFFT length.
BAND_HZ = (1.20e9, 2.60e9)
N_FFT = 65536

# Reference eps_r = 25 sphere FDTD run.
N_TOTAL = 16000                   # full run length [time steps]
N_CUT_75PCT = 4000                # 75 %-removed early-stop cut [time steps]

# Huygens-surface angular sampling (theta x phi nodes) and recorded components.
N_THETA = 24
N_PHI = 48
COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
