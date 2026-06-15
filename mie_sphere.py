"""Analytic Mie-series scattering for a homogeneous dielectric sphere.

Reference solution to validate the FDTD + NTFF RCS pipeline. Non-magnetic
sphere (mu_r = 1) of radius a, relative permittivity eps_r, in vacuum.

Conventions
-----------
x  = k0 * a                    size parameter (k0 = 2*pi/lambda in vacuum)
m  = sqrt(eps_r)               relative refractive index
S1, S2(theta)                  Bohren & Huffman scattering amplitudes
Bistatic RCS in a principal plane (scattering angle theta from forward +z):
    sigma_perp(theta) = (4*pi / k0^2) * |S1(theta)|^2   (H-plane, E perp to plane)
    sigma_par (theta) = (4*pi / k0^2) * |S2(theta)|^2   (E-plane, E in plane)
Backscatter (theta = 180 deg): both polarizations coincide.
"""
from __future__ import annotations

import numpy as np


def _mie_ab(x: float, m: complex, nmax: int):
    """Mie coefficients a_n, b_n (n = 1..nmax) via Riccati-Bessel recurrence."""
    mx = m * x
    n = np.arange(1, nmax + 1)

    # spherical Bessel of real argument x
    jx = _sph_jn(nmax, x)              # j_0..j_nmax
    yx = _sph_yn(nmax, x)
    psi = x * jx                       # Riccati-Bessel psi_n, n=0..nmax
    chi = -x * yx                      # chi_n
    xi = psi - 1j * chi                # xi_n = psi_n - i chi_n

    psi_n = psi[1:]; psi_nm1 = psi[:-1]
    xi_n = xi[1:];   xi_nm1 = xi[:-1]
    dpsi = psi_nm1 - n / x * psi_n     # psi_n'
    dxi = xi_nm1 - n / x * xi_n        # xi_n'

    # logarithmic derivative D_n(mx) by downward recurrence
    D = _logderiv(mx, nmax)            # D_1..D_nmax
    a = ((D / m + n / x) * psi_n - psi_nm1) / ((D / m + n / x) * xi_n - xi_nm1)
    b = ((D * m + n / x) * psi_n - psi_nm1) / ((D * m + n / x) * xi_n - xi_nm1)
    return a, b


def _mie_ab_pec(x: float, nmax: int):
    """Mie coefficients a_n, b_n for a perfectly conducting (PEC) sphere.

    PEC limit of the dielectric coefficients (Bohren & Huffman, e^{-iwt}):
        a_n (TM/electric) = psi_n'(x) / xi_n'(x)
        b_n (TE/magnetic) = psi_n(x)  / xi_n(x)
    """
    n = np.arange(1, nmax + 1)
    jx = _sph_jn(nmax, x)
    yx = _sph_yn(nmax, x)
    psi = x * jx
    chi = -x * yx
    xi = psi - 1j * chi
    psi_n = psi[1:]; psi_nm1 = psi[:-1]
    xi_n = xi[1:];   xi_nm1 = xi[:-1]
    dpsi = psi_nm1 - n / x * psi_n
    dxi = xi_nm1 - n / x * xi_n
    # PEC limit of the dielectric Mie coefficients (m -> inf):
    #   a_n (TM/electric)  -> psi_n'(x) / xi_n'(x)
    #   b_n (TE/magnetic)  -> psi_n(x)  / xi_n(x)
    a = dpsi / dxi
    b = psi_n / xi_n
    return a, b


def _sph_jn(nmax, x):
    from scipy.special import spherical_jn
    return spherical_jn(np.arange(nmax + 1), x)


def _sph_yn(nmax, x):
    from scipy.special import spherical_yn
    return spherical_yn(np.arange(nmax + 1), x)


def _logderiv(mx: complex, nmax: int):
    """D_n(mx) = psi_n'/psi_n, downward recurrence (Bohren & Huffman)."""
    nmx = int(max(nmax + 15, abs(mx) + 15))
    D = np.zeros(nmx + 1, dtype=complex)
    for n in range(nmx, 0, -1):
        D[n - 1] = n / mx - 1.0 / (D[n] + n / mx)
    return D[1:nmax + 1]


def _pi_tau(theta, nmax):
    """Angle functions pi_n(cos th), tau_n(cos th), n=1..nmax."""
    mu = np.cos(theta)
    pi = np.zeros(nmax + 1)
    tau = np.zeros(nmax + 1)
    pi[1] = 1.0
    tau[1] = mu
    for n in range(2, nmax + 1):
        pi[n] = ((2 * n - 1) / (n - 1)) * mu * pi[n - 1] - (n / (n - 1)) * pi[n - 2]
        tau[n] = n * mu * pi[n] - (n + 1) * pi[n - 1]
    return pi[1:], tau[1:]


def mie_amplitudes(theta, eps_r, a_radius: float, f0: float, pec: bool = False):
    """Return (S1, S2) arrays over the scattering-angle array theta (radians).

    pec=True -> perfectly conducting sphere (eps_r ignored)."""
    c0 = 299_792_458.0
    k0 = 2 * np.pi * f0 / c0
    x = k0 * a_radius
    nmax = int(np.ceil(x + 4 * x ** (1 / 3) + 2)) + 2
    if pec:
        an, bn = _mie_ab_pec(x, nmax)
    else:
        an, bn = _mie_ab(x, np.sqrt(eps_r + 0j), nmax)
    n = np.arange(1, nmax + 1)
    fac = (2 * n + 1) / (n * (n + 1))
    theta = np.atleast_1d(theta)
    S1 = np.zeros(theta.shape, complex)
    S2 = np.zeros(theta.shape, complex)
    for i, th in enumerate(theta):
        pin, taun = _pi_tau(th, nmax)
        S1[i] = np.sum(fac * (an * pin + bn * taun))
        S2[i] = np.sum(fac * (an * taun + bn * pin))
    return S1, S2


def mie_rcs(theta, eps_r, a_radius: float, f0: float, pol: str = "E", pec: bool = False):
    """Bistatic RCS (m^2) vs scattering angle theta.

    pol='E' -> E-plane (parallel, uses S2);  pol='H' -> H-plane (uses S1).
    pec=True -> perfectly conducting sphere.
    """
    c0 = 299_792_458.0
    k0 = 2 * np.pi * f0 / c0
    S1, S2 = mie_amplitudes(theta, eps_r, a_radius, f0, pec=pec)
    S = S2 if pol.upper() == "E" else S1
    return (4 * np.pi / k0 ** 2) * np.abs(S) ** 2


def mie_backscatter(eps_r, a_radius: float, f0: float, pec: bool = False) -> float:
    """Monostatic (backscatter) RCS (m^2)."""
    return float(mie_rcs(np.array([np.pi]), eps_r, a_radius, f0, pol="E", pec=pec)[0])
