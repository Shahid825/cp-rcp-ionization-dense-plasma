# -*- coding: utf-8 -*-
"""
growth_rates.py
===============

Analytical growth rates of circularly polarized (CP) and right-circularly
polarized (RCP) electromagnetic modes in a dense, degenerate, unmagnetized
electron-ion plasma with an ionization source term, as derived in

    M. B. Rafiq, S. Idrees, A. A. Khan, A. Rasheed, B. Ramzan & M. Jamil,
    "Ionization-Driven Instabilities of Circularly Polarized Electromagnetic
     Modes in Dense Plasmas", Research in Astronomy and Astrophysics (2026).

Equation numbers below refer to that paper. Gaussian-cgs units throughout:
    k in cm^-1, n in cm^-3, alpha and all frequencies in s^-1,
    velocities in cm s^-1.

Functions
---------
fermi_velocity(n, m)          v_F = hbar (3 pi^2 n)^(1/3) / m
plasma_freq_sq(n, m)          omega_p^2 = 4 pi n e^2 / m
gamma1(k, a, ne, ni, th)      CP growth rate, Eq. (14)
gamma2_printed(k, ...)        RCP growth rate exactly as printed, Eq. (15)
gamma2(k, ...)                RCP growth rate, reduced form, Eq. (16)
gamma2_dev(k, ...)            gamma_2 - alpha/2, Eq. (16)
delta0(a, ne, ni)             small-k reduction below alpha/2, Eq. (17)
singular_points(a, ne, ni, th)  zeros of the denominator of Eq. (16)
analytic_k(a, ne, ni, th)     closed-form singular points k_e, k_i, Eq. (19)
gamma1_large_k(k, a, n, th)   large-k limit of the CP growth rate, Eq. (20)
plasma_scales(n)              quantities listed in Table 1
"""

import math
import numpy as np
from scipy.optimize import brentq

# --------------------------------------------------------------------------
# Physical constants (cgs). These are the values used for all numbers and
# figures in the paper; do not change them if you want to reproduce the
# published values digit for digit.
# --------------------------------------------------------------------------
E_CHARGE = 4.803e-10      # elementary charge [statC]
M_E = 9.109e-28           # electron mass [g]
M_I = 1.6726e-24          # ion (proton) mass [g]
HBAR = 1.0546e-27         # reduced Planck constant [erg s]
K_B = 1.3807e-16          # Boltzmann constant [erg K^-1]
C_LIGHT = 2.998e10        # speed of light [cm s^-1]
DEG = math.pi / 180.0     # degrees -> radians


# --------------------------------------------------------------------------
# Basic plasma quantities
# --------------------------------------------------------------------------
def fermi_velocity(n, m):
    """Fermi velocity v_F = hbar (3 pi^2 n)^(1/3) / m  [cm s^-1]."""
    return HBAR * (3.0 * math.pi**2 * n) ** (1.0 / 3.0) / m


def plasma_freq_sq(n, m):
    """Plasma frequency squared omega_p^2 = 4 pi n e^2 / m  [s^-2]."""
    return 4.0 * math.pi * n * E_CHARGE**2 / m


# --------------------------------------------------------------------------
# CP growth rate, Eq. (14)
# --------------------------------------------------------------------------
def gamma1(k, a, ne, ni, th):
    """CP growth rate gamma_1 [s^-1], Eq. (14).

    k  : wavenumber [cm^-1] (scalar or numpy array)
    a  : ionization frequency alpha [s^-1]
    ne, ni : equilibrium electron and ion densities [cm^-3]
    th : propagation angle to the z-axis [rad]
    """
    ve, vi = fermi_velocity(ne, M_E), fermi_velocity(ni, M_I)
    we2, wi2 = plasma_freq_sq(ne, M_E), plasma_freq_sq(ni, M_I)
    ky, kz = k * np.sin(th), k * np.cos(th)
    ae, ai = a**2 - ky**2 * ve**2, a**2 - ky**2 * vi**2
    g4e = ae**2 - ky**2 * kz**2 * ve**4          # Gamma_{alpha e}^4
    g4i = ai**2 - ky**2 * kz**2 * vi**4          # Gamma_{alpha i}^4
    num = a * wi2 * ai * g4e + a * we2 * ae * g4i
    den = 2.0 * we2 * ae * g4i - 2.0 * g4e * (g4i - wi2 * ai)
    return num / den


def gamma1_large_k(k, a, n, th):
    """Large-k limit of Eq. (14) for n_i0 = n_e0 = n and k_y v_Fi >> alpha, Eq. (20)."""
    ve = fermi_velocity(n, M_E)
    we2 = plasma_freq_sq(n, M_E)
    return 0.5 * a * (1.0 + (M_E / M_I) * k**2 * ve**2 / we2 * np.cos(2.0 * th))


# --------------------------------------------------------------------------
# RCP growth rate, Eqs. (15)-(16)
# --------------------------------------------------------------------------
def _rcp_terms(k, a, ne, ni, th):
    """Auxiliary quantities of Eqs. (15)-(16)."""
    ve, vi = fermi_velocity(ne, M_E), fermi_velocity(ni, M_I)
    wp2 = plasma_freq_sq(ne, M_E) + plasma_freq_sq(ni, M_I)
    X, Y = ve**2 + vi**2, ve**4 + vi**4
    ky, kz = k * np.sin(th), k * np.cos(th)
    qy = a**2 - ky**2 * X                        # Q_y
    qz = a**2 - kz**2 * X                        # Q_z
    g14 = qy * qz - ky**2 * kz**2 * Y            # Gamma_1^4
    g22 = 2.0 * a**2 - ky**2 * X                 # Gamma_2^2
    g38 = a**2 * ky**2 * kz**2 * Y * wp2         # Gamma_3^8
    dred = wp2 * (qz * g22 - ky**2 * kz**2 * Y) + 2.0 * a**2 * g14   # denominator of Eq. (16)
    return dict(wp2=wp2, X=X, Y=Y, ky=ky, kz=kz, qy=qy, qz=qz,
                g14=g14, g22=g22, g38=g38, dred=dred)


def gamma2_printed(k, a, ne, ni, th):
    """RCP growth rate exactly as printed in Eq. (15) (kept for verification)."""
    t = _rcp_terms(k, a, ne, ni, th)
    num = a * t["wp2"] * t["g14"] * t["g22"] + a * t["g38"]
    den = (4.0 * a**2 * t["qy"] * t["g14"] + 2.0 * t["wp2"] * t["g14"] * t["g22"]
           + 2.0 * t["g38"])
    return num / den


def gamma2(k, a, ne, ni, th):
    """RCP growth rate gamma_2 [s^-1], reduced form Eq. (16).

    The common factor Q_y = alpha^2 - k_y^2 (v_Fe^2 + v_Fi^2) of the numerator
    and denominator of Eq. (15) has been cancelled analytically, so the zero
    of Q_y (a removable singularity) causes no numerical problems.
    """
    t = _rcp_terms(k, a, ne, ni, th)
    s = t["qz"] * t["g22"] - t["ky"]**2 * t["kz"]**2 * t["Y"]
    return 0.5 * a * t["wp2"] * s / t["dred"]


def gamma2_dev(k, a, ne, ni, th):
    """Deviation gamma_2 - alpha/2 [s^-1], Eq. (16), without round-off loss."""
    t = _rcp_terms(k, a, ne, ni, th)
    return -a**3 * t["g14"] / t["dred"]


def delta0(a, ne, ni):
    """Small-k reduction of gamma_2 below alpha/2, Eq. (17) [s^-1]."""
    return a**3 / (2.0 * (plasma_freq_sq(ne, M_E) + plasma_freq_sq(ni, M_I) + a**2))


def singular_points(a, ne, ni, th, kmin=1.0, kmax=1e9, npts=300000):
    """Wavenumbers [cm^-1] at which the denominator of Eq. (16) vanishes.

    Sign changes are bracketed on a logarithmic grid and refined with
    Brent's method.
    """
    ks = np.geomspace(kmin, kmax, npts)
    d = _rcp_terms(ks, a, ne, ni, th)["dred"]
    idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    f = lambda kk: _rcp_terms(kk, a, ne, ni, th)["dred"]
    return [brentq(f, ks[i], ks[i + 1], xtol=1e-12, rtol=1e-15) for i in idx]


def analytic_k(a, ne, ni, th):
    """Closed-form singular points (k_e, k_i) [cm^-1], Eq. (19)."""
    s2, c2 = math.sin(th) ** 2, math.cos(th) ** 2
    ke = a / fermi_velocity(ne, M_E) * math.sqrt(2.0 / (1.0 + c2))
    ki = a / fermi_velocity(ni, M_I) * math.sqrt((1.0 + c2) / (2.0 * s2 * c2))
    return ke, ki


# --------------------------------------------------------------------------
# Table 1
# --------------------------------------------------------------------------
def plasma_scales(n):
    """Characteristic parameters of the electron gas listed in Table 1 (n_i0 = n_e0 = n)."""
    kf = (3.0 * math.pi**2 * n) ** (1.0 / 3.0)
    ve = fermi_velocity(n, M_E)
    wpe = math.sqrt(plasma_freq_sq(n, M_E))
    ef = HBAR**2 * kf**2 / (2.0 * M_E)
    a_ws = (3.0 / (4.0 * math.pi * n)) ** (1.0 / 3.0)
    return {
        "n_e0_cm-3": n,
        "omega_pe_s-1": wpe,
        "v_Fe_cm_s-1": ve,
        "v_Fi_cm_s-1": fermi_velocity(n, M_I),
        "T_Fe_K": ef / K_B,
        "p_F_over_me_c": HBAR * kf / (M_E * C_LIGHT),
        "coupling_e2_over_aEF": E_CHARGE**2 / (a_ws * ef),
        "a_interelectron_cm": a_ws,
        "lambda_F_cm": 2.0 * math.pi / kf,
        "lambda_TF_cm": ve / (math.sqrt(3.0) * wpe),
        "d_e_cm": C_LIGHT / wpe,
    }
