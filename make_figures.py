# -*- coding: utf-8 -*-
"""
make_figures.py
===============

Regenerates Figures 1-7 of Rafiq et al. (2026, RAA) and writes the numerical
data underlying every figure and table to CSV files.

Usage
-----
    python make_figures.py

Output
------
    figures/fig1_cp_angle.pdf|png      ... figures/fig7_rcp_kres.pdf|png
    data/fig1_cp_angle.csv             ... data/fig7c_kres_vs_density.csv
    data/table1_plasma_scales.csv
    data/table3_rcp_characteristics.csv

All formulas are implemented in growth_rates.py. Run time: about 1-2 minutes.
"""

import csv
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from growth_rates import (DEG, M_E, M_I, gamma1, gamma2_dev, delta0,
                          singular_points, analytic_k, plasma_scales,
                          _rcp_terms)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
DATA_DIR = os.path.join(HERE, "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 11,
                     "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True})

# Reference parameter set (Section 3 of the paper)
A0 = 3e10          # ionization frequency alpha [s^-1]
N0 = 1e25          # n_e0 = n_i0 [cm^-3]
TH0 = 20 * DEG     # propagation angle theta


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"))
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=300)
    plt.close(fig)


def write_csv(name, header, columns, comment=None):
    """Write equal-length columns to data/<name>.csv (NaN written as 'nan')."""
    path = os.path.join(DATA_DIR, name + ".csv")
    with open(path, "w", newline="") as f:
        if comment:
            for line in comment.strip().splitlines():
                f.write("# " + line.strip() + "\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for row in zip(*columns):
            w.writerow(["%.8e" % v if isinstance(v, (float, np.floating)) else v for v in row])


# ==========================================================================
# CP figures (Figs. 1-3): linear k axis up to 3e8 cm^-1
# ==========================================================================
K_CP = np.concatenate([[0.0], np.linspace(1e6, 3e8, 600)])


def cp_axes(ax, ylabel):
    ax.set_xlabel(r"Wave vector $k$ ($10^{8}$ cm$^{-1}$)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 3.0)
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.legend(loc="upper left")


def figure1():
    sets = [(20, "blue", "-"), (15, "black", "-."), (10, "red", ":")]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    cols = []
    for t, col, ls in sets:
        g = gamma1(K_CP, A0, N0, N0, t * DEG)
        cols.append(g)
        ax.plot(K_CP / 1e8, g / 1e10, color=col, ls=ls, lw=1.8, label=r"$\theta=%d^\circ$" % t)
    cp_axes(ax, r"Growth rate $\gamma_1$ ($10^{10}$ s$^{-1}$)")
    save_fig(fig, "fig1_cp_angle")
    write_csv("fig1_cp_angle", ["k_cm-1", "gamma1_theta20_s-1", "gamma1_theta15_s-1", "gamma1_theta10_s-1"],
              [K_CP] + cols,
              "Fig. 1: CP growth rate gamma_1, Eq. (14); alpha = 3e10 s^-1, n_e0 = n_i0 = 1e25 cm^-3")


def figure2():
    sets = [(3.000e10, "blue", "-"), (3.001e10, "black", ":"), (3.002e10, "orange", "-.")]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    cols = []
    for a, col, ls in sets:
        g = gamma1(K_CP, a, N0, N0, TH0)
        cols.append(g)
        ax.plot(K_CP / 1e8, g / 1e10, color=col, ls=ls, lw=1.8,
                label=r"$\alpha=%.3f\times10^{10}$ s$^{-1}$" % (a / 1e10))
    cp_axes(ax, r"Growth rate $\gamma_1$ ($10^{10}$ s$^{-1}$)")
    save_fig(fig, "fig2_cp_alpha")
    write_csv("fig2_cp_alpha", ["k_cm-1", "gamma1_alpha3.000e10_s-1", "gamma1_alpha3.001e10_s-1",
                                "gamma1_alpha3.002e10_s-1"],
              [K_CP] + cols,
              "Fig. 2: CP growth rate gamma_1, Eq. (14); n_e0 = n_i0 = 1e25 cm^-3, theta = 20 deg")


def figure3():
    sets = [(1e24, "blue", "-"), (1e25, "black", ":"), (1e26, "orange", "-.")]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    cols = []
    for n, col, ls in sets:
        g = gamma1(K_CP, A0, n, n, TH0)
        cols.append(g / g[0])
        ax.plot(K_CP / 1e8, g / g[0], color=col, ls=ls, lw=1.8,
                label=r"$n_{e0}=1.0\times10^{%d}$ cm$^{-3}$" % round(math.log10(n)))
    cp_axes(ax, r"$\gamma_1(k)\,/\,\gamma_1(k\rightarrow 0)$")
    save_fig(fig, "fig3_cp_density")
    write_csv("fig3_cp_density", ["k_cm-1", "gamma1_norm_ne1e24", "gamma1_norm_ne1e25", "gamma1_norm_ne1e26"],
              [K_CP] + cols,
              "Fig. 3: gamma_1(k)/gamma_1(k->0), Eq. (14); n_i0 = n_e0, alpha = 3e10 s^-1, theta = 20 deg")


# ==========================================================================
# RCP figures (Figs. 4-6): normalized deviation on a logarithmic k grid
# ==========================================================================
def rcp_curve(k, a, ne, ni, th):
    """(gamma_2 - alpha/2)/Delta_0 from Eq. (16), set to NaN at the singular points."""
    y = gamma2_dev(k, a, ne, ni, th) / delta0(a, ne, ni)
    d = _rcp_terms(k, a, ne, ni, th)["dred"]
    flip = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    y = np.array(y, dtype=float)
    y[flip] = np.nan
    y[flip + 1] = np.nan
    return y


def rcp_figure(name, sets, kmax, colnames, comment):
    k = np.geomspace(1.0, kmax, 200000)
    fig, ax = plt.subplots(figsize=(8, 5.6))
    cols = []
    for (a, ne, ni, th, col, ls, lab) in sets:
        y = rcp_curve(k, a, ne, ni, th)
        cols.append(y)
        ax.plot(k, y, color=col, ls=ls, lw=1.6, label=lab)
        for kr in analytic_k(a, ne, ni, th):
            if 1.0 < kr < kmax:
                ax.axvline(kr, color=col, ls=":", lw=0.9, alpha=0.8)
    ax.axhline(-1, color="0.5", lw=0.6)
    ax.axhline(-2, color="0.5", lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(1.0, kmax)
    ax.set_ylim(-4.5, 2.5)
    ax.set_xlabel(r"Wave vector $k$ (cm$^{-1}$)")
    ax.set_ylabel(r"$(\gamma_2-\alpha/2)\,/\,\Delta_0$")
    ax.legend(loc="lower left", frameon=True)
    save_fig(fig, name)
    write_csv(name, ["k_cm-1"] + colnames, [k] + cols, comment)


def figure4():
    sets = [(A0, N0, N0, 20 * DEG, "blue", "-", r"$\theta=20^\circ$"),
            (A0, N0, N0, 15 * DEG, "red", "--", r"$\theta=15^\circ$"),
            (A0, N0, N0, 10 * DEG, "green", "-.", r"$\theta=10^\circ$")]
    rcp_figure("fig4_rcp_angle", sets, 1e7,
               ["dev_norm_theta20", "dev_norm_theta15", "dev_norm_theta10"],
               "Fig. 4: (gamma_2 - alpha/2)/Delta_0 from Eq. (16); alpha = 3e10 s^-1, n_e0 = n_i0 = 1e25 cm^-3\n"
               "Delta_0 = 4.24e-4 s^-1 for all three curves; nan marks the singular points")


def figure5():
    sets = [(3e10, N0, N0, TH0, "blue", "-", r"$\alpha_1=3\times10^{10}$ s$^{-1}$"),
            (3e11, N0, N0, TH0, "red", "--", r"$\alpha_2=3\times10^{11}$ s$^{-1}$"),
            (3e12, N0, N0, TH0, "green", "-.", r"$\alpha_3=3\times10^{12}$ s$^{-1}$")]
    rcp_figure("fig5_rcp_alpha", sets, 5e7,
               ["dev_norm_alpha3e10", "dev_norm_alpha3e11", "dev_norm_alpha3e12"],
               "Fig. 5: (gamma_2 - alpha/2)/Delta_0 from Eq. (16); n_e0 = n_i0 = 1e25 cm^-3, theta = 20 deg\n"
               "Delta_0 = 4.24e-4, 0.424 and 424 s^-1; nan marks the singular points")


def figure6():
    sets = [(A0, 1e24, 1e24, TH0, "blue", "-", r"$n_{e0}=10^{24}$ cm$^{-3}$"),
            (A0, 1e25, 1e25, TH0, "red", "--", r"$n_{e0}=10^{25}$ cm$^{-3}$"),
            (A0, 1e26, 1e26, TH0, "green", "-.", r"$n_{e0}=10^{26}$ cm$^{-3}$")]
    rcp_figure("fig6_rcp_density", sets, 1e7,
               ["dev_norm_ne1e24", "dev_norm_ne1e25", "dev_norm_ne1e26"],
               "Fig. 6: (gamma_2 - alpha/2)/Delta_0 from Eq. (16); n_i0 = n_e0, alpha = 3e10 s^-1, theta = 20 deg\n"
               "Delta_0 = 4.24e-3, 4.24e-4 and 4.24e-5 s^-1; nan marks the singular points")


# ==========================================================================
# Fig. 7: positions of the singular points
# ==========================================================================
def figure7():
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6))
    sty = [("blue", "-", r"$k_{e}$ (electron)"), ("green", "-.", r"$k_{i}$ (ion)")]

    panels = [
        ("fig7a_kres_vs_alpha", "alpha_s-1", np.geomspace(1e10, 3e12, 60), np.geomspace(1e10, 3e12, 6),
         lambda x: (x, N0, N0, TH0), r"$\alpha$ (s$^{-1}$)",
         r"(a) $n_{e0}=10^{25}$ cm$^{-3}$, $\theta=20^\circ$", True),
        ("fig7b_kres_vs_theta", "theta_deg", np.linspace(5, 30, 60), np.array([5, 10, 15, 20, 25, 30.0]),
         lambda x: (A0, N0, N0, x * DEG), r"$\theta$ (deg)",
         r"(b) $\alpha=3\times10^{10}$ s$^{-1}$, $n_{e0}=10^{25}$ cm$^{-3}$", False),
        ("fig7c_kres_vs_density", "ne0_cm-3", np.geomspace(1e24, 1e26, 60), np.geomspace(1e24, 1e26, 5),
         lambda x: (A0, x, x, TH0), r"$n_{e0}=n_{i0}$ (cm$^{-3}$)",
         r"(c) $\alpha=3\times10^{10}$ s$^{-1}$, $\theta=20^\circ$", True),
    ]
    for ax, (name, xname, xline, xpts, args, xlabel, title, logx) in zip(axs, panels):
        line = np.array([analytic_k(*args(x)) for x in xline])
        pts = np.array([singular_points(*args(x)) for x in xpts])
        for j in range(2):
            ax.plot(xline, line[:, j], color=sty[j][0], ls=sty[j][1], label=sty[j][2])
            ax.plot(xpts, pts[:, j], "o", mfc="none", color=sty[j][0])
        if logx:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=11)
        # data: analytic line + numerical roots at the same x values
        num_line = np.array([singular_points(*args(x)) for x in xline])
        write_csv(name, [xname, "k_e_analytic_cm-1", "k_i_analytic_cm-1",
                         "k_e_numerical_cm-1", "k_i_numerical_cm-1"],
                  [xline, line[:, 0], line[:, 1], num_line[:, 0], num_line[:, 1]],
                  "Fig. 7: singular points of Eq. (16); analytic = Eq. (19), numerical = zeros of the "
                  "denominator of Eq. (16)")
    axs[0].set_ylabel(r"$k_{\rm res}$ (cm$^{-1}$)")
    axs[0].legend(loc="upper left")
    save_fig(fig, "fig7_rcp_kres")


# ==========================================================================
# Tables 1 and 3
# ==========================================================================
def table1():
    rows = [plasma_scales(n) for n in (1e24, 1e25, 1e26)]
    keys = list(rows[0].keys())
    write_csv("table1_plasma_scales", keys, [[r[k] for r in rows] for k in keys],
              "Table 1: characteristic parameters and length scales (n_i0 = n_e0)")


def table3():
    sets = [(3e10, 1e25, 20), (3e10, 1e25, 15), (3e10, 1e25, 10), (3e11, 1e25, 20),
            (3e12, 1e25, 20), (3e10, 1e24, 20), (3e10, 1e26, 20)]
    cols = {k: [] for k in ("alpha_s-1", "ne0_cm-3", "theta_deg", "alpha_over_2_s-1", "Delta0_s-1",
                            "k_e_cm-1", "k_i_cm-1", "k_e_Eq19_cm-1", "k_i_Eq19_cm-1")}
    for a, n, t in sets:
        num = singular_points(a, n, n, t * DEG)
        an = analytic_k(a, n, n, t * DEG)
        for key, val in zip(cols, (a, n, float(t), a / 2, delta0(a, n, n), num[0], num[1], an[0], an[1])):
            cols[key].append(val)
    write_csv("table3_rcp_characteristics", list(cols), list(cols.values()),
              "Table 3: RCP characteristics; k_e, k_i from the zeros of the denominator of Eq. (16), "
              "compared with Eq. (19)")


if __name__ == "__main__":
    for step in (figure1, figure2, figure3, figure4, figure5, figure6, figure7, table1, table3):
        print("running", step.__name__)
        step()
    print("done: figures in ./figures, data in ./data")
