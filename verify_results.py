# -*- coding: utf-8 -*-
"""
verify_results.py
=================

Independent numerical checks of the analytical results of Rafiq et al.
(2026, RAA). Prints a report and saves it to results/verification_report.txt.

Checks
------
1. Eq. (16) (reduced RCP growth rate) equals Eq. (15) as printed.
2. Small-k and large-k limits of gamma_2: -Delta_0 and -2 Delta_0, and the
   intermediate plateau -2 Delta_0/(1 + cos^2 theta)  (Eq. 17 and text).
3. Closed-form singular points, Eq. (19), versus zeros of the denominator
   of Eq. (16) over the full parameter range.
4. Large-k limit of the CP growth rate, Eq. (20), versus Eq. (14).
5. Numbers quoted in the text for Figs. 1, 3, 4-6 and Tables 1 and 3.

Usage
-----
    python verify_results.py
"""

import io
import math
import os
import sys

import numpy as np

from growth_rates import (DEG, M_E, M_I, gamma1, gamma1_large_k, gamma2, gamma2_printed,
                          gamma2_dev, delta0, singular_points, analytic_k,
                          plasma_scales, fermi_velocity)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
os.makedirs(OUT_DIR, exist_ok=True)

buf = io.StringIO()


def say(*args):
    line = " ".join(str(a) for a in args)
    print(line)
    buf.write(line + "\n")


A0, N0, TH0 = 3e10, 1e25, 20 * DEG

say("=" * 72)
say("Verification report: Rafiq et al. (2026), RAA")
say("=" * 72)

# 1 ----------------------------------------------------------------------
rng = np.random.default_rng(1)
worst = 0.0
for _ in range(20000):
    a = 10 ** rng.uniform(10, 12.5)
    n = 10 ** rng.uniform(24, 26)
    th = rng.uniform(5, 30) * DEG
    k = 10 ** rng.uniform(0, 7.5)
    g15 = gamma2_printed(k, a, n, n, th)
    g16 = gamma2(k, a, n, n, th)
    worst = max(worst, abs(g16 - g15) / abs(g15))
say("\n[1] Eq. (16) vs Eq. (15), 20000 random points:")
say("    max |gamma2(16) - gamma2(15)| / gamma2(15) = %.2e" % worst)

# 2 ----------------------------------------------------------------------
say("\n[2] Plateaus of (gamma_2 - alpha/2)/Delta_0 (alpha = 3e10, n = 1e25):")
for t in (20, 15, 10):
    d0 = delta0(A0, N0, N0)
    small = gamma2_dev(1.0, A0, N0, N0, t * DEG) / d0
    mid = gamma2_dev(1e3, A0, N0, N0, t * DEG) / d0
    large = gamma2_dev(1e9, A0, N0, N0, t * DEG) / d0
    say("    theta=%2d deg: k=1: %.4f (expect -1)   k=1e3: %.4f (expect %.4f)   k=1e9: %.4f (expect -2)"
        % (t, small, mid, -2 / (1 + math.cos(t * DEG) ** 2), large))

# 3 ----------------------------------------------------------------------
worst = 0.0
count = 0
for a in np.geomspace(1e10, 3e12, 7):
    for n in np.geomspace(1e24, 1e26, 5):
        for t in (5, 10, 15, 20, 25, 30):
            num = singular_points(a, n, n, t * DEG)
            an = analytic_k(a, n, n, t * DEG)
            assert len(num) == 2, "expected exactly two singular points"
            worst = max(worst, max(abs(x - y) / y for x, y in zip(num, an)))
            count += 1
say("\n[3] Singular points: Eq. (19) vs numerical zeros of the denominator of Eq. (16)")
say("    %d parameter sets (alpha 1e10-3e12, n 1e24-1e26, theta 5-30 deg)" % count)
say("    exactly two singular points found in every case; max relative difference = %.2e" % worst)

# 4 ----------------------------------------------------------------------
say("\n[4] CP large-k limit, Eq. (20) vs Eq. (14) at k = 3e8 cm^-1 (alpha = 3e10):")
say("    n_e0      theta   gamma1/(alpha/2)-1: Eq.(14)   Eq.(20)    k*lambda_TF")
for n in (1e24, 1e25, 1e26):
    ltf = plasma_scales(n)["lambda_TF_cm"]
    for t in (20, 15, 10):
        ex = gamma1(3e8, A0, n, n, t * DEG) / (A0 / 2) - 1
        ap = gamma1_large_k(3e8, A0, n, t * DEG) / (A0 / 2) - 1
        say("    %.0e   %2d      %.4e          %.4e   %.2f" % (n, t, ex, ap, 3e8 * ltf))

# 5 ----------------------------------------------------------------------
say("\n[5] Numbers quoted in the text")
say("    Fig. 1, gamma_1(k=3e8) for theta = 20, 15, 10 deg: "
    + ", ".join("%.5e" % gamma1(3e8, A0, N0, N0, t * DEG) for t in (20, 15, 10)))
say("    Fig. 3, gamma_1(3e8)/gamma_1(0) - 1 for n = 1e24, 1e25, 1e26: "
    + ", ".join("%.2e" % (gamma1(3e8, A0, n, n, TH0) / gamma1(0.0, A0, n, n, TH0) - 1)
                for n in (1e24, 1e25, 1e26)))
say("    CP singular points (alpha=3e10, n=1e25, theta=20):")
ks = np.geomspace(1, 3e8, 300000)
g = gamma1(ks, A0, N0, N0, TH0)
# locate sign changes of the denominator through jumps of gamma_1
from growth_rates import plasma_freq_sq
ve, vi = fermi_velocity(N0, M_E), fermi_velocity(N0, M_I)
we2, wi2 = plasma_freq_sq(N0, M_E), plasma_freq_sq(N0, M_I)
ky, kz = ks * math.sin(TH0), ks * math.cos(TH0)
ae, ai = A0**2 - ky**2 * ve**2, A0**2 - ky**2 * vi**2
g4e, g4i = ae**2 - ky**2 * kz**2 * ve**4, ai**2 - ky**2 * kz**2 * vi**4
den = 2 * we2 * ae * g4i - 2 * g4e * (g4i - wi2 * ai)
idx = np.where(np.sign(den[:-1]) != np.sign(den[1:]))[0]
say("        k = " + ", ".join("%.3e" % ks[i] for i in idx)
    + "  (alpha/(v_Fe sin th) = %.3e, alpha/(v_Fi sin th) = %.3e)"
    % (A0 / (ve * math.sin(TH0)), A0 / (vi * math.sin(TH0))))

say("\n    Table 1:")
for n in (1e24, 1e25, 1e26):
    s = plasma_scales(n)
    say("      n=%.0e  w_pe=%.3e  v_Fe=%.3e  v_Fi=%.3e  T_Fe=%.2e K  pF/mec=%.3f  "
        "e2/(aEF)=%.2f  a=%.2e  lam_F=%.2e  lam_TF=%.2e  d_e=%.2e"
        % (n, s["omega_pe_s-1"], s["v_Fe_cm_s-1"], s["v_Fi_cm_s-1"], s["T_Fe_K"],
           s["p_F_over_me_c"], s["coupling_e2_over_aEF"], s["a_interelectron_cm"],
           s["lambda_F_cm"], s["lambda_TF_cm"], s["d_e_cm"]))

say("\n    Table 3 (k_e, k_i numerical | Eq. 19):")
for a, n, t in [(3e10, 1e25, 20), (3e10, 1e25, 15), (3e10, 1e25, 10), (3e11, 1e25, 20),
                (3e12, 1e25, 20), (3e10, 1e24, 20), (3e10, 1e26, 20)]:
    num = singular_points(a, n, n, t * DEG)
    an = analytic_k(a, n, n, t * DEG)
    say("      alpha=%.0e n=%.0e theta=%2d  Delta0=%.3e  k_e=%.4g k_i=%.4g | %.4g %.4g"
        % (a, n, t, delta0(a, n, n), num[0], num[1], an[0], an[1]))

with open(os.path.join(OUT_DIR, "verification_report.txt"), "w") as f:
    f.write(buf.getvalue())
say("\nreport written to results/verification_report.txt")
