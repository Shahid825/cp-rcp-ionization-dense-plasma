# Ionization-driven instabilities of circularly polarized modes in dense plasmas

Code and data for the paper

> M. Bilal Rafiq, Shahid Idrees, Arroj A. Khan, A. Rasheed, B. Ramzan and M. Jamil,
> **"Ionization-Driven Instabilities of Circularly Polarized Electromagnetic Modes in Dense Plasmas"**,
> *Research in Astronomy and Astrophysics* (2026, submitted).

<!-- After Zenodo has created the DOI, replace XXXXXXX below (two places) -->
Archived version: [https://doi.org/10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

The repository contains the Python scripts that evaluate the analytical growth
rates of the paper, Equations (14)–(20), reproduce Figures 1–7, and write the
numerical data underlying every figure and table.

---

## Contents

| Path | Description |
|------|-------------|
| `growth_rates.py` | All formulas of the paper: CP growth rate (Eq. 14), RCP growth rate as printed (Eq. 15) and in reduced form (Eq. 16), small-k reduction Δ₀ (Eq. 17), singular points (Eqs. 18–19), large-k CP limit (Eq. 20) and the plasma parameters of Table 1 |
| `make_figures.py` | Regenerates Figures 1–7 in `figures/` and writes the data files in `data/` |
| `verify_results.py` | Independent numerical checks of the analytical results; writes `results/verification_report.txt` |
| `figures/` | Figures 1–7 as PDF (used in the paper) and PNG |
| `data/` | CSV files with the numerical values plotted in each figure and listed in Tables 1 and 3 |
| `results/verification_report.txt` | Output of `verify_results.py` |
| `requirements.txt` | Python packages needed |
| `CITATION.cff` | Citation metadata (read by GitHub and Zenodo) |
| `LICENSE` | MIT licence |

### Data files

| File | Content | Paper |
|------|---------|-------|
| `data/fig1_cp_angle.csv` | γ₁(k) for θ = 20°, 15°, 10° | Fig. 1 |
| `data/fig2_cp_alpha.csv` | γ₁(k) for α = 3.000, 3.001, 3.002 × 10¹⁰ s⁻¹ | Fig. 2 |
| `data/fig3_cp_density.csv` | γ₁(k)/γ₁(k→0) for n_e0 = 10²⁴, 10²⁵, 10²⁶ cm⁻³ | Fig. 3 |
| `data/fig4_rcp_angle.csv` | (γ₂ − α/2)/Δ₀ for θ = 20°, 15°, 10° | Fig. 4 |
| `data/fig5_rcp_alpha.csv` | (γ₂ − α/2)/Δ₀ for α = 3 × 10¹⁰, 3 × 10¹¹, 3 × 10¹² s⁻¹ | Fig. 5 |
| `data/fig6_rcp_density.csv` | (γ₂ − α/2)/Δ₀ for n_e0 = 10²⁴, 10²⁵, 10²⁶ cm⁻³ | Fig. 6 |
| `data/fig7a_kres_vs_alpha.csv`, `fig7b_kres_vs_theta.csv`, `fig7c_kres_vs_density.csv` | Singular points k_e and k_i: closed form (Eq. 19) and numerical roots | Fig. 7 |
| `data/table1_plasma_scales.csv` | Characteristic parameters and length scales | Table 1 |
| `data/table3_rcp_characteristics.csv` | α/2, Δ₀, k_e, k_i for the parameter sets of Figs. 4–6 | Table 3 |

Each CSV file starts with comment lines beginning with `#` that state the
parameters, followed by a header row. Units are given in the column names
(Gaussian-cgs: k in cm⁻¹, frequencies and growth rates in s⁻¹, densities in
cm⁻³). In the files for Figs. 4–6 the value `nan` marks the grid points on
either side of a singular point, where the curve is interrupted.

Reading a file in Python:

```python
import pandas as pd
df = pd.read_csv("data/fig4_rcp_angle.csv", comment="#")
```

---

## Requirements and usage

Python 3.8 or newer with NumPy, SciPy and Matplotlib:

```bash
pip install -r requirements.txt
python make_figures.py      # figures/ and data/  (about 15 s)
python verify_results.py    # results/verification_report.txt
```

The scripts were run with Python 3.12, NumPy 2.4, SciPy 1.17 and Matplotlib 3.10.

---

## Parameters and conventions

* Gaussian-cgs units; constants in `growth_rates.py`
  (e = 4.803 × 10⁻¹⁰ statC, m_e = 9.109 × 10⁻²⁸ g, m_i = 1.6726 × 10⁻²⁴ g,
  ħ = 1.0546 × 10⁻²⁷ erg s).
* Fermi velocity v_Fj = ħ(3π²n_j0)^(1/3)/m_j; quasi-neutrality n_i0 = n_e0 (Z = 1).
* Reference parameter set: α = 3 × 10¹⁰ s⁻¹, n_e0 = 10²⁵ cm⁻³, θ = 20°;
  one parameter at a time is varied.
* CP figures (1–3): linear k grid, 0–3 × 10⁸ cm⁻¹ (601 points).
* RCP figures (4–6): logarithmic k grid of 2 × 10⁵ points. Because γ₂ differs
  from α/2 only by an amount of order Δ₀ = α³/[2(ω_pe² + ω_pi² + α²)], the
  figures show the normalized deviation (γ₂ − α/2)/Δ₀, computed from the
  reduced form, Eq. (16), which avoids round-off errors.
* The positions of the singular points are robust and agree with Eq. (19) to
  better than 10⁻⁶. The heights of the curves near the singular points depend
  on the numerical grid and have no physical meaning (see Section 3 of the paper).
* Scope: the growth rates are derived in the ionization-dominated ordering
  α² ≫ ω² discussed in Section 2 of the paper.

---

## Citation

If you use this code or data, please cite the paper above and the archived
software version (DOI above). Citation metadata are provided in `CITATION.cff`
("Cite this repository" button on GitHub).

## Licence

MIT licence; see `LICENSE`.

## Contact

Shahid Idrees, National Astronomical Observatories, Chinese Academy of Sciences,
Beijing, China — shahid@bao.ac.cn
