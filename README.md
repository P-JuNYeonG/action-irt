# Action-IRT: Identifying Important Actions in Problem-Solving Process Data via an Extended Item Response Model

> **Paper:** *Item Response Model for Online Educational Assessment Dataset with Log Sequence Dataset*
>
> Junyeong Park, Seyoung Park, Ick Hoon Jin, Minjeong Jeon
>
> Submitted to *Psychometrika*

---

## Overview

This repository contains the code and supplementary materials for an end-to-end framework that identifies behaviorally important actions in problem-solving process data from large-scale computer-based assessments.

The framework transforms raw log sequences into interpretable action-level effect estimates within a psychometric Item Response Theory (IRT) model. It consists of three stages:

```
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────────────┐
│  1. Action       │     │  2. Dimension        │     │  3. Action-IRT with      │
│     Embedding    │────▶│     Reduction         │────▶│     Spike-and-Slab       │
│  (Hybrid W2V)   │     │  (LSTM AutoEncoder)  │     │     Variable Selection   │
└─────────────────┘     └─────────────────────┘     └──────────────────────────┘
 Raw log sequences       Variable-length             Binary responses +
 → Dense action           embedding sequences         latent action values
   vectors                → D-dimensional             → Important action
                            latent values                identification
```

**Applied to:** 14 PSTRE items from the OECD PIAAC (U.S. sample, 1,996 respondents).

**Key result:** 126 out of 2,025 action–item combinations (6.2%) were identified as important, with a parametric-bootstrap simulation confirming action-level AUC of 0.94 and PIP-based sensitivity of 0.93.

---

## Repository Structure

```
action-irt/
├── README.md
├── LICENSE                          # MIT License
├── .gitignore
├── setup_r_env.R                    # R package dependency setup
│
├── manuscript/                      # LaTeX source for the paper
│   ├── main.tex
│   ├── reference.bib
│   ├── figures/
│   └── tables/
│
├── code/
│   ├── 01_preprocessing/            # Data cleaning (rule-based + LLM-assisted)
│   │   └── README.md
│   ├── 02_embedding/                # Hybrid Word2Vec for action embedding
│   │   └── README.md
│   ├── 03_dimension_reduction/      # LSTM AutoEncoder
│   │   └── README.md
│   ├── 04_mcmc/                     # Action-IRT model (C++/Rcpp + R)
│   │   ├── README.md
│   │   ├── MCMC.cpp
│   │   └── run_mcmc.R
│   └── 05_simulation/              # Parametric bootstrap simulation
│       └── README.md
│
├── analysis/                        # Post-estimation analysis and visualization
│   ├── convergence_diagnostics.R
│   ├── important_actions.R
│   └── figures/
│
├── supplementary/                   # Appendix materials
│   ├── llm_prompts/                 # LLM preprocessing prompts
│   └── full_item_tables/            # Complete action tables for all 14 items
│
└── docs/                            # GitHub Pages website
    └── index.md
```

---

## Data Availability

The analysis uses log process data from the **OECD Programme for the International Assessment of Adult Competencies (PIAAC)**, Problem Solving in Technology-Rich Environments (PSTRE) domain.

- **Access:** PIAAC public-use data files are available from the [OECD PIAAC Data Portal](https://www.oecd.org/skills/piaac/data/).
- **Restriction:** Raw log files are not redistributed in this repository due to OECD data use terms.
- **Reproducibility:** All analysis code is provided. Users who obtain the PIAAC log data can reproduce the full pipeline by following the instructions below.

---

## Reproduction Guide

### Prerequisites

| Software | Version | Purpose |
|----------|---------|---------|
| R        | ≥ 4.3   | MCMC estimation, diagnostics, visualization |
| Python   | ≥ 3.9   | Action embedding (Word2Vec), LSTM AutoEncoder |
| C++ compiler | C++17 compatible | Rcpp/RcppArmadillo for MCMC sampler |

### Step-by-step

```bash
# 1. Clone the repository
git clone https://github.com/P-JuNYeonG/action-irt.git
cd action-irt

# 2. Install R dependencies
Rscript setup_r_env.R

# 3. Install Python dependencies
pip install -r requirements.txt  # if provided

# 4. Run the pipeline in order
#    Step 1: Preprocessing (see code/01_preprocessing/README.md)
#    Step 2: Action embedding (see code/02_embedding/README.md)
#    Step 3: Dimension reduction (see code/03_dimension_reduction/README.md)
#    Step 4: MCMC estimation (see code/04_mcmc/README.md)
#    Step 5: Simulation study (see code/05_simulation/README.md)
```

### MCMC Settings (Empirical Analysis)

| Parameter | Value |
|-----------|-------|
| Total iterations | 50,000 |
| Burn-in | 10,000 |
| Thinning | 10 |
| Saved samples | 4,000 |
| Spike variance τ² | 0.001 |
| Slab variance ν² | 2.5 |
| Latent dimension D | 1 |

---

## Key Results Summary

| Metric | Value |
|--------|-------|
| Respondents | 1,996 |
| Items | 14 |
| Total action–item combinations | 2,025 |
| Important actions identified | 126 (6.2%) |
| Simulation AUC (mean ± SD) | 0.943 ± 0.011 |
| Simulation sensitivity (mean ± SD) | 0.935 ± 0.013 |

---

## Citation

If you use this code or framework, please cite:

```bibtex
@article{park2026actionirt,
  title={Item Response Model for Online Educational Assessment Dataset 
         with Log Sequence Dataset},
  author={Park, Junyeong and Park, Seyoung and Jin, Ick Hoon and Jeon, Minjeong},
  journal={Psychometrika},
  year={2026},
  note={Submitted}
}
```

---

## License

Code is released under the [MIT License](LICENSE). The manuscript text is © the authors.

---

## Contact

For questions about the code or analysis, please open an issue or contact: [corresponding author email]
