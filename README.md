# Action-IRT: A Representation-Learning Item Response Model for Identifying Behaviorally Important Actions in PIAAC Process Data

> Junyeong Park, Daeun Hwangbo, Seyoung Park, Ick Hoon Jin, Minjeong Jeon

---

## Overview

This repository contains code and supplementary materials for a three-stage framework that identifies behaviorally important actions in problem-solving process data. Raw log sequences are embedded via a hybrid Word2Vec model, compressed to low-dimensional latent values via an LSTM AutoEncoder, and entered into an extended Rasch model with a spike-and-slab prior for automatic variable selection.

Applied to 14 PSTRE items from the U.S. PIAAC sample (1,996 respondents), the framework identified 126 of 2,025 action–item combinations (6.2%) as important. A parametric-bootstrap simulation yielded action-level AUC of 0.943 and PIP-threshold sensitivity of 0.935.

Project documentation is available on the [Action-IRT companion site](https://p-junyeong.github.io/action-irt/).

---

## Repository Structure

```
action-irt/
├── Data_Preprocess/
│   ├── 01_preprocessing_notebook.ipynb  # Original item-wise preprocessing workflow
│   ├── 01_preprocess_rules.py           # Reusable preprocessing utilities
│   ├── 02_llm_clean_descriptions.py     # Optional LLM-assisted description cleaning
│   └── mapping_tables/                   # Reviewed mappings for 14 PSTRE items
├── Embedding/
│   ├── HW2V/                             # Hybrid token-unit Word2Vec
│   ├── build_embedding_matrices.py       # Time-augmented embedding matrices
│   └── LSTM_AE/                          # LSTM AutoEncoder and long-format export
├── MCMC/
│   ├── MCMC.cpp                          # RcppArmadillo Action-IRT sampler
│   ├── run_mcmc.R                        # Empirical model wrapper
│   ├── GR_PIP_stability.R                # Convergence and PIP diagnostics
│   └── Simulation.R                      # Parametric-bootstrap evaluation
├── Analysis/                              # Loss curves, trace plots, and diagnostics
├── docs/                                  # GitHub Pages companion site
├── setup_r_env.R                          # R package setup
├── LICENSE
└── README.md
```

---

## Requirements

| Software | Version |
|---|---|
| R | ≥ 4.3 |
| Python | ≥ 3.9 |
| C++ compiler | C++17 compatible |

```bash
Rscript setup_r_env.R
```

Core Python packages: `numpy`, `pandas`, `torch`, and `matplotlib`. The `openai` package is required only when regenerating the optional OpenRouter-assisted preprocessing mappings.

---

## Configuration

Raw data, intermediate files, and generated outputs are not included in the repository. Input and output directories are intentionally user-configured: set the path arguments and placeholders in the relevant scripts for your local environment.

Reviewed LLM mapping tables for the 14 PSTRE items are included under `Data_Preprocess/mapping_tables/`. Regenerating them through OpenRouter is optional.

---

## Workflow

1. `Data_Preprocess/` — clean raw logs and construct action sequences.
2. `Embedding/HW2V/` — train hybrid action embeddings.
3. `Embedding/build_embedding_matrices.py` — combine action embeddings with time features.
4. `Embedding/LSTM_AE/` — reduce the time-augmented sequences and export latent values.
5. `MCMC/` — estimate the Action-IRT model and assess convergence and PIP stability.
6. `MCMC/Simulation.R` — run the parametric-bootstrap recovery evaluation.

---

## Data Availability

PIAAC public-use files are available from the [OECD PIAAC Data Portal](https://www.oecd.org/en/data/datasets/piaac.html). Raw log files are not redistributed here due to OECD data-use terms.

---

## Citation

```bibtex
@article{park2026actionirt,
  title={A Representation-Learning Item Response Model for Identifying
         Behaviorally Important Actions in {PIAAC} Process Data},
  author={Park, Junyeong and Hwangbo, Daeun and Park, Seyoung
          and Jin, Ick Hoon and Jeon, Minjeong},
  journal={Psychometrika},
  year={2026},
  note={Submitted}
}
```

---

## License

Code is released under the [MIT License](LICENSE). Manuscript and supplementary text remain copyright of the authors.