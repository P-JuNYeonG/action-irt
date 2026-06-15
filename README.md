# Action-IRT: A Representation-Learning Item Response Model for Identifying Behaviorally Important Actions in PIAAC Process Data

> Junyeong Park, Daeun Hwangbo, Seyoung Park, Ick Hoon Jin, Minjeong Jeon
>
> Submitted to *Psychometrika*

---

## Overview

This repository contains code and supplementary materials for a three-stage framework that identifies behaviorally important actions in problem-solving process data. Raw log sequences are embedded via a hybrid Word2Vec model, compressed to low-dimensional latent values via an LSTM AutoEncoder, and entered into an extended Rasch model with a spike-and-slab prior for automatic variable selection.

Applied to 14 PSTRE items from the U.S. PIAAC sample (1,996 respondents), the framework identified 126 of 2,025 action–item combinations (6.2%) as important. A parametric-bootstrap simulation yielded action-level AUC of 0.943 and PIP-threshold sensitivity of 0.935.

---

## Repository Structure

```
action-irt/
├── setup_r_env.R           # R package setup
│
├── Data_Preprocess/        # Rule-based + LLM-assisted action-sequence construction
├── Embedding/HW2V/         # Hybrid Word2Vec action embedding
├── Embedding/LSTM_AE/      # LSTM AutoEncoder dimension reduction
├── MCMC/                   # Action-IRT sampler (C++/Rcpp) and R wrapper, Parametric-bootstrap recovery evaluation
└── Analysis/               # Trace plots
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

Core Python packages: `numpy`, `pandas`, `torch`, `scikit-learn`.

---

## Usage

Run stages in order:

1. `Data/Preprocessing/` — clean raw logs and construct action sequences
2. `Module/HW2V/` — train action embeddings
3. `Module/LSTM_AE/` — reduce embeddings to latent values
4. `MCMC/` — estimate the Action-IRT model
5. `Simulation/` — run parametric-bootstrap simulation

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

## Contact

For questions, please open a GitHub issue or contact the corresponding author: Ick Hoon Jin (ijin@yonsei.ac.kr).