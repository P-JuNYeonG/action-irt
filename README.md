# Action-IRT: Identifying Important Actions in Problem-Solving Process Data via an Extended Item Response Model

> **Paper:** *Item Response Model for Online Educational Assessment Dataset with Log Sequence Dataset*
>
> Junyeong Park, Seyoung Park, Ick Hoon Jin, Minjeong Jeon
>
> Submitted to *Psychometrika*

---

## Overview

This repository contains code and supplementary materials for an end-to-end Action-IRT workflow. The workflow converts raw problem-solving process logs into action sequences, learns action-level representations, reduces those representations to low-dimensional latent values, and estimates an Item Response Theory model with action effects and spike-and-slab variable selection.

The empirical application uses PIAAC Problem Solving in Technology-Rich Environments (PSTRE) log data. Raw OECD data and generated intermediate files are not redistributed in this repository.

```
Raw PIAAC logs
    |
    v
Data/Preprocessing
    Rule-based and LLM-assisted action-sequence construction
    |
    v
Module/HW2V
    Hybrid Word2Vec action embedding
    |
    v
Module/LSTM_AE
    LSTM AutoEncoder dimension reduction
    |
    v
MCMC
    Action-IRT estimation with spike-and-slab selection
    |
    v
Simulation
    Parametric-bootstrap recovery evaluation
```

---

## Repository Structure

```
action-irt/
├── README.md
├── LICENSE
├── config.yaml                 # Central configuration values used by analysis scripts
├── setup_r_env.R               # R package setup helper
│
├── Data/
│   ├── README.md
│   └── Preprocessing/
│       ├── README.md
│       ├── 01_preprocessing_notebook.ipynb
│       ├── 01_preprocess_rules.py
│       ├── 02_llm_clean_descriptions.py
│       ├── openrouter_client.py
│       ├── prompts.py
│       └── mapping_tables/
│
├── Module/
│   ├── README.md
│   ├── build_embedding_matrices.py
│   ├── HW2V/
│   │   ├── train_word2vec.py
│   │   ├── data_loader.py
│   │   ├── model.py
│   │   ├── trainer.py
│   │   ├── utils.py
│   │   └── vocab_builder.py
│   └── LSTM_AE/
│       ├── README.md
│       ├── train_lstm_ae.py
│       └── export_latent_long.py
│
├── MCMC/
│   ├── README.md
│   ├── MCMC.cpp
│   └── run_mcmc.R
│
├── Simulation/
│   ├── README.md
│   └── multi_seed_simulation.R
│
├── Figures/
│   └── Selected figures for the manuscript and supplementary materials
│
└── Supplementary/
    ├── manuscript/
    ├── docs/
    ├── full_item_tables/
    └── llm_prompts/
```

---

## Data Availability

The analysis uses log process data from the OECD Programme for the International Assessment of Adult Competencies (PIAAC), specifically the PSTRE domain.

- PIAAC public-use files are available from the [OECD PIAAC Data Portal](https://www.oecd.org/skills/piaac/data/).
- Raw log files are not included here because redistribution is governed by OECD data-use terms.
- Scripts assume that users place locally obtained raw and processed data in the expected local input paths before running the full pipeline.

---

## Software Requirements

| Software | Purpose |
|----------|---------|
| R >= 4.3 | MCMC estimation, simulation, diagnostics |
| Python >= 3.9 | Preprocessing utilities, Hybrid Word2Vec, LSTM AutoEncoder |
| C++17-compatible compiler | Rcpp/RcppArmadillo compilation for the MCMC sampler |

Install R dependencies from the repository root:

```bash
Rscript setup_r_env.R
```

Python dependencies depend on the local environment used for embedding and LSTM-AE training. The main Python scripts use common scientific packages such as `numpy`, `pandas`, `torch`, `matplotlib`, and `scikit-learn`.

---

## Pipeline Guide

Run each stage from the directory that contains the relevant script unless you adapt the paths.

1. **Data preprocessing**

   See `Data/Preprocessing/README.md`.

   Main outputs:
   - `model_input/HW2V/test_us_{item}.txt`
   - `model_input/IRT/test_us_{item}.csv`
   - third-pass item pickle files

2. **Action embedding**

   See `Module/README.md`.

   Main scripts:
   - `Module/HW2V/train_word2vec.py`
   - `Module/build_embedding_matrices.py`

3. **Dimension reduction**

   See `Module/LSTM_AE/README.md`.

   Main scripts:
   - `Module/LSTM_AE/train_lstm_ae.py`
   - `Module/LSTM_AE/export_latent_long.py`

4. **Action-IRT estimation**

   See `MCMC/README.md`.

   Main scripts:
   - `MCMC/MCMC.cpp`
   - `MCMC/run_mcmc.R`

5. **Parametric-bootstrap simulation**

   See `Simulation/README.md`.

   Main script:
   - `Simulation/multi_seed_simulation.R`

---

## Empirical Settings

| Setting | Value |
|---------|-------|
| Respondents | 1,996 |
| PSTRE items | 14 |
| Action-item combinations | 2,025 |
| Primary latent dimension | D = 1 |
| MCMC iterations | 50,000 |
| Burn-in | 10,000 |
| Thinning | 10 |
| Saved posterior samples | 4,000 |
| Spike variance | 0.001 |
| Slab variance | 2.5 |

---

## Reported Results

The empirical analysis identified 126 important action-item combinations out of 2,025 candidates. The parametric-bootstrap simulation reported an action-level AUC of approximately 0.94 and PIP-threshold sensitivity of approximately 0.93.

See the manuscript and supplementary materials for the full interpretation, item-level tables, and simulation details.

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

Code is released under the [MIT License](LICENSE). Manuscript and supplementary text remain copyright of the authors.
