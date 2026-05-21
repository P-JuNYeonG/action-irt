---
layout: home
title: Action-IRT
---

# Action-IRT: Identifying Important Actions in Problem-Solving Process Data

**Paper:** *Item Response Model for Online Educational Assessment Dataset with Log Sequence Dataset*

**Authors:** Junyeong Park, Seyoung Park, Ick Hoon Jin, Minjeong Jeon

**Status:** Submitted to *Psychometrika*

---

## Abstract

Problem-solving log process data collected in online environments contain rich information, including item difficulty, respondent ability, and problem-solving strategies. However, such log process data are complex and noisy, making it challenging to identify behaviors that substantially contribute to problem-solving. In this paper, we propose an extended Item Response Theory (IRT) model that incorporates a new behavioral term by embedding raw log data through hierarchical relationships with surrounding behaviors using an embedding framework from Natural Language Processing. We apply the spike-and-slab prior, a Bayesian variable selection method, to selectively identify behaviors that significantly influence correct or incorrect responses. We apply our framework to data from the OECD Programme for the International Assessment of Adult Competencies (PIAAC) for measuring Problem Solving in Technology-Rich Environments (PSTRE).

---

## Framework Overview

The proposed framework consists of three stages:

| Stage | Method | Purpose |
|-------|--------|---------|
| 1. Action Embedding | Hybrid Word2Vec (skip-gram + FastText) | Encode sequential context and token structure of actions |
| 2. Dimension Reduction | LSTM AutoEncoder | Compress variable-length sequences to D-dimensional latent values |
| 3. Action-IRT | Extended Rasch model with spike-and-slab prior | Estimate action weights with automatic variable selection |

---

## Key Results

- **14 PSTRE items** analyzed from the U.S. PIAAC study (1,996 respondents)
- **126 important actions** identified out of 2,025 action–item combinations (6.2%)
- **Simulation validation:** Action-level AUC = 0.943, PIP-based sensitivity = 0.935

---

## Navigation

- [Methodology](pages/method.html) — Detailed description of the three-stage framework
- [Results](pages/results.html) — Empirical findings and item-level case studies
- [Simulation](pages/simulation.html) — Parametric bootstrap simulation design and results
- [Appendix](pages/appendix.html) — Additional tables and implementation details

---

## Code & Data

- **Code repository:** [GitHub](https://github.com/P-JuNYeonG/action-irt)
- **Data:** PIAAC public-use files available from the [OECD PIAAC Data Portal](https://www.oecd.org/skills/piaac/data/)
