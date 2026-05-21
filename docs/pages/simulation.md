---
layout: page
title: Simulation
permalink: /pages/simulation.html
---

# Parametric Bootstrap Simulation

## Design

The simulation evaluates whether the spike-and-slab Action-IRT model can recover a known set of important actions under data conditions that approximate the empirical application.

**Key design choice:** This is a parametric bootstrap, not a fully synthetic simulation. The observed respondent–item structure, action occurrence patterns, and LSTM latent values are held fixed at their empirical values. Only binary responses are regenerated from the estimated true parameters.

---

## True Parameter Sources

| Component | Source |
|-----------|--------|
| Respondent ability (α) | Posterior mean from empirical fit |
| Item difficulty (β) | Posterior mean from empirical fit |
| Important action weights (ω) | Slab-conditional posterior mean |
| Non-important action weights (ω) | Single draw from N(0, 0.001), fixed across seeds |
| Important actions | 126 actions identified via 95% HPD criterion |

---

## Settings

| Setting | Value |
|---------|-------|
| Respondents | 1,996 |
| Items | 14 |
| Total action–item combinations | 2,025 |
| True important actions | 126 (6.22%) |
| Replications | 5 |
| MCMC iterations per seed | 50,000 |
| Burn-in | 10,000 |
| Thinning | 10 |
| Saved samples | 4,000 |
| τ² (spike) | 0.001 |
| ν² (slab) | 2.5 |

---

## Results

| Metric | Mean | SD | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Simulated correct rate | 0.4300 | 0.0028 | 0.4303 | 0.4269 | 0.4329 |
| Action AUC | 0.9431 | 0.0108 | 0.9460 | 0.9257 | 0.9550 |
| Action sensitivity (PIP ≥ 0.5) | 0.9349 | 0.0130 | 0.9365 | 0.9127 | 0.9444 |

---

## Interpretation

- PIP reliably separates truly important actions from non-important ones (AUC > 0.92 in all 5 seeds).
- The PIP ≥ 0.5 threshold recovers over 91% of true important actions in every replication.
- Results are stable across seeds, with low variability in all metrics.
- The parametric bootstrap design is more realistic than fully synthetic simulations because it preserves the actual covariate structure, sparsity level, and missingness pattern from the PIAAC data.
