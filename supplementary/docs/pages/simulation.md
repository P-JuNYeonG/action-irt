---
layout: page
title: Simulation
permalink: /pages/simulation.html
---

# Parametric Bootstrap Simulation

The simulation evaluates whether the spike-and-slab Action-IRT model can recover a known set of important actions under data conditions close to the empirical application.

<p class="section-note">The design is a parametric bootstrap rather than a fully synthetic simulation. The empirical respondent-item structure, action occurrence patterns, and LSTM latent values are held fixed. Binary responses are regenerated from specified true parameters.</p>

## Data-Generating Components

| Component | Source |
|-----------|--------|
| Respondent ability \(\alpha_i\) | Posterior mean from the empirical model |
| Item difficulty \(\beta_j\) | Posterior mean from the empirical model |
| Important action weights \(\omega\) | Slab-conditional posterior mean |
| Non-important action weights \(\omega\) | Single draw from \(N(0, 0.001)\), fixed across seeds |
| True important actions | 126 actions selected in the empirical analysis |

## Simulation Settings

| Setting | Value |
|---------|------:|
| Respondents | 1,996 |
| Items | 14 |
| Total action-item combinations | 2,025 |
| True important actions | 126 (6.22%) |
| Replications | 5 |
| MCMC iterations per replication | 50,000 |
| Burn-in | 10,000 |
| Thinning | 10 |
| Saved samples | 4,000 |
| Spike variance \(\tau^2\) | 0.001 |
| Slab variance \(\nu^2\) | 2.5 |

## Recovery Results

| Metric | Mean | SD | Median | Min | Max |
|--------|-----:|---:|-------:|----:|----:|
| Simulated correct-response rate | 0.4300 | 0.0028 | 0.4303 | 0.4269 | 0.4329 |
| Action-level AUC | 0.9431 | 0.0108 | 0.9460 | 0.9257 | 0.9550 |
| Sensitivity at PIP >= 0.5 | 0.9349 | 0.0130 | 0.9365 | 0.9127 | 0.9444 |

## Interpretation

The action-level AUC is above 0.92 in all five replications, indicating that posterior inclusion probabilities separate true important actions from non-important actions under the bootstrap design. Sensitivity under the PIP >= 0.5 rule is also above 0.91 in every replication. These results suggest that the variable-selection component can recover sparse action effects when the data-generating structure resembles the empirical PIAAC application.
