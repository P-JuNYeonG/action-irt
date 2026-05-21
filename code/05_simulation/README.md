# Stage 5: Parametric Bootstrap Simulation

## Purpose

Evaluate the variable-selection performance of the Action-IRT model by generating bootstrap response data from empirically estimated parameters and assessing recovery of known important actions.

## Design

This is a **parametric bootstrap** simulation, not a fully synthetic study. The following are held fixed at their empirical values:

- Observed respondent–item structure (missingness pattern)
- Action occurrence patterns per (respondent, item)
- LSTM latent values C̄_ijl

Only the binary responses Y_ij are regenerated:

```
η_ij = α_i^true + β_j^true + Σ_l ω_jl^true · C̄_ijl
Y_ij ~ Bernoulli(logistic(η_ij))
```

### True Parameter Construction

| Parameter | Source |
|-----------|--------|
| α_i^true | Posterior mean from empirical fit |
| β_j^true | Posterior mean from empirical fit |
| ω_jl^true (important) | Slab-conditional posterior mean |
| ω_jl^true (non-important) | Drawn once from N(0, τ²), fixed across seeds |
| λ_jl^true (important) | 1 |
| λ_jl^true (non-important) | 0 |

### Evaluation Metrics

| Metric | Definition |
|--------|------------|
| AUC | ROC AUC using true λ as label, estimated PIP as score |
| Sensitivity | Proportion of true important actions with PIP ≥ 0.5 |

## Settings

| Setting | Value |
|---------|-------|
| Seeds | 5 (randomly generated) |
| Iterations per seed | 50,000 |
| Burn-in | 10,000 |
| Thinning | 10 |
| True important actions | 126 / 2,025 (6.22%) |

## Results

| Metric | Mean | SD | Range |
|--------|------|----|-------|
| Simulated correct rate | 0.4300 | 0.0028 | 0.4269–0.4329 |
| Action AUC | 0.9431 | 0.0108 | 0.9257–0.9550 |
| Action sensitivity | 0.9349 | 0.0130 | 0.9127–0.9444 |

## Usage

```r
Rscript multi_seed_simulation.R --n_seeds 5 --iter 50000
```
