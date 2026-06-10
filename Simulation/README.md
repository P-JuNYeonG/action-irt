# Stage 5: Parametric Bootstrap Simulation

## Purpose

Evaluate the variable-selection performance of the Action-IRT model by regenerating binary responses from empirically estimated parameters and checking whether the sampler recovers the known important actions.

This is a parametric-bootstrap simulation, not a fully synthetic data-generation study.

## Design

The following empirical structures are held fixed:

- respondent-item observation pattern
- action occurrence patterns for each respondent-item pair
- Stage 3 latent action summaries

Only the binary responses are regenerated:

```text
eta_ij = alpha_i^true + beta_j^true + sum_l omega_jl^true * Cbar_ijl
Y_ij ~ Bernoulli(logistic(eta_ij))
```

## True Parameter Construction

| Parameter | Source |
|-----------|--------|
| `alpha_i^true` | Posterior mean from the empirical fit |
| `beta_j^true` | Posterior mean from the empirical fit |
| `omega_jl^true`, important actions | Slab-conditional posterior mean |
| `omega_jl^true`, non-important actions | One draw from the spike distribution, fixed across seeds |
| `lambda_jl^true`, important actions | 1 |
| `lambda_jl^true`, non-important actions | 0 |

## Evaluation Metrics

| Metric | Definition |
|--------|------------|
| AUC | ROC AUC using true inclusion labels and estimated PIP scores |
| Sensitivity | Proportion of true important actions with PIP >= 0.5 |
| HPD-based sensitivity | Proportion of true important actions selected by the HPD rule |

## Settings

| Setting | Value |
|---------|-------|
| Seeds | 5 |
| Iterations per seed | 50,000 |
| Burn-in | 10,000 |
| Thinning | 10 |
| True important actions | 126 / 2,025 |

## Reported Results

| Metric | Mean | SD | Range |
|--------|------|----|-------|
| Simulated correct rate | 0.4300 | 0.0028 | 0.4269-0.4329 |
| Action AUC | 0.9431 | 0.0108 | 0.9257-0.9550 |
| Action sensitivity | 0.9349 | 0.0130 | 0.9127-0.9444 |

## Files

| File | Description |
|------|-------------|
| `multi_seed_simulation.R` | Main parametric-bootstrap simulation runner |

## Usage

Run from `Simulation/`:

```bash
Rscript multi_seed_simulation.R
```

The current simulation script is configured inside `multi_seed_simulation.R`. Before running, check the model setting, latent dimension, MCMC settings, input paths, and output directory in the script configuration section.

## Notes

- The script compiles the MCMC sampler from `../MCMC/MCMC.cpp`.
- The simulation requires empirical MCMC outputs and Stage 3 latent-action inputs.
- Generated simulation result files are not redistributed.
