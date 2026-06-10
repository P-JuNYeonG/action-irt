# Stage 4: Action-IRT Model with Spike-and-Slab Selection

## Purpose

Estimate respondent ability, item difficulty, and action effects jointly with MCMC. The model uses a spike-and-slab prior to select action effects associated with response accuracy.

## Model

```text
logit(pi_ij) = alpha_i + beta_j + sum_l sum_d omega_jl^(d) * Cbar_ijl^(d) * I(l in A_ij)
```

where:

- `alpha_i` is respondent ability.
- `beta_j` is item difficulty.
- `Cbar_ijl^(d)` is the aggregated latent action value from Stage 3.
- `omega_jl^(d)` is the action weight.
- `lambda_jl^(d)` is the spike-and-slab inclusion indicator.

## Priors

| Parameter | Prior |
|-----------|-------|
| `alpha_i` | Normal prior with variance `sigma_alpha^2` |
| `beta_j` | Normal prior with fixed variance |
| `omega_jl^(d)`, when `lambda = 0` | Spike normal prior with variance `tau^2` |
| `omega_jl^(d)`, when `lambda = 1` | Slab normal prior with variance `nu^2` |
| `lambda_jl^(d)` | Bernoulli prior |
| `sigma_alpha^2` | Inverse-Gamma prior |

## MCMC Updates

| Parameter | Method |
|-----------|--------|
| `alpha_i` | Metropolis-Hastings random walk |
| `beta_j` | Metropolis-Hastings random walk |
| `omega_jl^(d)` | Metropolis-Hastings random walk |
| `lambda_jl^(d)` | Gibbs update |
| `sigma_alpha^2` | Conjugate Gibbs update |

## Input

The empirical MCMC script expects locally generated model inputs from previous stages, including:

- binary response data
- aggregated latent action values from Stage 3
- per-item action-count information

The exact local paths are configured inside `run_mcmc.R`; update them to match your local data layout before running.

## Output

Typical outputs include:

- posterior samples for `alpha`, `beta`, `W`, `lambda`, and `sigma`
- Metropolis-Hastings acceptance rates
- posterior inclusion probabilities (PIPs)
- saved `.RData` result objects

## Files

| File | Description |
|------|-------------|
| `MCMC.cpp` | C++/Rcpp implementation of the MCMC sampler |
| `run_mcmc.R` | R driver for data loading, scaling, sampler compilation, MCMC execution, and result saving |

## Empirical Settings

| Setting | Value |
|---------|-------|
| Iterations | 50,000 |
| Burn-in | 10,000 |
| Thinning | 10 |
| Saved samples | 4,000 |
| `proposal_sd_alpha` | 1.5 |
| `proposal_sd_beta` | 0.5 |
| `proposal_sd_w` | 0.4 |
| Spike variance `tau^2` | 0.001 |
| Slab variance `nu^2` | 2.5 |

## Usage

Run from `MCMC/`:

```bash
Rscript run_mcmc.R
```

The current script is configured primarily through variables and paths inside `run_mcmc.R`. If you need to change model type, latent dimension, preprocessing variant, iteration count, or output paths, edit the configuration section of that script before running.

## Computational Notes

- `run_mcmc.R` compiles `MCMC.cpp` via Rcpp before running the sampler.
- The sampler uses an incremental cache strategy for the action-effect term to reduce repeated linear-predictor computation.
- Runtime depends on the number of respondents, items, actions, latent dimensions, and MCMC iterations.
