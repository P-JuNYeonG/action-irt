# Stage 4: Action-IRT Model with Spike-and-Slab Selection

## Purpose

Estimate respondent ability, item difficulty, and action weights jointly via MCMC, with automatic variable selection over the action space using a spike-and-slab prior.

## Model

```
logit(π_ij) = α_i + β_j + Σ_l Σ_d ω^(d)_jl · C̄^(d)_ijl · I(l ∈ A_ij)
```

### Priors

| Parameter | Prior |
|-----------|-------|
| α_i | N(0, σ²_α) |
| β_j | N(0, 1) — fixed |
| ω^(d)_jl \| λ=0 | N(0, τ²) — spike |
| ω^(d)_jl \| λ=1 | N(0, ν²) — slab |
| λ^(d)_jl | Bernoulli(0.5) |
| σ²_α | Inverse-Gamma(2, 1) |

### MCMC Updates

| Parameter | Method |
|-----------|--------|
| α_i | Metropolis–Hastings (Gaussian random walk) |
| β_j | Metropolis–Hastings (Gaussian random walk) |
| ω^(d)_jl | Metropolis–Hastings (Gaussian random walk) |
| λ^(d)_jl | Gibbs (closed-form Bernoulli) |
| σ²_α | Gibbs (conjugate Inverse-Gamma) |

## Input

- Response matrix: n_resp × n_prob (binary)
- C_sum_list: per-item matrices of aggregated latent values (from Stage 3)
- N_j_vec: number of unique actions per item

## Output

- Posterior samples: α, β, W (action weights), λ (inclusion indicators), σ
- Acceptance rates for MH steps
- Stored as .RData

## Files

| File | Description |
|------|-------------|
| `MCMC.cpp` | C++/Rcpp implementation of the MCMC sampler |
| `run_mcmc.R` | R driver script: data loading, scaling, MCMC call, diagnostics |

## MCMC Settings

| Setting | Value |
|---------|-------|
| Iterations | 50,000 |
| Burn-in | 10,000 (20%) |
| Thinning | 10 |
| Saved samples | 4,000 |
| proposal_sd_alpha | 1.5 |
| proposal_sd_beta | 0.5 |
| proposal_sd_w | 0.4 |
| τ² (spike) | 0.001 |
| ν² (slab) | 2.5 |

## Usage

```r
# In R:
source("run_mcmc.R")

# Or from command line:
Rscript run_mcmc.R --model lstm_ae --dim 1 --pre robust --iter 50000
```

## Computational Notes

- The sampler uses an incremental WC_cache strategy to avoid recomputing the full linear predictor at each W update.
- Typical runtime: ~45–90 minutes for 50,000 iterations on a single core (depends on action vocabulary size).
