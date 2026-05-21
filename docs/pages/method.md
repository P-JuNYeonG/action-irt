---
layout: page
title: Methodology
permalink: /pages/method.html
---

# Methodology

## Overview

The framework converts raw log sequences into interpretable action-level effect estimates through three stages. Each stage addresses a specific challenge: encoding discrete behavioral events, reducing dimensionality while preserving sequential context, and performing principled variable selection within a psychometric model.

---

## Stage 1: Hybrid Word2Vec for Action Embedding

Each action unit is embedded into a dense vector using a modified skip-gram objective. The model exploits two sources of information:

1. **Sequential context** (Word2Vec): Actions appearing in similar local contexts receive similar representations.
2. **Compositional structure** (FastText): Action units are decomposed into tokens, and the unit embedding is the sum of unit-level and token-level vectors.

$$\mathbf{e}(u) = \mathbf{v}_u + \sum_{r=1}^{m} \mathbf{v}_{g_r}$$

**Temporal augmentation:** Each embedding is concatenated with three standardized temporal features (elapsed time, squared time, inter-action interval).

---

## Stage 2: LSTM AutoEncoder for Dimension Reduction

The augmented embedding sequence is compressed through an LSTM encoder-decoder architecture:

- **Encoder:** Processes the sequence step-by-step, producing a hidden state that summarizes preceding actions.
- **Bottleneck:** A linear projection maps each hidden state to the D-dimensional latent space.
- **Decoder:** Reconstructs the original embedding from the latent representation.

The latent value at each position encodes not only the action itself but also the sequential context in which it occurred.

$$\mathbf{C}_{ijn} = \mathbf{W}_{\text{proj}} \mathbf{h}_{ijn} + \mathbf{b}_{\text{proj}} \in \mathbb{R}^D$$

---

## Stage 3: Action-IRT Model with Spike-and-Slab Selection

The latent action representations enter an extended Rasch model:

$$\text{logit}(\pi_{ij}) = \alpha_i + \beta_j + \sum_{l=1}^{N_{.j.}} \left( \sum_{d=1}^{D} \omega_{jl}^{(d)} \bar{C}_{ijl}^{(d)} \right) I(l \in A_{ij})$$

**Variable selection** is achieved through a spike-and-slab prior on each action weight:

$$\omega_{jl}^{(d)} \mid \lambda_{jl}^{(d)} \sim \begin{cases} N(0, \tau^2) & \text{if } \lambda_{jl}^{(d)} = 0 \text{ (spike)} \\ N(0, \nu^2) & \text{if } \lambda_{jl}^{(d)} = 1 \text{ (slab)} \end{cases}$$

The posterior inclusion probability (PIP) quantifies the evidence that each action contributes to response accuracy beyond ability and difficulty.

### Discriminative Action Identification

An action is classified as **discriminative** when the 95% HPD interval of the group-contrast statistic does not contain zero:

$$\Delta E_{jl}^{(t)} = E_{jl}^{\mathcal{C}(t)} - E_{jl}^{\mathcal{I}(t)}$$

where the correct-group and incorrect-group effects are computed from posterior draws of action weights and observed group-level latent representations.
