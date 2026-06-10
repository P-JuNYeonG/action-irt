---
layout: home
title: Action-IRT
---

<section class="paper-hero">

<h1>A Representation-Learning Item Response Model for Identifying Behaviorally Important Actions in PIAAC Process Data</h1>

<p class="paper-meta"><strong>Junyeong Park, Daeun Hwangbo, Seyoung Park, Ick Hoon Jin, and Minjeong Jeon</strong></p>
<p class="paper-meta">Submitted manuscript for <em>Psychometrika</em></p>

<div class="paper-actions">
  <a class="button-link" href="https://github.com/P-JuNYeonG/action-irt">Repository</a>
  <a class="button-link" href="{{ '/pages/method.html' | relative_url }}">Method</a>
  <a class="button-link" href="{{ '/pages/results.html' | relative_url }}">Results</a>
  <a class="button-link" href="{{ '/pages/simulation.html' | relative_url }}">Simulation</a>
</div>

</section>

## Abstract

Problem-solving log process data collected in online environments contain rich information about item difficulty, respondent ability, and problem-solving strategies. These logs are complex and noisy, however, which makes it difficult to identify behaviors that substantially contribute to successful problem solving. This project proposes a representation-learning item response theory model that encodes raw log sequences through action embeddings, compresses action-level representations with an LSTM autoencoder, and estimates action effects in an extended Rasch model with spike-and-slab variable selection. The framework is applied to log process data from the OECD Programme for the International Assessment of Adult Competencies (PIAAC), Problem Solving in Technology-Rich Environments (PSTRE) domain.

## Study At A Glance

<div class="summary-grid">
  <div class="summary-item"><strong>14</strong>PSTRE items from the U.S. PIAAC sample</div>
  <div class="summary-item"><strong>1,996</strong>respondents in the empirical analysis</div>
  <div class="summary-item"><strong>2,025</strong>action-item combinations evaluated</div>
  <div class="summary-item"><strong>126</strong>important actions identified</div>
</div>

## Framework

<div class="pipeline">
  <div class="pipeline-step">
    <strong>1. Action embedding</strong>
    Hybrid Word2Vec combines local sequential context with token-level structure in action labels.
  </div>
  <div class="pipeline-step">
    <strong>2. Dimension reduction</strong>
    An LSTM autoencoder compresses time-augmented action embeddings into low-dimensional latent values.
  </div>
  <div class="pipeline-step">
    <strong>3. Action-IRT</strong>
    An extended Rasch model uses spike-and-slab priors to select actions associated with response accuracy.
  </div>
</div>

## Repository Contents

| Path | Contents |
|------|----------|
| `code/` | Preprocessing, action embedding, LSTM autoencoder, MCMC, and simulation code |
| `manuscript/` | LaTeX source for the manuscript |
| `docs/` | This GitHub Pages companion site |
| `supplementary/` | Additional prompts, tables, and appendix materials |

## Data Availability

The analysis uses PIAAC PSTRE log process data. Raw log files are not redistributed in this repository because of data-use restrictions. Public-use PIAAC data files are available from the [OECD PIAAC Data Portal](https://www.oecd.org/skills/piaac/data/).
