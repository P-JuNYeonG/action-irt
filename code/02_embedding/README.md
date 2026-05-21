# Stage 2: Hybrid Word2Vec for Action Embedding

## Purpose

Learn dense vector representations of action units by exploiting sequential context (skip-gram) and internal token structure (FastText-style composition).

## Model

```
Composite embedding:  e(u) = v_u + Σ v_g_r

Training pairs:
  (1) Unit–Unit:   center unit → neighboring units within context window
  (2) Token–Unit:  each token of center unit → same neighboring units

Objective: Skip-gram with negative sampling
```

## Input

- Cleaned action sequences from Stage 1

## Output

- Action embedding vectors: one vector per unique action unit (dimension: D_Action = 20)
- Per-respondent embedding matrices: N_ij × D_Action per (respondent, item) pair

## Temporal Augmentation

After embedding, each action vector is augmented with three temporal features:
- Elapsed time: t
- Squared elapsed time: t²
- Inter-action interval: t_n − t_{n−1}

Temporal features are standardized within each item.

Final output dimension per action: D_Action + 3

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding dimension (D_Action) | 20 |
| Context window size | 5 |
| Negative samples | 5 |
| Training epochs | 50 |
| Learning rate | 0.025 |

## Usage

```bash
python train_word2vec.py --config config.yaml
```
