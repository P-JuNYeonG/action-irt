# Stage 3: LSTM AutoEncoder for Dimension Reduction

## Purpose

Compress the augmented action embedding sequence (N_ij × (D_Action + 3)) into a low-dimensional latent representation (N_ij × D) that preserves sequential context.

## Architecture

```
Input: augmented embedding sequence (N_ij × 23)
    │
    ▼
┌──────────────────┐
│  LSTM Encoder     │  → hidden state h_n at each position
│  (H_enc units)    │
└──────────────────┘
    │
    ▼  Linear projection: W_proj · h_n + b_proj
┌──────────────────┐
│  Latent space     │  → C_n ∈ R^D  (bottleneck)
│  (D dimensions)   │
└──────────────────┘
    │
    ▼  Linear expansion + LSTM Decoder
┌──────────────────┐
│  LSTM Decoder     │  → reconstructed embedding ê_n
│  (H_dec units)    │
└──────────────────┘
    │
    ▼
Output: reconstructed sequence (N_ij × 23)

Loss: MSE over non-padded positions
```

## Input

- Augmented embedding matrices from Stage 2: N_ij × (D_Action + 3) per (respondent, item)

## Output

- Latent matrices: N_ij × D per (respondent, item)
- Saved as long-format CSV: `seq_id, problem_num, behavior_id, C_value1, ..., C_valueD`

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Encoder hidden dim (H_enc) | 64 |
| Latent dimension (D) | 1 (primary), 2–5 (sensitivity) |
| Decoder hidden dim (H_dec) | 64 |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 200 |

## Scaling

The latent values C are scaled within each item using robust scaling before entering the IRT model:

```
C_scaled = (C − median(C)) / IQR(C)
```

## Usage

```bash
python train_lstm_ae.py --item ps1_1 --latent_dim 1
```
