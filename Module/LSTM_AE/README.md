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
- Stored as a pickle dictionary: `{(seq_id, problem_num): numpy.ndarray}`

## Output

- Reduced latent matrices: N_ij × D per (respondent, item)
- Saved model weights and reconstruction-loss curves
- Optional long-format CSV for Action-IRT:
  `seq_id, problem_num, behavior_id, action_name, outcome, N_ij, C_value1, ..., C_valueD`

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Encoder hidden dim (H_enc) | 64 |
| Latent dimension (D) | 1 (primary), 2–5 (sensitivity) |
| Decoder hidden dim (H_dec) | 64 |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 200 |
| Early stopping patience | 20 |

## Scaling

The training script exports raw LSTM-AE latent values. The export script robust-scales latent values within each item before writing the Action-IRT CSV unless `--no-robust-scale` is used:

```
C_scaled = (C − median(C)) / IQR(C)
```

## Files

| File | Description |
|------|-------------|
| `train_lstm_ae.py` | Trains item-wise LSTM autoencoders and writes reduced latent pickle files |
| `export_latent_long.py` | Converts reduced latent pickle files to Action-IRT long-format CSV |

## Usage

Train LSTM autoencoders for a problem group:

```bash
python train_lstm_ae.py \
  --input-pkl ../02_embedding/outputs/embed_mat_ps1_20.pkl \
  --output-dir outputs/lstm_ae_ps1 \
  --item-group ps1 \
  --latent-dims 1,2,3,4,5 \
  --hidden-dim 64 \
  --batch-size 64 \
  --epochs 200 \
  --lr 0.001
```

Train one item and one latent dimension:

```bash
python train_lstm_ae.py \
  --input-pkl ../02_embedding/outputs/embed_mat_ps1_20.pkl \
  --output-dir outputs/lstm_ae_ps1_1_D1 \
  --items ps1_1 \
  --latent-dims 1
```

Export a reduced latent pickle to the long-format CSV used by Action-IRT:

```bash
python export_latent_long.py \
  --latent-pkl outputs/lstm_ae_ps1/lstm_ae_reduced_D1.pkl \
  --output-csv outputs/long_format_lstm_ae_ps1_D1.csv \
  --outcome-csv ../../data/prgusap1.csv \
  --seq-event-dir ../../model_input/IRT \
  --unique-action-dir ../../data/unique_actions
```

Large trained model files, reduced latent pickle files, loss-curve images, and generated CSV outputs are not redistributed in this repository.
