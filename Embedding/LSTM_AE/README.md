# Stage 3: LSTM AutoEncoder Dimension Reduction

## Purpose

Compress augmented action embedding sequences into low-dimensional latent action values for the Action-IRT model.

The input to this stage is a variable-length sequence matrix for each respondent-item pair. The output is a latent sequence matrix with the same action order and a smaller feature dimension.

## Architecture

```text
Input sequence: N_ij x (D_Action + 3)
    |
    v
LSTM encoder
    |
    v
Linear bottleneck projection
    |
    v
Latent sequence: N_ij x D
    |
    v
Linear expansion + LSTM decoder
    |
    v
Reconstructed input sequence
```

The loss is mean squared error over non-padded sequence positions.

## Input

`train_lstm_ae.py` expects a pickle dictionary:

```python
{(seq_id, problem_num): numpy.ndarray}
```

where each array is an `N_ij x (D_Action + 3)` embedding matrix produced by Stage 2.

## Output

Typical outputs include:

| Output | Description |
|--------|-------------|
| `lstm_ae_reduced_D{D}.pkl` | Reduced latent matrices for a latent dimension |
| model checkpoint files | Optional saved model weights |
| loss-curve figures | Optional training/validation loss plots |
| long-format CSV | Optional Action-IRT input exported by `export_latent_long.py` |

The long-format CSV includes columns such as:

```text
seq_id, problem_num, behavior_id, action_name, outcome, N_ij, C_value1, ...
```

## Key Hyperparameters

| Parameter | Default |
|-----------|---------|
| Encoder/decoder hidden dimension | 64 |
| Latent dimensions | 1,2,3,4,5 |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 200 |
| Early-stopping patience | 20 |
| Validation ratio | 0.1 |

## Scaling

`train_lstm_ae.py` exports raw latent values. `export_latent_long.py` robust-scales latent values within each item before writing the Action-IRT CSV unless `--no-robust-scale` is used:

```text
C_scaled = (C - median(C)) / IQR(C)
```

## Files

| File | Description |
|------|-------------|
| `train_lstm_ae.py` | Trains item-wise LSTM autoencoders and writes reduced latent pickle files |
| `export_latent_long.py` | Converts reduced latent pickle files to Action-IRT long-format CSV |

## Usage

Run commands from `Module/LSTM_AE/`.

Train LSTM autoencoders for all `ps1` items and multiple latent dimensions:

```bash
python train_lstm_ae.py \
  --input-pkl ../outputs/embed_mat_ps1_20.pkl \
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
  --input-pkl ../outputs/embed_mat_ps1_20.pkl \
  --output-dir outputs/lstm_ae_ps1_1_D1 \
  --items ps1_1 \
  --latent-dims 1
```

Export reduced latent values to the long-format CSV used by Action-IRT:

```bash
python export_latent_long.py \
  --latent-pkl outputs/lstm_ae_ps1/lstm_ae_reduced_D1.pkl \
  --output-csv outputs/long_format_lstm_ae_ps1_D1.csv \
  --outcome-csv ../../data/prgusap1.csv \
  --seq-event-dir ../../model_input/IRT \
  --unique-action-dir ../../data/unique_actions
```

Adjust the data paths to match the local location of response files, sequence files, and action-name mappings.

Large trained model files, reduced latent pickle files, loss-curve images, and generated CSV outputs are not redistributed.
