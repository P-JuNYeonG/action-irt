# Stage 2: Hybrid Word2Vec Action Embedding

## Purpose

Learn dense vector representations of action units by using both sequential context and the internal token structure of each action unit.

This stage receives cleaned action sequences from Stage 1 and produces respondent-level embedding matrices for Stage 3.

## Model

```
Composite action embedding:
  e(u) = v_u + sum_r v_g_r

Training pairs:
  1. Unit-Unit:  center action unit -> neighboring action units
  2. Token-Unit: tokens of center action unit -> neighboring action units

Objective:
  Skip-gram with negative sampling
```

## Input

Stage 1 writes one text file per item:

```text
model_input/HW2V/test_us_{item}.txt
```

Each line is a respondent-item action sequence.

## Output

The training script writes item-level embedding artifacts, including:

| Output | Description |
|--------|-------------|
| `fasttext_style_word2vec.pkl` | Trained Hybrid Word2Vec model |
| `vocabulary.txt` | Token and unit vocabulary |
| `training_log.json` | Training-loss and validation information |
| `unit_embeddings_fasttext.npz` | Action-unit embedding matrix |
| `token_embeddings.npz` | Token embedding matrix |

`build_embedding_matrices.py` then creates respondent-level matrices:

```text
{output_dir}/embed_mat_ps1_{suffix}.pkl
{output_dir}/embed_mat_ps2_{suffix}.pkl
```

Each dictionary key is a respondent-item identifier and each value is an `N_ij x (D_Action + 3)` matrix when timestamp features are included.

## Temporal Augmentation

After action embedding, each action vector can be augmented with:

- elapsed time
- squared elapsed time
- inter-action interval

Temporal features are scaled to be comparable with the embedding dimensions unless `--no-scale-to-embedding` is used.

## Key Hyperparameters

| Parameter | Default |
|-----------|---------|
| Embedding dimension | 20 |
| Context window size | 1 |
| Negative samples | 5 |
| Training epochs | 100 |
| Learning rate | 0.001 |
| Minimum count | 1 |

## Files

| File | Description |
|------|-------------|
| `HW2V/data_loader.py` | Loads action sequences and creates training pairs |
| `HW2V/vocab_builder.py` | Builds token/unit vocabularies and negative-sampling distributions |
| `HW2V/model.py` | Implements the FastText-style Token-Unit Word2Vec model |
| `HW2V/trainer.py` | Implements optimization, negative sampling, validation, and checkpoint saving |
| `HW2V/train_word2vec.py` | Command-line training entry point |
| `build_embedding_matrices.py` | Builds respondent-level embedding matrices and appends timestamp features |
| `HW2V/utils.py` | Optional embedding analysis and export helpers |
| `LSTM_AE/` | Stage 3 dimension-reduction scripts |

## Usage

Run the following from `Module/HW2V/` to train one item-level embedding model:

```bash
python train_word2vec.py \
  --input_file ../../model_input/HW2V/test_us_ps1_1.txt \
  --output_dir ../outputs/HW2V_ps1_1_20 \
  --embed_dim 20 \
  --window_size 1 \
  --negative_samples 5 \
  --learning_rate 0.001 \
  --epochs 100
```

Run the following from `Module/` after item-level models have been trained:

```bash
python build_embedding_matrices.py \
  --data-template ../Data/Preprocessing/input_data/3rd_data/us_{problem_num}.pkl \
  --model-template outputs/HW2V_{problem_num}_20/fasttext_style_word2vec.pkl \
  --sequence-template ../model_input/HW2V/test_us_{problem_num}.txt \
  --output-dir outputs
```

To build matrices for selected items only:

```bash
python build_embedding_matrices.py \
  --items ps1_1 ps1_2 \
  --data-template ../Data/Preprocessing/input_data/3rd_data/us_{problem_num}.pkl \
  --model-template outputs/HW2V_{problem_num}_20/fasttext_style_word2vec.pkl \
  --sequence-template ../model_input/HW2V/test_us_{problem_num}.txt \
  --output-dir outputs
```

Large trained models and generated embedding matrices are not redistributed.
