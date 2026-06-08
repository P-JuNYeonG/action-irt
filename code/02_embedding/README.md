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
| Context window size | 1 |
| Negative samples | 5 |
| Training epochs | 100 |
| Learning rate | 0.001 |

## Files

| File | Description |
|------|-------------|
| `data_loader.py` | Loads action-sequence text files and creates Unit-Unit and Token-Unit training pairs |
| `vocab_builder.py` | Builds token/unit vocabularies and negative-sampling distributions |
| `model.py` | Implements the FastText-style Token-Unit Word2Vec model |
| `trainer.py` | Implements Adam optimization, negative sampling, validation, and checkpoint saving |
| `train_word2vec.py` | Command-line training entry point |
| `build_embedding_matrices.py` | Builds respondent-level embedding matrices and appends timestamp features |
| `utils.py` | Optional embedding analysis and export helpers |

## Usage

Train one item-level embedding model:

```bash
python train_word2vec.py \
  --input_file ../../model_input/HW2V/test_us_ps1_1.txt \
  --output_dir outputs/HW2V_ps1_1_20 \
  --embed_dim 20 \
  --window_size 1 \
  --negative_samples 5 \
  --learning_rate 0.001 \
  --epochs 100
```

Build embedding matrices with timestamp features after item-level models have been trained:

```bash
python build_embedding_matrices.py \
  --data-template ../01_preprocessing/input_data/3rd_data/us_{problem_num}.pkl \
  --model-template outputs/HW2V_{problem_num}_20/fasttext_style_word2vec.pkl \
  --sequence-template ../../model_input/HW2V/test_us_{problem_num}.txt \
  --output-dir ../../model_output
```

Large trained model files and generated embedding matrices are not redistributed in this repository.
