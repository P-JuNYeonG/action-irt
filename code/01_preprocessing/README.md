# Stage 1: Data Preprocessing

## Purpose

Transform raw PIAAC PSTRE log event data into clean, ordered action sequences suitable for embedding.

## Pipeline

```
Raw PIAAC logs (.csv)
    │
    ├── 1st pass (rule-based, human)
    │   ├── Restart timestamp adjustment
    │   ├── Consecutive identical action merging
    │   ├── Redundant row removal
    │   └── Keypress collapsing (consecutive → count)
    │
    ├── 2nd pass (human + LLM)
    │   ├── Noisy description cleaning
    │   └── Unit-Token pair normalization
    │
    └── 3rd pass (rule-based, human)
        └── Final unit = event_type + cleaned_description
            │
            ▼
    Cleaned action sequences per (respondent, item)
```

## Input

- Raw PIAAC log files with columns: `SEQID`, `BookletID`, `BlockName`, `event_type`, `event_description`, `time`
- Source: [OECD PIAAC Data Portal](https://www.oecd.org/skills/piaac/data/)

## Output

- Per-item CSV files with columns: `seq_id`, `problem_num`, `action_unit`, `timestamp`, `outcome`
- Action unit vocabulary per item

## Files

| File / directory | Description |
|------------------|-------------|
| `01_preprocessing_notebook.ipynb` | Legacy notebook preserving the original item-by-item preprocessing workflow |
| `01_preprocess_rules.py` | Reusable rule-based preprocessing utilities and final sequence builders |
| `02_llm_clean_descriptions.py` | Script for LLM-assisted description normalization |
| `openrouter_client.py` | OpenRouter API wrapper; reads `OPENROUTER_API_KEY` from the environment |
| `prompts.py` | Prompt text for description normalization and verification |
| `mapping_tables/` | Existing LLM mapping outputs used in the final action-unit construction step |

## LLM Preprocessing

- Model: Claude Sonnet 4.5 (Anthropic, 2025) via OpenRouter API
- Full prompt text: see `prompts.py`
- Constraint: event types processed independently; human review of all outputs
- API key: set `OPENROUTER_API_KEY`; do not hard-code API keys

## Usage

```bash
# 1. Prepare second-pass event-description dictionaries after first-pass pickles exist
python 01_preprocess_rules.py prepare-second-pass \
  --first-pass-dir input_data/1st_data \
  --second-pass-dir input_data/2nd_data \
  --token-units-dir input_data/token_units

# 2. Optionally regenerate LLM mapping tables
export OPENROUTER_API_KEY="..."
python 02_llm_clean_descriptions.py \
  --token-units-dir input_data/token_units \
  --output-dir mapping_tables

# 3. Build final action sequences using mapping tables
python 01_preprocess_rules.py build-sequences \
  --second-pass-dir input_data/2nd_data \
  --mapping-dir mapping_tables \
  --third-pass-dir input_data/3rd_data \
  --irt-output-dir ../../model_input/IRT \
  --embedding-output-dir ../../model_input/HW2V
```

Raw PIAAC log files and generated intermediate data are not redistributed in this repository.
