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

## LLM Preprocessing

- Model: Claude Sonnet 4.5 (Anthropic, 2025) via OpenRouter API
- Full prompt text: see `supplementary/llm_prompts/`
- Constraint: event types processed independently; human review of all outputs

## Usage

```bash
# Detailed scripts to be added
Rscript 01_preprocess_rules.R      # 1st and 3rd pass
python  02_llm_cleaning.py         # 2nd pass (requires API key)
```
