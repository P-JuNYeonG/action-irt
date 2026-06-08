"""
Export LSTM-AE latent matrices to long-format CSV for Action-IRT.

Input
-----
Pickle containing {(seq_id, problem_num): Nij x D numpy array}.

Output
------
Long-format CSV with one row per action occurrence:
seq_id, problem_num, behavior_id, action_name, outcome, N_ij, C_value1, ...
"""

import argparse
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_PROBLEM_MAPPING = {
    "U01a000S": "ps1_1",
    "U01b000S": "ps1_2",
    "U03a000S": "ps1_3",
    "U06a000S": "ps1_4",
    "U06b000S": "ps1_5",
    "U21x000S": "ps1_6",
    "U04a000S": "ps1_7",
    "U19a000S": "ps2_1",
    "U19b000S": "ps2_2",
    "U07x000S": "ps2_3",
    "U02x000S": "ps2_4",
    "U16x000S": "ps2_5",
    "U11b000S": "ps2_6",
    "U23x000S": "ps2_7",
}

POLYTOMOUS_COLUMNS = {"U01a000S", "U04a000S", "U19b000S", "U02x000S", "U11b000S", "U23x000S"}


def load_problem_mapping(mapping_path: Optional[str]) -> Dict[str, str]:
    if mapping_path is None:
        return DEFAULT_PROBLEM_MAPPING

    mapping = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"Invalid mapping line: {line}")
            csv_col, problem_num = line.split("=", 1)
            mapping[csv_col.strip()] = problem_num.strip()
    return mapping


def load_outcome_data(
    csv_path: Optional[str],
    column_mapping: Dict[str, str],
    seq_id_column: str = "SEQID",
    seq_id_prefix: str = "US_",
) -> Dict[Tuple[str, str], float]:
    if csv_path is None:
        return {}

    df = pd.read_csv(csv_path, dtype={seq_id_column: str})
    outcome_dict = {}

    for _, row in df.iterrows():
        seq_id = f"{seq_id_prefix}{str(row[seq_id_column])}"
        for csv_col, problem_num in column_mapping.items():
            if csv_col not in row:
                continue
            value = row[csv_col]
            if pd.isna(value) or str(value).strip() == "":
                continue
            try:
                if csv_col in POLYTOMOUS_COLUMNS:
                    outcome = 1 if float(value) >= 2 else 0
                else:
                    outcome = 1 if float(value) == 1 else 0
                outcome_dict[(seq_id, problem_num)] = outcome
            except ValueError:
                continue

    print(f"Outcome loaded: {len(outcome_dict):,} respondent-item pairs")
    return outcome_dict


def load_action_names(
    seq_event_dir: Optional[str],
    target_problem: str,
    seq_id_column: str = "SEQID",
    seq_event_column: str = "seq_event",
    seq_id_prefix: str = "",
) -> Dict[Tuple[str, str], List[str]]:
    if seq_event_dir is None:
        return {}

    csv_path = os.path.join(seq_event_dir, f"test_us_{target_problem}.csv")
    if not os.path.exists(csv_path):
        print(f"[warning] action sequence file not found: {csv_path}")
        return {}

    df = pd.read_csv(csv_path, dtype={seq_id_column: str})
    action_name_dict = {}
    for _, row in df.iterrows():
        seq_id = f"{seq_id_prefix}{str(row[seq_id_column])}"
        seq_event = str(row[seq_event_column]).strip()
        action_name_dict[(seq_id, target_problem)] = seq_event.split()

    print(f"Action names loaded for {target_problem}: {len(action_name_dict):,}")
    return action_name_dict


def load_unique_action_mapping(unique_action_dir: Optional[str], target_problem: str) -> Dict[str, int]:
    if unique_action_dir is None:
        return {}

    filepath = os.path.join(unique_action_dir, f"us_{target_problem}.txt")
    if not os.path.exists(filepath):
        print(f"[warning] unique action mapping not found: {filepath}")
        return {}

    mapping = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(". ", 1)
            if len(parts) != 2:
                continue
            try:
                mapping[parts[1].strip()] = int(parts[0])
            except ValueError:
                continue

    print(f"Unique action mapping loaded for {target_problem}: {len(mapping):,}")
    return mapping


def robust_scale_by_item(latent_data: Dict[Tuple[str, str], np.ndarray]) -> Dict[Tuple[str, str], np.ndarray]:
    scaled = {}
    problems = sorted({problem_num for _, problem_num in latent_data.keys()})

    for problem_num in problems:
        keys = [key for key in latent_data if key[1] == problem_num]
        stacked = np.vstack([latent_data[key] for key in keys])
        median = np.median(stacked, axis=0)
        q75 = np.percentile(stacked, 75, axis=0)
        q25 = np.percentile(stacked, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr == 0] = 1.0

        for key in keys:
            scaled[key] = (latent_data[key] - median) / iqr

    return scaled


def to_long_format(
    latent_data: Dict[Tuple[str, str], np.ndarray],
    outcome_dict: Dict[Tuple[str, str], float],
    seq_event_dir: Optional[str],
    unique_action_dir: Optional[str],
    action_seq_id_prefix: str = "",
) -> pd.DataFrame:
    rows = []
    problems = sorted({problem_num for _, problem_num in latent_data.keys()})

    action_names_by_problem = {
        problem: load_action_names(seq_event_dir, problem, seq_id_prefix=action_seq_id_prefix)
        for problem in problems
    }
    mappings_by_problem = {
        problem: load_unique_action_mapping(unique_action_dir, problem)
        for problem in problems
    }

    unmapped_actions = set()
    for (seq_id, problem_num), latent_matrix in sorted(latent_data.items()):
        action_name_dict = action_names_by_problem.get(problem_num, {})
        action_mapping = mappings_by_problem.get(problem_num, {})
        action_list = action_name_dict.get((seq_id, problem_num))
        n_steps, latent_dim = latent_matrix.shape

        if action_list is not None and len(action_list) != n_steps:
            print(
                f"[warning] length mismatch for {(seq_id, problem_num)}: "
                f"latent={n_steps}, actions={len(action_list)}"
            )
            action_list = None

        for step in range(n_steps):
            if action_list is None:
                action_name = np.nan
                behavior_id = step + 1
            else:
                action_name = action_list[step]
                behavior_id = action_mapping.get(action_name, step + 1)
                if action_mapping and action_name not in action_mapping:
                    unmapped_actions.add(action_name)

            row = {
                "seq_id": seq_id,
                "problem_num": problem_num,
                "behavior_id": behavior_id,
                "action_name": action_name,
                "outcome": outcome_dict.get((seq_id, problem_num), np.nan),
                "N_ij": n_steps,
            }
            for d in range(latent_dim):
                row[f"C_value{d + 1}"] = latent_matrix[step, d]
            rows.append(row)

    if unmapped_actions:
        preview = sorted(unmapped_actions)[:10]
        suffix = "..." if len(unmapped_actions) > 10 else ""
        print(f"[warning] unmapped actions: {len(unmapped_actions):,} {preview}{suffix}")

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LSTM-AE latent pkl to Action-IRT long CSV.")
    parser.add_argument("--latent-pkl", required=True, help="Reduced latent pickle from train_lstm_ae.py.")
    parser.add_argument("--output-csv", required=True, help="Path to write the long-format CSV.")
    parser.add_argument("--outcome-csv", default=None, help="Wide-format response CSV.")
    parser.add_argument("--problem-map", default=None, help="Optional mapping file with CSV_COLUMN=problem_id lines.")
    parser.add_argument("--seq-event-dir", default=None, help="Directory containing test_us_{problem}.csv files.")
    parser.add_argument("--unique-action-dir", default=None, help="Directory containing us_{problem}.txt mapping files.")
    parser.add_argument("--outcome-seq-prefix", default="US_", help="Prefix added to outcome SEQID values.")
    parser.add_argument("--action-seq-prefix", default="", help="Prefix added to action-sequence SEQID values.")
    parser.add_argument("--no-robust-scale", action="store_true", help="Do not robust-scale latent values by item.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.latent_pkl, "rb") as f:
        latent_data = pickle.load(f)
    if not isinstance(latent_data, dict) or not latent_data:
        raise ValueError("--latent-pkl must contain a non-empty dictionary.")

    if not args.no_robust_scale:
        latent_data = robust_scale_by_item(latent_data)

    problem_mapping = load_problem_mapping(args.problem_map)
    outcome_dict = load_outcome_data(
        csv_path=args.outcome_csv,
        column_mapping=problem_mapping,
        seq_id_prefix=args.outcome_seq_prefix,
    )

    df_long = to_long_format(
        latent_data=latent_data,
        outcome_dict=outcome_dict,
        seq_event_dir=args.seq_event_dir,
        unique_action_dir=args.unique_action_dir,
        action_seq_id_prefix=args.action_seq_prefix,
    )

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df_long.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df_long):,} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
