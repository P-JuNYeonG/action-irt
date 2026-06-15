"""Reusable preprocessing utilities for PIAAC PSTRE action logs.

This script is a cleaned, repository-facing extraction of the legacy
notebook `01_preprocessing_notebook.ipynb`. The notebook preserves the
original item-by-item workflow; this module provides reusable functions for
the common preprocessing operations.
"""

from __future__ import annotations

import argparse
import pickle
import re
from collections import defaultdict
from itertools import product
from pathlib import Path

import pandas as pd


ITEMS = [f"ps{i}_{j}" for i, j in product(range(1, 3), range(1, 8))]

MAINTAIN_ACTION_TYPES = {
    "mail_drag",
    "mail_drop",
    "folder_viewed",
    "mail_viewed",
    "menuitem",
    "get_help",
    "restart",
    "folder_unfolded",
    "folder_folded",
    "radio_btn",
    "breakoff",
    "shortcut",
    "checkbox",
    "tab",
    "success",
    "failure",
}

LLM_ACTION_TYPES = {
    "toolbar",
    "menu",
    "button",
    "textbox_onfocus",
    "textbox_killfocus",
    "combobox",
    "textlink",
}

DEFAULT_REMOVE_EVENTS = {
    "DOACTION",
    "NEXT_INQUIRY",
    "NEXT_BUTTON",
    "CONFIRMATION_OPENED",
    "CONFIRMATION_CLOSED",
}

REMOVE_VARS_BY_ITEM = {
    "ps1_1": {"test_time", "end", "value"},
    "ps1_2": {"test_time", "end", "value"},
    "ps1_3": {"test_time", "end", "value"},
    "ps1_4": {"test_time", "end", "value"},
    "ps1_5": {"test_time", "end", "value"},
    "ps1_6": {"test_time", "end", "value", "href"},
    "ps1_7": {"test_time", "end", "value"},
    "ps2_1": {"test_time", "end", "value"},
    "ps2_2": {"test_time", "end", "value"},
    "ps2_3": {"test_time", "end", "value", "href"},
    "ps2_4": {"test_time", "end", "value", "href"},
    "ps2_5": {"test_time", "end", "value"},
    "ps2_6": {"test_time", "end", "value"},
    "ps2_7": {"test_time", "end", "value", "href"},
}


def fix_restart_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Adjust timestamps after RESTART events within each respondent sequence."""
    adjusted_groups = []
    for _, group in df.groupby("SEQID", sort=False):
        group = group.copy()
        adjusted_timestamps = group["timestamp"].copy()
        restart_positions = [
            pos for pos, value in enumerate(group["event_type"].tolist()) if value == "RESTART"
        ]

        for restart_pos in restart_positions:
            if restart_pos == 0 or restart_pos >= len(group) - 1:
                continue

            prev_timestamp = adjusted_timestamps.iloc[restart_pos - 1]
            next_original_timestamp = group["timestamp"].iloc[restart_pos + 1]
            restart_timestamp = prev_timestamp + round(next_original_timestamp / 2)
            adjusted_timestamps.iloc[restart_pos] = restart_timestamp

            end_pos = len(group) - 1
            for pos in range(restart_pos + 1, len(group)):
                if group["event_type"].iloc[pos] == "END":
                    end_pos = pos
                    break

            for pos in range(restart_pos + 1, end_pos + 1):
                adjusted_timestamps.iloc[pos] = group["timestamp"].iloc[pos] + restart_timestamp

        group["timestamp"] = adjusted_timestamps
        adjusted_groups.append(group)

    return pd.concat(adjusted_groups, ignore_index=True)


def process_keypress_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse consecutive KEYPRESS events for a single sequence."""
    rows = []
    i = 0
    while i < len(df):
        current_row = df.iloc[i].copy()
        if current_row["event_type"] != "KEYPRESS":
            rows.append(current_row)
            i += 1
            continue

        count = 1
        j = i + 1
        while j < len(df) and df.iloc[j]["event_type"] == "KEYPRESS":
            count += 1
            j += 1

        current_row["event_description"] = f"count={count}"
        rows.append(current_row)
        i = j

    return pd.DataFrame(rows)


def preprocess_keypress_data(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse consecutive KEYPRESS events sequence by sequence."""
    processed = []
    for _, seq_data in df.groupby("SEQID", sort=False):
        processed.append(process_keypress_sequences(seq_data.reset_index(drop=True)))
    return pd.concat(processed, ignore_index=True)


def normalize_basic_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cross-item button label normalization used in the legacy workflow."""
    df = df.copy()
    df["event_description"] = df["event_description"].replace(
        {
            "id=endunit_txt1": "id=next",
            "id=endunit_txt2": "id=next",
            "id=endunit_txt3": "id=ok",
            "id=endunit_txt4": "id=cancel",
        }
    )
    return df


def first_pass_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apply common first-pass preprocessing rules."""
    data = normalize_basic_descriptions(df)
    data = fix_restart_timestamps(data)
    data = data[~data["event_type"].isin(DEFAULT_REMOVE_EVENTS)].copy()
    data = data.drop_duplicates()
    data = preprocess_keypress_data(data)
    return data.reset_index(drop=True)


def process_event_description(description: str, item: str) -> str:
    """Remove system-only variables from an event description."""
    components = re.split(r"[|*$]", str(description))
    components = [component.strip() for component in components if component.strip()]
    remove_vars = REMOVE_VARS_BY_ITEM[item]
    remove_details = {"nan", ",", "."}

    filtered = []
    for component in components:
        if "=" in component or component == "end":
            var_name = component.split("=")[0].strip()
            if var_name not in remove_vars:
                filtered.append(component)
        elif not any(detail in component for detail in remove_details):
            filtered.append(component)

    return "|".join(filtered)


def create_event_type_dict(df: pd.DataFrame) -> tuple[dict, dict, dict, dict]:
    """Create unit/token dictionaries and count dictionaries by event type."""
    event_dict = defaultdict(set)
    event_count_dict = defaultdict(lambda: defaultdict(int))
    token_dict = defaultdict(set)
    token_count_dict = defaultdict(lambda: defaultdict(int))

    for _, row in df.iterrows():
        event_type = row["event_type"]
        event_desc = row["event_desc_list"]
        event_dict[event_type].add(event_desc)
        event_count_dict[event_type][event_desc] += 1

        for token in event_desc.split("|"):
            token_dict[event_type].add(token)
            token_count_dict[event_type][token] += 1

    unit_dict = {key: list(value) for key, value in event_dict.items()}
    unit_count_dict = {key: dict(value) for key, value in event_count_dict.items()}
    token_dict = {key: list(value) for key, value in token_dict.items()}
    token_count_dict = {key: dict(value) for key, value in token_count_dict.items()}
    return unit_dict, unit_count_dict, token_dict, token_count_dict


def second_pass_prepare(
    first_pass_dir: Path,
    second_pass_dir: Path,
    token_units_dir: Path,
) -> None:
    """Create second-pass files and LLM input dictionaries for all items."""
    second_pass_dir.mkdir(parents=True, exist_ok=True)
    token_units_dir.mkdir(parents=True, exist_ok=True)

    for item in ITEMS:
        data = pd.read_pickle(first_pass_dir / f"us_{item}.pkl")
        data = data.copy()
        data["event_type"] = data["event_type"].str.lower()
        data["event_description"] = data["event_description"].str.lower()
        data["event_desc_list"] = data["event_description"].apply(
            lambda desc: process_event_description(desc, item)
        )

        unit_dict, unit_count, token_dict, token_count = create_event_type_dict(data)
        data.to_pickle(second_pass_dir / f"us_{item}.pkl")

        for prefix, obj in {
            "us_unit": unit_dict,
            "us_unit_cnt": unit_count,
            "us_token": token_dict,
            "us_token_cnt": token_count,
        }.items():
            with (token_units_dir / f"{prefix}_{item}.pkl").open("wb") as handle:
                pickle.dump(obj, handle)


def parse_text_table(text_path: Path) -> pd.DataFrame:
    """Parse a markdown table with event_type, description, and substitute columns."""
    content = text_path.read_text(encoding="utf-8")
    table_lines = [line.strip() for line in content.split("\n") if "|" in line]
    if len(table_lines) < 2:
        raise ValueError(f"No markdown table found in {text_path}")

    headers = [column.strip() for column in table_lines[0].split("|") if column.strip()]
    rows = []
    for line in table_lines[2:]:
        temp = re.sub(r"\\\|", "[[PIPE]]", line)
        parts = [part.strip() for part in temp.split("|") if part.strip()]
        row = [part.replace("[[PIPE]]", "|").replace("\\", "") for part in parts]
        if len(row) == len(headers):
            rows.append(row)

    return pd.DataFrame(rows, columns=headers)


def transform_mapping_dict(mapping_df: pd.DataFrame) -> dict[tuple[str, str], str]:
    return {
        (row["event_type"], row["description"]): row["substitute"]
        for _, row in mapping_df.iterrows()
    }


def parse_description(desc: str) -> dict[str, str]:
    if pd.isna(desc) or desc == "":
        return {}

    result = {}
    for pair in desc.split("|"):
        if "=" in pair:
            key, value = pair.split("=", 1)
            result[key] = value
    return result


def transform_event(row: pd.Series, llm_mapping: dict[tuple[str, str], str]) -> str:
    event_type = row["event_type"]
    event_desc = row["event_desc_list"]
    event_type_converted = event_type.replace("_", "-")

    if event_type in MAINTAIN_ACTION_TYPES:
        parsed = parse_description(event_desc)
        detailed_info = []
        for value in parsed.values():
            converted_value = re.sub(r"(u\d+[A-Za-z0-9]*)(_|$)", r"\1-", value)
            detailed_info.append(converted_value)
        return (
            f"{event_type_converted}_{'_'.join(detailed_info)}"
            if detailed_info
            else event_type_converted
        )

    if event_type == "keypress":
        parsed = parse_description(event_desc)
        count = int(parsed.get("count", 0))
        count_str = "10+" if count > 10 else str(count)
        return f"keypress{count_str}"

    if event_type == "cell_change":
        parsed = parse_description(event_desc)
        detailed_info = list(parsed.values())[0] if parsed else ""
        converted_info = detailed_info.replace(
            "content_spreadsheet_colad", "content-spreadsheet-colad"
        )
        return f"cell-change_{converted_info}"

    if event_type in LLM_ACTION_TYPES:
        key = (event_type, event_desc)
        substitute = llm_mapping.get(key, event_desc)
        return f"{event_type_converted}_{substitute}"

    return f"{event_type_converted}_{event_desc}" if event_desc else event_type_converted


def third_pass_build_sequences(
    second_pass_dir: Path,
    mapping_dir: Path,
    third_pass_dir: Path,
    irt_output_dir: Path,
    embedding_output_dir: Path,
) -> None:
    """Create final action units and sequence files for IRT and embedding."""
    third_pass_dir.mkdir(parents=True, exist_ok=True)
    irt_output_dir.mkdir(parents=True, exist_ok=True)
    embedding_output_dir.mkdir(parents=True, exist_ok=True)

    for item in ITEMS:
        data = pd.read_pickle(second_pass_dir / f"us_{item}.pkl")
        mapping_df = parse_text_table(mapping_dir / item / "step1_llm.txt")
        llm_mapping = transform_mapping_dict(mapping_df)

        data = data.copy()
        data["processed_event"] = data.apply(
            lambda row: transform_event(row, llm_mapping), axis=1
        )
        data = data[~data["processed_event"].isin({"start", "button_next", "button_ok", "end"})]
        data["processed_event"] = data["processed_event"].apply(
            lambda value: "_".join(dict.fromkeys(value.split("_")))
        )

        result = data[["SEQID", "event_type", "event_desc_list", "timestamp", "processed_event"]]
        result.to_pickle(third_pass_dir / f"us_{item}.pkl")

        sequence_rows = []
        for seqid, group in result.groupby("SEQID", sort=False):
            events = group.sort_values("timestamp")["processed_event"].tolist()
            sequence_rows.append(
                {
                    "SEQID": seqid,
                    "seq_event": " ".join(events),
                    "event_count": len(events),
                }
            )

        sequence_df = pd.DataFrame(sequence_rows)
        sequence_df.to_csv(irt_output_dir / f"test_us_{item}.csv", index=False)

        with (embedding_output_dir / f"test_us_{item}.txt").open("w", encoding="utf-8") as handle:
            for sequence in sequence_df["seq_event"].dropna():
                handle.write(f"{sequence}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    second = subparsers.add_parser("prepare-second-pass")
    second.add_argument("--first-pass-dir", type=Path, required=True)
    second.add_argument("--second-pass-dir", type=Path, required=True)
    second.add_argument("--token-units-dir", type=Path, required=True)

    third = subparsers.add_parser("build-sequences")
    third.add_argument("--second-pass-dir", type=Path, required=True)
    third.add_argument("--mapping-dir", type=Path, required=True)
    third.add_argument("--third-pass-dir", type=Path, required=True)
    third.add_argument("--irt-output-dir", type=Path, required=True)
    third.add_argument("--embedding-output-dir", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "prepare-second-pass":
        second_pass_prepare(args.first_pass_dir, args.second_pass_dir, args.token_units_dir)
    elif args.command == "build-sequences":
        third_pass_build_sequences(
            args.second_pass_dir,
            args.mapping_dir,
            args.third_pass_dir,
            args.irt_output_dir,
            args.embedding_output_dir,
        )


if __name__ == "__main__":
    main()
