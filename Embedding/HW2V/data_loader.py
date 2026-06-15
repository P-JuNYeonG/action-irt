"""
Data Loader Module
[Action sequence] -> [Action unit] -> [Action token] splitting
Token-Unit training pair generation
Unit-Unit training pair generation
"""

import os
from typing import List, Tuple
from collections import Counter


class ActionDataLoader:
    """
    Class for loading action sequence data and generating token-unit training pairs.

    Attributes:
        sequences (List[str]): Raw action sequences.
        action_units (List[List[str]]): Action units for each sequence.
        action_tokens (List[List[str]]): Action tokens for each sequence.
        token_counter (Counter): Frequency counts of all tokens.

    Example:
        sequences : [
        "start toolbar_menu toolbar_ss next button_end",
        "click file_dialog select_option confirm_action close",
        "open menu_edit copy_text paste_text save_file"]

        action_units : [
        ['start', 'toolbar_menu', 'toolbar_ss', 'next', 'button_end'],
        ['click', 'file_dialog', 'select_option', 'confirm_action', 'close'],
        ['open', 'menu_edit', 'copy_text', 'paste_text', 'save_file']]

        action_tokens : [
        ['toolbar', 'menu', 'toolbar', 'ss', 'button', 'end'],
        ['file', 'dialog', 'select', 'option', 'confirm', 'action'],
        ['menu', 'edit', 'copy', 'text', 'paste', 'text', 'save', 'file']]

        token_counter = Counter({})
    """

    def __init__(self):
        """Initialize the data loader."""
        self.sequences     = []
        self.action_units  = []
        self.action_tokens = []
        self.token_counter = Counter()

    def load_from_file(self, file_path: str) -> None:
        """
        Load action sequence data from a text file.

        Parameters:
            file_path (str): Path to the input text file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.sequences.append(line)

        print(f"Loaded {len(self.sequences)} action sequences.")

    def tokenize_sequences(self) -> None:
        """
        Split loaded sequences into action units and action tokens.

        Processing steps:
            1. Split each sequence by whitespace to extract action units.
            2. Split each action unit by '_' to extract tokens; skip single-token units.
            3. Count token frequencies.
        """
        self.action_units  = []
        self.action_tokens = []

        for sequence in self.sequences:
            units = sequence.split()
            self.action_units.append(units)

            sequence_tokens = []
            for unit in units:
                tokens = unit.split('_')
                if len(tokens) == 1:
                    continue
                sequence_tokens.extend(tokens)
                self.token_counter.update(tokens)

            self.action_tokens.append(sequence_tokens)

        print(f"Found {len(set(self.token_counter.keys()))} unique action tokens.")
        print(f"Top 5 most frequent tokens: {self.token_counter.most_common(5)}")

    def get_training_pairs(self, window_size: int = 1) -> List[Tuple[str, str]]:
        """
        Generate Hybrid-style Token-Unit and Unit-Unit training pairs.

        Parameters:
            window_size (int): Context window size in units (default: 1).

        Returns:
            List[Tuple[str, str]]: List of (center_item, context_unit) pairs.

        Note:
            Similar to FastText, both tokens and units are used as center items.
            Example: 'start button_next end'
            - Token-Unit pairs: ('button', 'start'), ('button', 'end'),
                                ('next', 'start'), ('next', 'end')
            - Unit-Unit pairs:  ('button_next', 'start'), ('button_next', 'end')
        """
        training_pairs   = []
        token_unit_pairs = 0
        unit_unit_pairs  = 0

        for _, units in enumerate(self.action_units):
            # 1) Generate Token-Unit pairs
            unit_positions = {}  # token -> [unit_indices]

            for unit_idx, unit in enumerate(units):
                tokens = unit.split('_')
                if len(tokens) == 1:
                    continue
                for token in tokens:
                    if token not in unit_positions:
                        unit_positions[token] = []
                    unit_positions[token].append(unit_idx)

            for token, token_unit_indices in unit_positions.items():
                for token_unit_idx in token_unit_indices:
                    start = max(0, token_unit_idx - window_size)
                    end   = min(len(units), token_unit_idx + window_size + 1)

                    for context_unit_idx in range(start, end):
                        if token_unit_idx != context_unit_idx:  # Exclude self
                            context_unit = units[context_unit_idx]
                            training_pairs.append((token, context_unit))
                            token_unit_pairs += 1

            # 2) Generate Unit-Unit pairs
            for unit_idx, unit in enumerate(units):
                start = max(0, unit_idx - window_size)
                end   = min(len(units), unit_idx + window_size + 1)

                for context_unit_idx in range(start, end):
                    if unit_idx != context_unit_idx:  # Exclude self
                        context_unit = units[context_unit_idx]
                        training_pairs.append((unit, context_unit))
                        unit_unit_pairs += 1

        print(f"FastText-style training pairs generated:")
        print(f"  Token-Unit pairs: {token_unit_pairs:,}")
        print(f"  Unit-Unit pairs:  {unit_unit_pairs:,}")
        print(f"  Total pairs:      {len(training_pairs):,}")
        return training_pairs

    def get_vocabulary_stats(self) -> dict:
        """
        Return vocabulary statistics.

        Returns:
            dict: Vocabulary statistics.
        """
        all_units = set()
        for units_list in self.action_units:
            all_units.update(units_list)

        return {
            'total_sequences'       : len(self.sequences),
            'total_tokens'          : sum(self.token_counter.values()),
            'unique_tokens'         : len(self.token_counter),
            'unique_units'          : len(all_units),
            'average_sequence_length': (
                sum(len(units) for units in self.action_units) / len(self.action_units)
                if self.action_units else 0
            )
        }

    def split_sequences(self, train_ratio: float = 0.8,
                        random_seed: int = 42) -> Tuple[List[str], List[str]]:
        """
        Randomly split sequences into train and validation sets.

        Parameters:
            train_ratio (float): Fraction of sequences for training (default: 0.8).
            random_seed (int):   Random seed for reproducibility.

        Returns:
            Tuple[List[str], List[str]]: (train_sequences, valid_sequences).

        Note:
            Shuffles self.sequences before splitting.
        """
        import random

        if not self.sequences:
            raise ValueError("No sequences loaded. Call load_from_file() first.")

        # 1. Generate and shuffle indices
        num_sequences = len(self.sequences)
        indices = list(range(num_sequences))
        random.seed(random_seed)
        random.shuffle(indices)

        # 2. Compute split point
        split_point   = int(num_sequences * train_ratio)
        train_indices = indices[:split_point]
        valid_indices = indices[split_point:]

        train_sequences = [self.sequences[i] for i in train_indices]
        valid_sequences = [self.sequences[i] for i in valid_indices]

        print(f"Data split complete:")
        print(f"  Train sequences: {len(train_sequences):,} ({train_ratio*100:.1f}%)")
        print(f"  Valid sequences: {len(valid_sequences):,} ({(1-train_ratio)*100:.1f}%)")

        return train_sequences, valid_sequences

    def get_training_pairs_from_sequences(self, sequences: List[str],
                                          window_size: int = 1) -> List[Tuple[str, str]]:
        """
        Generate training pairs from a given list of sequences.
        (Same logic as get_training_pairs(), applied to a specified subset of sequences.)

        Parameters:
            sequences (List[str]): Sequences from which to generate pairs.
            window_size (int):     Context window size in units (default: 1).

        Returns:
            List[Tuple[str, str]]: List of (center_item, context_unit) pairs.

        Note:
            Generates FastText-style pairs (Token-Unit + Unit-Unit).
            Intended to be called separately for train and validation sets.
        """
        training_pairs   = []
        token_unit_pairs = 0
        unit_unit_pairs  = 0

        action_units_list = [seq.split() for seq in sequences]

        for _, units in enumerate(action_units_list):
            # 1) Generate Token-Unit pairs
            unit_positions = {}  # token -> [unit_indices]

            for unit_idx, unit in enumerate(units):
                tokens = unit.split('_')
                if len(tokens) == 1:
                    continue
                for token in tokens:
                    if token not in unit_positions:
                        unit_positions[token] = []
                    unit_positions[token].append(unit_idx)

            for token, token_unit_indices in unit_positions.items():
                for token_unit_idx in token_unit_indices:
                    start = max(0, token_unit_idx - window_size)
                    end   = min(len(units), token_unit_idx + window_size + 1)

                    for context_unit_idx in range(start, end):
                        if token_unit_idx != context_unit_idx:  # Exclude self
                            context_unit = units[context_unit_idx]
                            training_pairs.append((token, context_unit))
                            token_unit_pairs += 1

            # 2) Generate Unit-Unit pairs
            for unit_idx, unit in enumerate(units):
                start = max(0, unit_idx - window_size)
                end   = min(len(units), unit_idx + window_size + 1)

                for context_unit_idx in range(start, end):
                    if unit_idx != context_unit_idx:  # Exclude self
                        context_unit = units[context_unit_idx]
                        training_pairs.append((unit, context_unit))
                        unit_unit_pairs += 1

        print(f"FastText-style training pairs generated:")
        print(f"  Token-Unit pairs: {token_unit_pairs:,}")
        print(f"  Unit-Unit pairs:  {unit_unit_pairs:,}")
        print(f"  Total pairs:      {len(training_pairs):,}")
        return training_pairs