"""
Vocabulary Building Module
Indexing and vocabulary management for both tokens and units.
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple


class VocabularyBuilder:
    """
    Class for building and managing vocabulary indices for action tokens and action units.

    Attributes:
        token_to_idx (Dict[str, int]): Mapping from token to index.
        idx_to_token (Dict[int, str]): Mapping from index to token.
        unit_to_idx (Dict[str, int]):  Mapping from unit to index.
        idx_to_unit (Dict[int, str]):  Mapping from index to unit.
        token_counts (Counter):        Frequency counts for each token.
        unit_counts (Counter):         Frequency counts for each unit.
        vocab_size (int):              Token vocabulary size.
        unit_vocab_size (int):         Unit vocabulary size.
    """

    def __init__(self, min_count: int = 1):
        """
        Initialize the vocabulary builder.

        Parameters:
            min_count (int): Minimum frequency threshold for vocabulary inclusion (default: 1).
        """
        self.min_count    = min_count

        self.token_to_idx = {}
        self.idx_to_token = {}
        self.token_counts = Counter()
        self.vocab_size   = 0

        self.unit_to_idx     = {}
        self.idx_to_unit     = {}
        self.unit_counts     = Counter()
        self.unit_vocab_size = 0

    def build_vocabulary(self, token_counter: Counter,
                         action_units: List[List[str]]) -> None:
        """
        Build the vocabulary from a token counter and action units.

        Parameters:
            token_counter (Counter):        Token counter produced by the data loader.
            action_units (List[List[str]]): Action units for all sequences.

        Note:
            Applies min_count filtering to both tokens and units.
        """
        self._build_token_vocabulary(token_counter)
        self._build_unit_vocabulary(action_units)

        print(f"Vocabulary built:")
        print(f"  Token vocab size: {self.vocab_size}")
        print(f"  Unit vocab size:  {self.unit_vocab_size}")

    def _build_token_vocabulary(self, token_counter: Counter) -> None:
        """Build the token vocabulary."""
        filtered_tokens = [token for token, count in token_counter.items()
                           if count >= self.min_count]

        sorted_tokens = sorted(filtered_tokens,
                               key=lambda x: token_counter[x],
                               reverse=True)

        for idx, token in enumerate(sorted_tokens):
            self.token_to_idx[token] = idx
            self.idx_to_token[idx]   = token
            self.token_counts[token] = token_counter[token]

        self.vocab_size = len(self.token_to_idx)

    def _build_unit_vocabulary(self, action_units: List[List[str]]) -> None:
        """Build the unit vocabulary."""
        unit_counter = Counter()
        for units_list in action_units:
            unit_counter.update(units_list)

        filtered_units = [unit for unit, count in unit_counter.items()
                          if count >= self.min_count]

        sorted_units = sorted(filtered_units,
                              key=lambda x: unit_counter[x],
                              reverse=True)

        for idx, unit in enumerate(sorted_units):
            self.unit_to_idx[unit] = idx
            self.idx_to_unit[idx]  = unit
            self.unit_counts[unit] = unit_counter[unit]

        self.unit_vocab_size = len(self.unit_to_idx)

    def get_token_index(self, token: str) -> Optional[int]:
        """Return the index of the given token."""
        return self.token_to_idx.get(token)

    def get_token_by_index(self, idx: int) -> Optional[str]:
        """Return the token corresponding to the given index."""
        return self.idx_to_token.get(idx)

    def get_unit_index(self, unit: str) -> Optional[int]:
        """Return the index of the given unit."""
        return self.unit_to_idx.get(unit)

    def get_unit_by_index(self, idx: int) -> Optional[str]:
        """Return the unit corresponding to the given index."""
        return self.idx_to_unit.get(idx)

    def get_unit_composition(self, unit: str) -> List[Tuple[str, int]]:
        """
        Return the composition indices for a unit in the FastText style.

        Parameters:
            unit (str): Unit name (e.g., 'button_next').

        Returns:
            List[Tuple[str, int]]: List of (type, index) tuples where type is 'token' or 'unit'.

        Note:
            FastText approach: unit_embedding = sum([token1, token2, ..., unit])
            Example: 'button_next' → [('token', button_idx), ('token', next_idx), ('unit', button_next_idx)]
        """
        composition_indices = []

        if '_' in unit:
            tokens = unit.split('_')
            for token in tokens:
                token_idx = self.get_token_index(token)
                if token_idx is not None:
                    composition_indices.append(('token', token_idx))

        unit_idx = self.get_unit_index(unit)
        if unit_idx is not None:
            composition_indices.append(('unit', unit_idx))

        return composition_indices

        # # 1. Add indices of component tokens
        # #    (modified to disambiguate single-token units such as 'start' using the unit index)
        # tokens = unit.split('_')
        # for token in tokens:
        #     token_idx = self.get_token_index(token)
        #     if token_idx is not None:
        #         composition_indices.append(('token', token_idx))

        # # 2. Add the unit's own index
        # unit_idx = self.get_unit_index(unit)
        # if unit_idx is not None:
        #     composition_indices.append(('unit', unit_idx))

        # return composition_indices

    def is_valid_item(self, item: str) -> bool:
        """
        Check whether an item (token or unit) is present in the vocabulary.

        Parameters:
            item (str): Item to check.

        Returns:
            bool: True if the item is in the vocabulary.
        """
        if self.get_token_index(item) is not None:
            return True

        if self.get_unit_index(item) is not None:
            if '_' in item:
                tokens = item.split('_')
                return all(self.get_token_index(token) is not None for token in tokens)
            return True

        return False

        # # Check if token (no underscore means token)
        # if '_' not in item:
        #     return self.get_token_index(item) is not None
        # else:
        #     # Unit: both the unit itself and all its component tokens must be in the vocabulary
        #     unit_idx = self.get_unit_index(item)
        #     if unit_idx is None:
        #         return False
        #     tokens = item.split('_')
        #     return all(self.get_token_index(token) is not None for token in tokens)

    def filter_training_pairs(self, training_pairs: List[tuple]) -> List[tuple]:
        """
        Filter out training pairs containing items absent from the vocabulary.

        Parameters:
            training_pairs (List[tuple]): List of (center_item, context_unit) pairs.

        Returns:
            List[tuple]: Filtered training pairs.
        """
        filtered_pairs = []
        original_count = len(training_pairs)

        for center_item, context_unit in training_pairs:
            if self.is_valid_item(center_item) and self.is_valid_item(context_unit):
                filtered_pairs.append((center_item, context_unit))

        filtered_count = len(filtered_pairs)
        removed_count  = original_count - filtered_count

        token_unit_pairs = 0
        unit_unit_pairs  = 0

        for center, _ in filtered_pairs:
            if center in self.token_to_idx and center not in self.unit_to_idx:
                token_unit_pairs += 1
            elif center in self.unit_to_idx:
                unit_unit_pairs += 1

        # token_unit_pairs = sum(1 for center, _ in filtered_pairs if '_' not in center)
        # unit_unit_pairs  = sum(1 for center, _ in filtered_pairs if '_' in center)

        print(f"FastText-style training pairs filtered:")
        print(f"  Original pairs:  {original_count:,}")
        print(f"  Retained pairs:  {filtered_count:,}")
        print(f"    - Token-Unit:  {token_unit_pairs:,}")
        print(f"    - Unit-Unit:   {unit_unit_pairs:,}")
        print(f"  Removed pairs:   {removed_count:,}")

        return filtered_pairs

    def get_negative_sampling_distribution(self) -> np.ndarray:
        """
        Generate the negative sampling probability distribution over units.

        Returns:
            np.ndarray: Negative sampling probability distribution over all units.

        Note:
            In Token-Unit training, units are used as context items,
            so the distribution is defined over units.
        """
        if self.unit_vocab_size == 0:
            raise ValueError("Unit vocabulary has not been built.")

        frequencies = np.zeros(self.unit_vocab_size)
        for idx in range(self.unit_vocab_size):
            unit            = self.idx_to_unit[idx]
            frequencies[idx] = self.unit_counts[unit]

        # Apply 3/4 power (standard for negative sampling)
        powered_frequencies = np.power(frequencies, 0.75)
        distribution        = powered_frequencies / np.sum(powered_frequencies)

        return distribution

    def get_vocabulary_info(self) -> Dict:
        """Return vocabulary information as a dictionary."""
        if self.vocab_size == 0 or self.unit_vocab_size == 0:
            return {"error": "Vocabulary has not been built."}

        token_total_count = sum(self.token_counts.values())
        unit_total_count  = sum(self.unit_counts.values())

        return {
            "token_vocab_size"       : self.vocab_size,
            "unit_vocab_size"        : self.unit_vocab_size,
            "min_count"              : self.min_count,
            "total_token_count"      : token_total_count,
            "total_unit_count"       : unit_total_count,
            "most_common_tokens"     : self.token_counts.most_common(5),
            "most_common_units"      : self.unit_counts.most_common(5),
            "average_token_frequency": token_total_count / self.vocab_size,
            "average_unit_frequency" : unit_total_count  / self.unit_vocab_size
        }

    def save_vocabulary(self, file_path: str) -> None:
        """Save the built vocabulary to a file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Token vocab size: {self.vocab_size}\n")
            f.write(f"# Unit vocab size:  {self.unit_vocab_size}\n")
            f.write(f"# Min count:        {self.min_count}\n")
            f.write("\n=== Token Vocabulary ===\n")
            f.write("# Format: index\ttoken\tfrequency\n")

            for idx in range(self.vocab_size):
                token = self.idx_to_token[idx]
                count = self.token_counts[token]
                f.write(f"{idx}\t{token}\t{count}\n")

            f.write("\n=== Unit Vocabulary ===\n")
            f.write("# Format: index\tunit\tfrequency\n")

            for idx in range(self.unit_vocab_size):
                unit  = self.idx_to_unit[idx]
                count = self.unit_counts[unit]
                f.write(f"{idx}\t{unit}\t{count}\n")

        print(f"Vocabulary saved: {file_path}")