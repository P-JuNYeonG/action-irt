"""
Extract embedding matrices keyed by (SEQID, problem_num) for multiple item IDs.
Version including timestamp features.
Computes per-item time statistics separately, then rescales time features to the embedding scale before combining.
"""

import argparse
import pickle
from itertools import product
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np

from model import Word2VecModel
from vocab_builder import VocabularyBuilder
from data_loader import ActionDataLoader


def load_model_and_rebuild_vocab(model_path, data_path, min_count=1):
    """Load the model and rebuild the vocabulary builder."""
    print("Loading model...")
    model = Word2VecModel.load_model(model_path)

    print("Rebuilding vocabulary...")
    data_loader = ActionDataLoader()
    data_loader.load_from_file(data_path)
    data_loader.tokenize_sequences()

    vocab_builder = VocabularyBuilder(min_count=min_count)
    vocab_builder.build_vocabulary(data_loader.token_counter, data_loader.action_units)

    print(f"Model loaded: {vocab_builder.vocab_size} tokens, {vocab_builder.unit_vocab_size} units")

    return model, vocab_builder


def create_timestamp_features(timestamps: np.ndarray) -> np.ndarray:
    """
    Create raw timestamp features [t, t^2, delta_t] without scaling.
    Scaling is performed separately after embedding statistics are collected.

    Parameters:
        timestamps (np.ndarray): Array of timestamp values.

    Returns:
        np.ndarray: shape (n, 3) — [t, t^2, delta_t] on the original scale.
    """
    timestamps = np.array(timestamps, dtype=np.float64)
    t          = timestamps.copy()
    t_squared  = t ** 2

    time_deltas    = np.zeros_like(timestamps)
    time_deltas[1:] = np.diff(timestamps)
    time_deltas[0]  = timestamps[0]  # First delta defaults to the original timestamp value

    return np.column_stack([t, t_squared, time_deltas])


def scale_time_features_to_embedding(time_features: np.ndarray,
                                      emb_mean: float,
                                      emb_std: float,
                                      time_means: np.ndarray,
                                      time_stds: np.ndarray) -> np.ndarray:
    """
    Rescale time features to match the embedding space scale.

    Transformation steps:
        1. Z-score standardize each time feature using per-item statistics (mean 0, std 1).
        2. Multiply by the embedding standard deviation (σ_emb) and add the embedding mean (μ_emb).
           The time features end up on the same distributional scale as the embeddings.

    Parameters:
        time_features (np.ndarray): shape (n, 3) — raw time features.
        emb_mean (float):           Global embedding mean for the item (μ_emb).
        emb_std (float):            Global embedding standard deviation for the item (σ_emb).
        time_means (np.ndarray):    shape (3,) — per-feature time means for the item.
        time_stds (np.ndarray):     shape (3,) — per-feature time standard deviations for the item.

    Returns:
        np.ndarray: shape (n, 3) — time features rescaled to embedding space.
    """
    # Guard against zero std (constant feature)
    safe_time_stds = np.where(time_stds > 0, time_stds, 1.0)

    # Step 1: Z-score standardization using per-item time statistics
    z_scored = (time_features - time_means) / safe_time_stds

    # Step 2: Rescale to embedding space
    scaled = z_scored * emb_std + emb_mean

    return scaled


def create_embedding_matrices_by_seqid(df: pd.DataFrame,
                                        processed_event_column: str,
                                        timestamp_column: str,
                                        model,
                                        vocab_builder,
                                        include_timestamp: bool = True,
                                        scale_to_embedding: bool = True) -> Dict[str, np.ndarray]:
    """
    Build per-SEQID action-unit embedding matrices (with optional timestamp features).

    Two-pass design:
        Pass 1: Generate embeddings and time features separately,
                collecting per-item statistics.
        Pass 2: Rescale time features to embedding scale, then concatenate.

    Parameters:
        include_timestamp (bool):   Whether to include timestamp features.
        scale_to_embedding (bool):  Whether to rescale time features to embedding scale.
    """
    grouped = df.groupby('SEQID')

    # ========== Pass 1: Generate embeddings and time features, collect statistics ==========
    emb_dict  = {}  # {seqid: embedding_matrix (n, dim)}
    time_dict = {}  # {seqid: time_features (n, 3)}

    all_embeddings    = []  # collect all embeddings for per-item statistics
    all_time_features = []  # collect all time features for per-item statistics

    for seqid, group in grouped:
        group = group.sort_values(by=timestamp_column)

        processed_events = group[processed_event_column].tolist()
        timestamps       = group[timestamp_column].values

        embeddings = []
        for processed_event in processed_events:
            event_str = str(processed_event)
            embedding = model.get_fasttext_unit_embedding(event_str, vocab_builder)
            if embedding is not None:
                embeddings.append(embedding)

        if embeddings:
            embedding_matrix  = np.stack(embeddings)
            emb_dict[seqid]   = embedding_matrix
            all_embeddings.append(embedding_matrix)

            if include_timestamp:
                time_features       = create_timestamp_features(timestamps)
                time_dict[seqid]    = time_features
                all_time_features.append(time_features)

    # ========== Pass 2: Rescale and concatenate ==========
    embedding_matrices = {}

    if include_timestamp and scale_to_embedding and all_embeddings and all_time_features:
        # Compute per-item embedding statistics (flattened across all dimensions)
        all_emb_concat = np.concatenate(all_embeddings, axis=0)  # (total_actions, embedding_dim)
        emb_mean = all_emb_concat.mean()  # scalar: global embedding mean
        emb_std  = all_emb_concat.std()   # scalar: global embedding std

        # Compute per-item time feature statistics (per feature column)
        all_time_concat = np.concatenate(all_time_features, axis=0)  # (total_actions, 3)
        time_means = all_time_concat.mean(axis=0)  # shape (3,)
        time_stds  = all_time_concat.std(axis=0)   # shape (3,)

        print(f"  [scaling] embedding μ={emb_mean:.4f}, σ={emb_std:.4f}")
        print(f"  [scaling] time feature μ={time_means}, σ={time_stds}")

        for seqid in emb_dict:
            scaled_time = scale_time_features_to_embedding(
                time_dict[seqid], emb_mean, emb_std, time_means, time_stds
            )
            embedding_matrices[seqid] = np.concatenate([emb_dict[seqid], scaled_time], axis=1)

    elif include_timestamp and not scale_to_embedding:
        # Concatenate without rescaling
        for seqid in emb_dict:
            embedding_matrices[seqid] = np.concatenate([emb_dict[seqid], time_dict[seqid]], axis=1)

    else:
        # No timestamp features: return embeddings only
        embedding_matrices = emb_dict

    return embedding_matrices


def extract_multi_problem_embeddings(problem_nums: list,
                                     base_data_path: str,
                                     base_model_path: str,
                                     base_test_path: str,
                                     include_timestamp: bool = True,
                                     scale_to_embedding: bool = True) -> Dict[tuple, np.ndarray]:
    """
    Extract embedding matrices keyed by (SEQID, problem_num) for multiple items.

    Parameters:
        problem_nums (list):       List of item IDs (e.g., ['ps1_1', 'ps1_2', 'ps2_7']).
        base_data_path (str):      Path template for data files.
        base_model_path (str):     Path template for model files.
        base_test_path (str):      Path template for sequence text files.
        include_timestamp (bool):  Whether to include timestamp features.
        scale_to_embedding (bool): Whether to rescale time features to embedding scale.

    Returns:
        Dict[tuple, np.ndarray]: {(SEQID, problem_num): embedding_matrix}
    """
    result_dict = {}

    for problem_num in problem_nums:
        print(f"\n{'=' * 60}")
        print(f"Processing: {problem_num}")
        print(f"{'=' * 60}")

        data_path  = base_data_path.format(problem_num=problem_num)
        model_path = base_model_path.format(problem_num=problem_num)
        test_path  = base_test_path.format(problem_num=problem_num)

        try:
            model, vocab_builder = load_model_and_rebuild_vocab(model_path, test_path)

            df = pd.read_pickle(data_path)

            matrices = create_embedding_matrices_by_seqid(
                df, 'processed_event', 'timestamp', model, vocab_builder,
                include_timestamp=include_timestamp,
                scale_to_embedding=scale_to_embedding
            )

            for seqid, matrix in matrices.items():
                result_dict[(seqid, problem_num)] = matrix

            print(f"✓ {problem_num}: {len(matrices)} SEQID entries processed")
            if matrices:
                sample_shape     = list(matrices.values())[0].shape
                timestamp_status = "included" if include_timestamp else "excluded"
                print(f"  Sample matrix shape: {sample_shape} (timestamp {timestamp_status})")

        except Exception as e:
            print(f"✗ {problem_num}: error — {str(e)}")
            continue

    print(f"\n{'=' * 60}")
    print(f"All items processed: {len(result_dict)} total entries")
    print(f"{'=' * 60}")

    return result_dict


def all_items() -> list[str]:
    return [f"ps{i}_{j}" for i, j in product(range(1, 3), range(1, 8))]


def save_grouped_outputs(embedding_dict: Dict[tuple, np.ndarray],
                         output_dir: Path, suffix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = {"ps1": {}, "ps2": {}}
    for key, matrix in embedding_dict.items():
        _, problem_num = key
        if problem_num.startswith("ps1_"):
            grouped["ps1"][key] = matrix
        elif problem_num.startswith("ps2_"):
            grouped["ps2"][key] = matrix

    for group_name, group_dict in grouped.items():
        output_path = output_dir / f"embed_mat_{group_name}_{suffix}.pkl"
        with output_path.open("wb") as handle:
            pickle.dump(group_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{group_name.upper()} saved: {output_path} ({len(group_dict)} entries)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build respondent-level action embedding matrices with timestamp features."
    )
    parser.add_argument("--items", nargs="*", default=all_items())
    parser.add_argument(
        "--data-template",
        required=True,
        help="Template for third-pass pickle files, e.g. input_data/3rd_data/us_{problem_num}.pkl",
    )
    parser.add_argument(
        "--model-template",
        required=True,
        help="Template for trained model files, e.g. outputs/HW2V_{problem_num}_20/fasttext_style_word2vec.pkl",
    )
    parser.add_argument(
        "--sequence-template",
        required=True,
        help="Template for Stage-1 sequence text files, e.g. model_input/HW2V/test_us_{problem_num}.txt",
    )
    parser.add_argument("--output-dir",     type=Path, required=True)
    parser.add_argument("--output-suffix",  default="20")
    parser.add_argument("--no-timestamp",          action="store_true")
    parser.add_argument("--no-scale-to-embedding", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedding_dict = extract_multi_problem_embeddings(
        problem_nums      = args.items,
        base_data_path    = args.data_template,
        base_model_path   = args.model_template,
        base_test_path    = args.sequence_template,
        include_timestamp = not args.no_timestamp,
        scale_to_embedding= not args.no_scale_to_embedding,
    )
    save_grouped_outputs(embedding_dict, args.output_dir, args.output_suffix)


if __name__ == "__main__":
    main()