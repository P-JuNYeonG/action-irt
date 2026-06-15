"""
Hybrid-Style Token-Unit Word2Vec Training Script
Training pipeline using both tokens and units as center items.
"""

import argparse
import os
import time
import json
from typing import Dict, Any
import numpy as np

from data_loader import ActionDataLoader
from vocab_builder import VocabularyBuilder
from model import Word2VecModel
from trainer import Word2VecTrainer


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Hybrid-style Token-Unit Word2Vec model training script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument("--input_file", "-i", type=str, required=True,
                        help="Path to the input data file (.txt).")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="Path to the output directory.")

    # Model hyperparameters (Hybrid-style)
    parser.add_argument("--embed_dim", type=int, default=20,
                        help="Embedding vector dimension.")
    parser.add_argument("--window_size", type=int, default=1,
                        help="Context window size in units.")
    parser.add_argument("--min_count", type=int, default=1,
                        help="Minimum frequency threshold for vocabulary inclusion.")
    parser.add_argument("--negative_samples", type=int, default=5,
                        help="Number of negative samples.")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Adam optimizer learning rate.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    parser.add_argument("--early_stopping", action="store_true",
                        help="Enable early stopping.")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (default: 3).")

    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Train/validation split ratio (default: 0.8).")
    parser.add_argument("--no_validation", action="store_true",
                        help="Disable validation and train on the full dataset.")

    return parser.parse_args()


def setup_output_directory(output_dir: str) -> Dict[str, str]:
    """Set up the output directory and return file paths."""
    os.makedirs(output_dir, exist_ok=True)

    paths = {
        'model'              : os.path.join(output_dir, 'fasttext_style_word2vec.pkl'),
        'vocabulary'         : os.path.join(output_dir, 'vocabulary.txt'),
        'training_log'       : os.path.join(output_dir, 'training_log.json'),
        'config'             : os.path.join(output_dir, 'config.json'),
        'token_embeddings'   : os.path.join(output_dir, 'token_embeddings.npz'),
        'unit_embeddings'    : os.path.join(output_dir, 'unit_embeddings_fasttext.npz'),
        'fasttext_analysis'  : os.path.join(output_dir, 'fasttext_comprehensive_analysis.json'),
        'token_tsv'          : os.path.join(output_dir, 'token_embeddings.tsv'),
        'token_metadata_tsv' : os.path.join(output_dir, 'token_metadata.tsv'),
        'unit_tsv'           : os.path.join(output_dir, 'unit_embeddings_fasttext.tsv'),
        'unit_metadata_tsv'  : os.path.join(output_dir, 'unit_metadata_fasttext.tsv')
    }

    return paths


def save_configuration(config: Dict[str, Any], config_path: str) -> None:
    """Save training configuration to a JSON file."""
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"Configuration saved: {config_path}")


def save_fasttext_embeddings(model: Word2VecModel, vocab_builder: VocabularyBuilder,
                             token_file: str, unit_file: str) -> None:
    """Save FastText-style token and unit embeddings."""
    token_embeddings = []
    token_names      = []

    for idx in range(vocab_builder.vocab_size):
        token     = vocab_builder.get_token_by_index(idx)
        embedding = model.get_token_embedding(idx)
        token_embeddings.append(embedding)
        token_names.append(token)

    np.savez_compressed(token_file,
                        embeddings=np.stack(token_embeddings),
                        tokens=np.array(token_names),
                        embed_dim=model.embed_dim,
                        model_type='fasttext_style_token')

    unit_embeddings  = []
    unit_names       = []
    unit_compositions = []

    for idx in range(vocab_builder.unit_vocab_size):
        unit      = vocab_builder.get_unit_by_index(idx)
        embedding = model.get_fasttext_unit_embedding(unit, vocab_builder)
        if embedding is not None:
            unit_embeddings.append(embedding)
            unit_names.append(unit)

            # Store composition metadata
            composition = vocab_builder.get_unit_composition(unit)
            comp_info   = []
            for comp_type, comp_idx in composition:
                if comp_type == 'token':
                    comp_info.append(f"token:{vocab_builder.get_token_by_index(comp_idx)}")
                elif comp_type == 'unit':
                    comp_info.append(f"unit:{vocab_builder.get_unit_by_index(comp_idx)}")
            unit_compositions.append(','.join(comp_info))

    if unit_embeddings:
        np.savez_compressed(unit_file,
                            embeddings=np.stack(unit_embeddings),
                            units=np.array(unit_names),
                            compositions=np.array(unit_compositions),
                            embed_dim=model.embed_dim,
                            model_type='fasttext_style_unit')

    print(f"FastText-style embeddings saved:")
    print(f"  Token embeddings: {token_file} ({len(token_names)} tokens)")
    print(f"  Unit embeddings:  {unit_file} ({len(unit_names)} units, FastText-style)")
    print(f"  Unit composition metadata included.")


def main():
    """Main entry point."""
    print("=" * 70)
    print("FastText-style Token-Unit Word2Vec model training")
    print("unit_embedding = sum([token_embeddings, unit_embedding])")
    print("=" * 70)

    args       = parse_arguments()
    file_paths = setup_output_directory(args.output_dir)

    config = {
        'input_file'  : args.input_file,
        'output_dir'  : args.output_dir,
        'model_type'  : 'fasttext_style_token_unit',
        'model_description': 'FastText style: unit_embedding = sum([token_embeddings, unit_embedding])',
        'learning_approach': 'Token-Unit and Unit-Unit pairs with FastText gradient distribution',
        'model_hyperparameters': {
            'embed_dim'       : args.embed_dim,
            'window_size'     : args.window_size,
            'min_count'       : args.min_count,
            'negative_samples': args.negative_samples,
            'random_seed'     : args.random_seed
        },
        'training_hyperparameters': {
            'learning_rate': args.learning_rate,
            'epochs'       : args.epochs,
            'early_stopping': args.early_stopping,
            'patience'     : args.patience
        },
        'data_split_options': {
            'use_validation': not args.no_validation,
            'train_ratio'   : args.train_ratio if not args.no_validation else 1.0
        }
    }
    save_configuration(config, file_paths['config'])

    # --- Step 1: Data loading and preprocessing ---
    print("\n--- Step 1: Data loading and preprocessing ---")
    start_time  = time.time()
    data_loader = ActionDataLoader()
    data_loader.load_from_file(args.input_file)
    data_loader.tokenize_sequences()

    data_stats = data_loader.get_vocabulary_stats()
    print(f"Data loaded ({time.time() - start_time:.1f}s):")
    for key, value in data_stats.items():
        print(f"  {key}: {value}")

    # --- Step 2: Building vocabulary from full dataset ---
    print("\n--- Step 2: Building vocabulary from full dataset ---")
    start_time    = time.time()
    vocab_builder = VocabularyBuilder(min_count=args.min_count)
    vocab_builder.build_vocabulary(data_loader.token_counter, data_loader.action_units)
    vocab_builder.save_vocabulary(file_paths['vocabulary'])

    vocab_info = vocab_builder.get_vocabulary_info()
    print(f"Vocabulary built ({time.time() - start_time:.1f}s):")
    print(f"  Token vocab size: {vocab_info['token_vocab_size']:,}")
    print(f"  Unit vocab size:  {vocab_info['unit_vocab_size']:,}")

    # --- Step 3: Pair generation ---
    if not args.no_validation:
        print("\n--- Step 3: Train/validation split and pair generation ---")
        start_time = time.time()

        train_sequences, valid_sequences = data_loader.split_sequences(
            train_ratio=args.train_ratio,
            random_seed=args.random_seed
        )

        print("\n[Generating Train pairs]")
        train_pairs          = data_loader.get_training_pairs_from_sequences(
            train_sequences, window_size=args.window_size
        )
        train_pairs_filtered = vocab_builder.filter_training_pairs(train_pairs)

        print("\n[Generating Validation pairs]")
        valid_pairs          = data_loader.get_training_pairs_from_sequences(
            valid_sequences, window_size=args.window_size
        )
        valid_pairs_filtered = vocab_builder.filter_training_pairs(valid_pairs)

        print(f"\nPairs generated ({time.time() - start_time:.1f}s):")
        print(f"  Train pairs: {len(train_pairs_filtered):,}")
        print(f"  Valid pairs: {len(valid_pairs_filtered):,}")

    else:
        print("\n--- Step 3: Pair generation without validation ---")
        start_time = time.time()

        train_pairs          = data_loader.get_training_pairs(window_size=args.window_size)
        train_pairs_filtered = vocab_builder.filter_training_pairs(train_pairs)
        valid_pairs_filtered = None
        train_sequences      = data_loader.sequences
        valid_sequences      = []

        print(f"\nPairs generated ({time.time() - start_time:.1f}s):")
        print(f"  Total pairs: {len(train_pairs_filtered):,}")

    # --- Step 4: FastText-style model initialization ---
    print("\n--- Step 4: FastText-style model initialization ---")
    model = Word2VecModel(
        token_vocab_size=vocab_builder.vocab_size,
        unit_vocab_size=vocab_builder.unit_vocab_size,
        embed_dim=args.embed_dim,
        random_seed=args.random_seed
    )
    model_info = model.get_model_info()
    print(f"Total model parameters: {model_info['total_parameters']:,}")

    # --- Step 5: FastText-style model training ---
    print("\n--- Step 5: FastText-style model training ---")
    start_time = time.time()

    trainer = Word2VecTrainer(
        model=model,
        vocab_builder=vocab_builder,
        learning_rate=args.learning_rate,
        negative_samples=args.negative_samples,
        early_stopping=args.early_stopping,
        patience=args.patience
    )

    training_history = trainer.train(
        training_pairs=train_pairs_filtered,
        validation_pairs=valid_pairs_filtered,
        epochs=args.epochs,
        save_path=file_paths['model']
    )

    total_training_time = time.time() - start_time
    print(f"\nTraining complete (total: {total_training_time:.1f}s)")

    # --- Step 5: Saving training log ---
    print("\n--- Step 5: Saving training log ---")

    token_unit_pairs = sum(1 for center, _ in train_pairs_filtered
                           if center in vocab_builder.token_to_idx and
                              center not in vocab_builder.unit_to_idx)
    unit_unit_pairs  = sum(1 for center, _ in train_pairs_filtered
                           if center in vocab_builder.unit_to_idx)

    training_log = {
        'model_type'          : 'fasttext_style_token_unit',
        'training_history'    : training_history,
        'total_training_time' : total_training_time,
        'final_train_loss'    : training_history[-1]['avg_loss'] if training_history else 0.0,
        'final_valid_loss'    : training_history[-1].get('valid_loss', None) if training_history else None,
        'data_split': {
            'use_validation'       : not args.no_validation,
            'train_ratio'          : args.train_ratio if not args.no_validation else 1.0,
            'train_sequences'      : len(train_sequences),
            'valid_sequences'      : len(valid_sequences),
            'train_pairs'          : len(train_pairs_filtered),
            'valid_pairs'          : len(valid_pairs_filtered) if valid_pairs_filtered else 0,
            'train_token_unit_pairs': token_unit_pairs,
            'train_unit_unit_pairs' : unit_unit_pairs
        },
        'data_stats'  : data_stats,
        'vocab_info'  : vocab_info,
        'model_info'  : model_info,
        'config'      : config,
        'fasttext_characteristics': {
            'unit_composition_method': 'sum([token_embeddings, unit_embedding])',
            'learning_pairs'         : 'Token-Unit + Unit-Unit',
            'gradient_distribution'  : 'FastText style',
            'semantic_approach'      : 'Compositional + Holistic'
        }
    }

    with open(file_paths['training_log'], 'w', encoding='utf-8') as f:
        json.dump(training_log, f, ensure_ascii=False, indent=2)
    print(f"Training log saved: {file_paths['training_log']}")

    # --- Step 6: Saving FastText-style embeddings ---
    print("\n--- Step 6: Saving FastText-style embeddings ---")
    start_time = time.time()

    save_fasttext_embeddings(
        model=model,
        vocab_builder=vocab_builder,
        token_file=file_paths['token_embeddings'],
        unit_file=file_paths['unit_embeddings']
    )
    print(f"Embeddings saved ({time.time() - start_time:.1f}s)")

    # --- Training summary ---
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)

    print(f"\nData:")
    print(f"  Total sequences:  {data_stats['total_sequences']:,}")
    if not args.no_validation:
        print(f"  Train sequences:  {len(train_sequences):,} ({args.train_ratio*100:.0f}%)")
        print(f"  Valid sequences:  {len(valid_sequences):,} ({(1-args.train_ratio)*100:.0f}%)")

    print(f"\nVocabulary:")
    print(f"  Token vocab size: {vocab_info['token_vocab_size']:,}")
    print(f"  Unit vocab size:  {vocab_info['unit_vocab_size']:,}")

    print(f"\nTraining pairs:")
    print(f"  Train pairs:  {len(train_pairs_filtered):,}")
    print(f"    - Token-Unit: {token_unit_pairs:,}")
    print(f"    - Unit-Unit:  {unit_unit_pairs:,}")
    if valid_pairs_filtered:
        print(f"  Valid pairs:  {len(valid_pairs_filtered):,}")

    print(f"\nFinal loss:")
    print(f"  Train loss: {training_history[-1]['avg_loss']:.6f}" if training_history else "  N/A")
    if training_history and training_history[-1].get('valid_loss'):
        print(f"  Valid loss: {training_history[-1]['valid_loss']:.6f}")

    print(f"\nSaved files:")
    print(f"  Model:            {file_paths['model']}")
    print(f"  Vocabulary:       {file_paths['vocabulary']}")
    print(f"  Training log:     {file_paths['training_log']}")
    print(f"  Config:           {file_paths['config']}")
    print(f"  Token embeddings: {file_paths['token_embeddings']}")
    print(f"  Unit embeddings:  {file_paths['unit_embeddings']}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except FileNotFoundError as e:
        print(f"\n\nFile not found: {e}")
        print("Please verify the input file path.")
    except Exception as e:
        print(f"\n\nAn error occurred: {str(e)}")
        print("\nTraceback:")
        import traceback
        traceback.print_exc()
    finally:
        print("\nExiting.")