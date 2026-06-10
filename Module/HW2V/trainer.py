"""
Word2Vec Model Trainer (FastText Style)
Adam optimizer with Negative Sampling using both tokens and units as center items.
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import random

from model import Word2VecModel
from vocab_builder import VocabularyBuilder


class AdamOptimizer:
    """
    Adam optimizer implementation for the FastText-style model.
    Manages both token and unit embeddings.
    """

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        """Initialize the Adam optimizer."""
        self.learning_rate = learning_rate
        self.beta1         = beta1
        self.beta2         = beta2
        self.epsilon       = epsilon
        self.t             = 0  # time step

        self.m_token = None
        self.v_token = None
        self.m_unit  = None
        self.v_unit  = None

    def initialize(self, model: Word2VecModel) -> None:
        """Initialize moment variables to match the model."""
        self.m_token = np.zeros_like(model.W_token)
        self.v_token = np.zeros_like(model.W_token)
        self.m_unit  = np.zeros_like(model.W_unit)
        self.v_unit  = np.zeros_like(model.W_unit)

        print("FastText-style Adam optimizer initialized.")

    def update_token_embedding(self, token_idx: int, grad: np.ndarray,
                               model: Word2VecModel) -> None:
        """Apply Adam update to a token embedding."""
        # First moment update
        self.m_token[token_idx] = (self.beta1 * self.m_token[token_idx]
                                   + (1 - self.beta1) * grad)
        # Second moment update
        self.v_token[token_idx] = (self.beta2 * self.v_token[token_idx]
                                   + (1 - self.beta2) * (grad ** 2))
        # Bias correction
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t
        m_corrected = self.m_token[token_idx] / bias_correction1
        v_corrected = self.v_token[token_idx] / bias_correction2

        model.W_token[token_idx] -= (self.learning_rate * m_corrected /
                                     (np.sqrt(v_corrected) + self.epsilon))

    def update_unit_embedding(self, unit_idx: int, grad: np.ndarray,
                              model: Word2VecModel) -> None:
        """Apply Adam update to a unit embedding."""
        # First moment update
        self.m_unit[unit_idx] = (self.beta1 * self.m_unit[unit_idx]
                                 + (1 - self.beta1) * grad)
        # Second moment update
        self.v_unit[unit_idx] = (self.beta2 * self.v_unit[unit_idx]
                                 + (1 - self.beta2) * (grad ** 2))
        # Bias correction
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t
        m_corrected = self.m_unit[unit_idx] / bias_correction1
        v_corrected = self.v_unit[unit_idx] / bias_correction2

        model.W_unit[unit_idx] -= (self.learning_rate * m_corrected /
                                   (np.sqrt(v_corrected) + self.epsilon))


class Word2VecTrainer:
    """
    Class for managing FastText-style Token-Unit Word2Vec model training.
    """

    def __init__(self, model: Word2VecModel, vocab_builder: VocabularyBuilder,
                 learning_rate: float = 0.001, negative_samples: int = 5,
                 early_stopping: bool = False, patience: int = 3):
        """
        Initialize the FastText-style trainer.

        Parameters:
            model (Word2VecModel):          FastText-style model to train.
            vocab_builder (VocabularyBuilder): Vocabulary builder for tokens and units.
            learning_rate (float):          Learning rate.
            negative_samples (int):         Number of negative samples.
            early_stopping (bool):          Whether to use early stopping.
            patience (int):                 Early stopping patience.
        """
        self.model            = model
        self.vocab_builder    = vocab_builder
        self.negative_samples = negative_samples

        self.optimizer = AdamOptimizer(learning_rate=learning_rate)
        self.optimizer.initialize(model)

        self.unit_sampling_distribution = vocab_builder.get_negative_sampling_distribution()
        self._build_negative_sampling_table()

        self.early_stopping    = early_stopping
        self.patience          = patience
        self.best_loss         = float('inf')
        self.patience_counter  = 0
        self.best_model_state  = None

        print(f"FastText-style Word2Vec trainer initialized:")
        print(f"  Learning rate:     {learning_rate}")
        print(f"  Negative samples:  {negative_samples}")
        print(f"  Token vocab size:  {vocab_builder.vocab_size}")
        print(f"  Unit vocab size:   {vocab_builder.unit_vocab_size}")
        print(f"  Early stopping:    {'enabled' if early_stopping else 'disabled'}")
        if early_stopping:
            print(f"  Patience:          {patience} epochs")

    def _build_negative_sampling_table(self, table_size: int = 1000000) -> None:
        """Build a lookup table for fast negative sampling over units."""
        self.negative_sampling_table = np.zeros(table_size, dtype=np.int32)

        unit_vocab_size = len(self.unit_sampling_distribution)
        table_idx = 0

        for unit_idx in range(unit_vocab_size):
            count = int(self.unit_sampling_distribution[unit_idx] * table_size)
            for _ in range(count):
                if table_idx < table_size:
                    self.negative_sampling_table[table_idx] = unit_idx
                    table_idx += 1

        # Fill remaining slots randomly
        while table_idx < table_size:
            self.negative_sampling_table[table_idx] = np.random.randint(0, unit_vocab_size)
            table_idx += 1

        print(f"Negative sampling table built (size: {table_size:,})")

    def _sample_negative_units(self, positive_unit: str,
                               num_samples: int) -> List[str]:
        """
        Sample negative units.

        Parameters:
            positive_unit (str): Positive unit to exclude from samples.
            num_samples (int):   Number of negative samples to draw.

        Returns:
            List[str]: Names of sampled negative units.
        """
        positive_unit_idx = self.vocab_builder.get_unit_index(positive_unit)
        negative_samples  = []

        while len(negative_samples) < num_samples:
            table_idx     = np.random.randint(0, len(self.negative_sampling_table))
            candidate_idx = self.negative_sampling_table[table_idx]

            if candidate_idx != positive_unit_idx:
                candidate_unit = self.vocab_builder.get_unit_by_index(candidate_idx)
                if candidate_unit is not None and candidate_unit not in negative_samples:
                    negative_samples.append(candidate_unit)

        return negative_samples

    def evaluate(self, validation_pairs: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Evaluate the model on validation data (forward pass only; no gradient updates).

        Parameters:
            validation_pairs (List[Tuple[str, str]]): Validation training pairs.

        Returns:
            Dict[str, float]: {
                'valid_loss': average loss,
                'valid_pairs_count': number of evaluated pairs,
                'valid_total_pairs': total number of pairs
            }

        Note:
            Validation-only vocabulary items are included but treated as reference only.
            Validation loss is more informative when vocabulary overlaps heavily with training.
        """
        total_loss  = 0.0
        valid_count = 0

        for center_item, context_unit in validation_pairs:
            if not (self.vocab_builder.is_valid_item(center_item) and
                    self.vocab_builder.is_valid_item(context_unit)):
                continue

            negative_units = self._sample_negative_units(context_unit, self.negative_samples)

            loss, _ = self.model.forward_pass(
                center_item, context_unit, negative_units, self.vocab_builder
            )

            if not np.isinf(loss):
                total_loss  += loss
                valid_count += 1

        avg_loss = total_loss / valid_count if valid_count > 0 else float('inf')

        return {
            'valid_loss'        : avg_loss,
            'valid_pairs_count' : valid_count,
            'valid_total_pairs' : len(validation_pairs)
        }

    def _apply_fasttext_gradients(self, gradients: Dict[str, np.ndarray]) -> None:
        """
        Distribute and apply gradients in the FastText style.

        Parameters:
            gradients (Dict[str, np.ndarray]): Per-item gradients.
        """
        self.optimizer.t += 1

        for item, grad in gradients.items():
            unit_idx = self.vocab_builder.get_unit_index(item)
            if unit_idx is not None:  # Unit: distribute gradients FastText-style
                composition_indices = self.vocab_builder.get_unit_composition(item)
                if composition_indices:
                    # FastText: distribute gradient across composition components
                    # grad_per_component = grad / len(composition_indices)
                    for comp_type, idx in composition_indices:
                        if comp_type == 'token':
                            self.optimizer.update_token_embedding(idx, grad, self.model)
                        elif comp_type == 'unit':
                            self.optimizer.update_unit_embedding(idx, grad, self.model)
            else:  # Token: direct update
                token_idx = self.vocab_builder.get_token_index(item)
                if token_idx is not None:
                    self.optimizer.update_token_embedding(token_idx, grad, self.model)

    def train_epoch(self, training_pairs: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Train the FastText-style model for one epoch.

        Parameters:
            training_pairs (List[Tuple[str, str]]): List of (center_item, context_unit) pairs.

        Returns:
            Dict[str, float]: Epoch training statistics.
        """
        epoch_loss       = 0.0
        processed_pairs  = 0
        start_time       = time.time()

        shuffled_pairs   = training_pairs.copy()
        random.shuffle(shuffled_pairs)

        batch_gradients  = defaultdict(lambda: np.zeros(self.model.embed_dim))
        batch_size       = 100

        for pair_idx, (center_item, context_unit) in enumerate(shuffled_pairs):
            if not (self.vocab_builder.is_valid_item(center_item) and
                    self.vocab_builder.is_valid_item(context_unit)):
                continue

            negative_units = self._sample_negative_units(context_unit, self.negative_samples)

            loss, forward_cache = self.model.forward_pass(
                center_item, context_unit, negative_units, self.vocab_builder
            )

            if np.isinf(loss):
                continue  # Skip invalid loss

            center_grads, context_grads, negative_grads = self.model.backward_pass(
                forward_cache, self.vocab_builder
            )

            all_gradients = defaultdict(lambda: np.zeros(self.model.embed_dim))
            for item, grad in center_grads.items():
                all_gradients[item] += grad
            for item, grad in context_grads.items():
                all_gradients[item] += grad
            for item, grad in negative_grads.items():
                all_gradients[item] += grad

            for item, grad in all_gradients.items():
                batch_gradients[item] += grad

            epoch_loss      += loss
            processed_pairs += 1

            # Apply batch update
            if (pair_idx + 1) % batch_size == 0 or pair_idx == len(shuffled_pairs) - 1:
                self._apply_fasttext_gradients(dict(batch_gradients))
                batch_gradients.clear()

        avg_loss   = epoch_loss / processed_pairs if processed_pairs > 0 else 0.0
        epoch_time = time.time() - start_time

        return {
            'avg_loss'       : avg_loss,
            'total_loss'     : epoch_loss,
            'processed_pairs': processed_pairs,
            'epoch_time'     : epoch_time,
            'pairs_per_sec'  : processed_pairs / epoch_time if epoch_time > 0 else 0
        }

    def train(self, training_pairs: List[Tuple[str, str]],
              validation_pairs: Optional[List[Tuple[str, str]]] = None,
              epochs: int = 5,
              save_path: Optional[str] = None,
              evaluation_words: Optional[List[str]] = None) -> List[Dict]:
        """
        Run the full FastText-style training procedure.

        Parameters:
            training_pairs (List[Tuple[str, str]]):   Training pairs.
            validation_pairs (Optional[List[...]]]):  Validation pairs (if None, validation is skipped).
            epochs (int):                             Number of training epochs.
            save_path (Optional[str]):                Path to save the model.
            evaluation_words (Optional[List[str]]):   Token list for qualitative evaluation.

        Returns:
            List[Dict]: Per-epoch training and validation statistics.
        """
        print(f"\n=== FastText-style Token-Unit Word2Vec training started ===")
        print(f"Training pairs:   {len(training_pairs):,}")
        if validation_pairs is not None:
            print(f"Validation pairs: {len(validation_pairs):,}")
        print(f"Epochs:           {epochs}")
        print(f"Token vocab size: {self.vocab_builder.vocab_size:,}")
        print(f"Unit vocab size:  {self.vocab_builder.unit_vocab_size:,}")
        print(f"Embedding dim:    {self.model.embed_dim}")
        print(f"Unit embedding:   FastText-style (sum of tokens + unit)")
        print("=" * 50)

        training_history = []

        for epoch in range(1, epochs + 1):
            print(f"\n--- Epoch {epoch}/{epochs} ---")

            epoch_stats        = self.train_epoch(training_pairs)
            epoch_stats['epoch'] = epoch
            train_loss         = epoch_stats['avg_loss']

            print(f"Epoch {epoch} complete:")
            print(f"  Train Loss:   {train_loss:.6f}")
            print(f"  Elapsed time: {epoch_stats['epoch_time']:.1f}s")

            if validation_pairs is not None:
                print(f"  Evaluating on validation set...")
                valid_stats = self.evaluate(validation_pairs)

                epoch_stats['valid_loss']        = valid_stats['valid_loss']
                epoch_stats['valid_pairs_count'] = valid_stats['valid_pairs_count']

                valid_loss = valid_stats['valid_loss']
                print(f"  Valid Loss:             {valid_loss:.6f}")
                print(f"  Loss Gap (Valid-Train): {valid_loss - train_loss:+.6f}")

            training_history.append(epoch_stats)

            # Early stopping (based on validation loss)
            if self.early_stopping and validation_pairs is not None:
                current_loss = valid_stats['valid_loss']

                if (self.best_loss - current_loss) > 1e-5:
                    self.best_loss       = current_loss
                    self.patience_counter = 0
                    # Save best model state
                    self.best_model_state = {
                        'W_token'   : self.model.W_token.copy(),
                        'W_unit'    : self.model.W_unit.copy(),
                        'epoch'     : epoch,
                        'train_loss': train_loss,
                        'valid_loss': valid_loss
                    }
                    print(f"  Best model updated (Valid Loss: {valid_loss:.6f})")
                else:
                    self.patience_counter += 1
                    print(f"  No improvement ({self.patience_counter}/{self.patience})")

                    if self.patience_counter >= self.patience:
                        print(f"\n" + "=" * 50)
                        print(f"Early Stopping")
                        print(f"  Best epoch:       {self.best_model_state['epoch']}")
                        print(f"  Best train loss:  {self.best_model_state['train_loss']:.6f}")
                        print(f"  Best valid loss:  {self.best_model_state['valid_loss']:.6f}")
                        print("=" * 50)

                        # Restore best model
                        self.model.W_token = self.best_model_state['W_token']
                        self.model.W_unit  = self.best_model_state['W_unit']

                        if save_path:
                            final_path = save_path if save_path.endswith('.pkl') else f"{save_path}.pkl"
                            self.model.save_model(final_path)
                            print(f"  Best model saved: {final_path}")
                        break

            # Early stopping (based on train loss when no validation set)
            elif self.early_stopping and validation_pairs is None:
                current_loss = train_loss

                if current_loss < self.best_loss:
                    self.best_loss        = current_loss
                    self.patience_counter = 0
                    # Save best model state
                    self.best_model_state = {
                        'W_token'   : self.model.W_token.copy(),
                        'W_unit'    : self.model.W_unit.copy(),
                        'epoch'     : epoch,
                        'train_loss': train_loss
                    }
                    print(f"  Best model updated (Train Loss: {train_loss:.6f})")
                else:
                    self.patience_counter += 1
                    print(f"  No improvement ({self.patience_counter}/{self.patience})")

                    if self.patience_counter >= self.patience:
                        print(f"\nEarly Stopping")
                        print(f"  Best epoch:      {self.best_model_state['epoch']}")
                        print(f"  Best train loss: {self.best_model_state['train_loss']:.6f}")

                        # Restore best model
                        self.model.W_token = self.best_model_state['W_token']
                        self.model.W_unit  = self.best_model_state['W_unit']

                        if save_path:
                            final_path = save_path if save_path.endswith('.pkl') else f"{save_path}.pkl"
                            self.model.save_model(final_path)
                            print(f"  Best model saved: {final_path}")
                        break

            # Optional: qualitative evaluation
            if evaluation_words and epoch % 2 == 0:
                self._evaluate_model(evaluation_words)

            # Optional: save intermediate model
            if save_path and epoch % 5 == 0:
                intermediate_path = save_path.replace('.pkl', f'_epoch_{epoch}.pkl')
                self.model.save_model(intermediate_path)

        # Save final model if early stopping was not triggered
        if save_path:
            early_stopped = (self.early_stopping and
                             ((validation_pairs is not None and self.patience_counter >= self.patience) or
                              (validation_pairs is None and self.patience_counter >= self.patience)))
            if not early_stopped:
                final_path = save_path if save_path.endswith('.pkl') else f"{save_path}.pkl"
                self.model.save_model(final_path)
                print(f"\nFinal model saved: {final_path}")

        print(f"\n=== FastText-style training complete ===")
        return training_history

    def _evaluate_model(self, evaluation_words: List[str], top_k: int = 5) -> None:
        """Print FastText-style analysis for a set of evaluation words."""
        print(f"\n--- FastText-style model evaluation ---")

        for word in evaluation_words[:3]:
            unit_index = self.vocab_builder.get_unit_index(word)

            # Token
            if unit_index is None:
                token_embed = self.model.get_item_embedding(word, self.vocab_builder)
                if token_embed is not None:
                    print(f"Token '{word}' embedding norm: {np.linalg.norm(token_embed):.4f}")

                    # Find units similar to this token
                    similar_units = []
                    for unit_idx in range(min(10, self.vocab_builder.unit_vocab_size)):
                        unit = self.vocab_builder.get_unit_by_index(unit_idx)
                        if unit and '_' in unit:
                            similarity = self.model.get_similarity(word, unit, self.vocab_builder)
                            similar_units.append((unit, similarity))

                    if similar_units:
                        similar_units.sort(key=lambda x: x[1], reverse=True)
                        print(f"  Similar units:")
                        for unit, sim in similar_units[:3]:
                            print(f"    {unit}: {sim:.4f}")

            # Unit: FastText-style analysis
            elif self.vocab_builder.is_valid_item(word):
                composition = self.vocab_builder.get_unit_composition(word)
                if composition:
                    print(f"Unit '{word}' FastText composition:")
                    component_names = []
                    for comp_type, idx in composition:
                        if comp_type == 'token':
                            token = self.vocab_builder.get_token_by_index(idx)
                            component_names.append(f"token:{token}")
                        elif comp_type == 'unit':
                            unit = self.vocab_builder.get_unit_by_index(idx)
                            component_names.append(f"unit:{unit}")

                    print(f"  Components: {', '.join(component_names)}")

                    fasttext_embed = self.model.get_fasttext_unit_embedding(word, self.vocab_builder)
                    if fasttext_embed is not None:
                        print(f"  FastText embedding norm: {np.linalg.norm(fasttext_embed):.4f}")

        print("-" * 40)