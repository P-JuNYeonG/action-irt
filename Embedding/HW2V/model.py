"""
Word2Vec Model Implementation (Hybrid Style)
Token-Unit Skip-gram with Negative Sampling
Hybrid approach: unit_embedding = SUM([token_embeddings, unit_embedding])
"""

import numpy as np
from typing import Tuple, Optional, List
import pickle


class Word2VecModel:
    """
    FastText-style Token-Unit Skip-gram with Negative Sampling model.

    Attributes:
        token_vocab_size (int):  Token vocabulary size.
        unit_vocab_size (int):   Unit vocabulary size.
        embed_dim (int):         Embedding dimension.
        W_token (np.ndarray):    Token embedding matrix (token_vocab_size x embed_dim).
        W_unit (np.ndarray):     Unit embedding matrix  (unit_vocab_size  x embed_dim).
    """

    def __init__(self, token_vocab_size: int, unit_vocab_size: int,
                 embed_dim: int = 300, random_seed: int = 42):
        """
        Initialize the FastText-style Token-Unit Word2Vec model.

        Parameters:
            token_vocab_size (int): Token vocabulary size.
            unit_vocab_size (int):  Unit vocabulary size.
            embed_dim (int):        Embedding vector dimension (default: 300).
            random_seed (int):      Random seed for reproducibility.
        """
        self.token_vocab_size = token_vocab_size
        self.unit_vocab_size  = unit_vocab_size
        self.embed_dim        = embed_dim

        np.random.seed(random_seed)

        # Xavier initialization
        token_std = np.sqrt(1.0 / embed_dim)
        unit_std  = np.sqrt(1.0 / embed_dim)

        self.W_token = np.random.normal(0, token_std, (token_vocab_size, embed_dim))
        self.W_unit  = np.random.normal(0, unit_std,  (unit_vocab_size,  embed_dim))

        print(f"FastText-style Token-Unit Word2Vec model initialized:")
        print(f"  Token vocab size:  {token_vocab_size}")
        print(f"  Unit vocab size:   {unit_vocab_size}")
        print(f"  Embedding dim:     {embed_dim}")
        print(f"  Token matrix:      {self.W_token.shape}")
        print(f"  Unit matrix:       {self.W_unit.shape}")
        print(f"  Unit embedding:    FastText-style (sum of tokens + unit)")

    def get_token_embedding(self, token_idx: int) -> np.ndarray:
        """Return the embedding vector for the given token index."""
        if token_idx >= self.token_vocab_size or token_idx < 0:
            raise IndexError(f"Token index out of range: {token_idx}")
        return self.W_token[token_idx].copy()

    def get_unit_embedding(self, unit_idx: int) -> np.ndarray:
        """Return the embedding vector for the given unit index."""
        if unit_idx >= self.unit_vocab_size or unit_idx < 0:
            raise IndexError(f"Unit index out of range: {unit_idx}")
        return self.W_unit[unit_idx].copy()

    def get_fasttext_unit_embedding(self, unit: str, vocab_builder) -> Optional[np.ndarray]:
        """
        Compute the final embedding for a unit using the FastText approach.

        Parameters:
            unit (str):   Unit name (e.g., 'button_next').
            vocab_builder: Vocabulary builder.

        Returns:
            Optional[np.ndarray]: FastText-style unit embedding vector.

        Note:
            FastText approach: unit_embedding = sum([token1_emb, token2_emb, unit_emb])
            Example: button_next = sum([button, next, button_next])
        """
        composition_indices = vocab_builder.get_unit_composition(unit)
        if not composition_indices:
            return None

        embeddings = []
        for comp_type, idx in composition_indices:
            if comp_type == 'token':
                embeddings.append(self.W_token[idx])
            elif comp_type == 'unit':
                embeddings.append(self.W_unit[idx])

        if embeddings:
            # FastText: sum all component embeddings
            return np.sum(embeddings, axis=0)

        return None

    def get_item_embedding(self, item: str, vocab_builder) -> Optional[np.ndarray]:
        """
        Return the embedding for an item (token or unit).

        Parameters:
            item (str):    Token or unit name.
            vocab_builder: Vocabulary builder.

        Returns:
            Optional[np.ndarray]: Item embedding vector.
        """
        if vocab_builder.get_unit_index(item) is not None:
            return self.get_fasttext_unit_embedding(item, vocab_builder)

        token_idx = vocab_builder.get_token_index(item)
        if token_idx is not None:
            return self.get_token_embedding(token_idx)

        return None

    def forward_pass(self, center_item: str, context_item: str,
                     negative_items: List[str], vocab_builder) -> Tuple[float, dict]:
        """
        Perform the FastText-style forward pass.

        Parameters:
            center_item (str):          Center item (token or unit).
            context_item (str):         Context item (unit).
            negative_items (List[str]): Negative sample items (units).
            vocab_builder:              Vocabulary builder.

        Returns:
            Tuple[float, dict]: Loss value and intermediate computation cache.
        """
        # 1. Retrieve center item embedding
        center_embed = self.get_item_embedding(center_item, vocab_builder)
        if center_embed is None:
            return float('inf'), {}

        # 2. Compute positive context embedding (FastText-style)
        context_embed = self.get_fasttext_unit_embedding(context_item, vocab_builder)
        if context_embed is None:
            return float('inf'), {}

        positive_score = np.dot(center_embed, context_embed)
        positive_prob  = self._sigmoid(positive_score)

        # 3. Compute negative sample embeddings and scores
        negative_embeds      = []
        negative_scores      = []
        valid_negative_items = []

        for neg_item in negative_items:
            neg_embed = self.get_fasttext_unit_embedding(neg_item, vocab_builder)
            if neg_embed is not None:
                negative_embeds.append(neg_embed)
                negative_scores.append(np.dot(center_embed, neg_embed))
                valid_negative_items.append(neg_item)

        if not negative_scores:
            return float('inf'), {}

        negative_scores = np.array(negative_scores)
        negative_probs  = self._sigmoid(-negative_scores)

        # 4. Compute loss
        positive_loss = -np.log(positive_prob + 1e-10)
        negative_loss = -np.sum(np.log(negative_probs + 1e-10))
        total_loss    = positive_loss + negative_loss

        forward_cache = {
            'center_item'    : center_item,
            'center_embed'   : center_embed,
            'context_item'   : context_item,
            'context_embed'  : context_embed,
            'negative_items' : valid_negative_items,
            'negative_embeds': negative_embeds,
            'positive_score' : positive_score,
            'positive_prob'  : positive_prob,
            'negative_scores': negative_scores,
            'negative_probs' : negative_probs
        }

        return total_loss, forward_cache

    def backward_pass(self, forward_cache: dict, vocab_builder) -> Tuple[dict, dict, dict]:
        """
        Perform the backward pass and compute gradients.

        Parameters:
            forward_cache (dict): Intermediate values cached during the forward pass.
            vocab_builder:        Vocabulary builder.

        Returns:
            Tuple[dict, dict, dict]: Gradients for center, context, and negative items.
        """
        center_item     = forward_cache['center_item']
        center_embed    = forward_cache['center_embed']
        context_item    = forward_cache['context_item']
        context_embed   = forward_cache['context_embed']
        negative_items  = forward_cache['negative_items']
        negative_embeds = forward_cache['negative_embeds']
        positive_prob   = forward_cache['positive_prob']
        negative_probs  = forward_cache['negative_probs']

        # 1. Gradient with respect to center item
        positive_grad = (positive_prob - 1) * context_embed
        negative_grad = np.zeros_like(center_embed)
        for i, neg_embed in enumerate(negative_embeds):
            negative_grad += (1 - negative_probs[i]) * neg_embed
        center_grad = positive_grad + negative_grad

        # 2. Gradient with respect to context item
        context_grad = (positive_prob - 1) * center_embed

        # 3. Gradients with respect to negative items
        negative_grads = {}
        for i, neg_item in enumerate(negative_items):
            negative_grads[neg_item] = (1 - negative_probs[i]) * center_embed

        return {center_item: center_grad}, {context_item: context_grad}, negative_grads

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def get_similarity(self, item1: str, item2: str, vocab_builder) -> float:
        """
        Compute cosine similarity between two items.

        Parameters:
            item1 (str):   First item (token or unit).
            item2 (str):   Second item (token or unit).
            vocab_builder: Vocabulary builder.

        Returns:
            float: Cosine similarity in [-1, 1].
        """
        embed1 = self.get_item_embedding(item1, vocab_builder)
        embed2 = self.get_item_embedding(item2, vocab_builder)

        if embed1 is None or embed2 is None:
            return 0.0

        norm1 = np.linalg.norm(embed1)
        norm2 = np.linalg.norm(embed2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(embed1, embed2) / (norm1 * norm2)

    def save_model(self, file_path: str) -> None:
        """Save the trained model to a file."""
        model_data = {
            'model_type'      : 'fasttext_style_token_unit',
            'token_vocab_size': self.token_vocab_size,
            'unit_vocab_size' : self.unit_vocab_size,
            'embed_dim'       : self.embed_dim,
            'W_token'         : self.W_token,
            'W_unit'          : self.W_unit
        }

        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"FastText-style Token-Unit model saved: {file_path}")

    @classmethod
    def load_model(cls, file_path: str) -> 'Word2VecModel':
        """Load a saved model from a file."""
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)

        model         = cls(model_data['token_vocab_size'],
                            model_data['unit_vocab_size'],
                            model_data['embed_dim'])
        model.W_token = model_data['W_token']
        model.W_unit  = model_data['W_unit']

        print(f"FastText-style Token-Unit model loaded: {file_path}")
        return model

    def get_model_info(self) -> dict:
        """Return model information as a dictionary."""
        return {
            'model_type'          : 'fasttext_style_token_unit',
            'token_vocab_size'    : self.token_vocab_size,
            'unit_vocab_size'     : self.unit_vocab_size,
            'embed_dim'           : self.embed_dim,
            'total_parameters'    : (self.token_vocab_size + self.unit_vocab_size) * self.embed_dim,
            'token_matrix_shape'  : self.W_token.shape,
            'unit_matrix_shape'   : self.W_unit.shape,
            'token_matrix_norm'   : np.linalg.norm(self.W_token),
            'unit_matrix_norm'    : np.linalg.norm(self.W_unit),
            'unit_embedding_method': 'fasttext_style_sum'
        }