"""
Utility Functions (FastText Style)
FastText-style Token-Unit embedding analysis and helper functions.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import json
from collections import defaultdict

from model import Word2VecModel
from vocab_builder import VocabularyBuilder


class FastTextStyleEmbeddingAnalyzer:
    """
    Class for analyzing FastText-style Token-Unit embeddings.

    Attributes:
        model (Word2VecModel):        Trained FastText-style Word2Vec model.
        vocab_builder (VocabularyBuilder): Vocabulary builder.
        fasttext_cache (Dict):        Cache for FastText-style unit embeddings.
    """

    def __init__(self, model: Word2VecModel, vocab_builder: VocabularyBuilder):
        """
        Initialize the FastText-style embedding analyzer.

        Parameters:
            model (Word2VecModel):          Trained FastText-style Word2Vec model.
            vocab_builder (VocabularyBuilder): Vocabulary builder.
        """
        self.model         = model
        self.vocab_builder = vocab_builder
        self.fasttext_cache = {}  # cache for FastText embeddings

        print("FastText-style embedding analyzer initialized.")

    def get_fasttext_unit_embedding_cached(self, unit: str) -> Optional[np.ndarray]:
        """
        Efficiently compute the FastText-style unit embedding using a cache.

        Parameters:
            unit (str): Unit name (e.g., 'button_next').

        Returns:
            Optional[np.ndarray]: FastText-style unit embedding vector.
        """
        if unit in self.fasttext_cache:
            return self.fasttext_cache[unit]

        fasttext_embedding = self.model.get_fasttext_unit_embedding(unit, self.vocab_builder)

        if fasttext_embedding is not None:
            self.fasttext_cache[unit] = fasttext_embedding

        return fasttext_embedding

    def analyze_compositional_semantics(self, units: List[str]) -> Dict:
        """
        Analyze FastText-style compositional semantics.

        Parameters:
            units (List[str]): Units to analyze.

        Returns:
            Dict: FastText-style compositional semantics analysis results.
        """
        analysis_results = {
            'fasttext_composition_analysis': {},
            'component_contribution_analysis': {},
            'semantic_coherence': {}
        }

        for unit in units:
            if not self.vocab_builder.is_valid_item(unit) or '_' not in unit:
                continue

            composition_indices = self.vocab_builder.get_unit_composition(unit)
            if len(composition_indices) < 2:
                continue  # Skip units with fewer than 2 components

            fasttext_embedding = self.get_fasttext_unit_embedding_cached(unit)
            if fasttext_embedding is None:
                continue

            component_contributions = {}
            component_similarities  = {}
            individual_embeddings   = []

            for comp_type, idx in composition_indices:
                if comp_type == 'token':
                    comp_name      = f"token:{self.vocab_builder.get_token_by_index(idx)}"
                    comp_embedding = self.model.get_token_embedding(idx)
                elif comp_type == 'unit':
                    comp_name      = f"unit:{self.vocab_builder.get_unit_by_index(idx)}"
                    comp_embedding = self.model.get_unit_embedding(idx)
                else:
                    continue

                individual_embeddings.append(comp_embedding)

                comp_norm     = np.linalg.norm(comp_embedding)
                fasttext_norm = np.linalg.norm(fasttext_embedding)
                if comp_norm > 0 and fasttext_norm > 0:
                    similarity = np.dot(comp_embedding, fasttext_embedding) / (comp_norm * fasttext_norm)
                    component_similarities[comp_name] = float(similarity)

                    # Contribution weighted by 1/n
                    contribution = similarity / len(composition_indices)
                    component_contributions[comp_name] = float(contribution)

            coherence_score = 0.0
            if individual_embeddings:
                sum_embedding = np.sum(individual_embeddings, axis=0)
                sum_norm      = np.linalg.norm(sum_embedding)
                fasttext_norm = np.linalg.norm(fasttext_embedding)

                if sum_norm > 0 and fasttext_norm > 0:
                    coherence_score = np.dot(sum_embedding, fasttext_embedding) / (sum_norm * fasttext_norm)

            analysis_results['fasttext_composition_analysis'][unit] = {
                'composition_formula'  : f"sum([{', '.join(component_similarities.keys())}])",
                'component_similarities': component_similarities,
                'component_contributions': component_contributions,
                'average_contribution' : float(np.mean(list(component_contributions.values()))) if component_contributions else 0.0,
                'coherence_with_sum'   : float(coherence_score)
            }

        return analysis_results

    def demonstrate_fasttext_composition(self, unit: str) -> Dict:
        """
        Demonstrate the FastText-style composition for a specific unit.

        Parameters:
            unit (str): Unit name to demonstrate.

        Returns:
            Dict: FastText-style composition demonstration results.
        """
        if not self.vocab_builder.is_valid_item(unit) or '_' not in unit:
            return {'error': f"Invalid unit: {unit}"}

        composition_indices = self.vocab_builder.get_unit_composition(unit)
        if not composition_indices:
            return {'error': f"No composition found for: {unit}"}

        demonstration = {
            'unit'              : unit,
            'fasttext_formula'  : f"{unit} = sum([components])",
            'components'        : [],
            'individual_embeddings': [],
            'fasttext_embedding': None,
            'embedding_comparison': {}
        }

        individual_embeddings = []
        for comp_type, idx in composition_indices:
            if comp_type == 'token':
                comp_name      = self.vocab_builder.get_token_by_index(idx)
                comp_embedding = self.model.get_token_embedding(idx)
                comp_full_name = f"token:{comp_name}"
            elif comp_type == 'unit':
                comp_name      = self.vocab_builder.get_unit_by_index(idx)
                comp_embedding = self.model.get_unit_embedding(idx)
                comp_full_name = f"unit:{comp_name}"
            else:
                continue

            demonstration['components'].append({
                'name'           : comp_full_name,
                'embedding_norm' : float(np.linalg.norm(comp_embedding)),
                'embedding_shape': comp_embedding.shape
            })
            individual_embeddings.append(comp_embedding)

        fasttext_embedding = self.get_fasttext_unit_embedding_cached(unit)
        if fasttext_embedding is not None:
            demonstration['fasttext_embedding'] = {
                'embedding_norm' : float(np.linalg.norm(fasttext_embedding)),
                'embedding_shape': fasttext_embedding.shape
            }

            # Verify against manually computed sum
            if individual_embeddings:
                manual_sum = np.sum(individual_embeddings, axis=0)

                # Check identity up to numerical precision
                is_identical    = np.allclose(fasttext_embedding, manual_sum, rtol=1e-10)
                difference_norm = np.linalg.norm(fasttext_embedding - manual_sum)

                demonstration['embedding_comparison'] = {
                    'manual_sum_norm' : float(np.linalg.norm(manual_sum)),
                    'fasttext_norm'   : float(np.linalg.norm(fasttext_embedding)),
                    'is_identical'    : is_identical,
                    'difference_norm' : float(difference_norm),
                    'verification'    : (
                        'FastText embedding matches manual sum.'
                        if is_identical else
                        'FastText embedding differs from manual sum.'
                    )
                }

        return demonstration

    def compare_token_vs_fasttext_unit(self, token: str, unit: str) -> Dict:
        """
        Compare a token embedding with a FastText-style unit embedding.

        Parameters:
            token (str): Token to compare.
            unit (str):  Unit to compare.

        Returns:
            Dict: Comparison results.
        """
        comparison = {
            'token'                   : token,
            'unit'                    : unit,
            'token_embedding'         : None,
            'fasttext_unit_embedding' : None,
            'similarity'              : 0.0,
            'analysis'                : {}
        }

        token_embedding = self.model.get_item_embedding(token, self.vocab_builder)
        if token_embedding is not None:
            comparison['token_embedding'] = {
                'norm' : float(np.linalg.norm(token_embedding)),
                'shape': token_embedding.shape
            }

        fasttext_embedding = self.get_fasttext_unit_embedding_cached(unit)
        if fasttext_embedding is not None:
            comparison['fasttext_unit_embedding'] = {
                'norm' : float(np.linalg.norm(fasttext_embedding)),
                'shape': fasttext_embedding.shape
            }

        if token_embedding is not None and fasttext_embedding is not None:
            similarity     = self.model.get_similarity(token, unit, self.vocab_builder)
            comparison['similarity'] = float(similarity)

            unit_tokens    = unit.split('_')
            contains_token = token in unit_tokens

            comparison['analysis'] = {
                'unit_contains_token'       : contains_token,
                'unit_composition'          : unit_tokens,
                'expected_high_similarity'  : contains_token,
                'similarity_interpretation' : self._interpret_similarity(similarity, contains_token)
            }

        return comparison

    def _interpret_similarity(self, similarity: float, contains_token: bool) -> str:
        """Interpret a similarity value."""
        if contains_token:
            if similarity > 0.7:
                return "High similarity: token contributes strongly to unit composition"
            elif similarity > 0.4:
                return "Moderate similarity: token partially contributes to unit composition"
            else:
                return "Low similarity: token is part of unit but contributes weakly"
        else:
            if similarity > 0.5:
                return "Unexpectedly high similarity: semantic relation exists"
            elif similarity > 0.2:
                return "Moderate similarity: slight semantic relation"
            else:
                return "Low similarity: weak semantic relation"

    def analyze_unit_complexity(self, units: List[str]) -> Dict:
        """
        Analyze the complexity of units (number of components, FastText embedding characteristics, etc.).

        Parameters:
            units (List[str]): Units to analyze.

        Returns:
            Dict: Complexity analysis results.
        """
        complexity_analysis = {
            'unit_complexity_stats': {},
            'complexity_distribution': {
                'simple_units'  : [],   # exactly 2 components
                'moderate_units': [],   # 3-4 components
                'complex_units' : []    # 5 or more components
            },
            'embedding_characteristics': {}
        }

        for unit in units:
            if not self.vocab_builder.is_valid_item(unit) or '_' not in unit:
                continue

            composition_indices = self.vocab_builder.get_unit_composition(unit)
            num_components      = len(composition_indices)

            fasttext_embedding = self.get_fasttext_unit_embedding_cached(unit)
            if fasttext_embedding is None:
                continue

            unit_stats = {
                'num_components' : num_components,
                'num_tokens'     : sum(1 for comp_type, _ in composition_indices if comp_type == 'token'),
                'num_units'      : sum(1 for comp_type, _ in composition_indices if comp_type == 'unit'),
                'embedding_norm' : float(np.linalg.norm(fasttext_embedding))
            }

            complexity_analysis['unit_complexity_stats'][unit] = unit_stats

            if num_components == 2:
                complexity_analysis['complexity_distribution']['simple_units'].append(unit)
            elif num_components <= 4:
                complexity_analysis['complexity_distribution']['moderate_units'].append(unit)
            else:
                complexity_analysis['complexity_distribution']['complex_units'].append(unit)

        all_norms      = [s['embedding_norm'] for s in complexity_analysis['unit_complexity_stats'].values()]
        all_components = [s['num_components']  for s in complexity_analysis['unit_complexity_stats'].values()]

        if all_norms:
            complexity_analysis['embedding_characteristics'] = {
                'mean_embedding_norm'          : float(np.mean(all_norms)),
                'std_embedding_norm'           : float(np.std(all_norms)),
                'mean_num_components'          : float(np.mean(all_components)),
                'std_num_components'           : float(np.std(all_components)),
                'norm_vs_complexity_correlation': float(np.corrcoef(all_norms, all_components)[0, 1]) if len(all_norms) > 1 else 0.0
            }

        return complexity_analysis

    def save_comprehensive_fasttext_analysis(self, output_file: str,
                                             sample_size: int = 50) -> None:
        """
        Save a comprehensive FastText-style analysis to a file.

        Parameters:
            output_file (str): Path to the output file.
            sample_size (int): Number of top units to analyze.
        """
        top_tokens = [token for token, _ in self.vocab_builder.token_counts.most_common(15)]
        top_units  = [unit for unit, _ in self.vocab_builder.unit_counts.most_common(sample_size)
                      if self.vocab_builder.is_valid_item(unit) and '_' in unit]

        comprehensive_analysis = {
            'model_info': {
                'model_type'      : 'fasttext_style_token_unit',
                'token_vocab_size': self.model.token_vocab_size,
                'unit_vocab_size' : self.model.unit_vocab_size,
                'embed_dim'       : self.model.embed_dim,
                'fasttext_formula': 'unit_embedding = sum([token_embeddings, unit_embedding])'
            },
            'fasttext_demonstrations'  : {},
            'compositional_semantics'  : self.analyze_compositional_semantics(top_units),
            'token_unit_comparisons'   : {},
            'unit_complexity_analysis' : self.analyze_unit_complexity(top_units),
            'sample_statistics': {
                'analyzed_tokens'       : len(top_tokens),
                'analyzed_units'        : len(top_units),
                'total_demonstrations'  : min(5, len(top_units))
            }
        }

        # FastText composition demonstrations
        for unit in top_units[:5]:
            demonstration = self.demonstrate_fasttext_composition(unit)
            comprehensive_analysis['fasttext_demonstrations'][unit] = demonstration

        # Token-unit comparisons
        for token in top_tokens[:5]:
            token_comparisons = []
            for unit in top_units[:3]:
                comparison = self.compare_token_vs_fasttext_unit(token, unit)
                token_comparisons.append(comparison)
            comprehensive_analysis['token_unit_comparisons'][token] = token_comparisons

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_analysis, f, ensure_ascii=False, indent=2)

        print(f"Comprehensive FastText-style analysis saved: {output_file}")
        print(f"  - FastText demonstrations: {len(comprehensive_analysis['fasttext_demonstrations'])}")
        print(f"  - Token-unit comparisons:  {len(comprehensive_analysis['token_unit_comparisons'])}")
        print(f"  - Complexity analyses:     {len(comprehensive_analysis['unit_complexity_analysis']['unit_complexity_stats'])}")

    def clear_cache(self) -> None:
        """Clear the FastText embedding cache."""
        self.fasttext_cache.clear()
        print("FastText embedding cache cleared.")


def save_token_unit_embeddings_tsv(model: Word2VecModel, vocab_builder: VocabularyBuilder,
                                   token_embeddings_file: str, token_metadata_file: str,
                                   unit_embeddings_file: str, unit_metadata_file: str) -> None:
    """
    Save FastText-style token and unit embeddings in TensorBoard Projector TSV format.

    Parameters:
        model (Word2VecModel):          Trained model.
        vocab_builder (VocabularyBuilder): Vocabulary builder.
        token_embeddings_file (str):    Path to token embeddings file (.tsv).
        token_metadata_file (str):      Path to token metadata file (.tsv).
        unit_embeddings_file (str):     Path to unit embeddings file (.tsv).
        unit_metadata_file (str):       Path to unit metadata file (.tsv).
    """
    with open(token_embeddings_file, 'w', encoding='utf-8') as emb_f, \
         open(token_metadata_file,   'w', encoding='utf-8') as meta_f:

        meta_f.write("Token\tFrequency\tType\n")

        for idx in range(vocab_builder.vocab_size):
            token     = vocab_builder.get_token_by_index(idx)
            embedding = model.get_token_embedding(idx)
            frequency = vocab_builder.token_counts[token]

            emb_f.write('\t'.join(map(str, embedding)) + '\n')
            meta_f.write(f"{token}\t{frequency}\ttoken\n")

    with open(unit_embeddings_file, 'w', encoding='utf-8') as emb_f, \
         open(unit_metadata_file,   'w', encoding='utf-8') as meta_f:

        meta_f.write("Unit\tFrequency\tType\tComponents\n")

        for idx in range(vocab_builder.unit_vocab_size):
            unit      = vocab_builder.get_unit_by_index(idx)
            embedding = model.get_fasttext_unit_embedding(unit, vocab_builder)
            if embedding is not None:
                frequency    = vocab_builder.unit_counts[unit]
                components   = len(vocab_builder.get_unit_composition(unit))

                emb_f.write('\t'.join(map(str, embedding)) + '\n')
                meta_f.write(f"{unit}\t{frequency}\tfasttext_unit\t{components}\n")

    print(f"FastText-style TensorBoard Projector files saved:")
    print(f"  Token embeddings:  {token_embeddings_file}")
    print(f"  Token metadata:    {token_metadata_file}")
    print(f"  Unit embeddings:   {unit_embeddings_file} (FastText-style)")
    print(f"  Unit metadata:     {unit_metadata_file}")


def extract_all_action_units(sequences: List[str]) -> List[str]:
    """Extract all unique action units from action sequences."""
    all_units = set()
    for sequence in sequences:
        all_units.update(sequence.strip().split())
    return sorted(list(all_units))


def extract_all_action_tokens(sequences: List[str]) -> List[str]:
    """Extract all unique action tokens from action sequences."""
    all_tokens = set()
    for sequence in sequences:
        for unit in sequence.strip().split():
            all_tokens.update(unit.split('_'))
    return sorted(list(all_tokens))


def demonstrate_fasttext_capabilities(model: Word2VecModel,
                                      vocab_builder: VocabularyBuilder) -> None:
    """
    Demonstrate the capabilities of the FastText-style model.

    Parameters:
        model (Word2VecModel):          Trained model.
        vocab_builder (VocabularyBuilder): Vocabulary builder.
    """
    print("=" * 60)
    print("FastText-style Model Capabilities")
    print("=" * 60)

    analyzer = FastTextStyleEmbeddingAnalyzer(model, vocab_builder)

    # 1. FastText-style unit embedding computation
    print("\n1. FastText-style unit embedding computation")
    sample_units = []
    for idx in range(min(5, vocab_builder.unit_vocab_size)):
        unit = vocab_builder.get_unit_by_index(idx)
        if unit and vocab_builder.is_valid_item(unit) and '_' in unit:
            sample_units.append(unit)

    for unit in sample_units[:3]:
        composition     = vocab_builder.get_unit_composition(unit)
        component_names = []
        for comp_type, idx in composition:
            if comp_type == 'token':
                component_names.append(f"token:{vocab_builder.get_token_by_index(idx)}")
            elif comp_type == 'unit':
                component_names.append(f"unit:{vocab_builder.get_unit_by_index(idx)}")

        fasttext_embedding = model.get_fasttext_unit_embedding(unit, vocab_builder)
        if fasttext_embedding is not None:
            print(f"  '{unit}' = sum([{', '.join(component_names)}])")
            print(f"    → shape: {fasttext_embedding.shape}, norm: {np.linalg.norm(fasttext_embedding):.4f}")

    print("\n2. FastText-style composition demonstration")
    if sample_units:
        demonstration = analyzer.demonstrate_fasttext_composition(sample_units[0])
        print(f"  Unit:    {demonstration.get('unit', 'N/A')}")
        print(f"  Formula: {demonstration.get('fasttext_formula', 'N/A')}")
        if 'embedding_comparison' in demonstration:
            comp = demonstration['embedding_comparison']
            print(f"  Verify:  {comp.get('verification', 'N/A')}")

    print("\n3. Compositional semantics analysis")
    if len(sample_units) >= 2:
        composition_analysis = analyzer.analyze_compositional_semantics(sample_units[:2])

        for unit, analysis in composition_analysis.get('fasttext_composition_analysis', {}).items():
            print(f"  '{unit}' composition:")
            print(f"    Formula:       {analysis.get('composition_formula', 'N/A')}")
            print(f"    Avg contrib:   {analysis.get('average_contribution', 0):.4f}")
            print(f"    Sum coherence: {analysis.get('coherence_with_sum', 0):.4f}")

    print("\n4. Token-unit similarity comparison")
    top_tokens = []
    for idx in range(min(3, vocab_builder.vocab_size)):
        token = vocab_builder.get_token_by_index(idx)
        if token:
            top_tokens.append(token)

    for token in top_tokens:
        for unit in sample_units[:2]:
            similarity   = model.get_similarity(token, unit, vocab_builder)
            unit_tokens  = unit.split('_')
            contains     = token in unit_tokens
            print(f"  '{token}' ↔ '{unit}': {similarity:.4f} {'(contains)' if contains else '(absent)'}")

    print(f"\n5. FastText-style model characteristics:")
    print(f"  ✓ Unit embedding = sum([component tokens, unit itself])")
    print(f"  ✓ Compositional + Holistic semantics")
    print(f"  ✓ FastText-style gradient distribution")
    print(f"  ✓ Both tokens and units used as center items during training")