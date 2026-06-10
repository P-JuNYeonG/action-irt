"""
유틸리티 함수들 (FastText 스타일)
FastText 방식 Token-Unit 임베딩 분석 및 기타 도우미 함수들
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
    FastText 스타일 Token-Unit 임베딩을 분석하는 클래스
    
    Attributes:
        model (Word2VecModel): 학습된 FastText 스타일 Word2Vec 모델
        vocab_builder (VocabularyBuilder): 어휘 구축기
        fasttext_cache (Dict): FastText 방식 유닛 임베딩 캐시
    """
    
    def __init__(self, model: Word2VecModel, vocab_builder: VocabularyBuilder):
        """
        FastText 스타일 임베딩 분석기 초기화
        
        Parameters:
            model (Word2VecModel): 학습된 FastText 스타일 Word2Vec 모델
            vocab_builder (VocabularyBuilder): 어휘 구축기
        """
        self.model = model
        self.vocab_builder = vocab_builder
        self.fasttext_cache = {}  # 성능을 위한 FastText 임베딩 캐시
        
        print("FastText 스타일 임베딩 분석기 초기화 완료")
    
    def get_fasttext_unit_embedding_cached(self, unit: str) -> Optional[np.ndarray]:
        """
        캐시를 사용하여 FastText 방식 유닛 임베딩을 효율적으로 계산합니다.
        
        Parameters:
            unit (str): 유닛명 (예: 'button_next')
            
        Returns:
            Optional[np.ndarray]: FastText 방식 유닛 임베딩 벡터
        """
        if unit in self.fasttext_cache:
            return self.fasttext_cache[unit]
        
        fasttext_embedding = self.model.get_fasttext_unit_embedding(unit, self.vocab_builder)
        
        if fasttext_embedding is not None:
            self.fasttext_cache[unit] = fasttext_embedding
        
        return fasttext_embedding
    
    def analyze_compositional_semantics(self, units: List[str]) -> Dict:
        """
        FastText 방식 compositional semantics를 분석합니다.
        
        Parameters:
            units (List[str]): 분석할 유닛들
            
        Returns:
            Dict: FastText 방식 compositional semantics 분석 결과
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
                continue  # 구성요소가 1개 이하면 분석하지 않음
            
            # FastText 방식 유닛 임베딩
            fasttext_embedding = self.get_fasttext_unit_embedding_cached(unit)
            if fasttext_embedding is None:
                continue
            
            # 각 구성요소의 기여도 분석
            component_contributions = {}
            component_similarities = {}
            individual_embeddings = []
            
            for comp_type, idx in composition_indices:
                if comp_type == 'token':
                    comp_name = f"token:{self.vocab_builder.get_token_by_index(idx)}"
                    comp_embedding = self.model.get_token_embedding(idx)
                elif comp_type == 'unit':
                    comp_name = f"unit:{self.vocab_builder.get_unit_by_index(idx)}"
                    comp_embedding = self.model.get_unit_embedding(idx)
                else:
                    continue
                
                individual_embeddings.append(comp_embedding)
                
                # 구성요소와 FastText 임베딩 간의 유사도
                comp_norm = np.linalg.norm(comp_embedding) #구성요소
                fasttext_norm = np.linalg.norm(fasttext_embedding) #전체유닛
                if comp_norm > 0 and fasttext_norm > 0:
                    similarity = np.dot(comp_embedding, fasttext_embedding) / (comp_norm * fasttext_norm)
                    component_similarities[comp_name] = float(similarity)
                    
                    # 평균에서의 기여도 (1/n의 가중치로 계산)
                    contribution = similarity / len(composition_indices)
                    component_contributions[comp_name] = float(contribution)
            
            # 평균 임베딩과 FastText 임베딩 비교
            coherence_score = 0.0
            if individual_embeddings:
                sum_embedding = np.sum(individual_embeddings, axis=0)
                sum_norm = np.linalg.norm(sum_embedding)
                fasttext_norm = np.linalg.norm(fasttext_embedding)
                
                if sum_norm > 0 and fasttext_norm > 0:
                    coherence_score = np.dot(sum_embedding, fasttext_embedding) / (sum_norm * fasttext_norm)
            
            analysis_results['fasttext_composition_analysis'][unit] = {
                'composition_formula': f"sum([{', '.join(component_similarities.keys())}])",
                'component_similarities': component_similarities,
                'component_contributions': component_contributions,
                'average_contribution': float(np.mean(list(component_contributions.values()))) if component_contributions else 0.0,
                'coherence_with_sum': float(coherence_score)
            }
        
        return analysis_results
    
    def demonstrate_fasttext_composition(self, unit: str) -> Dict:
        """
        특정 유닛에 대한 FastText 방식 구성을 시연합니다.
        
        Parameters:
            unit (str): 시연할 유닛명
            
        Returns:
            Dict: FastText 방식 구성 시연 결과
        """
        if not self.vocab_builder.is_valid_item(unit) or '_' not in unit:
            return {'error': f"유효하지 않은 유닛: {unit}"}
        
        composition_indices = self.vocab_builder.get_unit_composition(unit)
        if not composition_indices:
            return {'error': f"구성요소를 찾을 수 없음: {unit}"}
        
        demonstration = {
            'unit': unit,
            'fasttext_formula': f"{unit} = sum([구성요소들])",
            'components': [],
            'individual_embeddings': [],
            'fasttext_embedding': None,
            'embedding_comparison': {}
        }
        
        # 구성요소별 임베딩 수집
        individual_embeddings = []
        for comp_type, idx in composition_indices:
            if comp_type == 'token':
                comp_name = self.vocab_builder.get_token_by_index(idx)
                comp_embedding = self.model.get_token_embedding(idx)
                comp_full_name = f"token:{comp_name}"
            elif comp_type == 'unit':
                comp_name = self.vocab_builder.get_unit_by_index(idx)
                comp_embedding = self.model.get_unit_embedding(idx)
                comp_full_name = f"unit:{comp_name}"
            else:
                continue
            
            demonstration['components'].append({
                'name': comp_full_name,
                'embedding_norm': float(np.linalg.norm(comp_embedding)),
                'embedding_shape': comp_embedding.shape
            })
            individual_embeddings.append(comp_embedding)
        
        # FastText 방식 임베딩 계산
        fasttext_embedding = self.get_fasttext_unit_embedding_cached(unit)
        if fasttext_embedding is not None:
            demonstration['fasttext_embedding'] = {
                'embedding_norm': float(np.linalg.norm(fasttext_embedding)),
                'embedding_shape': fasttext_embedding.shape
            }
            
            # 수동 계산한 평균과 비교
            if individual_embeddings:
                manual_sum = np.sum(individual_embeddings, axis=0)
                
                # 두 임베딩이 동일한지 확인 (수치 오차 고려)
                is_identical = np.allclose(fasttext_embedding, manual_sum, rtol=1e-10)
                difference_norm = np.linalg.norm(fasttext_embedding - manual_sum)
                
                demonstration['embedding_comparison'] = {
                    'manual_sum_norm': float(np.linalg.norm(manual_sum)),
                    'fasttext_norm': float(np.linalg.norm(fasttext_embedding)),
                    'is_identical': is_identical,
                    'difference_norm': float(difference_norm),
                    'verification': 'FastText 방식이 수동 평균과 일치함' if is_identical else 'FastText 방식과 수동 평균이 다름'
                }
        
        return demonstration
    
    def compare_token_vs_fasttext_unit(self, token: str, unit: str) -> Dict:
        """
        토큰과 FastText 방식 유닛 임베딩을 비교합니다.
        
        Parameters:
            token (str): 비교할 토큰
            unit (str): 비교할 유닛
            
        Returns:
            Dict: 비교 결과
        """
        comparison = {
            'token': token,
            'unit': unit,
            'token_embedding': None,
            'fasttext_unit_embedding': None,
            'similarity': 0.0,
            'analysis': {}
        }
        
        # 토큰 임베딩
        token_embedding = self.model.get_item_embedding(token, self.vocab_builder)
        if token_embedding is not None:
            comparison['token_embedding'] = {
                'norm': float(np.linalg.norm(token_embedding)),
                'shape': token_embedding.shape
            }
        
        # FastText 방식 유닛 임베딩
        fasttext_embedding = self.get_fasttext_unit_embedding_cached(unit)
        if fasttext_embedding is not None:
            comparison['fasttext_unit_embedding'] = {
                'norm': float(np.linalg.norm(fasttext_embedding)),
                'shape': fasttext_embedding.shape
            }
        
        # 유사도 계산
        if token_embedding is not None and fasttext_embedding is not None:
            similarity = self.model.get_similarity(token, unit, self.vocab_builder)
            comparison['similarity'] = float(similarity)
            
            # 유닛이 해당 토큰을 포함하는지 확인
            unit_tokens = unit.split('_')
            contains_token = token in unit_tokens
            
            comparison['analysis'] = {
                'unit_contains_token': contains_token,
                'unit_composition': unit_tokens,
                'expected_high_similarity': contains_token,
                'similarity_interpretation': self._interpret_similarity(similarity, contains_token)
            }
        
        return comparison
    
    def _interpret_similarity(self, similarity: float, contains_token: bool) -> str:
        """유사도 값을 해석합니다."""
        if contains_token:
            if similarity > 0.7:
                return "높은 유사도: 토큰이 유닛 구성에 강하게 기여"
            elif similarity > 0.4:
                return "중간 유사도: 토큰이 유닛 구성에 부분적으로 기여"
            else:
                return "낮은 유사도: 토큰이 유닛에 포함되지만 기여도 낮음"
        else:
            if similarity > 0.5:
                return "예상보다 높은 유사도: 의미적 연관성 존재"
            elif similarity > 0.2:
                return "중간 유사도: 약간의 의미적 연관성"
            else:
                return "낮은 유사도: 의미적 연관성 낮음"
    
    def analyze_unit_complexity(self, units: List[str]) -> Dict:
        """
        유닛들의 복잡도를 분석합니다 (구성요소 수, FastText 임베딩 특성 등).
        
        Parameters:
            units (List[str]): 분석할 유닛들
            
        Returns:
            Dict: 복잡도 분석 결과
        """
        complexity_analysis = {
            'unit_complexity_stats': {},
            'complexity_distribution': {
                'simple_units': [],  # 구성요소 2개
                'moderate_units': [],  # 구성요소 3-4개
                'complex_units': []  # 구성요소 5개 이상
            },
            'embedding_characteristics': {}
        }
        
        for unit in units:
            if not self.vocab_builder.is_valid_item(unit) or '_' not in unit:
                continue
            
            composition_indices = self.vocab_builder.get_unit_composition(unit)
            num_components = len(composition_indices)
            
            fasttext_embedding = self.get_fasttext_unit_embedding_cached(unit)
            if fasttext_embedding is None:
                continue
            
            # 복잡도 통계
            unit_stats = {
                'num_components': num_components,
                'num_tokens': sum(1 for comp_type, _ in composition_indices if comp_type == 'token'),
                'num_units': sum(1 for comp_type, _ in composition_indices if comp_type == 'unit'),
                'embedding_norm': float(np.linalg.norm(fasttext_embedding))
            }
            
            complexity_analysis['unit_complexity_stats'][unit] = unit_stats
            
            # 복잡도별 분류
            if num_components == 2:
                complexity_analysis['complexity_distribution']['simple_units'].append(unit)
            elif num_components <= 4:
                complexity_analysis['complexity_distribution']['moderate_units'].append(unit)
            else:
                complexity_analysis['complexity_distribution']['complex_units'].append(unit)
        
        # 전체 통계
        all_norms = [stats['embedding_norm'] for stats in complexity_analysis['unit_complexity_stats'].values()]
        all_components = [stats['num_components'] for stats in complexity_analysis['unit_complexity_stats'].values()]
        
        if all_norms:
            complexity_analysis['embedding_characteristics'] = {
                'mean_embedding_norm': float(np.mean(all_norms)),
                'std_embedding_norm': float(np.std(all_norms)),
                'mean_num_components': float(np.mean(all_components)),
                'std_num_components': float(np.std(all_components)),
                'norm_vs_complexity_correlation': float(np.corrcoef(all_norms, all_components)[0, 1]) if len(all_norms) > 1 else 0.0
            }
        
        return complexity_analysis
    
    def save_comprehensive_fasttext_analysis(self, output_file: str, sample_size: int = 50) -> None:
        """
        종합적인 FastText 스타일 분석 결과를 파일로 저장합니다.
        
        Parameters:
            output_file (str): 결과 저장 파일 경로
            sample_size (int): 분석할 샘플 크기
        """
        # 샘플 데이터 준비
        top_tokens = [token for token, _ in self.vocab_builder.token_counts.most_common(15)]
        top_units = [unit for unit, _ in self.vocab_builder.unit_counts.most_common(sample_size) 
                    if self.vocab_builder.is_valid_item(unit) and '_' in unit]
        
        comprehensive_analysis = {
            'model_info': {
                'model_type': 'fasttext_style_token_unit',
                'token_vocab_size': self.model.token_vocab_size,
                'unit_vocab_size': self.model.unit_vocab_size,
                'embed_dim': self.model.embed_dim,
                'fasttext_formula': 'unit_embedding = sum([token_embeddings, unit_embedding])'
            },
            'fasttext_demonstrations': {},
            'compositional_semantics': self.analyze_compositional_semantics(top_units),
            'token_unit_comparisons': {},
            'unit_complexity_analysis': self.analyze_unit_complexity(top_units),
            'sample_statistics': {
                'analyzed_tokens': len(top_tokens),
                'analyzed_units': len(top_units),
                'total_demonstrations': min(5, len(top_units))
            }
        }
        
        # FastText 방식 시연
        for unit in top_units[:5]:
            demonstration = self.demonstrate_fasttext_composition(unit)
            comprehensive_analysis['fasttext_demonstrations'][unit] = demonstration
        
        # 토큰-유닛 비교
        for token in top_tokens[:5]:
            token_comparisons = []
            for unit in top_units[:3]:
                comparison = self.compare_token_vs_fasttext_unit(token, unit)
                token_comparisons.append(comparison)
            comprehensive_analysis['token_unit_comparisons'][token] = token_comparisons
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_analysis, f, ensure_ascii=False, indent=2)
        
        print(f"FastText 스타일 종합 분석 결과 저장 완료: {output_file}")
        print(f"  - FastText 시연: {len(comprehensive_analysis['fasttext_demonstrations'])}개")
        print(f"  - 토큰-유닛 비교: {len(comprehensive_analysis['token_unit_comparisons'])}개")
        print(f"  - 복잡도 분석: {len(comprehensive_analysis['unit_complexity_analysis']['unit_complexity_stats'])}개")
    
    def clear_cache(self) -> None:
        """FastText 임베딩 캐시를 삭제합니다."""
        self.fasttext_cache.clear()
        print("FastText 임베딩 캐시가 삭제되었습니다.")


def save_token_unit_embeddings_tsv(model: Word2VecModel, vocab_builder: VocabularyBuilder,
                                   token_embeddings_file: str, token_metadata_file: str,
                                   unit_embeddings_file: str, unit_metadata_file: str) -> None:
    """
    FastText 스타일 Token과 Unit 임베딩을 TensorBoard Projector용 TSV 형식으로 저장합니다.
    
    Parameters:
        model (Word2VecModel): 학습된 모델
        vocab_builder (VocabularyBuilder): 어휘 구축기
        token_embeddings_file (str): 토큰 임베딩 파일 경로 (.tsv)
        token_metadata_file (str): 토큰 메타데이터 파일 경로 (.tsv)
        unit_embeddings_file (str): 유닛 임베딩 파일 경로 (.tsv)
        unit_metadata_file (str): 유닛 메타데이터 파일 경로 (.tsv)
    """
    # 토큰 임베딩 TSV 저장
    with open(token_embeddings_file, 'w', encoding='utf-8') as emb_f, \
         open(token_metadata_file, 'w', encoding='utf-8') as meta_f:
        
        # 메타데이터 헤더
        meta_f.write("Token\tFrequency\tType\n")
        
        for idx in range(vocab_builder.vocab_size):
            token = vocab_builder.get_token_by_index(idx)
            embedding = model.get_token_embedding(idx)
            frequency = vocab_builder.token_counts[token]
            
            # 임베딩 저장 (탭으로 구분)
            emb_f.write('\t'.join(map(str, embedding)) + '\n')
            
            # 메타데이터 저장
            meta_f.write(f"{token}\t{frequency}\ttoken\n")
    
    # 유닛 임베딩 TSV 저장 (FastText 방식)
    with open(unit_embeddings_file, 'w', encoding='utf-8') as emb_f, \
         open(unit_metadata_file, 'w', encoding='utf-8') as meta_f:
        
        # 메타데이터 헤더
        meta_f.write("Unit\tFrequency\tType\tComponents\n")
        
        for idx in range(vocab_builder.unit_vocab_size):
            unit = vocab_builder.get_unit_by_index(idx)
            # FastText 방식으로 유닛 임베딩 계산
            embedding = model.get_fasttext_unit_embedding(unit, vocab_builder)
            if embedding is not None:
                frequency = vocab_builder.unit_counts[unit]
                components = len(vocab_builder.get_unit_composition(unit))
                
                # 임베딩 저장 (탭으로 구분)
                emb_f.write('\t'.join(map(str, embedding)) + '\n')
                
                # 메타데이터 저장
                meta_f.write(f"{unit}\t{frequency}\tfasttext_unit\t{components}\n")
    
    print(f"FastText 스타일 TensorBoard Projector용 파일 저장 완료:")
    print(f"  토큰 임베딩: {token_embeddings_file}")
    print(f"  토큰 메타데이터: {token_metadata_file}")
    print(f"  유닛 임베딩: {unit_embeddings_file} (FastText 방식)")
    print(f"  유닛 메타데이터: {unit_metadata_file}")


def extract_all_action_units(sequences: List[str]) -> List[str]:
    """액션 시퀀스들에서 모든 고유한 액션 유닛을 추출합니다."""
    all_units = set()
    
    for sequence in sequences:
        units = sequence.strip().split()
        all_units.update(units)
    
    return sorted(list(all_units))


def extract_all_action_tokens(sequences: List[str]) -> List[str]:
    """액션 시퀀스들에서 모든 고유한 액션 토큰을 추출합니다."""
    all_tokens = set()
    
    for sequence in sequences:
        units = sequence.strip().split()
        for unit in units:
            tokens = unit.split('_')
            all_tokens.update(tokens)
    
    return sorted(list(all_tokens))


def demonstrate_fasttext_capabilities(model: Word2VecModel, vocab_builder: VocabularyBuilder) -> None:
    """
    FastText 스타일 모델의 capabilities를 시연합니다.
    
    Parameters:
        model (Word2VecModel): 학습된 모델
        vocab_builder (VocabularyBuilder): 어휘 구축기
    """
    print("=" * 60)
    print("FastText 스타일 모델 Capabilities 시연")
    print("=" * 60)
    
    analyzer = FastTextStyleEmbeddingAnalyzer(model, vocab_builder)
    
    # 1. FastText 방식 임베딩 계산 시연
    print("\n1. FastText 방식 유닛 임베딩 계산")
    sample_units = []
    for idx in range(min(5, vocab_builder.unit_vocab_size)):
        unit = vocab_builder.get_unit_by_index(idx)
        if unit and vocab_builder.is_valid_item(unit) and '_' in unit:
            sample_units.append(unit)
    
    for unit in sample_units[:3]:
        composition = vocab_builder.get_unit_composition(unit)
        component_names = []
        for comp_type, idx in composition:
            if comp_type == 'token':
                component_names.append(f"token:{vocab_builder.get_token_by_index(idx)}")
            elif comp_type == 'unit':
                component_names.append(f"unit:{vocab_builder.get_unit_by_index(idx)}")
        
        fasttext_embedding = model.get_fasttext_unit_embedding(unit, vocab_builder)
        if fasttext_embedding is not None:
            print(f"  '{unit}' = sum([{', '.join(component_names)}])")
            print(f"    → 임베딩 크기: {fasttext_embedding.shape}, 노름: {np.linalg.norm(fasttext_embedding):.4f}")
    
    # 2. FastText 방식 시연
    print("\n2. FastText 방식 구성 시연")
    if sample_units:
        demonstration = analyzer.demonstrate_fasttext_composition(sample_units[0])
        print(f"  유닛: {demonstration.get('unit', 'N/A')}")
        print(f"  공식: {demonstration.get('fasttext_formula', 'N/A')}")
        if 'embedding_comparison' in demonstration:
            comp = demonstration['embedding_comparison']
            print(f"  검증: {comp.get('verification', 'N/A')}")
    
    # 3. Compositional semantics 분석
    print("\n3. Compositional Semantics 분석")
    if len(sample_units) >= 2:
        composition_analysis = analyzer.analyze_compositional_semantics(sample_units[:2])
        
        for unit, analysis in composition_analysis.get('fasttext_composition_analysis', {}).items():
            print(f"  '{unit}' 구성 분석:")
            print(f"    공식: {analysis.get('composition_formula', 'N/A')}")
            print(f"    평균 기여도: {analysis.get('average_contribution', 0):.4f}")
            print(f"    평균과의 일관성: {analysis.get('coherence_with_sum', 0):.4f}")
    
    # 4. 토큰-유닛 비교
    print("\n4. 토큰-유닛 유사도 비교")
    top_tokens = []
    for idx in range(min(3, vocab_builder.vocab_size)):
        token = vocab_builder.get_token_by_index(idx)
        if token:
            top_tokens.append(token)
    
    for token in top_tokens:
        for unit in sample_units[:2]:
            similarity = model.get_similarity(token, unit, vocab_builder)
            unit_tokens = unit.split('_')
            contains = token in unit_tokens
            print(f"  '{token}' ↔ '{unit}': {similarity:.4f} {'(포함됨)' if contains else '(미포함)'}")
    
    print(f"\n5. FastText 스타일 모델의 특징:")
    print(f"  ✓ 유닛 임베딩 = sum([구성_토큰들, 유닛_자체])")
    print(f"  ✓ Compositional + Holistic semantics")
    print(f"  ✓ 그래디언트 FastText 방식 분배")
    print(f"  ✓ 토큰과 유닛 모두 중심 아이템으로 학습")


# 사용 예시
if __name__ == "__main__":
    # 테스트용 간단한 데이터
    test_sequences = [
        "start button_next toolbar_menu end",
        "click button_save file_dialog close",
        "open menu_file select_option confirm_dialog"
    ]
    
    # 모든 액션 유닛과 토큰 추출
    all_units = extract_all_action_units(test_sequences)
    all_tokens = extract_all_action_tokens(test_sequences)
    
    print(f"추출된 액션 유닛들: {all_units}")
    print(f"추출된 액션 토큰들: {all_tokens}")
    
    print("\n실제 사용 예시:")
    print("# 학습된 FastText 스타일 모델 로드")
    print("model = Word2VecModel.load_model('fasttext_model.pkl')")
    print("vocab_builder = VocabularyBuilder()")
    print("# ... 어휘 구축 ...")
    print()
    print("# FastText 스타일 분석기 초기화")
    print("analyzer = FastTextStyleEmbeddingAnalyzer(model, vocab_builder)")
    print()
    print("# FastText 방식 유닛 임베딩")
    print("fasttext_embedding = model.get_fasttext_unit_embedding('button_next', vocab_builder)")
    print("# = sum([button_token_emb, next_token_emb, button_next_unit_emb])")
    print()
    print("# FastText 방식 구성 시연")
    print("demonstration = analyzer.demonstrate_fasttext_composition('button_next')")
    print()
    print("# Compositional semantics 분석")
    print("composition_analysis = analyzer.analyze_compositional_semantics(['button_next'])")
    print()
    print("# 종합 분석")
    print("analyzer.save_comprehensive_fasttext_analysis('fasttext_analysis.json')")
    print()
    print("# 모델 capabilities 시연")
    print("demonstrate_fasttext_capabilities(model, vocab_builder)")