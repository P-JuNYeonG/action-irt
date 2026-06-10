"""
Word2Vec 모델 학습기 (FastText 스타일)
Token과 Unit 모두를 중심 아이템으로 사용하는 Adam 옵티마이저와 Negative Sampling
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import random

from model import Word2VecModel
from vocab_builder import VocabularyBuilder

"""
    FastText 스타일 모델을 위한 Adam 옵티마이저 구현
    토큰과 유닛 임베딩 모두 관리
"""

class AdamOptimizer:
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                beta2: float = 0.999, epsilon: float = 1e-8):
        """Adam 옵티마이저 초기화"""
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # 시간 스텝
        
        # 토큰과 유닛 임베딩용 모멘트 변수들
        self.m_token = None
        self.v_token = None
        self.m_unit = None
        self.v_unit = None
    
    def initialize(self, model: Word2VecModel) -> None:
        """모델에 맞춰 모멘트 변수들을 초기화합니다."""
        # 토큰 임베딩용 모멘트
        self.m_token = np.zeros_like(model.W_token)
        self.v_token = np.zeros_like(model.W_token)
        
        # 유닛 임베딩용 모멘트
        self.m_unit = np.zeros_like(model.W_unit)
        self.v_unit = np.zeros_like(model.W_unit)
        
        print("FastText 스타일 Adam 옵티마이저 초기화 완료")
    
    def update_token_embedding(self, token_idx: int, grad: np.ndarray, model: Word2VecModel) -> None:
        """토큰 임베딩에 대해 Adam 업데이트를 수행합니다."""
        # 1차 모멘트 업데이트
        self.m_token[token_idx] = self.beta1 * self.m_token[token_idx] + (1 - self.beta1) * grad
        
        # 2차 모멘트 업데이트
        self.v_token[token_idx] = self.beta2 * self.v_token[token_idx] + (1 - self.beta2) * (grad ** 2)
        
        # Bias correction 적용
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t
        
        m_corrected = self.m_token[token_idx] / bias_correction1
        v_corrected = self.v_token[token_idx] / bias_correction2
        
        # 파라미터 업데이트
        model.W_token[token_idx] -= (self.learning_rate * m_corrected / 
                                    (np.sqrt(v_corrected) + self.epsilon))
    
    def update_unit_embedding(self, unit_idx: int, grad: np.ndarray, model: Word2VecModel) -> None:
        """유닛 임베딩에 대해 Adam 업데이트를 수행합니다."""
        # 1차 모멘트 업데이트
        self.m_unit[unit_idx] = self.beta1 * self.m_unit[unit_idx] + (1 - self.beta1) * grad
        
        # 2차 모멘트 업데이트
        self.v_unit[unit_idx] = self.beta2 * self.v_unit[unit_idx] + (1 - self.beta2) * (grad ** 2)
        
        # Bias correction 적용
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t
        
        m_corrected = self.m_unit[unit_idx] / bias_correction1
        v_corrected = self.v_unit[unit_idx] / bias_correction2
        
        # 파라미터 업데이트
        model.W_unit[unit_idx] -= (self.learning_rate * m_corrected / 
                                (np.sqrt(v_corrected) + self.epsilon))


class Word2VecTrainer:
    """
    FastText 스타일 Token-Unit Word2Vec 모델 학습을 관리하는 클래스
    """
    
    def __init__(self, model: Word2VecModel, vocab_builder: VocabularyBuilder,
                learning_rate: float = 0.001, negative_samples: int = 5,
                early_stopping: bool = False, patience: int = 3):
        """
        FastText 스타일 학습기 초기화
        
        Parameters:
            model (Word2VecModel): 학습할 FastText 스타일 모델
            vocab_builder (VocabularyBuilder): 토큰과 유닛 어휘 구축기
            learning_rate (float): 학습률
            negative_samples (int): 음성 샘플 개수
        """
        self.model = model
        self.vocab_builder = vocab_builder
        self.negative_samples = negative_samples
        
        # Adam 옵티마이저 초기화
        self.optimizer = AdamOptimizer(learning_rate=learning_rate)
        self.optimizer.initialize(model)
        
        # 유닛에 대한 음성 샘플링을 위한 확률 분포 생성
        self.unit_sampling_distribution = vocab_builder.get_negative_sampling_distribution()
        
        # 빠른 음성 샘플링을 위한 누적 분포 테이블 생성
        self._build_negative_sampling_table()
        
        #Early stopping 관련 변수 추가
        self.early_stopping = early_stopping
        self.patience = patience
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        print(f"FastText 스타일 Word2Vec 학습기 초기화 완료:")
        print(f"  학습률: {learning_rate}")
        print(f"  음성 샘플 개수: {negative_samples}")
        print(f"  토큰 어휘 크기: {vocab_builder.vocab_size}")
        print(f"  유닛 어휘 크기: {vocab_builder.unit_vocab_size}")
        print(f"  Early stopping: {'활성화' if early_stopping else '비활성화'}")  # 추가
        if early_stopping:
            print(f"  Patience: {patience} 에포크")  # 추가
            
    
    def _build_negative_sampling_table(self, table_size: int = 1000000) -> None:
        """유닛에 대한 빠른 음성 샘플링을 위한 테이블을 구축합니다."""
        self.negative_sampling_table = np.zeros(table_size, dtype=np.int32)
        
        # 확률 분포에 따라 테이블 채우기
        unit_vocab_size = len(self.unit_sampling_distribution)
        table_idx = 0
        
        for unit_idx in range(unit_vocab_size):
            # 각 유닛이 차지할 테이블 슬롯 수 계산
            count = int(self.unit_sampling_distribution[unit_idx] * table_size)
            
            # 테이블에 유닛 인덱스 채우기
            for _ in range(count):
                if table_idx < table_size:
                    self.negative_sampling_table[table_idx] = unit_idx
                    table_idx += 1
        
        # 남은 슬롯들을 랜덤하게 채우기
        while table_idx < table_size:
            random_idx = np.random.randint(0, unit_vocab_size)
            self.negative_sampling_table[table_idx] = random_idx
            table_idx += 1
        
        print(f"유닛 음성 샘플링 테이블 구축 완료 (크기: {table_size:,})")
    
    def _sample_negative_units(self, positive_unit: str, num_samples: int) -> List[str]:
        """
        음성 유닛 샘플을 추출합니다.
        
        Parameters:
            positive_unit (str): 양성 유닛 (제외할 유닛)
            num_samples (int): 추출할 음성 샘플 개수
            
        Returns:
            List[str]: 음성 샘플 유닛명들
        """
        positive_unit_idx = self.vocab_builder.get_unit_index(positive_unit)
        negative_samples = []
        
        while len(negative_samples) < num_samples:
            # 테이블에서 랜덤 인덱스 선택
            table_idx = np.random.randint(0, len(self.negative_sampling_table))
            candidate_idx = self.negative_sampling_table[table_idx]
            
            # 양성 유닛과 다르고 이미 선택되지 않은 경우만 추가
            if candidate_idx != positive_unit_idx:
                candidate_unit = self.vocab_builder.get_unit_by_index(candidate_idx)
                if candidate_unit is not None and candidate_unit not in negative_samples:
                    negative_samples.append(candidate_unit)
        
        return negative_samples
    
    def evaluate(self, validation_pairs: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Validation 데이터로 모델 평가 (역전파 없이 순전파만)
        
        Parameters:
            validation_pairs (List[Tuple[str, str]]): Valid 학습 페어
            
        Returns:
            Dict[str, float]: {'valid_loss': 평균 손실, 'valid_pairs_count': 평가된 페어 수, 'valid_total_pairs': 전체 페어 수}
            
        Note:
            - 순전파만 수행 (그래디언트 업데이트 없음)
            - Valid-only 어휘도 포함되어 있지만, 참고용으로만 사용
            - Train에서 본 어휘가 많을수록 Valid loss가 더 의미있음
        """
        total_loss = 0.0
        valid_count = 0
        
        for center_item, context_unit in validation_pairs:
            # 1. 어휘 검증
            if not (self.vocab_builder.is_valid_item(center_item) and 
                    self.vocab_builder.is_valid_item(context_unit)):
                continue
            
            # 2. 음성 샘플 추출
            negative_units = self._sample_negative_units(
                context_unit, 
                self.negative_samples
            )
            
            # 3. 순전파만 수행 (역전파 없음)
            loss, _ = self.model.forward_pass(
                center_item, context_unit, negative_units, self.vocab_builder
            )
            
            # 4. 유효한 손실만 누적
            if not np.isinf(loss):
                total_loss += loss
                valid_count += 1
        
        # 5. 평균 손실 계산
        avg_loss = total_loss / valid_count if valid_count > 0 else float('inf')
        
        return {
            'valid_loss': avg_loss,
            'valid_pairs_count': valid_count,
            'valid_total_pairs': len(validation_pairs)
        }
    
    def _apply_fasttext_gradients(self, gradients: Dict[str, np.ndarray]) -> None:
        """
        FastText 방식으로 그래디언트를 분배하여 업데이트를 적용합니다.
        
        Parameters:
            gradients (Dict[str, np.ndarray]): 아이템별 그래디언트
        """
        self.optimizer.t += 1
        
        for item, grad in gradients.items():
            unit_idx = self.vocab_builder.get_unit_index(item)
            if unit_idx is not None:  # 유닛인 경우 - FastText 방식으로 분배
                composition_indices = self.vocab_builder.get_unit_composition(item)
                if composition_indices:
                    # FastText 방식: 그래디언트를 구성요소 수로 나누어 분배
                    # grad_per_component = grad / len(composition_indices)
                    
                    for comp_type, idx in composition_indices:
                        if comp_type == 'token':
                            self.optimizer.update_token_embedding(idx, grad, self.model)
                        elif comp_type == 'unit':
                            self.optimizer.update_unit_embedding(idx, grad, self.model)
            else:  # 토큰인 경우
                token_idx = self.vocab_builder.get_token_index(item)
                if token_idx is not None:
                    self.optimizer.update_token_embedding(token_idx, grad, self.model)
    
    def train_epoch(self, training_pairs: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        한 에포크 동안 FastText 스타일 모델을 학습합니다.
        
        Parameters:
            training_pairs (List[Tuple[str, str]]): (중심_아이템, 컨텍스트_유닛) 학습 페어 리스트
            
        Returns:
            Dict[str, float]: 에포크 학습 통계
        """
        epoch_loss = 0.0
        processed_pairs = 0
        start_time = time.time()
        
        # 학습 페어를 랜덤하게 섞기
        shuffled_pairs = training_pairs.copy()
        random.shuffle(shuffled_pairs)
        
        # 배치별 그래디언트 누적
        batch_gradients = defaultdict(lambda: np.zeros(self.model.embed_dim))
        batch_size = 100  # 배치 크기
        
        for pair_idx, (center_item, context_unit) in enumerate(shuffled_pairs):
            # 아이템들이 유효한지 확인
            if not (self.vocab_builder.is_valid_item(center_item) and 
                    self.vocab_builder.is_valid_item(context_unit)):
                continue
            
            # 음성 유닛 샘플 추출
            negative_units = self._sample_negative_units(context_unit, self.negative_samples)
            
            # 순전파
            loss, forward_cache = self.model.forward_pass(
                center_item, context_unit, negative_units, self.vocab_builder
            )
            
            if np.isinf(loss):
                continue  # 무효한 손실이면 스킵
            
            # 역전파
            center_grads, context_grads, negative_grads = self.model.backward_pass(
                forward_cache, self.vocab_builder
            )
            
            # 그래디언트 수집 및 누적
            all_gradients = defaultdict(lambda: np.zeros(self.model.embed_dim))
            for item, grad in center_grads.items():
                all_gradients[item] += grad
            for item, grad in context_grads.items():
                all_gradients[item] += grad
            for item, grad in negative_grads.items():
                all_gradients[item] += grad
            
            # 배치에 그래디언트 누적
            for item, grad in all_gradients.items():
                batch_gradients[item] += grad
            
            epoch_loss += loss
            processed_pairs += 1
            
            # 배치 단위로 업데이트
            if (pair_idx + 1) % batch_size == 0 or pair_idx == len(shuffled_pairs) - 1:
                self._apply_fasttext_gradients(dict(batch_gradients))
                batch_gradients.clear()
            
            # 진행상황 출력
            if processed_pairs % 10000 == 0:
                avg_loss = epoch_loss / processed_pairs
                elapsed_time = time.time() - start_time
                pairs_per_sec = processed_pairs / elapsed_time
                #print(f"  처리된 페어: {processed_pairs:,} | "
                #    f"평균 손실: {avg_loss:.6f} | "
                #    f"속도: {pairs_per_sec:.0f} pairs/sec")
        
        # 에포크 통계 계산
        avg_loss = epoch_loss / processed_pairs if processed_pairs > 0 else 0.0
        epoch_time = time.time() - start_time
        
        return {
            'avg_loss': avg_loss,
            'total_loss': epoch_loss,
            'processed_pairs': processed_pairs,
            'epoch_time': epoch_time,
            'pairs_per_sec': processed_pairs / epoch_time if epoch_time > 0 else 0
        }
    
    def train(self, training_pairs: List[Tuple[str, str]], 
             validation_pairs: Optional[List[Tuple[str, str]]] = None,
             epochs: int = 5,
             save_path: Optional[str] = None, 
             evaluation_words: Optional[List[str]] = None) -> List[Dict]:
        """
        전체 FastText 스타일 학습 과정을 실행합니다.
        
        Parameters:
            training_pairs (List[Tuple[str, str]]): Train 학습 페어
            validation_pairs (Optional[List[Tuple[str, str]]]): Valid 학습 페어 (None이면 Valid 평가 안 함)
            epochs (int): 학습 에포크 수
            save_path (Optional[str]): 모델 저장 경로
            evaluation_words (Optional[List[str]]): 평가용 토큰 리스트
            
        Returns:
            List[Dict]: 각 에포크별 학습/검증 통계
        """
        print(f"\n=== FastText 스타일 Token-Unit Word2Vec 학습 시작 ===")
        print(f"학습 페어 수: {len(training_pairs):,}")
        if validation_pairs is not None:
            print(f"검증 페어 수: {len(validation_pairs):,}")
        print(f"에포크 수: {epochs}")
        print(f"토큰 어휘 크기: {self.vocab_builder.vocab_size:,}")
        print(f"유닛 어휘 크기: {self.vocab_builder.unit_vocab_size:,}")
        print(f"임베딩 차원: {self.model.embed_dim}")
        print(f"유닛 임베딩: FastText 방식 (토큰들 + 유닛 평균)")
        print("=" * 50)
        
        training_history = []
        
        for epoch in range(1, epochs + 1):
            print(f"\n--- 에포크 {epoch}/{epochs} ---")
            
            # 1. Train 에포크
            epoch_stats = self.train_epoch(training_pairs)
            epoch_stats['epoch'] = epoch
            
            train_loss = epoch_stats['avg_loss']
            
            # 2. 에포크 결과 출력
            print(f"에포크 {epoch} 완료:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  처리 시간: {epoch_stats['epoch_time']:.1f}초")
            
            # 3. Validation 평가
            if validation_pairs is not None:
                print(f"  Validation 평가 중...")
                valid_stats = self.evaluate(validation_pairs)
                
                # epoch_stats에 valid 정보 추가
                epoch_stats['valid_loss'] = valid_stats['valid_loss']
                epoch_stats['valid_pairs_count'] = valid_stats['valid_pairs_count']
                
                valid_loss = valid_stats['valid_loss']
                print(f"  Valid Loss: {valid_loss:.6f}")
                print(f"  Loss Gap (Valid-Train): {valid_loss - train_loss:+.6f}")
            
            training_history.append(epoch_stats)
            
            # 4. Early Stopping (Validation loss 기준)
            if self.early_stopping and validation_pairs is not None:
                current_loss = valid_stats['valid_loss']
                
                if (self.best_loss - current_loss) > 1e-5:
                    # 손실 개선됨
                    self.best_loss = current_loss
                    self.patience_counter = 0
                    
                    # 최적 모델 상태 저장
                    self.best_model_state = {
                        'W_token': self.model.W_token.copy(),
                        'W_unit': self.model.W_unit.copy(),
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'valid_loss': valid_loss
                    }
                    print(f"  최적 모델 갱신 (Valid Loss: {valid_loss:.6f})")
                else:
                    # 손실 개선 안 됨
                    self.patience_counter += 1
                    print(f"  개선 없음 ({self.patience_counter}/{self.patience})")
                    
                    if self.patience_counter >= self.patience:
                        print(f"\n" + "="*50)
                        print(f"Early Stopping")
                        print(f"  최적 에포크: {self.best_model_state['epoch']}")
                        print(f"  최적 Train Loss: {self.best_model_state['train_loss']:.6f}")
                        print(f"  최적 Valid Loss: {self.best_model_state['valid_loss']:.6f}")
                        print("="*50)
                        
                        # 최적 모델로 복원
                        self.model.W_token = self.best_model_state['W_token']
                        self.model.W_unit = self.best_model_state['W_unit']
                        
                        # 최적 모델 저장
                        if save_path:
                            final_path = save_path if save_path.endswith('.pkl') else f"{save_path}.pkl"
                            self.model.save_model(final_path)
                            print(f"  최적 모델 저장: {final_path}")
                        
                        break  # 학습 중단
            
            # 5. Early Stopping (Validation 없을 때, Train loss 기준)
            elif self.early_stopping and validation_pairs is None:
                current_loss = train_loss
                
                if current_loss < self.best_loss:
                    improvement = self.best_loss - current_loss
                    self.best_loss = current_loss
                    self.patience_counter = 0
                    
                    # 최적 모델 상태 저장
                    self.best_model_state = {
                        'W_token': self.model.W_token.copy(),
                        'W_unit': self.model.W_unit.copy(),
                        'epoch': epoch,
                        'train_loss': train_loss
                    }
                    print(f" 최적 모델 갱신 (Train Loss: {train_loss:.6f})")
                else:
                    self.patience_counter += 1
                    print(f" 개선 없음 ({self.patience_counter}/{self.patience})")
                    
                    if self.patience_counter >= self.patience:
                        print(f"\nEarly Stopping")
                        print(f"  최적 에포크: {self.best_model_state['epoch']}")
                        print(f"  최적 Train Loss: {self.best_model_state['train_loss']:.6f}")
                        
                        # 최적 모델로 복원
                        self.model.W_token = self.best_model_state['W_token']
                        self.model.W_unit = self.best_model_state['W_unit']
                        
                        # 최적 모델 저장
                        if save_path:
                            final_path = save_path if save_path.endswith('.pkl') else f"{save_path}.pkl"
                            self.model.save_model(final_path)
                            print(f"  최적 모델 저장: {final_path}")
                        
                        break  # 학습 중단
            
            # 6. 평가 (선택사항)
            if evaluation_words and epoch % 2 == 0:
                self._evaluate_model(evaluation_words)
            
            # 7. 중간 모델 저장 (선택사항)
            if save_path and epoch % 5 == 0:
                intermediate_path = save_path.replace('.pkl', f'_epoch_{epoch}.pkl')
                self.model.save_model(intermediate_path)
        
        # 8. 최종 모델 저장 (Early stopping 안 된 경우)
        if save_path:
            early_stopped = (self.early_stopping and 
                            ((validation_pairs is not None and self.patience_counter >= self.patience) or
                            (validation_pairs is None and self.patience_counter >= self.patience)))
            
            if not early_stopped:
                final_path = save_path if save_path.endswith('.pkl') else f"{save_path}.pkl"
                self.model.save_model(final_path)
                print(f"\n최종 모델 저장됨: {final_path}")
        
        print(f"\n=== FastText 스타일 학습 완료 ===")
        return training_history
    
    def _evaluate_model(self, evaluation_words: List[str], top_k: int = 5) -> None:
        """평가용 단어들에 대해 FastText 스타일 분석을 출력합니다."""
        print(f"\n--- FastText 스타일 모델 평가 ---")
        
        for word in evaluation_words[:3]:
            # 토큰인 경우
            unit_index = self.vocab_builder.get_unit_index(word)
            if unit_index is None:
                token_embed = self.model.get_item_embedding(word, self.vocab_builder)
                if token_embed is not None:
                    print(f"토큰 '{word}' 임베딩 노름: {np.linalg.norm(token_embed):.4f}")
                    
                    # 이 토큰과 유사한 유닛들 찾기
                    similar_units = []
                    for unit_idx in range(min(10, self.vocab_builder.unit_vocab_size)):
                        unit = self.vocab_builder.get_unit_by_index(unit_idx)
                        if unit and '_' in unit:
                            similarity = self.model.get_similarity(word, unit, self.vocab_builder)
                            similar_units.append((unit, similarity))
                    
                    # 상위 3개 유사 유닛 출력
                    if similar_units:
                        similar_units.sort(key=lambda x: x[1], reverse=True)
                        print(f"  유사한 유닛들:")
                        for unit, sim in similar_units[:3]:
                            print(f"    {unit}: {sim:.4f}")
            
            # 유닛인 경우 FastText 방식 분석
            elif self.vocab_builder.is_valid_item(word):
                composition = self.vocab_builder.get_unit_composition(word)
                if composition:
                    print(f"유닛 '{word}' FastText 구성:")
                    component_names = []
                    for comp_type, idx in composition:
                        if comp_type == 'token':
                            token = self.vocab_builder.get_token_by_index(idx)
                            component_names.append(f"token:{token}")
                        elif comp_type == 'unit':
                            unit = self.vocab_builder.get_unit_by_index(idx)
                            component_names.append(f"unit:{unit}")
                    
                    print(f"  구성: {', '.join(component_names)}")
                    
                    fasttext_embed = self.model.get_fasttext_unit_embedding(word, self.vocab_builder)
                    if fasttext_embed is not None:
                        print(f"  FastText 임베딩 노름: {np.linalg.norm(fasttext_embed):.4f}")
        
        print("-" * 40)


# 사용 예시
if __name__ == "__main__":
    from collections import Counter
    import numpy as np
    
    # 테스트용 데이터 생성
    test_token_counter = Counter({
        'button': 80, 'next': 60, 'toolbar': 90, 'menu': 70, 'save': 40, 'file': 50
    })
    
    test_action_units = [
        ['start', 'button_next', 'toolbar_menu', 'end'],
        ['click', 'button_save', 'file_dialog', 'close'],
        ['open', 'menu_file', 'select_option', 'confirm_dialog']
    ]
    
    # FastText 스타일 어휘 구축기 생성
    from vocab_builder import VocabularyBuilder
    vocab_builder = VocabularyBuilder(min_count=1)
    vocab_builder.build_vocabulary(test_token_counter, test_action_units)
    
    # FastText 스타일 모델 생성
    model = Word2VecModel(vocab_builder.vocab_size, vocab_builder.unit_vocab_size, embed_dim=50)
    
    # 가상의 FastText 스타일 학습 페어 생성
    training_pairs = []
    tokens = list(test_token_counter.keys())
    units = ['start', 'button_next', 'toolbar_menu', 'end']
    
    # Token-Unit 페어
    for _ in range(500):
        token = np.random.choice(tokens)
        unit = np.random.choice(units)
        if vocab_builder.is_valid_item(token) and vocab_builder.is_valid_item(unit):
            training_pairs.append((token, unit))
    
    # Unit-Unit 페어
    for _ in range(300):
        center_unit = np.random.choice(units)
        context_unit = np.random.choice(units)
        if (center_unit != context_unit and 
            vocab_builder.is_valid_item(center_unit) and 
            vocab_builder.is_valid_item(context_unit)):
            training_pairs.append((center_unit, context_unit))
    
    print(f"생성된 학습 페어 수: {len(training_pairs)}")
    
    # 학습기 생성
    trainer = Word2VecTrainer(model, vocab_builder, learning_rate=0.01, negative_samples=3)
    
    # 학습 실행 (1 에포크만)
    history = trainer.train(training_pairs, epochs=1, evaluation_words=['start', 'button', 'button_next'])
    
    print(f"\n학습 히스토리:")
    for epoch_stats in history:
        print(f"에포크 {epoch_stats['epoch']}: 손실 {epoch_stats['avg_loss']:.6f}")
    
    # FastText 스타일 임베딩 테스트
    print(f"\n=== FastText 스타일 임베딩 테스트 ===")
    
    # 토큰 임베딩
    start_embedding = model.get_item_embedding('start', vocab_builder)
    if start_embedding is not None:
        print(f"토큰 'start' 임베딩 크기: {start_embedding.shape}")
    
    # FastText 방식 유닛 임베딩
    if vocab_builder.is_valid_item('button_next'):
        fasttext_embedding = model.get_fasttext_unit_embedding('button_next', vocab_builder)
        if fasttext_embedding is not None:
            print(f"유닛 'button_next' FastText 임베딩 크기: {fasttext_embedding.shape}")
            
            # 구성요소 확인
            composition = vocab_builder.get_unit_composition('button_next')
            component_names = []
            for comp_type, idx in composition:
                if comp_type == 'token':
                    component_names.append(f"token:{vocab_builder.get_token_by_index(idx)}")
                elif comp_type == 'unit':
                    component_names.append(f"unit:{vocab_builder.get_unit_by_index(idx)}")
            
            print(f"구성: sum([{', '.join(component_names)}])")
    
    # Token-Unit 유사도
    similarity = model.get_similarity('button', 'button_next', vocab_builder)
    print(f"토큰 'button'과 유닛 'button_next' 유사도: {similarity:.4f}")
    
    print(f"\nFastText 스타일 모델 특징:")
    print(f"- 유닛 임베딩 = sum([구성_토큰들, 유닛_자체])")
    print(f"- Token-Unit + Unit-Unit 학습 페어")
    print(f"- FastText 방식 그래디언트 분배")
    print(f"- Compositional + Holistic semantics")
    
    # 모델 정보 출력
    model_info = model.get_model_info()
    print(f"\n모델 정보:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")