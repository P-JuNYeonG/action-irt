"""
Word2Vec 모델 구현 (Hybrid 스타일)
Token-Unit Skip-gram with Negative Sampling 알고리즘
Hybrid 방식: unit_embedding = SUM([token_embeddings, unit_embedding])
"""

import numpy as np
from typing import Tuple, Optional, List
import pickle


class Word2VecModel:
    """
    FastText 스타일 Token-Unit Skip-gram with Negative Sampling 모델
    
    Attributes:
        token_vocab_size (int): 토큰 어휘 크기
        unit_vocab_size (int): 유닛 어휘 크기
        embed_dim (int): 임베딩 차원
        W_token (np.ndarray): 토큰 임베딩 행렬 (token_vocab_size x embed_dim)
        W_unit (np.ndarray): 유닛 임베딩 행렬 (unit_vocab_size x embed_dim)
        
    Note:
        FastText 방식으로 유닛의 최종 임베딩은 구성 토큰들과 유닛 자체 임베딩의 평균입니다.
        예: button_next = SUM([button_emb, next_emb, button_next_emb])
    """
    
    def __init__(self, token_vocab_size: int, unit_vocab_size: int, 
                 embed_dim: int = 300, random_seed: int = 42):
        """
        FastText 스타일 Token-Unit Word2Vec 모델 초기화
        
        Parameters:
            token_vocab_size (int): 토큰 어휘 크기
            unit_vocab_size (int): 유닛 어휘 크기
            embed_dim (int): 임베딩 벡터 차원 (기본값: 300)
            random_seed (int): 재현 가능한 결과를 위한 랜덤 시드
        """
        self.token_vocab_size = token_vocab_size
        self.unit_vocab_size = unit_vocab_size
        self.embed_dim = embed_dim
        
        # 재현 가능한 결과를 위한 시드 설정
        np.random.seed(random_seed)
        
        # Xavier 초기화
        token_std = np.sqrt(1.0 / embed_dim)
        unit_std = np.sqrt(1.0 / embed_dim)
        
        # 토큰과 유닛 임베딩 행렬 모두 유지
        self.W_token = np.random.normal(0, token_std, (token_vocab_size, embed_dim))
        self.W_unit = np.random.normal(0, unit_std, (unit_vocab_size, embed_dim))
        
        print(f"FastText 스타일 Token-Unit Word2Vec 모델 초기화 완료:")
        print(f"  토큰 어휘 크기: {token_vocab_size}")
        print(f"  유닛 어휘 크기: {unit_vocab_size}")
        print(f"  임베딩 차원: {embed_dim}")
        print(f"  토큰 행렬 크기: {self.W_token.shape}")
        print(f"  유닛 행렬 크기: {self.W_unit.shape}")
        print(f"  유닛 임베딩: FastText 방식 (토큰들 + 유닛 평균)")
    
    def get_token_embedding(self, token_idx: int) -> np.ndarray:
        """토큰 인덱스에 해당하는 임베딩 벡터를 반환합니다."""
        if token_idx >= self.token_vocab_size or token_idx < 0:
            raise IndexError(f"토큰 인덱스가 범위를 벗어났습니다: {token_idx}")
        
        return self.W_token[token_idx].copy()
    
    def get_unit_embedding(self, unit_idx: int) -> np.ndarray:
        """유닛 인덱스에 해당하는 임베딩 벡터를 반환합니다."""
        if unit_idx >= self.unit_vocab_size or unit_idx < 0:
            raise IndexError(f"유닛 인덱스가 범위를 벗어났습니다: {unit_idx}")
        
        return self.W_unit[unit_idx].copy()
    
    def get_fasttext_unit_embedding(self, unit: str, vocab_builder) -> Optional[np.ndarray]:
        """
        FastText 방식으로 유닛의 최종 임베딩을 계산합니다.
        
        Parameters:
            unit (str): 유닛명 (예: 'button_next')
            vocab_builder: 어휘 구축기
            
        Returns:
            Optional[np.ndarray]: FastText 방식 유닛 임베딩 벡터
            
        Note:
            FastText 방식: unit_embedding = sum([token1_emb, token2_emb, unit_emb])
            예: button_next = sum([button, next, button_next])
        """
        #학습에 사용할 각 토큰, 유닛의 인덱스 반환
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
            # FastText 방식: 모든 구성요소의 summation
            return np.sum(embeddings, axis=0)
        
        return None
    
    def get_item_embedding(self, item: str, vocab_builder) -> Optional[np.ndarray]:
        """
        아이템(토큰 또는 유닛)의 임베딩을 반환합니다.
        
        Parameters:
            item (str): 토큰 또는 유닛명
            vocab_builder: 어휘 구축기
            
        Returns:
            Optional[np.ndarray]: 아이템 임베딩 벡터
        """
        
        # 먼저 유닛인지 확인
        if vocab_builder.get_unit_index(item) is not None:
            # 유닛인 경우: FastText 방식으로 계산
            return self.get_fasttext_unit_embedding(item, vocab_builder)
        
        # 유닛이 아니면 토큰으로 처리
        token_idx = vocab_builder.get_token_index(item)
        if token_idx is not None:
            return self.get_token_embedding(token_idx)
        
        return None
        
    def forward_pass(self, center_item: str, context_item: str, 
                    negative_items: List[str], vocab_builder) -> Tuple[float, dict]:
        """
        FastText 스타일 순전파 과정을 수행합니다.
        
        Parameters:
            center_item (str): 중심 아이템 (토큰 또는 유닛)
            context_item (str): 컨텍스트 아이템 (유닛)
            negative_items (List[str]): 음성 샘플 아이템들 (유닛들)
            vocab_builder: 어휘 구축기
            
        Returns:
            Tuple[float, dict]: 손실값과 중간 계산 결과들
        """
        # 1. 중심 아이템 임베딩 추출
        center_embed = self.get_item_embedding(center_item, vocab_builder)
        if center_embed is None:
            return float('inf'), {}
        
        # 2. 양성 컨텍스트 아이템 임베딩 계산 (FastText 방식)
        context_embed = self.get_fasttext_unit_embedding(context_item, vocab_builder)
        if context_embed is None:
            return float('inf'), {}
        
        positive_score = np.dot(center_embed, context_embed)
        positive_prob = self._sigmoid(positive_score)
        
        # 3. 음성 샘플들의 임베딩 계산 및 점수 계산
        negative_embeds = [] #NS에 사용할 negative sample의 embedding
        negative_scores = []
        valid_negative_items = [] #NS에 사용한 negative sample
        
        for neg_item in negative_items:
            neg_embed = self.get_fasttext_unit_embedding(neg_item, vocab_builder)
            if neg_embed is not None:
                negative_embeds.append(neg_embed)
                neg_score = np.dot(center_embed, neg_embed)
                negative_scores.append(neg_score)
                valid_negative_items.append(neg_item)
        
        if not negative_scores:
            return float('inf'), {}
        
        negative_scores = np.array(negative_scores)
        negative_probs = self._sigmoid(-negative_scores)
        
        # 4. 손실 계산
        positive_loss = -np.log(positive_prob + 1e-10)
        negative_loss = -np.sum(np.log(negative_probs + 1e-10))
        
        total_loss = positive_loss + negative_loss
        
        # 중간 계산 결과 저장
        forward_cache = {
            'center_item': center_item,
            'center_embed': center_embed,
            'context_item': context_item,
            'context_embed': context_embed,
            'negative_items': valid_negative_items,
            'negative_embeds': negative_embeds,
            'positive_score': positive_score,
            'positive_prob': positive_prob,
            'negative_scores': negative_scores,
            'negative_probs': negative_probs
        }
        
        return total_loss, forward_cache
    
    def backward_pass(self, forward_cache: dict, vocab_builder) -> Tuple[dict, dict, dict]:
        """
        역전파 과정을 수행하여 그래디언트를 계산합니다.
        
        Parameters:
            forward_cache (dict): 순전파에서 저장된 중간 계산 결과
            vocab_builder: 어휘 구축기
            
        Returns:
            Tuple[dict, dict, dict]: 중심_그래디언트, 컨텍스트_그래디언트, 음성_그래디언트들
        """
        # 캐시에서 값들 추출
        center_item = forward_cache['center_item']
        center_embed = forward_cache['center_embed']
        
        context_item = forward_cache['context_item']
        context_embed = forward_cache['context_embed']
        
        negative_items = forward_cache['negative_items']
        negative_embeds = forward_cache['negative_embeds']
        
        positive_prob = forward_cache['positive_prob']
        negative_probs = forward_cache['negative_probs']
        
        # 1. 중심 아이템에 대한 그래디언트
        positive_grad = (positive_prob - 1) * context_embed
        
        negative_grad = np.zeros_like(center_embed)
        for i, neg_embed in enumerate(negative_embeds):
            negative_grad += (1 - negative_probs[i]) * neg_embed
        
        center_grad = positive_grad + negative_grad
        
        # 2. 컨텍스트 아이템에 대한 그래디언트
        context_grad = (positive_prob - 1) * center_embed
        
        # 3. 음성 아이템들에 대한 그래디언트
        negative_grads = {}
        for i, neg_item in enumerate(negative_items):
            neg_grad = (1 - negative_probs[i]) * center_embed
            negative_grads[neg_item] = neg_grad
        
        return {center_item: center_grad}, {context_item: context_grad}, negative_grads
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """수치적으로 안정한 시그모이드 함수 구현"""
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    def get_similarity(self, item1: str, item2: str, vocab_builder) -> float:
        """
        두 아이템 간의 코사인 유사도를 계산합니다.
        
        Parameters:
            item1 (str): 첫 번째 아이템 (토큰 또는 유닛)
            item2 (str): 두 번째 아이템 (토큰 또는 유닛)
            vocab_builder: 어휘 구축기
            
        Returns:
            float: 코사인 유사도 (-1 ~ 1)
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
        """학습된 모델을 파일로 저장합니다."""
        model_data = {
            'model_type': 'fasttext_style_token_unit',
            'token_vocab_size': self.token_vocab_size,
            'unit_vocab_size': self.unit_vocab_size,
            'embed_dim': self.embed_dim,
            'W_token': self.W_token,
            'W_unit': self.W_unit
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"FastText 스타일 Token-Unit 모델이 저장되었습니다: {file_path}")
    
    @classmethod
    def load_model(cls, file_path: str) -> 'Word2VecModel':
        """저장된 모델을 로드합니다."""
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # 새 인스턴스 생성
        model = cls(model_data['token_vocab_size'], 
                    model_data['unit_vocab_size'],
                    model_data['embed_dim'])
        
        # 저장된 가중치 로드
        model.W_token = model_data['W_token']
        model.W_unit = model_data['W_unit']
        
        print(f"FastText 스타일 Token-Unit 모델이 로드되었습니다: {file_path}")
        return model
    
    def get_model_info(self) -> dict:
        """모델 정보를 반환합니다."""
        return {
            'model_type': 'fasttext_style_token_unit',
            'token_vocab_size': self.token_vocab_size,
            'unit_vocab_size': self.unit_vocab_size,
            'embed_dim': self.embed_dim,
            'total_parameters': (self.token_vocab_size + self.unit_vocab_size) * self.embed_dim,
            'token_matrix_shape': self.W_token.shape,
            'unit_matrix_shape': self.W_unit.shape,
            'token_matrix_norm': np.linalg.norm(self.W_token),
            'unit_matrix_norm': np.linalg.norm(self.W_unit),
            'unit_embedding_method': 'fasttext_style_sum'
        }


# 사용 예시
if __name__ == "__main__":
    # 테스트용 FastText 스타일 모델 생성
    token_vocab_size = 100
    unit_vocab_size = 50
    embed_dim = 64
    
    model = Word2VecModel(token_vocab_size, unit_vocab_size, embed_dim)
    
    # 모델 정보 출력
    info = model.get_model_info()
    print("\n모델 정보:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 가상의 어휘 구축기 (테스트용)
    class MockVocabBuilder:
        def __init__(self):
            self.token_to_idx = {'button': 0, 'next': 1, 'start': 2}
            self.unit_to_idx = {'button_next': 0, 'start': 1}
        
        def get_token_index(self, token):
            return self.token_to_idx.get(token)
            
        def get_unit_index(self, unit):
            return self.unit_to_idx.get(unit)
        
        def get_unit_composition(self, unit):
            if unit == 'button_next':
                return [('token', 0), ('token', 1), ('unit', 0)]  # button, next, button_next
            elif unit == 'start':
                return [('unit', 1)]  # start
            return []
    
    vocab_builder = MockVocabBuilder()
    
    # FastText 방식 유닛 임베딩 계산 테스트
    fasttext_embedding = model.get_fasttext_unit_embedding('button_next', vocab_builder)
    
    if fasttext_embedding is not None:
        print(f"\n'button_next' FastText 임베딩 크기: {fasttext_embedding.shape}")
        print(f"임베딩 노름: {np.linalg.norm(fasttext_embedding):.4f}")
        print(f"공식: sum([button_token, next_token, button_next_unit])")
    
    # 토큰-유닛 유사도 테스트
    similarity = model.get_similarity('button', 'button_next', vocab_builder)
    print(f"토큰 'button'과 유닛 'button_next'의 유사도: {similarity:.6f}")
    
    print(f"\nFastText 스타일 모델 특징:")
    print(f"- 유닛 임베딩 = sum([구성_토큰들, 유닛_자체])")
    print(f"- Compositional + Holistic semantics")
    print(f"- 토큰과 유닛 모두 중심 아이템으로 학습 가능")
