"""
어휘 구축 모듈
토큰과 유닛 모두를 위한 인덱싱 및 어휘 사전 관리
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple


class VocabularyBuilder:
    """
    액션 토큰과 액션 유닛의 어휘를 구축하고 인덱싱을 관리하는 클래스
    
    Attributes:
        token_to_idx (Dict[str, int]): 토큰을 인덱스로 매핑하는 사전
        idx_to_token (Dict[int, str]): 인덱스를 토큰으로 매핑하는 사전
        unit_to_idx (Dict[str, int]): 유닛을 인덱스로 매핑하는 사전
        idx_to_unit (Dict[int, str]): 인덱스를 유닛으로 매핑하는 사전
        token_counts (Counter): 각 토큰의 빈도수
        unit_counts (Counter): 각 유닛의 빈도수
        vocab_size (int): 토큰 어휘 크기
        unit_vocab_size (int): 유닛 어휘 크기
    """
    
    def __init__(self, min_count: int = 1):
        """
        어휘 구축기 초기화
        
        Parameters:
            min_count (int): 어휘에 포함시키기 위한 최소 빈도수 (기본값: 1)
        """
        self.min_count = min_count
        
        # 토큰 어휘
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.token_counts = Counter()
        self.vocab_size = 0
        
        # 유닛 어휘
        self.unit_to_idx = {}
        self.idx_to_unit = {}
        self.unit_counts = Counter()
        self.unit_vocab_size = 0
    
    def build_vocabulary(self, token_counter: Counter, action_units: List[List[str]]) -> None:
        """
        토큰 카운터와 액션 유닛들로부터 어휘 사전을 구축합니다.
        
        Parameters:
            token_counter (Counter): 데이터 로더에서 생성된 토큰 카운터
            action_units (List[List[str]]): 모든 시퀀스의 액션 유닛들
            
        Note:
            토큰과 유닛 모두에 대해 min_count 필터링을 적용합니다.
        """
        # 1. 토큰 어휘 구축
        self._build_token_vocabulary(token_counter)
        
        # 2. 유닛 어휘 구축
        self._build_unit_vocabulary(action_units)
        
        print(f"어휘 구축 완료:")
        print(f"  토큰 어휘 크기: {self.vocab_size}")
        print(f"  유닛 어휘 크기: {self.unit_vocab_size}")
    
    def _build_token_vocabulary(self, token_counter: Counter) -> None:
        """토큰 어휘를 구축합니다."""
        # 최소 빈도수 조건을 만족하는 토큰들만 필터링
        filtered_tokens = [token for token, count in token_counter.items() 
                        if count >= self.min_count]
        
        # 빈도수 순으로 정렬 (높은 빈도부터)
        sorted_tokens = sorted(filtered_tokens, 
                            key=lambda x: token_counter[x], 
                            reverse=True)
        
        # 토큰-인덱스 매핑 생성 - 높은 빈도가 먼저 정렬되기 때문에 이른 idx를 받는다.
        for idx, token in enumerate(sorted_tokens):
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
            self.token_counts[token] = token_counter[token]
        
        self.vocab_size = len(self.token_to_idx)
    
    def _build_unit_vocabulary(self, action_units: List[List[str]]) -> None:
        """유닛 어휘를 구축합니다."""
        # 모든 유닛의 빈도수 계산
        unit_counter = Counter()
        for units_list in action_units:
            unit_counter.update(units_list)
        
        # 최소 빈도수 조건을 만족하는 유닛들만 필터링
        filtered_units = [unit for unit, count in unit_counter.items() 
                        if count >= self.min_count]
        
        # 빈도수 순으로 정렬 (높은 빈도부터)
        sorted_units = sorted(filtered_units, 
                            key=lambda x: unit_counter[x], 
                            reverse=True)
        
        # 유닛-인덱스 매핑 생성
        for idx, unit in enumerate(sorted_units):
            self.unit_to_idx[unit] = idx
            self.idx_to_unit[idx] = unit
            self.unit_counts[unit] = unit_counter[unit]
        
        self.unit_vocab_size = len(self.unit_to_idx)
    
    def get_token_index(self, token: str) -> Optional[int]:
        """토큰의 인덱스를 반환합니다."""
        return self.token_to_idx.get(token)
    
    def get_token_by_index(self, idx: int) -> Optional[str]:
        """인덱스에 해당하는 토큰을 반환합니다."""
        return self.idx_to_token.get(idx)
    
    def get_unit_index(self, unit: str) -> Optional[int]:
        """유닛의 인덱스를 반환합니다."""
        return self.unit_to_idx.get(unit)
    
    def get_unit_by_index(self, idx: int) -> Optional[str]:
        """인덱스에 해당하는 유닛을 반환합니다."""
        return self.idx_to_unit.get(idx)
    
    def get_unit_composition(self, unit: str) -> List[Tuple[str, int]]:
        """
        FastText 방식으로 유닛의 구성요소 인덱스들을 반환합니다.
        
        Parameters:
            unit (str): 유닛명 (예: 'button_next')
            
        Returns:
            List[Tuple[str, int]]: (타입, 인덱스) 튜플 리스트
                                타입은 'token' 또는 'unit', ('token', 4), ('unit', 2)
            
        Note:
            FastText 방식: unit_embedding = sum([token1, token2, ..., unit])
            예: 'button_next' → [('token', button_idx), ('token', next_idx), ('unit', button_next_idx)]
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
        
        # # 1. 구성 토큰들의 인덱스 추가 (단일 객체, start와 같은 경우 unit의 인덱스로 구분하기 위해서 수정)
        # tokens = unit.split('_')
        # for token in tokens:
        #     token_idx = self.get_token_index(token)
        #     if token_idx is not None:
        #         composition_indices.append(('token', token_idx))
        
        # # 2. 유닛 자체의 인덱스 추가
        # unit_idx = self.get_unit_index(unit)
        # if unit_idx is not None:
        #     composition_indices.append(('unit', unit_idx))
            
        # return composition_indices
    
    def is_valid_item(self, item: str) -> bool:
        """
        아이템(토큰 또는 유닛)이 어휘에 있는지 확인합니다.
        
        Parameters:
            item (str): 확인할 아이템
            
        Returns:
            bool: 어휘에 있으면 True
        """
        
        # 주어진 item이 token으로 존재하는지 확인
        if self.get_token_index(item) is not None:
            return True
        
        # 주어진 item이 unit으로 존재하는지 확인
        if self.get_unit_index(item) is not None:
            # 토큰으로 구성되어 있다면 해당 토큰도 확인
            if '_' in item:
                tokens = item.split('_')
                return all(self.get_token_index(token) is not None for token in tokens)
            return True
        
        return False
    
        # # 토큰인지 확인 (언더스코어가 없으면 토큰)
        # if '_' not in item:
        #     return self.get_token_index(item) is not None
        # else:
        #     # 유닛인 경우: 유닛 자체와 모든 구성 토큰이 어휘에 있어야 함
        #     unit_idx = self.get_unit_index(item)
        #     if unit_idx is None:
        #         return False
            
        #     tokens = item.split('_')
        #     return all(self.get_token_index(token) is not None for token in tokens)
    
    def filter_training_pairs(self, training_pairs: List[tuple]) -> List[tuple]:
        """
        FastText 스타일 학습 페어에서 어휘에 없는 아이템들을 필터링합니다.
        
        Parameters:
            training_pairs (List[tuple]): (중심_아이템, 컨텍스트_유닛) 페어 리스트
            
        Returns:
            List[tuple]: 필터링된 학습 페어 리스트
        """
        filtered_pairs = []
        original_count = len(training_pairs)
        
        for center_item, context_unit in training_pairs:
            # 중심 아이템과 컨텍스트 유닛 모두 유효한지 확인
            if (self.is_valid_item(center_item) and self.is_valid_item(context_unit)):
                filtered_pairs.append((center_item, context_unit))
        
        filtered_count = len(filtered_pairs)
        removed_count = original_count - filtered_count
        
        # 타입별 통계
        token_unit_pairs = 0
        unit_unit_pairs = 0
        
        for center, _ in filtered_pairs:
            # 토큰 어휘에 존재하면 token
            if center in self.token_to_idx and center not in self.unit_to_idx:
                token_unit_pairs += 1
            elif center in self.unit_to_idx:
                unit_unit_pairs += 1
        
        # token_unit_pairs = sum(1 for center, _ in filtered_pairs if '_' not in center)
        # unit_unit_pairs = sum(1 for center, _ in filtered_pairs if '_' in center)
        
        print(f"FastText 스타일 학습 페어 필터링 완료:")
        print(f"  원본 페어 수: {original_count:,}")
        print(f"  필터링된 페어 수: {filtered_count:,}")
        print(f"    - Token-Unit 페어: {token_unit_pairs:,}")
        print(f"    - Unit-Unit 페어: {unit_unit_pairs:,}")
        print(f"  제거된 페어 수: {removed_count:,}")
        
        return filtered_pairs
    
    def get_negative_sampling_distribution(self) -> np.ndarray:
        """
        유닛에 대한 Negative Sampling 확률 분포를 생성합니다.
        
        Returns:
            np.ndarray: 각 유닛의 negative sampling 확률 분포
            
        Note:
            Token-Unit 학습에서는 유닛을 컨텍스트로 사용하므로
            유닛에 대한 negative sampling 분포를 생성합니다.
        """
        if self.unit_vocab_size == 0:
            raise ValueError("유닛 어휘가 구축되지 않았습니다.")
        
        # 유닛 빈도수 배열 생성
        frequencies = np.zeros(self.unit_vocab_size)
        for idx in range(self.unit_vocab_size):
            unit = self.idx_to_unit[idx]
            frequencies[idx] = self.unit_counts[unit]
        
        # 3/4 지수 적용
        powered_frequencies = np.power(frequencies, 0.75)
        
        # 확률 분포로 정규화
        distribution = powered_frequencies / np.sum(powered_frequencies)
        
        return distribution
    
    def get_vocabulary_info(self) -> Dict:
        """어휘 정보를 딕셔너리로 반환합니다."""
        if self.vocab_size == 0 or self.unit_vocab_size == 0:
            return {"error": "어휘가 구축되지 않았습니다."}
        
        token_total_count = sum(self.token_counts.values())
        unit_total_count = sum(self.unit_counts.values())
        
        return {
            "token_vocab_size": self.vocab_size,
            "unit_vocab_size": self.unit_vocab_size,
            "min_count": self.min_count,
            "total_token_count": token_total_count,
            "total_unit_count": unit_total_count,
            "most_common_tokens": self.token_counts.most_common(5),
            "most_common_units": self.unit_counts.most_common(5),
            "average_token_frequency": token_total_count / self.vocab_size,
            "average_unit_frequency": unit_total_count / self.unit_vocab_size
        }
    
    def save_vocabulary(self, file_path: str) -> None:
        """구축된 어휘를 파일로 저장합니다."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# 토큰 어휘 크기: {self.vocab_size}\n")
            f.write(f"# 유닛 어휘 크기: {self.unit_vocab_size}\n")
            f.write(f"# 최소 빈도수: {self.min_count}\n")
            f.write("\n=== 토큰 어휘 ===\n")
            f.write("# 형식: 인덱스\t토큰\t빈도수\n")
            
            for idx in range(self.vocab_size):
                token = self.idx_to_token[idx]
                count = self.token_counts[token]
                f.write(f"{idx}\t{token}\t{count}\n")
            
            f.write("\n=== 유닛 어휘 ===\n")
            f.write("# 형식: 인덱스\t유닛\t빈도수\n")
            
            for idx in range(self.unit_vocab_size):
                unit = self.idx_to_unit[idx]
                count = self.unit_counts[unit]
                f.write(f"{idx}\t{unit}\t{count}\n")
        
        print(f"어휘가 저장되었습니다: {file_path}")


# 사용 예시
if __name__ == "__main__":
    from collections import Counter
    
    # 테스트용 데이터
    test_token_counter = Counter({
        'start': 100, 'toolbar': 80, 'menu': 60, 'button': 120,
        'end': 90, 'next': 70, 'ss': 40, 'file': 50, 'dialog': 30
    })
    
    test_action_units = [
        ['start', 'toolbar_menu', 'toolbar_ss', 'next', 'button_end'],
        ['click', 'file_dialog', 'select_option', 'confirm_action', 'close'],
        ['open', 'menu_edit', 'copy_text', 'paste_text', 'save_file']
    ]
    
    # 어휘 구축기 생성
    vocab_builder = VocabularyBuilder(min_count=1)
    
    #token_counter, action_unit 입력
    vocab_builder.build_vocabulary(test_token_counter, test_action_units)
    
    # 어휘 정보 출력
    info = vocab_builder.get_vocabulary_info()
    print("\n어휘 정보:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 테스트용 Token-Unit 학습 페어
    test_pairs = [
        ('start', 'toolbar_menu'),
        ('toolbar', 'start'),
        ('menu', 'toolbar_ss'),
        ('unknown_token', 'toolbar_menu'),  # 필터링될 페어
        ('start', 'unknown_unit')  # 필터링될 페어
    ]
    
    # 페어 필터링 테스트
    filtered_pairs = vocab_builder.filter_training_pairs(test_pairs)
    print(f"\n필터링된 페어들:")
    for token, unit in filtered_pairs:
        print(f"  ('{token}', '{unit}')")
    
    # Negative sampling 분포 생성
    neg_dist = vocab_builder.get_negative_sampling_distribution()
    print(f"\nNegative sampling 분포 크기: {len(neg_dist)}")
    print(f"분포 합계: {np.sum(neg_dist):.6f}")