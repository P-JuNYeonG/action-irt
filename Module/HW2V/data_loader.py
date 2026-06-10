"""
데이터 로더 모듈
[Action sequence] -> [Action unit] -> [Action token] 분할 처리
Token-Unit 학습 쌍 생성
Unit-Unit 학습 쌍 생성
"""

import os
from typing import List, Tuple
from collections import Counter


class ActionDataLoader:
    """
    액션 시퀀스 데이터를 로드하고 토큰-유닛 학습 쌍을 생성하는 클래스
    
    Attributes:
        sequences (List[str]): 원본 액션 시퀀스들
        action_units (List[List[str]]): 각 시퀀스의 액션 유닛들
        action_tokens (List[List[str]]): 각 시퀀스의 액션 토큰들
        token_counter (Counter): 전체 토큰의 빈도수
        
    Example:
        sequences : [
        "start toolbar_menu toolbar_ss next button_end",
        "click file_dialog select_option confirm_action close",
        "open menu_edit copy_text paste_text save_file"]
        
        action_units : [
        ['start', 'toolbar_menu', 'toolbar_ss', 'next', 'button_end'],
        ['click', 'file_dialog', 'select_option', 'confirm_action', 'close'],
        ['open', 'menu_edit', 'copy_text', 'paste_text', 'save_file']]
        
        action_tokens : self.action_tokens = [
        ['toolbar', 'menu', 'toolbar', 'ss', 'button', 'end'],
        ['file', 'dialog', 'select', 'option', 'confirm', 'action'],
        ['menu', 'edit', 'copy', 'text', 'paste', 'text', 'save', 'file']]
        
        self.token_counter = Counter({})
    """
    
    def __init__(self):
        """데이터 로더 초기화"""
        self.sequences = []
        self.action_units = []
        self.action_tokens = []
        self.token_counter = Counter()
    
    def load_from_file(self, file_path: str) -> None:
        """
        텍스트 파일에서 액션 시퀀스 데이터를 로드합니다.
        
        Parameters:
            file_path (str): 입력 텍스트 파일 경로
            
        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 빈 줄 제외
                    self.sequences.append(line)
        
        print(f"총 {len(self.sequences)}개의 액션 시퀀스를 로드했습니다.")
    
    def tokenize_sequences(self) -> None:
        """
        로드된 시퀀스들을 액션 유닛과 액션 토큰으로 분할합니다.
        
        처리 과정:
        1. 각 시퀀스를 공백으로 분할하여 액션 유닛 추출
        2. 각 액션 유닛을 '_'로 분할하여 액션 토큰 추출, 단일 유닛인 경우 토큰으로 변환하지 않음
        3. 토큰 빈도수 계산
        """
        self.action_units = []
        self.action_tokens = []
        
        for sequence in self.sequences:
            # 1단계: 액션 유닛 분할 (공백 기준) - action sequence가 공백으로 연결되어 있기 때문
            units = sequence.split()
            self.action_units.append(units)
            
            # 2단계: 액션 토큰 분할 ('_' 기준) - action unit이 '_'로 연결되어 있기 때문
            sequence_tokens = []
            for unit in units:
                tokens = unit.split('_')
                
                #단일 유닛인 경우 continue
                if len(tokens) == 1: continue
                
                sequence_tokens.extend(tokens)
                # 토큰 빈도수 업데이트
                self.token_counter.update(tokens)
            
            self.action_tokens.append(sequence_tokens)
        
        print(f"총 {len(set(self.token_counter.keys()))}개의 고유 액션 토큰을 발견했습니다.")
        print(f"가장 빈번한 토큰 5개: {self.token_counter.most_common(5)}")
    
    def get_training_pairs(self, window_size: int = 1) -> List[Tuple[str, str]]:
        """
        Hybrid 스타일 Token-Unit + Unit-Unit 학습 페어를 생성합니다.
        
        Parameters:
            window_size (int): 유닛 기준 컨텍스트 윈도우 크기 (기본값: 1)
            
        Returns:
            List[Tuple[str, str]]: (중심_아이템, 컨텍스트_유닛) 페어 리스트
            
        Note:
            FastText 방식과 유사하게 토큰과 유닛 모두를 중심 아이템으로 사용합니다.
            예시: 'start button_next end'
            - Token-Unit 페어: ('button', 'start'), ('button', 'end'), ('next', 'start'), ('next', 'end')
            - Unit-Unit 페어: ('button_next', 'start'), ('button_next', 'end')
        """
        training_pairs = []
        token_unit_pairs = 0
        unit_unit_pairs = 0
        
        # units = ['start', 'toolbar_menu', 'toolbar_ss', 'next', 'button_end']
        for _, units in enumerate(self.action_units):
            # 1) Token-Unit 페어 생성
            unit_positions = {}  # token -> [unit_indices]
            
            # 토큰이 어느 유닛에 속하는지 매핑
            # unit_idx = 0, unit = 'start' // 'toolbar_menu', 'toolbar_ss', 'next', 'button_end']
            for unit_idx, unit in enumerate(units):
                tokens = unit.split('_')
                
                #단일 유닛인 경우 continue
                if len(tokens) == 1: continue
                
                for token in tokens:
                    if token not in unit_positions:
                        unit_positions[token] = []
                    # toolbar, menu는 unit_idx = 1에 저장, ...
                    unit_positions[token].append(unit_idx)
            
            # 각 토큰에 대해 Token-Unit 페어 생성 - toolbar, [1, 2]
            for token, token_unit_indices in unit_positions.items():
                for token_unit_idx in token_unit_indices:
                    # 윈도우 범위 계산 (유닛 기준)
                    start = max(0, token_unit_idx - window_size) #2일 경우 1
                    end = min(len(units), token_unit_idx + window_size + 1) #2일 경우 2 + 1, range를 이용하기 위해 + 1을 더 더해줌
                    
                    # 윈도우 범위 내의 모든 유닛과 페어 생성
                    for context_unit_idx in range(start, end):
                        if token_unit_idx != context_unit_idx:  # 자기 자신은 제외
                            context_unit = units[context_unit_idx]
                            training_pairs.append((token, context_unit))
                            token_unit_pairs += 1
            
            # 2) Unit-Unit 페어 생성
            # unit_idx = 0, unit = 'start' // 'toolbar_menu', 'toolbar_ss', 'next', 'button_end']
            for unit_idx, unit in enumerate(units):
                # 윈도우 범위 계산 (유닛 기준)
                start = max(0, unit_idx - window_size)
                end = min(len(units), unit_idx + window_size + 1)
                
                # 윈도우 범위 내의 다른 유닛들과 페어 생성
                for context_unit_idx in range(start, end):
                    if unit_idx != context_unit_idx:  # 자기 자신은 제외
                        context_unit = units[context_unit_idx]
                        training_pairs.append((unit, context_unit))
                        unit_unit_pairs += 1
        
        print(f"FastText 스타일 학습 페어 생성 완료:")
        print(f"  Token-Unit 페어: {token_unit_pairs:,}개")
        print(f"  Unit-Unit 페어: {unit_unit_pairs:,}개") 
        print(f"  총 페어 수: {len(training_pairs):,}개")
        return training_pairs
    
    def get_vocabulary_stats(self) -> dict:
        """
        어휘 통계 정보를 반환합니다.
        
        Returns:
            dict: 어휘 통계 정보
        """
        # 모든 고유 유닛 수집
        all_units = set()
        for units_list in self.action_units:
            all_units.update(units_list)
        
        return {
            'total_sequences': len(self.sequences),
            'total_tokens': sum(self.token_counter.values()),
            'unique_tokens': len(self.token_counter),
            'unique_units': len(all_units),
            'average_sequence_length': sum(len(units) for units in self.action_units) / len(self.action_units) if self.action_units else 0
        }
        
    def split_sequences(self, train_ratio: float = 0.8, random_seed: int = 42) -> Tuple[List[str], List[str]]:
        """
        시퀀스를 train/valid로 랜덤 분리합니다.
        
        Parameters:
            train_ratio (float): train 비율 (기본값: 0.8)
            random_seed (int): 재현성을 위한 랜덤 시드
            
        Returns:
            Tuple[List[str], List[str]]: (train_sequences, valid_sequences)
            
        Note:
            self.sequences를 랜덤하게 섞은 후 분리합니다.
        """
        import random
        
        if not self.sequences:
            raise ValueError("시퀀스가 로드되지 않았습니다. load_from_file()을 먼저 실행하세요.")
        
        # 1. 인덱스 생성 및 섞기
        num_sequences = len(self.sequences)
        indices = list(range(num_sequences))
        
        random.seed(random_seed)
        random.shuffle(indices)
        
        # 2. 분리점 계산
        split_point = int(num_sequences * train_ratio)
        
        # 3. 인덱스 분리
        train_indices = indices[:split_point]
        valid_indices = indices[split_point:]
        
        # 4. 시퀀스 추출
        train_sequences = [self.sequences[i] for i in train_indices]
        valid_sequences = [self.sequences[i] for i in valid_indices]
        
        print(f"데이터 분리 완료:")
        print(f"  Train 시퀀스: {len(train_sequences):,}개 ({train_ratio*100:.1f}%)")
        print(f"  Valid 시퀀스: {len(valid_sequences):,}개 ({(1-train_ratio)*100:.1f}%)")
        
        return train_sequences, valid_sequences
    
    def get_training_pairs_from_sequences(self, sequences: List[str], 
                                        window_size: int = 1) -> List[Tuple[str, str]]:
        """
        주어진 시퀀스들로부터 학습 페어를 생성합니다.
        (기존 get_training_pairs() 로직과 동일하지만 특정 시퀀스만 사용)
        
        Parameters:
            sequences (List[str]): 페어를 생성할 시퀀스들
            window_size (int): 유닛 기준 컨텍스트 윈도우 크기 (기본값: 1)
            
        Returns:
            List[Tuple[str, str]]: (중심_아이템, 컨텍스트_유닛) 페어 리스트
            
        Note:
            FastText 스타일 학습 페어 생성 (Token-Unit + Unit-Unit)
            Train/Valid 각각에 대해 호출될 예정
        """
        training_pairs = []
        token_unit_pairs = 0
        unit_unit_pairs = 0
        
        # 먼저 시퀀스들을 유닛으로 분할
        action_units_list = []
        for sequence in sequences:
            units = sequence.split()
            action_units_list.append(units)
        
        # 기존 get_training_pairs()와 동일한 로직
        for _, units in enumerate(action_units_list):
            # 1) Token-Unit 페어 생성
            unit_positions = {}  # token -> [unit_indices]
            
            # 토큰이 어느 유닛에 속하는지 매핑
            for unit_idx, unit in enumerate(units):
                tokens = unit.split('_')
                
                # 단일 유닛인 경우 continue
                if len(tokens) == 1: continue
                
                for token in tokens:
                    if token not in unit_positions:
                        unit_positions[token] = []
                    unit_positions[token].append(unit_idx)
            
            # 각 토큰에 대해 Token-Unit 페어 생성
            for token, token_unit_indices in unit_positions.items():
                for token_unit_idx in token_unit_indices:
                    # 윈도우 범위 계산 (유닛 기준)
                    start = max(0, token_unit_idx - window_size)
                    end = min(len(units), token_unit_idx + window_size + 1)
                    
                    # 윈도우 범위 내의 모든 유닛과 페어 생성
                    for context_unit_idx in range(start, end):
                        if token_unit_idx != context_unit_idx:  # 자기 자신은 제외
                            context_unit = units[context_unit_idx]
                            training_pairs.append((token, context_unit))
                            token_unit_pairs += 1
            
            # 2) Unit-Unit 페어 생성
            for unit_idx, unit in enumerate(units):
                # 윈도우 범위 계산 (유닛 기준)
                start = max(0, unit_idx - window_size)
                end = min(len(units), unit_idx + window_size + 1)
                
                # 윈도우 범위 내의 다른 유닛들과 페어 생성
                for context_unit_idx in range(start, end):
                    if unit_idx != context_unit_idx:  # 자기 자신은 제외
                        context_unit = units[context_unit_idx]
                        training_pairs.append((unit, context_unit))
                        unit_unit_pairs += 1
        
        print(f"FastText 스타일 학습 페어 생성 완료:")
        print(f"  Token-Unit 페어: {token_unit_pairs:,}개")
        print(f"  Unit-Unit 페어: {unit_unit_pairs:,}개") 
        print(f"  총 페어 수: {len(training_pairs):,}개")
        return training_pairs


# 사용 예시
if __name__ == "__main__":
    # 테스트용 데이터 생성
    test_data = """start toolbar_menu toolbar_ss next button_end
click file_dialog select_option confirm_action close
open menu_edit copy_text paste_text save_file"""
    
    # 테스트 파일 생성
    with open("test_data.txt", "w", encoding="utf-8") as f:
        f.write(test_data)
    
    # 데이터 로더 초기화
    loader = ActionDataLoader()
    
    # 파일에서 데이터 로드
    loader.load_from_file("test_data.txt")
    
    # 토큰화 수행
    loader.tokenize_sequences()
    
    # Token-Unit 학습 페어 생성
    pairs = loader.get_training_pairs(window_size=1)
    
    # 결과 출력
    print(f"\n=== 첫 번째 시퀀스 분석 ===")
    print(f"원본 시퀀스: {loader.sequences[0]}")
    print(f"액션 유닛들: {loader.action_units[0]}")
    
    # 첫 번째 시퀀스에 존재하는 unit이 context로 있는 학습 쌍 출력
    first_seq_pairs = []
    first_seq_units = loader.action_units[0]
    
    for token, unit in pairs:
        if unit in first_seq_units:
            first_seq_pairs.append((token, unit))
    
    print(f"생성된 Token-Unit 페어들:")
    for token, unit in sorted(set(first_seq_pairs)):
        print(f"  ('{token}', '{unit}')")
    
    # 통계 정보 출력
    stats = loader.get_vocabulary_stats()
    print(f"\n=== 통계 정보 ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")