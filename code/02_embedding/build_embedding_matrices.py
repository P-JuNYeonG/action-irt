"""
여러 문제 번호에 대해 (SEQID, problem_num) 키로 임베딩 행렬을 추출하는 코드
timestamp 정보 포함 버전
문항별로 시간 통계량을 분리해서 집계한 후, 임베딩 스케일에 맞춰 시간 피처를 변환하여 결합하는 방식
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
    """
    모델 로드 및 vocab_builder 재구축
    """
    print("모델 로드 중...")
    model = Word2VecModel.load_model(model_path)
    
    print("어휘 재구축 중...")
    data_loader = ActionDataLoader()
    data_loader.load_from_file(data_path)
    data_loader.tokenize_sequences()
    
    vocab_builder = VocabularyBuilder(min_count=min_count)
    vocab_builder.build_vocabulary(data_loader.token_counter, data_loader.action_units)
    
    print(f"모델 로드 완료: 토큰 {vocab_builder.vocab_size}개, 유닛 {vocab_builder.unit_vocab_size}개")
    
    return model, vocab_builder


def create_timestamp_features(timestamps: np.ndarray) -> np.ndarray:
    """
    timestamp로부터 [t, t^2, 시간_간격] 원본 특성 생성 (스케일링 미적용)
    스케일링은 임베딩 통계량 확보 후 별도로 수행
    
    Parameters:
        timestamps (np.ndarray): 시간 값 배열
    
    Returns:
        np.ndarray: shape (n, 3) - [t, t^2, delta_t] (원본 스케일)
    """
    timestamps = np.array(timestamps, dtype=np.float64)
    
    t = timestamps.copy()
    
    # t^2 값
    t_squared = t ** 2
    
    # 시간 간격 계산
    time_deltas = np.zeros_like(timestamps)
    time_deltas[1:] = np.diff(timestamps)
    time_deltas[0] = timestamps[0]  # 첫 번째는 원본 timestamp 값으로 설정
    
    # [t, t^2, delta_t] 형태로 결합
    timestamp_features = np.column_stack([t, t_squared, time_deltas])
    
    return timestamp_features


def scale_time_features_to_embedding(time_features: np.ndarray, 
                                      emb_mean: float, 
                                      emb_std: float,
                                      time_means: np.ndarray,
                                      time_stds: np.ndarray) -> np.ndarray:
    """
    시간 피처를 임베딩 공간의 스케일에 맞춰 변환
    
    변환 과정:
        1. 각 시간 피처를 문제별 통계량으로 z-score 표준화 (평균 0, 표준편차 1)
        2. 임베딩의 표준편차(σ_emb)를 곱하고 평균(μ_emb)을 더함
        => 시간 피처가 임베딩과 동일한 분포 스케일을 가지게 됨
    
    Parameters:
        time_features (np.ndarray): shape (n, 3) - 원본 시간 피처
        emb_mean (float): 해당 문제의 임베딩 전체 평균 (μ_emb)
        emb_std (float): 해당 문제의 임베딩 전체 표준편차 (σ_emb)
        time_means (np.ndarray): shape (3,) - 해당 문제의 시간 피처별 평균
        time_stds (np.ndarray): shape (3,) - 해당 문제의 시간 피처별 표준편차
    
    Returns:
        np.ndarray: shape (n, 3) - 임베딩 스케일에 맞춰 변환된 시간 피처
    """
    # 표준편차가 0인 경우 방지 (상수 피처)
    safe_time_stds = np.where(time_stds > 0, time_stds, 1.0)
    
    # Step 1: z-score 표준화 (문제별 시간 통계량 기준)
    z_scored = (time_features - time_means) / safe_time_stds
    
    # Step 2: 임베딩 스케일로 변환
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
    SEQID별로 action unit들의 임베딩 행렬을 생성 (timestamp 정보 포함)
    
    Two-pass 구조:
        Pass 1: 임베딩과 시간 피처를 각각 생성하고, 문제별 통계량 수집
        Pass 2: 시간 피처를 임베딩 스케일에 맞춰 변환 후 결합
    
    Parameters:
        include_timestamp (bool): timestamp 특성을 포함할지 여부
        scale_to_embedding (bool): 시간 피처를 임베딩 스케일에 맞출지 여부
    """
    grouped = df.groupby('SEQID')
    
    # ========== Pass 1: 임베딩 & 시간 피처 생성, 통계량 수집 ==========
    emb_dict = {}       # {seqid: embedding_matrix (n, dim)}
    time_dict = {}      # {seqid: time_features (n, 3)}
    
    all_embeddings = []  # 문제 전체 임베딩 값 수집 (통계량 계산용)
    all_time_features = []  # 문제 전체 시간 피처 수집 (통계량 계산용)
    
    for seqid, group in grouped:
        group = group.sort_values(by=timestamp_column)
        
        processed_events = group[processed_event_column].tolist()
        timestamps = group[timestamp_column].values
        
        embeddings = []
        for processed_event in processed_events:
            event_str = str(processed_event)
            embedding = model.get_fasttext_unit_embedding(event_str, vocab_builder)
            if embedding is not None:
                embeddings.append(embedding)
        
        if embeddings:
            embedding_matrix = np.stack(embeddings)
            emb_dict[seqid] = embedding_matrix
            all_embeddings.append(embedding_matrix)
            
            if include_timestamp:
                time_features = create_timestamp_features(timestamps)
                time_dict[seqid] = time_features
                all_time_features.append(time_features)
    
    # ========== Pass 2: 스케일링 및 결합 ==========
    embedding_matrices = {}
    
    if include_timestamp and scale_to_embedding and all_embeddings and all_time_features:
        # 문제별 임베딩 통계량 계산 (전체 차원을 하나로 펼쳐서)
        all_emb_concat = np.concatenate(all_embeddings, axis=0)  # (total_actions, embedding_dim)
        emb_mean = all_emb_concat.mean()    # 스칼라: 전체 임베딩의 평균
        emb_std = all_emb_concat.std()      # 스칼라: 전체 임베딩의 표준편차
        
        # 문제별 시간 피처 통계량 계산 (각 피처별로)
        all_time_concat = np.concatenate(all_time_features, axis=0)  # (total_actions, 3)
        time_means = all_time_concat.mean(axis=0)  # shape (3,)
        time_stds = all_time_concat.std(axis=0)    # shape (3,)
        
        print(f"  [스케일링 정보] 임베딩 μ={emb_mean:.4f}, σ={emb_std:.4f}")
        print(f"  [스케일링 정보] 시간 피처 μ={time_means}, σ={time_stds}")
        
        for seqid in emb_dict:
            scaled_time = scale_time_features_to_embedding(
                time_dict[seqid], emb_mean, emb_std, time_means, time_stds
            )
            embedding_matrices[seqid] = np.concatenate([emb_dict[seqid], scaled_time], axis=1)
    
    elif include_timestamp and not scale_to_embedding:
        # 스케일링 없이 원본 시간 피처를 그대로 결합
        for seqid in emb_dict:
            embedding_matrices[seqid] = np.concatenate([emb_dict[seqid], time_dict[seqid]], axis=1)
    
    else:
        # 시간 피처 미포함: 임베딩만 반환
        embedding_matrices = emb_dict
    
    return embedding_matrices


def extract_multi_problem_embeddings(problem_nums: list, 
                                    base_data_path: str,
                                    base_model_path: str,
                                    base_test_path: str,
                                    include_timestamp: bool = True,
                                    scale_to_embedding: bool = True) -> Dict[tuple, np.ndarray]:
    """
    여러 문제에 대해 (SEQID, problem_num) 키로 임베딩 행렬을 추출
    
    Parameters:
        problem_nums (list): 문제 번호 리스트 (예: ['ps1_1', 'ps1_2', 'ps2_7'])
        base_data_path (str): 데이터 파일 경로 템플릿
        base_model_path (str): 모델 파일 경로 템플릿
        base_test_path (str): 테스트 데이터 경로 템플릿
        include_timestamp (bool): timestamp 특성 포함 여부
        scale_to_embedding (bool): 시간 피처를 임베딩 스케일에 맞출지 여부
        
    Returns:
        Dict[tuple, np.ndarray]: {(SEQID, problem_num): embedding_matrix}
    """
    
    result_dict = {}
    
    for problem_num in problem_nums:
        print(f"\n{'='*60}")
        print(f"처리 중: {problem_num}")
        print(f"{'='*60}")
        
        # 파일 경로 생성
        data_path = base_data_path.format(problem_num=problem_num)
        model_path = base_model_path.format(problem_num=problem_num)
        test_path = base_test_path.format(problem_num=problem_num)
        
        try:
            # 모델 및 어휘 로드
            model, vocab_builder = load_model_and_rebuild_vocab(model_path, test_path)
            
            # 데이터 로드
            df = pd.read_pickle(data_path)
            
            # 임베딩 행렬 생성 (문제별 스케일링 적용)
            matrices = create_embedding_matrices_by_seqid(
                df, 'processed_event', 'timestamp', model, vocab_builder,
                include_timestamp=include_timestamp,
                scale_to_embedding=scale_to_embedding
            )
            
            # (SEQID, problem_num) 키로 결과 저장
            for seqid, matrix in matrices.items():
                key = (seqid, problem_num)
                result_dict[key] = matrix
            
            print(f"✓ {problem_num}: {len(matrices)}개 SEQID 처리 완료")
            if matrices:
                sample_shape = list(matrices.values())[0].shape
                timestamp_status = "포함" if include_timestamp else "미포함"
                print(f"  샘플 행렬 shape: {sample_shape} (timestamp {timestamp_status})")
            
        except Exception as e:
            print(f"✗ {problem_num} 처리 중 오류 발생: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print(f"전체 처리 완료: 총 {len(result_dict)}개 항목")
    print(f"{'='*60}")
    
    return result_dict


def all_items() -> list[str]:
    return [f"ps{i}_{j}" for i, j in product(range(1, 3), range(1, 8))]


def save_grouped_outputs(embedding_dict: Dict[tuple, np.ndarray], output_dir: Path, suffix: str) -> None:
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-suffix", default="20")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--no-scale-to-embedding", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedding_dict = extract_multi_problem_embeddings(
        problem_nums=args.items,
        base_data_path=args.data_template,
        base_model_path=args.model_template,
        base_test_path=args.sequence_template,
        include_timestamp=not args.no_timestamp,
        scale_to_embedding=not args.no_scale_to_embedding,
    )
    save_grouped_outputs(embedding_dict, args.output_dir, args.output_suffix)


if __name__ == "__main__":
    main()
