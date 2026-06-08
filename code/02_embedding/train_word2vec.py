"""
Hybrid 스타일 Token-Unit Word2Vec 학습 메인 스크립트
토큰과 유닛 모두를 중심 아이템으로 사용하는 학습 파이프라인
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
    """명령행 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="Hybrid 스타일 Token-Unit Word2Vec 모델 학습 스크립트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 필수 인자
    parser.add_argument("--input_file", "-i", type=str, required=True,
                        help="입력 데이터 파일 경로 (.txt)")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="출력 디렉토리 경로")
    
    # 모델 하이퍼파라미터 (Hybrid 스타일)
    parser.add_argument("--embed_dim", type=int, default=20,
                        help="임베딩 벡터 차원")
    parser.add_argument("--window_size", type=int, default=1,
                        help="유닛 기준 컨텍스트 윈도우 크기")
    parser.add_argument("--min_count", type=int, default=1,
                        help="어휘에 포함할 최소 빈도수")
    parser.add_argument("--negative_samples", type=int, default=5,
                        help="음성 샘플 개수")
    
    # 학습 하이퍼파라미터
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Adam 옵티마이저 학습률")
    parser.add_argument("--epochs", type=int, default=100,
                        help="학습 에포크 수")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="재현 가능한 결과를 위한 랜덤 시드")
    
    # Early stopping 옵션 추가
    parser.add_argument("--early_stopping", action="store_true",
                        help="Early stopping 사용 여부")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (기본값: 3)")
    
    # Train/Valid 분리 옵션 추가
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Train/Valid 분리 비율 (기본값: 0.8)")
    parser.add_argument("--no_validation", action="store_true",
                        help="Validation 사용 안 함 (전체 데이터로 학습)")
    
    return parser.parse_args()


def setup_output_directory(output_dir: str) -> Dict[str, str]:
    """출력 디렉토리를 설정하고 파일 경로들을 반환합니다."""
    os.makedirs(output_dir, exist_ok=True)
    
    paths = {
        'model': os.path.join(output_dir, 'fasttext_style_word2vec.pkl'),
        'vocabulary': os.path.join(output_dir, 'vocabulary.txt'),
        'training_log': os.path.join(output_dir, 'training_log.json'),
        'config': os.path.join(output_dir, 'config.json'),
        'token_embeddings': os.path.join(output_dir, 'token_embeddings.npz'),
        'unit_embeddings': os.path.join(output_dir, 'unit_embeddings_fasttext.npz'),
        'fasttext_analysis': os.path.join(output_dir, 'fasttext_comprehensive_analysis.json'),
        'token_tsv': os.path.join(output_dir, 'token_embeddings.tsv'),
        'token_metadata_tsv': os.path.join(output_dir, 'token_metadata.tsv'),
        'unit_tsv': os.path.join(output_dir, 'unit_embeddings_fasttext.tsv'),
        'unit_metadata_tsv': os.path.join(output_dir, 'unit_metadata_fasttext.tsv')
    }
    
    return paths


def save_configuration(config: Dict[str, Any], config_path: str) -> None:
    """학습 설정을 JSON 파일로 저장합니다."""
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"설정 파일 저장: {config_path}")


def save_fasttext_embeddings(model: Word2VecModel, vocab_builder: VocabularyBuilder,
                            token_file: str, unit_file: str) -> None:
    """FastText 스타일 토큰과 유닛 임베딩을 저장합니다."""
    # 토큰 임베딩 저장 (직접 저장)
    token_embeddings = []
    token_names = []
    
    for idx in range(vocab_builder.vocab_size):
        token = vocab_builder.get_token_by_index(idx)
        embedding = model.get_token_embedding(idx)
        token_embeddings.append(embedding)
        token_names.append(token)
    
    np.savez_compressed(token_file, 
                       embeddings=np.stack(token_embeddings),
                       tokens=np.array(token_names),
                       embed_dim=model.embed_dim,
                       model_type='fasttext_style_token')
    
    # 유닛 임베딩 저장 (FastText 방식으로 계산)
    unit_embeddings = []
    unit_names = []
    unit_compositions = []
    
    for idx in range(vocab_builder.unit_vocab_size):
        unit = vocab_builder.get_unit_by_index(idx)
        # FastText 방식으로 유닛 임베딩 계산
        embedding = model.get_fasttext_unit_embedding(unit, vocab_builder)
        if embedding is not None:
            unit_embeddings.append(embedding)
            unit_names.append(unit)
            
            # 구성 정보도 저장
            composition = vocab_builder.get_unit_composition(unit)
            comp_info = []
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
    
    print(f"FastText 스타일 임베딩 저장 완료:")
    print(f"  토큰 임베딩: {token_file} ({len(token_names)}개)")
    print(f"  유닛 임베딩: {unit_file} ({len(unit_names)}개, FastText 방식)")
    print(f"  유닛 구성 정보도 함께 저장됨")



def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("FastText 스타일 Token-Unit Word2Vec 모델 학습 시작")
    print("unit_embedding = sum([token_embeddings, unit_embedding])")
    print("=" * 70)
    
    # 1. 명령행 인자 파싱
    args = parse_arguments()
    
    # 2. 출력 디렉토리 설정
    file_paths = setup_output_directory(args.output_dir)
    
    # 3. 설정 저장
    config = {
        'input_file': args.input_file,
        'output_dir': args.output_dir,
        'model_type': 'fasttext_style_token_unit',
        'model_description': 'FastText style: unit_embedding = sum([token_embeddings, unit_embedding])',
        'learning_approach': 'Token-Unit and Unit-Unit pairs with FastText gradient distribution',
        'model_hyperparameters': {
            'embed_dim': args.embed_dim,
            'window_size': args.window_size,
            'min_count': args.min_count,
            'negative_samples': args.negative_samples,
            'random_seed': args.random_seed
        },
        'training_hyperparameters': {
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'early_stopping': args.early_stopping,
            'patience': args.patience
        },
        'data_split_options': {
            'use_validation': not args.no_validation,
            'train_ratio': args.train_ratio if not args.no_validation else 1.0
        }
    }
    save_configuration(config, file_paths['config'])
    
    # 4. 데이터 로딩 및 전처리
    print("\n--- 1단계: 데이터 로딩 및 전처리 ---")
    start_time = time.time()
    
    # 데이터 로더 초기화 및 데이터 로드
    data_loader = ActionDataLoader()
    data_loader.load_from_file(args.input_file)
    data_loader.tokenize_sequences()
    
    # 데이터 통계 출력
    data_stats = data_loader.get_vocabulary_stats()
    print(f"데이터 로딩 완료 ({time.time() - start_time:.1f}초):")
    for key, value in data_stats.items():
        print(f"  {key}: {value}")
    
    # 5. 전체 데이터로 어휘 구축
    print("\n--- 2단계: 전체 데이터로 어휘 구축 ---")
    start_time = time.time()
    
    vocab_builder = VocabularyBuilder(min_count=args.min_count)
    vocab_builder.build_vocabulary(data_loader.token_counter, data_loader.action_units)
    vocab_builder.save_vocabulary(file_paths['vocabulary'])
    
    vocab_info = vocab_builder.get_vocabulary_info()
    print(f"FastText 스타일 어휘 구축 완료 ({time.time() - start_time:.1f}초):")
    print(f"  토큰 어휘 크기: {vocab_info['token_vocab_size']:,}")
    print(f"  유닛 어휘 크기: {vocab_info['unit_vocab_size']:,}")
    
    # 6. Train/Valid 분리 및 페어 생성
    if not args.no_validation:
        print("\n--- 3단계: Train/Valid 분리 및 페어 생성 ---")
        start_time = time.time()
        
        # 6-1. 시퀀스 분리
        train_sequences, valid_sequences = data_loader.split_sequences(
            train_ratio=args.train_ratio,
            random_seed=args.random_seed
        )
        
        # 6-2. Train 페어 생성
        print("\n[Train 페어 생성]")
        train_pairs = data_loader.get_training_pairs_from_sequences(
            train_sequences, 
            window_size=args.window_size
        )
        train_pairs_filtered = vocab_builder.filter_training_pairs(train_pairs)
        
        # 6-3. Valid 페어 생성
        print("\n[Valid 페어 생성]")
        valid_pairs = data_loader.get_training_pairs_from_sequences(
            valid_sequences, 
            window_size=args.window_size
        )
        valid_pairs_filtered = vocab_builder.filter_training_pairs(valid_pairs)
        
        print(f"\n페어 생성 완료 ({time.time() - start_time:.1f}초):")
        print(f"  Train 페어: {len(train_pairs_filtered):,}개")
        print(f"  Valid 페어: {len(valid_pairs_filtered):,}개")
        
    else:
        print("\n--- 3단계: 전체 데이터로 페어 생성 (No Validation) ---")
        start_time = time.time()
        
        train_pairs = data_loader.get_training_pairs(window_size=args.window_size)
        train_pairs_filtered = vocab_builder.filter_training_pairs(train_pairs)
        valid_pairs_filtered = None
        
        train_sequences = data_loader.sequences
        valid_sequences = []
        
        print(f"\n페어 생성 완료 ({time.time() - start_time:.1f}초):")
        print(f"  전체 페어: {len(train_pairs_filtered):,}개")
    
    # 7. FastText 스타일 모델 초기화
    print("\n--- 4단계: FastText 스타일 모델 초기화 ---")
    model = Word2VecModel(
        token_vocab_size=vocab_builder.vocab_size,
        unit_vocab_size=vocab_builder.unit_vocab_size,
        embed_dim=args.embed_dim,
        random_seed=args.random_seed
    )
    
    # 모델 정보 출력
    model_info = model.get_model_info()
    print(f"모델 파라미터 수: {model_info['total_parameters']:,}")
    
    # 8. 학습기 초기화 및 학습
    print("\n--- 5단계: FastText 스타일 모델 학습 ---")
    start_time = time.time()
    
    trainer = Word2VecTrainer(
        model=model,
        vocab_builder=vocab_builder,
        learning_rate=args.learning_rate,
        negative_samples=args.negative_samples,
        early_stopping=args.early_stopping,
        patience=args.patience
    )
    
    # 학습 실행 (validation_pairs 전달)
    training_history = trainer.train(
        training_pairs=train_pairs_filtered,
        validation_pairs=valid_pairs_filtered,  # None일 수 있음
        epochs=args.epochs,
        save_path=file_paths['model']
    )
    
    total_training_time = time.time() - start_time
    print(f"\n학습 완료 (총 {total_training_time:.1f}초)")
    
    # 8. 학습 로그 저장
    print("\n--- 5단계: 학습 로그 저장 ---")
    
    # 타입별 페어 수 계산
    token_unit_pairs = sum(1 for center, _ in train_pairs_filtered 
                           if center in vocab_builder.token_to_idx and 
                              center not in vocab_builder.unit_to_idx)
    unit_unit_pairs = sum(1 for center, _ in train_pairs_filtered 
                          if center in vocab_builder.unit_to_idx)
    
    training_log = {
        'model_type': 'fasttext_style_token_unit',
        'training_history': training_history,
        'total_training_time': total_training_time,
        'final_train_loss': training_history[-1]['avg_loss'] if training_history else 0.0,
        'final_valid_loss': training_history[-1].get('valid_loss', None) if training_history else None,
        'data_split': {
            'use_validation': not args.no_validation,
            'train_ratio': args.train_ratio if not args.no_validation else 1.0,
            'train_sequences': len(train_sequences),
            'valid_sequences': len(valid_sequences),
            'train_pairs': len(train_pairs_filtered),
            'valid_pairs': len(valid_pairs_filtered) if valid_pairs_filtered else 0,
            'train_token_unit_pairs': token_unit_pairs,
            'train_unit_unit_pairs': unit_unit_pairs
        },
        'data_stats': data_stats,
        'vocab_info': vocab_info,
        'model_info': model_info,
        'config': config,
        'fasttext_characteristics': {
            'unit_composition_method': 'sum([token_embeddings, unit_embedding])',
            'learning_pairs': 'Token-Unit + Unit-Unit',
            'gradient_distribution': 'FastText style',
            'semantic_approach': 'Compositional + Holistic'
        }
    }
    
    with open(file_paths['training_log'], 'w', encoding='utf-8') as f:
        json.dump(training_log, f, ensure_ascii=False, indent=2)
    print(f"학습 로그 저장: {file_paths['training_log']}")
    
    # 9. FastText 스타일 임베딩 저장
    print("\n--- 6단계: FastText 스타일 임베딩 저장 ---")
    start_time = time.time()
    
    save_fasttext_embeddings(
        model=model,
        vocab_builder=vocab_builder,
        token_file=file_paths['token_embeddings'],
        unit_file=file_paths['unit_embeddings']
    )
    
    print(f"임베딩 저장 완료 ({time.time() - start_time:.1f}초)")

    # 10. 학습 결과 종합 요약
    print("\n" + "=" * 70)
    print("학습 완료 요약")
    print("=" * 70)
    
    print(f"\n데이터:")
    print(f"  총 시퀀스 수: {data_stats['total_sequences']:,}")
    if not args.no_validation:
        print(f"  Train 시퀀스: {len(train_sequences):,} ({args.train_ratio*100:.0f}%)")
        print(f"  Valid 시퀀스: {len(valid_sequences):,} ({(1-args.train_ratio)*100:.0f}%)")
    
    print(f"\n어휘:")
    print(f"  토큰 어휘 크기: {vocab_info['token_vocab_size']:,}")
    print(f"  유닛 어휘 크기: {vocab_info['unit_vocab_size']:,}")
    
    print(f"\n학습 페어:")
    print(f"  Train 페어: {len(train_pairs_filtered):,}개")
    print(f"    - Token-Unit: {token_unit_pairs:,}개")
    print(f"    - Unit-Unit: {unit_unit_pairs:,}개")
    if valid_pairs_filtered:
        print(f"  Valid 페어: {len(valid_pairs_filtered):,}개")
    
    print(f"\n최종 손실:")
    print(f"  Train Loss: {training_history[-1]['avg_loss']:.6f}" if training_history else "N/A")
    if training_history and training_history[-1].get('valid_loss'):
        print(f"  Valid Loss: {training_history[-1]['valid_loss']:.6f}")
    
    print(f"\n저장된 파일:")
    print(f"  모델: {file_paths['model']}")
    print(f"  어휘: {file_paths['vocabulary']}")
    print(f"  학습 로그: {file_paths['training_log']}")
    print(f"  설정 파일: {file_paths['config']}")
    print(f"  토큰 임베딩: {file_paths['token_embeddings']}")
    print(f"  유닛 임베딩: {file_paths['unit_embeddings']}")
    
    print("=" * 70)


'''
사용 예시:

# Validation 사용 (기본)
python modified_main.py \
    --input_file data.txt \
    --output_dir output/ \
    --train_ratio 0.8 \
    --embed_dim 50 \
    --window_size 1 \
    --epochs 10 \
    --early_stopping \
    --patience 3

# Validation 사용 안 함 (전체 데이터 학습)
python modified_main.py \
    --input_file data.txt \
    --output_dir output/ \
    --no_validation \
    --epochs 50
'''


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n학습이 사용자에 의해 중단되었습니다.")
    except FileNotFoundError as e:
        print(f"\n\n파일을 찾을 수 없습니다: {e}")
        print("입력 파일 경로를 확인해주세요.")
    except Exception as e:
        print(f"\n\n오류가 발생했습니다: {str(e)}")
        print("\n상세 오류 정보:")
        import traceback
        traceback.print_exc()
    finally:
        print("\n프로그램을 종료합니다.")
