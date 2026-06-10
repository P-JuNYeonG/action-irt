"""
LSTM AutoEncoder Dimension Reduction for Action Embedding Matrices
==================================================================
- Input: {(SEQ_ID, problem_num): Nij x input_dim numpy array} pickle
- Output: {D: {(SEQ_ID, problem_num): Nij x D numpy array}} pickle
- 문제(problem_num)별로 data_list를 구성하여 별도 LSTM AE 학습
- D = [1, 2, 3, 4, 5] 비교
- StandardScaler 미적용 (시간 정보가 이미 임베딩 스케일로 조정됨)
- pack_padded_sequence로 padding의 bias term 오염 방지
- 문제별 학습 곡선(loss curve) 시각화 포함
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from typing import Dict, Tuple, List, Optional, Sequence
import os
import pickle
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. 시퀀스 Dataset (padding + 길이 정보 포함)
# ============================================================
class SequenceDataset(Dataset):
    """
    가변 길이 시퀀스를 배치로 묶기 위한 Dataset.
    collate_fn에서 padding을 수행한다.
    """

    def __init__(self, sequences: List[np.ndarray]):
        """
        Parameters
        ----------
        sequences : list of np.ndarray
            각 원소가 Nij × input_dim 행렬인 리스트
        """
        self.sequences = [
            torch.FloatTensor(seq.astype(np.float32)) for seq in sequences
        ]
        self.lengths = [seq.shape[0] for seq in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.lengths[idx]


def collate_fn(batch):
    """
    가변 길이 시퀀스를 padding하여 배치로 묶는 함수.
    길이 내림차순으로 정렬 (pack_padded_sequence 요구사항).

    Returns
    -------
    padded : (batch_size, max_len, input_dim)
    lengths : list of int (내림차순 정렬)
    sort_indices : 원래 순서 복원용 인덱스
    """
    # 길이 내림차순 정렬
    batch.sort(key=lambda x: x[1], reverse=True)
    sequences, lengths = zip(*batch)

    max_len = lengths[0]  # 정렬 후 첫 번째가 최장
    input_dim = sequences[0].shape[1]
    batch_size = len(sequences)

    padded = torch.zeros(batch_size, max_len, input_dim)
    for i, seq in enumerate(sequences):
        padded[i, :seq.shape[0], :] = seq

    return padded, list(lengths)


# ============================================================
# 2. LSTM AutoEncoder 모델 정의
# ============================================================
class LSTMAutoEncoder(nn.Module):
    """
    LSTM AutoEncoder for Action Embedding Sequences.

    구조:
        Encoder: LSTM(input=input_dim, hidden=64) -> FC(64 -> D)
        Decoder: FC(D -> 64) -> LSTM(input=64, hidden=64) -> FC(64 -> input_dim)

    - pack_padded_sequence를 사용하여 padding 위치에서
      LSTM 연산을 수행하지 않음 (bias term 오염 방지)
    - 시점별(time-step-wise) latent 출력: Nij x D
    """

    def __init__(self, input_dim: int = 23, hidden_dim: int = 64, latent_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_fc_in = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.decoder_fc_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x_packed, lengths):
        """
        Parameters
        ----------
        x_packed : PackedSequence 또는 (batch_size, max_len, input_dim)
        lengths : list of int

        Returns
        -------
        latent_padded : (batch_size, max_len, latent_dim)
        """
        # Encoder LSTM
        if isinstance(x_packed, torch.nn.utils.rnn.PackedSequence):
            enc_out_packed, _ = self.encoder_lstm(x_packed)
        else:
            packed = pack_padded_sequence(x_packed, lengths, batch_first=True, enforce_sorted=True)
            enc_out_packed, _ = self.encoder_lstm(packed)

        # Unpack → (batch_size, max_len, hidden_dim)
        enc_out_padded, _ = pad_packed_sequence(enc_out_packed, batch_first=True)

        # FC → latent (batch_size, max_len, latent_dim)
        latent_padded = self.encoder_fc(enc_out_padded)

        return latent_padded

    def decode(self, latent_padded, lengths):
        """
        Parameters
        ----------
        latent_padded : (batch_size, max_len, latent_dim)
        lengths : list of int

        Returns
        -------
        recon_padded : (batch_size, max_len, input_dim)
        """
        # FC → hidden dim
        dec_input = torch.relu(self.decoder_fc_in(latent_padded))

        # Pack → Decoder LSTM → Unpack
        dec_packed = pack_padded_sequence(dec_input, lengths, batch_first=True, enforce_sorted=True)
        dec_out_packed, _ = self.decoder_lstm(dec_packed)
        dec_out_padded, _ = pad_packed_sequence(dec_out_packed, batch_first=True)

        # FC → reconstruction
        recon_padded = self.decoder_fc_out(dec_out_padded)

        return recon_padded

    def forward(self, x_padded, lengths):
        """
        Parameters
        ----------
        x_padded : (batch_size, max_len, input_dim)
        lengths : list of int

        Returns
        -------
        recon_padded : (batch_size, max_len, input_dim)
        latent_padded : (batch_size, max_len, latent_dim)
        """
        # Pack input
        x_packed = pack_padded_sequence(x_padded, lengths, batch_first=True, enforce_sorted=True)

        latent_padded = self.encode(x_packed, lengths)
        recon_padded = self.decode(latent_padded, lengths)

        return recon_padded, latent_padded


# ============================================================
# 3. Masked MSE Loss
# ============================================================
def masked_mse_loss(recon, target, lengths):
    """
    Padding 위치를 제외한 MSE loss 계산.

    Parameters
    ----------
    recon : (batch_size, max_len, input_dim)
    target : (batch_size, max_len, input_dim)
    lengths : list of int
    """
    batch_size, max_len, input_dim = target.shape

    # mask: (batch_size, max_len, 1) → broadcast over input_dim
    mask = torch.zeros(batch_size, max_len, 1, device=target.device)
    for i, length in enumerate(lengths):
        mask[i, :length, :] = 1.0

    # MSE only on valid positions
    sq_error = (recon - target) ** 2 * mask
    loss = sq_error.sum() / (mask.sum() * input_dim)

    return loss


# ============================================================
# 4. LSTM AE Reducer 클래스 (단일 문제용)
# ============================================================
class LSTMAEReducer:
    """
    단일 문제에 대해 LSTM AutoEncoder 학습/변환을 수행하는 클래스.
    인터페이스: PCAReducer, MLPAEReducer와 동일 (fit, transform)
    """

    def __init__(self,
                 D: int = 1,
                 input_dim: int = 23,
                 hidden_dim: int = 64,
                 epochs: int = 200,
                 batch_size: int = 32,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 patience: int = 20,
                 device: Optional[str] = None,
                 seed: int = 42):
        """
        Parameters
        ----------
        D : int
            latent dimension
        input_dim : int
            입력 차원
        hidden_dim : int
            LSTM hidden state 차원 (64)
        epochs : int
            최대 학습 에포크
        batch_size : int
            배치 크기 (시퀀스 단위이므로 MLP AE보다 작게 설정)
        lr : float
            학습률
        weight_decay : float
            L2 정규화
        patience : int
            Early stopping patience
        device : str, optional
            'cuda' or 'cpu'
        seed : int
            재현성 시드
        """
        self.D = D
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.seed = seed

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model: Optional[LSTMAutoEncoder] = None
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_epoch: int = 0

    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def fit(self, data_list: List[np.ndarray], val_ratio: float = 0.1) -> None:
        """
        Parameters
        ----------
        data_list : list of np.ndarray
            각 원소가 Nij x input_dim 행렬인 리스트 (시퀀스 단위)
        val_ratio : float
            validation set 비율
        """
        self._set_seed()

        n_sequences = len(data_list)
        if n_sequences == 0:
            raise ValueError("data_list is empty.")

        # Train / Validation 분리 (시퀀스 단위)
        indices = np.random.permutation(n_sequences)
        if n_sequences == 1:
            val_indices = indices
            train_indices = indices
        else:
            n_val = max(1, int(n_sequences * val_ratio))
            n_val = min(n_val, n_sequences - 1)
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

        train_sequences = [data_list[i] for i in train_indices]
        val_sequences = [data_list[i] for i in val_indices]

        train_dataset = SequenceDataset(train_sequences)
        val_dataset = SequenceDataset(val_sequences)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=False
        )

        # 모델 초기화
        self.model = LSTMAutoEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.D
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # 학습
        self.train_losses = []
        self.val_losses = []
        best_val_loss = float('inf')
        best_state_dict = None
        patience_counter = 0

        for epoch in range(self.epochs):
            # --- Train ---
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for padded, lengths in train_loader:
                padded = padded.to(self.device)
                recon, _ = self.model(padded, lengths)
                loss = masked_mse_loss(recon, padded, lengths)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            train_loss = epoch_loss / n_batches
            self.train_losses.append(train_loss)

            # --- Validation ---
            self.model.eval()
            with torch.no_grad():
                val_loss = self._evaluate(val_dataset)
            self.val_losses.append(val_loss)

            # --- Early Stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"    Early stopping at epoch {epoch+1} "
                      f"(best epoch: {self.best_epoch}, best val loss: {best_val_loss:.6f})")
                break

        # best 모델 복원
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            self.model.to(self.device)

        total_actions = sum(seq.shape[0] for seq in data_list)
        print(f"    학습 완료 | 시퀀스 수: {n_sequences} (train: {len(train_indices)}, "
              f"val: {len(val_indices)}) | 총 행동 수: {total_actions} | "
              f"best epoch: {self.best_epoch} | "
              f"train loss: {self.train_losses[self.best_epoch-1]:.6f} | "
              f"val loss: {best_val_loss:.6f}")

    def _evaluate(self, dataset: SequenceDataset) -> float:
        """Validation set에 대한 masked MSE 계산"""
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False
        )

        total_loss = 0.0
        n_batches = 0

        for padded, lengths in loader:
            padded = padded.to(self.device)
            recon, _ = self.model(padded, lengths)
            loss = masked_mse_loss(recon, padded, lengths)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def transform(self, data_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Parameters
        ----------
        data_list : list of np.ndarray
            각 원소가 Nij x input_dim 행렬인 리스트

        Returns
        -------
        list of np.ndarray
            각 원소가 Nij x D 행렬인 리스트 (입력과 동일한 순서)
        """
        if self.model is None:
            raise ValueError("fit()을 먼저 수행하세요.")

        self.model.eval()
        results = []

        with torch.no_grad():
            for matrix in data_list:
                # 단일 시퀀스를 배치 차원 추가: (1, Nij, input_dim)
                x = torch.FloatTensor(matrix.astype(np.float32)).unsqueeze(0).to(self.device)
                lengths = [matrix.shape[0]]

                x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
                latent = self.model.encode(x_packed, lengths)

                # (1, Nij, D) -> (Nij, D)
                results.append(latent.squeeze(0).cpu().numpy())

        return results

    def fit_transform(self, data_list: List[np.ndarray]) -> List[np.ndarray]:
        self.fit(data_list)
        return self.transform(data_list)

    def get_reconstruction_error(self, data_list: List[np.ndarray]) -> List[np.ndarray]:
        """각 시퀀스의 행동별 reconstruction error 반환 (진단용)"""
        if self.model is None:
            raise ValueError("fit()을 먼저 수행하세요.")

        self.model.eval()
        errors = []

        with torch.no_grad():
            for matrix in data_list:
                x = torch.FloatTensor(matrix.astype(np.float32)).unsqueeze(0).to(self.device)
                lengths = [matrix.shape[0]]

                recon, _ = self.model(x, lengths)
                mse_per_row = ((recon.squeeze(0) - x.squeeze(0)) ** 2).mean(dim=1)
                errors.append(mse_per_row.cpu().numpy())

        return errors


# ============================================================
# 5. 학습 곡선 시각화 (개별 문제용)
# ============================================================
def plot_loss_curve_single(problem_num: str,
                           train_losses: List[float],
                           val_losses: List[float],
                           best_epoch: int,
                           D: int,
                           save_path: Optional[str] = None) -> None:
    """단일 문제, 단일 D에 대한 학습 곡선"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train Loss', color='steelblue', linewidth=1.5)
    ax.plot(epochs, val_losses, label='Val Loss', color='coral', linewidth=1.5)
    ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Masked MSE Loss', fontsize=11)
    ax.set_title(f'LSTM AE Loss Curve - {problem_num} (D={D})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 6. 전체 문제 × D 종합 학습 곡선
# ============================================================
def plot_loss_curves_all(problem_reducers: Dict[str, Dict[int, LSTMAEReducer]],
                         save_path: Optional[str] = None) -> None:
    """모든 문제 × D 조합의 학습 곡선을 하나의 figure에 생성"""
    problem_nums = sorted(problem_reducers.keys())
    D_list = sorted(list(problem_reducers[problem_nums[0]].keys()))
    n_problems = len(problem_nums)
    n_D = len(D_list)

    fig, axes = plt.subplots(n_problems, n_D, figsize=(4 * n_D, 3 * n_problems))
    if n_problems == 1:
        axes = axes[np.newaxis, :]
    if n_D == 1:
        axes = axes[:, np.newaxis]

    for i, problem_num in enumerate(problem_nums):
        for j, D in enumerate(D_list):
            ax = axes[i, j]
            reducer = problem_reducers[problem_num][D]

            epochs = range(1, len(reducer.train_losses) + 1)
            ax.plot(epochs, reducer.train_losses, color='steelblue', linewidth=1, label='Train')
            ax.plot(epochs, reducer.val_losses, color='coral', linewidth=1, label='Val')
            ax.axvline(x=reducer.best_epoch, color='red', linestyle='--', alpha=0.4, linewidth=0.8)

            ax.set_title(f'{problem_num} | D={D}', fontsize=9, fontweight='bold')
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2)

            if i == 0 and j == 0:
                ax.legend(fontsize=7)
            if i == n_problems - 1:
                ax.set_xlabel('Epoch', fontsize=8)
            if j == 0:
                ax.set_ylabel('Masked MSE', fontsize=8)

    fig.suptitle('LSTM AE Loss Curves (All Problems × D)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n종합 Loss curve 저장: {save_path}")
    plt.close(fig)


# ============================================================
# 7. 결과 요약 출력
# ============================================================
def print_summary(problem_reducers: Dict[str, Dict[int, LSTMAEReducer]],
                  D_list: List[int]) -> None:
    """문제별, D별 최종 validation loss 요약"""
    problem_nums = sorted(problem_reducers.keys())

    print("\n" + "=" * 70)
    print("LSTM AE Validation Loss (Best Epoch)")
    print("=" * 70)

    header = f"{'Problem':<15}" + "".join([f"{'D=' + str(d):<14}" for d in D_list])
    print(header)
    print("-" * 70)

    for problem_num in problem_nums:
        row = f"{str(problem_num):<15}"
        for d in D_list:
            if d in problem_reducers[problem_num]:
                reducer = problem_reducers[problem_num][d]
                best_val = reducer.val_losses[reducer.best_epoch - 1]
                row += f"{best_val:<14.6f}"
            else:
                row += f"{'N/A':<14}"
        print(row)

    print("=" * 70)


def parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def infer_problems(data: Dict[Tuple, np.ndarray], item_group: Optional[str]) -> List[str]:
    problems = sorted({key[1] for key in data.keys()})
    if item_group:
        prefix = f"{item_group}_"
        problems = [problem for problem in problems if str(problem).startswith(prefix)]
    return problems


def run_lstm_ae(
    data: Dict[Tuple, np.ndarray],
    problems: Sequence[str],
    d_list: Sequence[int],
    output_dir: str,
    hidden_dim: int = 64,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 20,
    seed: int = 42,
    val_ratio: float = 0.1,
    save_models: bool = True,
    save_plots: bool = True,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, "models")
    loss_dir = os.path.join(output_dir, "loss_curves")
    if save_models:
        os.makedirs(model_dir, exist_ok=True)
    if save_plots:
        os.makedirs(loss_dir, exist_ok=True)

    sample_matrix = next(iter(data.values()))
    input_dim = int(sample_matrix.shape[1])

    print("=" * 70)
    print("LSTM AutoEncoder dimension reduction")
    print("=" * 70)
    print(f"  Sequences: {len(data):,}")
    print(f"  Input dim: {input_dim}")
    print(f"  Problems: {', '.join(map(str, problems))}")
    print(f"  Latent dims: {', '.join(map(str, d_list))}")
    print(f"  Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    seq_lengths = [matrix.shape[0] for matrix in data.values()]
    print(f"  Sequence length: min={min(seq_lengths)}, max={max(seq_lengths)}, "
          f"mean={np.mean(seq_lengths):.1f}, median={np.median(seq_lengths):.1f}")

    problem_reducers: Dict[str, Dict[int, LSTMAEReducer]] = {}
    all_results: Dict[int, Dict[Tuple, np.ndarray]] = {D: {} for D in d_list}

    for prob_idx, target_problem in enumerate(problems):
        print(f"\n{'=' * 50}")
        print(f"  [{prob_idx + 1}/{len(problems)}] item '{target_problem}'")
        print(f"{'=' * 50}")

        data_list = []
        keys_list = []
        for key, matrix in data.items():
            if key[1] == target_problem:
                data_list.append(matrix)
                keys_list.append(key)

        if not data_list:
            print(f"    [warning] no sequences for item '{target_problem}'. Skipping.")
            continue

        prob_lengths = [m.shape[0] for m in data_list]
        print(f"    Respondents: {len(data_list):,} | "
              f"length min={min(prob_lengths)}, max={max(prob_lengths)}, "
              f"mean={np.mean(prob_lengths):.1f}")

        problem_reducers[target_problem] = {}

        for D in d_list:
            print(f"\n  --- D = {D} ---")
            reducer = LSTMAEReducer(
                D=D,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                patience=patience,
                seed=seed
            )
            reducer.fit(data_list, val_ratio=val_ratio)

            transformed_list = reducer.transform(data_list)
            for key, transformed_matrix in zip(keys_list, transformed_list):
                all_results[D][key] = transformed_matrix

            problem_reducers[target_problem][D] = reducer

            if save_plots:
                loss_path = os.path.join(loss_dir, f"loss_curve_{target_problem}_D{D}.png")
                plot_loss_curve_single(
                    problem_num=target_problem,
                    train_losses=reducer.train_losses,
                    val_losses=reducer.val_losses,
                    best_epoch=reducer.best_epoch,
                    D=D,
                    save_path=loss_path
                )

            if save_models and reducer.model is not None:
                model_path = os.path.join(model_dir, f"lstm_ae_{target_problem}_D{D}.pt")
                torch.save(reducer.model.state_dict(), model_path)

    print("\n" + "=" * 70)
    print("Saving reduced latent matrices")
    print("=" * 70)
    for D in d_list:
        output_path = os.path.join(output_dir, f"lstm_ae_reduced_D{D}.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(all_results[D], f)
        print(f"  D={D}: {len(all_results[D]):,} sequences -> {output_path}")

    if problem_reducers:
        print_summary(problem_reducers, list(d_list))
        if save_plots:
            plot_loss_curves_all(
                problem_reducers,
                save_path=os.path.join(loss_dir, "loss_curves_all.png")
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train item-wise LSTM autoencoders for action embedding matrices."
    )
    parser.add_argument("--input-pkl", required=True, help="Pickle containing {(seq_id, item): matrix}.")
    parser.add_argument("--output-dir", default="outputs/lstm_ae", help="Directory for reduced outputs.")
    parser.add_argument("--items", default=None, help="Comma-separated item IDs, e.g. ps1_1,ps1_2.")
    parser.add_argument("--item-group", default=None, help="Optional item prefix such as ps1 or ps2.")
    parser.add_argument("--latent-dims", default="1,2,3,4,5", help="Comma-separated latent dimensions.")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--no-save-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_pickle(args.input_pkl)
    if not isinstance(data, dict) or not data:
        raise ValueError("--input-pkl must contain a non-empty dictionary.")

    if args.items:
        problems = [item.strip() for item in args.items.split(",") if item.strip()]
    else:
        problems = infer_problems(data, args.item_group)

    if not problems:
        raise ValueError("No target items found. Check --items or --item-group.")

    run_lstm_ae(
        data=data,
        problems=problems,
        d_list=parse_int_list(args.latent_dims),
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        val_ratio=args.val_ratio,
        save_models=not args.no_save_models,
        save_plots=not args.no_save_plots,
    )


if __name__ == "__main__":
    main()
    print("=" * 70)
    print(f"\n저장된 파일 목록:")
    for root, dirs, files in os.walk(save_dir):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            fsize = os.path.getsize(fpath) / 1024
            rel_path = os.path.relpath(fpath, save_dir)
            print(f"  {rel_path} ({fsize:.1f} KB)")
