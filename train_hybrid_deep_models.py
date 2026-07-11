from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset

from train_deep_sequence_models import (
    BATCH_SIZE,
    LABEL_COL,
    LEARNING_RATE,
    ORDERS_PATH,
    PATIENCE,
    RESULTS_DIR,
    SEED,
    SEQUENCE_FEATURES,
    SEQ_LEN,
    WEIGHT_DECAY,
    Chomp1d,
    compute_metrics,
    find_best_threshold,
    load_inputs,
    print_section,
    set_seed,
    split_indices,
    build_sequence_dataset,
    standardize_sequence,
)


MODEL_DIR = RESULTS_DIR / "hybrid_deep_models"
EPOCHS = 90

TABULAR_FEATURES = [
    "total_orders_before_target",
    "avg_gap_before_target",
    "std_gap_before_target",
    "min_gap_before_target",
    "max_gap_before_target",
    "recent_3_avg_gap",
    "recent_5_avg_gap",
    "last_gap_before_target",
    "gap_trend",
    "active_span_days",
    "order_frequency",
    "weekend_order_ratio",
    "dow_variability",
]


def standardize_tabular(
    df: pd.DataFrame,
    feature_cols: list[str],
    split: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"model_base_hv.csv missing tabular columns: {missing}")

    X = df[feature_cols].astype(float).to_numpy(dtype=np.float32)
    train = X[split["train"]]
    medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)

    X_filled = X.copy()
    nan_rows, nan_cols = np.where(np.isnan(X_filled))
    if len(nan_rows):
        X_filled[nan_rows, nan_cols] = medians[nan_cols]

    train_filled = X_filled[split["train"]]
    mean = train_filled.mean(axis=0)
    std = train_filled.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    X_scaled = (X_filled - mean.reshape(1, -1)) / std.reshape(1, -1)

    scaler = {
        "feature_names": feature_cols,
        "median": medians.astype(float).tolist(),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
    }
    return X_scaled.astype(np.float32), scaler


class HybridDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, X_tab: np.ndarray, y: np.ndarray):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_tab = torch.tensor(X_tab, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_seq[idx], self.X_tab[idx], self.y[idx]


class TabularEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 64, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlattenSequenceEncoder(nn.Module):
    def __init__(self, seq_len: int, input_dim: int, embedding_dim: int = 64, dropout: float = 0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        embedding_dim: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.proj(out[:, -1, :])


class GRUSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        embedding_dim: int = 64,
        dropout: float = 0.25,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        rnn_dim = hidden_dim * (2 if bidirectional else 1)
        self.proj = nn.Sequential(
            nn.Linear(rnn_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.proj(out[:, -1, :])


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class TCNSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: tuple[int, ...] = (64, 96, 96),
        kernel_size: int = 3,
        embedding_dim: int = 64,
        dropout: float = 0.20,
    ):
        super().__init__()
        blocks = []
        in_channels = input_dim
        for level, out_channels in enumerate(channels):
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2**level,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.tcn = nn.Sequential(*blocks)
        self.proj = nn.Sequential(
            nn.Linear(channels[-1], embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.tcn(x.transpose(1, 2))
        return self.proj(y[:, :, -1])


class TransformerSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 2,
        embedding_dim: int = 64,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=192,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.pos_embedding
        encoded = self.encoder(h)
        return self.head(encoded.mean(dim=1))


class TabularOnlyDeepMLP(nn.Module):
    def __init__(self, tabular_dim: int, dropout: float = 0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tabular_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        return self.net(x_tab).squeeze(1)


class HybridClassifier(nn.Module):
    def __init__(
        self,
        sequence_encoder: nn.Module,
        tabular_dim: int,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.sequence_encoder = sequence_encoder
        self.tabular_encoder = TabularEncoder(tabular_dim, embedding_dim=64, dropout=0.20)
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        seq_emb = self.sequence_encoder(x_seq)
        tab_emb = self.tabular_encoder(x_tab)
        return self.head(torch.cat([seq_emb, tab_emb], dim=1)).squeeze(1)


@dataclass
class ModelSpec:
    name: str
    family: str
    model: nn.Module
    hyperparameters: dict[str, Any]
    note: str


def build_model_specs(seq_len: int, seq_input_dim: int, tabular_dim: int) -> list[ModelSpec]:
    return [
        ModelSpec(
            name="TabularDeepMLP",
            family="deep_learning_tabular",
            model=TabularOnlyDeepMLP(tabular_dim=tabular_dim, dropout=0.30),
            hyperparameters={"hidden_layers": [256, 128, 64], "dropout": 0.30},
            note="기존 고객 요약 변수만 사용하는 PyTorch MLP 기준선.",
        ),
        ModelSpec(
            name="HybridMLP_Flatten",
            family="deep_learning_hybrid",
            model=HybridClassifier(
                sequence_encoder=FlattenSequenceEncoder(seq_len=seq_len, input_dim=seq_input_dim, dropout=0.30),
                tabular_dim=tabular_dim,
                dropout=0.30,
            ),
            hyperparameters={"sequence_encoder": "flatten_mlp", "fusion_hidden": [128, 64], "dropout": 0.30},
            note="최근 주문 sequence를 펼친 표현과 고객 요약 변수를 함께 사용하는 MLP.",
        ),
        ModelSpec(
            name="HybridLSTM",
            family="deep_learning_hybrid_sequence",
            model=HybridClassifier(
                sequence_encoder=LSTMSequenceEncoder(seq_input_dim, hidden_dim=128, num_layers=2, dropout=0.25),
                tabular_dim=tabular_dim,
                dropout=0.30,
            ),
            hyperparameters={"sequence_encoder": "lstm", "hidden_dim": 128, "num_layers": 2, "dropout": 0.25},
            note="최근 주문 순서 패턴과 고객 요약 변수를 함께 반영하는 LSTM 하이브리드 모델.",
        ),
        ModelSpec(
            name="HybridGRU",
            family="deep_learning_hybrid_sequence",
            model=HybridClassifier(
                sequence_encoder=GRUSequenceEncoder(seq_input_dim, hidden_dim=128, num_layers=2, dropout=0.25),
                tabular_dim=tabular_dim,
                dropout=0.30,
            ),
            hyperparameters={"sequence_encoder": "gru", "hidden_dim": 128, "num_layers": 2, "dropout": 0.25},
            note="LSTM보다 단순한 gate 구조로 sequence와 요약 변수를 결합하는 GRU 하이브리드 모델.",
        ),
        ModelSpec(
            name="HybridBiGRU",
            family="deep_learning_hybrid_sequence",
            model=HybridClassifier(
                sequence_encoder=GRUSequenceEncoder(
                    seq_input_dim,
                    hidden_dim=96,
                    num_layers=2,
                    dropout=0.25,
                    bidirectional=True,
                ),
                tabular_dim=tabular_dim,
                dropout=0.30,
            ),
            hyperparameters={"sequence_encoder": "bidirectional_gru", "hidden_dim": 96, "num_layers": 2, "dropout": 0.25},
            note="target 이전 sequence 전체를 양방향으로 요약한 뒤 고객 요약 변수와 결합하는 모델.",
        ),
        ModelSpec(
            name="HybridTCN",
            family="deep_learning_hybrid_sequence",
            model=HybridClassifier(
                sequence_encoder=TCNSequenceEncoder(seq_input_dim, channels=(64, 96, 96), kernel_size=3, dropout=0.20),
                tabular_dim=tabular_dim,
                dropout=0.30,
            ),
            hyperparameters={"sequence_encoder": "tcn", "channels": [64, 96, 96], "kernel_size": 3, "dropout": 0.20},
            note="시간축 convolution으로 최근 주문 간격 패턴을 잡고 고객 요약 변수와 결합하는 모델.",
        ),
        ModelSpec(
            name="HybridTransformer",
            family="deep_learning_hybrid_sequence",
            model=HybridClassifier(
                sequence_encoder=TransformerSequenceEncoder(
                    input_dim=seq_input_dim,
                    seq_len=seq_len,
                    d_model=96,
                    nhead=4,
                    num_layers=2,
                    dropout=0.20,
                ),
                tabular_dim=tabular_dim,
                dropout=0.30,
            ),
            hyperparameters={"sequence_encoder": "transformer_encoder", "d_model": 96, "nhead": 4, "num_layers": 2, "dropout": 0.20},
            note="self-attention 기반 sequence 표현과 고객 요약 변수를 결합하는 모델.",
        ),
    ]


def make_loaders(
    X_seq: np.ndarray,
    X_tab: np.ndarray,
    y: np.ndarray,
    split: dict[str, np.ndarray],
) -> dict[str, DataLoader]:
    loaders = {}
    for key, shuffle in [("train", True), ("validation", False), ("test", False)]:
        ds = HybridDataset(X_seq[split[key]], X_tab[split[key]], y[split[key]])
        loaders[key] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)
    return loaders


def collect_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for x_seq, x_tab, _ in loader:
            x_seq = x_seq.to(device)
            x_tab = x_tab.to(device)
            logits = model(x_seq, x_tab)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs)


def train_one_model(
    spec: ModelSpec,
    loaders: dict[str, DataLoader],
    y_split: dict[str, np.ndarray],
    device: torch.device,
    pos_weight: float,
) -> tuple[nn.Module, pd.DataFrame, dict[str, Any], np.ndarray]:
    model = spec.model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_score = -1.0
    best_state = None
    patience_counter = 0
    history_rows = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for x_seq, x_tab, yb in loaders["train"]:
            x_seq = x_seq.to(device)
            x_tab = x_tab.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(x_seq, x_tab)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += float(loss.item()) * len(yb)
            seen += len(yb)

        train_loss /= max(seen, 1)
        val_prob = collect_probs(model, loaders["validation"], device)
        val_auc = roc_auc_score(y_split["validation"], val_prob)
        val_ap = average_precision_score(y_split["validation"], val_prob)
        val_threshold, val_f1 = find_best_threshold(y_split["validation"], val_prob)
        scheduler.step(val_ap)

        history_rows.append(
            {
                "model": spec.name,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_roc_auc": val_auc,
                "val_average_precision": val_ap,
                "val_best_threshold": val_threshold,
                "val_best_f1": val_f1,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"{spec.name} | epoch {epoch:03d} | loss={train_loss:.4f} | "
            f"val_ap={val_ap:.4f} | val_auc={val_auc:.4f} | val_f1={val_f1:.4f}"
        )

        if val_ap > best_score:
            best_score = float(val_ap)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"{spec.name} early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_prob = collect_probs(model, loaders["validation"], device)
    threshold, val_best_f1 = find_best_threshold(y_split["validation"], val_prob)
    test_prob = collect_probs(model, loaders["test"], device)
    metrics = compute_metrics(y_split["test"], test_prob, threshold)
    metrics.update(
        {
            "model": spec.name,
            "family": spec.family,
            "decision_threshold": threshold,
            "val_best_f1": val_best_f1,
            "best_val_average_precision": best_score,
            "epochs_trained": len(history_rows),
            "interpretation_note": spec.note,
        }
    )
    return model, pd.DataFrame(history_rows), metrics, test_prob


def save_curves(results_df: pd.DataFrame, curve_records: dict[str, Any], positive_ratio: float) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Random")
    for name, data in curve_records.items():
        plt.plot(data["fpr"], data["tpr"], linewidth=2, label=f"{name} (AUC={data['roc_auc']:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Hybrid Deep Models - ROC Curve")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "hybrid_deep_roc_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.axhline(positive_ratio, linestyle="--", color="#999999", label=f"Positive ratio ({positive_ratio:.3f})")
    for name, data in curve_records.items():
        plt.plot(data["recall"], data["precision"], linewidth=2, label=f"{name} (AP={data['average_precision']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Hybrid Deep Models - Precision-Recall Curve")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "hybrid_deep_pr_curve.png", dpi=150)
    plt.close()

    plot_df = results_df.sort_values("f1_score", ascending=True)
    metrics = ["recall", "f1_score", "roc_auc", "average_precision"]
    ax = plot_df.set_index("model")[metrics].plot(kind="barh", figsize=(10, 6), width=0.82)
    ax.set_title("Hybrid Deep Model Comparison")
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(["Recall", "F1-score", "ROC-AUC", "Average Precision"], loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "hybrid_deep_model_comparison.png", dpi=150)
    plt.close()


def save_architecture_diagram() -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis("off")

    def box(text: str, x: float, y: float, w: float = 0.16, h: float = 0.20, color: str = "#EAF3F0") -> None:
        rect = plt.Rectangle((x, y), w, h, fill=True, color=color, ec="#0E5A53", lw=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, color="#1E2E2B")

    box("Recent N orders\nsequence", 0.05, 0.62)
    box("Sequence encoder\nLSTM / GRU / TCN\nTransformer", 0.28, 0.62, w=0.18)
    box("Sequence\nembedding", 0.55, 0.62)
    box("Customer summary\nfeatures", 0.05, 0.24)
    box("Tabular MLP\nencoder", 0.28, 0.24, w=0.18)
    box("Tabular\nembedding", 0.55, 0.24)
    box("Concatenate\nembeddings", 0.73, 0.43)
    box("Dense head\n+ sigmoid", 0.88, 0.43, w=0.11)

    arrows = [
        ((0.21, 0.72), (0.28, 0.72)),
        ((0.46, 0.72), (0.55, 0.72)),
        ((0.21, 0.34), (0.28, 0.34)),
        ((0.46, 0.34), (0.55, 0.34)),
        ((0.71, 0.72), (0.73, 0.53)),
        ((0.71, 0.34), (0.73, 0.48)),
        ((0.89, 0.53), (0.88, 0.53)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.8, color="#66736D"))

    ax.text(0.5, 0.93, "Hybrid Deep Learning Pipeline", ha="center", fontsize=15, weight="bold", color="#0E5A53")
    ax.text(
        0.5,
        0.08,
        "The model combines recent order sequence patterns with aggregate customer history features. Inputs use only records before the target order.",
        ha="center",
        fontsize=10,
        color="#66736D",
    )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "hybrid_deep_model_architecture.png", dpi=150)
    plt.close()


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None) -> list[str]:
    view = df.copy()
    if columns is not None:
        view = view[columns]
    view = view.fillna("")
    header = "| " + " | ".join(view.columns.astype(str)) + " |"
    separator = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = [header, separator]
    for _, row in view.iterrows():
        rows.append("| " + " | ".join(str(value) for value in row.tolist()) + " |")
    return rows


def save_summary(results_df: pd.DataFrame, metadata: dict[str, Any], hyperparams_df: pd.DataFrame) -> None:
    best_f1 = results_df.iloc[0]
    best_recall = results_df.sort_values("recall", ascending=False).iloc[0]
    best_auc = results_df.sort_values("roc_auc", ascending=False).iloc[0]
    best_ap = results_df.sort_values("average_precision", ascending=False).iloc[0]

    lines = [
        "# Hybrid Deep Model Summary",
        "",
        "이 파일은 최근 주문 sequence와 기존 고객 요약 feature를 함께 사용하는 하이브리드 딥러닝 실험 요약이다.",
        "모든 입력은 target 주문 이전 이력만 사용하며, label은 target 주문의 `target_gap > 15`로 생성한다.",
        "",
        "## Dataset",
        "",
        f"- Samples: {metadata['samples']:,}",
        f"- Positive ratio: {metadata['positive_ratio']:.4f}",
        f"- Sequence length: {metadata['sequence_length']}",
        f"- Sequence features: {', '.join(metadata['sequence_features'])}",
        f"- Tabular features: {', '.join(TABULAR_FEATURES)}",
        f"- Split method: {metadata['split_method']}",
        "",
        "## Best Models By Metric",
        "",
        f"- F1-score: {best_f1['model']} ({best_f1['f1_score']:.4f})",
        f"- Recall: {best_recall['model']} ({best_recall['recall']:.4f})",
        f"- ROC-AUC: {best_auc['model']} ({best_auc['roc_auc']:.4f})",
        f"- Average Precision: {best_ap['model']} ({best_ap['average_precision']:.4f})",
        "",
        "## Model Comparison",
        "",
    ]
    table_cols = ["rank_by_f1", "model", "precision", "recall", "f1_score", "roc_auc", "average_precision", "decision_threshold", "epochs_trained"]
    lines.extend(markdown_table(results_df[table_cols]))
    lines.extend(
        [
            "",
            "## Hyperparameters",
            "",
        ]
    )
    hyper_cols = ["model", "family", "batch_size", "max_epochs", "patience", "learning_rate", "weight_decay", "pos_weight", "note"]
    lines.extend(markdown_table(hyperparams_df[hyper_cols]))
    lines.extend(
        [
            "",
            "## Presentation Note",
            "",
            "정형 요약 변수만 사용하는 모델과 비교했을 때, 하이브리드 모델은 고객의 장기 요약 정보와 최근 주문 순서 패턴을 함께 반영한다.",
            "따라서 최종 발표에서는 단순히 성능표를 나열하기보다, 입력 표현을 강화했을 때 딥러닝 모델이 어떤 지표에서 개선되는지를 중심으로 해석하는 것이 좋다.",
        ]
    )
    (RESULTS_DIR / "hybrid_deep_model_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print_section("Load data")
    model_base, orders = load_inputs()
    if not ORDERS_PATH.exists():
        raise FileNotFoundError(f"orders file not found: {ORDERS_PATH}")

    print_section("Build sequence and tabular inputs")
    X_seq, y, kept, metadata = build_sequence_dataset(model_base, orders, seq_len=SEQ_LEN)
    split = split_indices(y)
    X_seq_scaled, seq_scaler = standardize_sequence(X_seq, split)
    X_tab_scaled, tab_scaler = standardize_tabular(kept, TABULAR_FEATURES, split)

    y_split = {key: y[idx] for key, idx in split.items()}
    loaders = make_loaders(X_seq_scaled, X_tab_scaled, y, split)
    metadata.update(
        {
            "split_method": "stratified random split 70/15/15 using the same target-anchor samples as tabular baselines",
            "split": {
                key: {
                    "samples": int(len(idx)),
                    "positive_count": int(y[idx].sum()),
                    "positive_ratio": float(y[idx].mean()),
                }
                for key, idx in split.items()
            },
            "sequence_scaler": seq_scaler,
            "tabular_scaler": tab_scaler,
            "tabular_features": TABULAR_FEATURES,
            "hybrid_input_definition": "Each model receives a recent order sequence plus aggregate customer-history features computed before the target order.",
        }
    )
    (RESULTS_DIR / "hybrid_deep_dataset_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in metadata.items() if not k.endswith("scaler")}, ensure_ascii=False, indent=2))

    pos = float(y_split["train"].sum())
    neg = float(len(y_split["train"]) - y_split["train"].sum())
    pos_weight = neg / max(pos, 1.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"pos_weight: {pos_weight:.4f}")

    specs = build_model_specs(seq_len=X_seq_scaled.shape[1], seq_input_dim=X_seq_scaled.shape[2], tabular_dim=X_tab_scaled.shape[1])
    hyperparams_rows = []
    for spec in specs:
        row = {
            "model": spec.name,
            "family": spec.family,
            "batch_size": BATCH_SIZE,
            "max_epochs": EPOCHS,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "pos_weight": pos_weight,
            "note": spec.note,
        }
        row.update({f"hp_{k}": json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in spec.hyperparameters.items()})
        hyperparams_rows.append(row)
    hyperparams_df = pd.DataFrame(hyperparams_rows)
    hyperparams_df.to_csv(RESULTS_DIR / "hybrid_deep_hyperparameters.csv", index=False, encoding="utf-8-sig")

    all_results = []
    all_history = []
    prediction_frames = []
    curve_records: dict[str, Any] = {}
    threshold_records: dict[str, Any] = {
        "_note": "All hybrid deep models use validation F1 maximization for decision threshold selection.",
        "_split_method": metadata["split_method"],
        "_sequence_length": SEQ_LEN,
    }

    for spec in specs:
        print_section(f"Train {spec.name}")
        model, history_df, metrics, test_prob = train_one_model(
            spec=spec,
            loaders=loaders,
            y_split=y_split,
            device=device,
            pos_weight=pos_weight,
        )
        torch.save(model.state_dict(), MODEL_DIR / f"{spec.name}.pt")
        history_df.to_csv(MODEL_DIR / f"{spec.name}_history.csv", index=False, encoding="utf-8-sig")
        all_history.append(history_df)
        all_results.append(metrics)

        pred_df = kept.loc[split["test"], ["user_id", "target_order_id", "target_order_number", "target_gap", LABEL_COL]].copy()
        pred_df.insert(0, "model", spec.name)
        pred_df["y_prob"] = test_prob
        pred_df["y_pred"] = (test_prob >= metrics["decision_threshold"]).astype(int)
        prediction_frames.append(pred_df)

        fpr, tpr, _ = roc_curve(y_split["test"], test_prob)
        pr_precision, pr_recall, _ = precision_recall_curve(y_split["test"], test_prob)
        curve_records[spec.name] = {
            "fpr": fpr,
            "tpr": tpr,
            "precision": pr_precision,
            "recall": pr_recall,
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["average_precision"],
        }
        threshold_records[spec.name] = {
            "decision_threshold": metrics["decision_threshold"],
            "val_best_f1": metrics["val_best_f1"],
            "best_val_average_precision": metrics["best_val_average_precision"],
        }

        print(
            f"{spec.name} test | threshold={metrics['decision_threshold']:.2f} | "
            f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1_score']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}, AP={metrics['average_precision']:.4f}"
        )

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(["f1_score", "recall", "average_precision"], ascending=False).reset_index(drop=True)
    results_df.insert(0, "rank_by_f1", np.arange(1, len(results_df) + 1))
    results_df.to_csv(RESULTS_DIR / "hybrid_deep_model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_history, ignore_index=True).to_csv(RESULTS_DIR / "hybrid_deep_training_history.csv", index=False, encoding="utf-8-sig")
    pd.concat(prediction_frames, ignore_index=True).to_csv(RESULTS_DIR / "hybrid_deep_test_predictions.csv", index=False, encoding="utf-8-sig")
    (RESULTS_DIR / "hybrid_deep_thresholds.json").write_text(json.dumps(threshold_records, ensure_ascii=False, indent=2), encoding="utf-8")

    save_curves(results_df, curve_records, positive_ratio=float(y_split["test"].mean()))
    save_architecture_diagram()
    save_summary(results_df, metadata, hyperparams_df)

    print_section("Saved")
    for path in [
        RESULTS_DIR / "hybrid_deep_model_comparison.csv",
        RESULTS_DIR / "hybrid_deep_hyperparameters.csv",
        RESULTS_DIR / "hybrid_deep_dataset_metadata.json",
        RESULTS_DIR / "hybrid_deep_training_history.csv",
        RESULTS_DIR / "hybrid_deep_test_predictions.csv",
        RESULTS_DIR / "hybrid_deep_thresholds.json",
        RESULTS_DIR / "hybrid_deep_model_comparison.png",
        RESULTS_DIR / "hybrid_deep_roc_curve.png",
        RESULTS_DIR / "hybrid_deep_pr_curve.png",
        RESULTS_DIR / "hybrid_deep_model_architecture.png",
        RESULTS_DIR / "hybrid_deep_model_summary.md",
    ]:
        print(path)


if __name__ == "__main__":
    main()
