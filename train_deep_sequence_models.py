from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_BASE_PATH = PROJECT_ROOT / "model_base_hv.csv"
ORDERS_PATH = PROJECT_ROOT / "incoming_inspect" / "raw_archive" / "archive" / "orders.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = RESULTS_DIR / "deep_sequence_models"

SEED = 42
SEQ_LEN = 20
BATCH_SIZE = 512
EPOCHS = 80
PATIENCE = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_COL = "delay_risk"


SEQUENCE_FEATURES = [
    "gap",
    "gap_delta",
    "rolling_3_gap",
    "order_dow_sin",
    "order_dow_cos",
    "order_hour_sin",
    "order_hour_cos",
    "relative_order_position",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not MODEL_BASE_PATH.exists():
        raise FileNotFoundError(f"model base file not found: {MODEL_BASE_PATH}")
    if not ORDERS_PATH.exists():
        raise FileNotFoundError(f"orders file not found: {ORDERS_PATH}")

    model_base = pd.read_csv(MODEL_BASE_PATH)
    required_base_cols = {"user_id", "target_order_number", "target_gap", LABEL_COL}
    missing_base = required_base_cols - set(model_base.columns)
    if missing_base:
        raise ValueError(f"model_base_hv.csv missing columns: {sorted(missing_base)}")

    orders = pd.read_csv(
        ORDERS_PATH,
        usecols=[
            "order_id",
            "user_id",
            "order_number",
            "order_dow",
            "order_hour_of_day",
            "days_since_prior_order",
        ],
    )
    orders = orders.sort_values(["user_id", "order_number"]).reset_index(drop=True)
    return model_base, orders


def make_user_order_lookup(orders: pd.DataFrame) -> dict[int, pd.DataFrame]:
    lookup: dict[int, pd.DataFrame] = {}
    for user_id, group in orders.groupby("user_id", sort=False):
        lookup[int(user_id)] = group.sort_values("order_number").reset_index(drop=True)
    return lookup


def build_one_sequence(user_orders: pd.DataFrame, target_order_number: int, seq_len: int) -> np.ndarray:
    history = user_orders[user_orders["order_number"] < target_order_number].copy()
    if history.empty:
        raise ValueError("target order has no history")

    history = history.tail(seq_len).copy()
    gaps = history["days_since_prior_order"].fillna(0).astype(float).to_numpy()
    dows = history["order_dow"].astype(float).to_numpy()
    hours = history["order_hour_of_day"].astype(float).to_numpy()
    order_numbers = history["order_number"].astype(float).to_numpy()

    gap_delta = np.zeros_like(gaps)
    if len(gaps) > 1:
        gap_delta[1:] = np.diff(gaps)

    rolling_3 = pd.Series(gaps).rolling(window=3, min_periods=1).mean().to_numpy()
    dow_angle = 2 * math.pi * dows / 7.0
    hour_angle = 2 * math.pi * hours / 24.0
    relative_position = order_numbers / max(float(target_order_number - 1), 1.0)

    seq = np.column_stack(
        [
            gaps,
            gap_delta,
            rolling_3,
            np.sin(dow_angle),
            np.cos(dow_angle),
            np.sin(hour_angle),
            np.cos(hour_angle),
            relative_position,
        ]
    ).astype(np.float32)

    if len(seq) < seq_len:
        pad = np.zeros((seq_len - len(seq), seq.shape[1]), dtype=np.float32)
        seq = np.vstack([pad, seq])

    return seq


def build_sequence_dataset(
    model_base: pd.DataFrame,
    orders: pd.DataFrame,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    lookup = make_user_order_lookup(orders)
    sequences: list[np.ndarray] = []
    keep_rows: list[int] = []
    missing_users = 0
    short_histories = 0

    for idx, row in model_base.iterrows():
        user_id = int(row["user_id"])
        target_order_number = int(row["target_order_number"])
        user_orders = lookup.get(user_id)
        if user_orders is None:
            missing_users += 1
            continue
        history_len = int((user_orders["order_number"] < target_order_number).sum())
        if history_len == 0:
            short_histories += 1
            continue
        sequences.append(build_one_sequence(user_orders, target_order_number, seq_len))
        keep_rows.append(idx)

    if not sequences:
        raise ValueError("no sequence samples were created")

    X = np.stack(sequences).astype(np.float32)
    kept = model_base.loc[keep_rows].reset_index(drop=True).copy()
    y = kept[LABEL_COL].astype(int).to_numpy()

    metadata = {
        "sequence_source": "raw orders.csv matched to model_base_hv target anchors",
        "sequence_length": seq_len,
        "sequence_features": SEQUENCE_FEATURES,
        "samples": int(len(kept)),
        "positive_count": int(y.sum()),
        "positive_ratio": float(y.mean()),
        "missing_users_skipped": int(missing_users),
        "short_histories_skipped": int(short_histories),
        "feature_label_separation": "Each sequence uses orders before the target order only. The label is computed from target_gap on the target order.",
    }
    return X, y, kept, metadata


def split_indices(y: np.ndarray) -> dict[str, np.ndarray]:
    indices = np.arange(len(y))
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices,
        y,
        test_size=0.30,
        stratify=y,
        random_state=SEED,
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=SEED,
    )
    return {"train": train_idx, "validation": val_idx, "test": test_idx}


def standardize_sequence(
    X: np.ndarray,
    split: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, list[float]]]:
    train_flat = X[split["train"]].reshape(-1, X.shape[-1])
    mean = train_flat.mean(axis=0)
    std = train_flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    X_scaled = (X - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    scaler = {
        "feature_names": SEQUENCE_FEATURES,
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
    }
    return X_scaled.astype(np.float32), scaler


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class MLPFlattenClassifier(nn.Module):
    def __init__(self, seq_len: int, input_dim: int, dropout: float = 0.30):
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
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(1)


class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.25,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        rnn_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(rnn_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(1)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


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


class TCNClassifier(nn.Module):
    def __init__(self, input_dim: int, channels: tuple[int, ...] = (64, 64, 64), kernel_size: int = 3, dropout: float = 0.20):
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
        self.head = nn.Sequential(
            nn.Linear(channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1d expects batch, channels, seq.
        y = self.tcn(x.transpose(1, 2))
        return self.head(y[:, :, -1]).squeeze(1)


class TransformerSequenceClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.pos_embedding
        encoded = self.encoder(h)
        pooled = encoded.mean(dim=1)
        return self.head(pooled).squeeze(1)


@dataclass
class ModelSpec:
    name: str
    family: str
    model: nn.Module
    hyperparameters: dict[str, Any]
    note: str


def build_model_specs(seq_len: int, input_dim: int) -> list[ModelSpec]:
    return [
        ModelSpec(
            name="DeepMLP_SequenceFlatten",
            family="deep_learning",
            model=MLPFlattenClassifier(seq_len=seq_len, input_dim=input_dim, dropout=0.30),
            hyperparameters={"hidden_layers": [256, 128, 64], "dropout": 0.30},
            note="Sequence를 펼쳐서 쓰는 강한 MLP 기준선.",
        ),
        ModelSpec(
            name="LSTM",
            family="deep_learning_sequence",
            model=LSTMClassifier(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0.25),
            hyperparameters={"hidden_dim": 128, "num_layers": 2, "dropout": 0.25},
            note="최근 주문 sequence를 순서대로 읽는 recurrent model.",
        ),
        ModelSpec(
            name="GRU",
            family="deep_learning_sequence",
            model=GRUClassifier(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0.25),
            hyperparameters={"hidden_dim": 128, "num_layers": 2, "dropout": 0.25, "bidirectional": False},
            note="LSTM보다 단순한 gate 구조의 recurrent model.",
        ),
        ModelSpec(
            name="BiGRU",
            family="deep_learning_sequence",
            model=GRUClassifier(input_dim=input_dim, hidden_dim=96, num_layers=2, dropout=0.25, bidirectional=True),
            hyperparameters={"hidden_dim": 96, "num_layers": 2, "dropout": 0.25, "bidirectional": True},
            note="이미 알고 있는 target 이전 sequence 전체를 양방향으로 요약하는 GRU.",
        ),
        ModelSpec(
            name="TCN",
            family="deep_learning_sequence",
            model=TCNClassifier(input_dim=input_dim, channels=(64, 64, 64), kernel_size=3, dropout=0.20),
            hyperparameters={"channels": [64, 64, 64], "kernel_size": 3, "dropout": 0.20},
            note="시간축 1D convolution으로 주문 간격 패턴을 잡는 sequence model.",
        ),
        ModelSpec(
            name="TransformerEncoder",
            family="deep_learning_sequence",
            model=TransformerSequenceClassifier(input_dim=input_dim, seq_len=seq_len, d_model=64, nhead=4, num_layers=2, dropout=0.20),
            hyperparameters={"d_model": 64, "nhead": 4, "num_layers": 2, "dropout": 0.20},
            note="Self-attention으로 최근 주문 sequence 내 위치 간 관계를 요약하는 model.",
        ),
    ]


def make_loaders(X: np.ndarray, y: np.ndarray, split: dict[str, np.ndarray]) -> dict[str, DataLoader]:
    loaders = {}
    for key, shuffle in [("train", True), ("validation", False), ("test", False)]:
        ds = SequenceDataset(X[split[key]], y[split[key]])
        loaders[key] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)
    return loaders


def collect_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs)


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.05, 0.96, 0.01):
        pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(score)
    return best_threshold, best_f1


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def train_one_model(
    spec: ModelSpec,
    loaders: dict[str, DataLoader],
    y_split: dict[str, np.ndarray],
    device: torch.device,
    pos_weight: float,
) -> tuple[nn.Module, pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray]:
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
        for xb, yb in loaders["train"]:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += float(loss.item()) * len(xb)
            seen += len(xb)

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

    return model, pd.DataFrame(history_rows), metrics, val_prob, test_prob


def save_curves(results_df: pd.DataFrame, curve_records: dict[str, Any], positive_ratio: float) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Random")
    for name, data in curve_records.items():
        plt.plot(data["fpr"], data["tpr"], linewidth=2, label=f"{name} (AUC={data['roc_auc']:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Deep Sequence Models - ROC Curve")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "deep_sequence_roc_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.axhline(positive_ratio, linestyle="--", color="#999999", label=f"Positive ratio ({positive_ratio:.3f})")
    for name, data in curve_records.items():
        plt.plot(data["recall"], data["precision"], linewidth=2, label=f"{name} (AP={data['average_precision']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Deep Sequence Models - Precision-Recall Curve")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "deep_sequence_pr_curve.png", dpi=150)
    plt.close()

    plot_df = results_df.sort_values("f1_score", ascending=True)
    metrics = ["recall", "f1_score", "roc_auc", "average_precision"]
    ax = plot_df.set_index("model")[metrics].plot(kind="barh", figsize=(10, 6), width=0.82)
    ax.set_title("Deep Sequence Model Comparison")
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(["Recall", "F1-score", "ROC-AUC", "Average Precision"], loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "deep_sequence_model_comparison.png", dpi=150)
    plt.close()


def save_architecture_diagram() -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    boxes = [
        ("Recent N orders\n(sequence)", 0.07),
        ("Per-order inputs\n gap, dow, hour,\ntrend signals", 0.25),
        ("Sequence model\nLSTM / GRU / TCN\nTransformer", 0.47),
        ("Dense head\n+ sigmoid", 0.68),
        ("delay_risk\nprobability", 0.86),
    ]
    for text, x in boxes:
        rect = plt.Rectangle((x, 0.35), 0.14, 0.30, fill=True, color="#EAF3F0", ec="#0E5A53", lw=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.07, 0.50, text, ha="center", va="center", fontsize=10, color="#1E2E2B")
    for i in range(len(boxes) - 1):
        x0 = boxes[i][1] + 0.14
        x1 = boxes[i + 1][1]
        ax.annotate("", xy=(x1, 0.50), xytext=(x0, 0.50), arrowprops=dict(arrowstyle="->", lw=1.8, color="#66736D"))
    ax.text(0.5, 0.82, "Deep Sequence Modeling Pipeline", ha="center", fontsize=15, weight="bold", color="#0E5A53")
    ax.text(0.5, 0.18, "Feature-label separation: only orders before the target order are used as model input.", ha="center", fontsize=10, color="#66736D")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "deep_sequence_model_architecture.png", dpi=150)
    plt.close()


def save_summary(
    results_df: pd.DataFrame,
    metadata: dict[str, Any],
    split: dict[str, np.ndarray],
    y: np.ndarray,
    hyperparams_df: pd.DataFrame,
) -> None:
    def markdown_table(df: pd.DataFrame, columns: list[str] | None = None) -> list[str]:
        view = df.copy()
        if columns is not None:
            view = view[columns]
        view = view.fillna("")
        header = "| " + " | ".join(view.columns.astype(str)) + " |"
        separator = "| " + " | ".join(["---"] * len(view.columns)) + " |"
        rows = []
        for _, row in view.iterrows():
            values = []
            for value in row:
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            rows.append("| " + " | ".join(values) + " |")
        return [header, separator, *rows]

    best_f1 = results_df.sort_values(["f1_score", "recall"], ascending=False).iloc[0]
    best_recall = results_df.sort_values(["recall", "f1_score"], ascending=False).iloc[0]
    best_ap = results_df.sort_values(["average_precision", "f1_score"], ascending=False).iloc[0]

    split_rows = []
    for key in ["train", "validation", "test"]:
        idx = split[key]
        split_rows.append(
            {
                "split": key,
                "samples": int(len(idx)),
                "positives": int(y[idx].sum()),
                "positive_ratio": float(y[idx].mean()),
            }
        )

    lines = [
        "# Deep Sequence Model Summary",
        "",
        "## 실험 목적",
        "",
        "기존 집계 feature 실험을 보완하기 위해 target 주문 이전 최근 주문 이력을 실제 sequence 입력으로 구성하고, LSTM/GRU/BiGRU/TCN/Transformer 계열 딥러닝 모델을 비교한다.",
        "",
        "## 데이터 구성",
        "",
        f"- 기준 데이터: `model_base_hv.csv`의 target anchor와 label",
        f"- 원본 주문 이력: `incoming_inspect/raw_archive/archive/orders.csv`",
        f"- sequence length: `{metadata['sequence_length']}`",
        f"- sequence features: `{', '.join(metadata['sequence_features'])}`",
        f"- samples: `{metadata['samples']:,}`",
        f"- positive ratio: `{metadata['positive_ratio']:.4f}`",
        "- feature-label separation: target 이전 주문만 입력으로 사용하고, label은 target 주문의 target_gap으로 생성",
        "",
        "## split",
        "",
        "| split | samples | positives | positive_ratio |",
        "| --- | ---: | ---: | ---: |",
        *[
            f"| {row['split']} | {row['samples']:,} | {row['positives']:,} | {row['positive_ratio']:.4f} |"
            for row in split_rows
        ],
        "",
        "## 지표별 대표 모델",
        "",
        f"- F1-score 기준: `{best_f1['model']}` (F1={best_f1['f1_score']:.4f}, Recall={best_f1['recall']:.4f}, AP={best_f1['average_precision']:.4f})",
        f"- Recall 기준: `{best_recall['model']}` (Recall={best_recall['recall']:.4f}, F1={best_recall['f1_score']:.4f})",
        f"- Average Precision 기준: `{best_ap['model']}` (AP={best_ap['average_precision']:.4f}, ROC-AUC={best_ap['roc_auc']:.4f})",
        "",
        "## 발표용 해석 방향",
        "",
        "- 정형 요약 feature만 사용한 실험은 강한 baseline으로 유지한다.",
        "- 최종 딥러닝 실험은 최근 주문 sequence를 직접 입력으로 넣어 주문 간격 변화와 순차 패턴을 학습한다는 점에 의미가 있다.",
        "- 최종 모델은 단순 accuracy가 아니라 Recall, F1-score, Average Precision을 함께 고려해 선정한다.",
        "- 성능 차이가 작다면, 순차 정보를 직접 반영할 수 있는 구조적 타당성과 지표 균형을 함께 근거로 제시한다.",
        "",
        "## 하이퍼파라미터 요약",
        "",
        *markdown_table(
            hyperparams_df,
            [
                "model",
                "batch_size",
                "max_epochs",
                "patience",
                "learning_rate",
                "weight_decay",
                "pos_weight",
                "note",
            ],
        ),
        "",
        "## 상세 성능",
        "",
        *markdown_table(
            results_df,
            [
                "rank_by_f1",
                "model",
                "decision_threshold",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "average_precision",
                "tn",
                "fp",
                "fn",
                "tp",
            ],
        ),
        "",
    ]
    (RESULTS_DIR / "deep_sequence_model_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_section("Load and build sequence dataset")
    model_base, orders = load_inputs()
    X, y, kept, metadata = build_sequence_dataset(model_base, orders, SEQ_LEN)
    split = split_indices(y)
    X, scaler = standardize_sequence(X, split)
    loaders = make_loaders(X, y, split)
    y_split = {key: y[idx] for key, idx in split.items()}

    metadata.update(
        {
            "device": str(device),
            "random_state": SEED,
            "split_method": "stratified random split on model_base_hv sequence samples",
            "split": {
                key: {
                    "samples": int(len(idx)),
                    "positive_count": int(y[idx].sum()),
                    "positive_ratio": float(y[idx].mean()),
                }
                for key, idx in split.items()
            },
            "scaler": scaler,
        }
    )
    (RESULTS_DIR / "deep_sequence_dataset_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in metadata.items() if k != "scaler"}, ensure_ascii=False, indent=2))

    pos = float(y_split["train"].sum())
    neg = float(len(y_split["train"]) - y_split["train"].sum())
    pos_weight = neg / max(pos, 1.0)
    specs = build_model_specs(seq_len=X.shape[1], input_dim=X.shape[2])

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
    hyperparams_df.to_csv(RESULTS_DIR / "deep_sequence_hyperparameters.csv", index=False, encoding="utf-8-sig")

    all_results = []
    all_history = []
    prediction_frames = []
    curve_records: dict[str, Any] = {}
    threshold_records: dict[str, Any] = {
        "_note": "All deep sequence models use validation F1 maximization for decision threshold selection.",
        "_split_method": metadata["split_method"],
        "_sequence_length": SEQ_LEN,
    }

    for spec in specs:
        print_section(f"Train {spec.name}")
        model, history_df, metrics, _, test_prob = train_one_model(
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

        test_idx = split["test"]
        pred_df = kept.loc[test_idx, ["user_id", "target_order_id", "target_order_number", "target_gap", LABEL_COL]].copy()
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
    results_df.to_csv(RESULTS_DIR / "deep_sequence_model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_history, ignore_index=True).to_csv(RESULTS_DIR / "deep_sequence_training_history.csv", index=False, encoding="utf-8-sig")
    pd.concat(prediction_frames, ignore_index=True).to_csv(RESULTS_DIR / "deep_sequence_test_predictions.csv", index=False, encoding="utf-8-sig")
    (RESULTS_DIR / "deep_sequence_thresholds.json").write_text(json.dumps(threshold_records, ensure_ascii=False, indent=2), encoding="utf-8")

    save_curves(results_df, curve_records, positive_ratio=float(y_split["test"].mean()))
    save_architecture_diagram()
    save_summary(results_df, metadata, split, y, hyperparams_df)

    print_section("Saved")
    for path in [
        RESULTS_DIR / "deep_sequence_model_comparison.csv",
        RESULTS_DIR / "deep_sequence_hyperparameters.csv",
        RESULTS_DIR / "deep_sequence_dataset_metadata.json",
        RESULTS_DIR / "deep_sequence_training_history.csv",
        RESULTS_DIR / "deep_sequence_test_predictions.csv",
        RESULTS_DIR / "deep_sequence_thresholds.json",
        RESULTS_DIR / "deep_sequence_model_comparison.png",
        RESULTS_DIR / "deep_sequence_roc_curve.png",
        RESULTS_DIR / "deep_sequence_pr_curve.png",
        RESULTS_DIR / "deep_sequence_model_architecture.png",
        RESULTS_DIR / "deep_sequence_model_summary.md",
    ]:
        print(path)


if __name__ == "__main__":
    main()
