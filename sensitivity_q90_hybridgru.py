from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from train_deep_sequence_models import (
    BATCH_SIZE,
    LABEL_COL,
    LEARNING_RATE,
    ORDERS_PATH,
    PATIENCE,
    RESULTS_DIR,
    SEED,
    SEQ_LEN,
    WEIGHT_DECAY,
    build_sequence_dataset,
    compute_metrics,
    find_best_threshold,
    load_inputs,
    print_section,
    set_seed,
    split_indices,
    standardize_sequence,
)
from train_hybrid_deep_models import (
    TABULAR_FEATURES,
    GRUSequenceEncoder,
    HybridClassifier,
    ModelSpec,
    make_loaders,
    standardize_tabular,
    train_one_model,
)


DELAY_DAYS = 24
LABEL_NAME = "delay_gt_24_q90"
OUT_DIR = RESULTS_DIR / "q90_hybridgru_sensitivity"


def _round_metrics(row: dict[str, Any]) -> dict[str, Any]:
    rounded = {}
    for key, value in row.items():
        if isinstance(value, float):
            rounded[key] = round(value, 6)
        else:
            rounded[key] = value
    return rounded


def main() -> None:
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print_section("Load data")
    model_base, orders = load_inputs()
    if not ORDERS_PATH.exists():
        raise FileNotFoundError(f"orders file not found: {ORDERS_PATH}")

    print_section("Build q90 HybridGRU inputs")
    X_seq, _old_y, kept, metadata = build_sequence_dataset(model_base, orders, seq_len=SEQ_LEN)
    y = (kept["target_gap"] > DELAY_DAYS).astype(int).to_numpy()
    kept = kept.copy()
    kept[LABEL_NAME] = y

    split = split_indices(y)
    X_seq_scaled, seq_scaler = standardize_sequence(X_seq, split)
    X_tab_scaled, tab_scaler = standardize_tabular(kept, TABULAR_FEATURES, split)
    y_split = {key: y[idx] for key, idx in split.items()}
    loaders = make_loaders(X_seq_scaled, X_tab_scaled, y, split)

    positive_count = int(y.sum())
    positive_ratio = float(y.mean())
    pos = float(y_split["train"].sum())
    neg = float(len(y_split["train"]) - y_split["train"].sum())
    pos_weight = neg / max(pos, 1.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata.update(
        {
            "label_name": LABEL_NAME,
            "delay_days": DELAY_DAYS,
            "label_definition": f"target_gap > {DELAY_DAYS}",
            "positive_count": positive_count,
            "positive_ratio": positive_ratio,
            "split_method": "stratified random split 70/15/15 using q90 label",
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
            "pos_weight": pos_weight,
        }
    )

    print(json.dumps({k: v for k, v in metadata.items() if not k.endswith("scaler")}, ensure_ascii=False, indent=2))
    print(f"device: {device}")
    print(f"pos_weight: {pos_weight:.4f}")

    spec = ModelSpec(
        name="HybridGRU_q90",
        family="deep_learning_hybrid_sequence",
        model=HybridClassifier(
            sequence_encoder=GRUSequenceEncoder(
                input_dim=X_seq_scaled.shape[2],
                hidden_dim=128,
                num_layers=2,
                dropout=0.25,
            ),
            tabular_dim=X_tab_scaled.shape[1],
            dropout=0.30,
        ),
        hyperparameters={"sequence_encoder": "gru", "hidden_dim": 128, "num_layers": 2, "dropout": 0.25},
        note="q90=24일 고위험 기준에서 재학습한 HybridGRU 민감도 분석.",
    )

    print_section("Train HybridGRU q90")
    model, history_df, metrics, test_prob = train_one_model(
        spec=spec,
        loaders=loaders,
        y_split=y_split,
        device=device,
        pos_weight=pos_weight,
    )

    torch.save(model.state_dict(), OUT_DIR / "HybridGRU_q90.pt")
    history_df.to_csv(OUT_DIR / "hybridgru_q90_training_history.csv", index=False, encoding="utf-8-sig")

    pred_df = kept.loc[split["test"], ["user_id", "target_order_id", "target_order_number", "target_gap", LABEL_COL, LABEL_NAME]].copy()
    pred_df["model"] = spec.name
    pred_df["y_prob"] = test_prob
    pred_df["y_pred"] = (test_prob >= metrics["decision_threshold"]).astype(int)
    pred_df.to_csv(OUT_DIR / "hybridgru_q90_test_predictions.csv", index=False, encoding="utf-8-sig")

    row = {
        "label_name": LABEL_NAME,
        "delay_days": DELAY_DAYS,
        "positive_count": positive_count,
        "positive_ratio": positive_ratio,
        "train_samples": int(len(split["train"])),
        "validation_samples": int(len(split["validation"])),
        "test_samples": int(len(split["test"])),
        "model": spec.name,
        "decision_threshold": metrics["decision_threshold"],
        "val_best_f1": metrics["val_best_f1"],
        "best_val_average_precision": metrics["best_val_average_precision"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
        "average_precision": metrics["average_precision"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
        "epochs_trained": metrics["epochs_trained"],
    }
    row = _round_metrics(row)
    pd.DataFrame([row]).to_csv(RESULTS_DIR / "q90_hybridgru_sensitivity.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "hybridgru_q90_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# q90 HybridGRU Sensitivity",
        "",
        f"- Label: `{LABEL_NAME}` (`target_gap > {DELAY_DAYS}`)",
        f"- Positive count: {positive_count:,}",
        f"- Positive ratio: {positive_ratio:.4f}",
        f"- Decision threshold: {metrics['decision_threshold']:.2f}",
        f"- Precision: {metrics['precision']:.4f}",
        f"- Recall: {metrics['recall']:.4f}",
        f"- F1-score: {metrics['f1_score']:.4f}",
        f"- ROC-AUC: {metrics['roc_auc']:.4f}",
        f"- Average Precision: {metrics['average_precision']:.4f}",
        f"- Confusion matrix: TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}",
        "",
        "## 해석 메모",
        "",
        "q90=24일 기준은 더 심한 지연만 양성으로 정의하므로 positive class가 줄어든다.",
        "따라서 precision, recall, F1은 q80=15일 기준과 직접적으로 같은 난이도의 점수로 비교하기 어렵고, 라벨 기준 변화에 따른 민감도 결과로 해석해야 한다.",
    ]
    (RESULTS_DIR / "q90_hybridgru_sensitivity_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print_section("Saved")
    for path in [
        RESULTS_DIR / "q90_hybridgru_sensitivity.csv",
        RESULTS_DIR / "q90_hybridgru_sensitivity_summary.md",
        OUT_DIR / "hybridgru_q90_test_predictions.csv",
        OUT_DIR / "hybridgru_q90_training_history.csv",
        OUT_DIR / "hybridgru_q90_metadata.json",
    ]:
        print(path)


if __name__ == "__main__":
    main()
