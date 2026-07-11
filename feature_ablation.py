from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =========================
# 경로 및 기본 설정
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "model_base_hv.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "delay_risk"
META_COLS = ["user_id", "target_order_id", "target_order_number", "target_gap"]
RANDOM_STATE = 42

ALL_FEATURES = [
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

# 현재 최고 MLP 설정
BEST_MLP_PARAMS = {
    "hidden_layer_sizes": (128, 64),
    "alpha": 1e-4,
    "learning_rate_init": 1e-3,
    "batch_size": 64,
}


def print_section(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if LABEL_COL not in df.columns:
        raise ValueError(f"라벨 컬럼이 없습니다: {LABEL_COL}")
    return df


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def build_mlp_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=BEST_MLP_PARAMS["hidden_layer_sizes"],
            activation="relu",
            alpha=BEST_MLP_PARAMS["alpha"],
            learning_rate_init=BEST_MLP_PARAMS["learning_rate_init"],
            batch_size=BEST_MLP_PARAMS["batch_size"],
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_STATE,
        )),
    ])


def run_single_experiment(
    df: pd.DataFrame,
    experiment_name: str,
    feature_cols: list[str],
) -> dict:
    X = df[feature_cols].copy()
    y = df[LABEL_COL].astype(int).copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    model = build_mlp_pipeline()
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    best_threshold, best_val_f1 = find_best_threshold(y_val.to_numpy(), val_prob)

    test_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.to_numpy(), test_prob, best_threshold)

    return {
        "experiment_name": experiment_name,
        "n_features": len(feature_cols),
        "features": ", ".join(feature_cols),
        "threshold": best_threshold,
        "val_best_f1": best_val_f1,
        **metrics,
    }


def main() -> None:
    df = load_data()

    available_features = [col for col in ALL_FEATURES if col in df.columns]
    if len(available_features) == 0:
        raise ValueError("사용 가능한 feature가 없습니다.")

    print_section("사용 가능한 전체 feature")
    for col in available_features:
        print("-", col)

    experiments = {}

    # 1) 전체 feature
    experiments["all_features"] = available_features.copy()

    # 2) active_span_days 제거
    experiments["drop_active_span_days"] = [
        col for col in available_features if col != "active_span_days"
    ]

    # 3) 약한 feature 제거
    experiments["drop_weak_time_features"] = [
        col for col in available_features
        if col not in ["weekend_order_ratio", "dow_variability"]
    ]

    # 4) 최근성/패턴 중심 feature만
    experiments["recent_pattern_only"] = [
        col for col in [
            "recent_3_avg_gap",
            "recent_5_avg_gap",
            "last_gap_before_target",
            "gap_trend",
            "std_gap_before_target",
            "avg_gap_before_target",
            "max_gap_before_target",
            "order_frequency",
            "total_orders_before_target",
        ]
        if col in available_features
    ]

    # 5) 최근성 + 활동기간 제외
    experiments["recent_pattern_no_active_span"] = [
        col for col in [
            "recent_3_avg_gap",
            "recent_5_avg_gap",
            "last_gap_before_target",
            "gap_trend",
            "std_gap_before_target",
            "avg_gap_before_target",
            "max_gap_before_target",
            "order_frequency",
            "total_orders_before_target",
            "weekend_order_ratio",
            "dow_variability",
        ]
        if col in available_features
    ]

    print_section("Ablation 실험 시작")
    results = []

    for exp_name, feature_cols in experiments.items():
        print(f"\n[실험] {exp_name}")
        print(f"feature 수: {len(feature_cols)}")
        result = run_single_experiment(df, exp_name, feature_cols)
        results.append(result)
        print(
            f"Precision={result['precision']:.4f}, "
            f"Recall={result['recall']:.4f}, "
            f"F1={result['f1_score']:.4f}, "
            f"ROC-AUC={result['roc_auc']:.4f}, "
            f"AP={result['average_precision']:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values(
        ["f1_score", "roc_auc", "average_precision"], ascending=False
    )

    print_section("Ablation 결과 요약")
    print(
        results_df[
            [
                "experiment_name",
                "n_features",
                "threshold",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "average_precision",
            ]
        ].round(4)
    )

    results_df.to_csv(
        RESULTS_DIR / "feature_ablation_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 시각화
    plot_df = results_df.copy()
    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["experiment_name"], plot_df["f1_score"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("F1-score")
    plt.title("Feature Ablation Comparison")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_ablation_f1.png", dpi=150)
    plt.close()

    # 요약 markdown
    best_row = results_df.iloc[0]
    lines = [
        "# Feature Ablation Summary",
        "",
        f"- 데이터 파일: `{DATA_PATH.name}`",
        f"- 실험 수: {len(results_df)}",
        "",
        "## 결과 요약",
    ]

    for _, row in results_df.iterrows():
        lines.extend([
            f"### {row['experiment_name']}",
            f"- feature 수: {row['n_features']}",
            f"- threshold: {row['threshold']:.2f}",
            f"- Precision: {row['precision']:.4f}",
            f"- Recall: {row['recall']:.4f}",
            f"- F1-score: {row['f1_score']:.4f}",
            f"- ROC-AUC: {row['roc_auc']:.4f}",
            f"- Average Precision: {row['average_precision']:.4f}",
            "",
        ])

    lines.extend([
        "## 최종 해석",
        f"- 최고 설정(F1 기준): {best_row['experiment_name']}",
        "- `drop_active_span_days` 성능이 크게 떨어지면 active_span_days 의존도가 높다고 볼 수 있다.",
        "- `drop_weak_time_features` 성능이 거의 유지되면 weekend_order_ratio, dow_variability는 제거 후보가 될 수 있다.",
        "- `recent_pattern_only` 성능이 잘 나오면 최근성 중심 lightweight 모델 설명이 가능하다.",
    ])

    with open(RESULTS_DIR / "feature_ablation_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print_section("저장 완료")
    print(RESULTS_DIR / "feature_ablation_results.csv")
    print(RESULTS_DIR / "feature_ablation_f1.png")
    print(RESULTS_DIR / "feature_ablation_summary.md")


if __name__ == "__main__":
    main()