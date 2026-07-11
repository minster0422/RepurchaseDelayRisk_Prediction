from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "model_base_hv.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_GAP_COL = "target_gap"
RANDOM_STATE = 42

THRESHOLDS = [
    {"label_name": "delay_gt_15_q80", "delay_days": 15, "basis": "q80 조기탐지 기준"},
    {"label_name": "delay_gt_24_q90", "delay_days": 24, "basis": "q90 고위험 기준"},
]

FEATURES = [
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


def load_data() -> tuple[pd.DataFrame, list[str]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if TARGET_GAP_COL not in df.columns:
        raise ValueError(f"`{TARGET_GAP_COL}` 컬럼이 없습니다.")

    feature_cols = [col for col in FEATURES if col in df.columns]
    if not feature_cols:
        raise ValueError("사용 가능한 feature 컬럼을 찾지 못했습니다.")

    return df, feature_cols


def make_pipeline(model: Any, scale: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def build_models(y_train: pd.Series) -> list[dict[str, Any]]:
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / pos if pos else 1.0

    models: list[dict[str, Any]] = [
        {
            "model": "DummyClassifier",
            "family": "baseline",
            "estimator": make_pipeline(DummyClassifier(strategy="prior"), scale=False),
            "threshold_tuning": False,
        },
        {
            "model": "LogisticRegression",
            "family": "linear",
            "estimator": make_pipeline(
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
                scale=True,
            ),
            "threshold_tuning": True,
        },
        {
            "model": "MLP",
            "family": "neural_network",
            "estimator": make_pipeline(
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    batch_size=64,
                    max_iter=400,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=RANDOM_STATE,
                ),
                scale=True,
            ),
            "threshold_tuning": True,
        },
    ]

    try:
        from lightgbm import LGBMClassifier

        models.append(
            {
                "model": "LightGBM",
                "family": "tree_boosting",
                "estimator": make_pipeline(
                    LGBMClassifier(
                        n_estimators=400,
                        learning_rate=0.03,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                    scale=False,
                ),
                "threshold_tuning": True,
            }
        )
    except ImportError:
        pass

    try:
        from xgboost import XGBClassifier

        models.append(
            {
                "model": "XGBoost",
                "family": "tree_boosting",
                "estimator": make_pipeline(
                    XGBClassifier(
                        n_estimators=400,
                        learning_rate=0.03,
                        max_depth=4,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        scale_pos_weight=scale_pos_weight,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                    scale=False,
                ),
                "threshold_tuning": True,
            }
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier

        models.append(
            {
                "model": "CatBoost",
                "family": "tree_boosting",
                "estimator": make_pipeline(
                    CatBoostClassifier(
                        iterations=400,
                        learning_rate=0.03,
                        depth=6,
                        loss_function="Logloss",
                        eval_metric="AUC",
                        auto_class_weights="Balanced",
                        random_seed=RANDOM_STATE,
                        verbose=False,
                    ),
                    scale=False,
                ),
                "threshold_tuning": True,
            }
        )
    except ImportError:
        pass

    return models


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(score)
    return best_threshold, best_f1


def predict_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def metric_row(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
    }


def run_for_label(df: pd.DataFrame, feature_cols: list[str], threshold_info: dict[str, Any]) -> list[dict[str, Any]]:
    label_name = threshold_info["label_name"]
    delay_days = int(threshold_info["delay_days"])
    y = (df[TARGET_GAP_COL] > delay_days).astype(int)
    X = df[feature_cols].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    rows = []
    for spec in build_models(y_train):
        model = spec["estimator"]
        model.fit(X_train, y_train)

        if spec["threshold_tuning"]:
            val_prob = predict_probability(model, X_val)
            decision_threshold, val_best_f1 = find_best_threshold(y_val.to_numpy(), val_prob)
        else:
            decision_threshold = 0.5
            val_best_f1 = None

        test_prob = predict_probability(model, X_test)
        metrics = metric_row(y_test.to_numpy(), test_prob, decision_threshold)

        rows.append(
            {
                "label_name": label_name,
                "basis": threshold_info["basis"],
                "delay_days": delay_days,
                "positive_count": int(y.sum()),
                "positive_ratio": float(y.mean()),
                "train_samples": int(len(y_train)),
                "validation_samples": int(len(y_val)),
                "test_samples": int(len(y_test)),
                "model": spec["model"],
                "family": spec["family"],
                "threshold_tuned_on_val": bool(spec["threshold_tuning"]),
                "decision_threshold": decision_threshold,
                "val_best_f1": val_best_f1,
                **metrics,
            }
        )

    return rows


def save_plot(results: pd.DataFrame) -> None:
    plot_df = results[results["model"] != "DummyClassifier"].copy()
    pivot = plot_df.pivot(index="model", columns="label_name", values="f1_score")
    pivot = pivot.sort_values("delay_gt_15_q80", ascending=True)

    ax = pivot.plot(kind="barh", figsize=(9, 5), color=["#2f6f73", "#d98f45"])
    ax.set_title("Sensitivity Analysis: Delay Threshold")
    ax.set_xlabel("F1-score")
    ax.set_xlim(0, max(0.6, float(pivot.max().max()) + 0.05))
    ax.grid(axis="x", alpha=0.25)
    ax.legend(["target_gap > 15 (q80)", "target_gap > 24 (q90)"], loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "delay_threshold_sensitivity_f1.png", dpi=150)
    plt.close()


def save_summary(results: pd.DataFrame) -> None:
    best_15 = results[results["label_name"] == "delay_gt_15_q80"].sort_values("f1_score", ascending=False).iloc[0]
    best_24 = results[results["label_name"] == "delay_gt_24_q90"].sort_values("f1_score", ascending=False).iloc[0]

    pos_info = (
        results[["label_name", "basis", "delay_days", "positive_count", "positive_ratio"]]
        .drop_duplicates()
        .sort_values("delay_days")
    )

    lines = [
        "# Delay Threshold Sensitivity Summary",
        "",
        "## 목적",
        "`target_gap > 15` q80 조기탐지 기준과 `target_gap > 24` q90 고위험 기준을 비교해, 라벨 기준 선택의 영향을 확인한다.",
        "",
        "## 라벨별 양성 비율",
    ]

    for _, row in pos_info.iterrows():
        lines.append(
            f"- `{row['label_name']}`: threshold `{int(row['delay_days'])}`일, "
            f"양성 수 `{int(row['positive_count']):,}`, 양성 비율 `{row['positive_ratio']:.4f}` ({row['basis']})"
        )

    lines.extend(
        [
            "",
            "## F1 기준 best model",
            f"- q80 기준(`target_gap > 15`): {best_15['model']} (F1={best_15['f1_score']:.4f}, Recall={best_15['recall']:.4f}, ROC-AUC={best_15['roc_auc']:.4f})",
            f"- q90 기준(`target_gap > 24`): {best_24['model']} (F1={best_24['f1_score']:.4f}, Recall={best_24['recall']:.4f}, ROC-AUC={best_24['roc_auc']:.4f})",
            "",
            "## 발표용 해석",
            "- q80 기준은 양성 클래스가 더 많아 조기 위험 탐지와 모델 학습 안정성에 유리하다.",
            "- q90 기준은 더 엄격한 고위험 지연을 정의하지만 양성 클래스가 줄어들어 Recall/F1 해석이 더 민감해질 수 있다.",
            "- 중간발표에서는 q80 기준을 기본 실험으로 사용하고, q90 기준은 교수님 피드백을 받아 최종 기준으로 검토할 대안이라고 설명한다.",
        ]
    )

    with open(RESULTS_DIR / "delay_threshold_sensitivity_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    df, feature_cols = load_data()
    rows = []
    for threshold_info in THRESHOLDS:
        print(f"실행 중: {threshold_info['label_name']}")
        rows.extend(run_for_label(df, feature_cols, threshold_info))

    results = pd.DataFrame(rows)
    results.to_csv(
        RESULTS_DIR / "delay_threshold_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "thresholds": THRESHOLDS,
        "random_state": RANDOM_STATE,
        "split": "stratified random train/validation/test = 70/15/15",
    }
    with open(RESULTS_DIR / "delay_threshold_sensitivity_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    save_plot(results)
    save_summary(results)

    print("저장 완료:")
    print(RESULTS_DIR / "delay_threshold_sensitivity.csv")
    print(RESULTS_DIR / "delay_threshold_sensitivity_metadata.json")
    print(RESULTS_DIR / "delay_threshold_sensitivity_f1.png")
    print(RESULTS_DIR / "delay_threshold_sensitivity_summary.md")


if __name__ == "__main__":
    main()
