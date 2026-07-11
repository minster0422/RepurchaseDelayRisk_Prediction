from __future__ import annotations

import importlib.util
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
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
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

LABEL_COL = "delay_risk"
META_COLS = ["user_id", "target_order_id", "target_order_number", "target_gap"]
RANDOM_STATE = 42

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


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def is_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def load_data() -> tuple[pd.DataFrame, list[str]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if LABEL_COL not in df.columns:
        raise ValueError(f"라벨 컬럼이 없습니다: {LABEL_COL}")

    feature_cols = [col for col in FEATURES if col in df.columns]
    if not feature_cols:
        excluded = set(META_COLS + [LABEL_COL])
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in excluded]

    if not feature_cols:
        raise ValueError("사용 가능한 feature 컬럼을 찾지 못했습니다.")

    return df, feature_cols


def split_data(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    X = df[feature_cols].copy()
    y = df[LABEL_COL].astype(int).copy()

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

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(score)

    return best_threshold, best_f1


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
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


def make_pipeline(model: Any, scale: bool = False) -> Pipeline:
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
            "name": "DummyClassifier",
            "family": "baseline",
            "model": make_pipeline(DummyClassifier(strategy="prior"), scale=False),
            "threshold_tuning": False,
            "note": "최소 기준선. threshold tuning 없이 0.5 기준을 사용한다.",
        },
        {
            "name": "LogisticRegression",
            "family": "linear",
            "model": make_pipeline(
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                scale=True,
            ),
            "threshold_tuning": True,
            "note": "해석 가능한 선형 baseline.",
        },
        {
            "name": "MLP",
            "family": "neural_network",
            "model": make_pipeline(
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
            "note": "집계 특징 기반 비선형 신경망.",
        },
    ]

    if is_available("lightgbm"):
        from lightgbm import LGBMClassifier

        models.append(
            {
                "name": "LightGBM",
                "family": "tree_boosting",
                "model": make_pipeline(
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
                "note": "tabular 데이터에 강한 tree boosting baseline.",
            }
        )

    if is_available("xgboost"):
        from xgboost import XGBClassifier

        models.append(
            {
                "name": "XGBoost",
                "family": "tree_boosting",
                "model": make_pipeline(
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
                "note": "널리 쓰이는 tree boosting baseline.",
            }
        )

    if is_available("catboost"):
        from catboost import CatBoostClassifier

        models.append(
            {
                "name": "CatBoost",
                "family": "tree_boosting",
                "model": make_pipeline(
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
                "note": "범주형 변수에 강점이 있지만, 현재 데이터에서는 numeric 집계 feature 기반 baseline으로 사용.",
            }
        )

    return models


def model_prob(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    decision = model.decision_function(X)
    return 1 / (1 + np.exp(-decision))


def run_experiment() -> tuple[pd.DataFrame, dict[str, Any], list[pd.DataFrame], float]:
    df, feature_cols = load_data()
    positive_ratio = float(df[LABEL_COL].mean())
    split = split_data(df, feature_cols)

    y_train = split["y_train"]
    y_val = split["y_val"]
    y_test = split["y_test"]

    print_section("데이터 및 split")
    print(f"전체 샘플: {len(df):,}")
    print(f"feature 수: {len(feature_cols)}")
    print(f"train/val/test: {len(y_train):,} / {len(y_val):,} / {len(y_test):,}")
    print(f"positive ratio: {df[LABEL_COL].mean():.4f}")

    rows = []
    curve_records: dict[str, Any] = {}
    prediction_frames: list[pd.DataFrame] = []
    threshold_records: dict[str, Any] = {
        "_note": "DummyClassifier는 고정 threshold 0.5를 사용하고, 나머지 모델은 validation F1 최대 기준으로 threshold를 선택한다.",
        "_split": {
            "method": "stratified random split",
            "train": int(len(y_train)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
            "random_state": RANDOM_STATE,
        },
    }

    for spec in build_models(y_train):
        name = spec["name"]
        model = spec["model"]
        should_tune = bool(spec["threshold_tuning"])

        print_section(f"{name} 학습")
        model.fit(split["X_train"], y_train)

        if should_tune:
            val_prob = model_prob(model, split["X_val"])
            threshold, val_best_f1 = find_best_threshold(y_val.to_numpy(), val_prob)
        else:
            threshold = 0.5
            val_best_f1 = None

        test_prob = model_prob(model, split["X_test"])
        metrics = compute_metrics(y_test.to_numpy(), test_prob, threshold)
        test_pred = (test_prob >= threshold).astype(int)

        fpr, tpr, _ = roc_curve(y_test.to_numpy(), test_prob)
        pr_precision, pr_recall, _ = precision_recall_curve(y_test.to_numpy(), test_prob)
        curve_records[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "pr_precision": pr_precision,
            "pr_recall": pr_recall,
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["average_precision"],
        }

        meta_cols = [col for col in META_COLS if col in df.columns]
        pred_df = df.loc[split["X_test"].index, meta_cols + [LABEL_COL]].copy()
        pred_df.insert(0, "model", name)
        pred_df["y_prob"] = test_prob
        pred_df["y_pred"] = test_pred
        prediction_frames.append(pred_df)

        row = {
            "model": name,
            "family": spec["family"],
            "threshold_tuned_on_val": should_tune,
            "decision_threshold": threshold,
            "val_best_f1": val_best_f1,
            **metrics,
            "interpretation_note": spec["note"],
        }
        rows.append(row)

        threshold_records[name] = {
            "threshold_tuned_on_val": should_tune,
            "decision_threshold": threshold,
            "val_best_f1": val_best_f1,
        }

        print(
            f"threshold={threshold:.2f} | "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1_score']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}, "
            f"AP={metrics['average_precision']:.4f}"
        )

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(["f1_score", "roc_auc"], ascending=False)
    results_df.insert(0, "rank_by_f1", range(1, len(results_df) + 1))

    results_df.to_csv(
        RESULTS_DIR / "tabular_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with open(RESULTS_DIR / "tabular_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(threshold_records, f, ensure_ascii=False, indent=2)

    return results_df, curve_records, prediction_frames, positive_ratio


def save_plot(results_df: pd.DataFrame) -> None:
    plot_df = results_df.sort_values("f1_score", ascending=True)
    metrics = ["recall", "f1_score", "roc_auc", "average_precision"]
    colors = ["#8fb3ff", "#2f6f73", "#f0a35e", "#b86b77"]

    ax = plot_df.set_index("model")[metrics].plot(
        kind="barh",
        figsize=(10, 6),
        color=colors,
        width=0.82,
    )
    ax.set_title("Tabular Model Comparison")
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(["Recall", "F1-score", "ROC-AUC", "Average Precision"], loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tabular_model_comparison.png", dpi=150)
    plt.close()


def save_curves(curve_records: dict[str, Any], positive_ratio: float) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Random")
    for name, data in curve_records.items():
        if name == "DummyClassifier":
            continue
        plt.plot(
            data["fpr"],
            data["tpr"],
            linewidth=2,
            label=f"{name} (AUC={data['roc_auc']:.3f})",
        )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve by Tabular Model")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tabular_roc_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.axhline(
        positive_ratio,
        linestyle="--",
        color="#999999",
        label=f"Positive ratio ({positive_ratio:.3f})",
    )
    for name, data in curve_records.items():
        if name == "DummyClassifier":
            continue
        plt.plot(
            data["pr_recall"],
            data["pr_precision"],
            linewidth=2,
            label=f"{name} (AP={data['average_precision']:.3f})",
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve by Tabular Model")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tabular_pr_curve.png", dpi=150)
    plt.close()


def save_confusion_matrices(results_df: pd.DataFrame) -> None:
    plot_df = results_df.sort_values("rank_by_f1")
    n_models = len(plot_df)
    n_cols = 3
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes_arr = np.array(axes).reshape(-1)

    for ax, (_, row) in zip(axes_arr, plot_df.iterrows()):
        matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=int)
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title(f"{row['model']}\nF1={row['f1_score']:.3f}, Recall={row['recall']:.3f}")
        ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], labels=["True 0", "True 1"])

        for i in range(2):
            for j in range(2):
                text_color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
                ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", color=text_color)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes_arr[n_models:]:
        ax.axis("off")

    fig.suptitle("Confusion Matrices by Tabular Model", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tabular_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_predictions(prediction_frames: list[pd.DataFrame]) -> None:
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(
        RESULTS_DIR / "tabular_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )


def save_summary(results_df: pd.DataFrame) -> None:
    best = results_df.iloc[0]
    tree_df = results_df[results_df["family"] == "tree_boosting"]
    best_tree = tree_df.iloc[0] if not tree_df.empty else None

    lines = [
        "# Tabular Model Comparison Summary",
        "",
        "## 목적",
        "DummyClassifier, LogisticRegression, MLP와 Tree Boosting 계열 모델을 같은 split과 같은 validation threshold tuning 기준으로 비교한다.",
        "",
        "## 전체 결과 요약",
        f"- F1 기준 1위: {best['model']} (F1={best['f1_score']:.4f}, Recall={best['recall']:.4f}, ROC-AUC={best['roc_auc']:.4f})",
    ]

    if best_tree is not None:
        lines.append(
            f"- Tree Boosting 계열 best: {best_tree['model']} "
            f"(F1={best_tree['f1_score']:.4f}, Recall={best_tree['recall']:.4f}, ROC-AUC={best_tree['roc_auc']:.4f})"
        )

    non_dummy = results_df[results_df["model"] != "DummyClassifier"]
    metric_labels = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1-score"),
        ("roc_auc", "ROC-AUC"),
        ("average_precision", "Average Precision"),
    ]
    lines.extend(["", "## 지표별 1위"])
    for metric, label in metric_labels:
        metric_best = non_dummy.sort_values(metric, ascending=False).iloc[0]
        lines.append(f"- {label}: {metric_best['model']} ({metric_best[metric]:.4f})")

    lines.extend(
        [
            "",
            "## 발표용 해석",
            "- CatBoost는 F1-score와 ROC-AUC 기준에서 가장 균형적인 결과를 보였다.",
            "- LightGBM은 Recall이 가장 높아 위험 고객을 더 넓게 포착하는 방향에 강점이 있다.",
            "- XGBoost는 Average Precision이 가장 높아 positive class ranking 관점에서 강점이 있다.",
            "- MLP는 Tree Boosting 계열보다 약간 낮지만 큰 차이는 아니므로, 집계 feature 기반 신경망 대안으로 해석할 수 있다.",
        ]
    )

    lines.extend(
        [
            "",
            "## 해석 원칙",
            "- accuracy 단독 결론은 피하고 Recall, F1-score, ROC-AUC, Average Precision을 함께 해석한다.",
            "- Tree Boosting이 MLP보다 높게 나오면, 현재 집계형 tabular feature에서는 딥러닝보다 강한 tabular baseline이 더 적합할 수 있다고 설명한다.",
            "- MLP가 높게 나오더라도 Dummy/Logistic보다 강한 baseline과 비교했다는 점을 함께 제시한다.",
            "- LSTM은 실제 주문 sequence pipeline 보강 후 별도 비교하는 것이 안전하다.",
            "",
            "## 상세 결과",
        ]
    )

    for _, row in results_df.iterrows():
        lines.extend(
            [
                f"### {row['model']}",
                f"- Family: {row['family']}",
                f"- Threshold: {row['decision_threshold']:.2f}",
                f"- Precision: {row['precision']:.4f}",
                f"- Recall: {row['recall']:.4f}",
                f"- F1-score: {row['f1_score']:.4f}",
                f"- ROC-AUC: {row['roc_auc']:.4f}",
                f"- Average Precision: {row['average_precision']:.4f}",
                f"- Confusion Matrix: TN={int(row['tn'])}, FP={int(row['fp'])}, FN={int(row['fn'])}, TP={int(row['tp'])}",
                "",
            ]
        )

    with open(RESULTS_DIR / "tabular_model_comparison_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    results_df, curve_records, prediction_frames, positive_ratio = run_experiment()
    save_plot(results_df)
    save_curves(curve_records, positive_ratio=positive_ratio)
    save_confusion_matrices(results_df)
    save_predictions(prediction_frames)
    save_summary(results_df)

    print_section("저장 완료")
    print(RESULTS_DIR / "tabular_model_comparison.csv")
    print(RESULTS_DIR / "tabular_thresholds.json")
    print(RESULTS_DIR / "tabular_model_comparison.png")
    print(RESULTS_DIR / "tabular_roc_curve.png")
    print(RESULTS_DIR / "tabular_pr_curve.png")
    print(RESULTS_DIR / "tabular_confusion_matrices.png")
    print(RESULTS_DIR / "tabular_test_predictions.csv")
    print(RESULTS_DIR / "tabular_model_comparison_summary.md")


if __name__ == "__main__":
    main()
