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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================
# 기본 설정
# =========================
BASE_DIR = Path(r"C:\Users\user\20221275")
DATA_PATH = BASE_DIR / "model_base_hv.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
LABEL_COL = "delay_risk"

# 발표/문서 기준 핵심 컬럼들
META_COLS = ["user_id", "target_order_id", "target_order_number", "target_gap"]

# feature 후보
PREFERRED_FEATURES = [
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
    line = "=" * 60
    print(f"\n{line}\n{title}\n{line}")


def load_data() -> tuple[pd.DataFrame, list[str]]:
    print_section("1. 데이터 로드")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"데이터 경로: {DATA_PATH}")
    print(f"데이터 크기: {df.shape}")

    missing_required = [col for col in [LABEL_COL] if col not in df.columns]
    if missing_required:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_required}")

    available_features = [col for col in PREFERRED_FEATURES if col in df.columns]
    if not available_features:
        excluded = set(META_COLS + [LABEL_COL])
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        available_features = [col for col in numeric_cols if col not in excluded]

    if not available_features:
        raise ValueError("사용 가능한 feature 컬럼을 찾지 못했습니다.")

    print("사용 feature 컬럼:")
    for col in available_features:
        print(f"- {col}")

    positive_ratio = df[LABEL_COL].mean()
    print(f"양성 비율(delay_risk=1): {positive_ratio:.4f}")

    return df, available_features


def split_data(df: pd.DataFrame, feature_cols: list[str]):
    print_section("2. Train / Val / Test 분할")

    X = df[feature_cols].copy()
    y = df[LABEL_COL].astype(int).copy()

    # 70 / 15 / 15
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

    print(f"Train: {X_train.shape}, 양성 비율={y_train.mean():.4f}")
    print(f"Val  : {X_val.shape}, 양성 비율={y_val.mean():.4f}")
    print(f"Test : {X_test.shape}, 양성 비율={y_test.mean():.4f}")

    return X_train, X_val, X_test, y_train, y_val, y_test


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


def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "model_name": model_name,
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "average_precision": float(ap),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def save_confusion_matrix(cm: np.ndarray, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ylabel="True label",
        xlabel="Predicted label",
        title=f"{model_name} Confusion Matrix",
    )

    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"confusion_matrix_{model_name.lower()}.png", dpi=150)
    plt.close(fig)


def save_roc_curve(all_probs: dict[str, np.ndarray], y_test: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name, y_prob in all_probs.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_score:.4f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)


def save_pr_curve(all_probs: dict[str, np.ndarray], y_test: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name, y_prob in all_probs.items():
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap_score = average_precision_score(y_test, y_prob)
        ax.plot(recall, precision, linewidth=2, label=f"{model_name} (AP={ap_score:.4f})")

    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pr_curve.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df, feature_cols = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, feature_cols)

    print_section("3. 모델 학습")

    # 공통 전처리
    preprocessor = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    # Dummy
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)
    dummy_val_prob = dummy.predict_proba(X_val)[:, 1]
    dummy_test_prob = dummy.predict_proba(X_test)[:, 1]
    dummy_threshold = 0.5

    # Logistic Regression
    logistic = Pipeline(
        preprocessor
        + [
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            )
        ]
    )
    logistic.fit(X_train, y_train)
    logistic_val_prob = logistic.predict_proba(X_val)[:, 1]
    logistic_test_prob = logistic.predict_proba(X_test)[:, 1]
    logistic_threshold, logistic_val_f1 = find_best_threshold(y_val.to_numpy(), logistic_val_prob)

    # MLP
    mlp = Pipeline(
        preprocessor
        + [
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=300,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=RANDOM_STATE,
                ),
            )
        ]
    )
    mlp.fit(X_train, y_train)
    mlp_val_prob = mlp.predict_proba(X_val)[:, 1]
    mlp_test_prob = mlp.predict_proba(X_test)[:, 1]
    mlp_threshold, mlp_val_f1 = find_best_threshold(y_val.to_numpy(), mlp_val_prob)

    print(f"Logistic best threshold (val F1 기준): {logistic_threshold:.2f}, val F1={logistic_val_f1:.4f}")
    print(f"MLP best threshold (val F1 기준): {mlp_threshold:.2f}, val F1={mlp_val_f1:.4f}")

    print_section("4. 테스트셋 평가")

    results = []
    all_test_probs = {
        "Dummy": dummy_test_prob,
        "LogisticRegression": logistic_test_prob,
        "MLP": mlp_test_prob,
    }

    dummy_metrics = evaluate_model("Dummy", y_test.to_numpy(), dummy_test_prob, dummy_threshold)
    logistic_metrics = evaluate_model("LogisticRegression", y_test.to_numpy(), logistic_test_prob, logistic_threshold)
    mlp_metrics = evaluate_model("MLP", y_test.to_numpy(), mlp_test_prob, mlp_threshold)

    results.extend([dummy_metrics, logistic_metrics, mlp_metrics])

    for metrics in results:
        print(
            f"{metrics['model_name']}: "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1_score']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}, "
            f"AP={metrics['average_precision']:.4f}"
        )

    print_section("5. 결과 저장")

    results_df = pd.DataFrame(results).sort_values(
        ["f1_score", "roc_auc", "recall"], ascending=False
    )
    results_df.to_csv(RESULTS_DIR / "final_metrics.csv", index=False, encoding="utf-8-sig")

    thresholds = {
        "dummy_classifier": dummy_threshold,
        "logistic_regression": logistic_threshold,
        "mlp": mlp_threshold,
        "lstm": None,
        "_note": "LogisticRegression과 MLP는 validation set 기준 F1 최대 threshold를 사용했다. LSTM은 아직 미실행 상태다.",
    }
    with open(RESULTS_DIR / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)

    save_roc_curve(all_test_probs, y_test.to_numpy())
    save_pr_curve(all_test_probs, y_test.to_numpy())

    for metrics in results:
        cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
        save_confusion_matrix(cm, metrics["model_name"])

    best_row = results_df.iloc[0]
    run_summary = [
        "# Run Summary",
        "",
        f"- 데이터 파일: `{DATA_PATH.name}`",
        f"- 샘플 수: {len(df):,}",
        f"- 양성 비율: {df[LABEL_COL].mean():.4f}",
        f"- 사용 feature 수: {len(feature_cols)}",
        "",
        "## 모델별 결과",
    ]

    for _, row in results_df.iterrows():
        run_summary.extend(
            [
                f"### {row['model_name']}",
                f"- Threshold: {row['threshold']:.2f}",
                f"- Precision: {row['precision']:.4f}",
                f"- Recall: {row['recall']:.4f}",
                f"- F1-score: {row['f1_score']:.4f}",
                f"- ROC-AUC: {row['roc_auc']:.4f}",
                f"- Average Precision: {row['average_precision']:.4f}",
                f"- Confusion Matrix: TN={row['tn']}, FP={row['fp']}, FN={row['fn']}, TP={row['tp']}",
                "",
            ]
        )

    run_summary.extend(
        [
            "## 최종 요약",
            f"- 현재 최고 모델(F1 기준): {best_row['model_name']}",
            "- 이번 실행은 Dummy / LogisticRegression / MLP만 수행했다.",
            "- LSTM은 sequence pipeline 준비 후 별도 실행 필요.",
        ]
    )

    with open(RESULTS_DIR / "run_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(run_summary))

    print(f"결과 저장 완료: {RESULTS_DIR.resolve()}")
    print("저장 파일:")
    print("- final_metrics.csv")
    print("- thresholds.json")
    print("- roc_curve.png")
    print("- pr_curve.png")
    print("- confusion_matrix_*.png")
    print("- run_summary.md")


if __name__ == "__main__":
    main()