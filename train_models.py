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
# 경로 및 기본 설정
# =========================
BASE_DIR = Path(r"C:\Users\user\20221275")
DATA_PATH = BASE_DIR / "model_base_hv.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "delay_risk"
META_COLS = ["user_id", "target_order_id", "target_order_number", "target_gap"]

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

SEEDS = [42, 52, 62, 72, 82]

MLP_PARAM_GRID = [
    {
        "hidden_layer_sizes": (64, 32),
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
        "batch_size": 64,
    },
    {
        "hidden_layer_sizes": (128, 64),
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
        "batch_size": 64,
    },
    {
        "hidden_layer_sizes": (128, 64, 32),
        "alpha": 1e-4,
        "learning_rate_init": 5e-4,
        "batch_size": 64,
    },
    {
        "hidden_layer_sizes": (128, 64),
        "alpha": 1e-3,
        "learning_rate_init": 5e-4,
        "batch_size": 128,
    },
]


def print_section(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")


def load_data() -> tuple[pd.DataFrame, list[str]]:
    print_section("1. 데이터 로드")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"데이터 경로: {DATA_PATH}")
    print(f"데이터 크기: {df.shape}")

    if LABEL_COL not in df.columns:
        raise ValueError(f"라벨 컬럼이 없습니다: {LABEL_COL}")

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

    print(f"양성 비율(delay_risk=1): {df[LABEL_COL].mean():.4f}")
    return df, available_features


def split_data(df: pd.DataFrame, feature_cols: list[str], seed: int):
    X = df[feature_cols].copy()
    y = df[LABEL_COL].astype(int).copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=seed,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=seed,
    )
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


def build_logistic_pipeline(seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        )),
    ])


def build_mlp_pipeline(seed: int, params: dict) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            activation="relu",
            alpha=params["alpha"],
            learning_rate_init=params["learning_rate_init"],
            batch_size=params["batch_size"],
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        )),
    ])


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
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
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

    print_section("2. Logistic / MLP 반복 실험")
    search_rows = []

    # Logistic 반복
    for seed in SEEDS:
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, feature_cols, seed)
        logistic = build_logistic_pipeline(seed)
        logistic.fit(X_train, y_train)

        val_prob = logistic.predict_proba(X_val)[:, 1]
        threshold, _ = find_best_threshold(y_val.to_numpy(), val_prob)
        test_prob = logistic.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test.to_numpy(), test_prob, threshold)

        search_rows.append({
            "model_name": "LogisticRegression",
            "seed": seed,
            "hidden_layer_sizes": None,
            "alpha": None,
            "learning_rate_init": None,
            "batch_size": None,
            "threshold": threshold,
            **metrics,
        })

    # MLP 반복 + 하이퍼파라미터 탐색
    for params in MLP_PARAM_GRID:
        for seed in SEEDS:
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, feature_cols, seed)
            mlp = build_mlp_pipeline(seed, params)
            mlp.fit(X_train, y_train)

            val_prob = mlp.predict_proba(X_val)[:, 1]
            threshold, _ = find_best_threshold(y_val.to_numpy(), val_prob)
            test_prob = mlp.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(y_test.to_numpy(), test_prob, threshold)

            search_rows.append({
                "model_name": "MLP",
                "seed": seed,
                "hidden_layer_sizes": str(params["hidden_layer_sizes"]),
                "alpha": params["alpha"],
                "learning_rate_init": params["learning_rate_init"],
                "batch_size": params["batch_size"],
                "threshold": threshold,
                **metrics,
            })

    search_df = pd.DataFrame(search_rows)
    search_df.to_csv(RESULTS_DIR / "tuning_search_results.csv", index=False, encoding="utf-8-sig")

    print_section("3. 반복 실험 요약")
    summary_rows = []

    logistic_df = search_df[search_df["model_name"] == "LogisticRegression"].copy()
    summary_rows.append({
        "model_name": "LogisticRegression",
        "setting": "baseline",
        "precision_mean": logistic_df["precision"].mean(),
        "precision_std": logistic_df["precision"].std(),
        "recall_mean": logistic_df["recall"].mean(),
        "recall_std": logistic_df["recall"].std(),
        "f1_mean": logistic_df["f1_score"].mean(),
        "f1_std": logistic_df["f1_score"].std(),
        "roc_auc_mean": logistic_df["roc_auc"].mean(),
        "roc_auc_std": logistic_df["roc_auc"].std(),
        "ap_mean": logistic_df["average_precision"].mean(),
        "ap_std": logistic_df["average_precision"].std(),
    })

    mlp_grouped = (
        search_df[search_df["model_name"] == "MLP"]
        .groupby(["hidden_layer_sizes", "alpha", "learning_rate_init", "batch_size"], dropna=False)
        .agg(
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1_score", "mean"),
            f1_std=("f1_score", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            ap_mean=("average_precision", "mean"),
            ap_std=("average_precision", "std"),
        )
        .reset_index()
        .sort_values(["f1_mean", "roc_auc_mean"], ascending=False)
    )

    for _, row in mlp_grouped.iterrows():
        summary_rows.append({
            "model_name": "MLP",
            "setting": f"hidden={row['hidden_layer_sizes']}, alpha={row['alpha']}, lr={row['learning_rate_init']}, batch={row['batch_size']}",
            "precision_mean": row["precision_mean"],
            "precision_std": row["precision_std"],
            "recall_mean": row["recall_mean"],
            "recall_std": row["recall_std"],
            "f1_mean": row["f1_mean"],
            "f1_std": row["f1_std"],
            "roc_auc_mean": row["roc_auc_mean"],
            "roc_auc_std": row["roc_auc_std"],
            "ap_mean": row["ap_mean"],
            "ap_std": row["ap_std"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "tuning_summary.csv", index=False, encoding="utf-8-sig")
    print(summary_df.round(4).head(10))

    # 최고 MLP 설정 선택
    best_mlp = mlp_grouped.iloc[0]
    best_mlp_params = {
        "hidden_layer_sizes": eval(best_mlp["hidden_layer_sizes"]),
        "alpha": float(best_mlp["alpha"]),
        "learning_rate_init": float(best_mlp["learning_rate_init"]),
        "batch_size": int(best_mlp["batch_size"]),
    }

    print_section("4. 최종 대표 실행")
    print("선택된 최고 MLP 설정:", best_mlp_params)

    seed = 42
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, feature_cols, seed)

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)
    dummy_prob = dummy.predict_proba(X_test)[:, 1]
    dummy_metrics = compute_metrics(y_test.to_numpy(), dummy_prob, 0.5)

    logistic = build_logistic_pipeline(seed)
    logistic.fit(X_train, y_train)
    logistic_val_prob = logistic.predict_proba(X_val)[:, 1]
    logistic_threshold, _ = find_best_threshold(y_val.to_numpy(), logistic_val_prob)
    logistic_test_prob = logistic.predict_proba(X_test)[:, 1]
    logistic_metrics = compute_metrics(y_test.to_numpy(), logistic_test_prob, logistic_threshold)

    mlp = build_mlp_pipeline(seed, best_mlp_params)
    mlp.fit(X_train, y_train)
    mlp_val_prob = mlp.predict_proba(X_val)[:, 1]
    mlp_threshold, _ = find_best_threshold(y_val.to_numpy(), mlp_val_prob)
    mlp_test_prob = mlp.predict_proba(X_test)[:, 1]
    mlp_metrics = compute_metrics(y_test.to_numpy(), mlp_test_prob, mlp_threshold)

    final_rows = [
        {"model_name": "Dummy", "threshold": 0.5, **dummy_metrics},
        {"model_name": "LogisticRegression", "threshold": logistic_threshold, **logistic_metrics},
        {"model_name": "MLP", "threshold": mlp_threshold, **mlp_metrics},
    ]
    final_df = pd.DataFrame(final_rows).sort_values(["f1_score", "roc_auc"], ascending=False)
    final_df.to_csv(RESULTS_DIR / "final_metrics.csv", index=False, encoding="utf-8-sig")

    thresholds = {
        "dummy_classifier": 0.5,
        "logistic_regression": float(logistic_threshold),
        "mlp": float(mlp_threshold),
        "lstm": None,
        "best_mlp_params": {
            "hidden_layer_sizes": list(best_mlp_params["hidden_layer_sizes"]),
            "alpha": best_mlp_params["alpha"],
            "learning_rate_init": best_mlp_params["learning_rate_init"],
            "batch_size": best_mlp_params["batch_size"],
        },
        "_note": "LogisticRegression과 MLP는 validation set 기준 F1 최대 threshold를 사용했다. LSTM은 아직 미실행 상태다.",
    }
    with open(RESULTS_DIR / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)

    all_probs = {
        "Dummy": dummy_prob,
        "LogisticRegression": logistic_test_prob,
        "MLP": mlp_test_prob,
    }
    save_roc_curve(all_probs, y_test.to_numpy())
    save_pr_curve(all_probs, y_test.to_numpy())

    for row in final_rows:
        cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        save_confusion_matrix(cm, row["model_name"])

    best_row = final_df.iloc[0]
    summary_lines = [
        "# Tuned Run Summary",
        "",
        f"- 데이터 파일: `{DATA_PATH.name}`",
        f"- 샘플 수: {len(df):,}",
        f"- 양성 비율: {df[LABEL_COL].mean():.4f}",
        f"- 사용 feature 수: {len(feature_cols)}",
        f"- 반복 seed 수: {len(SEEDS)}",
        f"- MLP 후보 설정 수: {len(MLP_PARAM_GRID)}",
        "",
        "## 선택된 최고 MLP 설정",
        f"- hidden_layer_sizes: {best_mlp_params['hidden_layer_sizes']}",
        f"- alpha: {best_mlp_params['alpha']}",
        f"- learning_rate_init: {best_mlp_params['learning_rate_init']}",
        f"- batch_size: {best_mlp_params['batch_size']}",
        "",
        "## 최종 대표 실행 결과",
    ]

    for _, row in final_df.iterrows():
        summary_lines.extend([
            f"### {row['model_name']}",
            f"- Threshold: {row['threshold']:.2f}",
            f"- Precision: {row['precision']:.4f}",
            f"- Recall: {row['recall']:.4f}",
            f"- F1-score: {row['f1_score']:.4f}",
            f"- ROC-AUC: {row['roc_auc']:.4f}",
            f"- Average Precision: {row['average_precision']:.4f}",
            f"- Confusion Matrix: TN={row['tn']}, FP={row['fp']}, FN={row['fn']}, TP={row['tp']}",
            "",
        ])

    summary_lines.extend([
        "## 최종 요약",
        f"- 현재 최고 모델(F1 기준): {best_row['model_name']}",
        "- 이번 실행은 Dummy / LogisticRegression / MLP만 수행했다.",
        "- LSTM은 sequence pipeline 준비 후 별도 실행 필요.",
    ])

    with open(RESULTS_DIR / "run_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print_section("5. 저장 완료")
    print(f"결과 저장 폴더: {RESULTS_DIR.resolve()}")
    print("생성 파일:")
    print("- tuning_search_results.csv")
    print("- tuning_summary.csv")
    print("- final_metrics.csv")
    print("- thresholds.json")
    print("- roc_curve.png")
    print("- pr_curve.png")
    print("- confusion_matrix_*.png")
    print("- run_summary.md")


if __name__ == "__main__":
    main()