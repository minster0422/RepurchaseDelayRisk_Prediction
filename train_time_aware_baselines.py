from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


PROJECT_ROOT = Path(__file__).resolve().parent
TIME_DIR = PROJECT_ROOT / "time_aware"
DATA_PATH = TIME_DIR / "time_aware_model_base.csv"
OUT_CSV = TIME_DIR / "time_aware_model_comparison.csv"
OUT_JSON = TIME_DIR / "time_aware_thresholds.json"
OUT_MD = TIME_DIR / "time_aware_model_summary.md"
OUT_PNG = TIME_DIR / "time_aware_model_comparison.png"

SEED = 42

DROP_COLS = {
    "user_id",
    "target_order_id",
    "target_eval_set",
    "target_gap",
    "delay_risk",
    "split",
    # Not available at prediction time or directly derived from the target gap.
    "total_orders",
    "days_since_first_order_to_target",
}


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    thresholds = np.round(np.arange(0.01, 0.81, 0.01), 2)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, float(best_f1)


def metric_row(
    model_name: str,
    family: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    val_best_f1: float | None,
) -> dict[str, float | int | str | bool | None]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model": model_name,
        "family": family,
        "threshold_tuned_on_val": val_best_f1 is not None,
        "decision_threshold": threshold,
        "val_best_f1": val_best_f1,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def plot_summary(results: pd.DataFrame) -> None:
    plot_df = results[results["model"] != "DummyClassifier"].copy()
    x = np.arange(len(plot_df))
    width = 0.22
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.bar(x - width, plot_df["f1_score"], width, label="F1-score", color="#2F7776")
    ax.bar(x, plot_df["recall"], width, label="Recall", color="#8FB3FF")
    ax.bar(x + width, plot_df["average_precision"], width, label="Average Precision", color="#E39D43")
    for offset, col in [(-width, "f1_score"), (0, "recall"), (width, "average_precision")]:
        for i, value in enumerate(plot_df[col]):
            ax.text(i + offset, value + 0.01, f"{value:.3f}", ha="center", fontsize=9)
    ax.set_title("Time-Aware Baseline Results\norder-lifecycle split, target_gap > 15", fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_xticks(x, plot_df["model"])
    ax.set_ylim(0, max(0.7, float(plot_df[["f1_score", "recall", "average_precision"]].max().max()) + 0.08))
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.01,
        0.01,
        "Note: validation/test positive ratios are much lower than the one-row-per-user baseline, so compare cautiously.",
        fontsize=9,
        color="#555555",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    feature_cols = [
        c for c in df.columns
        if c not in DROP_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]

    train = df[df["split"] == "train"]
    val = df[df["split"] == "validation"]
    test = df[df["split"] == "test"]

    x_train = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    x_val = val[feature_cols].replace([np.inf, -np.inf], np.nan)
    x_test = test[feature_cols].replace([np.inf, -np.inf], np.nan)
    medians = x_train.median()
    x_train = x_train.fillna(medians)
    x_val = x_val.fillna(medians)
    x_test = x_test.fillna(medians)

    y_train = train["delay_risk"].astype(int).to_numpy()
    y_val = val["delay_risk"].astype(int).to_numpy()
    y_test = test["delay_risk"].astype(int).to_numpy()

    models: list[tuple[str, str, object, bool]] = [
        ("DummyClassifier", "baseline", DummyClassifier(strategy="prior"), False),
        (
            "LogisticRegression",
            "linear",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=SEED,
                ),
            ),
            True,
        ),
    ]
    if LGBMClassifier is not None:
        scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
        models.append(
            (
                "LightGBM",
                "tree_boosting",
                LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    scale_pos_weight=scale_pos_weight,
                    random_state=SEED,
                    n_jobs=-1,
                    verbose=-1,
                ),
                True,
            )
        )

    rows: list[dict[str, float | int | str | bool | None]] = []
    thresholds: dict[str, object] = {
        "_note": "Time-aware v1 uses predefined order-lifecycle split. Thresholds except Dummy are tuned on validation F1.",
        "_feature_cols": feature_cols,
        "_split_counts": {
            "train": int(len(train)),
            "validation": int(len(val)),
            "test": int(len(test)),
        },
        "_positive_ratio": {
            "train": float(y_train.mean()),
            "validation": float(y_val.mean()),
            "test": float(y_test.mean()),
        },
    }

    for name, family, model, tune_threshold in models:
        model.fit(x_train, y_train)
        if hasattr(model, "predict_proba"):
            val_prob = model.predict_proba(x_val)[:, 1]
            test_prob = model.predict_proba(x_test)[:, 1]
        else:
            val_prob = model.decision_function(x_val)
            test_prob = model.decision_function(x_test)

        if tune_threshold:
            threshold, val_best_f1 = choose_threshold(y_val, val_prob)
        else:
            threshold, val_best_f1 = 0.5, None

        rows.append(metric_row(name, family, y_test, test_prob, threshold, val_best_f1))
        thresholds[name] = {
            "threshold_tuned_on_val": tune_threshold,
            "decision_threshold": threshold,
            "val_best_f1": val_best_f1,
        }

    results = pd.DataFrame(rows).sort_values("f1_score", ascending=False)
    results.insert(0, "rank_by_f1", np.arange(1, len(results) + 1))
    results.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_summary(results)

    best = results.iloc[0]
    lines = [
        "# Time-Aware Baseline Summary",
        "",
        "## 목적",
        "",
        "`time_aware_model_base.csv`의 order-lifecycle split에서 빠른 baseline 성능을 확인한다.",
        "",
        "## 중요한 관찰",
        "",
        "- 이 데이터셋은 사용자별 여러 target order를 만들기 때문에 기존 one-row-per-user 데이터와 양성 비율이 크게 다르다.",
        "- 특히 validation/test로 갈수록 `target_gap > 15` 양성 비율이 낮아져 성능 해석이 더 까다롭다.",
        "- 따라서 이 결과는 기존 random split 결과와 직접 비교하기보다, 최종발표용 time-aware pipeline 후보로 해석한다.",
        "",
        "## split별 양성 비율",
        "",
        f"- train: `{y_train.mean():.4f}`",
        f"- validation: `{y_val.mean():.4f}`",
        f"- test: `{y_test.mean():.4f}`",
        "",
        "## F1 기준 best",
        "",
        f"- {best['model']}: F1 `{best['f1_score']:.4f}`, Recall `{best['recall']:.4f}`, ROC-AUC `{best['roc_auc']:.4f}`, AP `{best['average_precision']:.4f}`",
        "",
        "## 상세 결과",
        "",
        "| model | threshold | precision | recall | f1 | roc_auc | average_precision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in results.iterrows():
        lines.append(
            f"| {row['model']} | {row['decision_threshold']:.2f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1_score']:.4f} | {row['roc_auc']:.4f} | {row['average_precision']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 해석",
            "",
            "현재 time-aware v1은 방법론적으로 더 예측 프로젝트답지만, target lifecycle이 뒤로 갈수록 지연 양성 비율이 급격히 낮아진다.",
            "최종발표에서는 이 구조를 발전시키되, 라벨 기준을 q80/q90 중 다시 잡거나 anchor sampling 방식을 조정하는 것이 필요하다.",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"created={OUT_CSV}")
    print(f"created={OUT_JSON}")
    print(f"created={OUT_MD}")
    print(f"created={OUT_PNG}")


if __name__ == "__main__":
    main()
