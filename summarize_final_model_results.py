from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"

METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "average_precision",
]

SELECTED_MODELS = [
    "DummyClassifier",
    "LogisticRegression",
    "MLP",
    "CatBoost",
    "LightGBM",
    "XGBoost",
    "LSTM",
    "HybridGRU",
    "HybridBiGRU",
    "HybridTransformer",
]


def load_result(path: Path, experiment: str, input_type: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["experiment"] = experiment
    df["input_type"] = input_type
    return df


def markdown_table(df: pd.DataFrame) -> list[str]:
    view = df.copy().fillna("")
    header = "| " + " | ".join(view.columns.astype(str)) + " |"
    separator = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = [header, separator]
    for _, row in view.iterrows():
        rows.append("| " + " | ".join(str(value) for value in row.tolist()) + " |")
    return rows


def save_metric_plot(df: pd.DataFrame) -> None:
    selected = df[df["model"].isin(SELECTED_MODELS)].copy()
    selected["model"] = pd.Categorical(selected["model"], categories=SELECTED_MODELS, ordered=True)
    selected = selected.sort_values("model")

    metrics = ["recall", "f1_score", "roc_auc", "average_precision"]
    plot_df = selected.set_index("model")[metrics]
    ax = plot_df.plot(kind="barh", figsize=(11, 7), width=0.82)
    ax.set_title("Final Model Comparison")
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(["Recall", "F1-score", "ROC-AUC", "Average Precision"], loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "final_model_comparison.png", dpi=150)
    plt.close()


def load_predictions(path: Path, experiment: str) -> pd.DataFrame:
    pred = pd.read_csv(path)
    pred["experiment"] = experiment
    return pred


def save_selected_curves() -> None:
    pred_frames = [
        load_predictions(RESULTS_DIR / "tabular_test_predictions.csv", "tabular_aggregate"),
        load_predictions(RESULTS_DIR / "deep_sequence_test_predictions.csv", "sequence_only"),
        load_predictions(RESULTS_DIR / "hybrid_deep_test_predictions.csv", "hybrid_sequence_tabular"),
    ]
    pred = pd.concat(pred_frames, ignore_index=True)
    selected = pred[pred["model"].isin(SELECTED_MODELS)].copy()

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Random")
    for model in SELECTED_MODELS:
        model_df = selected[selected["model"] == model]
        if model_df.empty:
            continue
        y_true = model_df["delay_risk"].astype(int).to_numpy()
        y_prob = model_df["y_prob"].astype(float).to_numpy()
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, linewidth=2, label=f"{model} ({auc:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Selected Models - ROC Curve")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "final_selected_roc_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    positive_ratio = float(selected["delay_risk"].mean())
    plt.axhline(positive_ratio, linestyle="--", color="#999999", label=f"Positive ratio ({positive_ratio:.3f})")
    for model in SELECTED_MODELS:
        model_df = selected[selected["model"] == model]
        if model_df.empty:
            continue
        y_true = model_df["delay_risk"].astype(int).to_numpy()
        y_prob = model_df["y_prob"].astype(float).to_numpy()
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, linewidth=2, label=f"{model} ({ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Selected Models - Precision-Recall Curve")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "final_selected_pr_curve.png", dpi=150)
    plt.close()


def save_summary(df: pd.DataFrame) -> None:
    sorted_df = df.sort_values(["f1_score", "recall", "average_precision"], ascending=False).reset_index(drop=True)
    best_f1 = sorted_df.iloc[0]
    best_recall = df.sort_values("recall", ascending=False).iloc[0]
    best_auc = df.sort_values("roc_auc", ascending=False).iloc[0]
    best_ap = df.sort_values("average_precision", ascending=False).iloc[0]

    presentation_df = sorted_df[
        [
            "rank_by_f1_final",
            "model",
            "experiment",
            "input_type",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "average_precision",
            "decision_threshold",
        ]
    ].copy()
    for col in ["precision", "recall", "f1_score", "roc_auc", "average_precision", "decision_threshold"]:
        presentation_df[col] = presentation_df[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    lines = [
        "# Final Model Comparison Summary",
        "",
        "이 문서는 최종 발표용 전체 모델 비교 요약이다. 같은 test split에서 정형 요약 feature 모델, 순수 sequence 딥러닝 모델, 하이브리드 딥러닝 모델을 함께 비교한다.",
        "",
        "## Key Result",
        "",
        f"- F1-score 기준 최상위 모델: {best_f1['model']} ({best_f1['f1_score']:.4f})",
        f"- Recall 기준 최상위 모델: {best_recall['model']} ({best_recall['recall']:.4f})",
        f"- ROC-AUC 기준 최상위 모델: {best_auc['model']} ({best_auc['roc_auc']:.4f})",
        f"- Average Precision 기준 최상위 모델: {best_ap['model']} ({best_ap['average_precision']:.4f})",
        "",
        "## Presentation Interpretation",
        "",
        "기존 정형 요약 feature만 사용한 모델에서는 CatBoost와 LightGBM이 강한 기준선으로 작동했다.",
        "추가 실험에서는 target 이전 최근 주문 sequence를 구성했고, 이를 기존 고객 요약 변수와 결합한 하이브리드 딥러닝 모델을 학습했다.",
        "그 결과 HybridGRU가 F1-score 기준 가장 높은 결과를 보였으며, 이는 재구매 지연 위험 고객을 precision과 recall의 균형 관점에서 탐지하는 목적에 가장 적합한 후보로 해석할 수 있다.",
        "다만 ROC-AUC와 Average Precision에서는 CatBoost, XGBoost, HybridBiGRU가 매우 근접하므로, 최종 발표에서는 지표별 장단점을 함께 설명하는 것이 안전하다.",
        "",
        "## Full Comparison",
        "",
    ]
    lines.extend(markdown_table(presentation_df))
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `final_model_comparison.csv`: 전체 모델 성능 비교표",
            "- `final_model_comparison.png`: 주요 지표 막대 그래프",
            "- `final_selected_roc_curve.png`: 발표용 ROC curve",
            "- `final_selected_pr_curve.png`: 발표용 PR curve",
            "",
            "## Caution",
            "",
            "Accuracy는 음성 class 비율이 높은 불균형 데이터에서 과대평가될 수 있으므로, 최종 해석은 F1-score, Recall, ROC-AUC, Average Precision을 중심으로 진행한다.",
        ]
    )
    (RESULTS_DIR / "final_model_comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    tabular = load_result(
        RESULTS_DIR / "tabular_model_comparison.csv",
        experiment="tabular_aggregate",
        input_type="customer aggregate features",
    )
    sequence = load_result(
        RESULTS_DIR / "deep_sequence_model_comparison.csv",
        experiment="sequence_only",
        input_type="recent order sequence",
    )
    hybrid = load_result(
        RESULTS_DIR / "hybrid_deep_model_comparison.csv",
        experiment="hybrid_sequence_tabular",
        input_type="recent order sequence + customer aggregate features",
    )

    combined = pd.concat([tabular, sequence, hybrid], ignore_index=True, sort=False)
    combined = combined.sort_values(["f1_score", "recall", "average_precision"], ascending=False).reset_index(drop=True)
    combined.insert(0, "rank_by_f1_final", range(1, len(combined) + 1))
    combined.to_csv(RESULTS_DIR / "final_model_comparison.csv", index=False, encoding="utf-8-sig")

    save_metric_plot(combined)
    save_selected_curves()
    save_summary(combined)

    print("Saved final comparison files:")
    for name in [
        "final_model_comparison.csv",
        "final_model_comparison.png",
        "final_selected_roc_curve.png",
        "final_selected_pr_curve.png",
        "final_model_comparison_summary.md",
    ]:
        print(RESULTS_DIR / name)


if __name__ == "__main__":
    main()
