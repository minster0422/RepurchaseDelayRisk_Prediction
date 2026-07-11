from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_BASE_PATH = PROJECT_ROOT / "model_base_hv.csv"
PRED_PATH = RESULTS_DIR / "hybrid_deep_test_predictions.csv"

MODEL_NAME = "HybridGRU"
KEY_FEATURES = [
    "target_gap",
    "total_orders_before_target",
    "avg_gap_before_target",
    "recent_3_avg_gap",
    "recent_5_avg_gap",
    "last_gap_before_target",
    "gap_trend",
    "active_span_days",
    "order_frequency",
    "weekend_order_ratio",
    "dow_variability",
]


def assign_group(row: pd.Series) -> str:
    if row["delay_risk"] == 1 and row["y_pred"] == 1:
        return "TP"
    if row["delay_risk"] == 0 and row["y_pred"] == 1:
        return "FP"
    if row["delay_risk"] == 1 and row["y_pred"] == 0:
        return "FN"
    return "TN"


def main() -> None:
    if not PRED_PATH.exists():
        raise FileNotFoundError(PRED_PATH)
    if not MODEL_BASE_PATH.exists():
        raise FileNotFoundError(MODEL_BASE_PATH)

    preds = pd.read_csv(PRED_PATH)
    preds = preds[preds["model"] == MODEL_NAME].copy()
    base = pd.read_csv(MODEL_BASE_PATH)

    joined = preds.merge(
        base,
        on=["user_id", "target_order_id", "target_order_number", "target_gap", "delay_risk"],
        how="left",
    )
    joined["error_group"] = joined.apply(assign_group, axis=1)

    group_summary = (
        joined.groupby("error_group")
        .agg(
            samples=("user_id", "count"),
            avg_prob=("y_prob", "mean"),
            positive_rate=("delay_risk", "mean"),
            avg_target_gap=("target_gap", "mean"),
            median_target_gap=("target_gap", "median"),
        )
        .reset_index()
        .sort_values("error_group")
    )
    group_summary.to_csv(RESULTS_DIR / "hybridgru_error_group_summary.csv", index=False, encoding="utf-8-sig")

    feature_contrast = (
        joined.groupby("error_group")[KEY_FEATURES]
        .mean()
        .reset_index()
        .sort_values("error_group")
    )
    feature_contrast.to_csv(RESULTS_DIR / "hybridgru_error_feature_contrast.csv", index=False, encoding="utf-8-sig")

    y_true = joined["delay_risk"].astype(int)
    y_pred = joined["y_pred"].astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5.4, 4.3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["Actual 0", "Actual 1"],
        annot_kws={"fontsize": 14, "weight": "bold"},
    )
    plt.title("HybridGRU Confusion Matrix")
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "final_hybridgru_confusion_matrix.png", dpi=150)
    plt.close()

    fp = group_summary[group_summary["error_group"] == "FP"].iloc[0]
    fn = group_summary[group_summary["error_group"] == "FN"].iloc[0]
    tp = group_summary[group_summary["error_group"] == "TP"].iloc[0]
    tn = group_summary[group_summary["error_group"] == "TN"].iloc[0]

    fn_features = feature_contrast[feature_contrast["error_group"] == "FN"].iloc[0]
    fp_features = feature_contrast[feature_contrast["error_group"] == "FP"].iloc[0]

    lines = [
        "# HybridGRU Error Analysis",
        "",
        "이 문서는 최종 후보 모델인 HybridGRU의 test set 예측 오류를 요약한다.",
        "",
        "## Confusion Matrix",
        "",
        f"- TN: {int(tn['samples']):,}",
        f"- FP: {int(fp['samples']):,}",
        f"- FN: {int(fn['samples']):,}",
        f"- TP: {int(tp['samples']):,}",
        "",
        "## Error Pattern",
        "",
        f"- FP는 실제로는 지연 위험이 아니지만 모델이 위험으로 판단한 경우이며, 평균 예측 확률은 {fp['avg_prob']:.3f}이다.",
        f"- FN은 실제로는 지연 위험이지만 모델이 놓친 경우이며, 평균 target_gap은 {fn['avg_target_gap']:.2f}일이다.",
        f"- FN 그룹의 평균 최근 3회 주문 간격은 {fn_features['recent_3_avg_gap']:.2f}일, 마지막 주문 간격은 {fn_features['last_gap_before_target']:.2f}일이다.",
        f"- FP 그룹의 평균 최근 3회 주문 간격은 {fp_features['recent_3_avg_gap']:.2f}일, 마지막 주문 간격은 {fp_features['last_gap_before_target']:.2f}일이다.",
        "",
        "## Presentation Note",
        "",
        "오류 분석은 모델이 모든 지연 고객을 완벽히 잡는다는 주장이 아니라, 어떤 고객을 위험으로 과대 탐지하거나 놓치는지 확인하기 위한 보조 해석이다.",
        "최종 발표에서는 FP/FN 숫자와 함께, 지연 위험 탐지 문제에서는 FN을 줄이는 것이 중요하지만 FP가 너무 많아지면 마케팅 비용이 증가할 수 있다는 trade-off를 설명하면 된다.",
    ]
    (RESULTS_DIR / "hybridgru_error_analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("Saved HybridGRU error analysis files:")
    for name in [
        "hybridgru_error_group_summary.csv",
        "hybridgru_error_feature_contrast.csv",
        "final_hybridgru_confusion_matrix.png",
        "hybridgru_error_analysis_summary.md",
    ]:
        print(RESULTS_DIR / name)


if __name__ == "__main__":
    main()
