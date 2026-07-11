from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "model_base_hv.csv"
PRED_PATH = BASE_DIR / "results" / "tabular_test_predictions.csv"
RESULTS_DIR = BASE_DIR / "results"

MODEL_NAME = "CatBoost"
LABEL_COL = "delay_risk"

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

FEATURE_DESCRIPTIONS = {
    "total_orders_before_target": "target 이전 누적 주문 수. 고객의 활동성 규모를 나타낸다.",
    "avg_gap_before_target": "target 이전 평균 주문 간격. 평소 재구매 주기가 긴 고객인지 확인한다.",
    "std_gap_before_target": "주문 간격의 변동성. 구매 주기가 불안정한 고객을 포착한다.",
    "min_gap_before_target": "가장 짧았던 주문 간격. 매우 짧은 재구매 경험이 있는지 나타낸다.",
    "max_gap_before_target": "가장 길었던 주문 간격. 과거에도 장기 지연이 있었는지 나타낸다.",
    "recent_3_avg_gap": "최근 3회 평균 주문 간격. 단기적인 재구매 둔화를 포착한다.",
    "recent_5_avg_gap": "최근 5회 평균 주문 간격. 최근 주문 리듬을 조금 더 완만하게 본다.",
    "last_gap_before_target": "target 직전 주문 간격. 가장 최근의 주문 지연 신호다.",
    "gap_trend": "최근 주문 간격과 이전 주문 간격의 차이. 주문 주기가 길어지는 추세인지 본다.",
    "active_span_days": "target 이전 누적 활동 기간. 고객이 얼마나 오래 활동했는지 나타낸다.",
    "order_frequency": "활동 기간 대비 주문 빈도. 같은 기간 동안 얼마나 자주 주문했는지 나타낸다.",
    "weekend_order_ratio": "주말 주문 비율. 주문 요일 습관을 요약한다.",
    "dow_variability": "주문 요일 다양성. 특정 요일에 고정된 구매 습관이 있는지 확인한다.",
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {DATA_PATH}")
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"예측 파일이 없습니다: {PRED_PATH}")

    data = pd.read_csv(DATA_PATH)
    preds = pd.read_csv(PRED_PATH)
    preds = preds[preds["model"] == MODEL_NAME].copy()

    if preds.empty:
        raise ValueError(f"{MODEL_NAME} 예측 결과를 찾지 못했습니다.")

    return data, preds


def attach_features(data: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    cols = ["user_id"] + [col for col in FEATURES if col in data.columns]
    merged = preds.merge(data[cols], on="user_id", how="left", validate="many_to_one")

    def error_type(row: pd.Series) -> str:
        y_true = int(row[LABEL_COL])
        y_pred = int(row["y_pred"])
        if y_true == 1 and y_pred == 1:
            return "TP"
        if y_true == 0 and y_pred == 1:
            return "FP"
        if y_true == 1 and y_pred == 0:
            return "FN"
        return "TN"

    merged["error_type"] = merged.apply(error_type, axis=1)
    return merged


def summarize_by_error_type(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [col for col in FEATURES if col in df.columns]
    rows = []

    for error_type, group in df.groupby("error_type"):
        row = {
            "model": MODEL_NAME,
            "error_type": error_type,
            "n_samples": int(len(group)),
            "mean_target_gap": float(group["target_gap"].mean()),
            "mean_y_prob": float(group["y_prob"].mean()),
        }
        for col in feature_cols:
            row[f"mean_{col}"] = float(group[col].mean())
        rows.append(row)

    order = {"TP": 0, "FP": 1, "FN": 2, "TN": 3}
    out = pd.DataFrame(rows)
    out["sort_order"] = out["error_type"].map(order)
    return out.sort_values("sort_order").drop(columns=["sort_order"])


def summarize_feature_contrast(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [col for col in FEATURES if col in df.columns]
    rows = []
    groups = {name: group for name, group in df.groupby("error_type")}

    tp = groups.get("TP")
    fn = groups.get("FN")
    fp = groups.get("FP")
    tn = groups.get("TN")

    for col in feature_cols:
        row = {
            "feature": col,
            "description": FEATURE_DESCRIPTIONS.get(col, ""),
            "tp_mean": float(tp[col].mean()) if tp is not None else np.nan,
            "fn_mean": float(fn[col].mean()) if fn is not None else np.nan,
            "fp_mean": float(fp[col].mean()) if fp is not None else np.nan,
            "tn_mean": float(tn[col].mean()) if tn is not None else np.nan,
        }
        row["tp_minus_fn"] = row["tp_mean"] - row["fn_mean"]
        row["fp_minus_tn"] = row["fp_mean"] - row["tn_mean"]
        rows.append(row)

    out = pd.DataFrame(rows)
    out["abs_tp_minus_fn"] = out["tp_minus_fn"].abs()
    return out.sort_values("abs_tp_minus_fn", ascending=False).drop(columns=["abs_tp_minus_fn"])


def save_summary(error_summary: pd.DataFrame, contrast: pd.DataFrame) -> None:
    counts = error_summary.set_index("error_type")["n_samples"].to_dict()

    def count(name: str) -> int:
        return int(counts.get(name, 0))

    lines = [
        "# CatBoost Error Analysis Summary",
        "",
        "## 목적",
        "F1 기준 1위였던 CatBoost의 TP/FP/FN/TN 그룹을 비교해, 어떤 고객 패턴에서 맞고 틀리는지 해석한다.",
        "",
        "## Confusion Matrix 관점",
        f"- TP: {count('TP'):,}명. 실제 지연 위험 고객을 맞게 탐지한 경우.",
        f"- FP: {count('FP'):,}명. 지연 위험으로 예측했지만 실제로는 15일 이내에 재구매한 경우.",
        f"- FN: {count('FN'):,}명. 실제 지연 위험 고객이지만 모델이 놓친 경우.",
        f"- TN: {count('TN'):,}명. 비지연 고객을 맞게 비위험으로 예측한 경우.",
        "",
        "## feature 해석 메모",
    ]

    for _, row in contrast.head(8).iterrows():
        lines.append(
            f"- `{row['feature']}`: {row['description']} "
            f"TP-FN 평균 차이={row['tp_minus_fn']:.3f}, FP-TN 평균 차이={row['fp_minus_tn']:.3f}"
        )

    lines.extend(
        [
            "",
            "## 발표용 해석",
            "- 놓친 지연 고객(FN)과 잡아낸 지연 고객(TP)의 차이를 보면, 어떤 feature가 위험 신호를 더 분명하게 만드는지 설명할 수 있다.",
            "- FP는 모델이 위험 신호를 보았지만 실제로는 빠르게 재구매한 고객이므로, 쿠폰/리마인드 비용 관점에서 precision trade-off를 설명할 때 사용할 수 있다.",
            "- FN은 실제 지연 고객을 놓친 경우이므로, 위험 고객 탐지 목적에서는 recall을 함께 봐야 하는 이유를 설명한다.",
        ]
    )

    with open(RESULTS_DIR / "catboost_error_analysis_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    data, preds = load_inputs()
    merged = attach_features(data, preds)
    error_summary = summarize_by_error_type(merged)
    contrast = summarize_feature_contrast(merged)

    error_summary.to_csv(
        RESULTS_DIR / "catboost_error_group_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    contrast.to_csv(
        RESULTS_DIR / "catboost_error_feature_contrast.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_summary(error_summary, contrast)

    print("저장 완료:")
    print(RESULTS_DIR / "catboost_error_group_summary.csv")
    print(RESULTS_DIR / "catboost_error_feature_contrast.csv")
    print(RESULTS_DIR / "catboost_error_analysis_summary.md")


if __name__ == "__main__":
    main()
