from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ORDERS = PROJECT_ROOT / "incoming_inspect" / "raw_archive" / "archive" / "orders.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "time_aware"


def safe_std(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def safe_mean(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def make_gap_trend(hist_gaps: np.ndarray) -> float:
    if len(hist_gaps) < 4:
        return 0.0
    recent = hist_gaps[-3:]
    previous = hist_gaps[-6:-3] if len(hist_gaps) >= 6 else hist_gaps[:-3]
    if len(previous) == 0:
        return 0.0
    return float(np.mean(recent) - np.mean(previous))


def assign_lifecycle_split(df: pd.DataFrame) -> tuple[pd.Series, dict[str, float]]:
    """Split by target order-number lifecycle, not by random rows.

    Instacart does not provide an absolute calendar date. The closest stable
    time axis available in this orders-only v1 dataset is each user's
    order_number. We therefore sort samples by target_order_number and assign
    earlier lifecycle targets to train, middle targets to validation, and later
    lifecycle targets to test.
    """
    q70 = float(df["target_order_number"].quantile(0.70))
    q85 = float(df["target_order_number"].quantile(0.85))

    split = np.where(
        df["target_order_number"] <= q70,
        "train",
        np.where(df["target_order_number"] <= q85, "validation", "test"),
    )
    return pd.Series(split, index=df.index), {"train_cut_order_number_q70": q70, "validation_cut_order_number_q85": q85}


def build_dataset(
    orders_path: Path,
    output_dir: Path,
    high_frequency_threshold: int,
    min_history_orders: int,
    delay_threshold_days: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    usecols = [
        "order_id",
        "user_id",
        "eval_set",
        "order_number",
        "order_dow",
        "order_hour_of_day",
        "days_since_prior_order",
    ]
    orders = pd.read_csv(orders_path, usecols=usecols)
    orders = orders.sort_values(["user_id", "order_number"]).reset_index(drop=True)

    total_orders_by_user = orders.groupby("user_id")["order_number"].max()
    high_freq_users = total_orders_by_user[total_orders_by_user >= high_frequency_threshold].index
    hv_orders = orders[orders["user_id"].isin(high_freq_users)].copy()

    records: list[dict[str, float | int | str]] = []
    for user_id, g in hv_orders.groupby("user_id", sort=False):
        g = g.sort_values("order_number")
        order_ids = g["order_id"].to_numpy()
        eval_sets = g["eval_set"].astype(str).to_numpy()
        order_numbers = g["order_number"].to_numpy(dtype=int)
        dows = g["order_dow"].to_numpy(dtype=float)
        hours = g["order_hour_of_day"].to_numpy(dtype=float)
        gaps = g["days_since_prior_order"].fillna(0).to_numpy(dtype=float)
        total_orders = int(order_numbers.max())

        # target index i uses orders before i as feature history and order i gap as label.
        for i in range(min_history_orders, len(g)):
            target_gap = float(gaps[i])
            hist_gaps = gaps[1:i]  # exclude first-order NaN/0 and target gap
            hist_dows = dows[:i]
            hist_hours = hours[:i]

            total_before = int(i)
            active_span = float(np.sum(hist_gaps))
            order_frequency = float(total_before / active_span) if active_span > 0 else 0.0

            recent3 = hist_gaps[-3:] if len(hist_gaps) >= 3 else hist_gaps
            recent5 = hist_gaps[-5:] if len(hist_gaps) >= 5 else hist_gaps
            last_gap = float(hist_gaps[-1]) if len(hist_gaps) else 0.0

            records.append(
                {
                    "user_id": int(user_id),
                    "target_order_id": int(order_ids[i]),
                    "target_eval_set": eval_sets[i],
                    "target_order_number": int(order_numbers[i]),
                    "target_gap": target_gap,
                    "delay_risk": int(target_gap > delay_threshold_days),
                    "total_orders_before_target": total_before,
                    "total_orders": total_orders,
                    "avg_gap_before_target": safe_mean(hist_gaps),
                    "std_gap_before_target": safe_std(hist_gaps),
                    "min_gap_before_target": float(np.min(hist_gaps)) if len(hist_gaps) else 0.0,
                    "max_gap_before_target": float(np.max(hist_gaps)) if len(hist_gaps) else 0.0,
                    "recent_3_avg_gap": safe_mean(recent3),
                    "recent_5_avg_gap": safe_mean(recent5),
                    "last_gap_before_target": last_gap,
                    "gap_trend": make_gap_trend(hist_gaps),
                    "active_span_days": active_span,
                    "order_frequency": order_frequency,
                    "weekend_order_ratio": float(np.mean(np.isin(hist_dows, [0, 6]))) if len(hist_dows) else 0.0,
                    "dow_variability": int(len(np.unique(hist_dows))) if len(hist_dows) else 0,
                    "avg_order_hour": safe_mean(hist_hours),
                    "std_order_hour": safe_std(hist_hours),
                    "night_order_ratio": float(np.mean((hist_hours <= 6) | (hist_hours >= 22))) if len(hist_hours) else 0.0,
                    "morning_order_ratio": float(np.mean((hist_hours >= 6) & (hist_hours < 12))) if len(hist_hours) else 0.0,
                    "days_since_first_order_to_target": active_span + target_gap,
                }
            )

    dataset = pd.DataFrame.from_records(records)
    dataset["split"], split_meta = assign_lifecycle_split(dataset)

    # Keep split ordering stable and easy to read.
    split_order = pd.CategoricalDtype(["train", "validation", "test"], ordered=True)
    dataset["split"] = dataset["split"].astype(split_order)
    dataset = dataset.sort_values(["split", "target_order_number", "user_id", "target_order_id"]).reset_index(drop=True)

    out_csv = output_dir / "time_aware_model_base.csv"
    out_json = output_dir / "time_aware_metadata.json"
    dataset.to_csv(out_csv, index=False, encoding="utf-8-sig")

    split_summary = (
        dataset.groupby("split", observed=True)["delay_risk"]
        .agg(samples="size", positives="sum", positive_ratio="mean")
        .reset_index()
        .to_dict(orient="records")
    )

    metadata = {
        "source": str(orders_path.relative_to(PROJECT_ROOT) if orders_path.is_relative_to(PROJECT_ROOT) else orders_path),
        "high_frequency_definition": "total order count >= 24, equivalent to top 20% threshold in the current project",
        "high_frequency_threshold": high_frequency_threshold,
        "min_history_orders": min_history_orders,
        "first_target_order_number": min_history_orders + 1,
        "delay_threshold_days": delay_threshold_days,
        "split_method": "order-lifecycle time-aware split by target_order_number quantiles",
        "split_note": "Instacart has no absolute calendar date, so target_order_number is used as a pseudo-time axis.",
        **split_meta,
        "raw_total_users": int(orders["user_id"].nunique()),
        "raw_total_orders": int(len(orders)),
        "high_frequency_users": int(len(high_freq_users)),
        "samples": int(len(dataset)),
        "unique_users_in_samples": int(dataset["user_id"].nunique()),
        "positive_count": int(dataset["delay_risk"].sum()),
        "positive_ratio": float(dataset["delay_risk"].mean()),
        "target_gap_quantiles": {
            "q80": float(dataset["target_gap"].quantile(0.80)),
            "q90": float(dataset["target_gap"].quantile(0.90)),
            "q95": float(dataset["target_gap"].quantile(0.95)),
        },
        "split_summary": split_summary,
        "feature_label_separation": "All features are computed from orders before the target order. The label is computed from the target order's target_gap.",
    }
    out_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_md = output_dir / "time_aware_dataset_summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Time-Aware Dataset Summary",
                "",
                "## 목적",
                "",
                "최종발표 전 검토를 위해 원본 `orders.csv`에서 사용자별 주문 순서를 따라 여러 target sample을 다시 만든다.",
                "현재 Instacart 데이터에는 절대 calendar date가 없으므로 `target_order_number`를 고객 생애주기상의 pseudo-time 축으로 사용한다.",
                "",
                "## 핵심 설정",
                "",
                f"- 고빈도 고객 기준: 총 주문 횟수 `{high_frequency_threshold}`회 이상",
                f"- feature 최소 이력: target 이전 `{min_history_orders}`개 주문",
                f"- 첫 target 주문 번호: `{min_history_orders + 1}`",
                f"- 라벨 기준: `target_gap > {delay_threshold_days}`",
                f"- split: `target_order_number` q70/q85 기준 order-lifecycle time-aware split",
                "",
                "## 생성 결과",
                "",
                f"- 샘플 수: `{len(dataset):,}`",
                f"- 사용자 수: `{dataset['user_id'].nunique():,}`",
                f"- 양성 수: `{int(dataset['delay_risk'].sum()):,}`",
                f"- 양성 비율: `{dataset['delay_risk'].mean():.4f}`",
                f"- target gap 분위수: q80 `{dataset['target_gap'].quantile(0.80):.0f}`, q90 `{dataset['target_gap'].quantile(0.90):.0f}`, q95 `{dataset['target_gap'].quantile(0.95):.0f}`",
                "",
                "## split별 요약",
                "",
                "| split | samples | positives | positive_ratio |",
                "| --- | ---: | ---: | ---: |",
                *[
                    f"| {row['split']} | {int(row['samples']):,} | {int(row['positives']):,} | {float(row['positive_ratio']):.4f} |"
                    for row in split_summary
                ],
                "",
                "## 주의",
                "",
                "- 이 split은 사용자 단위 완전 분리가 아니라 같은 사용자의 더 이른 target은 train, 더 늦은 target은 validation/test에 들어갈 수 있는 미래 예측형 split이다.",
                "- 모델 feature에는 `user_id`를 사용하지 않는다.",
                "- LSTM을 제대로 비교하려면 이 데이터의 anchor별 최근 N개 주문 sequence를 별도 tensor로 만드는 추가 작업이 필요하다.",
            ]
        ),
        encoding="utf-8",
    )

    return out_csv, out_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders-path", type=Path, default=DEFAULT_ORDERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--high-frequency-threshold", type=int, default=24)
    parser.add_argument("--min-history-orders", type=int, default=23)
    parser.add_argument("--delay-threshold-days", type=int, default=15)
    args = parser.parse_args()

    csv_path, json_path = build_dataset(
        args.orders_path,
        args.output_dir,
        args.high_frequency_threshold,
        args.min_history_orders,
        args.delay_threshold_days,
    )
    print(f"created_csv={csv_path}")
    print(f"created_metadata={json_path}")


if __name__ == "__main__":
    main()
