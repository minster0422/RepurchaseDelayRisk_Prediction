from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =========================
# 기본 설정
# =========================
BASE_DIR = Path(__file__).resolve().parent
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

RANDOM_STATE = 42


def print_section(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n{title}\n{line}")


def load_data() -> tuple[pd.DataFrame, list[str]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if LABEL_COL not in df.columns:
        raise ValueError(f"라벨 컬럼이 없습니다: {LABEL_COL}")

    feature_cols = [col for col in PREFERRED_FEATURES if col in df.columns]
    if not feature_cols:
        excluded = set(META_COLS + [LABEL_COL])
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in excluded]

    if not feature_cols:
        raise ValueError("사용 가능한 feature 컬럼을 찾지 못했습니다.")

    return df, feature_cols


def build_best_mlp() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            batch_size=64,
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_STATE,
        )),
    ])


def main() -> None:
    print_section("1. 데이터 로드")
    df, feature_cols = load_data()
    print(f"데이터 크기: {df.shape}")
    print(f"사용 feature 수: {len(feature_cols)}")
    print("사용 feature:")
    for col in feature_cols:
        print(f"- {col}")

    X = df[feature_cols].copy()
    y = df[LABEL_COL].astype(int).copy()

    # train / val / test = 70 / 15 / 15
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

    print_section("2. 최종 MLP 학습")
    model = build_best_mlp()
    model.fit(X_train, y_train)
    print("MLP 학습 완료")

    print_section("3. Permutation Importance 계산")
    # scoring은 roc_auc 기준으로 계산
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=1,
    )

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)

    print(importance_df.round(6))

    importance_path = RESULTS_DIR / "feature_importance_mlp.csv"
    importance_df.to_csv(importance_path, index=False, encoding="utf-8-sig")

    print_section("4. 중요도 그래프 저장")
    top_n = min(10, len(importance_df))
    top_df = importance_df.head(top_n).sort_values("importance_mean", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(top_df["feature"], top_df["importance_mean"], xerr=top_df["importance_std"])
    plt.xlabel("Permutation Importance (ROC-AUC drop)")
    plt.title("Top Feature Importances - MLP")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance_mlp.png", dpi=150)
    plt.close()

    print(f"저장 완료: {importance_path}")
    print(f"저장 완료: {RESULTS_DIR / 'feature_importance_mlp.png'}")

    print_section("5. 해석 가이드")
    print("상위 feature는 현재 MLP 예측에 가장 큰 영향을 준 변수들입니다.")
    print("특히 recent_* / last_gap_before_target / gap_trend / order_frequency 계열이 높게 나오면")
    print("최근성, 주문 간격 변화, 빈도 관련 정보가 재구매 지연 예측에 중요하다고 해석할 수 있습니다.")


if __name__ == "__main__":
    main()
