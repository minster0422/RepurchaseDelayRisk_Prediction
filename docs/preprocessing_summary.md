# 전처리 summary

## source of truth

현재 저장소의 전처리 기준 파일은 [`model_base_hv.csv`](../model_base_hv.csv)다. README, 노트북, 결과 템플릿은 모두 이 CSV를 기준으로 설명한다.

## 현재 확인된 데이터 상태

- 샘플 수: `42,499`
- 특징 수: `18`
- 양성 수: `7,926`
- 양성 비율: `18.65%`
- 라벨 규칙: `target_gap > 15 -> delay_risk = 1`
- `target_order_number` 범위: `24 ~ 100`

## 해석 가능한 특징

- `avg_gap_before_target`
- `std_gap_before_target`
- `min_gap_before_target`
- `max_gap_before_target`
- `recent_3_avg_gap`
- `recent_5_avg_gap`
- `last_gap_before_target`
- `gap_trend`
- `active_span_days`
- `order_frequency`
- `weekend_order_ratio`
- `dow_variability`

즉, 현재 파일은 **target 이전 이력에서 만든 집계 특징 테이블**로 읽는 것이 자연스럽다.

## 발표용 핵심 원칙

1. `feature`는 target 이전 이력만 사용한다.
2. `label`은 target 주문의 `target_gap`으로 만든다.
3. high-frequency threshold는 `24`로 둔다.
4. delay threshold는 `15`로 둔다.
5. split은 사용자 단위 stratified `train / val / test = 70 / 15 / 15`를 권장한다.
6. `LogisticRegression`, `MLP`, `LSTM`은 모두 validation set에서 threshold tuning을 수행한다.

## 분위수 관련 NOTE

- 발표 기획에서는 `q90 ≈ 15일` 서사를 사용하고 있다.
- 하지만 현재 CSV 재계산 결과는 `q80 = 15`, `q90 = 24`, `q95 = 30`이다.
- 따라서 **15일 기준 자체는 현재 라벨과 일치하지만, q90 근거는 현재 CSV만으로는 재현되지 않는다.**

## 발표 자료에 꼭 들어갈 숫자

- high-frequency threshold = `24`
- delay threshold = `15`
- positive ratio = `18.65%`
- split 방식 = user-level stratified train / val / test
- 최종 모델 선정 기준 = Recall / F1 / ROC-AUC / AP 종합 해석

## 권장 결과 산출물

- `results/model_comparison_template.csv`
- `results/thresholds_template.json`
- `results/test_predictions_dummy.csv`
- `results/test_predictions_logreg.csv`
- `results/test_predictions_mlp.csv`
- `results/test_predictions_lstm.csv`
- `results/roc_curve.png`
- `results/pr_curve.png`
