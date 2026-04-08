# 전처리 요약

## 현재 source of truth

현재 저장소에는 원시 Instacart 전처리 코드가 없기 때문에, 아래 파일을 임시 source of truth로 둔다.

- 가공 데이터 기준: `model_base_hv.xlsx`
- 발표용 문제 정의 기준: `README.md`
- 노트북 역할 기준: `01_prepare_data.ipynb`
- 결과 기록 기준: `results/model_comparison_template.csv`

## 현재 확인된 데이터 상태

`model_base_hv.xlsx` 기준 확인값

- 샘플 수: 42,499
- 특징 수: 18개 열
- 사용자당 샘플 수: 1행
- 양성 비율: 약 18.65%
- 라벨 규칙: `target_gap > 15 -> delay_risk = 1`
- `target_order_number` 범위: 24 ~ 100
- 분위수: `q80 = 15`, `q90 = 24`, `q95 = 30`

## 현재 해석 가능한 특징

- `avg_gap_before_target`
- `std_gap_before_target`
- `recent_3_avg_gap`
- `recent_5_avg_gap`
- `last_gap_before_target`
- `gap_trend`
- `active_span_days`
- `order_frequency`
- `weekend_order_ratio`
- `dow_variability`

즉, 현재 저장소만으로는 **집계 특징 기반 모델** 설명은 가능하지만, **LSTM용 순차 입력**은 복원되어 있지 않다.

## 발표용으로 고정할 핵심 원칙

1. `feature`는 target 이전 이력만 사용한다.
2. `label`은 target 주문의 `target_gap`으로 만든다.
3. 고객 단위 split으로 데이터 누수를 막는다.
4. 불균형 데이터이므로 accuracy보다 Recall, F1, ROC-AUC, AP를 우선한다.
5. `LogisticRegression`, `MLP`, `LSTM`은 모두 validation set에서 threshold tuning을 수행한다.

## 현재 남은 TODO

- 원시 Instacart 파일 기준으로 high-frequency threshold = 24를 다시 산출하기
- `q90 = 15`와 현재 가공 데이터 사이의 불일치를 해소하기
- LSTM용 순차 입력 생성 규칙을 문서와 코드로 복원하기
- train / val / test split 결과를 파일로 저장하기

## 발표자료에 꼭 넣을 숫자와 그래프

- high-frequency threshold = 24
- 운영 기준 delay threshold = 15
- 현재 가공 데이터 기준 positive ratio = 18.65%
- split 비율과 split 기준
- ROC curve
- PR curve
- 모델 비교 표

## 권장 산출물

- `results/model_comparison_template.csv`
- `results/thresholds.json`
- `results/test_predictions_dummy.csv`
- `results/test_predictions_logreg.csv`
- `results/test_predictions_mlp.csv`
- `results/test_predictions_lstm.csv`
