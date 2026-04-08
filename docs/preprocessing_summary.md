# 전처리 summary

## source of truth

현재 저장소의 전처리 기준 파일은 [`model_base_hv.csv`](../model_base_hv.csv)다. README, 노트북, 결과 템플릿은 모두 이 CSV를 기준으로 설명한다.

## 현재 확인 가능한 상태

- 샘플 수: `42,499`
- 특징 수: `18`
- 양성 수: `7,926`
- 양성 비율: `18.65%`
- 라벨 규칙: `target_gap > 15 -> delay_risk = 1`
- `target_order_number` 범위: `24 ~ 100`
- `total_orders_before_target` 범위: `23 ~ 99`

## feature / label 시점 해석

- `feature`는 target 주문 이전 이력에서 만든 집계 특징으로 해석한다.
- `label`은 target 주문의 `target_gap`으로 생성된 것으로 해석한다.
- 따라서 이 프로젝트는 **현재 상태 분류**가 아니라 **다음 주문 지연 위험 예측** 구조로 보는 것이 적절하다.

## 고빈도 고객 정의

- 전체 고객 중 총 주문 횟수 상위 20%
- 현재 가공 데이터 기준 실제 threshold는 `24회`

현재 CSV에서 `target_order_number` 최소값이 `24`이고 `total_orders_before_target` 최소값이 `23`이므로, 최소 24번째 주문을 target으로 잡은 고빈도 고객 샘플로 이해하는 것이 자연스럽다.

## 분위수 관련 NOTE

- 운영 라벨 기준은 `target_gap > 15`
- 현재 CSV 재집계 결과는 `q80 = 15`, `q90 = 24`, `q95 = 30`
- 따라서 **15일 기준은 현재 데이터에서 q80 수준**이며, 기존 `q90 ≈ 15일` 설명은 재검증이 필요하다.

## 실험 기준

1. split은 사용자 단위 stratified `train / val / test = 70 / 15 / 15`를 권장한다.
2. `LogisticRegression`, `MLP`, `LSTM`은 모두 validation set에서 threshold tuning을 수행한다.
3. accuracy 단독 비교를 피하고 Recall, F1, ROC-AUC, AP를 함께 본다.

## 재실행 후 채워야 할 항목

- 모델별 최종 metric
- 모델별 threshold tuning 결과
- ROC curve / PR curve
- confusion matrix
- sequence pipeline 보강 후 LSTM 입력 생성 결과
