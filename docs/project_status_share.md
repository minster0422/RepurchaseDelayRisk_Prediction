# 중간발표 프로젝트 현황 공유문서

## 1. 현재 프로젝트 한 줄 요약

이 프로젝트는 Instacart 주문 이력을 이용해 **고빈도 고객의 다음 주문 지연 위험**을 예측하는 프로젝트다.  
현재 기준 데이터는 `model_base_hv.csv`이며, 사용자당 1행의 집계 feature와 `delay_risk` 라벨이 들어 있다.

발표 핵심 메시지:

> 고빈도 고객의 target 주문 이전 이력만 feature로 사용하고, target 주문의 `target_gap`으로 다음 주문 지연 위험을 예측한다.

## 2. 작업 역할 정리

### A의 전처리 산출물

- `model_base_hv.csv` 생성
- 고빈도 고객 기준 반영
- target 주문과 이전 이력 기반 feature 구조 제공
- `target_gap > 15` 기반 `delay_risk` 라벨 포함

### B의 초기 아이디어 문서

- 문제를 재구매 지연 위험 예측으로 바라보는 방향 제공
- feature/label 시점 분리, time-aware split, LSTM sequence 아이디어 제공
- 다만 B 문서의 일부 숫자와 구조는 현재 `model_base_hv.csv`와 다르므로 그대로 사용하지 않음

### 현재 보강한 작업

- q80/q90 라벨 기준 충돌 정리
- 데이터 규모와 split 방식 문서화
- Tree Boosting baseline 추가
- ROC/PR curve, confusion matrix, test prediction 저장
- feature importance와 오류 분석 정리
- q80 기준과 q90 기준 sensitivity analysis 추가
- 교수님 예상 질문 답변 문서 작성

## 3. 현재 source of truth

현재 프로젝트에서 우선적으로 믿을 파일은 아래 순서다.

| 구분 | 파일 | 역할 |
| --- | --- | --- |
| 기준 데이터 | `model_base_hv.csv` | 모델링용 최종 집계 데이터 |
| 보조 확인 | `model_base_hv.xlsx` | CSV 확인용 보조 파일 |
| 프로젝트 설명 | `README.md` | 발표용 전체 설명 |
| 전처리 요약 | `docs/preprocessing_summary.md` | threshold, split, 데이터 규모 정리 |
| 결과 안내 | `results/README.md` | 결과 파일별 역할 설명 |
| 예상 질문 | `docs/midterm_presentation_qna.md` | 교수님 질문 대비 답변 |

## 4. 데이터 규모

| 항목 | 값 |
| --- | ---: |
| 전체 고객 수 | 206,209명 |
| 전체 주문 수 | 3,421,083건 |
| 고빈도 고객 기준 | 총 주문 횟수 상위 20% |
| 고빈도 고객 threshold | 24회 |
| 고빈도 고객 수 | 42,499명 |
| 최종 모델링 샘플 수 | 42,499개 |
| 기본 라벨 양성 수 | 7,926개 |
| 기본 라벨 양성 비율 | 18.65% |

## 5. 라벨 기준

현재 최종 모델링 데이터의 `target_gap` 분위수:

| 분위수 | 값 |
| --- | ---: |
| q80 | 15일 |
| q90 | 24일 |
| q95 | 30일 |

현재 기본 라벨:

```text
target_gap > 15 -> delay_risk = 1
target_gap <= 15 -> delay_risk = 0
```

해석:

- 15일 기준은 q90이 아니라 **q80 수준의 조기탐지 기준**이다.
- q90 기준인 24일은 더 엄격한 고위험 기준 후보로 둔다.
- 중간발표에서는 q80 기준을 기본 실험으로 사용하고, q90 기준은 sensitivity analysis와 교수님 피드백용 대안으로 제시한다.

## 6. Split 방식

현재 `model_base_hv.csv`는 사용자당 1행이다. 따라서 stratified random split을 사용해도 같은 고객이 train, validation, test에 동시에 들어가지 않는다.

| split | 비율 | 샘플 수 |
| --- | ---: | ---: |
| train | 70% | 29,749 |
| validation | 15% | 6,375 |
| test | 15% | 6,375 |

주의:

- 현재 split은 calendar date 기준 time split은 아니다.
- 다만 target 주문은 label 생성에만 사용하고, feature는 target 이전 이력만 사용하므로 feature/label 시점 분리로 누수를 줄였다.
- 더 엄밀한 time-aware split은 고객별 여러 target 시점을 새로 만드는 추가 전처리가 필요하다.

## 7. 주요 feature와 의미

| feature | 발표용 의미 |
| --- | --- |
| `active_span_days` | target 이전 누적 활동 기간 |
| `recent_3_avg_gap` | 최근 3회 평균 주문 간격 |
| `recent_5_avg_gap` | 최근 5회 평균 주문 간격 |
| `last_gap_before_target` | target 직전 주문 간격 |
| `gap_trend` | 최근 주문 간격이 길어지는 추세 |
| `total_orders_before_target` | target 이전 누적 주문 수 |
| `avg_gap_before_target` | 평소 평균 주문 간격 |
| `std_gap_before_target` | 주문 간격 변동성 |
| `weekend_order_ratio` | 주말 주문 비율 |
| `dow_variability` | 주문 요일 다양성 |

MLP permutation importance 기준 상위 feature:

1. `active_span_days`
2. `recent_3_avg_gap`
3. `total_orders_before_target`
4. `std_gap_before_target`
5. `max_gap_before_target`

## 8. 모델 비교 결과

같은 split과 validation F1-score 기준 threshold tuning을 적용해 비교했다.

| 지표 관점 | 1위 모델 | 값 |
| --- | --- | ---: |
| F1-score | CatBoost | 0.5105 |
| Recall | LightGBM | 0.7157 |
| ROC-AUC | CatBoost | 0.8226 |
| Average Precision | XGBoost | 0.4807 |

주요 모델 결과:

| 모델 | Threshold | Precision | Recall | F1 | ROC-AUC | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CatBoost | 0.58 | 0.4035 | 0.6947 | 0.5105 | 0.8226 | 0.4795 |
| LightGBM | 0.55 | 0.3958 | 0.7157 | 0.5097 | 0.8181 | 0.4649 |
| XGBoost | 0.58 | 0.3990 | 0.6997 | 0.5082 | 0.8209 | 0.4807 |
| MLP | 0.24 | 0.4012 | 0.6863 | 0.5064 | 0.8179 | 0.4663 |
| LogisticRegression | 0.52 | 0.3410 | 0.6703 | 0.4521 | 0.7578 | 0.4090 |
| DummyClassifier | 0.50 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1865 |

발표 해석:

- Tree Boosting 계열이 MLP보다 근소하게 안정적인 성능을 보였다.
- CatBoost가 F1과 ROC-AUC 기준에서 균형적이었다.
- LightGBM은 Recall이 가장 높아 위험 고객을 넓게 잡는 관점에서 강점이 있다.
- XGBoost는 Average Precision 기준에서 강점이 있다.
- MLP는 Tree Boosting보다 약간 낮지만 큰 차이는 아니므로 집계 feature 기반 신경망 대안으로 볼 수 있다.

## 9. 라벨 기준 sensitivity analysis

| 기준 | 라벨 | 양성 수 | 양성 비율 | F1 best model | F1 |
| --- | --- | ---: | ---: | --- | ---: |
| q80 조기탐지 | `target_gap > 15` | 7,926 | 18.65% | CatBoost | 0.5105 |
| q90 고위험 | `target_gap > 24` | 4,207 | 9.90% | LightGBM | 0.4140 |

해석:

- q80 기준은 양성 샘플이 더 많아 조기 위험 탐지와 학습 안정성에 유리하다.
- q90 기준은 더 엄격한 고위험 기준이지만 양성 클래스가 줄어 F1-score가 낮아졌다.
- 중간발표에서는 q80 기준을 기본으로 사용하고, q90 기준은 최종 라벨 후보로 교수님 피드백을 받는다.

## 10. 오류 분석 요약

CatBoost 기준 confusion matrix:

| 유형 | 샘플 수 | 의미 |
| --- | ---: | --- |
| TP | 826 | 실제 지연 위험 고객을 맞게 탐지 |
| FP | 1,221 | 위험으로 예측했지만 실제로는 15일 이내 재구매 |
| FN | 363 | 실제 지연 위험 고객을 놓침 |
| TN | 3,965 | 비지연 고객을 맞게 비위험으로 예측 |

오류 분석에서 확인한 점:

- TP는 FN보다 `last_gap_before_target`, `recent_3_avg_gap`, `recent_5_avg_gap`, `gap_trend`가 높게 나타났다.
- 즉 최근 주문 간격이 이미 길어진 고객은 모델이 비교적 잘 탐지했다.
- 반대로 실제 target 주문은 지연됐지만 직전/최근 gap 신호가 약한 고객은 놓칠 수 있다.
- 이 분석은 인과관계 증명이 아니라, 모델이 어떤 feature 신호에 반응했는지 보는 해석 자료다.

## 11. LSTM 현재 위치

LSTM은 원래 순차 패턴 기반 모델로 비교하고 싶었던 후보지만, 현재 `model_base_hv.csv`는 사용자당 1행의 집계 feature다. 따라서 현재 루트의 LSTM 결과는 실제 주문 sequence 기반 결과라기보다 pseudo-sequence 참고 실험에 가깝다.

관련 파일:

- `train_lstm.py`
- `results/lstm_summary.csv`
- `results/threshold_comparison_lstm.csv`
- `results/lstm_model.pt`
- `docs/lstm_reference_note.md`

현재 참고 성능:

| 지표 | 값 |
| --- | ---: |
| ROC-AUC | 0.8146 |
| Average Precision | 0.4597 |
| best threshold | 0.60 |
| Precision | 0.4036 |
| Recall | 0.6776 |
| F1-score | 0.5059 |

단, 이 결과는 집계 feature를 `seq_len=4` pseudo-sequence로 바꾼 참고 실험이며, tabular 모델 비교와 같은 70/15/15 test split 기준 결과가 아니다. 따라서 최종 모델 비교표에는 직접 섞지 않고, 순차 모델 확장 가능성을 보여주는 보조 자료로만 사용한다.

중간발표에서는 다음처럼 말하는 것이 안전하다.

> 현재 중간발표는 tabular feature 기반 모델 비교를 중심으로 정리했고, LSTM은 실제 주문 sequence pipeline을 보강한 뒤 최종발표 확장 실험으로 다룰 계획이다.

## 12. 발표에 넣을 그래프

슬라이드에는 선이 겹치는 전체 ROC/PR 그래프보다 발표용으로 정리한 `presentation_*` 그래프를 우선 사용하는 것이 좋다. 자세한 기준은 `docs/presentation_figure_guide.md`에 정리했다.

우선순위가 높은 발표용 그래프:

1. `results/presentation_metric_summary.png`
2. `results/presentation_selected_roc_pr.png`
3. `results/presentation_threshold_sensitivity.png`
4. `results/presentation_catboost_confusion_matrix.png`
5. `results/feature_importance_mlp.png`

보조자료로 남길 그래프:

- `results/tabular_roc_curve.png`
- `results/tabular_pr_curve.png`
- `results/tabular_confusion_matrices.png`
- `results/delay_threshold_sensitivity_f1.png`

## 13. 교수님께 확인할 질문

발표 마지막 또는 질의응답에서 아래를 질문하면 좋다.

1. 현재 q80 기준인 `target_gap > 15`를 조기탐지 라벨로 유지해도 적절한지
2. q90 기준인 `target_gap > 24`를 최종 고위험 라벨로 바꾸는 것이 더 타당한지
3. 현재 중간발표에서는 tabular 비교 중심으로 정리하고, LSTM sequence pipeline은 최종발표 확장으로 두는 전략이 괜찮은지
4. 최종발표에서 time-aware split을 위해 고객별 여러 target 시점 샘플을 새로 만드는 것이 필요한지

## 14. 남은 TODO

중간발표 전:

- 발표 슬라이드에 핵심 숫자와 그래프 반영
- Q&A 문서에서 발표자가 맡을 답변 체크
- 결과표에서 과장 표현 제거

최종발표 전:

- q80/q90 라벨 기준 최종 확정
- LSTM용 실제 주문 sequence pipeline 복원 또는 제외 결정
- time-aware split 가능성 검토
- 최종 라벨 기준으로 전체 실험 재실행
- 최종 보고서용 ROC/PR curve와 confusion matrix 정리

## 15. 실행 명령

```bash
pip install -r requirements.txt
python train_tabular_baselines.py
python analyze_tabular_errors.py
python sensitivity_delay_threshold.py
```

각 명령은 `results/` 폴더에 결과 파일을 저장한다.
