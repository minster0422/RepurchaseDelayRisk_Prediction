# results 폴더 안내

이 폴더는 현재 프로젝트에서 실제로 생성된 실험 결과와, 앞으로 채워야 할 결과 템플릿을 구분해 저장하는 위치다.

## 현재 실제로 존재하는 결과

- `presentation_metric_summary.png`
  - 발표용 모델 성능 요약 그래프. F1-score, Recall, Average Precision을 간결하게 비교
- `presentation_selected_roc_pr.png`
  - 발표용 ROC/PR curve. LogisticRegression, MLP, CatBoost만 선택해 선 겹침을 줄인 버전
- `presentation_threshold_sensitivity.png`
  - q80 기준 15일과 q90 기준 24일 라벨의 F1-score 차이를 보여주는 발표용 그래프
- `presentation_catboost_confusion_matrix.png`
  - CatBoost 기준 단일 confusion matrix 발표용 그래프
- `tabular_model_comparison.csv`
  - DummyClassifier, LogisticRegression, MLP, LightGBM, XGBoost, CatBoost를 같은 split과 threshold tuning 기준으로 비교한 결과
- `tabular_thresholds.json`
  - validation F1-score 기준으로 선택한 모델별 decision threshold
- `tabular_model_comparison.png`
  - 주요 지표 비교 그래프
- `tabular_roc_curve.png`
  - tabular 모델별 ROC curve
- `tabular_pr_curve.png`
  - tabular 모델별 PR curve
- `tabular_confusion_matrices.png`
  - tabular 모델별 confusion matrix 통합 그림
- `tabular_test_predictions.csv`
  - test set의 모델별 예측 확률과 예측 라벨
- `tabular_model_comparison_summary.md`
  - tabular 모델 비교 해석 요약
- `catboost_error_group_summary.csv`
  - CatBoost 기준 TP/FP/FN/TN 그룹별 평균 feature 요약
- `catboost_error_feature_contrast.csv`
  - 오류 유형별 feature 평균 차이와 feature 설명
- `catboost_error_analysis_summary.md`
  - 발표용 오류 분석 해석 요약
- `delay_threshold_sensitivity.csv`
  - q80 기준 15일 라벨과 q90 기준 24일 라벨의 모델 성능 비교
- `delay_threshold_sensitivity_f1.png`
  - 라벨 기준별 F1-score 변화 그래프
- `delay_threshold_sensitivity_summary.md`
  - 라벨 기준 민감도 분석 요약
- `delay_threshold_sensitivity_metadata.json`
  - 민감도 분석 설정값
- `threshold_comparison_mlp.csv`
  - MLP decision threshold 변화에 따른 Precision, Recall, F1-score 비교 결과
- `threshold_comparison_mlp.png`
  - threshold별 Precision / Recall / F1-score 변화 그래프
- `threshold_comparison_summary.md`
  - MLP threshold 비교 해석 요약
- `feature_ablation_results.csv`
  - feature set을 줄였을 때 MLP 성능이 어떻게 변하는지 비교한 결과
- `feature_ablation_f1.png`
  - feature ablation별 F1-score 비교 그래프
- `feature_ablation_summary.md`
  - feature ablation 해석 요약
- `feature_importance_mlp.csv`
  - permutation importance 기반 MLP feature 중요도
- `feature_importance_mlp.png`
  - feature importance 시각화

## 현재 결과에서 말할 수 있는 것

- tabular 모델 비교에서는 CatBoost가 F1-score 기준 가장 높게 나타났다.
- CatBoost의 주요 지표는 Precision `0.4035`, Recall `0.6947`, F1-score `0.5105`, ROC-AUC `0.8226`, Average Precision `0.4795`이다.
- LightGBM은 Recall `0.7157`로 가장 높고, XGBoost는 Average Precision `0.4807`로 가장 높다.
- MLP는 threshold `0.24`에서 Precision `0.4012`, Recall `0.6863`, F1-score `0.5064`, ROC-AUC `0.8179`, Average Precision `0.4663`을 보였다.
- 지표별로 보면 CatBoost는 F1/ROC-AUC, LightGBM은 Recall, XGBoost는 Average Precision에서 강점이 있다.
- feature ablation에서는 전체 feature를 사용한 설정이 F1-score 기준 가장 높았다.
- permutation importance 기준 상위 feature는 `active_span_days`, `recent_3_avg_gap`, `total_orders_before_target`, `std_gap_before_target`, `max_gap_before_target` 순서다.
- CatBoost 오류 분석 기준, TP는 FN보다 `last_gap_before_target`, `recent_3_avg_gap`, `recent_5_avg_gap`, `gap_trend`가 높게 나타났다. 즉 최근 주문 간격이 이미 길어진 고객은 더 잘 탐지되고, 실제 target 주문은 지연됐지만 직전/최근 gap 신호가 약한 고객은 놓칠 가능성이 있다.
- 라벨 민감도 분석에서 q80 기준(`target_gap > 15`)은 양성 비율 `18.65%`, q90 기준(`target_gap > 24`)은 양성 비율 `9.90%`로 확인됐다. q90 기준은 더 엄격하지만 양성 클래스가 줄어 F1-score가 낮아졌다.

## 주의해서 해석할 것

- Tree Boosting 계열과 MLP의 차이는 근소하다. 따라서 특정 모델이 압도적으로 우수하다고 쓰지 않는다.
- `lstm_summary.csv`와 `threshold_comparison_lstm.csv`는 현재 루트 폴더의 `train_lstm.py`에서 생성된 참고 결과지만, 이 LSTM은 실제 주문 sequence가 아니라 집계 feature vector를 pseudo-sequence로 변환한 실험에 가깝다. 따라서 중간발표에서는 완성된 순차 모델 결과처럼 과장하지 않는다.
- 발표에서는 tabular 비교 결과를 중심으로 설명하고, LSTM은 sequence pipeline 보강 후 확장 실험으로 분리한다.
- 오류 분석은 원인 설명의 단서이지 인과관계 증명은 아니다. 그룹 평균 차이로 해석할 수 있는 범위 안에서만 사용한다.
- 전체 ROC/PR curve는 선이 겹치므로 발표 슬라이드에는 `presentation_selected_roc_pr.png`를 우선 사용한다.

## 아직 템플릿 또는 TODO인 결과

- `model_comparison_template.csv`
  - 모델별 최종 평가 지표를 한 표로 모으기 위한 템플릿
- `thresholds_template.json`
  - 모델별 decision threshold 기록 형식을 맞추기 위한 템플릿
- LSTM용 실제 주문 sequence 기반 결과
- 최종 라벨 기준 확정 후 전체 실험 재실행 여부

## 작성 원칙

- test set 기준 숫자만 최종 표에 남긴다.
- accuracy만으로 결론을 내리지 않는다.
- Recall, F1, ROC-AUC, Average Precision을 함께 기록한다.
- 해석 메모는 과장하지 않고, 지표별 장단점을 분리해서 쓴다.
