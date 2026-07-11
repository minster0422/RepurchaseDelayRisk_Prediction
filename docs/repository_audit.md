# 저장소 audit 요약

## 한눈에 보기

현재 작업 폴더는 A의 전처리 산출물(`model_base_hv.csv`)과 이후 보강된 실험 스크립트, 결과 파일, 발표용 문서가 함께 있는 상태다.

- 기준 데이터: `model_base_hv.csv`
- 보조 확인 파일: `model_base_hv.xlsx`
- 실험 스크립트: `threshold_compare.py`, `feature_ablation.py`, `feature_importance.py`, `train_lstm.py`
- 결과 폴더: `results/`
- GitHub 정리본: `repo_clone/`

## 현재 잘된 점

- `model_base_hv.csv`에 고빈도 고객 집계 특징과 `delay_risk` 라벨이 들어 있다.
- `target_gap > 15` 규칙이 현재 가공 데이터에 실제로 반영되어 있다.
- `target_order_number`의 최소값이 24이므로, 고빈도 고객 threshold = 24라는 발표 서사와 크게 어긋나지 않는다.
- 별도 문서에 "고가치 고객 churn"보다 "고빈도 고객의 재구매 지연 위험"이라는 표현이 더 적절하다는 정리가 이미 존재한다.

## 현재 문제점

- 루트 작업 폴더와 GitHub 정리본(`repo_clone`)의 포함 파일이 아직 완전히 같지는 않다.
- 최종 모델 비교표(`final_metrics.csv`)가 아직 없다.
- Tree Boosting 계열 baseline이 아직 없다.
- LSTM 결과는 pseudo-sequence 참고 실험으로, 실제 주문 sequence 기반 실험으로 보기 어렵다.
- 원시 Instacart 전처리 SQL과 BigQuery 실행 환경은 현재 저장소 안에 포함되어 있지 않다.

## 현재 확인된 사실

아래 수치는 현재 저장소에 남아 있는 `model_base_hv.csv`와 원본 Instacart `orders.csv`를 기준으로 재확인한 값이다.

| 항목 | 현재 확인값 |
| --- | --- |
| 전체 고객 수 | 206,209 |
| 전체 주문 수 | 3,421,083 |
| 고빈도 고객 threshold | 24회 |
| 고빈도 고객 수 | 42,499 |
| 행 수 | 42,499 |
| 열 수 | 18 |
| 양성 수 | 7,926 |
| 사용자당 행 수 | 1행 |
| 양성 비율 (`delay_risk = 1`) | 약 18.65% |
| `target_order_number` 최소값 | 24 |
| `target_order_number` 최대값 | 100 |
| 현재 확인된 라벨 규칙 | `target_gap > 15 -> delay_risk = 1` |
| `target_gap` q80 | 15 |
| `target_gap` q90 | 24 |
| `target_gap` q95 | 30 |
| 권장 split | train 29,749 / validation 6,375 / test 6,375 |

## 라벨 기준 정리

기존 발표 기획에는 15일 기준을 더 높은 분위수 기준처럼 설명하려는 문구가 있었지만, 현재 최종 모델링 데이터 기준 `target_gap` 재집계 결과는 **q80 = 15, q90 = 24, q95 = 30**이다.

따라서 발표 기준은 아래처럼 정리한다.

- 현재 실험 라벨은 `target_gap > 15`다.
- 15일은 q90이 아니라 **q80 수준의 조기탐지 기준**이다.
- q90 기준인 24일은 더 엄격한 고위험 지연 기준 후보로 남긴다.
- 중간발표에서는 q80 기준을 사용한 이유와 q90 기준 대안의 장단점을 함께 설명하고 교수님 피드백을 받는다.

이렇게 정리하면 15일 기준을 잘못된 분위수 근거로 단정하지 않으면서도, 현재까지 수행한 `15일` 기준 실험을 조기 위험 탐지 목적의 기준으로 설명할 수 있다.

## split 해석

현재 `model_base_hv.csv`는 사용자당 1행이므로 stratified random split을 사용해도 같은 사용자가 여러 split에 섞이지 않는다. 따라서 현재 모델링 단계의 분할은 user-level split으로 설명할 수 있다.

다만 이 분할은 calendar date 기준 time split은 아니다. 데이터 누수 방지의 핵심은 target 주문을 feature에 넣지 않고, target 이전 주문 이력만 집계한다는 점이다.

## 현재 결과물 상태

- `results/tabular_model_comparison.csv`: DummyClassifier, LogisticRegression, MLP, LightGBM, XGBoost, CatBoost를 같은 split과 같은 threshold tuning 기준으로 비교한 결과가 존재한다.
- `results/tabular_thresholds.json`: validation F1-score 기준 모델별 decision threshold 기록이 존재한다.
- `results/thresholds_template.json`: threshold 기록용 JSON 템플릿이 존재한다.
- `results/tabular_roc_curve.png`: tabular 모델별 ROC curve가 존재한다.
- `results/tabular_pr_curve.png`: tabular 모델별 PR curve가 존재한다.
- `results/tabular_confusion_matrices.png`: tabular 모델별 confusion matrix 통합 그림이 존재한다.
- `results/tabular_test_predictions.csv`: test set 모델별 예측 확률과 예측 라벨이 존재한다.
- `results/catboost_error_group_summary.csv`: CatBoost 기준 TP/FP/FN/TN 그룹별 평균 feature 요약이 존재한다.
- `results/catboost_error_feature_contrast.csv`: 오류 유형별 feature 평균 차이와 feature 설명이 존재한다.
- `results/delay_threshold_sensitivity.csv`: q80 기준 15일과 q90 기준 24일 라벨의 모델 성능 비교 결과가 존재한다.
- `results/delay_threshold_sensitivity_f1.png`: 라벨 기준별 F1-score 변화 그래프가 존재한다.
- `docs/midterm_presentation_qna.md`: 교수님 예상 질문 답변 문서가 존재한다.
- `docs/project_status_share.md`: 팀원이 빠르게 읽을 수 있는 중간발표 현황 공유 문서가 존재한다.
- `docs/lstm_reference_note.md`: 현재 LSTM 참고 실험의 위치와 한계를 정리한 문서가 존재한다.
- `results/threshold_comparison_mlp.csv`: MLP threshold별 Precision, Recall, F1-score 비교 결과가 존재한다.
- `results/feature_ablation_results.csv`: feature set별 MLP 성능 비교 결과가 존재한다.
- `results/feature_importance_mlp.csv`: permutation importance 기반 MLP feature 중요도 결과가 존재한다.
- `results/lstm_summary.csv`: LSTM 참고 결과가 존재하지만, 실제 주문 sequence가 아니라 pseudo-sequence 실험으로 해석해야 한다.
- tabular 비교에서는 CatBoost가 F1-score 기준 근소하게 가장 높고, LightGBM은 Recall, XGBoost는 Average Precision에서 강점을 보인다.

## 다음에 반드시 수정해야 할 항목

- q80 기준과 q90 기준 중 최종 라벨 기준을 교수님 피드백 후 확정하기
- GitHub 정리본에 실제 결과 파일과 스크립트를 반영할지 결정하기

## 있으면 좋은 개선점

- LSTM용 실제 주문 sequence 입력 생성 규칙 복원
- `docs/preprocessing_summary.md`를 전처리 기준 문서로 계속 유지
- `docs/project_status_share.md`를 팀 공유용 최신 요약 문서로 유지
- 최종 라벨 기준 확정 후 전체 실험 재실행 여부 검토
