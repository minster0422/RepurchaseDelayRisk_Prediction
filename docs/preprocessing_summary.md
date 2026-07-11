# 전처리 요약

## 현재 source of truth

현재 저장소의 기준 데이터는 아래 파일로 둔다.

- 가공 데이터 기준: `model_base_hv.csv`
- 보조 확인 파일: `model_base_hv.xlsx`
- 발표용 문제 정의 기준: `README.md`
- 팀 공유용 전체 현황: `docs/project_status_share.md`
- 노트북 역할 기준: `notebooks/01_prepare_data.ipynb`
- 결과 기록 기준: `results/final_model_comparison.csv`

## 현재 확인된 데이터 상태

`model_base_hv.csv` 기준 확인값

- 원본 전체 고객 수: 206,209명
- 원본 전체 주문 수: 3,421,083건
- 고빈도 고객 수: 42,499명
- 샘플 수: 42,499
- 특징 수: 18개 열
- 양성 수: 7,926
- 사용자당 샘플 수: 1행
- 양성 비율: 약 18.65%
- 라벨 규칙: `target_gap > 15 -> delay_risk = 1`
- `target_order_number` 범위: 24 ~ 100
- 분위수: `q80 = 15`, `q90 = 24`, `q95 = 30`

원본 Instacart `orders.csv` 기준으로도 고빈도 고객 threshold 24회가 확인된다.

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

즉, 현재 저장소만으로는 **집계 특징 기반 모델**과 **최근 주문 sequence 기반 모델**을 모두 설명할 수 있다. 최종 발표에서는 고객 요약 feature와 최근 주문 sequence를 결합한 HybridGRU를 딥러닝 최종 후보로 정리했다.

## 발표용으로 고정할 핵심 원칙

1. `feature`는 target 이전 이력만 사용한다.
2. `label`은 target 주문의 `target_gap`으로 만든다.
3. 최종 라벨은 `target_gap > 15`이며, 이 15일은 현재 target gap 분포에서 q80 수준이다.
4. q90 기준인 24일은 더 엄격한 고위험 기준 후보로 두고, q80 기준 15일과 비교하는 민감도 분석 대상으로 남긴다.
5. 불균형 데이터이므로 accuracy보다 Recall, F1, ROC-AUC, AP를 우선한다.

## split 기준

현재 `model_base_hv.csv`는 사용자당 1행이므로 stratified random split을 적용해도 같은 사용자가 train, validation, test에 동시에 들어가지 않는다. 따라서 현재 모델링 단계에서는 user-level split으로 해석할 수 있다.

| split | 비율 | 샘플 수 | 해석 |
| --- | ---: | ---: | --- |
| train | 70% | 29,749 | 모델 학습 |
| validation | 15% | 6,375 | threshold tuning 및 모델 선택 |
| test | 15% | 6,375 | 최종 성능 평가 |

주의할 점은 이 분할이 calendar date 기준 time split은 아니라는 점이다. 이 프로젝트의 핵심 누수 방지는 split 방식보다 **feature와 label의 시점 분리**에 있다. target 주문은 라벨 생성에만 사용하고, feature는 target 이전 이력에서만 만든다.

## 현재 남은 TODO

- GitHub 업로드 전 `requirements.txt`를 실제 사용 패키지 기준으로 정리하기
- `model_base_hv.csv` 공개 여부를 데이터 크기와 라이선스 기준으로 결정하기
- 교수님 피드백을 반영해 auxiliary-loss HybridGRU를 후속 실험으로 추가할지 결정하기
- train / validation / test split index를 별도 파일로 저장할지 결정하기

## 발표자료에 꼭 넣을 숫자와 그래프

- high-frequency threshold = 24
- 운영 기준 delay threshold = 15
- target gap 분위수 = q80 15 / q90 24 / q95 30
- 현재 가공 데이터 기준 positive ratio = 18.65%
- split 비율과 split 기준
- ROC curve
- PR curve
- 최종 모델 비교 표
- HybridGRU confusion matrix
- HybridGRU 오류 분석
- 예상 질문 답변: `presentation/final_presentation_qna_defense.md`
- 팀 공유용 요약: `docs/project_status_share.md`

## 권장 산출물

- `results/final_model_comparison.csv`
- `results/final_model_comparison_summary.md`
- `results/final_selected_roc_curve.png`
- `results/final_selected_pr_curve.png`
- `results/final_hybridgru_confusion_matrix.png`
- `results/hybrid_deep_model_summary.md`
- `results/delay_threshold_sensitivity.csv`
- `results/hybridgru_error_analysis_summary.md`
