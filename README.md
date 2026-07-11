# Repurchase Delay Risk Prediction

## 1. 프로젝트 개요

이 프로젝트는 Instacart 주문 이력을 바탕으로, **고빈도 고객의 다음 주문이 평소보다 오래 지연될 위험**을 예측하는 것을 목표로 한다. 발표 서사는 단순 분류 문제가 아니라, **과거 주문 이력으로 다음 주문의 지연 위험을 미리 탐지하는 예측 프로젝트**로 구성한다.

현재 저장소의 기준 데이터는 고빈도 고객 집계 특징과 라벨이 담긴 [`model_base_hv.csv`](./model_base_hv.csv)다. [`model_base_hv.xlsx`](./model_base_hv.xlsx)는 같은 데이터를 확인하기 위한 보조 파일로 둔다.

`최종 발표 이후 정리 메모`

- 최종 발표에서는 tabular baseline, sequence-only deep learning, hybrid deep learning을 함께 비교했다.
- 최종 딥러닝 후보는 `HybridGRU`로 정리하되, CatBoost와의 차이가 크지 않다는 점을 함께 명시한다.
- 발표 후 모델 설계 피드백에 따라, 향후에는 branch별 auxiliary loss를 추가한 HybridGRU를 개선 실험으로 둔다.
- GitHub 업로드 전 정리 기준은 [`docs/github_upload_checklist.md`](./docs/github_upload_checklist.md)에 둔다.
- 최종 발표 기준 요약은 [`docs/final_project_summary.md`](./docs/final_project_summary.md), 모델 설계 피드백은 [`docs/model_design_feedback.md`](./docs/model_design_feedback.md)에 정리했다.

## 2. 문제 정의

핵심 문제는 다음과 같다.

> 과거 주문 이력만을 사용해, 고빈도 고객의 **다음 주문이 지연 위험 상태에 들어갈지** 예측할 수 있는가?

발표에서는 아래 두 가지를 분명히 보여주는 것이 중요하다.

- `feature`와 `label`의 시점을 분리한다.
- `target` 주문은 라벨 생성용으로만 사용하고, 입력 특징은 반드시 **target 주문 이전 이력**에서 만든다.

즉, 이 프로젝트의 구조는 "현재 주문을 설명하는 분류"가 아니라, **다음 주문의 지연 위험을 예측하는 구조**여야 한다.

## 3. 왜 Instacart 데이터인가

Instacart는 반복 구매가 많은 식료품 주문 데이터이기 때문에, 재구매 간격과 고객별 주문 습관을 비교적 자연스럽게 관찰할 수 있다. 특히 `days_since_prior_order`를 통해 다음 주문까지의 간격을 정의할 수 있어, 재구매 지연 위험 예측 문제를 설계하기에 적합하다.

다만 Instacart에는 직접적인 구매 금액 정보가 없기 때문에, 본 프로젝트는 "고가치 고객"보다 **고빈도 고객**이라는 표현이 더 정직하다.

## 4. 고빈도 고객 정의

발표용 기준 정의는 다음과 같다.

- 전체 고객 중 **총 주문 횟수 상위 20%**를 고빈도 고객으로 본다.
- 최종 모델링 데이터 기준 이 경계는 **주문 횟수 24회**다.

현재 저장소에 남아 있는 [`model_base_hv.csv`](./model_base_hv.csv)를 보면 `target_order_number`의 최소값이 24이므로, **최종 모델링 데이터에는 threshold = 24**가 반영되어 있다.

## 5. 타깃 정의

현재 프로젝트의 타깃은 절대 기준형 이진 분류다.

- `target_gap > 15` 이면 `delay_risk = 1`
- 그렇지 않으면 `delay_risk = 0`

즉, 특정 target 주문의 실제 재구매 간격이 15일을 초과하면 지연 위험군으로 본다.

`NOTE`

- 현재 최종 모델링 데이터에서 고빈도 고객의 `target_gap` 분위수는 **q80 = 15, q90 = 24, q95 = 30**이다.
- 따라서 `15일` 기준은 q90 기준이 아니라, **q80 수준의 조기탐지 기준**으로 해석한다.
- q90 기준인 `24일`은 더 엄격한 고위험 지연 기준이 될 수 있으므로, 최종 정리에서는 `15일(q80)` 조기 탐지 기준과 `24일(q90)` 고위험 기준을 구분해 설명한다.

발표 자료에서는 `target_gap > 15`를 "위험 고객을 너무 좁게 잡기보다 조기 탐지를 우선한 운영 라벨 기준"으로 설명하는 편이 가장 안전하다.

## 6. 데이터 전처리 개요

전처리의 핵심 원칙은 아래와 같다.

- 고객별로 하나의 `target` 주문을 잡는다.
- `label`은 그 target 주문의 `target_gap`으로 만든다.
- `feature`는 target 주문 이전 이력만 사용해 만든다.
- 데이터 누수를 막기 위해, target 이후 정보는 어떤 형태로도 입력에 포함하지 않는다.

현재 포함된 가공 데이터 [`model_base_hv.csv`](./model_base_hv.csv)는 다음과 같은 **집계 특징 기반 표 형태**다.

- 원본 전체 고객 수: 206,209명
- 원본 전체 주문 수: 3,421,083건
- 고빈도 고객 수: 42,499명
- 샘플 수: 42,499
- 양성 수: 7,926
- 현재 확인된 양성 비율: 약 18.65%
- 현재 확인된 구조: 사용자당 1행
- 주요 열: `avg_gap_before_target`, `std_gap_before_target`, `recent_3_avg_gap`, `recent_5_avg_gap`, `gap_trend`, `order_frequency`, `weekend_order_ratio`

즉, 현재 저장소 기준으로는 **DummyClassifier / LogisticRegression / MLP / Tree Boosting 계열 모델을 위한 집계형 입력**과 **LSTM/GRU/TCN/Transformer 계열 순차 모델 및 하이브리드 모델을 위한 sequence 입력 생성 코드**가 모두 존재한다. 다만 GitHub 업로드 시에는 최종 발표에 사용한 코드와 결과만 선별해 정리하는 것이 좋다.

## 7. 모델 구성

최종 발표 기준 비교 모델은 세 그룹으로 정리한다.

- `Tabular baseline`
  - `DummyClassifier`, `LogisticRegression`, `MLP`, `CatBoost`, `LightGBM`, `XGBoost`
  - 고객 요약 feature만으로 재구매 지연 위험을 어느 정도 예측할 수 있는지 확인한다.
- `Sequence-only deep learning`
  - `LSTM`, `GRU`, `BiGRU`, `TCN`, `TransformerEncoder`
  - 최근 주문 sequence만으로 지연 위험을 예측할 수 있는지 확인한다.
- `Hybrid deep learning`
  - `HybridLSTM`, `HybridGRU`, `HybridBiGRU`, `HybridTCN`, `HybridTransformer`
  - 고객 요약 feature와 최근 주문 sequence를 함께 사용했을 때 sequence-only 모델보다 개선되는지 확인한다.

발표의 핵심 비교 질문은 다음과 같다.

> **고객의 장기 구매 성향과 최근 주문 흐름을 함께 보면, sequence-only 모델보다 더 균형적인 지연 위험 예측이 가능한가?**

최종 발표에서는 `HybridGRU`를 딥러닝 최종 후보로 정리했다. 다만 이는 모든 지표에서 압도적으로 우수한 모델이라는 뜻이 아니라, 문제 구조에 가장 잘 맞고 F1-score 기준 가장 균형적인 결과를 보인 후보라는 의미다.

## 8. 실험 설정

현재 저장소의 모델링 테이블은 사용자당 1행이므로, `train / val / test` 분할은 곧 사용자 단위 분할로 해석할 수 있다. 권장 재현 구조는 아래 기준으로 정리한다.

- split 단위: 사용자 단위. 현재 데이터가 사용자당 1행이기 때문에 같은 사용자가 여러 split에 동시에 들어가지 않는다.
- split 방식: `train / val / test = 70 / 15 / 15`
- 분할 원칙: `delay_risk` 비율을 유지하는 stratified random split
- 예상 샘플 수: train 29,749 / validation 6,375 / test 6,375
- 주의점: calendar date 기준 time split은 아니다. 대신 target 주문을 라벨로만 사용하고, feature는 target 이전 이력만 사용해 시점 누수를 줄인다.
- threshold tuning:
  - `DummyClassifier`는 고정 기준선으로 둔다.
  - `LogisticRegression`, `MLP`, `LightGBM`, `XGBoost`, `CatBoost`는 **동일하게 validation set에서 F1-score 기준 threshold tuning**을 수행한다.

평가 지표는 accuracy보다 아래 지표를 우선한다.

- `Recall`: 위험 고객을 놓치지 않는 정도
- `F1-score`: precision과 recall의 균형
- `ROC-AUC`: threshold에 덜 의존하는 분리 성능
- `Average Precision` 또는 `PR curve`: 양성 클래스 탐지 품질

정리하면, 이 프로젝트는 불균형 데이터이므로 **accuracy 단독 비교를 피하고 Recall / F1 / ROC-AUC / AP를 중심으로 해석**해야 한다.

## 9. 결과 요약

현재 루트 [`results/`](./results/) 폴더에는 tabular baseline, sequence-only deep learning, hybrid deep learning 결과가 함께 정리되어 있다. 최종 비교 기준 파일은 [`results/final_model_comparison.csv`](./results/final_model_comparison.csv)다.

핵심 결과는 다음과 같다.

| model | precision | recall | f1_score | roc_auc | average_precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| HybridGRU | 0.4116 | 0.6972 | 0.5176 | 0.8220 | 0.4737 |
| CatBoost | 0.4035 | 0.6947 | 0.5105 | 0.8226 | 0.4795 |
| LightGBM | 0.3958 | 0.7157 | 0.5097 | 0.8181 | 0.4649 |
| XGBoost | 0.3990 | 0.6997 | 0.5082 | 0.8209 | 0.4807 |
| MLP | 0.4012 | 0.6863 | 0.5064 | 0.8179 | 0.4663 |
| LSTM | 0.3626 | 0.6173 | 0.4569 | 0.7604 | 0.4179 |

지표별 강점은 다르게 나타난다.

- `HybridGRU`: F1-score 기준 가장 높은 균형 성능
- `LightGBM`: Recall 기준 가장 많은 지연 고객 탐지
- `XGBoost`: Average Precision 기준 positive class ranking 강점
- `CatBoost`: ROC-AUC와 F1에서 매우 경쟁력 있는 tabular baseline
- `LSTM`: sequence-only 모델 중 기준점 역할

test set에서 실제 지연 고객은 1,189명이었고, HybridGRU는 그중 829명을 탐지했다. 따라서 Recall은 `829 / 1,189 = 0.6972`다. Precision은 0.4116으로 단독으로 보면 낮아 보일 수 있지만, 전체 지연 고객 비율이 18.65%라는 점을 함께 보면 모델이 위험하다고 고른 고객군에는 실제 지연 고객이 평균보다 약 2.2배 더 많이 포함되어 있다.

따라서 이 결과는 완성된 운영 모델이라기보다, **고빈도 고객의 주문 이력만으로 재구매 지연 위험 고객군을 우선순위화할 가능성**을 보인 결과로 해석한다.

## 10. 최종 결론

이 프로젝트는 Instacart 주문 이력에서 고빈도 고객을 정의하고, target 이전 이력만으로 다음 주문의 지연 위험을 예측하는 구조를 만들었다. 집계형 tabular baseline, sequence-only 딥러닝 모델, hybrid 딥러닝 모델을 비교한 결과, HybridGRU는 고객 요약 feature와 최근 주문 sequence를 함께 사용해 sequence-only 모델보다 개선된 결과를 보였고, 같은 split에서 F1-score 기준 가장 균형적인 성능을 보였다.

다만 CatBoost와의 차이는 크지 않으므로 `HybridGRU가 압도적으로 우수하다`고 해석하지 않는다. 더 안전한 결론은 다음과 같다.

> HybridGRU는 이번 문제의 핵심 가설인 “고객의 장기 구매 성향과 최근 주문 흐름을 함께 본다”는 구조를 가장 직접적으로 구현했고, 같은 split에서 F1-score 기준 가장 균형적인 결과를 보여 딥러닝 최종 후보로 선정했다.

활용 측면에서는 모든 고빈도 고객에게 동일하게 쿠폰이나 알림을 보내기보다, 위험 score가 높은 고객부터 먼저 확인하고 고객 유지 전략을 적용하는 방식으로 사용할 수 있다.

## 11. 한계와 향후 개선 방향

현재 프로젝트의 한계와 개선 방향은 아래처럼 정리한다.

- 현재 split은 stratified random split이다. 사용자 중복 누수는 줄였지만, 실제 서비스 적용을 위해서는 시간 기준 검증이 더 엄격하다.
- `target_gap > 15`는 q80 기반의 조기 탐지 기준이다. q90인 24일 기준은 더 보수적인 고위험 기준으로 별도 민감도 분석 대상이다.
- 현재 HybridGRU는 GRU branch와 MLP branch를 결합한 뒤 최종 출력 하나에 대해 loss를 계산하는 single-loss late fusion 구조다.
- 교수님 피드백을 반영하면, GRU branch와 MLP branch 각각에 보조 prediction head와 auxiliary loss를 추가하는 구조가 더 엄밀하다.
- Instacart에는 상품 정보도 있으나, 현재 모델은 주문 간격과 주문 순서 정보에 집중했다. 이후 basket size, product diversity, reorder ratio 같은 상품 기반 feature를 추가할 수 있다.

모델 설계 피드백과 auxiliary-loss 개선안은 [`docs/model_design_feedback.md`](./docs/model_design_feedback.md)에 따로 정리했다.

## 12. 실행 방법

현재 저장소는 발표용 산출물과 실험 코드가 함께 들어 있는 작업 폴더 상태다. GitHub 업로드 전에는 [`docs/github_upload_checklist.md`](./docs/github_upload_checklist.md)를 기준으로 파일을 선별하는 것이 좋다.

권장 실행 흐름은 아래와 같다.

1. [`notebooks/01_prepare_data.ipynb`](./notebooks/01_prepare_data.ipynb)에서 문제 정의, 시점 분리, split 규칙을 확인한다.
2. `python train_tabular_baselines.py`로 tabular baseline 결과를 생성한다.
3. `python train_deep_sequence_models.py`로 sequence-only deep learning 결과를 생성한다.
4. `python train_hybrid_deep_models.py`로 hybrid deep learning 결과를 생성한다.
5. `python summarize_final_model_results.py`로 최종 비교 파일을 생성한다.
6. `python sensitivity_delay_threshold.py`로 q80 기준 15일과 q90 기준 24일 라벨 민감도 분석을 생성한다.
7. `python analyze_hybridgru_errors.py`로 HybridGRU 오류 분석을 생성한다.

## 13. 파일 구조

```text
.
|-- README.md
|-- requirements.txt
|-- model_base_hv.csv
|-- model_base_hv.xlsx
|-- train_tabular_baselines.py
|-- train_deep_sequence_models.py
|-- train_hybrid_deep_models.py
|-- summarize_final_model_results.py
|-- analyze_hybridgru_errors.py
|-- sensitivity_delay_threshold.py
|-- feature_ablation.py
|-- feature_importance.py
|-- notebooks/
|   |-- 01_prepare_data.ipynb
|   |-- 02_train_mlp.ipynb
|   |-- 03_train_lstm.ipynb
|   `-- 04_compare_models.ipynb
|-- presentation/
|   |-- final_repurchase_delay_presentation_v14.pptx
|   |-- 딥러닝_프로젝트_발표자료_이민성팀.pdf
|   `-- final_presentation_speaker_script_v14.docx
|-- make_presentation_figures.py
|-- docs/
|   |-- final_project_summary.md
|   |-- github_upload_checklist.md
|   |-- model_design_feedback.md
|   |-- repository_audit.md
|   |-- preprocessing_summary.md
|   |-- midterm_presentation_qna.md
|   |-- lstm_reference_note.md
|   |-- presentation_figure_guide.md
|   `-- project_status_share.md
`-- results/
    |-- README.md
    |-- final_model_comparison.csv
    |-- final_model_comparison_summary.md
    |-- final_model_comparison.png
    |-- final_selected_roc_curve.png
    |-- final_selected_pr_curve.png
    |-- final_hybridgru_confusion_matrix.png
    |-- hybrid_deep_model_comparison.csv
    |-- hybrid_deep_model_summary.md
    |-- deep_sequence_model_comparison.csv
    |-- deep_sequence_model_summary.md
    |-- delay_threshold_sensitivity.csv
    |-- q90_hybridgru_sensitivity_summary.md
    |-- hybridgru_error_analysis_summary.md
    |-- feature_importance_mlp.csv
    `-- feature_ablation_results.csv
```

## 발표 준비 메모

발표 슬라이드에는 최소한 아래 숫자와 그래프가 들어가는 것이 좋다.

- high-frequency threshold = 24
- 운영 기준 delay threshold = 15  
- 현재 target gap 분위수 = q80 15 / q90 24 / q95 30
- 데이터 규모 = 전체 고객 206,209명 / 전체 주문 3,421,083건 / 최종 샘플 42,499명
- 현재 가공 데이터 기준 positive ratio = 18.65%
- split 규모 = train 29,749 / validation 6,375 / test 6,375
- split 방식 = user-level stratified train / val / test
- ROC curve
- PR curve
- 최종 모델 비교 표
- tabular model comparison 그래프
- MLP threshold 비교 그래프
- feature importance 상위 feature

세부 근거와 라벨 기준 선택 이유는 [`docs/preprocessing_summary.md`](./docs/preprocessing_summary.md)와 [`docs/repository_audit.md`](./docs/repository_audit.md)에 정리했다. 발표 예상 질문 답변은 [`docs/midterm_presentation_qna.md`](./docs/midterm_presentation_qna.md)에 따로 정리했고, 팀원 공유용 전체 현황은 [`docs/project_status_share.md`](./docs/project_status_share.md)에서 빠르게 확인할 수 있다. 현재 LSTM 참고 실험의 위치와 한계는 [`docs/lstm_reference_note.md`](./docs/lstm_reference_note.md)에 따로 정리했다.
