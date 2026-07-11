# results 폴더 안내

이 폴더는 최종 프로젝트에서 생성한 모델 성능표, 시각화, threshold, 오류 분석 결과를 모아둔 위치다.

## 가장 먼저 볼 파일

| 파일 | 내용 |
| --- | --- |
| `final_model_comparison.csv` | 전체 모델 최종 비교 표 |
| `final_model_comparison_summary.md` | 최종 모델 비교 해석 요약 |
| `final_model_comparison.png` | 주요 모델 성능 비교 그래프 |
| `final_selected_roc_curve.png` | 최종 발표용 ROC curve |
| `final_selected_pr_curve.png` | 최종 발표용 PR curve |
| `final_hybridgru_confusion_matrix.png` | HybridGRU confusion matrix |

## 최종 모델 비교 요약

최종 실험에서는 세 그룹의 모델을 비교했다.

- Tabular baseline: DummyClassifier, LogisticRegression, MLP, CatBoost, LightGBM, XGBoost
- Sequence-only deep learning: LSTM, GRU, BiGRU, TCN, TransformerEncoder
- Hybrid deep learning: HybridLSTM, HybridGRU, HybridBiGRU, HybridTCN, HybridTransformer

핵심 결과는 다음과 같다.

| model | precision | recall | f1_score | roc_auc | average_precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| HybridGRU | 0.4116 | 0.6972 | 0.5176 | 0.8220 | 0.4737 |
| CatBoost | 0.4035 | 0.6947 | 0.5105 | 0.8226 | 0.4795 |
| LightGBM | 0.3958 | 0.7157 | 0.5097 | 0.8181 | 0.4649 |
| XGBoost | 0.3990 | 0.6997 | 0.5082 | 0.8209 | 0.4807 |
| MLP | 0.4012 | 0.6863 | 0.5064 | 0.8179 | 0.4663 |
| LSTM | 0.3626 | 0.6173 | 0.4569 | 0.7604 | 0.4179 |

HybridGRU는 모든 지표에서 압도적으로 우수한 모델은 아니지만, 고객 요약 feature와 최근 주문 sequence를 함께 보는 구조를 가장 직접적으로 구현했고 F1-score 기준 가장 균형적인 결과를 보였다.

## 모델군별 결과 파일

| 구분 | 파일 |
| --- | --- |
| Tabular baseline 비교 | `tabular_model_comparison.csv`, `tabular_model_comparison_summary.md` |
| Tabular ROC/PR | `tabular_roc_curve.png`, `tabular_pr_curve.png` |
| Sequence-only 비교 | `deep_sequence_model_comparison.csv`, `deep_sequence_model_summary.md` |
| Sequence-only ROC/PR | `deep_sequence_roc_curve.png`, `deep_sequence_pr_curve.png` |
| Hybrid 비교 | `hybrid_deep_model_comparison.csv`, `hybrid_deep_model_summary.md` |
| Hybrid ROC/PR | `hybrid_deep_roc_curve.png`, `hybrid_deep_pr_curve.png` |

## 추가 분석 파일

| 분석 | 파일 | 목적 |
| --- | --- | --- |
| 라벨 기준 민감도 | `delay_threshold_sensitivity.csv`, `delay_threshold_sensitivity_summary.md` | q80 15일 기준과 q90 24일 기준 비교 |
| q90 HybridGRU 실험 | `q90_hybridgru_sensitivity.csv`, `q90_hybridgru_sensitivity_summary.md` | 더 엄격한 라벨 기준에서 HybridGRU 성능 확인 |
| HybridGRU 오류 분석 | `hybridgru_error_group_summary.csv`, `hybridgru_error_feature_contrast.csv`, `hybridgru_error_analysis_summary.md` | FP/FN/TP/TN 그룹 차이 확인 |
| Feature importance | `feature_importance_mlp.csv`, `feature_importance_mlp.png` | MLP 기준 permutation importance |
| Feature ablation | `feature_ablation_results.csv`, `feature_ablation_summary.md`, `feature_ablation_f1.png` | feature set 축소에 따른 성능 변화 |

## 해석 시 주의점

- positive class 비율이 18.65%인 불균형 데이터이므로 accuracy만으로 결론을 내리지 않는다.
- HybridGRU와 CatBoost의 F1-score 차이는 크지 않다. 따라서 HybridGRU를 압도적 최고 모델로 표현하지 않는다.
- HybridGRU는 문제 구조에 적합하고 sequence-only 모델보다 개선된 딥러닝 최종 후보로 해석한다.
- Precision 0.4116은 전체 지연 고객 비율 18.65%와 함께 해석해야 한다. 위험 예측 고객군에는 실제 지연 고객이 평균보다 약 2.2배 많이 포함되어 있다.
- 오류 분석은 인과관계 증명이 아니라, 어떤 feature 패턴에서 오탐/미탐이 발생하는지 살펴보는 보조 분석이다.

## 발표자료와 연결

최종 발표자료는 `presentation/딥러닝_프로젝트_발표자료_이민성팀.pdf`에 있다. 발표자료에서 사용한 핵심 그래프는 아래 파일을 기반으로 한다.

- `final_model_comparison.png`
- `final_selected_roc_curve.png`
- `final_selected_pr_curve.png`
- `final_hybridgru_confusion_matrix.png`
- `feature_importance_mlp.png`
