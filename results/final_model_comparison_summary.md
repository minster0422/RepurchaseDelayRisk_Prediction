# Final Model Comparison Summary

이 문서는 최종 발표용 전체 모델 비교 요약이다. 같은 test split에서 정형 요약 feature 모델, 순수 sequence 딥러닝 모델, 하이브리드 딥러닝 모델을 함께 비교한다.

## Key Result

- F1-score 기준 최상위 모델: HybridGRU (0.5176)
- Recall 기준 최상위 모델: LightGBM (0.7157)
- ROC-AUC 기준 최상위 모델: HybridBiGRU (0.8232)
- Average Precision 기준 최상위 모델: XGBoost (0.4807)

## Presentation Interpretation

기존 정형 요약 feature만 사용한 모델에서는 CatBoost와 LightGBM이 강한 기준선으로 작동했다.
추가 실험에서는 target 이전 최근 주문 sequence를 구성했고, 이를 기존 고객 요약 변수와 결합한 하이브리드 딥러닝 모델을 학습했다.
그 결과 HybridGRU가 F1-score 기준 가장 높은 결과를 보였으며, 이는 재구매 지연 위험 고객을 precision과 recall의 균형 관점에서 탐지하는 목적에 가장 적합한 후보로 해석할 수 있다.
다만 ROC-AUC와 Average Precision에서는 CatBoost, XGBoost, HybridBiGRU가 매우 근접하므로, 최종 발표에서는 지표별 장단점을 함께 설명하는 것이 안전하다.

## Full Comparison

| rank_by_f1_final | model | experiment | input_type | precision | recall | f1_score | roc_auc | average_precision | decision_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | HybridGRU | hybrid_sequence_tabular | recent order sequence + customer aggregate features | 0.4116 | 0.6972 | 0.5176 | 0.8220 | 0.4737 | 0.6100 |
| 2 | CatBoost | tabular_aggregate | customer aggregate features | 0.4035 | 0.6947 | 0.5105 | 0.8226 | 0.4795 | 0.5800 |
| 3 | LightGBM | tabular_aggregate | customer aggregate features | 0.3958 | 0.7157 | 0.5097 | 0.8181 | 0.4649 | 0.5500 |
| 4 | XGBoost | tabular_aggregate | customer aggregate features | 0.3990 | 0.6997 | 0.5082 | 0.8209 | 0.4807 | 0.5800 |
| 5 | MLP | tabular_aggregate | customer aggregate features | 0.4012 | 0.6863 | 0.5064 | 0.8179 | 0.4663 | 0.2400 |
| 6 | HybridTransformer | hybrid_sequence_tabular | recent order sequence + customer aggregate features | 0.3943 | 0.7031 | 0.5053 | 0.8117 | 0.4680 | 0.5900 |
| 7 | HybridLSTM | hybrid_sequence_tabular | recent order sequence + customer aggregate features | 0.4040 | 0.6728 | 0.5049 | 0.8176 | 0.4703 | 0.5700 |
| 8 | HybridBiGRU | hybrid_sequence_tabular | recent order sequence + customer aggregate features | 0.4324 | 0.6030 | 0.5037 | 0.8232 | 0.4797 | 0.6700 |
| 9 | HybridTCN | hybrid_sequence_tabular | recent order sequence + customer aggregate features | 0.4195 | 0.6291 | 0.5034 | 0.8136 | 0.4649 | 0.6200 |
| 10 | HybridMLP_Flatten | hybrid_sequence_tabular | recent order sequence + customer aggregate features | 0.4073 | 0.6501 | 0.5008 | 0.8091 | 0.4644 | 0.6200 |
| 11 | TabularDeepMLP | hybrid_sequence_tabular | recent order sequence + customer aggregate features | 0.4205 | 0.6165 | 0.5000 | 0.8147 | 0.4672 | 0.6300 |
| 12 | LSTM | sequence_only | recent order sequence | 0.3626 | 0.6173 | 0.4569 | 0.7604 | 0.4179 | 0.5900 |
| 13 | BiGRU | sequence_only | recent order sequence | 0.3546 | 0.6350 | 0.4551 | 0.7573 | 0.4135 | 0.5900 |
| 14 | TransformerEncoder | sequence_only | recent order sequence | 0.3684 | 0.5896 | 0.4534 | 0.7600 | 0.4263 | 0.6200 |
| 15 | LogisticRegression | tabular_aggregate | customer aggregate features | 0.3410 | 0.6703 | 0.4521 | 0.7578 | 0.4090 | 0.5200 |
| 16 | TCN | sequence_only | recent order sequence | 0.3437 | 0.6594 | 0.4519 | 0.7542 | 0.4096 | 0.5100 |
| 17 | GRU | sequence_only | recent order sequence | 0.3667 | 0.5854 | 0.4509 | 0.7582 | 0.4151 | 0.6000 |
| 18 | DeepMLP_SequenceFlatten | sequence_only | recent order sequence | 0.3455 | 0.6426 | 0.4494 | 0.7507 | 0.3984 | 0.5600 |
| 19 | DummyClassifier | tabular_aggregate | customer aggregate features | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1865 | 0.5000 |

## Output Files

- `final_model_comparison.csv`: 전체 모델 성능 비교표
- `final_model_comparison.png`: 주요 지표 막대 그래프
- `final_selected_roc_curve.png`: 발표용 ROC curve
- `final_selected_pr_curve.png`: 발표용 PR curve

## Caution

Accuracy는 음성 class 비율이 높은 불균형 데이터에서 과대평가될 수 있으므로, 최종 해석은 F1-score, Recall, ROC-AUC, Average Precision을 중심으로 진행한다.