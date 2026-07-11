# Hybrid Deep Model Summary

이 파일은 최근 주문 sequence와 기존 고객 요약 feature를 함께 사용하는 하이브리드 딥러닝 실험 요약이다.
모든 입력은 target 주문 이전 이력만 사용하며, label은 target 주문의 `target_gap > 15`로 생성한다.

## Dataset

- Samples: 42,499
- Positive ratio: 0.1865
- Sequence length: 20
- Sequence features: gap, gap_delta, rolling_3_gap, order_dow_sin, order_dow_cos, order_hour_sin, order_hour_cos, relative_order_position
- Tabular features: total_orders_before_target, avg_gap_before_target, std_gap_before_target, min_gap_before_target, max_gap_before_target, recent_3_avg_gap, recent_5_avg_gap, last_gap_before_target, gap_trend, active_span_days, order_frequency, weekend_order_ratio, dow_variability
- Split method: stratified random split 70/15/15 using the same target-anchor samples as tabular baselines

## Best Models By Metric

- F1-score: HybridGRU (0.5176)
- Recall: HybridTransformer (0.7031)
- ROC-AUC: HybridBiGRU (0.8232)
- Average Precision: HybridBiGRU (0.4797)

## Model Comparison

| rank_by_f1 | model | precision | recall | f1_score | roc_auc | average_precision | decision_threshold | epochs_trained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | HybridGRU | 0.4116186693147964 | 0.6972245584524811 | 0.5176397127692788 | 0.822019446805902 | 0.47365321281812983 | 0.6100000000000001 | 25 |
| 2 | HybridTransformer | 0.39433962264150946 | 0.703111858704794 | 0.505288606829858 | 0.8116501793500454 | 0.4680147929323739 | 0.5900000000000002 | 18 |
| 3 | HybridLSTM | 0.40404040404040403 | 0.6728343145500421 | 0.5048911328494793 | 0.8176438668252528 | 0.47025692896641985 | 0.5700000000000002 | 22 |
| 4 | HybridBiGRU | 0.4324487334137515 | 0.6030277544154752 | 0.5036880927291886 | 0.8232244929335206 | 0.4796541692708176 | 0.6700000000000002 | 23 |
| 5 | HybridTCN | 0.4195176668536175 | 0.6291000841042893 | 0.5033647375504711 | 0.8135586623363608 | 0.4648811222864868 | 0.6200000000000001 | 20 |
| 6 | HybridMLP_Flatten | 0.40727081138040044 | 0.6501261564339781 | 0.5008098477486232 | 0.8091306996224876 | 0.4644175898439731 | 0.6200000000000001 | 16 |
| 7 | TabularDeepMLP | 0.42053930005737233 | 0.616484440706476 | 0.5 | 0.814715947736628 | 0.46720959662223727 | 0.6300000000000001 | 24 |

## Hyperparameters

| model | family | batch_size | max_epochs | patience | learning_rate | weight_decay | pos_weight | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TabularDeepMLP | deep_learning_tabular | 512 | 90 | 10 | 0.001 | 0.0001 | 4.36211247296323 | 기존 고객 요약 변수만 사용하는 PyTorch MLP 기준선. |
| HybridMLP_Flatten | deep_learning_hybrid | 512 | 90 | 10 | 0.001 | 0.0001 | 4.36211247296323 | 최근 주문 sequence를 펼친 표현과 고객 요약 변수를 함께 사용하는 MLP. |
| HybridLSTM | deep_learning_hybrid_sequence | 512 | 90 | 10 | 0.001 | 0.0001 | 4.36211247296323 | 최근 주문 순서 패턴과 고객 요약 변수를 함께 반영하는 LSTM 하이브리드 모델. |
| HybridGRU | deep_learning_hybrid_sequence | 512 | 90 | 10 | 0.001 | 0.0001 | 4.36211247296323 | LSTM보다 단순한 gate 구조로 sequence와 요약 변수를 결합하는 GRU 하이브리드 모델. |
| HybridBiGRU | deep_learning_hybrid_sequence | 512 | 90 | 10 | 0.001 | 0.0001 | 4.36211247296323 | target 이전 sequence 전체를 양방향으로 요약한 뒤 고객 요약 변수와 결합하는 모델. |
| HybridTCN | deep_learning_hybrid_sequence | 512 | 90 | 10 | 0.001 | 0.0001 | 4.36211247296323 | 시간축 convolution으로 최근 주문 간격 패턴을 잡고 고객 요약 변수와 결합하는 모델. |
| HybridTransformer | deep_learning_hybrid_sequence | 512 | 90 | 10 | 0.001 | 0.0001 | 4.36211247296323 | self-attention 기반 sequence 표현과 고객 요약 변수를 결합하는 모델. |

## Presentation Note

정형 요약 변수만 사용하는 모델과 비교했을 때, 하이브리드 모델은 고객의 장기 요약 정보와 최근 주문 순서 패턴을 함께 반영한다.
따라서 최종 발표에서는 단순히 성능표를 나열하기보다, 입력 표현을 강화했을 때 딥러닝 모델이 어떤 지표에서 개선되는지를 중심으로 해석하는 것이 좋다.