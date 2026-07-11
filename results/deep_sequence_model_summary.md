# Deep Sequence Model Summary

## 실험 목적

기존 집계 feature 실험을 보완하기 위해 target 주문 이전 최근 주문 이력을 실제 sequence 입력으로 구성하고, LSTM/GRU/BiGRU/TCN/Transformer 계열 딥러닝 모델을 비교한다.

## 데이터 구성

- 기준 데이터: `model_base_hv.csv`의 target anchor와 label
- 원본 주문 이력: `incoming_inspect/raw_archive/archive/orders.csv`
- sequence length: `20`
- sequence features: `gap, gap_delta, rolling_3_gap, order_dow_sin, order_dow_cos, order_hour_sin, order_hour_cos, relative_order_position`
- samples: `42,499`
- positive ratio: `0.1865`
- feature-label separation: target 이전 주문만 입력으로 사용하고, label은 target 주문의 target_gap으로 생성

## split

| split | samples | positives | positive_ratio |
| --- | ---: | ---: | ---: |
| train | 29,749 | 5,548 | 0.1865 |
| validation | 6,375 | 1,189 | 0.1865 |
| test | 6,375 | 1,189 | 0.1865 |

## 지표별 대표 모델

- F1-score 기준: `LSTM` (F1=0.4569, Recall=0.6173, AP=0.4179)
- Recall 기준: `TCN` (Recall=0.6594, F1=0.4519)
- Average Precision 기준: `TransformerEncoder` (AP=0.4263, ROC-AUC=0.7600)

## 발표용 해석 방향

- 정형 요약 feature만 사용한 실험은 강한 baseline으로 유지한다.
- 최종 딥러닝 실험은 최근 주문 sequence를 직접 입력으로 넣어 주문 간격 변화와 순차 패턴을 학습한다는 점에 의미가 있다.
- 최종 모델은 단순 accuracy가 아니라 Recall, F1-score, Average Precision을 함께 고려해 선정한다.
- 성능 차이가 작다면, 순차 정보를 직접 반영할 수 있는 구조적 타당성과 지표 균형을 함께 근거로 제시한다.

## 하이퍼파라미터 요약

| model | batch_size | max_epochs | patience | learning_rate | weight_decay | pos_weight | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeepMLP_SequenceFlatten | 512 | 80 | 10 | 0.0010 | 0.0001 | 4.3621 | Sequence를 펼쳐서 쓰는 강한 MLP 기준선. |
| LSTM | 512 | 80 | 10 | 0.0010 | 0.0001 | 4.3621 | 최근 주문 sequence를 순서대로 읽는 recurrent model. |
| GRU | 512 | 80 | 10 | 0.0010 | 0.0001 | 4.3621 | LSTM보다 단순한 gate 구조의 recurrent model. |
| BiGRU | 512 | 80 | 10 | 0.0010 | 0.0001 | 4.3621 | 이미 알고 있는 target 이전 sequence 전체를 양방향으로 요약하는 GRU. |
| TCN | 512 | 80 | 10 | 0.0010 | 0.0001 | 4.3621 | 시간축 1D convolution으로 주문 간격 패턴을 잡는 sequence model. |
| TransformerEncoder | 512 | 80 | 10 | 0.0010 | 0.0001 | 4.3621 | Self-attention으로 최근 주문 sequence 내 위치 간 관계를 요약하는 model. |

## 상세 성능

| rank_by_f1 | model | decision_threshold | precision | recall | f1_score | roc_auc | average_precision | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | LSTM | 0.5900 | 0.3626 | 0.6173 | 0.4569 | 0.7604 | 0.4179 | 3896 | 1290 | 455 | 734 |
| 2 | BiGRU | 0.5900 | 0.3546 | 0.6350 | 0.4551 | 0.7573 | 0.4135 | 3812 | 1374 | 434 | 755 |
| 3 | TransformerEncoder | 0.6200 | 0.3684 | 0.5896 | 0.4534 | 0.7600 | 0.4263 | 3984 | 1202 | 488 | 701 |
| 4 | TCN | 0.5100 | 0.3437 | 0.6594 | 0.4519 | 0.7542 | 0.4096 | 3689 | 1497 | 405 | 784 |
| 5 | GRU | 0.6000 | 0.3667 | 0.5854 | 0.4509 | 0.7582 | 0.4151 | 3984 | 1202 | 493 | 696 |
| 6 | DeepMLP_SequenceFlatten | 0.5600 | 0.3455 | 0.6426 | 0.4494 | 0.7507 | 0.3984 | 3739 | 1447 | 425 | 764 |
