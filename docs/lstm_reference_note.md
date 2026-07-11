# LSTM 참고 실험 정리

## 현재 결론

LSTM 코드는 현재 저장소에 존재한다. 다만 현재 `model_base_hv.csv`는 사용자당 1행의 집계 feature 데이터이므로, 이 LSTM은 실제 주문 sequence를 직접 입력받는 모델이 아니다.

따라서 중간발표에서는 LSTM을 **완성된 순차 모델 결과**로 제시하기보다, **순차 모델 확장을 위한 참고 실험**으로 설명하는 것이 안전하다.

## 관련 파일

| 파일 | 역할 |
| --- | --- |
| `train_lstm.py` | 집계 feature vector를 pseudo-sequence 형태로 변환해 LSTM을 학습하는 참고 코드 |
| `results/lstm_summary.csv` | LSTM 참고 실험 요약 지표 |
| `results/threshold_comparison_lstm.csv` | threshold별 Precision, Recall, F1-score 비교 |
| `results/lstm_model.pt` | 학습된 PyTorch 모델 가중치 |
| `notebooks/03_train_lstm.ipynb` | 발표용 LSTM 역할 설명 노트북 |

## 현재 LSTM 실험 방식

`train_lstm.py`는 다음 흐름으로 동작한다.

1. `model_base_hv.csv`를 읽는다.
2. `delay_risk`, `target_gap`, `user_id`를 제외한 numeric feature를 사용한다.
3. 결측값을 median으로 채운 뒤 `StandardScaler`를 적용한다.
4. feature vector를 `seq_len = 4` 기준으로 reshape해 `(sample, sequence_length, feature_dim)` 형태로 만든다.
5. 이 pseudo-sequence를 LSTM에 입력해 `delay_risk`를 예측한다.

주의할 점은 이 sequence가 실제 주문 순서를 복원한 것이 아니라는 점이다. 즉 `최근 1번째 주문 -> 최근 2번째 주문 -> ...` 같은 진짜 주문 흐름이 아니라, 집계 feature를 모양만 sequence처럼 바꾼 구조다.

## 참고 성능

현재 `results/lstm_summary.csv` 기준:

| 지표 | 값 |
| --- | ---: |
| ROC-AUC | 0.8146 |
| Average Precision | 0.4597 |
| best threshold | 0.60 |
| Precision | 0.4036 |
| Recall | 0.6776 |
| F1-score | 0.5059 |

이 성능은 MLP와 가까운 수준이지만, split과 입력 구조가 tabular 비교 실험과 완전히 같지 않으므로 최종 비교표에 같은 조건의 test 결과처럼 넣으면 안 된다.

## 발표에서 권장 표현

> LSTM 코드와 참고 실험은 존재하지만, 현재 입력 데이터가 실제 주문 sequence가 아니라 사용자 단위 집계 feature이기 때문에 최종 비교 실험에서는 tabular 모델 중심으로 해석했습니다. LSTM은 최종발표에서 실제 주문 sequence pipeline을 복원한 뒤 다시 비교할 확장 후보로 남겼습니다.

## 앞으로 보강할 점

- raw Instacart 주문 이력에서 사용자별 최근 N개 주문 sequence를 다시 구성하기
- 각 time step에 주문 간격, basket size, 요일, reorder 비율 등 순차 feature를 넣기
- MLP, Tree Boosting, LSTM이 같은 train/validation/test split에서 평가되도록 맞추기
- LSTM 결과를 ROC-AUC, Average Precision, Recall, F1-score 기준으로 다시 비교하기
