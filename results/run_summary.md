# Tuned Run Summary

- 데이터 파일: `model_base_hv.csv`
- 샘플 수: 42,499
- 양성 비율: 0.1865
- 사용 feature 수: 13
- 반복 seed 수: 5
- MLP 후보 설정 수: 4

## 선택된 최고 MLP 설정
- hidden_layer_sizes: (128, 64)
- alpha: 0.0001
- learning_rate_init: 0.001
- batch_size: 64

## 최종 대표 실행 결과
### MLP
- Threshold: 0.24
- Precision: 0.4012
- Recall: 0.6863
- F1-score: 0.5064
- ROC-AUC: 0.8179
- Average Precision: 0.4663
- Confusion Matrix: TN=3968, FP=1218, FN=373, TP=816

### LogisticRegression
- Threshold: 0.52
- Precision: 0.3410
- Recall: 0.6703
- F1-score: 0.4521
- ROC-AUC: 0.7578
- Average Precision: 0.4090
- Confusion Matrix: TN=3646, FP=1540, FN=392, TP=797

### Dummy
- Threshold: 0.50
- Precision: 0.0000
- Recall: 0.0000
- F1-score: 0.0000
- ROC-AUC: 0.5000
- Average Precision: 0.1865
- Confusion Matrix: TN=5186, FP=0, FN=1189, TP=0

## 최종 요약
- 현재 최고 모델(F1 기준): MLP
- 이번 실행은 Dummy / LogisticRegression / MLP만 수행했다.
- LSTM은 sequence pipeline 준비 후 별도 실행 필요.