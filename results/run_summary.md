# Run Summary

- 데이터 파일: `model_base_hv.csv`
- 샘플 수: 42,499
- 양성 비율: 0.1865
- 사용 feature 수: 13

## 모델별 결과
### MLP
- Threshold: 0.21
- Precision: 0.3788
- Recall: 0.6821
- F1-score: 0.4871
- ROC-AUC: 0.8017
- Average Precision: 0.4547
- Confusion Matrix: TN=3856, FP=1330, FN=378, TP=811

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