# Tabular Model Comparison Summary

## 목적
DummyClassifier, LogisticRegression, MLP와 Tree Boosting 계열 모델을 같은 split과 같은 validation threshold tuning 기준으로 비교한다.

## 전체 결과 요약
- F1 기준 1위: CatBoost (F1=0.5105, Recall=0.6947, ROC-AUC=0.8226)
- Tree Boosting 계열 best: CatBoost (F1=0.5105, Recall=0.6947, ROC-AUC=0.8226)

## 지표별 1위
- Precision: CatBoost (0.4035)
- Recall: LightGBM (0.7157)
- F1-score: CatBoost (0.5105)
- ROC-AUC: CatBoost (0.8226)
- Average Precision: XGBoost (0.4807)

## 발표용 해석
- CatBoost는 F1-score와 ROC-AUC 기준에서 가장 균형적인 결과를 보였다.
- LightGBM은 Recall이 가장 높아 위험 고객을 더 넓게 포착하는 방향에 강점이 있다.
- XGBoost는 Average Precision이 가장 높아 positive class ranking 관점에서 강점이 있다.
- MLP는 Tree Boosting 계열보다 약간 낮지만 큰 차이는 아니므로, 집계 feature 기반 신경망 대안으로 해석할 수 있다.

## 해석 원칙
- accuracy 단독 결론은 피하고 Recall, F1-score, ROC-AUC, Average Precision을 함께 해석한다.
- Tree Boosting이 MLP보다 높게 나오면, 현재 집계형 tabular feature에서는 딥러닝보다 강한 tabular baseline이 더 적합할 수 있다고 설명한다.
- MLP가 높게 나오더라도 Dummy/Logistic보다 강한 baseline과 비교했다는 점을 함께 제시한다.
- LSTM은 실제 주문 sequence pipeline 보강 후 별도 비교하는 것이 안전하다.

## 상세 결과
### CatBoost
- Family: tree_boosting
- Threshold: 0.58
- Precision: 0.4035
- Recall: 0.6947
- F1-score: 0.5105
- ROC-AUC: 0.8226
- Average Precision: 0.4795
- Confusion Matrix: TN=3965, FP=1221, FN=363, TP=826

### LightGBM
- Family: tree_boosting
- Threshold: 0.55
- Precision: 0.3958
- Recall: 0.7157
- F1-score: 0.5097
- ROC-AUC: 0.8181
- Average Precision: 0.4649
- Confusion Matrix: TN=3887, FP=1299, FN=338, TP=851

### XGBoost
- Family: tree_boosting
- Threshold: 0.58
- Precision: 0.3990
- Recall: 0.6997
- F1-score: 0.5082
- ROC-AUC: 0.8209
- Average Precision: 0.4807
- Confusion Matrix: TN=3933, FP=1253, FN=357, TP=832

### MLP
- Family: neural_network
- Threshold: 0.24
- Precision: 0.4012
- Recall: 0.6863
- F1-score: 0.5064
- ROC-AUC: 0.8179
- Average Precision: 0.4663
- Confusion Matrix: TN=3968, FP=1218, FN=373, TP=816

### LogisticRegression
- Family: linear
- Threshold: 0.52
- Precision: 0.3410
- Recall: 0.6703
- F1-score: 0.4521
- ROC-AUC: 0.7578
- Average Precision: 0.4090
- Confusion Matrix: TN=3646, FP=1540, FN=392, TP=797

### DummyClassifier
- Family: baseline
- Threshold: 0.50
- Precision: 0.0000
- Recall: 0.0000
- F1-score: 0.0000
- ROC-AUC: 0.5000
- Average Precision: 0.1865
- Confusion Matrix: TN=5186, FP=0, FN=1189, TP=0
