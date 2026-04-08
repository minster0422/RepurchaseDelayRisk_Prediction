# results 폴더 안내

이 폴더는 발표용 숫자와 그래프를 정리하는 위치다. 현재는 **템플릿 중심 상태**이며, 실제 수치와 그림은 아직 채워지지 않았다.

## 현재 포함된 파일

- `model_comparison_template.csv`
  - 모델별 핵심 평가 지표를 한 표에 모으는 템플릿
- `thresholds_template.json`
  - validation set에서 선택한 threshold를 기록하는 템플릿

## 실제 실행 후 추가되면 좋은 파일

- `final_metrics.csv`
- `roc_curve.png`
- `pr_curve.png`
- `confusion_matrix_mlp.png`
- `confusion_matrix_lstm.png`
- `test_predictions_dummy.csv`
- `test_predictions_logreg.csv`
- `test_predictions_mlp.csv`
- `test_predictions_lstm.csv`

## 기록 원칙

- 최종 보고용 수치는 test set 기준으로 남긴다.
- `DummyClassifier`를 제외한 `LogisticRegression`, `MLP`, `LSTM`은 같은 validation set에서 threshold tuning을 수행한다.
- accuracy만으로 결론을 내리지 않고 Recall, F1, ROC-AUC, Average Precision을 함께 기록한다.

## 해석 원칙

- `MLP`는 집계 특징 기반 분리력이 어느 정도인지 확인하는 모델이다.
- `LSTM`은 순차 정보가 실제로 추가적인 예측 도움을 주는지 확인하는 모델이다.
- 최종 결론은 실제 수치가 채워진 뒤에 정리하며, 템플릿 상태에서는 특정 모델을 확정적으로 최고라고 쓰지 않는다.
