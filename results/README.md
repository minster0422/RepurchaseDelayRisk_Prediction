# results 폴더 사용 규칙

이 폴더는 발표용 숫자와 그래프를 저장하는 위치다. 최종 결론은 이 폴더의 test set 기준 산출물을 중심으로 정리하는 것이 좋다.

## 최소 권장 산출물

- `model_comparison_template.csv`
  - 모델별 핵심 지표를 한 표에 정리한다.
- `thresholds_template.json`
  - validation set에서 고른 threshold를 남기는 형식 예시다.
- `roc_curve.png`
  - 모델별 ROC curve 비교 그림
- `pr_curve.png`
  - 모델별 PR curve 비교 그림

## 공정성 원칙

- `DummyClassifier`를 제외한 `LogisticRegression`, `MLP`, `LSTM`은 같은 validation set에서 threshold tuning을 수행한다.
- test set으로 threshold를 다시 맞추지 않는다.
- accuracy를 보조 지표로 두고, Recall / F1 / ROC-AUC / AP를 함께 기록한다.

## 결과 해석 원칙

- `MLP`는 ROC-AUC가 높다면 집계 특징 기반 분리력에 강점이 있다고 해석한다.
- `LogisticRegression`은 Recall이 높다면 위험 고객을 넓게 탐지하는 기준선으로 해석한다.
- `LSTM`은 F1-score가 가장 균형적일 때 최종 후보로 검토한다.
- 특정 모델을 무조건 최고라고 쓰지 않는다.
