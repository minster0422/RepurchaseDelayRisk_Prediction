# results 폴더 사용 규칙

이 폴더는 발표용 숫자와 그래프의 **source of truth**를 저장하는 위치다.

## 최소 권장 산출물

- `model_comparison_template.csv`
  - 모델별 최종 평가 지표를 한 표로 정리한다.
- `thresholds.json`
  - validation set에서 고른 decision threshold를 기록한다.
- `roc_curve.png`
  - 모델별 ROC curve 비교 그림
- `pr_curve.png`
  - 모델별 PR curve 비교 그림

## 작성 원칙

- test set 기준 숫자만 최종 표에 남긴다.
- accuracy만으로 결론을 내리지 않는다.
- Recall, F1, ROC-AUC, Average Precision을 함께 기록한다.
- 해석 메모는 과장하지 않고, 지표별 장단점을 분리해서 쓴다.
