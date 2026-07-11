# 발표용 그래프 사용 가이드

## 결론

기존 `tabular_roc_curve.png`, `tabular_pr_curve.png`는 모든 모델을 한 번에 보여주는 기록용 그래프다. 모델 수가 많아 선이 겹치므로 발표 슬라이드에서는 아래 `presentation_*` 그래프를 우선 사용하는 것이 좋다.

## 발표 슬라이드 우선 사용 그래프

| 우선순위 | 파일 | 용도 |
| ---: | --- | --- |
| 1 | `results/presentation_metric_summary.png` | 모델별 F1-score, Recall, Average Precision을 한 번에 요약 |
| 2 | `results/presentation_selected_roc_pr.png` | LogisticRegression, MLP, CatBoost만 골라 ROC/PR curve를 간결하게 비교 |
| 3 | `results/presentation_threshold_sensitivity.png` | q80 기준 15일과 q90 기준 24일 라벨의 F1-score 차이 설명 |
| 4 | `results/presentation_catboost_confusion_matrix.png` | CatBoost 기준 TP/FP/FN/TN 오류 해석 |

## 보조자료로만 쓰는 그래프

| 파일 | 이유 |
| --- | --- |
| `results/tabular_roc_curve.png` | 전체 모델 ROC curve 기록용. 선이 많아 발표용으로는 복잡함 |
| `results/tabular_pr_curve.png` | 전체 모델 PR curve 기록용. 불균형 데이터 설명에는 좋지만 발표 슬라이드에서는 복잡함 |
| `results/tabular_confusion_matrices.png` | 모든 모델 confusion matrix 기록용. 슬라이드에는 CatBoost 단일 행렬이 더 명확함 |

## 발표에서 말할 포인트

- 전체 실험은 모든 모델을 공정하게 비교했지만, 발표 그래프는 가독성을 위해 대표 모델만 선택했다.
- `presentation_selected_roc_pr.png`에서는 LogisticRegression, MLP, CatBoost를 보여준다.
- LogisticRegression은 선형 baseline, MLP는 집계 feature 기반 신경망, CatBoost는 현재 F1/ROC-AUC 기준 가장 균형적인 tabular 모델이다.
- LightGBM과 XGBoost는 curve에서는 생략하지만, `presentation_metric_summary.png`에서 지표별 장단점을 함께 보여준다.
- LSTM은 현재 pseudo-sequence 참고 실험이므로 발표 핵심 그래프에는 넣지 않고, 별도 참고 또는 향후 과제로 설명한다.

## 그래프 재생성 방법

```bash
python make_presentation_figures.py
```

위 스크립트는 `results/tabular_model_comparison.csv`, `results/tabular_test_predictions.csv`, `results/delay_threshold_sensitivity.csv`를 읽어 발표용 PNG 파일을 다시 생성한다.
