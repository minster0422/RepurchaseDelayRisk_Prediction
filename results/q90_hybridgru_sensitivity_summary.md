# q90 HybridGRU Sensitivity

- Label: `delay_gt_24_q90` (`target_gap > 24`)
- Positive count: 4,207
- Positive ratio: 0.0990
- Decision threshold: 0.71
- Precision: 0.3169
- Recall: 0.6101
- F1-score: 0.4171
- ROC-AUC: 0.8507
- Average Precision: 0.3372
- Confusion matrix: TN=4914, FP=830, FN=246, TP=385

## 해석 메모

q90=24일 기준은 더 심한 지연만 양성으로 정의하므로 positive class가 줄어든다.
따라서 precision, recall, F1은 q80=15일 기준과 직접적으로 같은 난이도의 점수로 비교하기 어렵고, 라벨 기준 변화에 따른 민감도 결과로 해석해야 한다.