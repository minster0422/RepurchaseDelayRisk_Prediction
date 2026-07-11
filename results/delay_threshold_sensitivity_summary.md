# Delay Threshold Sensitivity Summary

## 목적
`target_gap > 15` q80 조기탐지 기준과 `target_gap > 24` q90 고위험 기준을 비교해, 라벨 기준 선택의 영향을 확인한다.

## 라벨별 양성 비율
- `delay_gt_15_q80`: threshold `15`일, 양성 수 `7,926`, 양성 비율 `0.1865` (q80 조기탐지 기준)
- `delay_gt_24_q90`: threshold `24`일, 양성 수 `4,207`, 양성 비율 `0.0990` (q90 고위험 기준)

## F1 기준 best model
- q80 기준(`target_gap > 15`): CatBoost (F1=0.5105, Recall=0.6947, ROC-AUC=0.8226)
- q90 기준(`target_gap > 24`): LightGBM (F1=0.4140, Recall=0.6355, ROC-AUC=0.8430)

## 발표용 해석
- q80 기준은 양성 클래스가 더 많아 조기 위험 탐지와 모델 학습 안정성에 유리하다.
- q90 기준은 더 엄격한 고위험 지연을 정의하지만 양성 클래스가 줄어들어 Recall/F1 해석이 더 민감해질 수 있다.
- 중간발표에서는 q80 기준을 기본 실험으로 사용하고, q90 기준은 교수님 피드백을 받아 최종 기준으로 검토할 대안이라고 설명한다.