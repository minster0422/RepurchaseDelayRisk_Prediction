# Feature Ablation Summary

- 데이터 파일: `model_base_hv.csv`
- 실험 수: 5

## 결과 요약
### all_features
- feature 수: 13
- threshold: 0.24
- Precision: 0.4012
- Recall: 0.6863
- F1-score: 0.5064
- ROC-AUC: 0.8179
- Average Precision: 0.4663

### drop_active_span_days
- feature 수: 12
- threshold: 0.22
- Precision: 0.3876
- Recall: 0.6930
- F1-score: 0.4971
- ROC-AUC: 0.8050
- Average Precision: 0.4470

### drop_weak_time_features
- feature 수: 11
- threshold: 0.23
- Precision: 0.3940
- Recall: 0.6703
- F1-score: 0.4963
- ROC-AUC: 0.8075
- Average Precision: 0.4520

### recent_pattern_no_active_span
- feature 수: 11
- threshold: 0.21
- Precision: 0.3791
- Recall: 0.6754
- F1-score: 0.4856
- ROC-AUC: 0.7904
- Average Precision: 0.4445

### recent_pattern_only
- feature 수: 9
- threshold: 0.24
- Precision: 0.3836
- Recall: 0.6501
- F1-score: 0.4825
- ROC-AUC: 0.7892
- Average Precision: 0.4410

## 최종 해석
- 최고 설정(F1 기준): all_features
- `drop_active_span_days` 성능이 크게 떨어지면 active_span_days 의존도가 높다고 볼 수 있다.
- `drop_weak_time_features` 성능이 거의 유지되면 weekend_order_ratio, dow_variability는 제거 후보가 될 수 있다.
- `recent_pattern_only` 성능이 잘 나오면 최근성 중심 lightweight 모델 설명이 가능하다.