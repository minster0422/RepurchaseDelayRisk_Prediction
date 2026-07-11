# 최종 프로젝트 요약

## 프로젝트 목표

이 프로젝트는 Instacart 주문 이력을 사용해 **고빈도 고객의 다음 주문 재구매 지연 위험**을 예측하는 이진 분류 프로젝트다. 목적은 모든 고객에게 동일하게 쿠폰이나 알림을 보내기보다, 지연 위험 score가 높은 고객을 먼저 선별해 고객 유지 전략에 활용할 수 있는 가능성을 확인하는 것이다.

## 데이터와 라벨

- 데이터셋: Instacart 공개 주문 데이터
- 전체 고객 수: 206,209명
- 전체 주문 수: 3,421,083건
- 최종 모델링 샘플 수: 42,499명
- 고빈도 고객 기준: 총 주문 횟수 상위 20%
- 실제 threshold: 24회
- 라벨 기준: `target_gap > 15`
- target gap 분위수: q80 = 15, q90 = 24, q95 = 30
- positive ratio: 18.65%

`target_gap`은 모델 입력 변수가 아니라 label 생성용 변수다. feature는 target 주문 이전 이력에서 만들고, label은 target 주문의 실제 gap으로 만든다.

## 실험 구성

세 가지 모델군을 비교했다.

| 모델군 | 대표 모델 | 목적 |
| --- | --- | --- |
| Tabular baseline | LogisticRegression, MLP, CatBoost, LightGBM, XGBoost | 고객 요약 feature만으로 예측 가능한지 확인 |
| Sequence-only deep learning | LSTM, GRU, BiGRU, TCN, TransformerEncoder | 최근 주문 순서 정보만으로 예측 가능한지 확인 |
| Hybrid deep learning | HybridLSTM, HybridGRU, HybridBiGRU, HybridTCN, HybridTransformer | 고객 요약 feature와 최근 주문 sequence를 결합했을 때 개선되는지 확인 |

split은 stratified random split으로 train/validation/test = 70/15/15를 사용했다. 현재 데이터는 사용자당 1행이므로 같은 사용자가 여러 split에 동시에 들어가는 고객 중복 문제는 없다. 다만 실제 서비스 적용 단계에서는 시간 기준 검증이 더 엄격하다.

## 최종 후보 모델

최종 발표에서는 **HybridGRU**를 딥러닝 최종 후보로 해석했다.

HybridGRU는 다음 두 입력을 함께 사용한다.

- GRU branch: 최근 20개 주문 sequence
- MLP branch: 평균 주문 간격, 최근 주문 간격, 주문 빈도, 활동 기간 등 고객 요약 feature

이 두 표현을 결합해 최종 `delay_risk` score를 출력한다.

## 핵심 결과

| model | precision | recall | f1_score | roc_auc | average_precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| HybridGRU | 0.4116 | 0.6972 | 0.5176 | 0.8220 | 0.4737 |
| CatBoost | 0.4035 | 0.6947 | 0.5105 | 0.8226 | 0.4795 |
| LightGBM | 0.3958 | 0.7157 | 0.5097 | 0.8181 | 0.4649 |
| MLP | 0.4012 | 0.6863 | 0.5064 | 0.8179 | 0.4663 |
| LSTM | 0.3626 | 0.6173 | 0.4569 | 0.7604 | 0.4179 |

HybridGRU는 CatBoost보다 압도적으로 우수한 모델이라기보다, **최근 주문 흐름과 고객 요약 feature를 함께 본다는 문제 구조에 가장 잘 맞고 F1 기준 가장 균형적인 딥러닝 후보**로 해석하는 것이 안전하다.

## 결과 해석

test set에서 실제 지연 고객은 1,189명이었고, HybridGRU는 그중 829명을 탐지했다.

- Recall = 829 / 1,189 = 0.6972
- Precision = 0.4116
- 전체 지연 고객 비율 = 18.65%

Precision 41.2%는 단독으로 보면 낮아 보일 수 있지만, 전체 지연 고객 비율 18.65%와 함께 보면 모델이 위험하다고 고른 고객군에는 실제 지연 고객이 평균보다 약 2.2배 더 많이 포함되어 있다. 따라서 이 모델은 모든 예측을 정확히 맞히는 모델이라기보다, **고객 유지 전략 후보를 우선순위화하는 risk score 모델**로 해석한다.

## 교수님 피드백 이후 보완 방향

최종 발표 이후 받은 중요한 피드백은 HybridGRU의 loss 설계다. 현재 구현은 GRU branch와 MLP branch를 결합한 뒤 최종 출력 하나에 대해 loss를 계산하는 single-loss late fusion 구조다. 더 엄밀한 구조로는 GRU branch, MLP branch, fusion head 각각에 예측 head와 loss를 두는 auxiliary-loss 구조를 고려할 수 있다.

자세한 내용은 `docs/model_design_feedback.md`에 정리한다.
