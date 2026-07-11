# 모델 설계 피드백과 개선안

## 발표 당시 HybridGRU 구조

최종 발표에서 사용한 HybridGRU는 다음 구조다.

```text
최근 주문 sequence -> GRU encoder -> sequence embedding
고객 요약 feature -> MLP encoder -> tabular embedding
sequence embedding + tabular embedding -> fusion head -> delay_risk score
```

현재 구현은 `train_hybrid_deep_models.py`의 `HybridClassifier`에 해당한다. `forward()`에서는 GRU embedding과 MLP embedding을 concat한 뒤 최종 head를 통과시킨다. 학습 시에는 최종 출력 하나에 대해 `BCEWithLogitsLoss`를 계산한다.

```text
loss = BCEWithLogitsLoss(final_logit, y)
```

따라서 현재 구조는 **single-loss late fusion**으로 정리할 수 있다.

## 교수님 피드백의 핵심

교수님 피드백은 GRU와 MLP를 결합하는 하이브리드 구조라면 두 branch가 각각 의미 있게 학습되는지 확인할 수 있도록, branch별 loss를 함께 설계하는 것이 더 타당하다는 취지로 해석할 수 있다.

즉, 단순히 최종 출력 하나만 맞히게 하는 것이 아니라 다음 세 출력을 함께 학습시키는 방식이다.

```text
GRU branch -> sequence-only prediction -> loss_seq
MLP branch -> tabular-only prediction -> loss_tab
GRU + MLP fusion -> final prediction -> loss_fusion
```

전체 loss는 아래처럼 둘 수 있다.

```text
total_loss = loss_fusion + alpha * loss_seq + beta * loss_tab
```

여기서 `alpha`, `beta`는 보조 loss의 영향력을 조절하는 하이퍼파라미터다.

## 왜 이 구조가 더 타당한가

- GRU branch가 실제로 sequence 정보를 학습하는지 확인할 수 있다.
- MLP branch가 고객 요약 feature만으로 어느 정도 예측하는지 확인할 수 있다.
- fusion 결과가 branch 단독 결과보다 개선되는지 더 명확하게 비교할 수 있다.
- 한 branch가 다른 branch에 묻히는 현상을 줄일 수 있다.
- “왜 하이브리드 모델인가?”라는 질문에 구조적으로 더 강하게 답할 수 있다.

## 성능이 반드시 올라가는가?

보조 loss를 추가한다고 성능이 반드시 올라가는 것은 아니다. 이미 final loss만으로 충분히 학습됐을 수도 있고, `alpha`, `beta`를 잘못 설정하면 최종 예측 성능이 오히려 떨어질 수도 있다.

다만 성능 향상 여부와 별개로, auxiliary-loss 구조는 하이브리드 모델의 설계 타당성과 학습 안정성을 높이는 방향이다. 따라서 후속 실험에서는 기존 HybridGRU와 auxiliary-loss HybridGRU를 같은 split에서 비교하는 것이 좋다.

## 후속 실험 설계

권장 비교는 아래와 같다.

| 모델 | 설명 |
| --- | --- |
| TabularDeepMLP | 고객 요약 feature만 사용 |
| GRU sequence-only | 최근 주문 sequence만 사용 |
| HybridGRU single-loss | 기존 발표 모델 |
| HybridGRU auxiliary-loss | branch별 보조 loss 추가 모델 |

평가 지표는 기존과 동일하게 Precision, Recall, F1-score, ROC-AUC, Average Precision을 사용한다. 최종 판단은 F1-score만 보지 않고, sequence-only 대비 개선폭과 branch별 예측 성능을 함께 본다.

## 보고서용 문장

> 발표 당시 HybridGRU는 GRU branch와 MLP branch의 embedding을 결합한 뒤 최종 출력에 대해 하나의 loss를 계산하는 late-fusion 구조였다. 발표 후 피드백을 반영하면, 각 branch에 보조 prediction head와 auxiliary loss를 추가해 sequence branch와 tabular branch가 독립적으로도 의미 있는 표현을 학습하도록 설계하는 것이 더 엄밀하다. 이 개선안은 성능 향상을 보장하기보다는 하이브리드 모델의 학습 구조와 해석 가능성을 강화하는 방향이다.
