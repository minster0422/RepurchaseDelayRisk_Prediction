# Repurchase Delay Risk Prediction

## 1. 프로젝트 개요

이 프로젝트는 Instacart 주문 이력을 바탕으로, **고빈도 고객의 다음 주문이 평소보다 오래 지연될 위험**을 예측하는 것을 목표로 한다. 발표 서사는 단순 분류 문제가 아니라, **과거 주문 이력으로 다음 주문의 지연 위험을 미리 탐지하는 예측 프로젝트**로 구성한다.

현재 저장소에는 원시 Instacart 전체 파이프라인이 포함되어 있지 않고, 고빈도 고객 집계 특징이 담긴 [`model_base_hv.xlsx`](./model_base_hv.xlsx)만 남아 있다. 따라서 본 README는 **현재 확인 가능한 산출물**을 기준으로 작성하되, 검증이 필요한 부분은 `NOTE`와 `TODO`로 명시했다.

## 2. 문제 정의

핵심 문제는 다음과 같다.

> 과거 주문 이력만을 사용해, 고빈도 고객의 **다음 주문이 지연 위험 상태에 들어갈지** 예측할 수 있는가?

발표에서는 아래 두 가지를 분명히 보여주는 것이 중요하다.

- `feature`와 `label`의 시점을 분리한다.
- `target` 주문은 라벨 생성용으로만 사용하고, 입력 특징은 반드시 **target 주문 이전 이력**에서 만든다.

즉, 이 프로젝트의 구조는 "현재 주문을 설명하는 분류"가 아니라, **다음 주문의 지연 위험을 예측하는 구조**여야 한다.

## 3. 왜 Instacart 데이터인가

Instacart는 반복 구매가 많은 식료품 주문 데이터이기 때문에, 재구매 간격과 고객별 주문 습관을 비교적 자연스럽게 관찰할 수 있다. 특히 `days_since_prior_order`를 통해 다음 주문까지의 간격을 정의할 수 있어, 재구매 지연 위험 예측 문제를 설계하기에 적합하다.

다만 Instacart에는 직접적인 구매 금액 정보가 없기 때문에, 본 프로젝트는 "고가치 고객"보다 **고빈도 고객**이라는 표현이 더 정직하다.

## 4. 고빈도 고객 정의

발표용 기준 정의는 다음과 같다.

- 전체 고객 중 **총 주문 횟수 상위 20%**를 고빈도 고객으로 본다.
- 현재 프로젝트 설명에서는 이 경계가 **주문 횟수 24회**라고 가정한다.

현재 저장소에 남아 있는 [`model_base_hv.xlsx`](./model_base_hv.xlsx)를 보면 `target_order_number`의 최소값이 24이므로, 적어도 **가공 데이터 단계에서는 threshold = 24**가 반영된 것으로 보인다.

## 5. 타깃 정의

현재 프로젝트의 타깃은 절대 기준형 이진 분류다.

- `target_gap > 15` 이면 `delay_risk = 1`
- 그렇지 않으면 `delay_risk = 0`

즉, 특정 target 주문의 실제 재구매 간격이 15일을 초과하면 지연 위험군으로 본다.

`NOTE`

- 기존 발표 기획 문구에는 **"q90 ≈ 15일"** 이라는 설명이 포함되어 있다.
- 하지만 현재 저장소에 남아 있는 [`model_base_hv.xlsx`](./model_base_hv.xlsx)를 재계산하면 `target_gap` 분위수는 **q80 = 15, q90 = 24, q95 = 30**으로 나타난다.
- 따라서 **`target_gap > 15` 규칙 자체는 현재 데이터에서 확인되지만, `q90 = 15`라는 설명은 원본 전처리 코드 재검증이 필요하다.**

발표 자료에서는 이 불일치를 숨기기보다, 현재 산출물 기준 수치와 원래 기획 문구를 분리해서 설명하는 편이 더 안전하다.

## 6. 데이터 전처리 개요

전처리의 핵심 원칙은 아래와 같다.

- 고객별로 하나의 `target` 주문을 잡는다.
- `label`은 그 target 주문의 `target_gap`으로 만든다.
- `feature`는 target 주문 이전 이력만 사용해 만든다.
- 데이터 누수를 막기 위해, target 이후 정보는 어떤 형태로도 입력에 포함하지 않는다.

현재 포함된 가공 데이터 [`model_base_hv.xlsx`](./model_base_hv.xlsx)는 다음과 같은 **집계 특징 기반 표 형태**다.

- 샘플 수: 42,499
- 현재 확인된 양성 비율: 약 18.65%
- 현재 확인된 구조: 사용자당 1행
- 주요 열: `avg_gap_before_target`, `std_gap_before_target`, `recent_3_avg_gap`, `recent_5_avg_gap`, `gap_trend`, `order_frequency`, `weekend_order_ratio`

즉, 현재 저장소 기준으로는 **MLP / LogisticRegression / DummyClassifier를 위한 집계형 입력은 존재**하지만, **LSTM을 위한 순차 입력 생성 코드는 복원되어 있지 않다.**

## 7. 모델 구성

본 프로젝트는 아래 네 가지 모델을 비교 대상으로 둔다.

- `DummyClassifier`
  - 최소 기준선. 불균형 데이터에서 단순 기준선이 어느 정도 성능을 보이는지 확인한다.
- `LogisticRegression`
  - 해석 가능한 선형 baseline. 집계 특징만으로 어느 정도까지 위험을 탐지할 수 있는지 본다.
- `MLP`
  - 집계 특징 기반 비선형 모델. 핵심 질문은 "집계 특징만으로 충분한가?"이다.
- `LSTM`
  - 순차 패턴 기반 모델. 핵심 질문은 "주문 간격의 순서 정보가 실제로 추가 이득을 주는가?"이다.

발표의 핵심 비교 질문은 다음과 같다.

> **집계 특징만으로 충분한가, 아니면 순차 정보가 실제 예측에 도움이 되는가?**

## 8. 실험 설정

현재 저장소에는 실제 split 산출물이 포함되어 있지 않으므로, 발표용 재현 구조는 아래 기준으로 정리한다.

- split 단위: 사용자 단위
- split 방식: `train / val / test = 70 / 15 / 15`
- 분할 원칙: `delay_risk` 비율을 유지하는 stratified split 권장
- threshold tuning:
  - `DummyClassifier`는 고정 기준선으로 둔다.
  - `LogisticRegression`, `MLP`, `LSTM`은 **동일하게 validation set에서 threshold tuning**을 수행한다.

평가 지표는 accuracy보다 아래 지표를 우선한다.

- `Recall`: 위험 고객을 놓치지 않는 정도
- `F1-score`: precision과 recall의 균형
- `ROC-AUC`: threshold에 덜 의존하는 분리 성능
- `Average Precision` 또는 `PR curve`: 양성 클래스 탐지 품질

정리하면, 이 프로젝트는 불균형 데이터이므로 **accuracy 단독 비교를 피하고 Recall / F1 / ROC-AUC / AP를 중심으로 해석**해야 한다.

## 9. 결과 요약

현재 저장소에는 모델별 최종 metric 산출물이 포함되어 있지 않다. 따라서 발표 자료에서는 아래 순서로 결과를 정리하는 것이 좋다.

- 먼저 class imbalance와 positive ratio를 설명한다.
- 다음으로 baseline인 `DummyClassifier`, `LogisticRegression`과 신경망 모델 `MLP`, `LSTM`을 함께 놓고 비교한다.
- 단일 지표 1등만 강조하지 말고, 지표별 장단점을 분리해서 해석한다.

권장 해석 예시는 아래와 같다.

- `MLP`는 ROC-AUC가 상대적으로 높아 집계 특징 기반의 분리력에 강점이 있을 수 있다.
- `LogisticRegression`은 Recall이 높다면, 위험 고객을 넓게 포착하는 기준선으로 의미가 있다.
- `LSTM`은 F1-score가 가장 균형적일 경우, 순차 패턴을 활용한 최종 후보로 설명할 수 있다.

현재 저장소에는 실제 수치가 없으므로, 이 문구는 **해석 원칙**으로만 사용하고 최종 발표 수치는 실행 후 [`results/model_comparison_template.csv`](./results/model_comparison_template.csv)에 기록하는 것이 좋다.

## 10. 최종 결론

현재 보존된 산출물만으로는 어느 모델이 최종적으로 가장 적합한지 단정할 수 없다. 다만 프로젝트의 목적이 **위험 고객 탐지 우선**에 있다면, 최종 결론은 다음과 같이 신중하게 정리하는 것이 좋다.

> "Recall, F1-score, ROC-AUC를 함께 검토했을 때, 위험 고객 탐지를 우선하는 관점에서는 LSTM을 최종 후보로 고려할 수 있다. 다만 집계 특징만으로도 충분한 성능이 나온다면 MLP가 더 단순하고 재현 가능한 대안이 될 수 있다."

## 11. 한계와 향후 개선 방향

- 현재 저장소에는 원시 Instacart 데이터와 전처리 코드가 없다.
- `q90 = 15` 설명은 현재 가공 데이터와 충돌하므로 재검증이 필요하다.
- LSTM용 순차 입력 생성 코드와 실제 실험 결과가 빠져 있다.
- split 정보, threshold tuning 결과, test prediction 파일이 남아 있지 않아 재현성이 부족하다.

향후에는 아래 항목을 우선 보강하는 것이 좋다.

- 원시 Instacart 데이터에서 `01_prepare_data.ipynb`를 다시 만들고 threshold 근거를 자동으로 계산하기
- `results/` 폴더에 metric 요약 표와 threshold tuning 결과를 파일로 저장하기
- ROC curve와 PR curve를 함께 저장해 발표 자료에서 바로 재사용하기

## 12. 실행 방법

현재 저장소는 **발표용 구조 정리본**에 가깝다. 권장 실행 흐름은 아래와 같다.

1. [`01_prepare_data.ipynb`](./01_prepare_data.ipynb)에서 문제 정의, 시점 분리, split 규칙을 먼저 확인한다.
2. [`02_train_mlp.ipynb`](./02_train_mlp.ipynb)에서 `DummyClassifier`, `LogisticRegression`, `MLP` 실험을 정리한다.
3. [`03_train_lstm.ipynb`](./03_train_lstm.ipynb)에서 순차 모델 입력 정의와 LSTM 실험 조건을 맞춘다.
4. [`04_compare_models.ipynb`](./04_compare_models.ipynb)에서 threshold tuning 공정성과 최종 비교 기준을 점검한다.

## 13. 파일 구조

```text
.
|-- README.md
|-- 01_prepare_data.ipynb
|-- 02_train_mlp.ipynb
|-- 03_train_lstm.ipynb
|-- 04_compare_models.ipynb
|-- model_base_hv.xlsx
|-- docs/
|   |-- repository_audit.md
|   `-- preprocessing_summary.md
`-- results/
    |-- README.md
    `-- model_comparison_template.csv
```

## 발표 준비 메모

발표 슬라이드에는 최소한 아래 숫자와 그래프가 들어가는 것이 좋다.

- high-frequency threshold = 24
- 운영 기준 delay threshold = 15
- 현재 가공 데이터 기준 positive ratio = 18.65%
- split 방식 = user-level stratified train / val / test
- ROC curve
- PR curve
- 최종 모델 비교 표

세부 근거와 현재 불일치 사항은 [`docs/preprocessing_summary.md`](./docs/preprocessing_summary.md)와 [`docs/repository_audit.md`](./docs/repository_audit.md)에 정리했다.
