# Repurchase Delay Risk Prediction

## 1. 프로젝트 개요

이 프로젝트는 Instacart 주문 이력을 바탕으로, **고빈도 고객의 다음 주문이 평소보다 오래 지연될 위험**을 예측하는 것을 목표로 한다. 발표 서사는 단순 분류가 아니라, **과거 이력으로 다음 주문의 지연 위험을 예측하는 프로젝트**로 구성한다.

현재 저장소의 기준 데이터는 [`model_base_hv.csv`](./model_base_hv.csv)다. 이 파일은 이미 고객별 집계 특징과 타깃 라벨이 정리된 전처리 산출물이며, 본 저장소는 이 CSV를 중심으로 발표용 구조를 다시 정돈한 버전이다.

## 2. 문제 정의

핵심 질문은 다음과 같다.

> 과거 주문 이력만으로, 고빈도 고객의 **다음 주문이 지연 위험 상태에 들어갈지** 예측할 수 있는가?

이 프로젝트가 예측 프로젝트처럼 보이려면 아래 원칙이 반드시 드러나야 한다.

- `feature`와 `label`의 시점을 분리한다.
- `target` 주문은 라벨 생성용으로만 사용한다.
- 입력 특징은 반드시 **target 주문 이전 이력만** 사용한다.

즉, 현재 주문을 설명하는 분류가 아니라 **다음 주문의 지연 위험을 미리 탐지하는 예측 구조**로 해석해야 한다.

## 3. 왜 Instacart 데이터인가

Instacart는 반복 구매가 많은 식료품 주문 데이터이기 때문에, 고객별 재구매 간격과 주문 습관을 비교적 자연스럽게 관찰할 수 있다. 특히 `days_since_prior_order`에서 파생된 주문 간격 정보를 통해, 재구매 지연 위험 문제를 설계하기에 적합하다.

다만 Instacart에는 직접적인 구매 금액 정보가 없으므로, 본 프로젝트는 "고가치 고객"보다 **고빈도 고객**이라는 표현이 더 정확하다.

## 4. 고빈도 고객 정의

발표용 기준 정의는 다음과 같다.

- 전체 고객 중 **총 주문 횟수 상위 20%**를 고빈도 고객으로 본다.
- 현재 프로젝트 설명에서는 이 경계가 **주문 횟수 24회**라고 본다.

현재 CSV를 확인하면 `target_order_number`의 최소값이 `24`이므로, 적어도 현재 전처리 산출물 단계에서는 **high-frequency threshold = 24**가 반영된 것으로 해석할 수 있다.

## 5. 타깃 정의

현재 프로젝트의 타깃은 **절대 기준형 이진 분류 방식**이다.

- `target_gap > 15` 이면 `delay_risk = 1`
- `target_gap <= 15` 이면 `delay_risk = 0`

발표 기획에서는 `15일`을 고빈도 고객 집단의 지연 기준으로 두고, 이를 `q90 ≈ 15일` 서사와 연결해 설명하려고 한다.

`NOTE`

- 현재 저장소의 [`model_base_hv.csv`](./model_base_hv.csv)를 다시 계산하면 `target_gap` 분위수는 `q80 = 15`, `q90 = 24`, `q95 = 30`으로 나타난다.
- 즉, **`target_gap > 15` 규칙 자체는 현재 CSV와 정확히 일치**하지만, **`q90 ≈ 15일` 설명은 현재 CSV만으로는 재현되지 않는다.**
- 따라서 발표에서는 `q90 ≈ 15일`을 **초기 전처리 기획 기준**으로 소개하고, 현재 CSV 재계산 결과와는 차이가 있어 원본 전처리 코드 재검증이 필요하다는 점을 함께 밝히는 편이 안전하다.

## 6. 데이터 전처리 개요

전처리의 핵심 원칙은 아래와 같다.

- 고객별로 하나의 `target` 주문을 잡는다.
- `label`은 그 target 주문의 `target_gap`으로 만든다.
- `feature`는 target 주문 이전 이력만 사용해 만든다.
- target 이후 정보는 어떤 형태로도 입력에 포함하지 않는다.

현재 [`model_base_hv.csv`](./model_base_hv.csv)는 **사용자당 1행의 집계 특징 테이블**이다.

- 샘플 수: `42,499`
- 특징 수: `18`개 열
- 양성 비율: 약 `18.65%`
- 주요 열: `avg_gap_before_target`, `std_gap_before_target`, `recent_3_avg_gap`, `recent_5_avg_gap`, `last_gap_before_target`, `gap_trend`, `order_frequency`, `weekend_order_ratio`

즉, 현재 저장소는 **집계 특징 기반 MLP / LogisticRegression / DummyClassifier 실험에는 바로 연결 가능**하지만, **LSTM을 위한 순차 입력 생성 코드는 별도로 복원해야 한다.**

## 7. 모델 구성

비교 대상 모델은 다음 네 가지다.

- `DummyClassifier`
  - 최소 기준선이다. 불균형 데이터에서 단순 기준선이 어느 정도 성능을 보이는지 확인한다.
- `LogisticRegression`
  - 해석 가능한 선형 baseline이다. 집계 특징만으로 어느 정도까지 위험을 탐지할 수 있는지 본다.
- `MLP`
  - 집계 특징 기반 비선형 모델이다.
- `LSTM`
  - 순차 패턴 기반 모델이다.

핵심 비교 질문은 다음과 같다.

> **집계 특징만으로 충분한가, 아니면 순차 정보가 실제 예측에 도움이 되는가?**

## 8. 실험 설정

발표용 실험 구조는 아래 기준으로 정리한다.

- split 단위: 사용자 단위
- split 방식: `train / val / test = 70 / 15 / 15`
- 분할 원칙: `delay_risk` 비율을 유지하는 stratified split 권장
- threshold tuning:
  - `DummyClassifier`는 고정 기준선으로 둔다.
  - `LogisticRegression`, `MLP`, `LSTM`은 **같은 validation set에서 threshold tuning**을 수행한다.

불균형 데이터이므로 accuracy만으로 해석하면 위험 고객 탐지 성능을 과소평가할 수 있다. 따라서 아래 지표를 중심으로 본다.

- `Recall`
- `F1-score`
- `ROC-AUC`
- `Average Precision` 또는 `PR curve`

## 9. 결과 요약

현재 저장소에는 최종 실험 수치가 아직 기록되어 있지 않다. 따라서 발표 자료에서는 **지표별 장단점이 드러나는 방식**으로 결과를 정리하는 것이 중요하다.

예를 들어 다음과 같은 해석이 가능하다.

- `MLP`는 ROC-AUC가 가장 높아 집계 특징 기반 분리력에 강점이 있을 수 있다.
- `LogisticRegression`은 Recall이 가장 높다면 위험 고객을 넓게 포착하는 기준선으로 의미가 있다.
- `LSTM`은 F1-score가 가장 균형적이라면 최종 후보로 검토할 수 있다.

즉, 특정 모델을 무조건 최고라고 쓰기보다 **지표별 역할과 trade-off를 함께 설명**하는 것이 발표 설득력에 더 적합하다.

## 10. 최종 결론

현재 구조에서 가장 안전한 최종 결론 문장은 다음과 같다.

> 위험 고객 탐지를 우선하는 관점에서는 LSTM을 최종 후보로 선정할 수 있다. 다만 집계 특징만으로도 충분한 성능이 나온다면 MLP가 더 단순하고 재현 가능한 대안이 될 수 있다.

## 11. 한계와 향후 개선 방향

- 현재 저장소에는 원시 Instacart 전처리 코드가 없다.
- `q90 ≈ 15일` 설명은 현재 CSV 재계산 결과와 다르므로 재검증이 필요하다.
- LSTM용 sequence 입력 생성 코드와 실제 결과 파일이 없다.
- split 결과, threshold tuning 기록, test prediction 파일이 아직 저장돼 있지 않다.

향후에는 아래 항목을 보강하는 것이 좋다.

- 원시 Instacart 데이터 기준으로 threshold 근거를 다시 계산하기
- `results/` 폴더에 threshold, prediction, ROC/PR curve를 파일로 남기기
- 발표 슬라이드에서 사용할 최종 비교 표를 test set 기준으로 고정하기

## 12. 실행 방법

권장 실행 순서는 다음과 같다.

1. `pip install -r requirements.txt`로 기본 패키지를 설치한다.
2. [`01_prepare_data.ipynb`](./01_prepare_data.ipynb)에서 시점 분리, threshold 정의, split 규칙을 확인한다.
3. [`02_train_mlp.ipynb`](./02_train_mlp.ipynb)에서 `DummyClassifier`, `LogisticRegression`, `MLP` 흐름을 정리한다.
4. [`03_train_lstm.ipynb`](./03_train_lstm.ipynb)에서 sequence 입력과 LSTM 비교 관점을 정리한다.
5. [`04_compare_models.ipynb`](./04_compare_models.ipynb)에서 threshold tuning 공정성과 결과 해석 원칙을 점검한다.

## 13. 파일 구조

```text
.
|-- README.md
|-- requirements.txt
|-- model_base_hv.csv
|-- 01_prepare_data.ipynb
|-- 02_train_mlp.ipynb
|-- 03_train_lstm.ipynb
|-- 04_compare_models.ipynb
|-- docs/
|   |-- repository_audit.md
|   `-- preprocessing_summary.md
`-- results/
    |-- README.md
    |-- model_comparison_template.csv
    `-- thresholds_template.json
```

## 발표용 체크 숫자

발표 자료에는 최소한 아래 숫자와 그래프가 들어가는 것이 좋다.

- high-frequency threshold = `24`
- delay threshold = `15`
- positive ratio = `18.65%`
- split 방식 = user-level stratified train / val / test
- ROC curve
- PR curve
- 최종 모델 비교 표

세부 audit와 전처리 기준은 [`docs/repository_audit.md`](./docs/repository_audit.md), [`docs/preprocessing_summary.md`](./docs/preprocessing_summary.md)에 정리했다.
