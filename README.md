# Repurchase Delay Risk Prediction

## 1. 프로젝트 개요

이 프로젝트는 Instacart 주문 이력을 바탕으로, **고빈도 고객의 다음 주문이 평소보다 오래 지연될 위험**을 예측하는 것을 목표로 한다. 발표 서사는 단순 분류가 아니라, **과거 주문 이력으로 다음 주문의 지연 위험을 예측하는 프로젝트**로 구성한다.

현재 저장소는 발표용 구조 정리본과 일부 결과 템플릿을 포함한 상태이며, 기준 데이터는 [`model_base_hv.csv`](./model_base_hv.csv)다.

## 2. 문제 정의

핵심 질문은 다음과 같다.

> 과거 주문 이력만으로, 고빈도 고객의 **다음 주문이 지연 위험 상태에 들어갈지** 예측할 수 있는가?

이 프로젝트가 예측 프로젝트처럼 보이려면 아래 원칙이 분명해야 한다.

- `feature`와 `label`의 시점을 분리한다.
- `target` 주문은 라벨 생성용으로만 사용한다.
- 입력 특징은 반드시 **target 주문 이전 이력만** 사용한다.

즉, 현재 상태를 설명하는 분류가 아니라 **다음 주문 지연 위험을 미리 탐지하는 예측 구조**로 해석해야 한다.

## 3. 왜 Instacart 데이터인가

Instacart는 반복 구매가 많은 식료품 주문 데이터이기 때문에, 고객별 주문 간격과 재구매 습관을 비교적 자연스럽게 관찰할 수 있다. 특히 `days_since_prior_order`에서 파생된 간격 정보는 재구매 지연 위험 문제를 설계하기에 적합하다.

다만 Instacart에는 직접적인 구매 금액 정보가 없으므로, 본 프로젝트는 "고가치 고객"보다 **고빈도 고객**이라는 표현이 더 정확하다.

## 4. 고빈도 고객 정의

문서 전반에서 사용하는 표현은 다음과 같이 통일한다.

- 고빈도 고객은 전체 고객 중 **총 주문 횟수 상위 20%**
- 현재 가공 데이터 기준 실제 threshold는 **24회**

현재 [`model_base_hv.csv`](./model_base_hv.csv)에서 `target_order_number`의 최소값은 `24`이고, `total_orders_before_target`의 최소값은 `23`이다. 즉, 현재 샘플은 최소 24번째 주문을 target으로 삼은 고빈도 고객 집단으로 해석하는 것이 자연스럽다.

## 5. 타깃 정의

현재 프로젝트의 타깃은 **절대 기준형 이진 분류 방식**이다.

- `target_gap > 15` 이면 `delay_risk = 1`
- `target_gap <= 15` 이면 `delay_risk = 0`

발표 기획에서는 `15일`을 고빈도 고객 집단의 지연 기준으로 두고, 이를 `q90 ≈ 15일` 서사와 연결해 설명하려고 했다.

`NOTE`

- 현재 [`model_base_hv.csv`](./model_base_hv.csv)를 재집계하면 `target_gap` 분위수는 `q80 = 15`, `q90 = 24`, `q95 = 30`이다.
- 따라서 **15일 기준은 현재 가공 데이터에서 q80 수준**으로 해석하는 것이 맞고, **기존의 `q90 ≈ 15일` 설명은 재검증이 필요하다.**
- 즉, 현재 저장소에서 사실처럼 말할 수 있는 것은 `target_gap > 15` 운영 기준과 재집계 분위수 결과이며, `q90 ≈ 15일`은 초기 기획 설명으로만 다루는 편이 안전하다.

## 6. 데이터 전처리 개요

전처리의 핵심 원칙은 아래와 같다.

- 고객별로 하나의 `target` 주문을 잡는다.
- `label`은 그 target 주문의 `target_gap`으로 만든다.
- `feature`는 target 주문 이전 이력만 사용해 만든다.
- target 이후 정보는 어떤 형태로도 입력에 포함하지 않는다.

현재 [`model_base_hv.csv`](./model_base_hv.csv)는 **사용자당 1행의 집계 특징 테이블**이다.

- 샘플 수: `42,499`
- 특징 수: `18`개 열
- 양성 비율: `18.65%`
- 주요 열: `avg_gap_before_target`, `std_gap_before_target`, `recent_3_avg_gap`, `recent_5_avg_gap`, `last_gap_before_target`, `gap_trend`, `order_frequency`, `weekend_order_ratio`

즉, 현재 저장소는 **집계 특징 기반 MLP / LogisticRegression / DummyClassifier 실험에는 바로 연결 가능**하지만, **LSTM용 순차 입력 생성 파이프라인은 추가 보강이 필요하다.**

## 7. 모델 구성

비교 대상 모델은 다음 네 가지다.

- `DummyClassifier`
  - 최소 기준선이다.
- `LogisticRegression`
  - 해석 가능한 선형 baseline이다.
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

불균형 데이터이므로 accuracy만으로 결론을 내리기보다 아래 지표를 함께 보는 것이 적절하다.

- `Recall`
- `F1-score`
- `ROC-AUC`
- `Average Precision`
- `PR curve`

## 9. 결과 요약

현재 저장소에는 최종 실험 수치가 아직 채워져 있지 않다. 따라서 결과 해석은 **원칙 중심**으로 정리하고, 실제 수치는 재실행 후 템플릿에 채우는 방식이 적절하다.

현재 확인 가능한 상태:

- high-frequency threshold = `24`
- delay threshold = `15`
- positive ratio = `18.65%`
- `target_gap` 재집계 분위수 = `q80 15 / q90 24 / q95 30`
- 집계 특징 기반 비교 구조와 결과 템플릿

재실행 후 채워야 할 항목:

- 최종 metric 표
- threshold tuning 결과
- ROC curve / PR curve
- confusion matrix
- 모델별 test prediction 파일

## 10. 최종 결론

실제 결과가 채워지기 전에는 최종 모델을 단정적으로 확정하지 않는 것이 안전하다. 다만 발표용 결론 문구는 아래처럼 정리할 수 있다.

> 위험 고객 탐지를 우선하는 관점에서는 LSTM을 최종 후보로 검토할 수 있다. 다만 집계 특징만으로도 충분한 성능이 나온다면 MLP가 더 단순하고 재현 가능한 대안이 될 수 있다.

## 11. 한계와 향후 개선 방향

- 현재 저장소에는 원시 Instacart 전처리 코드가 없다.
- `q90 ≈ 15일` 설명은 현재 CSV 재집계 결과와 다르므로 재검증이 필요하다.
- LSTM용 sequence 입력 생성 코드와 실제 결과 파일이 없다.
- split 결과, threshold tuning 기록, test prediction 파일이 아직 저장돼 있지 않다.

향후에는 아래 항목을 보강하는 것이 좋다.

- 원시 Instacart 데이터 기준으로 threshold 근거를 다시 계산하기
- sequence pipeline을 보강해 LSTM 입력 생성을 재현하기
- `results/` 폴더에 실제 metric, ROC/PR curve, confusion matrix를 저장하기

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

## 14. 발표 준비 메모

- 한 장으로 요약하면, 이 프로젝트는 **고빈도 고객의 다음 주문 지연 위험 예측**이다.
- 발표에서는 `15일 기준`과 `현재 재집계 분위수`를 분리해서 설명하는 것이 중요하다.
- 안전한 표현은 다음과 같다.
  - 운영 라벨 기준은 `target_gap > 15`
  - 현재 가공 데이터 재집계 결과는 `q80 = 15`, `q90 = 24`, `q95 = 30`
  - 따라서 **15일 기준은 현재 데이터에서 q80 수준**이며, 기존 `q90 ≈ 15일` 설명은 재검증이 필요하다.
- 모델 비교 파트에서는 "`MLP`는 집계 특징 기반, `LSTM`은 순차 정보 기반"이라는 대비를 먼저 제시하는 편이 이해하기 쉽다.
- 실제 결과가 비어 있는 항목은 비워 둔 이유를 숨기지 말고, "발표용 구조와 재현 경로는 정리되었고, 최종 수치는 재실행 후 채워야 한다"는 식으로 설명하는 것이 적절하다.
