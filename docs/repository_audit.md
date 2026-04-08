# 현재 저장소 audit 결과

## 현재 잘된 점

- [`model_base_hv.csv`](../model_base_hv.csv)는 발표용 기준 데이터로 바로 사용할 수 있을 만큼 열 구성이 정리돼 있다.
- `delay_risk` 라벨은 현재 CSV에서 `target_gap > 15` 규칙과 정확히 일치한다.
- `target_order_number`의 최소값이 `24`라서, high-frequency threshold = 24 서사와는 맞는다.
- 양성 비율이 약 `18.65%`라서, 불균형 데이터 해석의 필요성을 분명하게 보여줄 수 있다.

## 현재 문제점

- 이전 정리 과정에서 README와 노트북이 다시 인코딩 이슈를 겪어 발표용 문서 품질이 불안정했다.
- 저장소 설명이 `model_base_hv.xlsx` 기준으로 남아 있었고, 이번에 제공된 CSV와 맞지 않았다.
- `requirements.txt`는 비어 있었고, 저장소 흐름에 도움이 되지 않았다.
- LSTM용 sequence 입력 생성 코드와 결과 파일이 없어, 순차 모델 비교는 아직 설계 문서 수준이다.
- `q90 ≈ 15일` 설명은 현재 CSV 재계산값과 일치하지 않는다.

## 반드시 수정해야 할 것

- 저장소의 기준 데이터를 `model_base_hv.csv`로 통일하기
- README와 4개 노트북을 CSV 기준으로 다시 작성하기
- `feature` / `label` 시점 분리와 threshold 정의를 문서에서 더 명확히 드러내기
- `LogisticRegression`, `MLP`, `LSTM`의 threshold tuning 공정성 기준을 동일하게 정리하기
- 결과 해석에서 accuracy 중심 설명을 피하고 Recall / F1 / ROC-AUC / AP 중심으로 재정리하기

## 있으면 좋은 개선점

- `results/thresholds.json`에 validation threshold 저장
- `results/test_predictions_*.csv`에 모델별 test prediction 저장
- `results/roc_curve.png`, `results/pr_curve.png` 저장
- 발표 슬라이드용 핵심 숫자와 그래프 체크리스트를 별도 문서로 유지

## 실제 점검 결과

현재 CSV 기준 주요 수치는 다음과 같다.

| 항목 | 값 |
| --- | --- |
| 행 수 | 42,499 |
| 열 수 | 18 |
| 양성 수 | 7,926 |
| 양성 비율 | 18.65% |
| `target_order_number` 최소값 | 24 |
| `target_order_number` 최대값 | 100 |
| `target_gap` q80 | 15 |
| `target_gap` q90 | 24 |
| `target_gap` q95 | 30 |

## 가장 중요한 해석상 이슈

현재 CSV를 기준으로 하면 `target_gap > 15` 규칙은 맞지만, `q90 ≈ 15일`은 재현되지 않는다. 따라서 발표에서는 아래처럼 분리해서 설명하는 것이 안전하다.

- **현재 저장소 기준 사실**: `target_gap > 15 -> delay_risk = 1`, q80 = 15, q90 = 24
- **발표 기획 기준 서사**: `15일`을 지연 위험 기준으로 두고, 기존 전처리 단계에서는 q90 근거였다고 설명

즉, 이 부분은 숨기기보다 `NOTE`로 명시하는 편이 더 설득력 있다.
