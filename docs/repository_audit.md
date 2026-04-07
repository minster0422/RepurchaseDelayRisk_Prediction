# 저장소 audit 요약

## 한눈에 보기

현재 작업 폴더는 완성된 GitHub 저장소가 아니라, **문서 2개와 가공 데이터 1개만 남아 있는 상태**다.

- `instacart_readme_revision_summary.docx`
- `instacart_readme_revision_summary.pdf`
- `model_base_hv.xlsx`

`.git`, `README.md`, 실험 노트북, 결과 요약 파일이 모두 없었기 때문에, 우선 발표용 저장소 구조를 새로 정리해야 했다.

## 현재 잘된 점

- `model_base_hv.xlsx`에 고빈도 고객 집계 특징과 `delay_risk` 라벨이 들어 있다.
- `target_gap > 15` 규칙이 현재 가공 데이터에 실제로 반영되어 있다.
- `target_order_number`의 최소값이 24이므로, 고빈도 고객 threshold = 24라는 발표 서사와 크게 어긋나지 않는다.
- 별도 문서에 "고가치 고객 churn"보다 "고빈도 고객의 재구매 지연 위험"이라는 표현이 더 적절하다는 정리가 이미 존재한다.

## 현재 문제점

- 실제 Git 저장소가 아니다.
- README가 없었다.
- `01_prepare_data.ipynb`, `02_train_mlp.ipynb`, `03_train_lstm.ipynb`, `04_compare_models.ipynb`가 없었다.
- 실험 결과 요약 파일이 모두 없다.
- LSTM용 순차 입력 생성 코드가 없다.
- 원시 Instacart 데이터 경로와 전처리 코드가 없다.

## 현재 확인된 사실

아래 수치는 현재 저장소에 남아 있는 `model_base_hv.xlsx`만을 기준으로 재확인한 값이다.

| 항목 | 현재 확인각 |
| --- | --- |
| 행 수 | 42,499 |
| 열 수 | 18 |
| 사용자당 행 수 | 1행 |
| 양성 비율 (`delay_risk = 1`) | 약 18.65% |
| `target_order_number` 최소값 | 24 |
| `target_order_number` 최대값 | 100 |
| 현재 확인된 라벨 규칙 | `target_gap > 15 -> delay_risk = 1` |
| `target_gap` q80 | 15 |
| `target_gap` q90 | 24 |
| `target_gap` q95 | 30 |

## 가장 중요한 불일치

기존 발표 기획 문구에는 `q90 ≈ 15일` 설몌이 들어 있지만, 현재 가공 데이터 기준 재집계 결과는 **q80 = 15, q90 = 24**다.

따라서 다음 두 문장은 현재 상태에서 동시에 사실일 수 없다.

- "threshold 15는 q90 근거다"
- "현재 `model_base_hv.xlsx`가 최종 전처리 산출물이다"

이 불일치는 발표 전에 반드시 정리해야 한다. 가장 안전한 방법은 다음 둘 둑 중하나다.

1. 원본 전처리 코드로 분위수를 다시 계산해 `q90 = 15`를 재검증한다.
2. 현재 산출물을 기준으로 README와 슬라인드를 수정하고, `15일`은 운영 기준 또는 `q80` 기반 설몙으로 정리한다.

## 반드시 수정해야 할 항목

- README를 발표 구조에 맞춰 새로 작성하기
- 노트북 4개의 역할과 실행 흐름을 명시하기
- threshold 근거와 split 규칙을 source of truth 문서로 분리하기
- 결과 표와 threshold tuning 기록을 `results/`에 남길 구조 만들기

## 있으면 좋은 개선점

- `results/thresholds.json`에 모델별 validation threshold 저장
- `results/test_predictions_*.csv`에 test prediction 저장
- `results/roc_curve.png`, `results/pr_curve.png` 저장
- `docs/preprocessing_summary.md`를 전처리 기준 문서로 계속 유지
