# 현재 저장소 audit 결과

## 현재 잘된 점

- [`model_base_hv.csv`](../model_base_hv.csv)가 저장소 기준 데이터로 정리돼 있다.
- `delay_risk` 라벨은 현재 CSV에서 `target_gap > 15` 규칙과 정확히 일치한다.
- `target_order_number` 최소값이 `24`, `total_orders_before_target` 최소값이 `23`이라서 고빈도 고객 정의와 현재 샘플 구조가 자연스럽게 이어진다.
- `README.md`, `docs/`, `results/`, 노트북 4개라는 발표용 구조 자체는 이미 잘 잡혀 있다.

## 현재 문제점

- 루트 README와 일부 노트북, `results/README.md`에 한글 깨짐 또는 어색한 표현이 남아 있었다.
- `q90 ≈ 15일` 기획 문구와 현재 CSV 재집계 결과(`q80 = 15`, `q90 = 24`, `q95 = 30`) 사이의 충돌 설명이 더 정교하게 정리될 필요가 있었다.
- `03_train_lstm.ipynb`는 sequence 입력 생성 코드가 없어, 현재 실행 가능한 수준과 설계/가이드 수준의 구분이 더 분명해야 했다.
- `results/` 폴더는 템플릿 구조는 괜찮지만, 실제 실행 후 어떤 파일이 추가돼야 하는지 안내가 더 필요했다.

## 반드시 수정해야 할 것

- 한글 깨짐과 어색한 문장을 자연스러운 한국어로 복구하기
- `15일 기준`과 `현재 재집계 분위수`의 충돌을 정직하고 일관되게 설명하기
- 고빈도 고객 정의를 `상위 20%`, `threshold 24회` 기준으로 문서 전반에 통일하기
- 노트북별 현재 역할과 한계를 더 명확하게 쓰기
- 결과 템플릿은 유지하되, 아직 실제 수치는 비어 있다는 점을 분명히 밝히기

## 있으면 좋은 개선점

- `results/final_metrics.csv`에 최종 지표 저장
- `results/roc_curve.png`, `results/pr_curve.png` 저장
- `results/confusion_matrix_mlp.png`, `results/confusion_matrix_lstm.png` 저장
- `results/test_predictions_*.csv` 저장

## 현재 확인 가능한 핵심 숫자

| 항목 | 값 |
| --- | --- |
| 행 수 | 42,499 |
| 열 수 | 18 |
| 양성 수 | 7,926 |
| 양성 비율 | 18.65% |
| `target_order_number` 최소값 | 24 |
| `total_orders_before_target` 최소값 | 23 |
| `target_gap` q80 | 15 |
| `target_gap` q90 | 24 |
| `target_gap` q95 | 30 |

## 해석상 주의점

- 현재 저장소에서 사실처럼 말할 수 있는 것은 `target_gap > 15` 운영 라벨 기준과 현재 CSV 재집계 결과다.
- 따라서 **15일 기준은 현재 가공 데이터에서 q80 수준**이며, **기존 `q90 ≈ 15일` 설명은 재검증 필요**라고 정리하는 것이 가장 안전하다.
