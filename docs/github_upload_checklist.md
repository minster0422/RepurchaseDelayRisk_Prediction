# GitHub 업로드 체크리스트

이 문서는 현재 작업 폴더를 나중에 GitHub 저장소로 정리할 때 기준으로 삼기 위한 메모다.

## 1. 업로드 전 핵심 원칙

- GitHub에는 **재현 가능한 코드, 문서, 핵심 결과 요약**을 중심으로 올린다.
- 큰 원본 데이터, 중간 zip, 영상, PowerPoint 미리보기 PNG, 모델 weight는 기본적으로 제외한다.
- `model_base_hv.csv`는 현재 모델링의 기준 데이터이지만, 공개 업로드 여부는 데이터 크기와 라이선스를 확인한 뒤 결정한다.
- 최종 README는 “중간발표 기록”이 아니라 “최종 프로젝트 설명서”로 유지한다.

## 2. GitHub에 포함하면 좋은 파일

| 구분 | 파일/폴더 | 이유 |
| --- | --- | --- |
| 프로젝트 설명 | `README.md` | 문제 정의, 데이터, 모델, 결과를 한 번에 설명 |
| 전처리 근거 | `docs/preprocessing_summary.md` | threshold, positive ratio, split 근거 |
| 최종 요약 | `docs/final_project_summary.md` | 발표 이후 기준의 짧은 프로젝트 요약 |
| 모델 피드백 | `docs/model_design_feedback.md` | 교수님 피드백과 개선 설계 기록 |
| 실험 코드 | `train_tabular_baselines.py`, `train_deep_sequence_models.py`, `train_hybrid_deep_models.py` | 주요 실험 재현 코드 |
| 분석 코드 | `analyze_hybridgru_errors.py`, `feature_importance.py`, `sensitivity_delay_threshold.py` | 오류 분석, feature 해석, 라벨 민감도 |
| 결과 요약 | `results/final_model_comparison.csv`, `results/final_model_comparison_summary.md` | 최종 모델 비교 결과 |
| 시각화 | `results/final_model_comparison.png`, `results/final_selected_roc_curve.png`, `results/final_selected_pr_curve.png` | README와 보고서용 핵심 그림 |
| 발표 자료 | `presentation/final_repurchase_delay_presentation_v14.pdf` | 최종 발표 산출물 |

## 3. 기본적으로 제외할 파일

- `incoming_inspect/`
- `repo_clone/`
- `__pycache__/`
- `catboost_info/`
- `*.zip`
- `*.mp4`
- `results/**/*.pt`
- `presentation/final_preview*/`
- 과거 발표 버전 `presentation/final_repurchase_delay_presentation_v*.pptx`
- 과거 대본 버전 `presentation/final_presentation_speaker_script_v*.docx`

## 4. README에 반드시 남길 핵심 숫자

- 전체 고객 수: 206,209명
- 전체 주문 수: 3,421,083건
- 최종 모델링 샘플 수: 42,499명
- 고빈도 고객 기준: 총 주문 횟수 상위 20%, threshold 24회
- 라벨 기준: `target_gap > 15`
- 분위수: q80 = 15, q90 = 24, q95 = 30
- positive ratio: 18.65%
- split: train/validation/test = 70/15/15
- split sample count: 29,749 / 6,375 / 6,375
- 최종 후보: HybridGRU
- HybridGRU 성능: Precision 0.4116, Recall 0.6972, F1 0.5176, ROC-AUC 0.8220, AP 0.4737

## 5. GitHub 정리 순서

1. 현재 작업 폴더에서 필요한 파일만 새 저장소 폴더로 복사한다.
2. `README.md`, `docs/`, `results/`의 숫자와 표현이 서로 맞는지 확인한다.
3. `.gitignore`가 큰 파일과 중간 산출물을 제외하는지 확인한다.
4. `requirements.txt`를 실제 실행 코드 기준으로 업데이트한다.
5. `python train_tabular_baselines.py`처럼 최소 1개 이상의 재현 명령이 동작하는지 확인한다.
6. GitHub 업로드 후 README의 이미지 링크가 깨지지 않는지 확인한다.

## 6. 업로드 전 남은 TODO

- `requirements.txt`에 PyTorch, LightGBM, XGBoost, CatBoost 등 실제 사용 패키지를 정확히 반영하기
- `model_base_hv.csv`를 공개할지, 샘플 데이터만 공개할지 결정하기
- 교수님 피드백을 반영한 auxiliary-loss HybridGRU를 별도 실험으로 추가할지 결정하기
- `README.md`의 파일 구조를 실제 GitHub 업로드 구조 기준으로 다시 축약하기
