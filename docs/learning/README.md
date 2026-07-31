# 4주 OpenCV 재구현·설명 학습 팩

대상은 코드를 읽을 수 있지만, 왜 그 코드를 선택했는지 설명하고 빈 파일에서 다시 조립하는 연습이 필요한 초급자입니다. 하루 30분, 4주 28일로 진행합니다. 결과를 외우기보다 **입력 → 진단 → 선택 → 검증 → 한계** 순서로 말하는 습관을 만듭니다.

## 사용하는 근거

- 프로젝트 흐름: [evidence map](../portfolio/evidence-map.md), [case study](../portfolio/case-study.md)
- 전처리 구현: [transforms.py](../../src/opencv_preprocessing_advisor/transforms.py), [diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py)
- 특징·분류·평가: [features.py](../../src/opencv_preprocessing_advisor/features.py), [classifiers.py](../../src/opencv_preprocessing_advisor/classifiers.py), [evaluation.py](../../src/opencv_preprocessing_advisor/evaluation.py)
- 실행 설정과 결과: [pipelines.yaml](../../src/opencv_preprocessing_advisor/config/pipelines.yaml), [experiment results](../portfolio/experiment-results.md)

## 사용 순서

1. 해당 주의 Day를 열고 `5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명`을 지킵니다.
2. 실습 뒤에는 결과 이미지·수치·파라미터를 한 줄로 기록합니다. 이미지 한 장의 휴리스틱 점수와 데이터셋 분류 평가는 같은 값이 아닙니다.
3. [실습 24개](exercises.md)는 E01부터 풀고, [면접 Q&A 32개](interview-qa.md)는 답을 보기 전 60초간 소리 내어 답합니다.
4. [진도 체크리스트](progress-checklist.md)에서 그날의 구현, 관찰, 설명을 모두 했을 때만 체크합니다.

## 주차별 산출물

| 주 | 핵심 질문 | 주말 산출물 |
| --- | --- | --- |
| 1 | 배열과 색 공간을 어떻게 안전하게 읽는가? | Unicode 경로를 포함한 작은 이미지 검사기 |
| 2 | 어떤 진단으로 전처리를 선택하는가? | 전처리 전후 비교와 진단 메모 |
| 3 | 특징·분류·평가를 어떻게 분리해 검증하는가? | 누수 없는 비교 결과 해설 |
| 4 | 프로젝트를 어떻게 설명·재구현·방어하는가? | 5분 발표, 실패 분석, 자기평가 |

다음 문서: [1주차](week-01-image-foundations.md) · [2주차](week-02-preprocessing-diagnostics.md) · [3주차](week-03-features-classifiers-evaluation.md) · [4주차](week-04-explanation-reimplementation.md)
