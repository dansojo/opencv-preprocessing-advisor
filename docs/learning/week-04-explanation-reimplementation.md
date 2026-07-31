# 4주차 — 설명, 재구현, 방어

이 주는 새 API를 많이 외우는 주가 아닙니다. 이미 있는 [서비스 흐름](../../src/opencv_preprocessing_advisor/services.py), [보고서](../../src/opencv_preprocessing_advisor/reports.py), [한계](../portfolio/limitations.md)를 근거로 의사결정을 재구성합니다.

## Day 22 — 1분 프로젝트 설명

시간: **5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명**

### 목표
문제·입력·출력·검증·한계를 포함한 1분 설명을 만든다.

### 개념
좋은 짧은 설명은 기능 목록이 아니라 흐름이다: BGR 이미지 진단, 전처리 후보의 설명 가능한 추천, 그리고 별도 교차검증 기반 데이터셋 비교. 단일 이미지 추천 점수는 정확도가 아니다.

### 코드 연결
[ImageAdvisorService](../../src/opencv_preprocessing_advisor/services.py)와 [case study](../portfolio/case-study.md)를 읽는다.

### 실습
60초 타이머로 초안을 녹음한다. “휴리스틱”, “교차검증”, “한계”가 각각 한 번씩 들어갔는지 표시한다.

### 말로 설명
“이 프로젝트는 전처리 후보를 투명하게 추천하고, 분류 성능은 별도의 데이터셋 프로토콜로 평가한다”라고 1분 내 말한다.

## Day 23 — 5분 프로젝트 설명

시간: **5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명**

### 목표
5분 발표를 진단·변환·특징·평가·실패 순서로 확장한다.

### 개념
발표의 각 주장에는 코드 또는 결과 근거가 필요하다. 숫자는 재현된 프로토콜 안에서만 말하고, MVTec의 공식 anomaly metric을 주장하지 않는다.

### 코드 연결
[experiment results](../portfolio/experiment-results.md), [limitations](../portfolio/limitations.md), [evidence map](../portfolio/evidence-map.md)을 읽는다.

### 실습
5개 슬라이드 없는 목차를 만든다: 문제, 진단/추천, 구현, 평가, 한계/다음 실험. 각 항목에 링크 하나를 붙인다.

### 말로 설명
“결과를 말한 뒤 반드시 데이터셋, 분할, 지표, 한계를 연결한다”는 규칙으로 5분 발표를 한다.

## Day 24 — 전처리 파이프라인 재구현

시간: **5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명**

### 목표
CLAHE 또는 filter 기반 파이프라인 하나를 빈 파일에서 조립한다.

### 개념
재구현은 입력 검증, 변환, 반환 dtype, 파라미터 검증, 전후 관찰의 순서다. 설정 파일은 후보 이름과 파라미터를 재현 가능하게 남긴다.

### 코드 연결
[TRANSFORMS](../../src/opencv_preprocessing_advisor/transforms.py), [pipelines.yaml](../../src/opencv_preprocessing_advisor/config/pipelines.yaml), [변환 테스트](../../tests/test_transforms.py)를 읽는다.

### 실습
문서를 덮고 `apply_lab_clahe`의 의사코드를 쓴 뒤 구현과 대조한다. L/a/b 중 어느 채널을 바꾸는지와 유효하지 않은 파라미터 하나를 적는다.

### 말로 설명
“재구현의 완료 조건은 이미지가 나오는 것이 아니라 입력 계약과 파라미터 실패 조건까지 재현하는 것”이라고 말한다.

## Day 25 — 진단 하나 추가하기

시간: **5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명**

### 목표
기존 추천을 과장하지 않는 보조 진단 하나를 설계한다.

### 개념
진단은 관찰값을 제공하고 추천의 근거를 보강한다. 새 점수는 정의, 범위, 예외 입력, 테스트 이미지, 오판 가능성을 먼저 적어야 한다.

### 코드 연결
[diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py), [scoring.py](../../src/opencv_preprocessing_advisor/scoring.py), [진단 테스트](../../tests/test_diagnostics.py)를 읽는다.

### 실습
‘어두운 픽셀 비율’ 또는 ‘큰 연결요소 비율’ 중 하나를 정의한다. 정상·극단 입력에서 기대값을 세 줄의 테스트 표로 쓴다.

### 말로 설명
“새 진단은 정답 레이블이 아니라 관찰 가능한 신호다. 점수에 넣으면 가중치와 실패 사례를 공개한다”라고 말한다.

## Day 26 — 두 파라미터 설정 비교

시간: **5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명**

### 목표
한 변수만 바꾼 비교로 파라미터 효과를 설명한다.

### 개념
공정한 비교는 입력과 나머지 설정을 고정한다. 결과는 전후 이미지뿐 아니라 진단, 특징, 평가 중 어떤 층에서 관찰했는지 밝혀야 한다.

### 코드 연결
[pipeline config](../../src/opencv_preprocessing_advisor/config/pipelines.yaml), [reports.py](../../src/opencv_preprocessing_advisor/reports.py), [실험 결과](../portfolio/experiment-results.md)를 읽는다.

### 실습
CLAHE clip limit 1.0과 4.0 또는 Gaussian kernel 3과 9를 비교한다. ‘더 좋아 보임’ 대신 관찰값 두 개와 부작용 한 개를 표로 쓴다.

### 말로 설명
“파라미터 비교는 한 번의 보기 좋은 결과가 아니라 고정한 조건과 측정 층을 함께 보고한다”라고 말한다.

## Day 27 — 실패 사례 진단

시간: **5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명**

### 목표
실패를 입력·전처리·특징·분류·평가 중 어느 단계의 가설로 분해한다.

### 개념
오분류는 하나의 원인이 아닐 수 있다. 혼동행렬, 샘플 관찰, 파라미터, feature shape, fold 규칙을 순서대로 확인하면 추측을 검증 가능한 가설로 바꾼다.

### 코드 연결
[confusion_matrix](../../src/opencv_preprocessing_advisor/evaluation.py), [reports.py](../../src/opencv_preprocessing_advisor/reports.py), [한계](../portfolio/limitations.md)를 읽는다.

### 실습
혼동행렬의 off-diagonal 셀 하나를 고른 뒤 가능한 원인을 세 계층으로 쓴다. 각 원인을 확인할 코드·산출물 링크를 붙인다.

### 말로 설명
“실패는 숨길 결과가 아니라 다음 실험을 설계하는 증거다. 관찰된 사실과 원인 가설을 구분한다”라고 말한다.

## Day 28 — 모의 면접과 자기평가

시간: **5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명**

### 목표
근거 링크를 보지 않고 질문에 답하고, 모르는 부분을 다음 학습 항목으로 바꾼다.

### 개념
설명 가능한 실력은 API 이름을 많이 말하는 것이 아니라 ‘왜/언제 피함/파라미터/근거’를 연결하는 능력이다. 불확실하면 사실과 가설을 분리해 답한다.

### 코드 연결
[면접 Q&A](interview-qa.md), [실습](exercises.md), [진도표](progress-checklist.md)를 사용하고 [전체 테스트 목록](../../tests/test_evaluation.py)도 확인한다.

### 실습
[Q01](interview-qa.md#q01-bgr-rgb)의 답을 보지 않고 녹음하고, E21 또는 E24를 빈 파일에서 다시 시작한다. 빠진 근거는 체크리스트에 ‘재학습’으로 적는다.

### 말로 설명
“나는 결과와 근거를 분리하고, 확인하지 않은 일반화는 하지 않는다”라고 말한 뒤 4주 중 다시 할 Day를 하나 고른다.
