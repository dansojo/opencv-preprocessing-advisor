# 2주차 — 전처리 선택과 진단

매일 공통 시간: **5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명**. 이 주에는 효과를 주장하기 전에 진단값과 전후 이미지를 함께 남깁니다. [진단 코드](../../src/opencv_preprocessing_advisor/diagnostics.py)와 [파이프라인 설정](../../src/opencv_preprocessing_advisor/config/pipelines.yaml)을 기준으로 합니다.

## Day 8 — 밝기, 대비, 히스토그램

시간: **5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명**

### 목표
밝기·대비·히스토그램을 서로 다른 관찰값으로 설명한다.

### 개념
평균 밝기는 전체 수준, 표준편차나 분포 폭은 대비의 단서다. 히스토그램은 공간 위치를 잃으므로 이미지와 함께 읽어야 한다.

### 코드 연결
[diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py)의 밝기·대비 계산과 [진단 테스트](../../tests/test_diagnostics.py)를 찾는다.

### 실습
어두운 그라데이션과 밝은 그라데이션을 만들고 평균·표준편차·히스토그램을 비교한다. 같은 평균인데 다르게 보이는 경우를 만든다.

### 말로 설명
“히스토그램은 무엇이 얼마나 있는지는 보여도 어디에 있는지는 말해 주지 않으므로, 진단은 추천의 근거이지 정답 판정은 아니다”라고 말한다.

## Day 9 — normalization과 gamma

시간: **5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명**

### 목표
정규화와 감마 보정의 목적과 파라미터 방향을 구분한다.

### 개념
min-max 정규화는 범위를 늘리고, 감마는 중간 밝기 분포를 비선형으로 바꾼다. 입력이 거의 검정·흰색이면 자동 감마가 원본을 복사하는 보호 조건도 있다.

### 코드 연결
[normalize_uint8와 apply_auto_gamma](../../src/opencv_preprocessing_advisor/transforms.py)를 읽고, target midpoint 검증을 찾는다.

### 실습
같은 어두운 이미지에 정규화와 감마를 적용한다. 평균·최소·최대와 눈에 띄는 차이를 한 줄씩 기록한다.

### 말로 설명
“정규화는 범위, 감마는 톤 곡선을 다룬다. 파라미터는 결과가 아니라 입력 진단을 보고 조절한다”라고 말한다.

## Day 10 — CLAHE

시간: **5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명**

### 목표
CLAHE의 clip limit와 tile grid가 과도한 대비 증폭을 어떻게 제어하는지 말한다.

### 개념
CLAHE는 국소 영역의 히스토그램을 다루며, clip limit는 특정 구간의 과도한 증폭을 제한한다. grid가 작아지면 더 국소적인 결과가 될 수 있어 부작용도 비교해야 한다.

### 코드 연결
[apply_lab_clahe](../../src/opencv_preprocessing_advisor/transforms.py)와 [pipeline 설정](../../src/opencv_preprocessing_advisor/config/pipelines.yaml)을 읽는다.

### 실습
clip limit 1.0, 2.0, 4.0과 grid 4, 8을 바꿔 4개 결과를 만든다. 가장 강한 결과가 선택 기준이 아닌 이유를 적는다.

### 말로 설명
“CLAHE는 저대비 영역을 드러낼 수 있지만 노이즈도 강조할 수 있어, 원본·진단·후속 특징을 함께 본다”라고 말한다.

## Day 11 — Gaussian, Median, Bilateral

시간: **5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명**

### 목표
세 필터를 노이즈 가정과 경계 보존 관점에서 비교한다.

### 개념
Gaussian은 주변값을 가중 평균해 부드럽게 하고, Median은 점 잡음에 강한 선택이 될 수 있으며, Bilateral은 색·공간 거리 모두를 고려해 경계를 남기려 한다. 비용과 효과는 이미지에 따라 확인한다.

### 코드 연결
[apply_gaussian, apply_median, apply_bilateral](../../src/opencv_preprocessing_advisor/transforms.py)과 [변환 테스트](../../tests/test_transforms.py)를 읽는다.

### 실습
그라데이션에 salt-and-pepper 잡음과 약한 Gaussian 잡음을 각각 넣고 세 필터를 적용한다. 경계 한 곳과 잡음 한 곳을 확대해 비교한다.

### 말로 설명
“필터 이름이 아니라 잡음 형태와 경계 보존 필요성이 선택 기준이다. 동일 kernel이라도 결과를 비교한다”라고 말한다.

## Day 12 — Sobel, Scharr, Laplacian, Canny

시간: **5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명**

### 목표
미분 기반 경계 연산자의 역할과 민감도를 구분한다.

### 개념
Sobel/Scharr는 방향별 변화, Laplacian은 2차 변화, Canny는 단계적 경계 검출 절차를 제공한다. 미분은 노이즈에도 반응하므로 전처리와 임계값을 함께 고려한다.

### 코드 연결
[TextureStatsExtractor](../../src/opencv_preprocessing_advisor/features.py)의 Sobel·Laplacian 사용과 [증거 표](../portfolio/evidence-map.md)를 읽는다.

### 실습
단순 사각형과 잡음 사각형에 네 연산자를 적용한다. Canny 임계값 두 쌍을 비교하고 끊긴 경계·잡음 경계를 표시한다.

### 말로 설명
“경계 연산자는 결함 자체를 확정하지 않고 밝기 변화 후보를 만든다. 그래서 잡음과 파라미터 영향을 분리해 설명한다”라고 말한다.

## Day 13 — morphology, threshold, contours, components

시간: **5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명**

### 목표
이진 마스크 뒤의 형태 연산과 영역 측정을 순서대로 설명한다.

### 개념
threshold는 마스크를 만들고, morphology는 작은 구멍·점들을 다루며, contours/components는 연결된 영역의 위치·면적 같은 측정 단위를 만든다. 구조 요소의 크기와 모양은 결과를 바꾼다.

### 코드 연결
[diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py)와 [서비스의 후보 설명](../../src/opencv_preprocessing_advisor/services.py)을 읽는다.

### 실습
점 노이즈와 작은 구멍을 가진 이진 마스크를 만든다. opening/closing 전후 컴포넌트 수와 가장 큰 면적을 비교한다.

### 말로 설명
“이진화는 끝이 아니라 측정의 시작이다. morphology와 연결요소 기준을 기록해야 면적 변화의 이유를 설명할 수 있다”라고 말한다.

## Day 14 — 진단 기반 선택과 주간 복습

시간: **5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명**

### 목표
진단 → 후보 → 비교 → 한계의 전처리 의사결정을 한 번 재현한다.

### 개념
프로젝트는 한 이미지의 휴리스틱 적합성 점수로 후보를 설명하고, 별도 데이터셋 평가로 분류 성능을 비교한다. 둘을 같은 정확도로 부르면 안 된다.

### 코드 연결
[ImageAdvisorService](../../src/opencv_preprocessing_advisor/services.py), [점수 구성](../../src/opencv_preprocessing_advisor/scoring.py), [한계](../portfolio/limitations.md)를 읽는다.

### 실습
저대비·점잡음·정상 이미지에 대해 ‘아무 변환/CLAHE/median’ 세 후보를 적용한다. 각 후보의 관찰, 선택, 반례를 표로 쓴다.

### 말로 설명
“추천은 진단에 근거한 가설이고, 데이터셋 평가는 별도 프로토콜로 검증한다”라고 1분간 말한 뒤 두 절차의 입력과 출력 차이를 적는다.
