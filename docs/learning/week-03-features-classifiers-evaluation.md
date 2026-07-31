# 3주차 — 특징, 분류, 평가

이 주의 핵심은 “특징이 좋다”가 아니라 어떤 특징·분류기·검증 규칙을 함께 썼는지 말하는 것입니다. 구현은 [features.py](../../src/opencv_preprocessing_advisor/features.py), [classifiers.py](../../src/opencv_preprocessing_advisor/classifiers.py), [evaluation.py](../../src/opencv_preprocessing_advisor/evaluation.py)에 있습니다.

## Day 15 — 색 히스토그램

시간: **5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명**

### 목표
색 히스토그램이 위치가 아닌 분포를 요약한다는 점을 설명한다.

### 개념
히스토그램은 픽셀 위치를 버리고 bin별 개수를 모은다. 프로젝트는 HSV H/S와 LAB L 채널의 히스토그램을 이어 붙이고 전체 합으로 정규화한다.

### 코드 연결
[ColorHistogramExtractor](../../src/opencv_preprocessing_advisor/features.py)와 [특징 테스트](../../tests/test_features.py)를 따라 bin 수와 범위를 확인한다.

### 실습
같은 색 비율이지만 위치가 다른 두 이미지를 만들어 히스토그램을 비교한다. 비슷한 벡터라도 이미지가 다른 이유를 적는다.

### 말로 설명
“히스토그램은 색 분포에는 민감하지만 배치에는 둔감하다. 그래서 단독 정답이 아니라 다른 특징과 결합한다”라고 말한다.

## Day 16 — HOG

시간: **5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명**

### 목표
HOG가 국소 방향성 분포를 쓰는 이유와 입력 크기 제약을 말한다.

### 개념
HOG는 셀·블록 안의 기울기 방향 히스토그램을 모아 윤곽·형상 단서를 만든다. 프로젝트 구현은 입력을 고정 크기로 resize하고, 크기는 16의 배수여야 한다.

### 코드 연결
[HOGExtractor](../../src/opencv_preprocessing_advisor/features.py)와 [HOG 검증](../../tests/test_features.py)를 읽는다.

### 실습
세로 줄·가로 줄·회전 사각형을 128×128로 만들어 HOG 벡터 길이를 출력한다. resize 전후 어떤 형상 정보가 바뀔지 적는다.

### 말로 설명
“HOG는 방향성 형상에 유용한 고정 길이 표현이지만, 입력 해상도와 정렬 방식이 특징 자체를 바꾼다”라고 말한다.

## Day 17 — Sobel, Laplacian, Gabor texture

시간: **5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명**

### 목표
여러 질감 응답을 통계량으로 요약하는 설계를 설명한다.

### 개념
프로젝트는 Sobel magnitude, Laplacian, 네 방향 Gabor 응답의 절댓값 평균·표준편차·백분위를 특징으로 만든다. 필터 이미지 전체를 그대로 모델에 넣는 것이 아니다.

### 코드 연결
[TextureStatsExtractor](../../src/opencv_preprocessing_advisor/features.py)와 [evidence map](../portfolio/evidence-map.md)을 읽는다.

### 실습
가로 줄·세로 줄·무작위 잡음에 0도와 90도 Gabor를 적용해 평균 절댓값을 비교한다. 방향 선택이 결과를 바꾼 사례를 기록한다.

### 말로 설명
“질감은 한 필터의 한 픽셀이 아니라 여러 방향 응답의 분포로 요약한다. 그래서 어떤 통계를 썼는지까지 밝혀야 한다”라고 말한다.

## Day 18 — SIFT와 BoW 개념

시간: **5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명**

### 목표
SIFT keypoint와 bag of visual words의 학습·변환 순서를 말한다.

### 개념
SIFT는 지역 keypoint의 descriptor를 만들고, BoW는 훈련 이미지 descriptor로 vocabulary를 만든 뒤 각 이미지의 단어 빈도를 벡터로 만든다. vocabulary는 평가 fold 바깥 데이터를 보면 안 된다.

### 코드 연결
[SiftBowExtractor](../../src/opencv_preprocessing_advisor/features.py)와 [SIFT 테스트](../../tests/test_features.py)를 읽는다.

### 실습
특징점이 거의 없는 단색 이미지와 모서리가 많은 이미지에서 descriptor 개수를 비교한다. vocabulary보다 descriptor가 적을 때 구현이 왜 실패해야 하는지 설명한다.

### 말로 설명
“BoW의 단어장은 학습 데이터에서만 만들어야 한다. 그렇지 않으면 테스트 이미지의 구조가 학습 표현에 새어 들어간다”라고 말한다.

## Day 19 — SVM

시간: **5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명**

### 목표
SVM을 비교 대상 분류기로 설명하고 스케일링 필요성을 연결한다.

### 개념
SVM은 특징 공간에서 분리 경계를 학습하는 분류기다. 거리·내적에 영향을 받는 특징은 스케일이 다르면 특정 성분이 과도하게 영향을 줄 수 있어 fold별 표준화가 중요하다.

### 코드 연결
[SVM 생성](../../src/opencv_preprocessing_advisor/classifiers.py)과 [교차검증](../../src/opencv_preprocessing_advisor/evaluation.py)을 읽는다.

### 실습
한 특징은 0~1, 다른 특징은 0~1000인 작은 표를 만들고 표준화 전후 평균·표준편차를 계산한다. 표준화 fit 데이터가 무엇인지 표시한다.

### 말로 설명
“SVM 결과는 커널 이름만으로 설명하지 않는다. 특징 스케일, 하이퍼파라미터, fold 규칙을 같은 실험 조건으로 제시한다”라고 말한다.

## Day 20 — kNN과 RTrees

시간: **5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명**

### 목표
kNN과 RTrees를 서로 다른 가정의 비교 기준으로 설명한다.

### 개념
kNN은 주변 학습 샘플의 거리에 의존하고, RTrees는 여러 결정 트리를 결합한다. 이 프로젝트는 둘을 같은 fold 계획에서 비교하며, 데이터셋·특징·seed가 바뀌면 순위도 달라질 수 있다.

### 코드 연결
[classifier factory](../../src/opencv_preprocessing_advisor/classifiers.py), [실험 결과](../portfolio/experiment-results.md), [분류 테스트](../../tests/test_classifiers.py)를 읽는다.

### 실습
2차원 점 6개에서 k=1과 k=3 예측이 달라지는 질의점을 만든다. RTrees에 대해 ‘왜 트리 수/seed를 기록해야 하는가’를 한 문장으로 쓴다.

### 말로 설명
“모델 선택은 이름의 우열이 아니라 같은 데이터 분할과 특징에서 관찰한 비교 결과다. 다른 조건에 일반화하지 않는다”라고 말한다.

## Day 21 — 교차검증, Macro F1, confusion matrix, leakage

시간: **5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명**

### 목표
fold-local scaling, Macro F1, 혼동행렬과 누수 방지를 하나의 평가 절차로 연결한다.

### 개념
교차검증은 분할마다 학습/평가를 반복한다. 표준화기는 train fold에만 fit하고 test fold에는 transform만 한다. Macro F1은 클래스별 F1의 평균이며, 혼동행렬은 어느 실제 클래스가 어디로 예측됐는지 보여 준다.

### 코드 연결
[cross_validate](../../src/opencv_preprocessing_advisor/evaluation.py), [metrics 구현](../../src/opencv_preprocessing_advisor/evaluation.py), [평가 테스트](../../tests/test_evaluation.py)를 읽는다.

### 실습
2×2 혼동행렬을 손으로 만들고 클래스별 precision·recall·F1과 Macro F1을 계산한다. 전체 데이터로 scaler를 fit한 잘못된 의사코드를 찾아 고친다.

### 말로 설명
“누수 방지는 성능을 낮추기 위한 제약이 아니라 배포 전에 볼 수 없는 정보를 평가에 넣지 않기 위한 규칙이다”라고 말한다.
