# 구현·실험 실습 24개

각 실습은 코드가 아니라 **관찰 기록**까지 제출합니다. E01~E08은 안내형, E09~E16은 제약형, E17~E24는 빈 페이지형입니다. 프로젝트 코드는 [transforms.py](../../src/opencv_preprocessing_advisor/transforms.py)와 [evaluation.py](../../src/opencv_preprocessing_advisor/evaluation.py)를 근거로 삼습니다.

## 안내형 — 코드를 따라 결과를 측정하기

## E01: 배열 검사
`uint8` BGR 배열을 만들고 shape, dtype, 세 픽셀을 출력한다. [입력 계약](../../src/opencv_preprocessing_advisor/io.py)과 비교해 허용되지 않는 shape 하나를 적는다.

## E02: BGR/Gray 왕복
BGR → Gray → BGR 변환 뒤 원본과의 차이를 계산한다. 색 정보를 잃는 픽셀 예시 하나를 제시한다.

## E03: HSV 범위 표
파랑·회색·검정을 HSV로 바꿔 H/S/V를 표로 쓴다. [히스토그램 범위](../../src/opencv_preprocessing_advisor/features.py)와 일치하는지 확인한다.

## E04: LAB L만 보정
LAB의 L에만 CLAHE를 적용하고 a/b는 유지한다. [구현](../../src/opencv_preprocessing_advisor/transforms.py)과 비교해 세 단계 의사코드를 쓴다.

## E05: Unicode I/O 회귀
한글 파일명으로 이미지 저장·읽기를 하고 실패 시 오류 메시지를 기록한다. [I/O 테스트](../../tests/test_io.py)의 기대 동작을 확인한다.

## E06: 밝기·대비 진단
어두운 이미지와 저대비 이미지를 만들어 평균·표준편차를 비교한다. [진단 구현](../../src/opencv_preprocessing_advisor/diagnostics.py)에서 같은 측정값을 찾는다.

## E07: gamma 대 normalization
동일 입력에 두 변환을 적용해 min/max/mean을 표로 남긴다. 어느 결과가 중간톤을 더 바꿨는지 관찰로만 쓴다.

## E08: 세 필터 비교
점 잡음과 부드러운 잡음에 Gaussian/Median/Bilateral을 적용한다. 경계 한 곳과 배경 한 곳을 확대해 비교한다.

## 제약형 — 한 변수를 고정해 가설 검증하기

## E09: CLAHE grid 실험
clip limit는 2.0으로 고정하고 grid 4, 8, 16만 바꾼다. 가장 강한 대비가 선택되지 않을 수 있는 반례를 기록한다.

## E10: Canny 임계값 실험
입력과 blur를 고정한 채 Canny의 두 임계값만 바꾼다. 끊긴 경계와 잡음 경계 수를 수동으로 세어 비교한다.

## E11: morphology kernel 실험
opening에 3×3, 5×5, 9×9 kernel을 적용한다. [파이프라인 설정](../../src/opencv_preprocessing_advisor/config/pipelines.yaml)과 다른 결과를 설명한다.

## E12: 연결요소 측정
threshold 후 connected components로 면적 상위 3개를 출력한다. 작은 점을 제거한 뒤 순위가 어떻게 바뀌는지 기록한다.

## E13: 히스토그램의 위치 손실
색 비율은 같고 위치가 다른 두 이미지를 만든다. [ColorHistogramExtractor](../../src/opencv_preprocessing_advisor/features.py) 벡터 차이를 계산하고 한계를 쓴다.

## E14: HOG 크기 계약
128×128과 130×128 입력에서 HOG 생성 조건을 확인한다. [HOGExtractor](../../src/opencv_preprocessing_advisor/features.py)의 16 배수 제약을 예외 메시지와 함께 기록한다.

## E15: Gabor 방향성
0도·90도 줄무늬에 0도·90도 Gabor를 적용한다. 각 응답 절댓값 평균을 표로 만들고 방향을 바꾸면 왜 값이 달라지는지 쓴다.

## E16: scaler 누수 찾기
전체 feature matrix에 fit하는 의사코드와 train fold에만 fit하는 의사코드를 나란히 쓴다. [cross_validate](../../src/opencv_preprocessing_advisor/evaluation.py)에서 올바른 위치를 찾는다.

## 빈 페이지형 — 설계와 검증을 스스로 연결하기

## E17: 입력 검증 함수
빈 파일에서 BGR `uint8` 입력 검증 함수를 작성하고 잘못된 ndim, 채널 수, dtype의 테스트를 세 개 만든다. [기존 테스트](../../tests/test_io.py)와 비교한다.

## E18: 안전한 밝기 보정
범위·dtype을 보존하는 밝기 보정 함수를 작성한다. 음수/큰 양수 파라미터에서 무엇을 허용할지 설계 메모를 남긴다.

## E19: 새 진단 설계
‘어두운 픽셀 비율’ 또는 ‘에지 밀도’를 정의하고 정상·극단·빈 입력의 기대 동작을 테스트로 쓴다. [점수 코드](../../src/opencv_preprocessing_advisor/scoring.py)에 바로 연결하지 말고 한계를 먼저 적는다.

## E20: 후보 추천 설명문
세 변환 후보에 대해 입력 진단, 파라미터, 기대 효과, 위험을 한 줄씩 작성한다. [서비스 결과](../../src/opencv_preprocessing_advisor/services.py)의 설명 구조와 비교한다.

## E21: 특징 조합 설계
색·HOG·질감 중 둘을 선택해 벡터 연결 순서와 각 길이를 설계한다. [CombinedExtractor](../../src/opencv_preprocessing_advisor/features.py)와 다른 선택을 했다면 이유를 쓴다.

## E22: 2×2 혼동행렬 계산
임의 truth/prediction으로 혼동행렬, 클래스별 F1, Macro F1을 손으로 계산하고 [metrics 구현](../../src/opencv_preprocessing_advisor/evaluation.py) 결과와 비교한다.

## E23: 실패 보고서
오분류 하나를 가정해 입력·전처리·특징·분류·평가의 가능한 원인을 각각 하나씩 쓴다. 관찰된 사실과 가설을 분리한다. [한계](../portfolio/limitations.md)를 참고한다.

## E24: 5분 데모 재구성
빈 문서에서 문제, 진단, 후보, 특징, 평가, 한계 순으로 5분 데모를 구성한다. 각 주장마다 [근거 표](../portfolio/evidence-map.md) 또는 코드 링크를 하나 붙인다.
