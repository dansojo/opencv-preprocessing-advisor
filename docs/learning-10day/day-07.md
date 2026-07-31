# Day 7 - OpenCV 특징 추출

전처리가 이미지를 바꾸는 일이라면 특징 추출은 분류기가 사용할 수 있는 숫자 벡터로 바꾸는 일이다. 오늘은 색상·형상·질감 특징이 무엇을 버리고 무엇을 남기는지, 그리고 SIFT Bag of Words를 공정하게 비교하려면 왜 fold 안에서 vocabulary를 학습해야 하는지를 다룬다.

## 오늘 답해야 할 핵심 질문

- HSV/LAB histogram은 BGR 원소를 그대로 쓰는 것보다 어떤 정보를 요약하는가?
- HOG의 cell, block, bin은 어떤 구조를 수치화하며 왜 resize가 필요한가?
- Sobel, Laplacian, Gabor 통계는 texture를 어떻게 요약하는가?
- 특징 결합은 언제 도움이 되고 언제 차원·스케일 문제를 만드는가?
- SIFT와 Bag of Words의 vocabulary는 왜 fold-local vocabulary여야 하는가?

## 개념과 원리

분류기는 원본 이미지를 직접 이해하지 못하고 행렬 `X`의 한 행당 하나의 feature vector를 받는다. 좋은 특징은 task와 관계 있는 변화를 남기고 관계 없는 변화를 줄인다. 하지만 모든 정보를 보존하는 특징은 없으므로, ‘좋은 특징’은 절대적인 말이 아니라 class를 가르는 데 도움이 되는 표현이라는 뜻이다. 전처리 후 결과가 매끈해도 실제 class 경계에 필요한 질감이나 작은 결함이 사라지면 downstream 분류기는 나빠질 수 있다.

ColorHistogramExtractor는 BGR 입력을 HSV와 LAB로 바꾼 뒤 HSV의 H/S, LAB의 L 채널 histogram을 각각 32 bin으로 만들고 합쳐서 전체 합이 1이 되도록 정규화한다. H는 색상 각도, S는 채도, L은 지각적 밝기에 가까운 축을 제공하므로 조명·색상 질문을 BGR 채널 하나보다 분리해 볼 수 있다. 그러나 histogram은 픽셀의 위치와 shape을 버린다. 색 분포가 같지만 균열 위치가 다른 두 이미지는 비슷하게 보일 수 있다.

HOG(Histogram of Oriented Gradients)는 resize한 grayscale의 지역 gradient 방향을 histogram으로 모은다. 여기서는 128×128 window, 8×8 cell, 16×16 block, 8×8 block stride, 9 direction bin을 쓴다. cell은 작은 공간 구역, bin은 gradient 방향 구간, block normalization은 조명 크기 변화에 대한 민감도를 줄이는 장치다. 그래서 HOG는 윤곽·방향성 shape에 강하지만, 색상 자체나 장거리 배치에는 제한적이다. 입력 크기를 고정하지 않으면 descriptor 길이가 이미지마다 달라져 classifier 행렬을 만들 수 없다.

TextureStatsExtractor는 grayscale float을 Sobel magnitude와 Laplacian, 네 방향의 Gabor response로 필터링하고 각 response의 절댓값 mean/std/75·95 percentile을 잇는다. Gabor는 방향과 주파수에 반응하는 filter bank라 반복 줄무늬나 방향성 표면을 요약하기 좋다. 여기서 값은 결함 위치가 아니라 response 분포의 통계다. 길이·위치·회전 변화를 완전히 해결하지 않으며 kernel 크기, sigma, wavelength, orientation은 데이터 가설이다.

CombinedExtractor는 color+HOG+texture를 concatenate한다. 이것은 서로 다른 단서를 분류기에게 주지만 차원이 큰 HOG가 distance나 margin을 지배할 수 있다. 그래서 평가에서는 feature matrix를 `float32`로 만들고 각 **훈련 fold**에서만 표준화한다. 결합이 항상 승리한다는 보장은 없다. 작은 데이터에서는 많은 feature가 noise와 overfitting 기회를 늘릴 수 있다.

SIFT는 keypoint 주변에서 scale/rotation에 비교적 강한 descriptor를 만들고, Bag of Words(BoW)는 training descriptor를 k-means cluster로 모아 visual word vocabulary를 만든 뒤 이미지마다 word 빈도 histogram을 만든다. `SiftBowExtractor`는 구현되어 있다. 하지만 현재 BenchmarkService의 feature profile은 `color`, `shape`, `texture`, `combined`만 공개하며 **SiftBowExtractor는 현재 benchmark profile로 노출되지 않는다.** 더 중요한 공정성 조건은 SIFT vocabulary를 전체 이미지에서 먼저 fit하면 test fold의 descriptor 분포가 cluster center에 들어가므로 leakage가 생긴다는 점이다. 각 split마다 학습 fold 이미지로 새 vocabulary를 fit하고, 그 vocabulary로 training/test를 transform하는 fold-local vocabulary가 필요하다.

## OpenCV API와 파라미터

[features.py](../../src/opencv_preprocessing_advisor/features.py)는 네 개의 현재 feature profile과 SiftBowExtractor를 제공한다. [services.py](../../src/opencv_preprocessing_advisor/services.py)의 `_feature_extractor()`는 공개 profile 선택을 제한한다. [test_features.py](../../tests/test_features.py)는 shape, dtype, normalization, SIFT fit contract를 확인하고 [test_services.py](../../tests/test_services.py)는 benchmark profile 선택을 검증한다.

| API/구현 | 핵심 파라미터 | 출력 | 해석 경계 |
| --- | --- | --- | --- |
| `cv2.calcHist` | 채널, bins, range | HSV/LAB histogram | 위치 정보는 포함하지 않는다. |
| `cv2.HOGDescriptor` | 128×128 window, 8×8 cell, 16×16 block, 9 bins | 고정 길이 gradient vector | resize 정책도 feature 정의의 일부다. |
| `cv2.Sobel`/`cv2.Laplacian` | `CV_32F`, `ksize=3` | texture response | noise도 response를 키울 수 있다. |
| `cv2.getGaborKernel` | 15×15, sigma 3, theta, lambda 8 | 방향별 Gabor filter | 네 orientation은 모든 texture를 대표하지 않는다. |
| `cv2.SIFT_create` + `BOWKMeansTrainer` | vocabulary size 32, seed 42 | SIFT BoW histogram | vocabulary는 학습 fold에서만 fit해야 한다. |

현재 코드의 HOG size는 `(128, 128)`이며 각 차원은 16의 배수여야 한다. 색 histogram은 H 범위 `[0, 180)`, S/L 범위 `[0, 256)`를 사용한다. Gabor orientation은 0, π/4, π/2, 3π/4다. 이것은 “보편 최적값”이 아니라 검증 가능한 기본 명세다. [features.py (main)](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/features.py)를 기준으로 말하고, 새로운 profile은 코드와 test를 함께 추가한 뒤에만 공개 선택지로 제시한다.

## 언제 사용하고 피하는가

색상 class나 조명/색조 차이가 중요한 경우 HSV/LAB histogram을 후보로 쓴다. 형상·방향이 차이를 만들면 HOG, 반복 결·거칠기·방향성 표면이면 Gabor 기반 texture statistics를 시도한다. 세 단서가 모두 그럴듯하면 CombinedExtractor를 baseline으로 삼을 수 있다. 단, feature 선택은 예쁜 visualization이 아니라 동일 split에서의 label 기반 평가로 정한다.

피할 일은 색상만 다른 task에 grayscale HOG만 고집하거나, 위치가 결정적인 task에서 global histogram만 믿는 것이다. HOG를 다양한 원본 크기에 그대로 적용해 서로 길이가 다른 vector를 만들 수도 없다. SIFT BoW를 전체 데이터로 fit한 뒤 ‘공정한 CV’라고 말하면 안 된다. SIFT의 존재와 current benchmark support는 다르다. fold-local vocabulary 구현 없이 SIFT를 현재 leaderboard와 직접 비교하지 않는다.

## 프로젝트 코드 연결

- [색·형상·질감·결합 특징: features.py](../../src/opencv_preprocessing_advisor/features.py)는 모든 결과를 finite `float32` matrix로 만든다.
- [benchmark profile 선택: services.py](../../src/opencv_preprocessing_advisor/services.py)는 현재 color/shape/texture/combined만 선택한다.
- [특징 테스트: test_features.py](../../tests/test_features.py)는 histogram 정규화, HOG 차원, SIFT fit 전 transform 실패를 검증한다.
- [서비스 테스트: test_services.py](../../tests/test_services.py)는 공개 profile과 benchmark 조합을 검증한다.
- [평가 코드: evaluation.py](../../src/opencv_preprocessing_advisor/evaluation.py)는 feature 이후 fold-local scaling을 수행한다.

Feature matrix의 행 순서는 manifest의 sample 순서와 같아야 한다. image가 하나도 없거나 NaN/inf가 있으면 `_as_float_matrix()`가 실패한다. 이 검사는 숫자만 나오면 된다는 착각을 막는다. experiment의 재현 근거는 [benchmark-evidence.json](../portfolio/benchmark-evidence.json)과 [evidence-map.md](../portfolio/evidence-map.md)에 연결한다.

## 직접 실험

합성 샘플의 원본과 좌우 반전본에 color/HOG/texture/combined vector를 만들고 shape·dtype·L2 거리를 출력한다. 이는 성능 실험이 아니라 각 표현이 어떤 변화에 민감한지 보는 smoke test다.

```python
from pathlib import Path

import cv2
import numpy as np

from opencv_preprocessing_advisor.features import (
    ColorHistogramExtractor, CombinedExtractor, HOGExtractor, TextureStatsExtractor,
)
from opencv_preprocessing_advisor.io import decode_image

image = decode_image(Path("data/samples/synthetic-tile.png"))
images = [image, cv2.flip(image, 1)]
extractors = {
    "color": ColorHistogramExtractor(),
    "hog": HOGExtractor(),
    "texture": TextureStatsExtractor(),
    "combined": CombinedExtractor(),
}
for name, extractor in extractors.items():
    matrix = extractor.transform(images)
    distance = np.linalg.norm(matrix[0] - matrix[1])
    print(name, "shape=", matrix.shape, "dtype=", matrix.dtype, "distance=", f"{distance:.4f}")
```

추가로 같은 이미지에 밝기만 조금 바꾼 버전과 Gaussian blur 버전을 만들어 표의 distance를 채운다. HOG의 `size=(64, 64)`를 시도할 수 있지만 서로 다른 size의 점수를 같은 scale로 해석하지 말고 descriptor 길이와 평가 조건까지 기록한다. SIFT를 연습한다면 최소 두 개의 image list에서 training subset만으로 `.fit()`한 뒤 남은 image에 `.transform()`한다. 전체 묶음으로 fit하는 shortcut은 fold-local vocabulary 실험이 아니다.

## 예상 결과와 해석

| 관찰 | 예상 결과 | 해석과 다음 질문 |
| --- | --- | --- |
| color histogram | 두 행의 합이 대체로 1 | 위치가 아니라 색 분포를 비교한다. 반전에 둔감할 수 있다. |
| HOG | 고정된 큰 차원, `float32` | edge 방향/배치 변화에 반응한다. resize가 정보를 바꾼다. |
| texture | 작은 통계 벡터, `float32` | Gabor/Sobel response 분포를 요약한다. 위치를 특정하지 않는다. |
| combined | 세 descriptor 길이의 합 | 정보가 늘지만 classifier scaling/복잡도 비용도 늘어난다. |
| SIFT BoW | fit 뒤 vocabulary 크기 histogram | vocabulary가 test를 보지 않았는지 split별로 확인한다. |

distance가 작다고 두 이미지가 같은 class라는 뜻은 아니고, 크다고 다른 class라는 뜻도 아니다. feature는 classifier·threshold·데이터 분포와 함께 해석한다. 특히 color histogram의 정규화는 이미지 면적 차이로 count가 커지는 효과를 줄이지만, 조명이나 white balance 변화가 없어지는 것은 아니다.

## 자주 하는 실수와 디버깅

1. **BGR histogram을 HSV/LAB이라고 부름**: 변환한 channel과 range를 명시한다. H의 최대는 255가 아니라 OpenCV HSV에서 180이다.
2. **HOG 크기 불일치**: 이미지마다 descriptor 길이가 다르면 matrix가 안 된다. resize와 16의 배수 제약을 확인한다.
3. **`uint8` feature 전달**: gradient/통계와 `cv2.ml` 입력은 `float32` 계약을 지킨다.
4. **Combined가 자동 승리한다고 가정**: 차원이 늘면 variance와 overfitting 위험도 늘 수 있다. CV metric으로 비교한다.
5. **SIFT leakage**: test image descriptor로 visual word center를 만들지 않는다. 매 fold의 학습 subset에 새 fold-local vocabulary를 fit한다.

dimension 오류면 extractor별 `matrix.shape`, input BGR shape, HOG size를 출력한다. SIFT가 “insufficient descriptors”로 실패하면 vocabulary size를 줄이는 것이 아니라 training fold에서 충분한 keypoint가 있는지 먼저 확인하고, 그 변경을 명세로 남긴다. profile 이름 오류는 서비스가 지원하지 않는 current benchmark profile을 요청했을 가능성이 크다.

## 본인 말로 설명하기

### 1분 설명

“특징 추출은 이미지를 classifier가 받을 `float32` 벡터로 바꾸는 단계입니다. 이 프로젝트는 HSV/LAB histogram으로 색 분포, HOG로 지역 gradient 방향, Sobel·Laplacian·Gabor 통계로 질감을 만들고 필요하면 합칩니다. histogram은 위치를 버리고 HOG는 resize 정책에 의존하므로 task에 맞춰야 합니다. SIFT BoW 구현은 있지만 현재 benchmark profile로 노출하지 않습니다. SIFT를 공정하게 평가하려면 test fold descriptor가 vocabulary 중심을 만드는 데 들어가지 않도록 각 학습 fold 안에서 vocabulary를 새로 fit해야 합니다.”

### 깊이 설명

“ColorHistogramExtractor는 HSV H/S와 LAB L의 32-bin 분포를 정규화해 색·밝기 단서를 남기고, HOG는 128×128 grayscale의 cell/block/bin gradient 구조를 고정 길이로 요약합니다. TextureStatsExtractor는 Sobel magnitude, Laplacian, 네 방향 Gabor response의 robust한 분포 통계를 사용합니다. Combined는 이 표현을 concatenate하지만 scaling과 sample 수에 비해 차원이 커지는 위험이 있어 CV로만 선택합니다. SiftBowExtractor는 SIFT descriptor를 k-means visual word로 양자화하지만 vocabulary는 학습 데이터 자체에서 추정한 parameter입니다. 전체 데이터를 먼저 보면 test 분포가 feature definition에 누출되므로 cross-validation 안에서 fold-local vocabulary를 fit/transform해야 합니다.”

## 완료 기준

- [ ] **이해**: HSV/LAB histogram, HOG, Gabor texture가 남기는 정보와 버리는 정보를 한 가지씩 설명했다.
- [ ] **구현**: 합성 이미지 두 장에 네 extractor를 실행해 `shape`, `float32`, distance를 출력했다.
- [ ] **해석**: 특징 distance를 class 정답으로 과해석할 수 없는 이유와 다음 label 기반 검증을 적었다.
- [ ] **설명**: SiftBowExtractor의 존재와 current benchmark profile 미노출, fold-local vocabulary 필요성을 구분해 말했다.
