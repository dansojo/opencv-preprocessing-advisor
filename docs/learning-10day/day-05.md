# Day 5 - 에지·경계·형태학

오늘은 밝기 영상에서 구조를 꺼내는 도구를 다룬다. 미분 기반 에지, threshold, morphology, contour와 connected components는 비슷해 보이는 흰/검정 mask를 만들지만 질문과 출력의 의미가 다르다.

## 오늘 답해야 할 핵심 질문

- Sobel, Scharr, Laplacian은 1차·2차 미분 관점에서 무엇이 다른가?
- 표준 Canny workflow의 선택적 blur와 `cv2.Canny` API 자체의 단계는 어떻게 다른가?
- global, adaptive, Otsu threshold는 각각 어떤 조명 가정에 의존하는가?
- erosion, dilation, opening, closing은 전경·배경 정의에 따라 어떻게 해석해야 하는가?
- contour와 connected components는 언제 서로 바꿔 쓰면 안 되는가?

## 개념과 원리

에지는 밝기 변화가 큰 위치다. grayscale `I`에서 Sobel은 x/y 방향의 1차 미분 근사 `Gx`, `Gy`를 convolution으로 구하고, 크기는 `sqrt(Gx² + Gy²)` 또는 빠른 근사 `|Gx|+|Gy|`로 본다. 방향은 `atan2(Gy, Gx)`다. `ksize=3` Sobel은 널리 쓰이는 시작점이며, **Scharr**는 3×3 크기에서 회전 대칭성/정확도를 개선한 커널로 작은 kernel 미분이 중요할 때 후보다. Laplacian은 2차 미분이라 급격한 변화에 반응하지만 noise에도 민감하고, 부호가 바뀌는 zero-crossing 해석이 필요하다.

전형적인 **Canny workflow**에서는 noise에 민감한 gradient를 계산하기 전에 Gaussian blur를 *선택적으로, 호출자가 명시해* 적용할 수 있다. 그 뒤 Canny 검출 단계는 (1) gradient magnitude와 direction 계산, (2) non-maximum suppression(NMS)으로 방향상 최대인 가는 선만 남기기, (3) low/high double threshold로 strong/weak/non-edge 나누기, (4) hysteresis로 strong edge에 연결된 weak edge만 살리기다. 중요한 API 경계는 `cv2.Canny는 Gaussian blur를 내부적으로 호출하지 않는다`는 점이다. 즉 `cv2.Canny(gray, low, high)`에 넣는 영상이 이미 smoothing됐는지와 blur kernel은 호출자가 정한다. high threshold만 바꾸거나, 필요할 수 있는 사전 blur의 효과를 threshold 효과로 설명하면 원인을 잘못 해석한다.

threshold는 grayscale을 이진 mask로 바꾸는 결정 경계다. global threshold는 하나의 `T`로 `I>T`를 전경으로 하므로 조명이 균일하고 히스토그램이 분리될 때 간단하다. Otsu는 히스토그램에서 class 간 분산을 크게 하는 T를 자동으로 찾지만, 두 분포가 잘 분리된다는 가정이 약하면 흔들린다. adaptive threshold는 주변 block마다 T를 잡아 조명 그라데이션에 대응할 수 있지만, blockSize와 C에 따라 정상 texture를 조각낼 수 있다.

morphology는 binary 또는 grayscale 이미지에서 구조 요소(structuring element)를 미끄러뜨리는 연산이다. 흰색이 전경이라는 약속을 먼저 확인한다. erosion은 전경을 줄이고 작은 연결을 끊으며, dilation은 전경을 넓혀 작은 틈을 메운다. opening=erosion→dilation은 작은 전경 점을 제거하는 데, closing=dilation→erosion은 전경 내부의 작은 구멍과 끊김을 메우는 데 유용하다. morphological gradient는 팽창-침식 차이, top-hat은 원본-열기, black-hat은 닫기-원본에 해당한다.

마지막 mask에서 contour는 경계의 점열과 계층을 얻어 둘레·근사 다각형·외곽 형상을 다루기 좋다. **connected components**는 연결된 전경 덩어리에 정수 label을 붙여 개수, 면적, bounding box, centroid를 다루기 좋다. contour는 구멍의 경계/계층 질문에, components는 각 blob의 통계와 크기 필터 질문에 더 자연스럽다. 같은 mask라도 4/8 connectivity 선택과 전경 극성에 따라 component 수가 달라진다.

## OpenCV API와 파라미터

[diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py)는 median 기반 Canny 뒤 `connectedComponentsWithStats(..., connectivity=8)`로 일정 면적 이상의 에지 성분 비율을 계산한다. [transforms.py](../../src/opencv_preprocessing_advisor/transforms.py)의 `apply_blackhat`은 ellipse/rect/cross 구조 요소와 홀수 kernel을 사용한 형태학 예다. [technique_explorer.py](../../ui/technique_explorer.py)는 UI에서 기법 결과를 보여 주는 연결점이며, [test_ui_imports.py](../../tests/test_ui_imports.py)는 UI 모듈이 처리 시작 없이 import되는지 확인한다.

| API | 핵심 인자 | 결과 | 주의점 |
| --- | --- | --- | --- |
| `cv2.Sobel` / `cv2.Scharr` | `ddepth`, 방향, ksize | 방향별 gradient | `uint8` 출력은 음수 gradient를 잃을 수 있어 `CV_32F`/`CV_64F`를 쓴다. |
| `cv2.GaussianBlur` | 홀수 kernel, sigma | 선택적인 사전 smoothing | `cv2.Canny`와 별도 호출이다. blur가 필요한지와 강도는 입력 noise에 따라 정한다. |
| `cv2.Canny` | low/high threshold, 선택적 aperture/L2gradient | 가는 binary edge | API는 gradient·NMS·double threshold·hysteresis를 수행하며 Gaussian blur를 내장하지 않는다. |
| `cv2.threshold(..., OTSU)` | global T + Otsu flag | 한 임계값 mask | 조명 그라데이션/단봉 분포에서 불안정할 수 있다. |
| `cv2.adaptiveThreshold` | blockSize, C | local threshold mask | blockSize는 홀수, 너무 작으면 texture가 분할된다. |
| `morphologyEx` | operation, kernel | opening/closing/black-hat 등 | 흰 전경/검정 배경 가정을 확인한다. |

[test_diagnostics.py](../../tests/test_diagnostics.py)는 blur가 sharpness를 낮춘다는 사실을, [test_transforms.py](../../tests/test_transforms.py)는 변환의 입력 계약을 검증한다. Notion에는 [diagnostics.py (main)](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/diagnostics.py) 같은 GitHub `main` 링크로 바꿔 추적성을 유지한다.

## 언제 사용하고 피하는가

Sobel/Scharr는 방향성 gradient를 특징으로 쓰거나 경계 후보를 탐색할 때, Canny는 가는 연결 에지가 필요한 빠른 구조 점검에 사용한다. threshold는 실제 전경/배경의 밝기 차이가 의미 있을 때, morphology는 이미 threshold된 mask의 작은 잡음·틈을 도메인 크기에 맞춰 정리할 때 사용한다. components는 불량점 개수/면적처럼 object statistics를, contour는 외곽 윤곽·구멍 계층처럼 boundary geometry를 원할 때 택한다.

피해야 할 경우는 조명과 표면 반사가 강한데 global threshold 하나만 믿는 경우다. adaptive threshold도 만능이 아니며, texture가 조각나면 component 수가 폭증할 수 있다. 큰 closing kernel은 실제 균열을 메워 결함을 없애고, 큰 opening은 작은 결함 자체를 제거한다. mask가 화면에서 더 깔끔해도 **시각적 개선은 분류 성능 개선을 보장하지 않는다**. 단일 이미지 heuristic·구조 관찰과 classifier의 label 기반 성능을 구분한다.

## 프로젝트 코드 연결

- [에지 진단: diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py)는 Canny threshold 설정과 connected components 기반 continuity 계산을 구현한다.
- [형태학 변환: transforms.py](../../src/opencv_preprocessing_advisor/transforms.py)는 black-hat과 구조 요소 shape을 제공한다.
- [기법 탐색 UI: technique_explorer.py](../../ui/technique_explorer.py)는 결과를 표시할 수 있는 사용자 접점이다.
- [진단 테스트: test_diagnostics.py](../../tests/test_diagnostics.py)는 알려진 blur/noise 입력의 방향성 검증이다.
- [UI import 테스트: test_ui_imports.py](../../tests/test_ui_imports.py)는 UI 모듈의 안전한 로딩을 검증한다.

프로젝트의 edge continuity는 “좋은 경계의 비율”이라는 절대 품질이 아니라, Canny 결과 중 area 8 이상 8-연결 component에 속한 pixel 비율이다. 기준선과 후보를 같은 Canny 규칙으로 비교할 때만 변화가 해석 가능하다.

## 직접 실험

아래 코드는 합성 타일에서 Sobel/Scharr/Canny와 Otsu/adaptive threshold를 만들고, closing 전후의 connected components 통계를 비교한다. `GaussianBlur`는 `cv2.Canny`에 앞서 **선택적으로 명시한 Gaussian blur**이며 API 내부 동작이 아니다. white foreground를 명시해 mask 해석을 고정한다.

```python
from pathlib import Path

import cv2
import numpy as np

from opencv_preprocessing_advisor.io import decode_image

image = decode_image(Path("data/samples/synthetic-tile.png"))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
scharr_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 140)
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 5)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)

for name, mask in {"otsu": otsu, "closed": closed, "adaptive": adaptive}.items():
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    print(name, "components=", count - 1, "largest=", int(areas.max()) if len(areas) else 0)
print("Sobel/Scharr mean abs:", np.mean(np.abs(sobel_x)), np.mean(np.abs(scharr_x)))
```

먼저 Otsu와 adaptive의 전경이 같은 의미인지 눈으로 확인한다. 반전이 필요하면 `THRESH_BINARY_INV`를 쓰되 표의 전경 정의도 바꾼다. 그 다음 closing kernel 3/5/9를 비교해 작은 구멍이 메워지는 것과 별개의 객체가 합쳐지는 것을 구분한다. `findContours`로 외곽 둘레를, components로 면적/중심을 각각 출력해 어떤 질문에 맞는지 기록한다.

## 예상 결과와 해석

| 관찰 | 예상 결과 | 해석과 다음 질문 |
| --- | --- | --- |
| Sobel vs Scharr | 둘 다 방향 gradient를 보이며 크기 scale은 다를 수 있음 | 같은 threshold로 단순 비교하지 말고 방향·kernel 목적을 본다. |
| Canny | 선택적으로 명시한 Gaussian blur 입력에서 API의 gradient/NMS/double threshold/hysteresis 뒤 가는 edge mask | blur 유무·kernel과 threshold를 별도 변수로 바꿔 약한 구조 연결을 기록한다. |
| Otsu | 전역 분포가 분리되면 간단한 mask | 조명 차이가 크면 배경까지 전경이 될 수 있다. |
| adaptive | 지역 밝기에 반응하는 조각 mask 가능 | blockSize/C가 texture를 결함처럼 분할했는지 확인한다. |
| closing + components | 틈은 줄고 component 면적·개수는 변할 수 있음 | 실제 객체가 합쳐졌다면 kernel이 과하다. |

예상 표의 ‘좋음’은 없다. component가 적어진 것이 잡음 제거일 수도, 두 개의 실제 결함이 합쳐진 것일 수도 있다. 원본과 overlay를 함께 보고 domain에서 의미 있는 최소 결함 크기보다 작은 구조 요소만 사용한다. 결과 image를 유일한 정답처럼 고르지 않는다.

## 자주 하는 실수와 디버깅

1. **Sobel을 `uint8`에 바로 저장**: 음수 미분이 잘려 방향 정보가 망가질 수 있다. 계산은 float depth로 하고 표시만 절댓값/스케일링한다.
2. **Canny smoothing의 위치를 혼동**: low/high와 NMS·hysteresis는 `cv2.Canny`의 검출 단계다. Gaussian blur는 필요하면 그 전에 별도 호출하며, API가 내부적으로 수행한다고 말하지 않는다.
3. **threshold 극성 미확인**: 흰색이 전경인지 확인하지 않으면 erosion/dilation의 설명이 반대가 된다.
4. **adaptive blockSize를 짝수로 지정**: OpenCV 요구를 확인하고 3 이상의 홀수를 사용한다.
5. **contour 수와 object 수를 동일시**: 구멍·계층·접촉 객체 때문에 달라질 수 있다. area 통계는 connected components가 더 직접적이다.

mask가 전부 흰색/검은색이면 gray range와 threshold 값을 먼저 출력한다. component 수가 갑자기 커지면 threshold보다 morphology 전의 mask와 connectivity를 확인한다. Canny가 텅 비면 먼저 사전 `GaussianBlur`를 쓸지와 그 kernel을 결정하고, 그 다음 `cv2.Canny`의 low/high와 영상 median 기반 기준을 차례대로 바꾼다.

## 본인 말로 설명하기

### 1분 설명

“Sobel과 Scharr는 밝기 1차 미분으로 방향 에지를 보고, Laplacian은 2차 미분이라 변화와 noise에 더 민감합니다. 전형적인 Canny workflow에서는 필요하면 `GaussianBlur`를 **별도 호출**한 뒤 `cv2.Canny`를 호출합니다. `cv2.Canny` 자체는 gradient, non-maximum suppression, double threshold, hysteresis를 수행하며 blur를 내부 호출하지 않습니다. threshold는 전역·Otsu·adaptive가 조명 가정을 달리하고, morphology는 흰 전경이라는 약속 아래 작은 점과 틈을 정리합니다. contour는 경계 기하, connected components는 객체 수·면적·중심 통계에 적합합니다. 깔끔한 mask가 분류 성능을 뜻하지는 않습니다.”

### 깊이 설명

“구조 추출은 연속 밝기 영상을 이산 결정으로 바꾸므로 가정이 크게 작용합니다. Sobel/Scharr의 gradient는 방향성 특징입니다. 표준 Canny 문헌의 noise 억제용 Gaussian smoothing은 선택적인 전처리이고, OpenCV에서는 호출자가 `cv2.GaussianBlur`로 명시한다는 API 경계를 지켜야 합니다. 그 입력을 받은 `cv2.Canny`는 gradient, NMS, double threshold, hysteresis로 edge 연결을 판정합니다. global/Otsu threshold는 한 값으로 분리된 분포를 가정하고, adaptive는 지역 조명에 대응하지만 texture를 과분할할 수 있습니다. morphology는 structuring element 크기를 도메인의 최소 의미 구조와 맞춰야 하며, opening/closing은 실제 결함을 제거하거나 합칠 위험이 있습니다. 마지막 mask에서 boundary 길이·계층은 contours, object statistics는 connected components로 나눠 다룹니다. 이 프로젝트의 continuity도 Canny component 기반 heuristic이므로 label 기반 classifier 결과와 혼동하지 않습니다.”

## 완료 기준

- [ ] **이해**: Sobel·Scharr·Laplacian·Canny의 미분/단계 차이와 threshold 가정을 설명했다.
- [ ] **구현**: 합성 샘플에 gradient, Canny, Otsu/adaptive, morphology를 실행하고 components 통계를 출력했다.
- [ ] **해석**: opening/closing 또는 threshold가 객체 개수·면적을 바꾼 한 사례와 원인을 기록했다.
- [ ] **설명**: contour와 connected components의 질문 차이, heuristic과 classifier 평가의 경계를 말했다.
