# Day 1 - 이미지 데이터와 OpenCV 기초

오늘의 목표는 “이미지를 보기 전에 배열로 읽을 수 있는가”이다. 이 프로젝트의 모든 전처리는 `uint8` BGR 배열을 입력 계약으로 삼는다. 따라서 색이 이상하거나 밝기가 깨졌을 때 곧바로 필터를 고르는 대신, shape·dtype·색 순서·범위를 먼저 확인하는 습관을 만든다.

## 오늘 답해야 할 핵심 질문

- 디지털 이미지는 왜 `height × width × channel` NumPy 배열인가?
- `uint8` 계산에서 overflow와 clipping은 각각 어떤 결과를 만드는가?
- OpenCV의 BGR을 matplotlib/Pillow의 RGB처럼 보여 주면 왜 색이 바뀌는가?
- HSV와 LAB는 어떤 질문에 더 적합하며, 변환 자체가 왜 공짜가 아닌가?
- 한글 경로에서도 안전하게 이미지를 읽고 저장하려면 왜 `imread` 대신 바이트 디코딩을 쓰는가?

## 개념과 원리

컬러 래스터 이미지는 각 화소(pixel)에 세 개의 수를 둔 3차원 배열이다. 행은 높이 `H`, 열은 너비 `W`, 마지막 축은 채널 `C`이므로 일반적인 입력의 shape은 `(H, W, 3)`이다. 이 저장 순서와 의미는 별개다. OpenCV의 `cv2.IMREAD_COLOR`는 채널을 **BGR** 순서로 돌려준다. 반면 웹·Pillow·matplotlib 예시는 대개 RGB이다. 같은 세 수라도 해석 순서를 바꾸면 빨강과 파랑이 교환된다.

프로젝트의 `validate_bgr_image`는 비어 있지 않은 `(H, W, 3)` `np.uint8`만 허용한다. `uint8`의 표현 범위는 0부터 255까지다. 즉 픽셀을 실수처럼 다룰 수 있다고 가정하면 안 된다. 예를 들어 NumPy의 `uint8` 배열에서 `250 + 10`은 산술 overflow로 4가 될 수 있다. 반대로 OpenCV의 `cv2.add`처럼 포화 연산(saturation)을 쓰면 결과를 255로 clipping한다. 둘 다 밝아진다는 의도와는 다른 세부 결과를 만들 수 있으므로, 계산 전 `float32`로 올리고 마지막에 `clip(0, 255).astype(np.uint8)` 하는 이유를 이해해야 한다.

정규화한 실수 영상은 보통 `x ∈ [0, 1]`로 쓰고, 감마나 선형 변환을 적용한 뒤 `y_8 = round(255 × clip(y, 0, 1))`로 되돌린다. 이때 입력의 최소·최대가 이미 0과 255에 닿아 있다면 무조건적인 min–max 정규화는 clipping 정보를 감출 수 있다. “범위를 넓혔다”와 “센서가 표현하지 못한 정보를 복원했다”는 전혀 다른 말이다.

색 공간은 작업 질문에 맞춘 좌표계다. BGR/RGB는 화면 표시와 저장에 편하고, grayscale은 밝기 구조만 빠르게 볼 때 편하다. HSV의 H는 색상, S는 채도, V는 명도에 가까워 색 영역 마스킹에 유용하지만 H의 원형성(0과 179가 이웃)을 조심해야 한다. **LAB**는 지각적 밝기 축 L과 색 축 a/b를 분리한다. 그래서 밝기만 조절하며 색 변화를 줄이고 싶은 경우 L을 다룬다. 단, `cvtColor`는 정보를 ‘더하는’ 것이 아니라 다른 표현으로 옮기는 연산이며, 양자화와 비용이 있다.

resize도 내용 보존 여부를 좌우한다. 축소에는 `INTER_AREA`가 평균화에 가까워 aliasing을 줄이는 출발점이고, 확대에는 `INTER_LINEAR` 또는 더 부드러운 `INTER_CUBIC`가 흔하다. 하지만 보간이 실제 결함의 미세 구조를 복원하는 것은 아니다. 분류기 입력 크기에 맞춘 resize는 데이터 파이프라인의 한 단계이며, 원본 관찰용 이미지까지 무심코 축소해서 판단하면 안 된다.

## OpenCV API와 파라미터

`cv2.imread`가 OS/빌드 조합의 유니코드 경로에서 실패할 수 있어 프로젝트는 `np.fromfile`로 바이트를 읽고 `cv2.imdecode`로 해석한다. [io.py](../../src/opencv_preprocessing_advisor/io.py)는 `decode_image`, `decode_image_bytes`, `encode_png`, `validate_bgr_image`에 이 계약을 한 곳으로 모아 둔다. [test_io.py](../../tests/test_io.py)는 `표면_이미지.png` 왕복으로 Unicode I/O를 검증한다.

| API | 핵심 인자 | 사용할 때 | 주의점 |
| --- | --- | --- | --- |
| `cv2.cvtColor(image, code)` | `COLOR_BGR2RGB`, `COLOR_BGR2HSV`, `COLOR_BGR2LAB` | 표시 목적 또는 색 공간별 분석 | 반환 dtype/shape은 대개 유지되지만 채널 의미가 변한다. |
| `cv2.resize(image, dsize, interpolation=...)` | 목표 `(width, height)`, 보간법 | 모델 입력 크기·미리보기 | OpenCV `dsize`는 shape과 달리 W,H 순서다. |
| `cv2.convertScaleAbs` | `alpha`, `beta` | 표시용 선형 스케일 | 절댓값과 포화 변환이 포함되어 분석용 원본을 대체하지 않는다. |
| `cv2.normalize` | `alpha=0`, `beta=255`, `NORM_MINMAX` | 비교 실험용 범위 맞춤 | 장면 간 밝기 차이와 clipping 신호가 사라질 수 있다. |

[transforms.py](../../src/opencv_preprocessing_advisor/transforms.py)의 `normalize_uint8`와 `apply_gray_bgr`은 이 입력·출력 계약을 유지하는 작은 변환 예다. [test_transforms.py](../../tests/test_transforms.py)에서 BGR shape과 dtype 보존을 확인한다. 이 링크들은 문서의 개념이 실제 구현에서 어디에 있는지 추적하는 출발점이다.

## 언제 사용하고 피하는가

처음 받은 단일 이미지는 먼저 `shape`, `dtype`, 최소/최대값, BGR 채널 평균을 기록한다. 화면에 올리기 위한 RGB 변환은 필요하지만, 분석 함수에 RGB를 다시 넣으면 프로젝트 진단이 잘못된 색으로 계산된다. HSV는 특정 색 범위를 분리할 때, LAB는 밝기와 색을 분리해 비교할 때 선택한다. 모두를 한꺼번에 변환해 지표를 많이 만들기보다, “무엇을 구분하려는가”라는 질문 하나에 맞춘다.

피해야 할 경우도 명확하다. 원본이 16-bit 과학·의료 영상인데 8-bit로 무조건 잘라내면 정보가 사라진다. 이 저장소의 공개 샘플과 서비스 계약은 `uint8` BGR이므로 그 밖의 dtype은 명시적으로 변환 규칙을 정하고 별도로 검증해야 한다. RGB로 읽었다고 생각하는 라이브러리와 섞어 쓰는 경우에도 변환 지점을 한 번만 두고 이름에 `bgr`, `rgb`, `lab`를 붙인다.

화면에서 더 선명하거나 더 밝아 보인다는 관찰은 탐색의 단서일 뿐이다. 이 프로젝트의 단일 이미지 진단과 추천 점수는 heuristic이며, **시각적 개선은 분류 성능 개선을 보장하지 않는다**. 분류기 영향은 라벨이 있는 분리된 데이터와 누수 없는 평가에서 따로 확인한다.

## 프로젝트 코드 연결

- [입출력 계약: io.py](../../src/opencv_preprocessing_advisor/io.py)는 BGR `uint8`과 빈 배열 오류를 강제한다.
- [변환 구현: transforms.py](../../src/opencv_preprocessing_advisor/transforms.py)는 `normalize_uint8`와 LAB 기반 변환의 반환 형태를 보장한다.
- [입출력 회귀 테스트: test_io.py](../../tests/test_io.py)는 Unicode 경로와 바이트 디코딩을 검증한다.
- [변환 회귀 테스트: test_transforms.py](../../tests/test_transforms.py)는 변환 뒤에도 shape/dtype이 유지되는지 확인한다.
- Notion에 붙일 때는 상대 링크 대신 [io.py (main)](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/io.py)처럼 GitHub `main` 링크를 사용한다.

`decode_image`는 경로 존재 여부를 먼저 확인하고, 바이트 배열을 `cv2.imdecode(..., IMREAD_COLOR)`에 전달한 뒤 같은 검증 함수를 호출한다. 이 순서 덕분에 업로드 바이트와 파일 경로가 같은 BGR 계약으로 합류한다. API를 호출하는 위치가 여러 곳이어도 이미지 계약을 각각 복제하지 않는 이유다.

## 직접 실험

저장소 루트에서 다음 예제를 실행한다. 공개된 `data/samples/synthetic-tile.png`만 사용하므로 개인 데이터나 MVTec 원본은 필요 없다. 출력 이미지는 임시 폴더에 두고 관찰 후 지워도 된다.

```python
from pathlib import Path

import cv2
import numpy as np

from opencv_preprocessing_advisor.io import decode_image, encode_png

source = Path("data/samples/synthetic-tile.png")
image_bgr = decode_image(source)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

print("BGR:", image_bgr.shape, image_bgr.dtype, image_bgr.min(), image_bgr.max())
print("channel means (B,G,R):", image_bgr.mean(axis=(0, 1)).round(1))
print("HSV/H range:", image_hsv[:, :, 0].min(), image_hsv[:, :, 0].max())
print("LAB/L range:", image_lab[:, :, 0].min(), image_lab[:, :, 0].max())

# uint8 overflow와 clipping을 비교한다.
pixel = np.array([[[250, 250, 250]]], dtype=np.uint8)
print("NumPy uint8 + 10:", (pixel + 10)[0, 0])
print("cv2.add + 10:", cv2.add(pixel, 10)[0, 0])

small = cv2.resize(image_bgr, (image_bgr.shape[1] // 2, image_bgr.shape[0] // 2),
                   interpolation=cv2.INTER_AREA)
Path("output/day01").mkdir(parents=True, exist_ok=True)
Path("output/day01/half.png").write_bytes(encode_png(small))
```

같은 파일을 matplotlib으로 표시한다면 `image_bgr`가 아니라 `image_rgb`를 넘긴다. 이어서 `INTER_NEAREST`, `INTER_AREA`, `INTER_LINEAR`로 각각 축소·확대해 타일 경계와 얇은 선이 어떻게 바뀌는지 비교한다. 관찰 표에는 숫자를 외워 쓰지 말고 실제 실행 환경에서 나온 범위와 해상도를 기록한다.

## 예상 결과와 해석

| 관찰 | 예상 결과 | 해석과 다음 질문 |
| --- | --- | --- |
| 배열 계약 | `(높이, 너비, 3)`, `uint8`, 값 0~255 | 서비스 함수에 바로 넣을 수 있는 BGR 입력이다. 범위 밖 값이면 변환 경로를 찾는다. |
| 색 표시 | BGR을 RGB로 바꾸지 않으면 빨강·파랑이 뒤바뀜 | 모델/진단용 배열과 표시용 배열의 변수명을 분리한다. |
| 산술 | NumPy 덧셈은 wrap-around, `cv2.add`는 255 포화 | 밝기 조절의 수학과 라이브러리 동작을 구분한다. |
| resize | 축소 `INTER_AREA`는 거친 aliasing이 덜할 수 있음 | 보간 선택은 목적·배율·실제 특징 보존을 함께 본다. |
| LAB L | L은 밝기 축, a/b는 색 축 | 이후 CLAHE에서 색 축을 유지할 근거가 된다. |

숫자 하나만으로 이미지가 “정상”이라고 결론내리지 않는다. 예를 들어 최소값 0과 최대값 255는 넓은 톤 범위일 수도, 그림자·하이라이트가 잘린 clipping일 수도 있다. Day 2의 진단으로 평균·대비·clipping 비율·에지 정보를 함께 본다.

## 자주 하는 실수와 디버깅

1. **RGB라고 가정한 BGR**: 색이 파랗거나 붉게 보이면 `cvtColor` 호출 전후의 변수명과 표시 라이브러리를 확인한다. 진단 함수에는 BGR을 전달한다.
2. **`float32`를 그대로 서비스에 전달**: 현재 `validate_bgr_image`는 의도적으로 거부한다. 분석용 실수 계산 후 명시적으로 clipping·캐스팅할지, 별도 파이프라인을 만들지 결정한다.
3. **`image.shape[:2]`를 `resize`에 그대로 전달**: shape은 H,W지만 `dsize`는 W,H다. 비정사각 이미지에서 특히 눈에 띈다.
4. **한글 파일명 실패를 조용히 무시**: `cv2.imread`가 `None`을 반환하는지 확인하고, 프로젝트의 `decode_image`를 사용한다. `test_unicode_path_round_trip`이 지키는 이유다.
5. **정규화를 품질 향상으로 해석**: 범위 재매핑은 표시 대비를 바꾼다. clipping과 레이블 성능은 독립된 관찰·평가 항목이다.

문제가 생기면 `print(image.dtype, image.shape, image.min(), image.max())`를 가장 먼저 남긴다. 이어서 BGR·RGB 두 표시를 나란히 보되, 원본 배열을 덮어쓰지 않는다. 실패 입력은 작은 합성 배열로 재현해 `validate_bgr_image` 테스트처럼 계약 자체를 검사한다.

## 본인 말로 설명하기

### 1분 설명

“OpenCV에서 컬러 이미지는 보통 `(H, W, 3)` 모양의 `uint8` BGR 배열입니다. 그래서 0~255 범위와 BGR 순서를 확인하지 않으면 색과 밝기 처리가 모두 흔들립니다. NumPy `uint8` 덧셈은 overflow할 수 있지만 OpenCV 포화 덧셈은 255에서 멈추므로 같은 ‘밝게 하기’라도 결과가 다릅니다. 표시할 때만 RGB로 바꾸고, 분석 파이프라인에는 BGR 계약을 유지합니다. 이 프로젝트는 Unicode 경로를 바이트로 읽어 `imdecode`하고, LAB처럼 밝기와 색을 분리해야 할 때만 색 공간을 바꿉니다.”

### 깊이 설명

“이미지 데이터의 기본 단위는 화소 값이지만, 실무에서 더 중요한 것은 계약입니다. 이 저장소의 `io.py`는 입력을 3채널 `uint8` BGR로 제한해 후속 진단과 transform이 같은 가정을 공유하게 합니다. `uint8`에는 256개 값만 있으므로 중간 계산을 어디서 실수로 올리고 어디서 양자화하는지가 결과를 바꿉니다. BGR/RGB는 채널 순서, HSV는 색상 기반 분할, LAB는 밝기와 색 분리를 위한 표현입니다. resize 보간도 특징을 새로 만들지 않으며, 표시가 좋아졌다는 사실을 모델 효과라고 부를 수 없습니다. 따라서 shape/dtype/range를 기록하고, 관찰용 RGB와 분석용 BGR을 분리한 뒤, 다음 단계에서 여러 진단 지표와 라벨 기반 평가를 연결합니다.”

## 완료 기준

- [ ] **이해**: `(H, W, 3)` BGR `uint8` 계약과 0~255 범위를 overflow·clipping 차이까지 설명했다.
- [ ] **구현**: `decode_image`로 합성 샘플을 읽고 BGR/RGB/HSV/LAB 배열의 shape·dtype·범위를 출력했다.
- [ ] **해석**: resize와 색 공간 변환이 정보를 보장해 복원하지 않는다는 점을 관찰 기록으로 남겼다.
- [ ] **설명**: 1분 설명을 말하고, 왜 heuristic 관찰과 분류기 평가는 분리되는지 답했다.
