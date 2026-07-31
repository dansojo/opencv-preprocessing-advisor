# Day 3 - 밝기와 대비 전처리

오늘은 어두운 이미지에 무조건 대비를 올리지 않는다. normalize, gamma, 전역 histogram equalization, CLAHE가 각각 무엇을 보존·왜곡하는지, 그리고 LAB의 밝기 채널만 조절하는 이유를 실험으로 구분한다.

## 오늘 답해야 할 핵심 질문

- min–max normalize는 어떤 입력 범위에서 유용하고 clipping 신호를 어떻게 감출 수 있는가?
- gamma의 지수 변환은 선형 밝기 조절과 무엇이 다른가?
- global histogram equalization과 CLAHE는 왜 같은 ‘대비 향상’이 아닌가?
- `clipLimit`와 `tileGridSize`는 각각 무엇을 제한하며, grid 수가 커질수록 local tile은 왜 작아지는가?
- BGR 전체가 아니라 **LAB L-channel**을 처리하면 어떤 색 변화 위험을 줄이는가?

## 개념과 원리

밝기 변환은 화소값 함수 `y = f(x)`를 바꾸는 일이다. min–max normalize는 보통 `y = 255 (x-min)/(max-min)`으로 현재 이미지의 최소·최대를 새 범위에 맞춘다. 흐린 장면의 분포를 펼쳐 비교하기에는 편하지만, 이미지마다 다른 기준을 쓰므로 절대 노출 차이를 지운다. 또 0과 255에 몰린 clipping이 있으면 이미 사라진 그림자·하이라이트 세부가 되살아나지 않는다.

gamma는 정규화 값 `x ∈ [0,1]`에 `y = x^γ`를 적용한다. `γ < 1`이면 어두운 중간톤이 상대적으로 밝아지고, `γ > 1`이면 중간톤이 어두워진다. 프로젝트의 auto gamma는 grayscale 평균을 target midpoint 0.5에 가깝게 보내는 gamma를 계산한 뒤 0.5~2.0으로 제한한다. 이것은 보편 정답이 아니라 과도한 곡선을 막는 안전장치다. 어두운 clipping이 심하면 gamma는 없는 세부를 만들지 못하고 noise만 더 보이게 할 수 있다.

global histogram equalization(HE)은 전체 히스토그램의 누적분포를 이용해 밝기 값을 재배치한다. 전역적으로 흐린 영상에서는 효과가 크지만, 한쪽에 조명이 몰린 장면에서는 어두운 영역의 noise를 과장하거나 밝은 부분의 톤을 평평하게 만들 수 있다. CLAHE(Contrast Limited Adaptive Histogram Equalization)는 영상을 작은 타일로 나눠 각 타일의 히스토그램을 평활화하고 경계는 보간한다. 그래서 지역 대비를 다루지만 타일마다 과장된 질감·격자 인상이라는 부작용도 생긴다.

CLAHE의 `clipLimit`는 타일 히스토그램 bin이 과도하게 커지는 것을 잘라 redistribution하는 한계다. 높이면 지역 대비와 noise 증폭 가능성이 함께 커진다. `tileGridSize=(8,8)`는 타일의 픽셀 크기가 아니라 **가로·세로 grid 개수**다. 같은 영상에서 grid 수가 커질수록 각 local tile은 작아지고 더 국소적인 변화에 민감해진다. `tileGridSize=(16,16)`을 “더 큰 타일”이라고 말하는 것은 흔한 반대 해석이다.

컬러 BGR 세 채널에 HE/CLAHE를 독립 적용하면 채널별 mapping이 달라져 색 균형이 흔들릴 수 있다. BGR→LAB로 옮기면 L은 지각적 밝기 축이고 a/b는 색 좌표다. 그러므로 L에만 CLAHE를 적용한 뒤 원래 a/b와 합쳐 BGR로 돌아오면 색조 변화를 줄일 수 있다. 완벽한 색 보존이라는 뜻은 아니며, conversion과 uint8 양자화 때문에 결과를 비교해야 한다.

## OpenCV API와 파라미터

[transforms.py](../../src/opencv_preprocessing_advisor/transforms.py)의 `apply_lab_clahe`는 BGR을 LAB로 바꾸고 L만 분리해 `cv2.createCLAHE(clipLimit=..., tileGridSize=(grid_size, grid_size))`를 적용한다. `apply_auto_gamma`는 LUT를 만들어 모든 BGR 채널에 같은 곡선을 적용한다. [test_transforms.py](../../tests/test_transforms.py)는 최소한 결과가 BGR shape과 `uint8` dtype을 보존하는지 확인한다.

| API | 주요 인자 | 효과 | 실험할 범위 |
| --- | --- | --- | --- |
| `cv2.normalize` | 0, 255, `NORM_MINMAX` | 전체 범위를 늘림 | clipping·상대 노출을 잃지 않는지 먼저 확인 |
| `cv2.LUT` | 256개 `uint8` lookup | gamma 곡선 적용 | gamma 0.5, 1.0, 1.5처럼 방향부터 비교 |
| `cv2.equalizeHist` | grayscale 1채널 | global HE | 조명 불균일, noise 증폭 여부 관찰 |
| `cv2.createCLAHE` | `clipLimit`, `tileGridSize` | 지역 대비 조정 | clipLimit 1~4, grid 4/8/16을 한 변수씩 변경 |
| `cv2.cvtColor` | `BGR2LAB`, `LAB2BGR` | LAB L-channel 처리 | a/b를 유지해 색 변화를 줄임 |

[diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py)로 전후의 local contrast, entropy, clipping, noise를 함께 본다. [test_diagnostics.py](../../tests/test_diagnostics.py)는 지표가 입력 변형에 반응한다는 회귀 증거다. Notion에는 [transforms.py (main)](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/transforms.py)를 절대 링크로 바꿔 넣는다.

## 언제 사용하고 피하는가

normalize는 동일 장면에서 상대적인 구조를 빠르게 보기 좋지만, 카메라 노출 자체가 중요한 검사에서는 원본 통계를 함께 보관한다. gamma는 전체적으로 어둡거나 밝은 중간톤을 부드럽게 바꿀 때 후보가 된다. global HE는 비교적 균일한 조명에서 톤 분포가 좁을 때, CLAHE는 지역적으로 대비가 부족한 경우에 출발점이 된다.

반대로 이미 bright/dark clipping이 높거나, fine texture가 모델의 특징이라면 강한 CLAHE는 위험하다. tileGridSize를 작게 해 너무 국소화하거나 clipLimit를 크게 하면 noise와 tile 경계 인상이 결함처럼 강해질 수 있다. 처리 전후가 더 또렷해 보여도 추천 heuristic은 가설 생성용이다. **시각적 개선은 분류 성능 개선을 보장하지 않는다**. classifier 영향은 라벨 데이터에서 원본과 동일한 split으로 비교한다.

## 프로젝트 코드 연결

- [CLAHE·gamma 구현: transforms.py](../../src/opencv_preprocessing_advisor/transforms.py)는 유효한 양수 파라미터와 BGR 반환을 보장한다.
- [진단 구현: diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py)는 대비·entropy·noise의 전후 변화를 측정한다.
- [변환 테스트: test_transforms.py](../../tests/test_transforms.py)는 LAB CLAHE 출력 계약을 검증한다.
- [진단 테스트: test_diagnostics.py](../../tests/test_diagnostics.py)는 blur/noise 변화에 대한 지표 방향을 검증한다.

프로젝트의 `grid_size=8`은 `tileGridSize=(8, 8)`이라는 grid 수다. 이미지가 640×480이라면 각 타일의 대략적 크기는 80×60 픽셀이다. 같은 이미지에서 16×16 grid라면 대략 40×30으로 더 작아져, 더 세밀한 조명 변화와 noise에 반응한다.

## 직접 실험

아래 코드는 합성 샘플에 gamma, global HE, LAB CLAHE를 나란히 적용하고 진단값을 출력한다. 결과 파일은 비교용이며 원본을 덮어쓰지 않는다.

```python
from pathlib import Path

import cv2

from opencv_preprocessing_advisor.diagnostics import analyze_image
from opencv_preprocessing_advisor.io import decode_image, encode_png
from opencv_preprocessing_advisor.transforms import apply_auto_gamma, apply_lab_clahe, normalize_uint8

image = decode_image(Path("data/samples/synthetic-tile.png"))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
global_he = cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR)
candidates = {
    "normalize": normalize_uint8(image),
    "gamma": apply_auto_gamma(image),
    "global-he": global_he,
    "lab-clahe": apply_lab_clahe(image, clip_limit=2.0, grid_size=8),
}
out = Path("output/day03")
out.mkdir(parents=True, exist_ok=True)
for name, result in candidates.items():
    d = analyze_image(result)
    print(name, "local_contrast=", round(d.local_contrast, 2),
          "entropy=", round(d.entropy, 3), "noise=", round(d.noise_estimate, 2))
    (out / f"{name}.png").write_bytes(encode_png(result))
```

첫 실행은 `clipLimit=2.0`, `tileGridSize=(8,8)`로 고정한다. 그 다음 **한 번에 하나만** 바꾼다: clipLimit 1→4, grid_size 4→8→16. 각 결과에서 어두운 홈, 타일 경계, 색조, noise처럼 보이는 점을 원본과 확대 비교한다. `global-he`는 grayscale 처리 후 BGR로 되돌린 예제이므로, 색을 보존한 결과가 아니라 전역 톤 대비 기준선이다.

## 예상 결과와 해석

| 관찰 | 예상 결과 | 해석과 다음 질문 |
| --- | --- | --- |
| normalize | 전체 min/max가 0/255 쪽으로 펴짐 | 기존 clipping 여부와 절대 밝기 의미를 별도 기록한다. |
| gamma | 중간톤 중심의 완만한 변화 | 평균이 목표에 가까워져도 shadow 세부가 복원됐는지 따로 본다. |
| global HE | 전체 히스토그램이 넓어질 수 있음 | 균일하지 않은 조명에서는 noise·강한 영역을 과장할 수 있다. |
| LAB CLAHE | 지역 대비가 커질 수 있음 | a/b를 유지해 색 변화는 줄지만 L 양자화와 tile artifact는 확인한다. |
| grid 16 | 더 작은 local tile로 더 국소적 변화 | ‘큰 grid’가 ‘큰 타일’이 아님을 결과와 함께 설명한다. |

특정 수치가 가장 높다고 그 설정을 고르지 않는다. local contrast와 entropy 상승이 noise estimate 상승, 에지의 인공적 강조, clipping 증가와 동반되는지 확인한다. 보고서에는 입력 파일·파라미터·원본 대비 변화·육안 관찰을 모두 남겨 다른 사람이 같은 결론을 재검토할 수 있게 한다.

## 자주 하는 실수와 디버깅

1. **`tileGridSize`를 픽셀 타일 크기로 이해**: OpenCV에서 이는 grid의 개수다. grid가 클수록 local tile은 작다.
2. **BGR 세 채널을 각각 equalize**: 색조가 바뀔 수 있다. 밝기 중심 목적이면 LAB L-channel을 먼저 실험한다.
3. **`clipLimit`를 높이면 항상 좋다고 생각**: 히스토그램 제한 완화는 대비뿐 아니라 noise 증폭도 키운다.
4. **HE 결과를 RGB 컬러 결과와 비교**: 예제의 global HE는 grayscale 기준선이다. 색 정보의 보존 여부를 따로 적는다.
5. **gamma가 노출 복구라고 주장**: gamma는 tone mapping이며 포화로 소실된 세부를 복원하지 않는다.

디버깅할 때는 LAB 변환 뒤 `L`, `a`, `b`를 분리해 L만 달라졌는지 확인하고, 색이 변하면 입력이 BGR인지와 `COLOR_LAB2BGR` 변환을 확인한다. 출력의 dtype이 float이면 프로젝트 계약에서 거부되므로 LUT·normalize 뒤 `uint8`인지 확인한다.

## 본인 말로 설명하기

### 1분 설명

“normalize는 현재 영상의 최소·최대를 0~255로 늘리고, gamma는 `x^gamma` 곡선으로 중간톤을 바꿉니다. global HE는 전체 히스토그램을, CLAHE는 local tile 히스토그램을 조정합니다. CLAHE의 `clipLimit`는 과도한 bin을 제한하고 `tileGridSize`는 타일 크기가 아니라 grid 개수라서 값이 클수록 각 타일은 작아집니다. 컬러에서는 BGR 채널을 따로 조정하면 색이 바뀔 수 있어 LAB의 L만 처리합니다. 하지만 보기 좋은 대비와 classifier 성능은 같은 측정이 아니므로 라벨 평가가 필요합니다.”

### 깊이 설명

“밝기 전처리는 픽셀 분포를 재매핑하는 선택입니다. normalize는 장면별 동적 범위를 같게 하지만 노출 차이와 clipping 신호를 희석할 수 있습니다. gamma는 target midpoint를 향한 비선형 LUT이고, equalization은 CDF 기반의 재배치입니다. CLAHE는 local histogram을 쓰며 clipLimit와 grid 수라는 두 축에서 국소성·noise 증폭의 trade-off를 만듭니다. 이 저장소는 `apply_lab_clahe`에서 BGR→LAB 변환 뒤 L만 바꾸고 a/b를 결합해 색 좌표를 되도록 유지합니다. 결과는 local contrast, entropy, noise, clip과 확대 관찰을 함께 해석하고, heuristic의 순위와 학습 모델의 일반화 성능을 혼동하지 않습니다.”

## 완료 기준

- [ ] **이해**: normalize, gamma, global HE, CLAHE의 서로 다른 mapping과 한계를 설명했다.
- [ ] **구현**: 합성 샘플에 gamma·global HE·LAB CLAHE를 실행하고 `clipLimit`와 `tileGridSize`를 한 변수씩 바꿨다.
- [ ] **해석**: grid 수가 커지면 local tile이 작아지는 관찰과 noise/색 부작용을 기록했다.
- [ ] **설명**: LAB L-channel 선택 이유와 heuristic 대 classifier 평가의 경계를 1분 안에 말했다.
