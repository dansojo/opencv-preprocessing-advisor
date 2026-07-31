# Day 4 - 노이즈와 필터링

필터는 지저분한 이미지를 자동으로 깨끗하게 하는 버튼이 아니다. 노이즈가 어떤 과정에서 생겼다고 가정하는지, 어떤 구조를 지우는지, 처리 시간이 허용되는지를 명시하는 모델 선택이다.

## 오늘 답해야 할 핵심 질문

- convolution kernel과 border 처리는 어떤 공간 정보를 섞는가?
- Gaussian, Median, Bilateral은 각각 어떤 noise model에 맞는가?
- kernel size, sigma, diameter는 어떤 trade-off를 바꾸는가?
- texture loss와 oversmoothing을 지표·확대 관찰로 어떻게 발견하는가?
- 빠른 필터와 edge-preserving 필터 중 무엇을 언제 선택하는가?

## 개념과 원리

공간 필터는 한 화소를 주변 이웃의 함수로 바꾼다. 선형 convolution은 `J(x,y)=Σ K(i,j) I(x-i,y-j)`로 쓸 수 있고, kernel이 넓을수록 더 멀리 있는 값을 섞는다. 가장자리에는 이웃이 없으므로 OpenCV는 기본 border 규칙으로 경계를 확장한다. border 반사·복제 방식은 얇은 검사 영역에서 결과에 보일 수 있으므로, crop 경계가 의미 있는 실험은 borderType까지 기록한다.

**Gaussian** blur는 거리 중심의 가중 평균이다. 가산적인, 대략 정규분포형 고주파 noise를 부드럽게 낮추는 합리적 출발점이다. `kernel_size`는 양의 홀수, `sigma`는 가중치 폭이다. sigma가 0이면 OpenCV가 kernel로부터 추정한다. 하지만 평균 계열 필터는 한 점의 강한 outlier를 주변으로 퍼뜨릴 수 있고, 가는 선·미세 texture의 gradient도 약하게 만든다.

**Median** filter는 이웃의 평균이 아니라 중앙값을 쓴다. salt-and-pepper 같은 impulse noise, 즉 드문 화소가 0이나 255로 튀는 경우에 outlier의 영향이 작다. 선형 convolution이 아니므로 Gaussian의 sigma 같은 파라미터는 없고 홀수 kernel 크기가 핵심이다. 정상 texture가 작고 촘촘하면 median도 그것을 평평하게 만들 수 있다.

**Bilateral** filter는 가까운 이웃일수록, 또 색/밝기 차이가 작을수록 크게 가중한다. 공간 거리와 range 거리의 두 가중치를 곱하므로 경계를 넘는 평균화를 줄이는 edge-preserving 시도다. `diameter`는 이웃 크기, `sigma_color`는 얼마나 다른 색까지 같은 표면으로 볼지, `sigma_space`는 얼마나 멀리 볼지다. 큰 값은 더 강한 smoothing을 주지만 비용이 커지고 경계 양쪽의 texture가 뭉개질 수 있다.

필터의 실패는 흔히 “noise가 줄었지만 판단에 필요한 세부도 줄었다”는 형태다. 이것이 oversmoothing이다. 진단의 noise estimate가 내려가도 sharpness·edge continuity가 크게 하락하거나, 작은 균열/타일 texture가 사라지면 좋은 결과가 아니다. 한 장의 필터 전후 품질은 heuristic 관찰이고, 실제 classifier 영향은 독립된 라벨 평가에서 확인한다.

## OpenCV API와 파라미터

[transforms.py](../../src/opencv_preprocessing_advisor/transforms.py)는 세 필터를 `apply_gaussian`, `apply_median`, `apply_bilateral`로 제공하고 입력 BGR `uint8` 및 유효 파라미터를 검사한다. [test_transforms.py](../../tests/test_transforms.py)는 impulse noise가 Median 뒤 감소하는 방향성을 검증한다. [diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py)의 median residual·Laplacian variance·에지 지표로 전후 부작용을 기록한다.

| 필터 | 핵심 파라미터 | 잘 맞는 가정 | 비용/위험 |
| --- | --- | --- | --- |
| Gaussian | 홀수 `kernel_size`, `sigma` | 가산적·완만한 고주파 noise | 경계와 texture도 평균화한다. |
| Median | 홀수 `kernel_size` | salt-and-pepper impulse noise | 작은 점·가느다란 texture가 사라질 수 있다. |
| Bilateral | `diameter`, `sigma_color`, `sigma_space` | 경계를 넘는 smoothing을 줄이고 싶은 noise | 일반 blur보다 느리고 큰 sigma에서 뭉개질 수 있다. |
| `cv2.filter2D` | kernel, borderType | 명시적 convolution 실험 | kernel 정규화·경계 처리를 직접 책임진다. |

`_validate_odd_kernel`가 4 같은 짝수 크기를 거부하는 이유는 중심 화소가 모호하기 때문이다. Bilateral의 모든 파라미터는 양수여야 한다. [test_pipelines.py](../../tests/test_pipelines.py)는 `clahe-bilateral` 파이프라인의 순서·출력 계약을, [scoring.py](../../src/opencv_preprocessing_advisor/scoring.py)는 과도한 sharpness 손실에 대한 경고를 확인할 수 있는 연결점이다.

## 언제 사용하고 피하는가

이미지에 드문 검정/흰 점이 뚜렷하면 Median을 작은 홀수 kernel부터 시험한다. 고르게 거친 고주파가 있고 경계 보존이 덜 중요하면 Gaussian이 빠른 기준선이다. 제품 외곽이나 결함 경계를 유지하면서 색/밝기 차이를 넘는 smoothing을 줄이고 싶으면 Bilateral을 후보로 둔다. 세 경우 모두 “무슨 noise인가”라는 가정이 틀리면 기대한 효과가 나오지 않는다.

피해야 할 패턴은 가장 깨끗해 보이는 결과만 고르는 것이다. Bilateral의 큰 diameter와 큰 sigma는 화면상 매끈하지만 texture descriptor에 필요한 미세 반복 무늬를 없앨 수 있다. Gaussian kernel 15를 기본값으로 고정하거나 Median 7을 모든 이미지에 적용하는 것도 위험하다. **시각적 개선은 분류 성능 개선을 보장하지 않는다**. 이 프로젝트의 점수는 실험 우선순위 heuristic이고, label을 본 classifier 성능은 아니다.

## 프로젝트 코드 연결

- [필터 구현: transforms.py](../../src/opencv_preprocessing_advisor/transforms.py)는 Gaussian, Median, Bilateral의 입력과 파라미터 검증을 담당한다.
- [진단 구현: diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py)는 noise estimate·sharpness·edge continuity를 비교한다.
- [변환 테스트: test_transforms.py](../../tests/test_transforms.py)는 Median이 impulse pixel을 줄이는지를 확인한다.
- [파이프라인 테스트: test_pipelines.py](../../tests/test_pipelines.py)는 설정된 변환 순서가 결과 계약을 지키는지 확인한다.
- Notion 절대 링크: [transforms.py (main)](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/transforms.py).

`apply_gaussian`의 sigma=0은 자동 선택이라는 편의값이다. 재현 가능한 비교에서는 실제 전달한 kernel·sigma를 기록한다. `apply_bilateral`에서 diameter는 이웃의 지름이고 sigma_color·sigma_space는 서로 다른 축이므로 하나를 바꾼 효과를 다른 하나의 효과로 설명하지 않는다.

## 직접 실험

다음은 합성 타일에 두 종류의 noise를 인위적으로 넣고 세 필터와 시간을 비교하는 최소 예제다. 출력 수치가 아닌 방향과 실행 시간 범주를 관찰한다.

```python
from pathlib import Path
from time import perf_counter

import numpy as np

from opencv_preprocessing_advisor.diagnostics import analyze_image
from opencv_preprocessing_advisor.io import decode_image
from opencv_preprocessing_advisor.transforms import apply_bilateral, apply_gaussian, apply_median

image = decode_image(Path("data/samples/synthetic-tile.png"))
rng = np.random.default_rng(42)
impulse = image.copy()
mask = rng.random(image.shape[:2]) < 0.02
impulse[mask] = 255
gaussian_noise = np.clip(image.astype(np.int16) + rng.normal(0, 12, image.shape), 0, 255).astype(np.uint8)

filters = {
    "Gaussian": lambda x: apply_gaussian(x, kernel_size=5, sigma=1.2),
    "Median": lambda x: apply_median(x, kernel_size=5),
    "Bilateral": lambda x: apply_bilateral(x, diameter=7, sigma_color=45, sigma_space=45),
}
for source_name, source in {"impulse": impulse, "gaussian": gaussian_noise}.items():
    print("\nsource:", source_name)
    for name, operation in filters.items():
        start = perf_counter(); result = operation(source); elapsed = perf_counter() - start
        d = analyze_image(result)
        print(name, f"{elapsed * 1000:.1f} ms", "noise=", round(d.noise_estimate, 2),
              "sharpness=", round(d.sharpness, 2))
```

Median은 impulse 입력에서, Gaussian은 가산 noise 입력에서 먼저 비교한다. Bilateral은 같은 입력에서 경계를 확대해 본다. 그 뒤 kernel 3/5/9 또는 bilateral sigma를 한 축씩 바꾸며 시간도 기록한다. 실행 시간은 기기·해상도에 의존하므로 절대 ms 비교 대신 “같은 입력·같은 기기에서 상대적으로”라고 적는다.

## 예상 결과와 해석

| 관찰 | 예상 결과 | 해석과 다음 질문 |
| --- | --- | --- |
| impulse + Median | 튀는 0/255 점과 noise estimate가 줄기 쉬움 | 작은 texture까지 사라지는지 확대 확인한다. |
| 가산 noise + Gaussian | 균일한 거침이 완화될 수 있음 | kernel이 커질수록 sharpness 하락을 기록한다. |
| Bilateral | 강한 경계를 상대적으로 남길 수 있음 | sigma/diameter가 크면 oversmoothing과 시간이 늘 수 있다. |
| 큰 kernel | noise 감소와 함께 edge continuity 감소 가능 | ‘깨끗함’보다 분석 목적의 구조 보존을 우선한다. |
| timing | Bilateral이 기준 blur보다 느린 경향 | 배치·UI 응답 시간 예산 안에서 파라미터를 정한다. |

이 표의 예상은 통계적 보장이 아니다. 실제 표면 texture가 median residual에 잡히면 noise estimate의 변화가 가정과 다를 수 있다. 결과 이미지, 파라미터, sharpness/noise/edge 변화, timing을 한 기록에 묶어 다음 사람이 같은 trade-off를 재현할 수 있게 한다.

## 자주 하는 실수와 디버깅

1. **noise type을 보지 않고 Gaussian 적용**: impulse는 평균 계열에서 번질 수 있다. outlier 위치를 확대해 본다.
2. **짝수 kernel 전달**: 중심이 없는 kernel은 구현에서 `ValueError`다. 3, 5, 7처럼 홀수를 쓴다.
3. **Bilateral sigma를 무한히 키움**: edge preservation도 약해지고 실행 시간·oversmoothing 위험이 커진다.
4. **noise 감소만 기록**: sharpness와 edge continuity, 관심 texture의 확대 crop을 같은 표에 남긴다.
5. **필터 timing을 다른 크기 입력끼리 비교**: 해상도와 warm-up을 고정하고 여러 번 실행해 median 시간을 비교한다.

필터 결과가 예상과 반대라면 입력의 dtype과 noise 생성 방식을 확인하고, noise 없는 원본에도 같은 필터를 걸어 texture loss 기준선을 만든다. project transform은 BGR `uint8` 계약을 강제하므로 float 실험 배열은 명시적으로 변환한다.

## 본인 말로 설명하기

### 1분 설명

“Gaussian은 주변 평균에 가까운 가중합이라 가산 고주파 noise의 빠른 기준선이고, Median은 중앙값이라 salt-and-pepper 같은 impulse에 강합니다. Bilateral은 공간 거리와 색 차이를 함께 고려해 경계를 넘는 smoothing을 줄이지만 더 느립니다. 세 필터 모두 kernel이나 sigma를 키우면 noise는 줄 수 있지만 texture와 edge도 잃을 수 있습니다. 그래서 noise estimate만 보지 않고 sharpness·edge continuity·확대 관찰·시간을 함께 기록합니다. 이 관찰은 heuristic이며 분류 성능은 따로 검증합니다.”

### 깊이 설명

“필터 선택은 noise model의 가정과 손실 함수의 선택입니다. Gaussian convolution은 outlier도 평균에 섞고, Median은 순위 기반이라 드문 이상값의 영향이 작습니다. Bilateral은 range weight로 서로 다른 표면의 혼합을 줄이지만 parameter 세 축과 계산 비용이 있습니다. `transforms.py`는 홀수 kernel과 양수 값을 강제해 명백한 설정 오류를 막고, diagnostics는 residual noise와 Laplacian/edge 변화를 보여 줍니다. 그러나 정상 texture도 고주파이므로 낮은 residual이 품질 보장은 아닙니다. 따라서 합성 noise에서 가정을 검증하고 실제 라벨 데이터에서 원본 대비 성능을 평가해야 합니다.”

## 완료 기준

- [ ] **이해**: Gaussian, Median, Bilateral의 noise-model 가정과 파라미터 역할을 설명했다.
- [ ] **구현**: impulse·가산 noise 합성 입력에 세 필터를 실행하고 kernel/sigma를 한 변수씩 바꿨다.
- [ ] **해석**: noise, sharpness, edge, timing으로 oversmoothing 또는 texture loss 한 사례를 기록했다.
- [ ] **설명**: 빠른 기준선과 edge-preserving 선택의 trade-off, heuristic과 모델 평가의 차이를 말했다.
