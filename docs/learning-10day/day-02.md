# Day 2 - 이미지 상태 진단

전처리는 “좋아 보이는 필터”를 고르는 일이 아니라 현재 이미지의 실패 징후를 측정하고 가설을 세우는 일이다. 오늘은 프로젝트의 model-free diagnostics가 무엇을 재고, 무엇을 말해 주지 않는지 배운다.

## 오늘 답해야 할 핵심 질문

- 평균 밝기와 clipping ratio는 어떤 서로 다른 문제를 포착하는가?
- global/local contrast, entropy, sharpness, noise는 왜 한 줄의 품질 점수로 합칠 수 없는가?
- Laplacian variance와 median residual은 어떤 가정 아래에서 유용한가?
- edge density와 edge continuity는 왜 구조 보존의 단서이지 정답이 아닌가?
- heuristic 추천과 학습된 classifier의 성능 평가는 왜 구분해야 하는가?

## 개념과 원리

좋은 진단은 이미지마다 숫자를 많이 출력하는 일이 아니라, 다음 실험을 좁혀 주는 관찰이다. 프로젝트의 `ImageDiagnostics`는 평균 밝기, 어두움/밝음 clipping 비율, 전역·지역 대비, entropy, sharpness, noise estimate, 조명 불균일, edge density·continuity, colorfulness, saturation spread를 함께 기록한다. 같은 사진도 노출 부족·블러·센서 노이즈·조명 그라데이션 중 무엇이 문제냐에 따라 필요한 처리와 위험이 달라진다.

평균 밝기 `μ = (1/N) Σ I(x,y)`는 전체 톤의 중심을 요약한다. 그러나 평균이 128 근처라고 해서 정상 노출이라는 뜻은 아니다. 어두운 절반과 밝은 절반이 섞여도 평균은 중간일 수 있다. `dark_clip_ratio = P(I ≤ 5)`, `bright_clip_ratio = P(I ≥ 250)`는 양 끝에서 정보가 눌렸을 가능성을 보여 준다. 단, 검은 배경이나 흰 제품 자체가 의도된 장면이라면 높은 비율이 오류는 아니다.

전역 대비는 grayscale 표준편차 `σ(I)`이고, 지역 대비는 16×16 타일 표준편차의 평균이다. 전자는 전체 분포 폭, 후자는 국소 패턴의 톤 변화를 본다. entropy는 히스토그램 확률 `p_i`에 대해 `H = -Σ p_i log₂ p_i`로 계산한다. 값이 높으면 많은 밝기 구간을 쓴다는 뜻이지만, 균일한 고주파 noise도 entropy를 올릴 수 있다. 따라서 entropy 증가는 자동으로 정보 증가가 아니다.

sharpness는 이 구현에서 `Var(∇²I)`, 즉 Laplacian 응답의 분산이다. 에지가 뚜렷하면 높아지는 경향이 있으나 질감·noise에도 반응한다. noise는 3×3 Median 필터로 얻은 부드러운 영상과 원본의 residual RMS, `sqrt(mean((I - median(I))²))`로 근사한다. impulse noise에는 유용하지만 정상적인 미세 질감도 residual로 보일 수 있다. 낮은 noise estimate가 무조건 좋은 결과가 아닌 이유다.

조명 불균일은 큰 Gaussian blur를 배경 근사로 보고 `std(background) / mean(background)`을 계산한다. edge density는 median 기반 Canny로 나온 에지 화소 비율, edge continuity는 그 중 8-연결 성분의 면적이 8 이상인 에지 비율이다. 흐림이면 sharpness와 continuity가 낮아질 수 있지만, 단순한 평면 이미지도 원래 낮다. colorfulness는 R-G와 Y-B 차이의 분포를 사용하고, HSV의 saturation spread는 채도 분산을 기록한다. 색이 많은 이미지가 언제나 더 좋은 입력이라는 규칙은 없다.

## OpenCV API와 파라미터

[diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py)의 `analyze_image`는 먼저 BGR을 grayscale과 HSV로 변환하고 모든 지표를 한 dataclass로 반환한다. [models.py](../../src/opencv_preprocessing_advisor/models.py)에는 이름이 고정된 `ImageDiagnostics`가 있어 보고서와 점수가 같은 필드를 사용한다. [test_diagnostics.py](../../tests/test_diagnostics.py)는 checkerboard의 대비, Gaussian blur 뒤 sharpness 하락, impulse noise 뒤 noise 증가라는 방향성만 검증한다.

| API/지표 | 파라미터 또는 수식 | 관찰하는 것 | 해석 한계 |
| --- | --- | --- | --- |
| `cv2.Laplacian(gray, CV_64F).var()` | 2차 미분 분산 | sharpness | 질감과 noise도 높일 수 있다. |
| `cv2.medianBlur(gray, 3)` residual | 3×3 median | noise | fine texture를 noise로 셀 수 있다. |
| `cv2.GaussianBlur` | 최대 51, 홀수 kernel | 조명 배경 | 큰 결함도 배경으로 섞일 수 있다. |
| `cv2.Canny(gray, low, high)` | median의 0.66, 1.33배 | edge density | 장면 의미나 정확한 결함 경계는 모른다. |
| `connectedComponentsWithStats` | 8-connectivity, area ≥ 8 | edge continuity | 끊긴 것이 항상 나쁜 것은 아니다. |

`compare_diagnostics(before, after)`는 절대 변화와 기준 이미지 대비 퍼센트를 만든다. 기준이 0에 가까울 때 퍼센트를 `None`으로 두는 이유는 0으로 나누어 극적인 숫자를 만들지 않기 위해서다. [reports.py](../../src/opencv_preprocessing_advisor/reports.py)는 이 비교를 CSV와 보고서에 기록한다.

## 언제 사용하고 피하는가

진단은 필터 전후를 같은 입력 크기와 색 계약에서 비교할 때 특히 유용하다. 평균 밝기가 낮고 어두운 clipping은 작지만, sharpness가 충분하면 gamma를 후보로 삼을 수 있다. noise estimate가 높되 에지 연속성도 이미 낮다면 강한 blur보다 원인 확인이 먼저다. 지역 대비만 낮은 영역은 CLAHE 실험 대상일 수 있으나, tile artifact와 색 보존을 함께 본다.

피해야 할 사용은 단일 지표의 순위화다. entropy가 높은 후보를 “최고”, sharpness가 낮은 후보를 “실패”라고 부르면 noise와 원래 장면 복잡도를 놓친다. 추천 점수는 여러 heuristic의 가중 조합으로 실험 우선순위를 정할 뿐, ground truth나 class label을 보지 않는다. **시각적 개선은 분류 성능 개선을 보장하지 않는다**. 라벨 기반 classifier 평가는 별도 split·fold에서 Accuracy와 Macro F1으로 측정해야 한다.

## 프로젝트 코드 연결

- [진단 계산: diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py)는 entropy, sharpness, noise와 에지 지표의 정확한 정의를 담고 있다.
- [지표 모델: models.py](../../src/opencv_preprocessing_advisor/models.py)는 모든 관찰값의 필드 이름을 고정한다.
- [진단 테스트: test_diagnostics.py](../../tests/test_diagnostics.py)는 값의 방향성만 요구하여 장비·버전 차이에 강하다.
- [점수 해석: scoring.py](../../src/opencv_preprocessing_advisor/scoring.py)는 진단을 추천 우선순위에 쓰되 경고도 추가한다.
- Notion용 절대 링크: [diagnostics.py (main)](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/diagnostics.py).

`_edges`는 Canny threshold를 영상 median의 0.66·1.33배로 정한다. 이는 모든 조명에서 최적인 마법 수가 아니라, 고정 threshold보다 샘플의 전체 톤에 맞춰 시작하기 위한 heuristic이다. 이 수치를 바꾸면 진단 수치도 함께 달라지므로 전후 비교에서는 같은 설정을 유지한다.

## 직접 실험

다음 코드는 합성 타일 원본, 의도적으로 blur한 이미지, salt-and-pepper noise를 섞은 이미지를 비교한다. random seed를 고정해 다시 실행해도 같은 노이즈 위치를 얻는다.

```python
from pathlib import Path

import cv2
import numpy as np

from opencv_preprocessing_advisor.diagnostics import analyze_image, compare_diagnostics
from opencv_preprocessing_advisor.io import decode_image

image = decode_image(Path("data/samples/synthetic-tile.png"))
blurred = cv2.GaussianBlur(image, (15, 15), 0)
rng = np.random.default_rng(42)
noisy = image.copy()
mask = rng.random(image.shape[:2]) < 0.02
noisy[mask] = np.where(rng.random((mask.sum(), 1)) < 0.5, 0, 255)

for name, candidate in {"original": image, "blurred": blurred, "noisy": noisy}.items():
    d = analyze_image(candidate)
    print(name, "sharpness=", round(d.sharpness, 2), "noise=", round(d.noise_estimate, 2),
          "entropy=", round(d.entropy, 3), "edge_density=", round(d.edge_density, 4))

delta = compare_diagnostics(analyze_image(image), analyze_image(blurred))
print("blurred sharpness delta:", round(delta["sharpness"].absolute_delta, 2))
```

원본과 blurred에서 sharpness·edge continuity가 함께 어떻게 변하는지 적고, noisy에서 entropy도 상승했는지 확인한다. 상승했다면 “더 많은 유용 정보”가 아니라 noise가 밝기 분포를 넓혔을 수 있다는 반대 가설을 적는다. 마지막으로 `cv2.medianBlur(noisy, 5)` 뒤의 noise와 sharpness를 비교해 제거와 질감 손실의 교환 관계를 기록한다.

## 예상 결과와 해석

| 관찰 | 예상 결과 | 해석과 다음 질문 |
| --- | --- | --- |
| blurred | sharpness와 edge continuity가 원본보다 낮아지는 경향 | 블러 가설을 지지하지만 원래 평면 장면인지 확인한다. |
| noisy | noise estimate가 증가, entropy도 변할 수 있음 | 정보량 증가가 아니라 임펄스가 히스토그램을 넓혔을 수 있다. |
| median 후 | noise는 감소할 수 있고 미세 에지도 약해질 수 있음 | 필터 크기를 줄이거나 다른 noise model을 비교한다. |
| clipping | 평균이 중간이어도 끝 비율이 높을 수 있음 | 노출/배경 의미를 눈으로 확인한 뒤 보정 여부를 정한다. |
| 여러 지표 | 서로 반대 방향의 변화가 가능 | 단일 score 대신 목적별 가설과 부작용을 적는다. |

실행 숫자는 OpenCV 버전과 이미지 크기에 따라 달라질 수 있다. 여기서 기대하는 것은 정확한 소수점이 아니라 변화 방향과 그것을 설명할 수 있는 원인이다. 특히 sharpness가 높아진 후보가 noise를 증폭했는지, edge density가 낮아진 후보가 결함 경계를 지웠는지 함께 확인한다.

## 자주 하는 실수와 디버깅

1. **entropy 하나로 품질을 판단**: salt-and-pepper noise는 entropy를 올릴 수 있다. residual과 시각 관찰을 같이 적는다.
2. **Laplacian variance를 초점의 절대 판정으로 사용**: 텍스처가 많은 타일은 흐리지 않아도 값이 다르다. 같은 대상의 전후 비교에 우선 쓴다.
3. **median residual을 센서 noise의 정확한 물리 측정으로 오해**: 3×3 이웃과 다른 세부 구조도 residual이다. noise model 가정이라고 명시한다.
4. **전후 이미지 크기·보간을 바꾼 뒤 수치 비교**: resize 자체가 edge와 sharpness를 바꾼다. 동일 해상도에서 측정한다.
5. **추천 점수를 모델 확률처럼 말함**: scoring은 heuristic ranking이며 label을 사용하지 않는다. 데이터셋 평가는 Day 9의 책임이다.

이상한 값이 나오면 작은 constant image, checkerboard, 사각형 blur, 임펄스 noise처럼 원인을 아는 합성 입력으로 되돌아간다. [test_diagnostics.py](../../tests/test_diagnostics.py)의 방향성 예제가 그 최소 재현법이다.

## 본인 말로 설명하기

### 1분 설명

“이 프로젝트는 한 개의 품질 점수 대신 밝기, clipping, 전역·지역 대비, entropy, sharpness, noise, 조명, 에지, 색 지표를 함께 봅니다. 평균 밝기는 중심만, clipping은 끝값 손실 가능성만 말합니다. Laplacian variance는 sharpness의 단서이고 median residual은 noise의 근사지만 둘 다 질감에도 반응합니다. 그래서 entropy가 올랐다고 좋아졌다고 말하지 않습니다. 이 수치들은 다음 전처리 실험의 우선순위를 정하는 heuristic이고, classifier 성능은 라벨 데이터로 별도 평가해야 합니다.”

### 깊이 설명

“진단의 핵심은 측정값과 원인 가설을 분리하는 것입니다. `diagnostics.py`는 BGR을 gray와 HSV로 바꾼 뒤, 평균/clip으로 노출, 표준편차와 타일 편차로 대비, 히스토그램 entropy로 분포 사용량, Laplacian 분산으로 고주파 구조, median residual로 이상 고주파, Gaussian 배경으로 조명 편차를 봅니다. Canny와 connected components는 구조가 얼마나 있고 이어지는지 보여 줍니다. 그러나 같은 수치가 결함·텍스처·noise 중 어느 의미인지 레이블 없이 알 수 없습니다. 따라서 전후 수치를 보고 강한 보정을 확정하지 않고, 원본·결과·경고를 함께 남기며, 실제 분류 영향은 누수 없는 데이터셋 실험으로 검증합니다.”

## 완료 기준

- [ ] **이해**: 모든 프로젝트 diagnostic의 의미와 ‘단일 metric은 품질이 아니다’라는 이유를 설명했다.
- [ ] **구현**: 원본·blurred·noisy 합성 샘플에 `analyze_image`와 `compare_diagnostics`를 실행했다.
- [ ] **해석**: entropy, sharpness, noise가 서로 충돌할 수 있는 실제 관찰 한 가지를 기록했다.
- [ ] **설명**: 1분 동안 heuristic 추천과 classifier 평가의 경계를 말하고 질문을 받았다.
