# OpenCV 기술 Q&A

10일 과정의 순서와 무관하게 찾아볼 수 있는 기술 참고서다. 실험은 [합성 타일](../../data/samples/synthetic-tile.png) 또는 본인이 사용 권한을 가진 이미지를 사용한다. 프로젝트의 수치는 [검증 근거](../portfolio/benchmark-evidence.json) 범위를 넘겨 해석하지 않는다.

## TQ1: OpenCV에서 기본 색 채널 순서는 무엇인가?
### 한 문장 답
`cv2.imread`와 이 프로젝트의 BGR 배열은 기본적으로 BGR 순서이며, RGB로 가정하면 색이 뒤바뀐다.
### 상세 설명
NumPy 배열의 마지막 축은 채널이지만 그 의미는 라이브러리마다 다르다. 화면 라이브러리에 넘기기 전에는 BGR→RGB 변환 여부를 명시하고, 회색화도 `COLOR_BGR2GRAY`를 사용한다.
### 프로젝트 근거
[io.py](../../src/opencv_preprocessing_advisor/io.py)의 `decode_image`와 [Day 1](day-01.md)은 BGR 계약을 먼저 검증한다.
### 주의/실패
RGB 이미지를 다시 BGR로 변환하면 색 보정 실험 자체가 틀어진다.

## TQ2: uint8 이미지에서 덧셈이 위험한 이유는?
### 한 문장 답
`uint8`은 0~255 범위라 NumPy 연산에서 wrap-around가 생길 수 있으므로 포화 연산이나 넓은 dtype을 의도적으로 선택한다.
### 상세 설명
밝기 이동은 값 범위를 벗어나면 clipping되어야 해석 가능하다. OpenCV의 `addWeighted`나 `normalize`는 범위 제어를 드러내며, 무심코 `image + 30`을 쓰는 것보다 안전하다.
### 프로젝트 근거
[transforms.py](../../src/opencv_preprocessing_advisor/transforms.py)의 `normalize_uint8`은 출력 dtype을 `uint8`로 명시한다.
### 주의/실패
오버플로를 대비 개선으로 오해하면 진단과 추천 점수가 모두 왜곡된다.

## TQ3: HSV와 LAB는 언제 구분해 쓰는가?
### 한 문장 답
HSV는 색상·채도 분리 관찰에, LAB는 밝기 L 채널을 색상과 덜 섞어 보정할 때 유용하다.
### 상세 설명
어떤 색 공간도 보편적으로 더 좋지 않다. 목표가 색상 임계값인지, 조명 변화 아래의 밝기 보정인지에 따라 변환 뒤의 분포와 결과를 비교한다.
### 프로젝트 근거
[features.py](../../src/opencv_preprocessing_advisor/features.py)는 HSV 히스토그램과 LAB L 히스토그램을 쓰고, `apply_lab_clahe`는 L만 바꾼다.
### 주의/실패
LAB가 색 보존을 보장하거나 HSV가 조명 불변이라고 주장하면 안 된다.

## TQ4: 이미지 shape와 dtype을 먼저 확인해야 하는 이유는?
### 한 문장 답
shape와 dtype은 채널 수, 해상도, 값 범위를 결정하므로 모든 OpenCV 전처리의 입력 계약이다.
### 상세 설명
같은 숫자 배열도 `(H,W)`와 `(H,W,3)`은 다른 의미이고, float 이미지는 0~1인지 0~255인지 확인해야 한다. 실패를 초기에 막으면 나중의 시각 오류를 줄인다.
### 프로젝트 근거
[io.py](../../src/opencv_preprocessing_advisor/io.py)의 `validate_bgr_image`가 `uint8`, 3채널, 비어 있지 않음을 검사한다.
### 주의/실패
알파 채널이나 회색 영상을 BGR 함수에 바로 넣어 생긴 오류를 필터 문제로 오진하지 않는다.

## TQ5: 보간법은 전처리 성능에 영향을 주는가?
### 한 문장 답
그렇다. resize는 픽셀 구조를 바꾸므로 보간법이 에지·질감·특징값에 영향을 준다.
### 상세 설명
축소에는 aliasing, 확대에는 계단 현상을 고려한다. 분류 비교에서는 파이프라인마다 같은 크기와 같은 resize 규칙을 적용해야 공정하다.
### 프로젝트 근거
[features.py](../../src/opencv_preprocessing_advisor/features.py)의 HOG와 질감 추출기는 입력을 고정 크기로 resize한다.
### 주의/실패
해상도 차이를 남긴 채 HOG 길이나 처리 시간을 비교하지 않는다.

## TQ6: Unicode 경로 이미지는 어떻게 읽는가?
### 한 문장 답
경로 문자열 인코딩 문제를 피하려면 바이트로 읽어 `cv2.imdecode`하는 방식이 이식성이 좋다.
### 상세 설명
일부 환경의 `cv2.imread`는 비ASCII 파일명에서 실패할 수 있다. 바이트 디코딩은 경로 처리와 이미지 디코딩을 분리해 실패 원인을 명확히 한다.
### 프로젝트 근거
[io.py](../../src/opencv_preprocessing_advisor/io.py)의 `decode_image`와 `decode_image_bytes`가 이 경로를 구현한다.
### 주의/실패
읽기 실패를 빈 배열로 넘기지 말고 예외와 입력 경로를 확인한다.

## TQ7: 엔트로피가 높으면 항상 좋은 이미지인가?
### 한 문장 답
아니다. 엔트로피는 분포의 다양성 지표일 뿐, 노이즈 증가도 높게 만들 수 있다.
### 상세 설명
엔트로피는 대비·질감·잡음을 구분하지 않는다. 밝기, clipping, 노이즈, sharpness와 함께 전후 변화를 본다.
### 프로젝트 근거
[diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py)의 `_entropy`는 여러 `ImageDiagnostics` 항목 중 하나다.
### 주의/실패
엔트로피 상승 하나만으로 추천 파이프라인이나 분류 성능을 단정하지 않는다.

## TQ8: 지역 대비와 전역 대비는 어떻게 다른가?
### 한 문장 답
전역 대비는 전체 표준편차이고, 지역 대비는 작은 타일 안의 변화를 평균해 국소 구조를 본다.
### 상세 설명
조명 그라데이션이 큰 이미지는 전역 대비가 있어도 결함 주변이 평평할 수 있다. 타일 크기는 관찰 스케일을 바꾸므로 고정해 비교한다.
### 프로젝트 근거
`_local_contrast`는 16픽셀 격자의 표준편차 평균을 계산한다([diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py)).
### 주의/실패
지역 대비 증가가 과도한 노이즈 증폭인지 에지 지표와 함께 확인한다.

## TQ9: 노이즈 추정치는 무엇을 측정하는가?
### 한 문장 답
이 프로젝트의 추정치는 3×3 median 결과와 원본의 RMS 잔차이며, 절대적인 센서 노이즈 측정값은 아니다.
### 상세 설명
median이 제거한 고주파 차이를 잡음 후보로 본다. 미세한 실제 질감도 잔차에 포함될 수 있으므로 재료별 비교가 필요하다.
### 프로젝트 근거
`_noise_estimate` 구현은 [diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py)에 있다.
### 주의/실패
낮은 추정치만 추구하면 유용한 결함 텍스처까지 지울 수 있다.

## TQ10: Laplacian 분산 sharpness는 무엇을 놓치는가?
### 한 문장 답
Laplacian 분산은 고주파 반응을 보는 간단한 선명도 대리 지표라 노이즈와 실제 경계를 구별하지 못한다.
### 상세 설명
블러에는 낮아지기 쉽지만, 노이즈나 반복 무늬에는 높아질 수 있다. 관찰 이미지와 노이즈·에지 지표를 함께 사용한다.
### 프로젝트 근거
`analyze_image`는 `cv2.Laplacian(...).var()`를 기록한다([diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py)).
### 주의/실패
sharpness가 커졌다는 사실을 품질 또는 분류 향상으로 번역하지 않는다.

## TQ11: clipping 비율은 왜 진단하는가?
### 한 문장 답
0 또는 255에 몰린 픽셀 비율은 암부·명부 세부가 이미 사라졌거나 보정으로 사라질 위험을 알려 준다.
### 상세 설명
이 프로젝트는 5 이하와 250 이상을 각각 센다. 목표 중간 밝기를 맞춰도 clipping이 늘면 회복 불가능한 정보 손실일 수 있다.
### 프로젝트 근거
`dark_clip_ratio`, `bright_clip_ratio`는 [diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py)와 [scoring.py](../../src/opencv_preprocessing_advisor/scoring.py)에 연결된다.
### 주의/실패
클리핑 임계값은 센서 포화의 물리적 정의가 아니라 진단용 운영 기준이다.

## TQ12: 조명 불균일도는 어떤 방식으로 계산하는가?
### 한 문장 답
큰 Gaussian blur로 배경을 근사한 뒤 그 표준편차를 평균으로 나눈 값이다.
### 상세 설명
구조보다 느리게 바뀌는 조명 성분을 보려는 근사다. 짧은 변의 길이에 맞춰 홀수 kernel을 조절해 아주 작은 영상도 다룬다.
### 프로젝트 근거
`_illumination_nonuniformity`는 [diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py)에 구현돼 있다.
### 주의/실패
넓은 실제 패턴을 조명 문제로 오인할 수 있으므로 원본과 배경 이미지를 같이 본다.

## TQ13: edge density와 edge continuity를 함께 보는 이유는?
### 한 문장 답
에지 수만 많으면 노이즈일 수 있어, 충분히 큰 연결 성분이 차지하는 비율도 함께 본다.
### 상세 설명
Canny 에지의 픽셀 비율과 8-연결 성분 중 면적 8 이상인 픽셀 비율을 분리한다. 이는 연속 구조 보존의 대리 신호다.
### 프로젝트 근거
`_edges`는 두 지표를 함께 반환한다([diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py)).
### 주의/실패
연속성 기준 8은 보편 임계값이 아니며 대상 크기에 맞춰 재검토한다.

## TQ14: 색 풍부도(colorfulness)는 색 정확도인가?
### 한 문장 답
아니다. 채널 차이의 분포에서 계산한 상대적 풍부도이며 색 보정의 정확도를 보장하지 않는다.
### 상세 설명
R-G와 Y-B 차이의 평균·분산을 조합해 변화량을 본다. 색표나 기준 조명 없이 실제 색 재현성을 판정할 수는 없다.
### 프로젝트 근거
`_colorfulness` 계산과 색 보존 점수는 각각 [diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py), [scoring.py](../../src/opencv_preprocessing_advisor/scoring.py)에 있다.
### 주의/실패
색 풍부도가 줄었다고 무조건 나쁜 것도, 늘었다고 색이 정확한 것도 아니다.

## TQ15: `normalize`는 언제 유용한가?
### 한 문장 답
입력마다 점유 범위가 좁을 때 0~255로 펼쳐 관찰을 돕지만, 영상 간 절대 밝기 관계는 바꾼다.
### 상세 설명
min-max 정규화는 단일 이미지의 대비를 늘릴 수 있다. 데이터셋 분류에서는 촬영 밝기 자체가 클래스 단서일 수 있어 영향도 측정해야 한다.
### 프로젝트 근거
`normalize_uint8`은 [transforms.py](../../src/opencv_preprocessing_advisor/transforms.py)의 하나의 후보 변환이다.
### 주의/실패
정규화 결과가 보기 좋다는 이유로 모든 생산 데이터에 고정하지 않는다.

## TQ16: gamma 보정의 핵심 파라미터는 무엇인가?
### 한 문장 답
gamma는 중간톤 곡선의 모양을 바꾸며, 이 프로젝트의 자동 gamma는 평균 밝기를 목표 중간값에 맞추려 한다.
### 상세 설명
LUT를 써서 각 픽셀에 같은 단조 변환을 적용한다. `target_midpoint`가 유효 범위인지와 입력이 거의 흑·백인지도 확인한다.
### 프로젝트 근거
`apply_auto_gamma`는 gamma를 0.5~2.0으로 제한한다([transforms.py](../../src/opencv_preprocessing_advisor/transforms.py)).
### 주의/실패
평균 밝기만 맞추면 국소 조명, clipping, 색 문제까지 해결된다고 보면 안 된다.

## TQ17: CLAHE의 `clipLimit`은 무엇을 제어하는가?
### 한 문장 답
`clipLimit`은 타일 히스토그램의 과도한 누적을 제한해 국소 대비 증폭을 완화하는 파라미터다.
### 상세 설명
값이 커질수록 국소 대비와 노이즈가 함께 두드러질 수 있다. 값 하나를 정답으로 정하지 말고 진단 변화와 다운스트림 결과를 비교한다.
### 프로젝트 근거
`apply_lab_clahe`의 기본 `clip_limit=2.0`은 [transforms.py](../../src/opencv_preprocessing_advisor/transforms.py)에 명시돼 있다.
### 주의/실패
OpenCV 구현의 clipLimit을 단순한 픽셀 개수로 설명하지 않는다.

## TQ18: CLAHE의 `tileGridSize`는 무엇을 뜻하는가?
### 한 문장 답
`tileGridSize`는 타일의 픽셀 크기가 아니라 영상을 나누는 타일 개수이며, 수가 커지면 각 타일은 더 작아진다.
### 상세 설명
더 작은 영역의 히스토그램을 쓰므로 조명 변화에 민감해질 수 있다. 작은 텍스처에 과도하게 반응하는지 결과를 봐야 한다.
### 프로젝트 근거
[Day 3](day-03.md)와 `cv2.createCLAHE(tileGridSize=(grid_size, grid_size))` 구현이 이 의미를 사용한다.
### 주의/실패
grid 값이 크면 더 넓은 문맥을 본다고 반대로 설명하지 않는다.

## TQ19: 왜 LAB의 L 채널에만 CLAHE를 적용하는가?
### 한 문장 답
밝기 채널만 조정해 a·b 색상 성분을 직접 변경하지 않으려는 설계다.
### 상세 설명
이는 BGR 각 채널을 독립 보정하는 것보다 색 관계 교란을 줄일 수 있다는 선택이지, 모든 조명·색상 문제의 해법은 아니다.
### 프로젝트 근거
`apply_lab_clahe`는 LAB를 split하고 L만 `clahe.apply`한다([transforms.py](../../src/opencv_preprocessing_advisor/transforms.py)).
### 주의/실패
변환 후 BGR로 돌아올 때의 양자화·색 이동 가능성도 관찰한다.

## TQ20: Gaussian filter는 어떤 가정에 맞는가?
### 한 문장 답
Gaussian blur는 공간적으로 부드러운 변동을 평균화하는 선형 필터라 일반적 고주파 잡음 완화에 쓴다.
### 상세 설명
kernel과 sigma가 커지면 노이즈와 에지가 함께 약해진다. 선형 평균이 적합한지, 미세 경계가 중요한지를 먼저 판단한다.
### 프로젝트 근거
`apply_gaussian`은 홀수 kernel과 음수가 아닌 sigma를 검증한다([transforms.py](../../src/opencv_preprocessing_advisor/transforms.py)).
### 주의/실패
blur 뒤 Canny가 잘 보인다는 인상만으로 최종 분류 성능을 결론내리지 않는다.

## TQ21: Median filter는 언제 Gaussian보다 낫나?
### 한 문장 답
점처럼 튀는 impulse noise에는 중앙값이 이상치에 덜 민감해 median이 유리할 수 있다.
### 상세 설명
median은 비선형이라 부드러운 Gaussian 노이즈에 항상 우세하지 않다. 질감과 가는 선을 얼마나 바꾸는지 비교한다.
### 프로젝트 근거
[tests/test_transforms.py](../../tests/test_transforms.py)는 impulse 픽셀 감소를 `apply_median`으로 검증한다.
### 주의/실패
kernel이 커지면 salt-and-pepper만 지우는 것이 아니라 작은 결함도 제거한다.

## TQ22: Bilateral filter의 장단점은?
### 한 문장 답
bilateral은 공간 거리와 색 차이를 함께 고려해 에지를 상대적으로 보존하며 평활화하지만 비용과 파라미터 민감도가 크다.
### 상세 설명
`diameter`, `sigmaColor`, `sigmaSpace`가 영향 범위와 유사성 허용치를 정한다. 반복 텍스처에는 실제 미세 구조도 뭉개질 수 있다.
### 프로젝트 근거
`apply_bilateral` 기본값은 [transforms.py](../../src/opencv_preprocessing_advisor/transforms.py)에 있고, benchmark 후보에도 있다.
### 주의/실패
에지가 남아 보인다고 지연시간과 특징 분포 변화가 무시되지는 않는다.

## TQ23: oversmoothing 경고는 어떻게 발생하는가?
### 한 문장 답
전처리 뒤 sharpness가 원본의 절반 미만이면 이 프로젝트는 oversmoothing 경고를 남긴다.
### 상세 설명
이는 안전장치용 휴리스틱이다. 선명도 대리 지표가 줄었다는 신호를 사용자에게 보여 주되, 필터 품질의 절대 판정은 하지 않는다.
### 프로젝트 근거
[scoring.py](../../src/opencv_preprocessing_advisor/scoring.py)의 `score_pipeline`이 경고 코드를 만든다.
### 주의/실패
절반이라는 임계값을 모든 카메라·재질에 보편적인 품질 기준으로 쓰지 않는다.

## TQ24: unsharp masking은 왜 노이즈를 키울 수 있나?
### 한 문장 답
원본에서 blur를 뺀 고주파 성분을 더하므로 실제 경계와 노이즈를 함께 강화할 수 있다.
### 상세 설명
amount와 threshold는 강화 정도와 작은 변화 무시 범위를 조절한다. 먼저 노이즈를 진단하고, 전후 에지 밀도·clipping을 점검한다.
### 프로젝트 근거
`apply_unsharp`은 threshold 아래의 저대비 위치를 원본으로 되돌린다([transforms.py](../../src/opencv_preprocessing_advisor/transforms.py)).
### 주의/실패
샤프닝을 블러 복원의 증명으로, 또는 경계 정확도의 보증으로 말하지 않는다.

## TQ25: Canny는 내부적으로 Gaussian blur를 하는가?
### 한 문장 답
이 과정에서 사용하는 `cv2.Canny` 호출은 Gaussian blur를 자동으로 포함하지 않으므로 필요하면 호출자가 앞에 명시한다.
### 상세 설명
Canny의 gradient, non-maximum suppression, 이중 임계값, hysteresis를 이해하고 평활화 단계는 별도 선택으로 둔다.
### 프로젝트 근거
[Day 5](day-05.md)와 [tests/test_learning_10day_content.py](../../tests/test_learning_10day_content.py)가 이 경계를 명시한다.
### 주의/실패
전처리 blur와 Canny API 자체의 연산을 섞어 설명하지 않는다.

## TQ26: Sobel과 Scharr는 어떻게 고르는가?
### 한 문장 답
Sobel은 일반적인 미분 근사, Scharr는 작은 kernel에서 회전 대칭성이 더 나은 대안으로 비교할 수 있다.
### 상세 설명
둘 다 방향별 변화량을 주므로 magnitude와 방향, kernel 크기, 스케일을 작업 목적에 맞춘다. 결과 이미지를 정량·정성으로 비교한다.
### 프로젝트 근거
[Day 5](day-05.md)와 [ui/technique_explorer.py](../../ui/technique_explorer.py)가 두 연산을 실험 경로로 연결한다.
### 주의/실패
Scharr가 모든 에지 검출 상황에서 더 정확하다고 일반화하지 않는다.

## TQ27: morphology kernel의 모양이 왜 중요한가?
### 한 문장 답
사각·타원·십자 kernel은 어떤 방향과 연결을 보존·제거하는지가 달라 결과의 구조적 의미를 바꾼다.
### 상세 설명
opening, closing, blackhat은 객체 크기와 방향에 의존한다. 결함의 최소 크기보다 큰 kernel을 무심코 쓰면 결함을 지울 수 있다.
### 프로젝트 근거
`apply_blackhat`은 세 kernel 모양을 지원한다([transforms.py](../../src/opencv_preprocessing_advisor/transforms.py)).
### 주의/실패
morphology를 단순한 노이즈 제거로만 보고 구조 손실을 누락하지 않는다.

## TQ28: contour와 connected components는 무엇이 다른가?
### 한 문장 답
contour는 경계 좌표를, connected components는 이진 영역의 연결된 라벨과 면적·통계를 다루기 좋다.
### 상세 설명
외곽선 모양·둘레가 중요하면 contour, 개수·면적·bounding box가 중요하면 components를 우선 고려한다. 이진화 품질이 두 방법 모두의 전제다.
### 프로젝트 근거
에지 연속성은 `connectedComponentsWithStats`를 사용한다([diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py)).
### 주의/실패
Canny 결과처럼 끊긴 경계를 하나의 물체 영역처럼 해석하지 않는다.

## TQ29: threshold는 왜 조명 변화에 취약한가?
### 한 문장 답
전역 threshold는 하나의 값으로 전체 픽셀을 나누므로 배경 밝기가 달라지면 같은 물체도 다르게 분할된다.
### 상세 설명
적응 임계값, 배경 보정, LAB L 기반 처리 등을 후보로 두되, false positive와 false negative를 표본별로 점검한다.
### 프로젝트 근거
[Day 5](day-05.md)는 threshold·morphology 실험을 [technique explorer](../../ui/technique_explorer.py)와 연결한다.
### 주의/실패
눈에 잘 맞는 한 장의 임계값을 데이터셋 전체에 그대로 적용하지 않는다.

## TQ30: HSV/LAB 히스토그램 특징은 무엇을 잃는가?
### 한 문장 답
히스토그램은 값의 분포를 요약하므로 픽셀의 정확한 공간 배치를 잃는다.
### 상세 설명
색 분포가 다른 클래스에는 도움이 될 수 있지만 동일한 색을 다른 모양으로 배치한 이미지는 구분하기 어렵다. 그래서 HOG·질감 통계와 결합한다.
### 프로젝트 근거
`ColorHistogramExtractor`와 `CombinedExtractor`는 [features.py](../../src/opencv_preprocessing_advisor/features.py)에 있다.
### 주의/실패
정규화된 히스토그램 길이가 고정이라는 사실을 조명·크기 불변성 보장으로 해석하지 않는다.

## TQ31: 이 저장소의 HOG 입력 크기는 왜 16의 배수여야 하나?
### 한 문장 답
이 저장소의 wrapper contract는 configured `size`의 각 차원을 16의 배수로 제한하지만, 이는 HOG의 보편 규칙이 아니다.
### 상세 설명
현재 `cv2.HOGDescriptor`는 `window=size`, `block=(16,16)`, `stride=(8,8)`, `cell=(8,8)`, `9 bins`로 구성된다. 이 wrapper는 그 구성에 맞춰 `size`를 16의 배수로 더 엄격하게 제한하고 입력을 그 window 크기로 resize한다. 다른 HOG window·block·stride 구성에는 다른 호환 조건이 적용될 수 있으므로 “16의 배수”를 보편 규칙으로 설명하면 안 된다.
### 프로젝트 근거
`HOGExtractor.__init__`의 검증과 `cv2.HOGDescriptor` 설정은 [features.py](../../src/opencv_preprocessing_advisor/features.py)에 있다.
### 주의/실패
원본이 130×128이라는 사실만으로 HOG가 실패한다고 말하면 구현을 잘못 이해한 것이다. 이 코드에서 검사하는 대상은 원본 크기가 아니라 configured `size`다.

## TQ32: Gabor 질감 통계는 무엇을 요약하는가?
### 한 문장 답
여러 방향 Gabor 응답의 절대값 평균·표준편차·상위 분위수로 방향성 질감 반응을 압축한다.
### 상세 설명
이 프로젝트는 0°, 45°, 90°, 135°와 Sobel·Laplacian 응답을 함께 쓴다. 이는 텍스처의 완전한 표현이 아니라 고정 길이 고전 특징이다.
### 프로젝트 근거
`TextureStatsExtractor`는 [features.py](../../src/opencv_preprocessing_advisor/features.py)에 구현돼 있다.
### 주의/실패
Gabor 파라미터를 바꾸면 특징 분포도 달라지므로 기존 모델과 섞지 않는다.

## TQ33: SIFT Bag-of-Words는 왜 `fit`이 먼저 필요한가?
### 한 문장 답
BoW 히스토그램은 SIFT descriptor를 어떤 visual word에 배정할지 정하는 vocabulary가 있어야 계산할 수 있다.
### 상세 설명
훈련 이미지 descriptor로 k-means vocabulary를 만들고 각 이미지의 할당 빈도를 정규화한다. vocabulary 크기보다 descriptor가 적으면 학습할 수 없다.
### 프로젝트 근거
`SiftBowExtractor.fit`과 명시적 오류는 [features.py](../../src/opencv_preprocessing_advisor/features.py)에 있다.
### 주의/실패
전체 데이터에서 vocabulary를 만든 뒤 CV test fold를 평가하면 누수다.

## TQ34: 현재 benchmark에 SIFT가 포함되는가?
### 한 문장 답
아니다. SIFT BoW는 구현되어 있지만 현재 `BenchmarkService`의 공개 feature profile은 color, shape, texture, combined다.
### 상세 설명
SIFT를 공정하게 넣으려면 각 train fold 안에서 vocabulary를 fit하는 파이프라인 설계가 추가로 필요하다. 존재하는 코드와 현재 실험 범위를 구분한다.
### 프로젝트 근거
[features.py](../../src/opencv_preprocessing_advisor/features.py)의 `SiftBowExtractor`와 [services.py](../../src/opencv_preprocessing_advisor/services.py)의 `_feature_extractor`를 대조한다.
### 주의/실패
구현돼 있다는 이유만으로 보고된 0.789 Macro F1에 SIFT 기여가 있었다고 말하지 않는다.

## TQ35: OpenCV 분류기 입력이 `float32`여야 하는 이유는?
### 한 문장 답
이 프로젝트의 feature matrix 계약은 OpenCV `cv2.ml` 호출에 맞춘 유한한 2차원 `float32`다.
### 상세 설명
dtype과 shape를 초기에 고정하면 모델별 암묵 변환과 예측 불일치를 줄인다. 라벨은 별도로 정수 벡터로 검증한다.
### 프로젝트 근거
[classifiers.py](../../src/opencv_preprocessing_advisor/classifiers.py)의 `_feature_matrix`, `_label_vector`가 계약을 강제한다.
### 주의/실패
`float64`가 항상 실패한다고 단정하기보다 이 저장소의 인터페이스 계약을 지킨다.

## TQ36: SVM, kNN, RTrees를 모두 비교하는 이유는?
### 한 문장 답
같은 특징·fold 조건에서 서로 다른 결정 방식의 baseline을 비교해 한 모델의 우연한 우세를 줄이기 위해서다.
### 상세 설명
SVM은 RBF kernel, kNN은 근접 이웃, RTrees는 랜덤화된 트리 앙상블을 사용한다. 데이터 크기와 특징 스케일에 따른 차이를 관찰한다.
### 프로젝트 근거
[classifiers.py](../../src/opencv_preprocessing_advisor/classifiers.py)와 [tests/test_classifiers.py](../../tests/test_classifiers.py)가 어댑터를 검증한다.
### 주의/실패
세 모델을 비교했어도 모든 산업 이미지에 대한 모델 선택 결론은 아니다.

## TQ37: 표준화는 왜 fold마다 fit해야 하나?
### 한 문장 답
test fold의 평균·표준편차를 표준화기에 쓰면 test 분포 정보를 학습 단계에 미리 전달하는 누수가 된다.
### 상세 설명
각 fold에서 train feature로만 평균·scale을 fit하고 train·test 모두에 같은 변환을 적용한다. 이는 평가 절차의 일부다.
### 프로젝트 근거
`cross_validate`의 `Standardizer().fit(matrix[fold.train_indices])`는 [evaluation.py](../../src/opencv_preprocessing_advisor/evaluation.py)에 있다.
### 주의/실패
전처리 또는 feature vocabulary도 데이터 의존적으로 학습한다면 같은 fold-local 원칙이 필요하다.

## TQ38: stratified K-fold의 장점은 무엇인가?
### 한 문장 답
각 fold에 클래스 비율을 가능한 한 유지해 소수 클래스가 특정 test fold에서 사라지는 위험을 줄인다.
### 상세 설명
이 구현은 클래스별 인덱스를 seed로 섞어 분할하고, 최소 클래스 개수에 맞춰 실제 split 수를 줄인다.
### 프로젝트 근거
`stratified_folds`는 [datasets.py](../../src/opencv_preprocessing_advisor/datasets.py)에 있고 benchmark seed는 42다.
### 주의/실패
stratification은 중복 이미지, 촬영 배치, 시간 순서 같은 다른 누수를 자동으로 막지 못한다.

## TQ39: Macro F1을 accuracy와 함께 보는 이유는?
### 한 문장 답
Macro F1은 클래스별 F1을 동등 가중 평균하므로 다수 클래스가 accuracy를 가리는 상황을 보완한다.
### 상세 설명
precision과 recall이 모두 낮은 클래스를 드러내며, support가 적은 클래스도 한 표를 갖는다. 따라서 accuracy와 함께 confusion matrix를 읽는다.
### 프로젝트 근거
`classification_metrics`는 per-class 값과 macro 평균을 계산한다([evaluation.py](../../src/opencv_preprocessing_advisor/evaluation.py)).
### 주의/실패
Macro F1 하나만으로 오류 비용이나 운영 임계값을 결정하지 않는다.

## TQ40: confusion matrix의 행과 열은 무엇인가?
### 한 문장 답
이 구현에서는 행이 실제 class, 열이 예측 class이므로 행별로 어떤 실제 클래스가 어디로 혼동됐는지 읽는다.
### 상세 설명
대각선은 맞춘 수이고 비대각선은 혼동이다. 표시 도구마다 축 방향이 다를 수 있어 보고서에 관례를 적어야 한다.
### 프로젝트 근거
`matrix[actual, guess] += 1`은 [evaluation.py](../../src/opencv_preprocessing_advisor/evaluation.py)에 명시돼 있다.
### 주의/실패
행·열을 뒤집은 그림을 보고 특정 클래스의 recall 문제를 precision 문제로 설명하지 않는다.

## TQ41: 추천 suitability score는 분류 정확도인가?
### 한 문장 답
아니다. 단일 이미지의 진단 변화와 프로필 가중치로 전처리 실험 우선순위를 정하는 휴리스틱 점수다.
### 상세 설명
점수는 local contrast, noise, edge, clipping 같은 대리 지표를 조합한다. 레이블이나 예측 정답을 사용하지 않으므로 accuracy가 될 수 없다.
### 프로젝트 근거
`rank_recommendations`는 [scoring.py](../../src/opencv_preprocessing_advisor/scoring.py), 데이터셋 평가는 [services.py](../../src/opencv_preprocessing_advisor/services.py)에 분리돼 있다.
### 주의/실패
Top 3의 1위가 benchmark에서도 1위일 것이라고 홍보하지 않는다.

## TQ42: profile weights는 왜 YAML에서 읽는가?
### 한 문장 답
작업 프로필별 진단 우선순위를 코드 변경 없이 명시·검토 가능하게 하기 위해 YAML로 분리한다.
### 상세 설명
가중치 합이 1인지, 지원하지 않는 metric이 없는지 검증해야 점수 해석이 안정적이다. 가중치는 전문가 판단이며 학습된 모델 파라미터가 아니다.
### 프로젝트 근거
[config/scoring.yaml](../../src/opencv_preprocessing_advisor/config/scoring.yaml)과 `load_profile_weights`가 연결된다.
### 주의/실패
가중치 파일이 있다는 이유로 데이터로 보정된 객관적 품질 모델이라고 부르지 않는다.

## TQ43: YAML 파이프라인의 장점은 무엇인가?
### 한 문장 답
파이프라인 이름, 순서, 파라미터, 경고를 선언적으로 기록해 같은 후보를 재실행하고 비교할 수 있게 한다.
### 상세 설명
코드는 catalog가 YAML을 읽어 transform 이름과 파라미터를 검증한 뒤 순서대로 실행한다. 실험에서 순서는 재현성의 일부다.
### 프로젝트 근거
[config/pipelines.yaml](../../src/opencv_preprocessing_advisor/config/pipelines.yaml)과 [pipelines.py](../../src/opencv_preprocessing_advisor/pipelines.py)를 본다.
### 주의/실패
YAML 설정을 바꾼 결과를 이전 config hash의 결과와 같은 실험으로 취급하지 않는다.

## TQ44: 보고서에 config hash와 OpenCV version을 남기는 이유는?
### 한 문장 답
같은 명령처럼 보여도 설정·라이브러리 버전이 다르면 결과가 달라질 수 있어 실행 문맥을 추적하기 위해서다.
### 상세 설명
시드, fold assignment, pipeline hash, version, 생성 시간은 결과 표의 숫자만으로는 알 수 없는 재현 조건이다.
### 프로젝트 근거
`ReportWriter.write_benchmark`는 run config와 hash를 기록한다([reports.py](../../src/opencv_preprocessing_advisor/reports.py)).
### 주의/실패
hash가 있다고 원본 데이터나 운영 환경까지 완벽하게 복제되는 것은 아니다.

## TQ45: 이 프로젝트의 benchmark 결과는 무엇을 뜻하는가?
### 한 문장 답
117장, 6개 `tile/test` 상태 폴더 클래스를 seed 42의 stratified 5-fold로 분류한 특정 설정의 비교 결과다.
### 상세 설명
Original + RTrees의 평균 Accuracy 0.804, Macro F1 0.789은 전처리 후보와 고전 특징·분류기 조합에 대한 관찰이다.
### 프로젝트 근거
정확한 표와 provenance는 [benchmark-evidence.json](../portfolio/benchmark-evidence.json)에 있다.
### 주의/실패
이 수치는 일반 산업 현장 성능이 아니며, This is not an official MVTec anomaly-detection metric.

## TQ46: Original pipeline이 이긴 결과는 실패인가?
### 한 문장 답
아니다. 해당 데이터와 특징·평가 조건에서는 추가 전처리가 정보 보존보다 이득이 적었다는 유용한 엔지니어링 결론이다.
### 상세 설명
보기 좋은 대비 강화가 색·질감·에지를 바꿔 고전 특징의 구분 단서를 약화할 수 있다. 그래서 원본을 baseline에 반드시 둔다.
### 프로젝트 근거
[benchmark-evidence.json](../portfolio/benchmark-evidence.json)의 3개 pipeline 순위와 [Day 10](day-10.md)을 확인한다.
### 주의/실패
원본 우세를 전처리 일반 무용론으로 확대하지 않는다.

## TQ47: 단일 이미지 추천과 benchmark를 왜 분리했는가?
### 한 문장 답
레이블 없는 한 장에서는 관찰 가능한 진단으로 후보를 정하고, 성능 주장은 레이블·fold가 있는 데이터셋에서 검증해야 하기 때문이다.
### 상세 설명
두 흐름은 입력 정보와 목표가 다르다. 분리하면 화면의 편의 점수를 분류 성능처럼 오해하는 구조적 위험을 줄인다.
### 프로젝트 근거
`ImageAdvisorService.analyze`와 `BenchmarkService.run`은 [services.py](../../src/opencv_preprocessing_advisor/services.py)에 별도 서비스로 구현된다.
### 주의/실패
추천 화면의 전후 진단 변화로 cross-validation 결과를 대신하지 않는다.

## TQ48: 테스트가 문서 학습에 어떤 역할을 하는가?
### 한 문장 답
테스트는 구현 계약과 주장 경계를 실행 가능한 증거로 만들어 문서의 설명을 코드와 연결한다.
### 상세 설명
변환 dtype, 특징 길이, fold-local scaling, 허브 링크·문구는 각각 회귀 테스트가 잡는다. 독자는 코드를 읽기 전 기대 동작을 확인할 수 있다.
### 프로젝트 근거
[tests](../../tests)와 [학습 허브 계약](../../tests/test_learning_10day_content.py)이 출발점이다.
### 주의/실패
테스트 통과가 모든 실제 이미지와 배포 환경의 적합성을 증명하지는 않는다.

## TQ49: OpenCV 고전 특징 접근의 한계는 무엇인가?
### 한 문장 답
고정 특징은 해석과 재현에는 좋지만 복잡한 변형·조명·도메인 변화에 대한 표현력은 제한적일 수 있다.
### 상세 설명
색 히스토그램, HOG, 질감 통계는 명시적이며 빠른 baseline이지만, 새 도메인에는 데이터와 오류 비용에 맞춘 재평가가 필요하다.
### 프로젝트 근거
[limitations.md](../portfolio/limitations.md)와 `CombinedExtractor`([features.py](../../src/opencv_preprocessing_advisor/features.py))가 범위를 설명한다.
### 주의/실패
딥러닝을 쓰지 않았다는 사실을 곧바로 부족함 또는 우월함으로 단정하지 않는다.

## TQ50: 새 이미지 도메인으로 옮길 때 첫 실험은 무엇인가?
### 한 문장 답
원본 baseline과 소수의 가설 기반 전처리 후보를 같은 데이터 분할·지표·보고서 조건에서 비교한다.
### 상세 설명
먼저 입력 검사, 진단 분포, 클래스 정의, 오류 비용을 기록한다. 그 다음 fold-local 처리와 seed를 고정해 비교하고 실패 사례를 확인한다.
### 프로젝트 근거
[services.py](../../src/opencv_preprocessing_advisor/services.py)의 benchmark 흐름과 [reports.py](../../src/opencv_preprocessing_advisor/reports.py)의 산출물을 재사용할 수 있다.
### 주의/실패
이 저장소의 타일 결과를 새 카메라·새 결함·새 라벨 체계로 전이 가능한 성능 수치로 복사하지 않는다.
