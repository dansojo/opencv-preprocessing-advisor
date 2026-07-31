# 실습 과제와 해설

모든 과제는 [합성 타일](../../data/samples/synthetic-tile.png) 또는 사용 권한이 있는 개인 이미지만 사용한다. MVTec 원본, GT mask, 비공개 경로는 요구하지 않는다. “검증”은 눈으로 보는 확인과 수치·테스트 확인을 함께 뜻한다.

## 유도형: 입력과 최소 코드를 함께 따라가는 10개

## EX1: BGR 입력 계약 확인
### 난이도
유도형 · ★☆☆
### 문제
이미지를 읽고 shape, dtype, 채널별 평균을 출력한 뒤 BGR인지 확인한다.
### 입력
`../../data/samples/synthetic-tile.png`.
### 요구 산출물
실행 코드, 출력 기록, BGR→RGB 표시 전환 한 장.
### 힌트
`decode_image`와 `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`를 비교한다.
### 해설
`decode_image(path)` 뒤 `image.shape == (H, W, 3)`와 `image.dtype == np.uint8`을 assert한다. 시각화 전만 RGB로 바꾸고 처리 배열은 BGR로 유지한다.
### 평가 기준
채널 순서·dtype·shape를 모두 명시하고, RGB 변환을 전처리로 오해하지 않으면 통과.

## EX2: uint8 overflow 재현과 수정
### 난이도
유도형 · ★☆☆
### 문제
`image + 40`과 포화 덧셈의 차이를 작은 배열로 재현한다.
### 입력
값 240, 250, 255가 든 1×3 `uint8` 배열.
### 요구 산출물
두 결과 표와 안전한 밝기 이동 코드.
### 힌트
`cv2.add` 또는 `int16` 변환 후 `np.clip`을 쓴다.
### 해설
NumPy `uint8` 덧셈은 modulo가 될 수 있으므로 `cv2.add(image, 40)` 또는 `np.clip(image.astype(np.int16)+40,0,255).astype(np.uint8)`를 사용한다.
### 평가 기준
wrap-around 원인과 clipping 결과를 숫자로 보여 주면 통과.

## EX3: BGR, HSV, LAB 채널 관찰
### 난이도
유도형 · ★☆☆
### 문제
한 이미지를 BGR, HSV, LAB로 변환해 각 채널의 평균·표준편차를 표로 만든다.
### 입력
합성 타일 또는 권한 있는 컬러 이미지.
### 요구 산출물
세 색 공간의 채널 통계와 사용 목적 한 줄씩.
### 힌트
`cv2.cvtColor`와 `cv2.split`을 사용한다.
### 해설
HSV는 hue/saturation 관찰, LAB는 밝기 L 분리 관찰에 쓴다고 기록한다. 채널 숫자 차이는 색 공간의 값 범위가 다르므로 직접 비교하지 않는다.
### 평가 기준
색 공간을 “더 좋은 공간”이 아니라 목적별 표현으로 설명하면 통과.

## EX4: 프로젝트 진단표 만들기
### 난이도
유도형 · ★★☆
### 문제
원본 이미지의 모든 `ImageDiagnostics` 필드를 표로 출력한다.
### 입력
합성 타일 한 장.
### 요구 산출물
필드명·값·해석 가설 3열 표.
### 힌트
`analyze_image(image)`의 dataclass를 `dataclasses.asdict`로 바꾼다.
### 해설
밝기, clipping, contrast, entropy, sharpness, noise, illumination, edge, color 항목을 빠짐없이 나열한다. 값 하나로 품질 결론을 내리지 않고 확인할 다음 관찰을 쓴다.
### 평가 기준
13개 항목과 “단일 지표는 충분하지 않다”는 해석이 있으면 통과.

## EX5: LAB CLAHE 전후 비교
### 난이도
유도형 · ★★☆
### 문제
`apply_lab_clahe` 전후의 local contrast와 clipping 비율을 비교한다.
### 입력
합성 타일.
### 요구 산출물
전후 진단 표와 이미지 두 장.
### 힌트
`before=analyze_image(image)`, `after=analyze_image(apply_lab_clahe(image))` 순서다.
### 해설
L 채널만 CLAHE 처리하므로 색상 성분을 직접 평활화하지 않는다고 설명한다. local contrast 상승과 clipping 변화가 함께 나타나는지 기록한다.
### 평가 기준
`clipLimit`, `tileGridSize` 의미와 trade-off를 적으면 통과.

## EX6: 세 필터의 noise 가정 비교
### 난이도
유도형 · ★★☆
### 문제
Gaussian, median, bilateral 결과를 원본과 비교한다.
### 입력
합성 타일에 임의의 salt-and-pepper noise를 추가한 배열.
### 요구 산출물
3열 비교 이미지, noise estimate·sharpness 표.
### 힌트
`apply_gaussian`, `apply_median`, `apply_bilateral`을 같은 입력에 적용한다.
### 해설
median은 impulse noise 후보, Gaussian은 일반 평활화, bilateral은 에지 보존 후보라는 가정을 적고 실제 수치와 시각 결과로 반례를 찾는다.
### 평가 기준
필터 이름만 나열하지 않고 oversmoothing 가능성을 판단하면 통과.

## EX7: 명시적 blur 뒤 Canny
### 난이도
유도형 · ★★☆
### 문제
원본 Canny와 Gaussian blur 후 Canny를 비교한다.
### 입력
합성 타일 또는 선이 많은 이미지.
### 요구 산출물
두 에지 맵, nonzero edge pixel 수, 선택 이유.
### 힌트
`cv2.GaussianBlur`를 먼저 호출하고 `cv2.Canny`는 별도 호출한다.
### 해설
blur는 Canny의 자동 단계가 아니므로 코드에 분리해 쓴다. 에지가 줄었다면 noise와 미세 경계가 함께 줄었을 가능성을 확인한다.
### 평가 기준
Canny 단계와 사전 평활화를 구분하면 통과.

## EX8: morphology blackhat 실험
### 난이도
유도형 · ★★☆
### 문제
밝은 배경의 어두운 선을 강조하는 blackhat을 세 kernel 모양으로 비교한다.
### 입력
합성 타일 또는 직접 그린 어두운 선 이미지.
### 요구 산출물
rect·ellipse·cross 출력과 선택 근거.
### 힌트
`apply_blackhat(image, kernel_size=9, shape=...)`를 사용한다.
### 해설
kernel은 물체보다 큰 배경 구조를 가정한다. 선 방향과 두께에 따라 반응이 달라지는지 관찰하고 결과를 BGR 3채널로 반환하는 구현 계약도 확인한다.
### 평가 기준
구조 요소의 크기·모양이 결과를 바꾼다는 설명이 있으면 통과.

## EX9: HOG descriptor 계약 확인
### 난이도
유도형 · ★★☆
### 문제
`HOGExtractor(size=(128,128))`의 두 번 transform 결과가 같은지 확인한다.
### 입력
합성 타일을 복제한 리스트 두 장.
### 요구 산출물
matrix shape, dtype, 결정성 assert.
### 힌트
`np.allclose(first, second)`와 `features.dtype`를 출력한다.
### 해설
입력은 extractor 내부에서 resize되고 출력은 `float32` 행렬이다. configured size가 16의 배수가 아니면 생성자 오류가 나는 것도 별도 확인한다.
### 평가 기준
원본 크기와 configured size를 구별하면 통과.

## EX10: 작은 feature matrix 표준화
### 난이도
유도형 · ★★☆
### 문제
훈련 행으로만 `Standardizer`를 fit하고 test 행을 transform한다.
### 입력
직접 만든 4×2 `float32` feature matrix와 train/test 인덱스.
### 요구 산출물
train 평균 0 근처 확인, test 변환값, 누수 설명.
### 힌트
`Standardizer().fit(matrix[train])`를 사용한다.
### 해설
전체 matrix의 평균으로 fit한 값과 train-only 값을 비교해 test 정보가 결과에 들어가는 방식을 보여 준다.
### 평가 기준
fit과 transform의 역할, test fold 미사용 이유를 설명하면 통과.

## 분석형: 관찰에서 가설과 검증을 설계하는 10개

## EX11: 엔트로피 상승의 원인 분석
### 난이도
분석형 · ★★☆
### 문제
CLAHE 뒤 엔트로피가 오른 사례에서 “개선”인지 “노이즈 증폭”인지 판단 절차를 설계한다.
### 입력
원본과 CLAHE 결과.
### 요구 산출물
최소 세 진단 지표, 시각 확인 항목, 결론 조건.
### 힌트
noise estimate, sharpness, edge density를 함께 본다.
### 해설
엔트로피만 보고 결론내리지 않는다. noise 상승, 과도한 edge, clipping 증가가 있으면 강화의 비용으로 기록하고 레이블 데이터가 있으면 같은 split에서 평가한다.
### 평가 기준
단일 수치 결론을 피하고 반증 조건을 쓰면 통과.

## EX12: 조명 불균일 이미지의 후보 순위
### 난이도
분석형 · ★★☆
### 문제
조명 불균일도가 높은 이미지에 대해 normalize, auto gamma, LAB CLAHE의 실험 순서를 제안한다.
### 입력
밝기 그라데이션을 합성한 타일.
### 요구 산출물
후보 순서, 각 후보의 가설, 중단 경고.
### 힌트
전후 `illumination_nonuniformity`, clipping, local contrast를 기록한다.
### 해설
후보 순서는 휴리스틱일 뿐 정확도가 아니다. 각 결과가 clipping을 키우거나 색·에지 경고를 내면 다음 후보를 무조건 채택하지 않고 원본도 남긴다.
### 평가 기준
원본 baseline과 실패 조건을 넣으면 통과.

## EX13: filter kernel ablation 표 설계
### 난이도
분석형 · ★★★
### 문제
Gaussian과 median의 kernel 3, 5, 7을 비교하는 표와 해석 규칙을 설계한다.
### 입력
노이즈가 추가된 합성 타일.
### 요구 산출물
실험 표 템플릿, 반복 조건, 최종 선택 규칙.
### 힌트
kernel 외에는 입력·dtype·평가 지표를 고정한다.
### 해설
행은 filter·kernel, 열은 noise estimate, sharpness, edge continuity, 처리시간, 시각 메모로 둔다. 하나의 최대값 대신 요구되는 구조 보존과 처리 제약을 기준으로 선택한다.
### 평가 기준
통제 변수와 oversmoothing 확인란이 있으면 통과.

## EX14: threshold 실패 사례 진단
### 난이도
분석형 · ★★☆
### 문제
전역 threshold가 배경 그라데이션에서 실패하는 이유와 대안을 비교한다.
### 입력
어두운 도형과 밝기 그라데이션을 합성한 grayscale 이미지.
### 요구 산출물
실패 이미지, 두 대안, false positive/negative 설명.
### 힌트
배경 보정, adaptive threshold, LAB L 전처리를 후보로 둔다.
### 해설
하나의 threshold는 위치별 배경 차이를 무시한다. 대안도 텍스처를 객체로 오인할 수 있으므로 mask의 개수·면적과 원본 overlay를 같이 확인한다.
### 평가 기준
대안의 새 실패 가능성까지 쓰면 통과.

## EX15: contour와 component 선택 근거
### 난이도
분석형 · ★★☆
### 문제
이진 마스크에서 “결함 개수와 면적”을 구할 때 contour와 components 중 하나를 선택하고 반대 선택의 장단점을 쓴다.
### 입력
직접 만든 서로 떨어진 도형 마스크.
### 요구 산출물
선택 API, 반환할 통계, 경계 조건.
### 힌트
면적·bounding box는 `connectedComponentsWithStats`가 직접 제공한다.
### 해설
연결 영역의 개수·면적이면 components를 우선 사용하고, 정밀한 외곽선·둘레·형상 기술이면 contour를 택한다. 전경/배경과 connectivity를 명시한다.
### 평가 기준
문제 요구에 따라 선택하고 이진화 품질을 전제로 쓰면 통과.

## EX16: 색 히스토그램의 공간 정보 손실 증명
### 난이도
분석형 · ★★★
### 문제
같은 색 픽셀 개수지만 배치가 다른 두 이미지를 만들어 히스토그램의 한계를 보인다.
### 입력
두 색을 좌우로 나눈 이미지와 체커보드 이미지.
### 요구 산출물
히스토그램 비교, HOG 또는 texture 차이, 해석.
### 힌트
두 이미지의 색 개수를 정확히 같게 만든다.
### 해설
색 히스토그램은 비슷하거나 같을 수 있지만 공간 배치가 다른 이미지는 HOG·Gabor에서 차이를 낼 수 있다. 그래서 combined feature를 쓰는 이유가 된다.
### 평가 기준
“히스토그램이 나쁘다” 대신 어떤 정보를 버리는지 보이면 통과.

## EX17: HOG resize 영향 분석
### 난이도
분석형 · ★★★
### 문제
같은 이미지를 다른 aspect ratio로 resize했을 때 HOG가 어떻게 달라질지 가설을 세운다.
### 입력
가로로 긴 도형 이미지.
### 요구 산출물
두 resize 결과, descriptor 차이 수치, 형태 왜곡 설명.
### 힌트
현재 extractor는 `cv2.resize(image, self.size)`를 사용한다.
### 해설
고정 크기는 descriptor 길이를 맞추지만 종횡비를 변형할 수 있다. padding 또는 crop을 대안으로 제시하되, feature 정의가 바뀌므로 benchmark 전체를 다시 실행해야 한다.
### 평가 기준
비교 조건을 다시 맞춰야 한다는 결론이 있으면 통과.

## EX18: Macro F1과 accuracy가 갈리는 예 만들기
### 난이도
분석형 · ★★★
### 문제
다수 클래스만 잘 맞혀 accuracy는 높고 Macro F1은 낮은 예측 배열을 만든다.
### 입력
직접 만든 truth/predicted 정수 배열과 class_count 3.
### 요구 산출물
confusion matrix, accuracy, Macro F1, 해석.
### 힌트
소수 클래스의 예측을 모두 다수 클래스로 보낸다.
### 해설
`classification_metrics`로 계산하고 per-class F1이 평균에 동등하게 들어감을 확인한다. 운영 목표가 소수 결함 검출이면 이 차이를 무시할 수 없다.
### 평가 기준
지표 정의와 실제 오류 사례를 연결하면 통과.

## EX19: stratified split 감사
### 난이도
분석형 · ★★★
### 문제
불균형 label 배열에 `stratified_folds`를 적용해 각 test fold의 클래스 수를 감사한다.
### 입력
클래스별 5, 7, 9개 표본으로 만든 label 배열.
### 요구 산출물
fold별 class count 표, seed 변경 비교, 한계.
### 힌트
각 fold의 `test_indices`로 labels를 인덱싱한다.
### 해설
seed가 바뀌면 개별 인덱스는 달라도 클래스별 균형 원칙은 유지된다. 동일 제품의 유사 이미지가 섞이면 stratification만으로는 leakage를 막지 못한다.
### 평가 기준
분포 확인과 group leakage 한계를 함께 쓰면 통과.

## EX20: 원본 우세 결과의 반증 계획
### 난이도
분석형 · ★★★
### 문제
Original이 CLAHE 후보보다 높은 Macro F1을 보인 결과에 대해 세 개의 가능한 가설과 검증 실험을 쓴다.
### 입력
[benchmark evidence](../portfolio/benchmark-evidence.json)의 리더보드.
### 요구 산출물
가설-관찰-다음 실험 표.
### 힌트
class별 혼동, feature 분포, parameter ablation을 분리한다.
### 해설
“전처리가 나쁘다”로 끝내지 않는다. 대비 강화가 특정 texture를 바꿨다는 가설은 original 포함 ablation과 동일 fold 보고서로 검증한다.
### 평가 기준
확정 인과 대신 반증 가능한 계획을 제시하면 통과.

## 재구현형: 빈 화면에서 프로젝트 경로를 다시 만드는 10개

## EX21: Unicode-safe decode 재구현
### 난이도
재구현형 · ★★★
### 문제
`decode_image(path)`를 별도 파일에 테스트 우선으로 재구현한다.
### 입력
합성 타일과 존재하지 않는 경로 문자열.
### 요구 산출물
실패 테스트, 구현, 성공 테스트 로그.
### 힌트
`Path.read_bytes`, `np.frombuffer`, `cv2.imdecode`를 조합한다.
### 해설
파일 부재·디코드 실패·BGR 계약을 각각 예외로 처리한다. 기존 [io.py](../../src/opencv_preprocessing_advisor/io.py)와 비교해 API와 실패 메시지를 점검한다.
### 평가 기준
RED→GREEN 기록과 3채널 uint8 검증이 있으면 통과.

## EX22: LAB CLAHE 함수 재구현
### 난이도
재구현형 · ★★★
### 문제
입력 shape/dtype을 보존하는 `apply_lab_clahe`를 테스트 우선으로 작성한다.
### 입력
저대비 BGR 배열과 잘못된 `clip_limit` 값.
### 요구 산출물
형태 보존 테스트, 양수 파라미터 오류 테스트, 구현.
### 힌트
LAB split 후 L에만 `cv2.createCLAHE(...).apply`한다.
### 해설
L,a,b를 merge해 BGR로 돌리고 출력이 입력과 같은 shape·dtype인지 확인한다. pixel 값이 반드시 좋아진다는 테스트는 쓰지 않는다.
### 평가 기준
색 채널 보존 의도와 parameter validation을 구현하면 통과.

## EX23: diagnostic delta 계산 재구현
### 난이도
재구현형 · ★★★
### 문제
두 진단 dataclass에서 absolute delta와 percent delta를 만드는 함수를 작성한다.
### 입력
0을 포함한 작은 mock diagnostics 두 개.
### 요구 산출물
0 baseline 처리 테스트와 반환 표.
### 힌트
분모 절댓값이 매우 작으면 percent를 `None`으로 둔다.
### 해설
각 필드를 같은 이름으로 매칭하고 `after-before`를 계산한다. 0으로 나누어 무한대가 나오는 것보다 “비교 불가”를 명시하는 편이 정직하다.
### 평가 기준
모든 필드 순회와 0 분모 케이스를 다루면 통과.

## EX24: YAML pipeline mini catalog 재구현
### 난이도
재구현형 · ★★★
### 문제
두 단계 transform을 가진 작은 YAML을 읽어 순서대로 실행하는 catalog를 만든다.
### 입력
`gray_bgr`와 `normalize` 두 step이 든 임시 YAML.
### 요구 산출물
순서 테스트, 미지 transform 오류 테스트, 실행 결과.
### 힌트
transform 이름을 callable registry에서 찾아 실행한다.
### 해설
각 step의 파라미터 사본과 intermediate output을 남긴다. 설정 순서를 보존해야 `A→B`와 `B→A`의 차이를 추적할 수 있다.
### 평가 기준
순서·유효성·중간 결과를 모두 검증하면 통과.

## EX25: profile-weight score 재구현
### 난이도
재구현형 · ★★★
### 문제
두 진단 변화와 가중치 dict를 받아 0~100 범위의 단순 suitability score를 작성한다.
### 입력
local contrast·noise·clipping이 다른 두 후보.
### 요구 산출물
가중치 합 오류 테스트, 순위 테스트, 이유 문자열.
### 힌트
가중치 합을 1.0으로 검사하고 clipping 증가에는 감점을 준다.
### 해설
이는 accuracy가 아닌 실험 우선순위 함수라는 docstring을 쓴다. 기존 [scoring.py](../../src/opencv_preprocessing_advisor/scoring.py)의 세부 수식을 복사하지 말고 최소 계약부터 검증한다.
### 평가 기준
점수 범위·가중치 검증·비성능 경계가 있으면 통과.

## EX26: 고정 길이 색 히스토그램 재구현
### 난이도
재구현형 · ★★★
### 문제
HSV H, HSV S, LAB L의 32-bin 히스토그램을 이어 붙이고 합이 1인 feature를 만든다.
### 입력
두 BGR 이미지 리스트.
### 요구 산출물
matrix shape, `float32`, 행별 합 테스트.
### 힌트
H 채널의 범위는 0~180, 나머지는 0~256이다.
### 해설
각 histogram을 concatenate 후 전체 합으로 나눈다. 빈·검은 이미지도 finite 값이 나와야 하며, 공간 배치를 잃는다는 한계를 문서화한다.
### 평가 기준
세 범위와 정규화 계약을 정확히 구현하면 통과.

## EX27: SIFT BoW fit/transform 재구현 계획
### 난이도
재구현형 · ★★★
### 문제
코드를 모두 쓰기 전에 fold-safe SIFT BoW의 인터페이스와 테스트를 설계한다.
### 입력
권한 있는 작은 이미지 리스트 또는 합성 도형 이미지.
### 요구 산출물
`fit(train_images)`, `transform(images)` 계약, vocabulary 부족 오류, CV 의사코드.
### 힌트
test fold descriptor는 vocabulary clustering에 절대 넣지 않는다.
### 해설
각 outer fold 안에서 SIFT vocabulary를 train 이미지로 fit하고 train/test histogram을 만든다. 이 규칙이 없으면 기존 구현을 benchmark에 연결하지 않는다.
### 평가 기준
누수 방지 테스트와 부족 descriptor 실패 경로가 있으면 통과.

## EX28: OpenCV classifier adapter 재구현
### 난이도
재구현형 · ★★★
### 문제
`fit(features, labels)`와 `predict(features)` 계약을 따르는 kNN 어댑터를 작성한다.
### 입력
작은 2차원 `float32` feature matrix와 int32 labels.
### 요구 산출물
fit 전 predict 오류, shape 오류, 성공 예측 테스트.
### 힌트
`cv2.ml.KNearest_create`, `ROW_SAMPLE`, `findNearest`를 사용한다.
### 해설
입력 행 수와 label 길이를 검사하고 k가 training rows보다 크지 않게 제한한다. accuracy보다 인터페이스·dtype 계약을 먼저 검증한다.
### 평가 기준
세 실패 경로와 one-pass prediction을 통과시키면 통과.

## EX29: fold-local cross validation 재구현
### 난이도
재구현형 · ★★★
### 문제
주어진 folds에 대해 scaler·classifier를 매 fold 새로 만들어 accuracy와 Macro F1을 평균내는 함수를 작성한다.
### 입력
합성 feature/label matrix와 `stratified_folds` 결과.
### 요구 산출물
train-only scaler spy 또는 기록, fold별 metric 표, 평균.
### 힌트
fold loop 안에서 `Standardizer().fit(features[train_indices])`를 호출한다.
### 해설
fit 객체를 fold 밖에 만들지 않는다. confusion matrix도 test labels와 prediction만으로 만들고, 성능은 작은 합성 데이터의 동작 확인일 뿐 일반화 결론이 아님을 적는다.
### 평가 기준
누수 없는 fit 위치와 fold별 기록이 있으면 통과.

## EX30: 재현 가능한 mini benchmark 보고서
### 난이도
재구현형 · ★★★
### 문제
합성 도형 세 클래스로 original과 한 전처리 후보를 비교하고 CSV/JSON 보고서를 만든다.
### 입력
코드로 생성한 클래스 폴더 또는 메모리 이미지·labels.
### 요구 산출물
seed, fold 수, pipeline 이름, Accuracy, Macro F1, config hash 또는 설정 전문.
### 힌트
원본 baseline, 동일 feature·classifier·fold를 반드시 포함한다.
### 해설
결과 순위보다 실험 메타데이터와 실패 사례를 남기는 것이 목표다. 결과가 좋더라도 공식 benchmark나 실사용 성능이라고 이름 붙이지 않는다.
### 평가 기준
동일 조건 비교, 재현 정보, 한계 문구, 원본 baseline이 모두 있으면 통과.
