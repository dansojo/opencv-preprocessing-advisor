# 기술 면접 Q&A 32개

답을 읽기 전 60초 동안 먼저 말해 보세요. 각 모범 답변은 **왜**, **피할 때**, **파라미터/조건**, **프로젝트 근거**를 한 흐름으로 연결합니다.

## Q01: BGR과 RGB를 왜 구분하나요?

### 모범 답변
OpenCV 기본 입력은 BGR이므로 RGB라고 가정하면 채널 의미가 뒤바뀝니다. 표시 라이브러리 경계에서만 변환하고, 입력 계약이 BGR인지 확인하지 못할 때는 색 기반 판단을 피합니다. 색상 범위와 채널 순서는 코드로 확인하며, 프로젝트 근거는 [I/O 검증](../../src/opencv_preprocessing_advisor/io.py)입니다.

## Q02: shape와 dtype을 먼저 보는 이유는 무엇인가요?

### 모범 답변
shape는 채널과 공간축, dtype은 값 범위와 산술 해석의 전제이기 때문입니다. 예상과 다른 배열에는 필터를 적용하지 않고 오류로 처리합니다. `uint8`/3채널 조건은 [validate_bgr_image](../../src/opencv_preprocessing_advisor/io.py)에 명시되어 있습니다.

## Q03: Gray 변환은 언제 피하나요?

### 모범 답변
색 차이가 클래스나 결함 단서일 때 Gray는 그 정보를 버리므로 피합니다. 밝기·경계만 필요한 단계라면 계산을 단순하게 할 수 있지만 목적을 확인해야 합니다. 프로젝트는 색 특징에 HSV/LAB을 별도로 쓰며 [features.py](../../src/opencv_preprocessing_advisor/features.py)가 근거입니다.

## Q04: HSV를 왜 쓰나요?

### 모범 답변
색상·채도·밝기를 분리해 색 분포를 요약하기 편하기 때문입니다. 매우 낮은 채도에서는 hue 해석을 과신하지 않으며 OpenCV 채널 범위를 코드에서 확인합니다. 실제 H/S 히스토그램 범위는 [ColorHistogramExtractor](../../src/opencv_preprocessing_advisor/features.py)에 있습니다.

## Q05: LAB L 채널만 CLAHE하는 이유는 무엇인가요?

### 모범 답변
명도 대비를 보정하면서 a/b 색 성분을 분리해 유지하려는 선택입니다. 색이 중요한 입력에 BGR 채널별 대비 증폭을 무조건 적용하지 않고 전후 색 변화를 비교합니다. L만 변환하는 구현은 [apply_lab_clahe](../../src/opencv_preprocessing_advisor/transforms.py)에 있습니다.

## Q06: CLAHE의 clip limit는 무엇을 바꾸나요?

### 모범 답변
국소 히스토그램의 과도한 증폭을 제한하는 조건입니다. 값을 크게 하면 대비뿐 아니라 잡음도 두드러질 수 있어 큰 값이 항상 낫지 않습니다. 프로젝트는 양수만 허용하며 [transforms.py](../../src/opencv_preprocessing_advisor/transforms.py)와 설정 [pipelines.yaml](../../src/opencv_preprocessing_advisor/config/pipelines.yaml)에서 확인합니다.

## Q07: gamma와 normalization의 차이는 무엇인가요?

### 모범 답변
normalization은 값 범위를 재조정하고 gamma는 중간톤을 비선형으로 이동합니다. 거의 완전한 검정·흰색 입력에는 자동 gamma를 과신하지 않으며 프로젝트는 원본 복사 보호 조건을 둡니다. 근거는 [apply_auto_gamma](../../src/opencv_preprocessing_advisor/transforms.py)입니다.

## Q08: Gaussian과 Median은 언제 구분하나요?

### 모범 답변
Gaussian은 주변값을 부드럽게 평균하는 선택이고 Median은 점 잡음처럼 극단값이 섞인 경우 비교할 후보입니다. 둘 다 경계를 바꿀 수 있으므로 노이즈 유형을 확인하지 않으면 선택을 보류합니다. 두 구현의 kernel은 [transforms.py](../../src/opencv_preprocessing_advisor/transforms.py)에서 양의 홀수로 검증됩니다.

## Q09: Bilateral을 무조건 쓰지 않는 이유는 무엇인가요?

### 모범 답변
색·공간 거리 조건으로 경계를 보존하려 하지만 계산 비용과 결과 특성이 입력에 따라 달라집니다. 빠른 처리나 경계 보존이 불필요한 경우에는 다른 후보를 먼저 비교합니다. diameter와 sigma 조건은 [apply_bilateral](../../src/opencv_preprocessing_advisor/transforms.py)에 있습니다.

## Q10: Sobel과 Scharr의 차이는 어떻게 설명하나요?

### 모범 답변
둘 다 방향별 밝기 변화의 근사이며, 작은 kernel에서 Scharr를 비교 후보로 둘 수 있습니다. 어느 것이 우수하다고 일반화하지 않고 입력 크기·노이즈·후속 임계값을 고정해 비교합니다. 프로젝트의 질감 파이프는 Sobel을 쓰며 [features.py](../../src/opencv_preprocessing_advisor/features.py)가 근거입니다.

## Q11: Laplacian을 쓸 때 위험은 무엇인가요?

### 모범 답변
2차 변화는 급격한 변화뿐 아니라 잡음에도 민감할 수 있습니다. 그래서 noise가 많은 입력에는 평활화와 결과 비교 없이 단독 임계값 판정을 피합니다. 프로젝트는 Laplacian 응답을 통계 특징으로 요약하며 [TextureStatsExtractor](../../src/opencv_preprocessing_advisor/features.py)에 있습니다.

## Q12: Canny의 threshold는 어떻게 조정하나요?

### 모범 답변
낮고 높은 임계값은 남길 경계와 연결될 약한 경계에 영향을 줍니다. 한 이미지만 보고 고정하지 않고 입력, blur, 두 threshold 외 조건을 고정해 비교합니다. 이 프로젝트는 Canny를 확정 결함 판정이 아닌 진단 후보로 다루며 [evidence map](../portfolio/evidence-map.md)을 참고합니다.

## Q13: morphology kernel 크기는 왜 기록하나요?

### 모범 답변
kernel은 제거·연결되는 구조의 규모를 정하므로 결과 영역 수와 면적을 바꿉니다. 대상 결함보다 큰 kernel을 무심코 쓰면 신호도 지울 수 있습니다. 설정 기반 후보는 [pipelines.yaml](../../src/opencv_preprocessing_advisor/config/pipelines.yaml)로 재현합니다.

## Q14: contours와 connected components는 무엇이 다른가요?

### 모범 답변
둘 다 이진 영역을 다루지만 contour는 경계 형태, connected components는 연결된 라벨 영역과 통계를 얻는 데 편합니다. 마스크 품질이 나쁘면 어느 결과도 과신하지 않습니다. 프로젝트의 영역 진단 근거는 [diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py)입니다.

## Q15: 히스토그램의 한계는 무엇인가요?

### 모범 답변
색이나 밝기 분포는 담지만 위치와 형상 배치는 버립니다. 위치가 중요한 결함을 히스토그램 단독으로 구분하려 하지 않고 HOG·질감과 결합합니다. 실제 조합은 [CombinedExtractor](../../src/opencv_preprocessing_advisor/features.py)에 있습니다.

## Q16: HOG 입력 크기를 고정하는 이유는 무엇인가요?

### 모범 답변
고정 셀·블록 구조에서 일관된 길이 벡터를 만들기 위해서입니다. resize가 작은 패턴을 왜곡할 수 있어 원본 종횡비와 목표 크기를 기록합니다. 프로젝트는 128×128 기본값과 16 배수 검증을 [HOGExtractor](../../src/opencv_preprocessing_advisor/features.py)에 둡니다.

## Q17: Gabor의 방향 파라미터는 왜 중요하나요?

### 모범 답변
Gabor는 특정 방향·주기의 질감에 반응하도록 설계되므로 theta가 관찰되는 응답을 바꿉니다. 한 방향만으로 모든 질감을 대표한다고 말하지 않고 여러 방향을 비교합니다. 프로젝트는 네 방향을 사용하며 [TextureStatsExtractor](../../src/opencv_preprocessing_advisor/features.py)가 근거입니다.

## Q18: SIFT BoW에서 vocabulary는 어디서 fit하나요?

### 모범 답변
학습 fold 이미지의 descriptor만으로 fit해야 합니다. test descriptor까지 포함하면 표현 공간에 평가 정보가 새어 들어가므로 피합니다. vocabulary 크기보다 descriptor가 적으면 실패시키는 보호는 [SiftBowExtractor](../../src/opencv_preprocessing_advisor/features.py)에 있습니다.

## Q19: SVM 전에 scaling이 필요한 이유는 무엇인가요?

### 모범 답변
특징 스케일 차이가 거리나 경계 학습에 과도하게 반영될 수 있기 때문입니다. 전체 데이터로 scaler를 fit하면 누수이므로 train fold에서만 fit합니다. 프로젝트의 순서는 [cross_validate](../../src/opencv_preprocessing_advisor/evaluation.py)에 구현돼 있습니다.

## Q20: kNN의 k는 무엇을 바꾸나요?

### 모범 답변
k는 참조하는 이웃 수를 바꿔 지역적 민감도와 평균화 정도에 영향을 줍니다. 클래스 불균형·특징 스케일·데이터 밀도를 보지 않고 기본 k를 일반화하지 않습니다. kNN은 [classifiers.py](../../src/opencv_preprocessing_advisor/classifiers.py)에서 동일 평가 흐름의 비교기입니다.

## Q21: RTrees 결과를 어떻게 해석하나요?

### 모범 답변
RTrees는 이 프로젝트에서 같은 데이터·feature·fold 조건에서 비교한 분류기 중 하나입니다. 다른 데이터셋에서도 최고라고 말하지 않으며 seed와 설정을 함께 남깁니다. 실제 비교 결과와 조건은 [experiment results](../portfolio/experiment-results.md)에 있습니다.

## Q22: accuracy만으로 충분하지 않은 이유는 무엇인가요?

### 모범 답변
클래스별 오류가 가려질 수 있기 때문입니다. 클래스가 불균형하거나 각 클래스의 균형을 보고 싶을 때 Macro F1과 혼동행렬을 함께 봅니다. 프로젝트는 accuracy와 Macro F1을 같이 계산하며 [evaluation.py](../../src/opencv_preprocessing_advisor/evaluation.py)가 근거입니다.

## Q23: Macro F1은 어떻게 계산하나요?

### 모범 답변
각 클래스의 precision·recall에서 F1을 계산한 뒤 클래스별 F1의 평균을 취합니다. support가 작아도 동일 비중이므로 표본 수와 per-class 지표를 함께 봅니다. 구현은 [classification_metrics](../../src/opencv_preprocessing_advisor/evaluation.py)에 있습니다.

## Q24: confusion matrix는 무엇을 알려 주나요?

### 모범 답변
행의 실제 클래스가 열의 예측 클래스로 어떻게 이동했는지 보여 주어 특정 혼동을 찾게 합니다. 원인이라고 단정하지 않고 샘플·전처리·특징을 추가 확인합니다. 행렬 생성은 [confusion_matrix](../../src/opencv_preprocessing_advisor/evaluation.py)에 있습니다.

## Q25: data leakage의 예를 하나 드세요.

### 모범 답변
테스트 fold를 포함한 전체 feature로 scaler나 BoW vocabulary를 fit하는 경우입니다. 성능이 좋아 보여도 배포 전에 알 수 없는 분포를 사용했으므로 피합니다. 프로젝트는 fold 안에서 scaler를 fit하는 [cross_validate](../../src/opencv_preprocessing_advisor/evaluation.py)를 사용합니다.

## Q26: heuristic 추천 점수는 accuracy인가요?

### 모범 답변
아닙니다. 단일 이미지의 진단 신호를 바탕으로 후보 적합성을 설명하는 점수이며 레이블 기반 정확도와 다른 출력입니다. 이를 성능 수치로 발표하지 않고 데이터셋 평가는 별도 실험으로 분리합니다. 근거는 [case study](../portfolio/case-study.md)와 [scoring.py](../../src/opencv_preprocessing_advisor/scoring.py)입니다.

## Q27: 파이프라인 설정 파일을 왜 쓰나요?

### 모범 답변
후보 이름과 파라미터를 코드 변경 없이 재현·비교하기 위해서입니다. 설정이 존재해도 입력 데이터나 평가 규칙을 기록하지 않으면 재현이 완전하지 않습니다. 현재 후보 설정은 [pipelines.yaml](../../src/opencv_preprocessing_advisor/config/pipelines.yaml)에 있습니다.

## Q28: 테스트는 문서화에 어떻게 도움이 되나요?

### 모범 답변
테스트는 함수가 어떤 입력과 실패 조건을 약속하는지 실행 가능한 증거로 남깁니다. 테스트가 없는 관찰을 보편적 사실로 쓰지 않고 재현 코드를 붙입니다. 변환·평가 계약은 [test_transforms.py](../../tests/test_transforms.py)와 [test_evaluation.py](../../tests/test_evaluation.py)에 있습니다.

## Q29: 실패 사례를 어떻게 보고하나요?

### 모범 답변
관찰된 입력·예측·지표를 먼저 제시하고, 원인은 전처리·특징·모델의 검증할 가설로 구분합니다. 한 개의 시각적 사례로 전체 성능을 단정하지 않습니다. 프로젝트의 공개 한계는 [limitations](../portfolio/limitations.md)에 있습니다.

## Q30: 결과가 좋지 않으면 무엇부터 확인하나요?

### 모범 답변
입력 계약, 데이터 분할, feature shape/dtype, scaler fit 위치, confusion matrix 순으로 확인합니다. 파라미터를 무작위로 넓히기 전에 재현 가능한 최소 실패를 만듭니다. 평가 흐름은 [evaluation.py](../../src/opencv_preprocessing_advisor/evaluation.py)에서 추적할 수 있습니다.

## Q31: 이 프로젝트의 가장 중요한 설계 분리는 무엇인가요?

### 모범 답변
단일 이미지 전처리 추천과 데이터셋 분류 평가는 서로 다른 목적·입력·출력을 가진다는 분리입니다. 추천 점수를 정확도로 해석하는 일을 피하고 각각의 근거를 따로 제시합니다. [services.py](../../src/opencv_preprocessing_advisor/services.py)와 [experiment results](../portfolio/experiment-results.md)가 근거입니다.

## Q32: 면접에서 모르는 API를 받으면 어떻게 답하나요?

### 모범 답변
확인하지 않은 동작을 단정하지 않고 입력·출력·파라미터·실패 조건을 먼저 확인하겠다고 말합니다. 그 API가 필요한 문제인지, 기존 후보와 무엇을 비교할지도 제안합니다. 이 프로젝트에서도 근거 링크와 테스트를 우선하는 방식은 [evidence map](../portfolio/evidence-map.md)으로 확인할 수 있습니다.
