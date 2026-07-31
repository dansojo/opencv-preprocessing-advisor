# 프로젝트·면접 질문과 모범 답안

답안을 외우기보다 자신의 실험 증거로 바꿔 말한다. 숫자는 [benchmark evidence](../portfolio/benchmark-evidence.json)의 117장·6클래스·seed 42 실험 범위이며, **not official MVTec anomaly-detection result**라는 경계를 유지한다.

## IQ1: 이 프로젝트는 어떤 문제를 해결하나요?
### 30초 답변
전처리를 예쁘게 보이게 하는 작업과 분류에 도움이 되는 작업을 분리해, 한 장에는 설명 가능한 후보를 제안하고 레이블 데이터에는 교차 검증을 적용한 프로젝트입니다.
### 2분 심화 답변
처음에는 대비를 높이면 좋아질 것이라는 가정을 두지 않았습니다. 진단 지표로 후보를 정하고, 별도 benchmark에서 원본과 후보를 같은 fold·특징·분류기 조건으로 비교해 전처리의 실제 영향을 확인하도록 설계했습니다.
### 근거 코드·결과
[services.py](../../src/opencv_preprocessing_advisor/services.py)의 `ImageAdvisorService`와 `BenchmarkService`가 분리돼 있습니다.
### 추가 질문
“후보 점수와 정확도를 분리하지 않으면 어떤 오해가 생기나요?”

## IQ2: 요구사항은 어떻게 진화했나요?
### 30초 답변
필터를 나열하는 도구에서, 진단·이유·경고를 가진 추천과 재현 가능한 데이터셋 평가를 제공하는 도구로 확장했습니다.
### 2분 심화 답변
단일 이미지에는 정답 레이블이 없으므로 추천 점수를 성능처럼 보이게 하지 않는 것이 핵심이었습니다. 따라서 profile별 가중치와 경고를 YAML로 분리하고, 성능 평가는 feature·classifier·fold가 있는 별도 서비스로 만들었습니다.
### 근거 코드·결과
[scoring.yaml](../../src/opencv_preprocessing_advisor/config/scoring.yaml), [pipelines.yaml](../../src/opencv_preprocessing_advisor/config/pipelines.yaml)을 확인할 수 있습니다.
### 추가 질문
“다음 요구 변경이 들어오면 어느 계층을 바꾸겠습니까?”

## IQ3: 왜 LAB L-channel CLAHE를 선택했나요?
### 30초 답변
밝기만 조정하고 색상 성분 a·b를 직접 바꾸지 않아 색 관계 교란을 줄이려는 후보였습니다.
### 2분 심화 답변
각 BGR 채널을 독립적으로 평활화하면 색조가 변할 수 있습니다. LAB에서 L만 CLAHE 처리하면 밝기 대비를 조절하는 의도가 코드에 드러나지만, 결과가 항상 우수하다는 뜻은 아니므로 원본과 benchmark로 검증했습니다.
### 근거 코드·결과
`apply_lab_clahe` 구현은 [transforms.py](../../src/opencv_preprocessing_advisor/transforms.py)에 있습니다.
### 추가 질문
“`tileGridSize`가 커지면 타일 크기는 어떻게 바뀌나요?”

## IQ4: CLAHE 파라미터는 어떻게 판단했나요?
### 30초 답변
`clipLimit`과 `tileGridSize`를 고정 정답으로 두지 않고 진단 변화, 경고, 검증 결과를 함께 봤습니다.
### 2분 심화 답변
clipLimit가 커지면 국소 대비와 노이즈가 같이 커질 수 있고, tile 수가 늘면 타일은 더 작아집니다. 그래서 한 장의 시각적 인상 대신 clipping, edge, noise와 분류 지표를 비교 대상으로 삼았습니다.
### 근거 코드·결과
[Day 3](day-03.md)와 [transforms.py](../../src/opencv_preprocessing_advisor/transforms.py)가 파라미터 의미를 문서화합니다.
### 추가 질문
“어떤 경우 global histogram equalization을 피하겠습니까?”

## IQ5: Gaussian, median, bilateral 중 무엇을 고르나요?
### 30초 답변
노이즈 가정과 보존해야 할 구조를 먼저 정하고, Gaussian은 일반 평활화, median은 impulse noise, bilateral은 에지 보존 후보로 비교합니다.
### 2분 심화 답변
필터 이름만으로 선택하지 않습니다. kernel이 커질 때 얇은 결함과 질감이 얼마나 사라지는지, 처리 시간이 허용되는지, feature 분포가 어떻게 바뀌는지를 원본 baseline과 함께 측정합니다.
### 근거 코드·결과
[transforms.py](../../src/opencv_preprocessing_advisor/transforms.py)와 [Day 4](day-04.md)의 실험 절차가 근거입니다.
### 추가 질문
“oversmoothing을 어떤 지표로 경고하나요?”

## IQ6: oversmoothing을 어떻게 처리했나요?
### 30초 답변
필터 후 sharpness가 원본의 절반 미만이면 경고를 표시해 사용자가 결과를 재검토하게 했습니다.
### 2분 심화 답변
Laplacian 분산은 완전한 품질 지표가 아니므로 자동 거부 규칙으로 쓰지 않았습니다. 노이즈 감소와 경계 보존의 trade-off를 보여 주는 보수적 신호로 사용했습니다.
### 근거 코드·결과
경고 조건은 [scoring.py](../../src/opencv_preprocessing_advisor/scoring.py)의 `score_pipeline`에 있습니다.
### 추가 질문
“왜 sharpness만으로 필터를 선택하지 않나요?”

## IQ7: Canny 앞에 blur를 왜 명시적으로 두나요?
### 30초 답변
이 프로젝트에서 `cv2.Canny`는 Gaussian blur를 자동 적용하지 않으므로, 평활화 여부를 호출자가 통제하도록 했습니다.
### 2분 심화 답변
Canny의 gradient·NMS·이중 임계값·hysteresis와 사전 평활화는 다른 단계입니다. 이 경계를 분명히 하면 noise 제거가 필요한 이미지와 원본 에지가 중요한 이미지를 같은 API 설명으로 혼동하지 않습니다.
### 근거 코드·결과
[Day 5](day-05.md)의 Canny 설명과 [학습 계약 테스트](../../tests/test_learning_10day_content.py)가 회귀를 막습니다.
### 추가 질문
“Scharr를 Sobel 대신 검토할 상황은?”

## IQ8: 진단 지표는 왜 여러 개인가요?
### 30초 답변
밝기·대비·엔트로피·노이즈·에지 중 하나만으로 이미지 품질이나 적합성을 정의할 수 없기 때문입니다.
### 2분 심화 답변
예를 들어 엔트로피와 sharpness는 노이즈로도 올라갈 수 있습니다. 그래서 각 지표를 원본 대비 변화로 보되, 추천의 이유와 경고를 사람이 검토할 수 있게 남겼습니다.
### 근거 코드·결과
모든 지표는 [diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py)의 `ImageDiagnostics`로 묶입니다.
### 추가 질문
“가장 신뢰하지 않는 지표 하나와 그 이유는?”

## IQ9: 추천 점수는 어떻게 계산되나요?
### 30초 답변
profile별 가중치로 대비·노이즈·에지·clipping 등의 진단 변화 점수를 합쳐 Top 3 실험 후보를 정합니다.
### 2분 심화 답변
점수 구성과 이유 코드를 노출해 black box처럼 보이지 않게 했습니다. profile weight는 YAML에서 읽고 합이 1인지 검증하지만, 학습된 정확도 모델은 아닙니다.
### 근거 코드·결과
[scoring.py](../../src/opencv_preprocessing_advisor/scoring.py)의 `score_pipeline`, `rank_recommendations`가 계산을 보여 줍니다.
### 추가 질문
“가중치를 사용자에게 열어 둘 때의 위험은?”

## IQ10: 추천 점수가 정확도가 아닌 이유는?
### 30초 답변
추천에는 레이블과 정답 예측이 없고, 단일 이미지의 대리 진단만 쓰므로 classification accuracy가 될 수 없습니다.
### 2분 심화 답변
이 경계를 UI·문서·서비스 구조 모두에 반영했습니다. 실제 성능 주장은 class folder 데이터, stratified fold, Macro F1과 confusion matrix를 쓰는 benchmark 흐름에서만 합니다.
### 근거 코드·결과
[services.py](../../src/opencv_preprocessing_advisor/services.py)의 두 서비스와 [README](../../README.md)의 경고 문구가 근거입니다.
### 추가 질문
“추천 후보를 benchmark의 search space로 쓰려면 무엇을 추가해야 하나요?”

## IQ11: 왜 YAML 파이프라인을 사용했나요?
### 30초 답변
변환 순서·파라미터·프로필·경고를 코드 밖에서 검토하고 같은 후보를 재현하기 위해서입니다.
### 2분 심화 답변
catalog가 YAML을 읽고 transform 이름과 kernel 조건을 검증합니다. 실행 결과에는 중간 단계와 config hash가 남아, ‘CLAHE를 썼다’보다 정확한 실험 문맥을 기록합니다.
### 근거 코드·결과
[pipelines.py](../../src/opencv_preprocessing_advisor/pipelines.py)와 [pipelines.yaml](../../src/opencv_preprocessing_advisor/config/pipelines.yaml)을 봅니다.
### 추가 질문
“YAML 값 검증을 하지 않으면 어떤 장애가 나나요?”

## IQ12: 왜 원본 파이프라인을 baseline으로 남겼나요?
### 30초 답변
전처리의 이득을 주장하려면 아무 처리도 하지 않은 기준과 같은 조건에서 비교해야 하기 때문입니다.
### 2분 심화 답변
실험 결과 Original + RTrees가 가장 높았고, 이는 특정 데이터에서 대비 강화가 고전 특징에 유리하지 않았다는 정보입니다. 원본 baseline이 없으면 이런 결론을 낼 수 없습니다.
### 근거 코드·결과
[benchmark-evidence.json](../portfolio/benchmark-evidence.json)은 Original의 Accuracy 0.804, Macro F1 0.789을 기록합니다.
### 추가 질문
“원본 우세 결과 뒤 어떤 ablation을 하겠습니까?”

## IQ13: benchmark의 정확한 범위는 무엇인가요?
### 30초 답변
MVTec AD `tile/test` 상태 폴더를 여섯 클래스 분류로 해석한 117장 사례입니다.
### 2분 심화 답변
seed 42, stratified 5-fold, combined feature, SVM·kNN·RTrees 및 세 후보 pipeline의 비교입니다. GT mask나 anomaly localization metric은 사용하지 않았으므로 not official MVTec anomaly-detection result입니다.
### 근거 코드·결과
[benchmark-evidence.json](../portfolio/benchmark-evidence.json)의 `dataset_interpretation`과 provenance가 기준입니다.
### 추가 질문
“왜 이 결과를 공개 benchmark 재현이라고 부르면 안 되나요?”

## IQ14: 데이터 누수는 어떻게 막았나요?
### 30초 답변
각 fold에서 표준화기를 train feature에만 fit하고 test feature에는 transform만 적용했습니다.
### 2분 심화 답변
전체 데이터 평균·표준편차를 쓰면 test 분포가 학습 단계에 들어갑니다. 데이터 의존적인 vocabulary나 feature selection을 추가할 때도 동일하게 fold 안에서 학습해야 합니다.
### 근거 코드·결과
`cross_validate`는 [evaluation.py](../../src/opencv_preprocessing_advisor/evaluation.py)에 구현돼 있습니다.
### 추가 질문
“SIFT BoW를 넣으면 누수 방지 절차가 어떻게 달라지나요?”

## IQ15: stratified fold를 택한 이유는?
### 30초 답변
클래스별 표본이 작은 상황에서 각 fold의 클래스 구성을 가능한 비슷하게 유지하기 위해서입니다.
### 2분 심화 답변
클래스별 인덱스를 seed로 섞고 최소 클래스 수보다 많은 split을 요구하면 실제 split 수를 제한합니다. 이는 class balance를 돕지만 중복 이미지나 촬영 배치 누수까지 해결하지는 않습니다.
### 근거 코드·결과
구현은 [datasets.py](../../src/opencv_preprocessing_advisor/datasets.py)의 `stratified_folds`입니다.
### 추가 질문
“시간 순서 데이터라면 어떤 split을 고려하나요?”

## IQ16: Accuracy와 Macro F1을 왜 같이 보고하나요?
### 30초 답변
Accuracy는 전체 비율, Macro F1은 각 클래스를 같은 비중으로 보는 지표라 함께 봐야 합니다.
### 2분 심화 답변
다수 클래스가 많으면 accuracy가 높아도 소수 클래스 실패를 가릴 수 있습니다. per-class precision·recall·F1과 confusion matrix를 함께 남겨 어떤 오류가 발생했는지 확인합니다.
### 근거 코드·결과
[evaluation.py](../../src/opencv_preprocessing_advisor/evaluation.py)의 `classification_metrics`가 두 지표를 계산합니다.
### 추가 질문
“운영 오류 비용이 비대칭이면 어떤 지표를 더 추가하겠습니까?”

## IQ17: confusion matrix는 어떻게 읽나요?
### 30초 답변
이 구현은 행이 실제 class, 열이 예측 class이므로 한 행의 비대각 원소가 해당 실제 클래스의 오분류 방향입니다.
### 2분 심화 답변
대각선만 보고 끝내지 않고 support가 적은 클래스와 특정 쌍의 혼동을 확인합니다. 파이프라인이 바뀌어 성능이 변하면 어떤 혼동이 변했는지도 분석 대상입니다.
### 근거 코드·결과
행·열 규약은 [evaluation.py](../../src/opencv_preprocessing_advisor/evaluation.py)의 `matrix[actual, guess]`입니다.
### 추가 질문
“precision과 recall을 matrix에서 각각 어떻게 읽나요?”

## IQ18: 어떤 특징을 사용했나요?
### 30초 답변
HSV/LAB 색 히스토그램, HOG 형태, Sobel·Laplacian·Gabor 질감 통계를 결합한 고전 특징을 사용했습니다.
### 2분 심화 답변
각 특징은 다른 단서를 보므로 combined profile로 결합했습니다. 이는 특정 데이터의 선택이며, 특징 차원·resize·표준화가 비교 조건에 포함됩니다.
### 근거 코드·결과
[features.py](../../src/opencv_preprocessing_advisor/features.py)의 `CombinedExtractor`가 구성 요소를 연결합니다.
### 추가 질문
“색 히스토그램이 놓치는 정보는 무엇인가요?”

## IQ19: HOG는 어떤 정보를 주나요?
### 30초 답변
HOG는 고정 격자에서 gradient 방향 분포를 요약해 윤곽·형태 단서를 제공합니다.
### 2분 심화 답변
입력을 configured size로 resize하고 block·cell 설정에 맞는 고정 길이 descriptor를 만듭니다. 그러므로 resize 자체가 특징 정의의 일부이며 해상도 조건을 명확히 해야 합니다.
### 근거 코드·결과
`HOGExtractor`는 [features.py](../../src/opencv_preprocessing_advisor/features.py)에 있습니다.
### 추가 질문
“왜 size가 16의 배수여야 하나요?”

## IQ20: SIFT를 구현했는데 benchmark에 넣지 않은 이유는?
### 30초 답변
SIFT BoW는 vocabulary를 fold 안에서 학습해야 공정한데, 현재 benchmark profile에는 그 fold-local 경로가 없기 때문입니다.
### 2분 심화 답변
전체 데이터 descriptor로 vocabulary를 만들면 test fold 정보가 섞입니다. 그래서 구현 존재와 보고된 비교 범위를 구분하고, 확장 과제로 fold-aware extractor를 제시했습니다.
### 근거 코드·결과
`SiftBowExtractor`는 [features.py](../../src/opencv_preprocessing_advisor/features.py), 현재 profile 목록은 [services.py](../../src/opencv_preprocessing_advisor/services.py)에 있습니다.
### 추가 질문
“vocabulary를 cache할 때도 누수가 생길 수 있나요?”

## IQ21: SVM 설정의 선택 근거는?
### 30초 답변
RBF SVM을 고전 특징의 비선형 baseline으로 두고, 다른 분류기와 동일 fold에서 비교했습니다.
### 2분 심화 답변
현재 C=2.0, gamma=0.01은 구현된 baseline 설정이지 전역 최적값 주장입니다. hyperparameter 탐색을 추가한다면 각 train fold 안에서 수행하고 검색 비용도 기록해야 합니다.
### 근거 코드·결과
설정은 [classifiers.py](../../src/opencv_preprocessing_advisor/classifiers.py)의 `OpenCvSvm`에 있습니다.
### 추가 질문
“검증 fold로 C를 고르면 어떤 nested 절차가 필요하나요?”

## IQ22: kNN에서 주의한 점은?
### 30초 답변
kNN은 거리 기반이라 feature scale에 민감하므로 fold-local 표준화와 training rows보다 큰 k 방지가 중요합니다.
### 2분 심화 답변
작은 fold에서 기본 이웃 수가 학습 표본보다 커질 수 있어 구현은 `min(neighbors, training_rows)`를 사용합니다. 그럼에도 작은 데이터의 이웃 관계는 불안정할 수 있습니다.
### 근거 코드·결과
[classifiers.py](../../src/opencv_preprocessing_advisor/classifiers.py)의 `OpenCvKnn`과 [evaluation.py](../../src/opencv_preprocessing_advisor/evaluation.py)를 봅니다.
### 추가 질문
“거리 기반 모델에서 어떤 feature scaling 대안을 쓰겠습니까?”

## IQ23: RTrees를 선택한 이유는?
### 30초 답변
RTrees는 비선형 관계를 다루는 OpenCV-native baseline이며 이 사례에서 원본과 결합해 가장 높은 Macro F1을 보였습니다.
### 2분 심화 답변
트리 깊이, 최소 표본 수, seed를 코드로 고정해 비교했습니다. 우세 결과는 이 데이터·feature·fold 조합의 관찰이며, 다른 산업 도메인으로 일반화하지 않습니다.
### 근거 코드·결과
`OpenCvRTrees` 설정은 [classifiers.py](../../src/opencv_preprocessing_advisor/classifiers.py)에 있습니다.
### 추가 질문
“트리 중요도를 해석에 추가하려면 무엇을 검증해야 하나요?”

## IQ24: 결과 재현성을 어떻게 확보했나요?
### 30초 답변
seed, fold assignment, pipeline config hash, OpenCV version과 보고서 산출물을 함께 기록하도록 했습니다.
### 2분 심화 답변
결과 숫자만 저장하면 어떤 설정으로 생성됐는지 알 수 없습니다. report writer는 CSV·JSON·confusion matrix와 실행 설정을 남겨 비교 가능한 실험 단위를 만듭니다.
### 근거 코드·결과
[reports.py](../../src/opencv_preprocessing_advisor/reports.py)와 [benchmark-evidence.json](../portfolio/benchmark-evidence.json)의 provenance가 근거입니다.
### 추가 질문
“시드가 같아도 달라질 수 있는 환경 요인은?”

## IQ25: 프로젝트 아키텍처를 설명해 보세요.
### 30초 답변
UI와 CLI는 서비스 계층을 호출하고, 서비스는 진단·파이프라인·특징·평가·보고서를 조합하므로 핵심 로직을 테스트 가능하게 분리했습니다.
### 2분 심화 답변
단일 이미지 흐름은 pipeline catalog와 scoring을, benchmark 흐름은 dataset manifest와 evaluation을 사용합니다. 공통 transform과 I/O 계약을 재사용하면서 성능 주장 경계는 서비스 단위로 분리했습니다.
### 근거 코드·결과
[architecture asset](../portfolio/assets/architecture.png), [services.py](../../src/opencv_preprocessing_advisor/services.py)를 함께 제시합니다.
### 추가 질문
“새 UI를 추가할 때 서비스 계약을 어떻게 유지하나요?”

## IQ26: 테스트 전략은 무엇인가요?
### 30초 답변
작은 배열 기반 단위 테스트로 변환·특징·평가 계약을 확인하고, 서비스·CLI·문서 validator까지 연결 테스트합니다.
### 2분 심화 답변
예를 들어 median은 impulse 감소, HOG는 결정성, evaluation은 누수 방지, 학습 문서는 링크와 사실 경계를 검사합니다. 테스트는 모든 실제 데이터의 보증이 아니라 회귀 방지와 설명 가능한 계약입니다.
### 근거 코드·결과
[tests](../../tests)와 특히 [test_evaluation.py](../../tests/test_evaluation.py), [test_learning_10day_content.py](../../tests/test_learning_10day_content.py)가 근거입니다.
### 추가 질문
“시각적 품질 검증을 자동화할 수 없는 부분은?”

## IQ27: 산업 이미지로 옮길 때 가장 먼저 확인할 것은?
### 30초 답변
클래스 정의, 촬영 조건, 오류 비용, 데이터 분할 단위와 원본 baseline을 먼저 확인합니다.
### 2분 심화 답변
새 카메라나 라인에서는 밝기·텍스처 분포와 결함 크기가 달라집니다. 기존 가중치와 파라미터를 복사하지 않고, 권한 있는 표본으로 진단 분포와 leakage 위험을 다시 측정합니다.
### 근거 코드·결과
[limitations.md](../portfolio/limitations.md)와 [Day 10](day-10.md)의 다음 실험 목록이 출발점입니다.
### 추가 질문
“제품 단위 중복을 어떻게 split에서 막겠습니까?”

## IQ28: 왜 MVTec 원본 이미지를 저장소에 넣지 않았나요?
### 30초 답변
데이터셋 배포·권한 범위와 저장소 경량성을 지키고, 공개 가능한 합성 샘플로 재현 가능한 경로를 제공하기 위해서입니다.
### 2분 심화 답변
코드는 로컬에서 권한 있는 class-folder dataset을 읽을 수 있지만, 커밋된 증거는 metrics, config hash, report hash처럼 원본을 노출하지 않는 정보입니다. 문서 예제는 synthetic tile을 사용합니다.
### 근거 코드·결과
[data/samples/synthetic-tile.png](../../data/samples/synthetic-tile.png)와 [benchmark evidence](../portfolio/benchmark-evidence.json)를 확인합니다.
### 추가 질문
“데이터셋 라이선스 검토 항목은 무엇인가요?”

## IQ29: 실패한 전처리 결과를 어떻게 설명하나요?
### 30초 답변
CLAHE+Bilateral과 LAB CLAHE가 이 사례에서 원본보다 낮았다는 사실을 숨기지 않고, 정보 보존과 feature 변화의 가설로 설명하되 확정 인과로 말하지 않습니다.
### 2분 심화 답변
전처리는 대비를 키우는 동시에 색·질감·경계의 상대 관계를 바꿉니다. 다음 단계는 class별 confusion matrix, 특징 분포, 파라미터 ablation으로 가설을 검증하는 것입니다.
### 근거 코드·결과
[benchmark-evidence.json](../portfolio/benchmark-evidence.json)의 순위는 Original 0.789, CLAHE+Bilateral 0.731, LAB CLAHE 0.594 Macro F1입니다.
### 추가 질문
“‘실패’와 ‘범위 내 관찰’을 어떻게 구분해 말하겠습니까?”

## IQ30: 어떤 한계를 가장 중요하게 말하나요?
### 30초 답변
고전 특징과 작은 특정 데이터셋의 분류 사례이므로 운영 성능이나 공식 anomaly-detection 성능을 주장하지 않는다는 한계입니다.
### 2분 심화 답변
GT mask를 쓰지 않았고, SIFT의 fold-local evaluation도 아직 benchmark에 없습니다. 도메인 이동, 클래스 불균형, 오류 비용, 처리 지연을 새 데이터에서 별도로 검증해야 합니다.
### 근거 코드·결과
[limitations.md](../portfolio/limitations.md)와 [README](../../README.md)의 범위 문구를 인용할 수 있습니다.
### 추가 질문
“한계를 말하면서도 프로젝트 가치를 어떻게 설명하나요?”

## IQ31: 우선순위를 어떻게 정했나요?
### 30초 답변
재현 가능한 baseline과 사실 경계를 먼저 만든 뒤, 추천 UX와 확장 특징을 그 위에 올렸습니다.
### 2분 심화 답변
초기에 모델을 늘리는 것보다 입력 계약, YAML pipeline, fold-safe evaluation, 보고서가 있어야 실험을 믿을 수 있습니다. 그래서 optional SIFT보다 현재 profile의 검증과 문서화를 우선했습니다.
### 근거 코드·결과
구조는 [evidence-map.md](../portfolio/evidence-map.md), 구현 우선순위는 [Day 6](day-06.md)에 정리돼 있습니다.
### 추가 질문
“시간이 절반이면 무엇을 제외하겠습니까?”

## IQ32: 다음 실험은 무엇인가요?
### 30초 답변
원본 우세의 원인을 class별 오류·파라미터 ablation·새 도메인 validation으로 좁히고, SIFT는 fold-local vocabulary로 추가하겠습니다.
### 2분 심화 답변
각 실험은 사전에 데이터 분할과 지표를 고정하고, original baseline을 포함합니다. 추천 가중치도 사용자 연구나 레이블 결과가 생기기 전에는 휴리스틱이라는 상태를 유지합니다.
### 근거 코드·결과
[Day 10](day-10.md)의 제한과 다음 실험, [reports.py](../../src/opencv_preprocessing_advisor/reports.py)의 보고서 형식을 재사용합니다.
### 추가 질문
“다음 실험의 성공 기준을 수치와 오류 사례로 어떻게 정의하나요?”

## IQ33: 한 장의 이미지를 평가해 달라는 요청에는 어떻게 답하나요?
### 30초 답변
진단과 후보를 제안할 수 있지만, 레이블 없는 한 장으로 분류 성능을 평가할 수는 없다고 명확히 답합니다.
### 2분 심화 답변
밝기, clipping, 대비, 노이즈, 에지 변화와 경고를 보여 주고 후보를 사람이 검토합니다. 성능 결론이 필요하면 대표성 있는 레이블 데이터와 독립적인 평가 계획을 요청합니다.
### 근거 코드·결과
`ImageAdvisorService.analyze`는 [services.py](../../src/opencv_preprocessing_advisor/services.py)에 있습니다.
### 추가 질문
“사용자가 점수 하나만 원하면 UI를 어떻게 설계하겠습니까?”

## IQ34: 문서의 기술적 진실성을 어떻게 유지했나요?
### 30초 답변
각 핵심 주장을 소스·테스트·설정·benchmark evidence로 연결하고, 문서 계약 테스트로 수치와 링크를 점검했습니다.
### 2분 심화 답변
특히 MVTec 문구는 공식 성능 주장으로 오해되지 않게 validator가 검사합니다. 학습 문서도 API 설명, 링크, 경계 문구를 코드 변경 시 다시 확인할 수 있습니다.
### 근거 코드·결과
[scripts/validate_portfolio.py](../../scripts/validate_portfolio.py)와 [test_learning_10day_content.py](../../tests/test_learning_10day_content.py)가 근거입니다.
### 추가 질문
“문서와 코드가 충돌하면 어느 쪽을 먼저 고치나요?”

## IQ35: 이 프로젝트에서 배운 가장 중요한 교훈은?
### 30초 답변
전처리는 기본값이 아니라 검증해야 할 가설이며, 보기 좋은 결과와 목적 지표의 개선은 다를 수 있다는 점입니다.
### 2분 심화 답변
Original + RTrees의 우세는 실패를 드러낸 것이 아니라 baseline, 분할, 지표, 보고서가 있어야 신뢰할 수 있는 결론을 얻는다는 사례입니다. 저는 다음 도메인에서도 같은 평가 규율을 먼저 적용하겠습니다.
### 근거 코드·결과
[benchmark-evidence.json](../portfolio/benchmark-evidence.json), [Day 9](day-09.md), [Day 10](day-10.md)을 함께 제시합니다.
### 추가 질문
“이 교훈이 비전 외의 ML 시스템에도 어떻게 적용되나요?”
