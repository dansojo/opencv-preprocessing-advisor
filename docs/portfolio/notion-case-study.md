# OpenCV Preprocessing Advisor - Explainable Preprocessing Portfolio

> Notion으로 옮길 원본 문서입니다. 구현과 검증 가능한 산출물을 분리해, 한 장의 이미지에 대한 **설명 가능한 탐색 추천**과 레이블 데이터셋에서의 **재현 가능한 분류 평가**를 혼동하지 않도록 구성했습니다.

**프로젝트 링크:** [GitHub 저장소](https://github.com/dansojo/opencv-preprocessing-advisor)

**PDF 포트폴리오:** [6페이지 PDF](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/output/pdf/opencv-preprocessing-advisor-portfolio.pdf)

**정본 근거:** [사례 연구](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/docs/portfolio/case-study.md), [실험 결과](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/docs/portfolio/experiment-results.md), [한계와 다음 검증](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/docs/portfolio/limitations.md), [증거 맵](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/docs/portfolio/evidence-map.md)

<table_of_contents/>

## 프로젝트 요약

이 프로젝트는 OpenCV 기반 전처리를 "더 보기 좋게 만드는 효과"가 아니라, **왜 이 후보를 먼저 실험해야 하는지 설명할 수 있는 의사결정 과정**으로 만들었다. 단일 이미지에서는 이미지 상태를 진단하고 Top 3 후보를 제시한다. 데이터셋이 있을 때는 같은 후보를 고전 특징과 `cv2.ml` 분류기에서 교차 검증해 성능 영향으로 확인한다.

구현의 흐름은 [서비스 조립](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/services.py), [파이프라인 카탈로그](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/pipelines.py), [명령행 진입점](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/cli.py), 그리고 [Streamlit 이미지 Advisor 화면](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/ui/image_advisor.py)으로 연결된다. 결과와 제약을 한 문서에 숨기지 않고, 원본 근거 문서와 생성 보고서로 역추적할 수 있게 했다.

## 배경과 요구사항 변화

초기 문제는 "어떤 전처리를 적용할까"였지만, 실제 사용에서는 정답 레이블이 없는 입력도 많고 과도한 전처리가 유용한 질감이나 색상 단서를 지울 수 있다. 그래서 요구사항을 다음처럼 발전시켰다.

- 후보를 즉시 하나로 단정하지 않고, 진단 기반 Top 3 탐색 순위를 제공한다.
- 후보 파이프라인과 프로필을 코드가 아닌 [YAML 파이프라인 설정](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/config/pipelines.yaml)과 [점수 가중치 설정](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/config/scoring.yaml)으로 공개한다.
- 레이블 데이터셋에서는 추천 점수 대신 교차 검증 Macro F1과 accuracy를 보고한다.
- 결과는 [보고서 생성기](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/reports.py)가 CSV, JSON, PNG, 실행 설정, OpenCV 버전, seed를 기록하도록 해 재검토할 수 있게 한다.

## 추천과 평가의 분리

Advisor는 한 장의 이미지에서 밝기, 대비, 노이즈, 에지, 색상, 클리핑 변화에 근거해 후보를 정렬한다. 반면 Benchmark는 클래스 폴더와 레이블을 입력으로 받아 전처리·특징·분류기 조합을 같은 fold에서 비교한다. 두 결과가 답하는 질문이 다르므로 같은 숫자로 보이게 만들지 않았다.

<callout icon="⚠️" color="yellow_bg">
	**휴리스틱 점수**
	Advisor의 적합도 점수는 **분류 정확도나 일반화 성능 추정치가 아니다**. 레이블이 없는 입력에서 다음 실험의 우선순위를 투명하게 정하는 휴리스틱이며, 성능 결론은 데이터셋 교차 검증에서만 낸다. 점수 계산과 경고 규칙은 [scoring.py](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/scoring.py)에서 확인할 수 있다.
</callout>

이 분리는 "시각적으로 강한 효과"를 "모델 성능 향상"으로 오해하지 않게 한다. 추천은 실험의 출발점이고, 평가는 그 출발점을 레이블 데이터에서 반증하거나 지지하는 별도 단계다.

## 이미지 진단

[진단 모듈](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/diagnostics.py)은 BGR 입력을 검증한 뒤 밝기, 전역·국소 대비, 엔트로피, Laplacian 기반 선명도, median residual 기반 노이즈 추정, 조명 불균일, Canny 에지 밀도·연속성, 색상성, 채도 분산, 밝고 어두운 클리핑을 계산한다. 입력 검증과 유니코드 안전 이미지 입출력은 [io.py](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/io.py)에 분리되어 있다.

후보를 적용한 뒤에는 전후 진단값을 다시 비교한다. 결과 모델은 [models.py](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/models.py)에 정의되어 있으며, 사용자는 총점만이 아니라 높은 기여를 한 구성 요소와 clipping·과도한 에지·과도한 평활화·색 손실 경고를 함께 본다. 이 구조가 "필터를 썼다"를 재현 가능한 판단 기록으로 바꾼다.

## 전처리 선택

전처리는 [transforms.py](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/transforms.py)에 있는 작은 OpenCV 연산으로 구성한다. normalize, 자동 gamma, LAB L-channel CLAHE, Gaussian, median, bilateral, unsharp, grayscale, blackhat을 조합하되 모든 후보를 정답처럼 취급하지 않는다.

LAB L-channel CLAHE는 BGR 각 채널을 독립적으로 평활화하지 않고 LAB의 밝기 L 채널에만 CLAHE를 적용한다. a/b 색상 채널을 보존하므로 색 자체가 단서인 이미지에서 채널별 BGR 평활화보다 색상 관계를 덜 교란한다. Gaussian은 넓은 고주파 노이즈, median은 임펄스성 노이즈, bilateral은 에지를 보존하고 싶을 때의 가정에 대응한다. 각각의 유효성과 위험은 [변환 단위 테스트](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/tests/test_transforms.py)로 확인한다.

## 특징

레이블 데이터셋에서는 `combined` 특징 프로필을 사용한다. 구성은 HSV/LAB 색상 히스토그램, HOG, Sobel/Laplacian/Gabor 텍스처 통계이며 [features.py](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/features.py)와 [특징 테스트](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/tests/test_features.py)에 구현돼 있다.

`SiftBowExtractor`도 독립 구현되어 있지만 **not exposed as a BenchmarkService feature profile** 이다. 즉, 현재 `BenchmarkService`에서 선택 가능한 특징 프로필로 연결되어 있지 않다. 따라서 현재 리더보드의 수치에 SIFT 결과를 포함하지 않는다. SIFT BoW를 공정하게 추가하려면 각 훈련 fold에만 vocabulary를 적합하는 **future fold-local vocabulary integration**(향후 fold별 어휘 학습 연결)이 필요하다.

## 분류기

분류 단계는 [classifiers.py](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/classifiers.py)의 OpenCV `cv2.ml` SVM, kNN, RTrees를 사용한다. 이 선택은 모델의 폭을 과장하려는 것이 아니라, 특징 추출부터 분류까지 OpenCV 중심의 재현 가능한 기준선을 만들기 위한 것이다. 모델별 입력 검증과 예측 형태는 [분류기 테스트](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/tests/test_classifiers.py)로 고정한다.

리더보드에서 분류기는 사전 선호로 고르지 않는다. 동일한 전처리·특징·fold 조건에서 Macro F1, accuracy, 처리 시간 순으로 정렬해 비교한다. 따라서 "RTrees가 항상 우수하다"가 아니라, 아래의 고정된 MVTec tile 사례에서 해당 조합이 가장 높았다는 제한된 관찰만 보고한다.

## 누수 방지와 재현성

[datasets.py](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/datasets.py)는 클래스 폴더를 결정적으로 탐색하고, seed 42로 stratified fold를 만든다. [evaluation.py](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/evaluation.py)는 각 fold의 훈련 특징에만 표준화기를 적합하고 테스트 특징에는 변환만 적용하는 **fold-local scaling**을 수행한다. 이는 fold마다 훈련 데이터만으로 스케일 기준을 정한다는 뜻이다. 이 경계가 없으면 테스트 분포가 훈련 통계에 섞여 성능이 낙관적으로 보이는 누수가 생긴다.

평가 코드와 [평가 테스트](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/tests/test_evaluation.py)는 같은 split에서 조합을 비교하고, 생성 보고서는 순위표·fold 지표·클래스 지표·혼동행렬·시간·실행 설정을 남긴다. 이 경로와 [benchmark-evidence.json](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/docs/portfolio/benchmark-evidence.json)의 hash 요약은 소스 이미지나 로컬 절대 경로를 커밋하지 않고도 재생성 근거를 제공한다.

## MVTec 실험

이 실험은 MVTec AD `tile/test`의 상태 폴더 `crack`, `glue_strip`, `good`, `gray_stroke`, `oil`, `rough`을 **6개 클래스**로 해석한 고전 분류 사례다. 총 **117** images를 seed 42의 stratified 5-fold 교차 검증으로 평가했고, SVM·kNN·RTrees와 `combined` 특징을 비교했다.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>Pipeline</td>
		<td>Classifier</td>
		<td>Accuracy</td>
		<td>Macro F1</td>
	</tr>
	<tr>
		<td>Original</td>
		<td>RTrees</td>
		<td>**0.804**</td>
		<td>**0.789**</td>
	</tr>
	<tr>
		<td>CLAHE + Bilateral</td>
		<td>RTrees</td>
		<td>0.766</td>
		<td>0.731</td>
	</tr>
	<tr>
		<td>LAB CLAHE</td>
		<td>RTrees</td>
		<td>0.664</td>
		<td>0.594</td>
	</tr>
</table>

<callout icon="⚠️" color="yellow_bg">
	**MVTec 공식 지표**
	이 결과는 `tile/test` 폴더 이름을 클래스 레이블로 사용한 분류 실험이며, **not an official MVTec anomaly-detection metric** 이다. GT mask, anomaly localization, AUROC, pixel-level AUROC, PRO를 사용하거나 주장하지 않는다. 정확한 프로토콜과 표시는 [실험 결과](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/docs/portfolio/experiment-results.md)와 [재생성 증거](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/docs/portfolio/benchmark-evidence.json)에 남겼다.
</callout>

## 실패 해석

<callout icon="💡" color="green_bg">
	**원본 파이프라인의 승리**
	Original + RTrees의 Accuracy **0.804**, Macro F1 **0.789**는 "전처리를 하지 못했다"가 아니라, 이 데이터와 고정 특징 조합에서는 원본 질감과 클래스 구분 정보가 이미 충분했음을 보여 주는 유용한 엔지니어링 결론이다.
</callout>

CLAHE + Bilateral의 Macro F1은 0.731, LAB CLAHE는 0.594로 원본보다 낮았다. 가능한 가설은 국소 대비 강화가 클래스 구분에 쓰이던 자연 질감을 바꾸거나, bilateral 평활화가 약한 표면 변화를 줄였다는 것이다. 그러나 이는 결과를 설명하기 위한 가설이지 인과관계나 다른 데이터셋에 대한 일반화 결론이 아니다.

혼동행렬은 [보고서 모듈](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/reports.py)에서 저장한다. accuracy와 Macro F1을 함께 읽어 전체 정답 비율과 클래스 간 균형을 분리해 확인하며, 한 클래스의 성능만으로 전체를 낙관적으로 해석하지 않는다.

## 트러블슈팅

<table fit-page-width="true" header-row="true">
	<tr>
		<td>증상</td>
		<td>먼저 확인할 근거</td>
		<td>대응</td>
	</tr>
	<tr>
		<td>추천 점수가 높지만 과도한 효과가 보임</td>
		<td>score warnings와 전후 진단</td>
		<td>clipping, excessive edges, oversmoothing, color loss 경고를 보고 다른 후보와 비교한다.</td>
	</tr>
	<tr>
		<td>색상이 어색하게 바뀜</td>
		<td>LAB L-channel 선택과 변환 결과</td>
		<td>BGR 채널별 평활화 대신 L 채널에만 CLAHE를 적용한 후보를 비교한다.</td>
	</tr>
	<tr>
		<td>같은 데이터에서 성능이 예상보다 높음</td>
		<td>fold-local scaling과 seed</td>
		<td>훈련 fold만으로 표준화했는지, seed 42와 동일 fold인지 확인한다.</td>
	</tr>
	<tr>
		<td>SIFT 수치가 리더보드에 없음</td>
		<td>Feature profile 노출 범위</td>
		<td>현재 `combined`만 보고하며, fold-local vocabulary 없이는 SIFT BoW를 비교에 넣지 않는다.</td>
	</tr>
	<tr>
		<td>보고서를 다시 만들 수 없음</td>
		<td>CLI와 산출물 경로</td>
		<td>[CLI 테스트](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/tests/test_cli.py)와 [reports 테스트](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/tests/test_reports.py)를 따라 `opencv-prep benchmark` 실행 조건과 출력물을 확인한다.</td>
	</tr>
</table>

문제 해결의 핵심은 단일 점수나 한 장의 결과 이미지에 의존하지 않는 것이다. 입력 검증, 진단 변화, pipeline 설정, fold 계획, 생성 보고서를 순서대로 확인하면 추천 문제와 평가 문제를 분리한 채 원인을 좁힐 수 있다.

## 한계와 다음 실험

현재 범위는 모든 OpenCV API, 모든 분류기, 깊은 특징, 모든 산업 이미지의 최적화를 주장하지 않는다. Advisor는 휴리스틱이고, MVTec 사례는 117 images·6개 클래스·seed 42·고정 특징·선택한 세 파이프라인의 제한된 관찰이다. GT mask와 공식 anomaly-detection 프로토콜도 범위 밖이다.

다음 실험에서는 (1) 다른 class/촬영 조건/seed에서 동일 보고서를 생성해 fold별 변동을 비교하고, (2) feature profile 및 전처리 parameter ablation을 수행하며, (3) 각 훈련 fold에서만 학습한 SIFT vocabulary를 통합하고, (4) GT mask와 공식 정의에 맞는 별도 anomaly-detection 평가를 구현한다. 각 단계는 지금의 [한계 문서](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/docs/portfolio/limitations.md)와 [증거 맵](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/docs/portfolio/evidence-map.md)를 갱신해, 새 주장보다 먼저 검증 경로를 남겨야 한다.
