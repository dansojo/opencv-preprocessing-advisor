# Day 10 - 프로젝트 전체 설명과 실전 대응

오늘은 코드를 외운 답변이 아니라 문제–설계–검증–한계를 하나의 이야기로 연결한다. 이 프로젝트의 핵심은 ‘전처리로 이미지를 예쁘게 만들기’가 아니라, 단일 이미지에는 설명 가능한 후보를 제안하고 레이블 데이터에는 같은 후보를 공정하게 비교하는 것이다. 마지막 결과가 Original + RTrees였다는 사실도 숨기지 않고 증거로 설명한다.

## 오늘 답해야 할 핵심 질문

- 이 프로젝트가 해결하는 문제와 일부러 해결하지 않는 문제는 무엇인가?
- 단일 이미지 Advisor와 Dataset Benchmark는 입력·출력·성공 기준이 어떻게 다른가?
- Streamlit/CLI, service layer, 진단, pipeline, feature, classifier, evaluation, report는 어떤 순서로 연결되는가?
- 117개·6개 class·stratified 5-fold·seed 42의 수치를 어떤 범위에서 말해야 하는가?
- Original + RTrees가 0.804/0.789로 가장 높았을 때 무엇을 결론내리고, 무엇을 결론내리면 안 되는가?

## 개념과 원리

프로젝트는 두 개의 의사결정 경로를 명시적으로 분리한다. 첫째, 이미지 한 장과 task profile을 받는 Advisor는 brightness, local contrast, entropy, sharpness, noise estimate, edge continuity, colorfulness, clipping 같은 진단을 계산한다. catalog의 전처리 후보를 실행하고 전후 heuristic component를 profile weight로 합쳐 Top 3를 제안한다. 이 출력은 후보 이름, suitability score, 사람이 읽을 reason, warning, intermediate image와 config hash다. 레이블이 없으므로 이 경로는 “무엇부터 확인할까?”에 답한다.

둘째, class folder와 레이블을 받는 Benchmark는 pipeline→feature→classifier 조합을 같은 fold 계획으로 반복 평가한다. 각 fold에서 feature scaling은 train fold에만 fit하고 test fold에는 transform만 한다. Macro F1, accuracy, class별 metric, confusion matrix, 시간과 metadata를 보고 leaderboard를 만든다. 이 경로는 “이 dataset 조건에서 어느 조합이 더 잘 분류되는가?”에 답한다. 둘은 같은 transform/catalog를 공유하지만 score의 의미가 다르다. heuristic recommendation을 정확도처럼 말하거나 benchmark metric을 단일 이미지 점수처럼 말하면 설계 경계가 무너진다.

표현 계층도 분리되어 있다. Streamlit UI와 CLI는 사용자 입력을 받고 service를 호출한다. ImageAdvisorService는 pipeline catalog, diagnostics, scoring을 조합한다. BenchmarkService는 manifest를 만들고 각 pipeline 이미지에서 color/HOG/texture/combined feature를 얻어 SVM, kNN, RTrees를 cross-validation으로 비교한다. reports는 CSV, JSON, PNG를 남긴다. UI가 algorithm을 직접 구현하지 않으므로 같은 service를 test할 수 있고, 입력 화면이 없어도 CLI·test에서 evidence를 재생성할 수 있다.

공개된 예시 결과는 로컬 MVTec AD `tile/test`의 status folder를 `crack`, `glue_strip`, `good`, `gray_stroke`, `oil`, `rough`의 6개 class folder로 해석한 117 image 분류 사례다. feature는 combined(HOG + HSV/LAB histogram + Sobel/Laplacian/Gabor texture statistics)이고 SVM/kNN/RTrees를 seed 42의 stratified 5-fold로 비교했다. 원본 이미지, GT mask, anomaly localization, anomaly-detection protocol은 이 실행에 사용하지 않았다. 따라서 이 수치는 **not official MVTec anomaly-detection metric**이며, official performance를 재현했다는 주장이 아니다.

그 정확한 범위에서 Original + RTrees의 평균 Accuracy는 0.804, Macro F1은 0.789였다. CLAHE + Bilateral + RTrees는 0.766/0.731, LAB CLAHE + RTrees는 0.664/0.594였다. ‘원본이 이겼다’는 관찰은 전처리가 자동으로 class 분리를 개선하지 않으며 대비·평활화가 유용한 texture를 바꿀 수 있다는 engineering conclusion이다. 이것은 실패한 프로젝트의 변명이 아니다. 같은 feature/fold 조건에서 반증 가능한 비교를 했고, 결과가 기대와 다를 때도 pipeline 선택의 근거를 남긴 것이다. 다른 데이터셋·다른 split·다른 downstream 목적에 일반화할 수는 없다.

## OpenCV API와 파라미터

[services.py](../../src/opencv_preprocessing_advisor/services.py)는 두 service 경로와 BenchmarkConfig를 정의한다. [diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py), [pipelines.py](../../src/opencv_preprocessing_advisor/pipelines.py), [scoring.py](../../src/opencv_preprocessing_advisor/scoring.py)가 단일 이미지 흐름을 담당한다. [features.py](../../src/opencv_preprocessing_advisor/features.py), [classifiers.py](../../src/opencv_preprocessing_advisor/classifiers.py), [evaluation.py](../../src/opencv_preprocessing_advisor/evaluation.py), [reports.py](../../src/opencv_preprocessing_advisor/reports.py)가 benchmark 흐름을 담당한다. [test_services.py](../../tests/test_services.py), [test_evaluation.py](../../tests/test_evaluation.py), [test_reports.py](../../tests/test_reports.py)는 그 경계를 검증한다.

| 계층 | 주요 API/설정 | 입력 → 출력 | 발표에서 말할 경계 |
| --- | --- | --- | --- |
| 진단 | `analyze_image` | BGR image → 관찰량 | 진단값은 label 기반 성능이 아니다. |
| Advisor | `ImageAdvisorService.analyze` | image/profile → Top 3 heuristic | score는 후보 우선순위다. |
| pipeline | `PipelineCatalog`, YAML steps | BGR image → intermediate/final | 순서와 params가 실험 조건이다. |
| 특징 | histogram/HOG/Gabor | processed images → `float32` matrix | SIFT는 존재하지만 current benchmark profile이 아니다. |
| 분류·평가 | `cv2.ml`, cross-validation | train/test folds → Macro F1/matrix | scaling은 훈련 fold에서만 fit한다. |
| 보고서 | `BenchmarkReportWriter` | result → CSV/JSON/PNG | hash/version/checksum이 재현 조건이다. |

CLI는 로컬 image 점검과 class-folder benchmark를 실행하는 사용자 진입점이다. Streamlit은 같은 service 결과를 표시한다. [cli.py](../../src/opencv_preprocessing_advisor/cli.py)와 [app.py](../../app.py)는 presentation layer이고, 핵심 주장은 UI screenshot이 아니라 service·test·report evidence로 뒷받침한다. [services.py (main)](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/services.py), [evaluation.py (main)](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/evaluation.py), [test_services.py](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/tests/test_services.py)로 구현을 추적할 수 있다.

## 언제 사용하고 피하는가

Advisor는 라벨 없는 새 이미지에서 조명·noise·clipping·edge 문제를 빠르게 살피고 어떤 전처리 후보부터 시각적으로 검토할지 정할 때 쓴다. Benchmark는 class folder와 충분한 label이 있고 pipeline/feature/classifier 결정을 재현 가능한 metric으로 비교할 때 쓴다. 둘을 함께 쓰면 “왜 이 후보를 시험했는가”와 “시험 후 label 기준으로 무엇이 나았는가”가 연결된다.

피할 일은 Advisor Top 1을 배포 모델로 간주하거나, 117-image status-folder classification을 anomaly localization 성능으로 소개하는 일이다. MVTec raw image나 GT mask를 문서·Notion·repository에 복제하지 않는다. SIFT vocabulary를 전체 데이터로 fit하고 CV라고 주장하지 않으며, test fold까지 scaling을 fit하지 않는다. 원본 pipeline이 leaderboard에서 이겼다는 이유만으로 전처리를 삭제할 필요도 없다. Advisor의 설명 가능성, 다른 dataset의 조건, 별도 task는 다시 검증해야 한다.

## 프로젝트 코드 연결

- [Advisor와 benchmark service: services.py](../../src/opencv_preprocessing_advisor/services.py)는 단일 이미지 heuristic과 label 평가를 분리한다.
- [진단: diagnostics.py](../../src/opencv_preprocessing_advisor/diagnostics.py)와 [점수: scoring.py](../../src/opencv_preprocessing_advisor/scoring.py)는 Top 3 이유·경고를 만든다.
- [특징: features.py](../../src/opencv_preprocessing_advisor/features.py)와 [분류기: classifiers.py](../../src/opencv_preprocessing_advisor/classifiers.py)는 `float32` feature·OpenCV model contract를 제공한다.
- [평가: evaluation.py](../../src/opencv_preprocessing_advisor/evaluation.py)와 [보고서: reports.py](../../src/opencv_preprocessing_advisor/reports.py)는 fold-local scaling·metric·artifact를 남긴다.
- [서비스 테스트: test_services.py](../../tests/test_services.py), [평가 테스트: test_evaluation.py](../../tests/test_evaluation.py), [보고서 테스트: test_reports.py](../../tests/test_reports.py)는 해당 경로가 UI 밖에서도 검증됨을 보인다.

사례 수치의 1차 근거는 [benchmark-evidence.json](../portfolio/benchmark-evidence.json), 설명 표와 해석은 [experiment-results.md](../portfolio/experiment-results.md), 적용 범위는 [limitations.md](../portfolio/limitations.md), 코드와 주장 연결은 [evidence-map.md](../portfolio/evidence-map.md)다. 발표에서 숫자를 말할 때 이 네 문서를 열어 범위와 rounding을 함께 보여 준다.

## 직접 실험

private MVTec data 없이 끝까지 실행할 수 있는 smoke test는 합성 타일로 Advisor와 CLI self-check를 돌리는 것이다. 아래 코드는 첫 추천의 score가 accuracy가 아니라는 메시지와 config hash를 출력한다.

```python
from pathlib import Path

from opencv_preprocessing_advisor.io import decode_image
from opencv_preprocessing_advisor.models import TaskProfile
from opencv_preprocessing_advisor.services import ImageAdvisorService

image = decode_image(Path("data/samples/synthetic-tile.png"))
result = ImageAdvisorService().analyze(image, TaskProfile.TEXTURE)
print("diagnostic profile:", result.profile.value)
print("candidate count:", len(result.recommendations))
print("pipeline hash:", result.pipeline_config_hash[:12])
for item in result.recommendations:
    print(item.pipeline_id, round(item.suitability_score, 2), item.warning_codes)
print("These are heuristic priorities, not classification accuracy.")
```

그 다음 PowerShell에서 `python -m opencv_preprocessing_advisor.cli self-check --output outputs/self-check`를 실행해 공개 sample의 전 경로가 동작하는지 확인한다. 자신의 class-folder data가 있다면 manifest 규칙(최소 두 class, class당 유효 image 다섯 개 이상)을 충족하는 별도 로컬 경로로 benchmark를 실행하고 report directory를 생성한다. 발표 자료에는 절대 경로나 원본 image를 붙이지 말고, 상대 sample 정보·hash·aggregate metric만 사용한다.

## 예상 결과와 해석

| 관찰 | 예상 결과 | 해석과 다음 질문 |
| --- | --- | --- |
| Advisor smoke test | Top 3, reason/warning, config hash | 후보 우선순위가 출력된 것이지 accuracy가 출력된 것이 아니다. |
| self-check | synthetic sample 기반 artifact/검사 | IO·service 경로가 실행된다는 기본 증거다. |
| benchmark evidence | Original + RTrees 0.804/0.789 | 117 image, 6 class, stratified 5-fold, seed 42 범위로만 말한다. |
| confusion matrix | class pair별 aggregate error | 전처리·feature·label·sample 가설을 다음 실험으로 만든다. |
| Original winner | 강화 파이프라인보다 높은 Macro F1 | 원본 정보 보존이 유리했을 가능성; 다른 data에 일반화하지 않는다. |

발표 중 “왜 CLAHE가 졌나요?”라는 질문에는 인과를 단정하지 않는다. 국소 대비 강화나 bilateral smoothing이 class 구분에 유용한 미세 texture를 바꿨을 수 있다는 hypothesis라고 답하고, pipeline별 ablation, parameter grid, 다른 seed, 새로운 held-out data로 확인하겠다고 말한다. score가 낮은 것이 코드 오류와 같은 뜻도 아니다. error log, report hash, config diff를 보고 버그와 실험 결과를 구분한다.

## 자주 하는 실수와 디버깅

1. **Top 3를 accuracy라고 말함**: heuristic에는 label이 없다. “추천 우선순위”와 “CV metric”을 같은 슬라이드에서도 별도 열로 쓴다.
2. **MVTec 사례의 범위를 생략**: 117 images, 6 status-folder classes, stratified 5-fold, seed 42, not official 범위를 문장으로 포함한다.
3. **원본 승리를 실패로 포장**: expected improvement가 없었다는 것 자체가 비교 실험의 결과다. 왜라는 가설과 다음 검증을 말한다.
4. **SIFT/standardization leakage**: vocabulary와 scaler를 전체 data에서 fit하지 않는다. training fold에서만 추정한다.
5. **UI screenshot만 증거로 사용**: test, config, report hash, evidence JSON, source link를 함께 제시한다.

실행이 안 되면 `self-check`로 sample IO와 service를 먼저 분리한다. Advisor 문제가 의심되면 BGR input, profile, YAML hash, diagnostics를 확인한다. benchmark 문제가 의심되면 manifest의 class count/sample count, feature `float32`, labels `int32`, fold indices, train-only scaler를 확인한다. metric이 바뀌면 OpenCV version, config hash, seed, sample checksum, requested/actual folds가 같은지 차례로 비교한다.

## 본인 말로 설명하기

### 1분 설명

“이 프로젝트는 전처리를 무조건 적용하는 도구가 아니라 두 경로를 분리한 검증 시스템입니다. 단일 이미지 Advisor는 진단값으로 Top 3 heuristic 후보와 clipping·oversmoothing 같은 경고를 제시하고, 점수는 accuracy가 아니라 실험 우선순위입니다. 레이블 class-folder Benchmark는 같은 pipeline을 HOG, HSV/LAB histogram, texture feature와 SVM/kNN/RTrees에 연결해 train-fold-only scaling과 stratified CV로 Macro F1을 비교합니다. 공개 사례에서는 117개 6-class status-folder 분류에서 Original+RTrees가 Accuracy 0.804, Macro F1 0.789였고, 이는 not official anomaly-detection metric입니다. 원본이 이긴 것은 전처리가 항상 유리하지 않다는 evidence입니다.”

### 깊이 설명

“UI와 CLI는 service layer를 호출하고, Advisor는 diagnostics→catalog pipeline→heuristic scoring으로, Benchmark는 manifest→pipeline→features→classifier→fold-local evaluation→report로 흐릅니다. Advisor의 component는 local contrast, entropy, noise, sharpness, edge, clipping 같은 label-free 변화라서 candidate ranking만 만들고, Benchmark의 Macro F1/confusion matrix는 labels와 held-out folds에서 downstream result를 측정합니다. MVTec 예시는 tile/test status folders를 여섯 class로 해석한 117 image run이며 GT mask·localization을 쓰지 않은 not official 사례입니다. 0.804/0.789는 config hash와 report evidence가 있는 조건부 관찰입니다. 다음에는 ablation, seeds, new data, fold-local SIFT, group/time split, GT 기반 별도 평가를 설계해 가설을 검증합니다.”

## 5분 발표 스크립트

“안녕하세요. 이 프로젝트는 OpenCV 전처리를 추천하고, 그 추천이 실제 분류에 도움이 되는지를 분리해서 검증하는 도구입니다. 문제는 이미지가 어둡거나 noisy할 때 필터를 하나 고르는 것처럼 보이지만, 화면이 좋아 보이는 것과 모델이 더 잘 맞히는 것은 다를 수 있다는 점입니다.

먼저 단일 이미지 Advisor입니다. 입력 이미지를 밝기, 국소 대비, entropy, sharpness, noise, edge continuity, clipping으로 진단합니다. YAML에 정의한 전처리 pipeline을 실행하고 전후 변화에 profile별 heuristic weight를 적용해 Top 3를 냅니다. 출력에는 점수뿐 아니라 이유와 clipping, excessive edges, oversmoothing, color loss 경고가 있습니다. 여기서 점수는 accuracy가 아니라 사람이 다음에 확인할 실험 우선순위입니다. label이 없기 때문입니다.

둘째는 label이 있는 Dataset Benchmark입니다. class-folder images를 같은 pipeline으로 처리한 뒤 HSV/LAB histogram, HOG, Gabor texture statistics를 만든 뒤 SVM, kNN, RTrees를 같은 stratified fold에서 비교합니다. scaler는 각 train fold에서만 fit해서 test information leakage를 막습니다. 결과는 accuracy만이 아니라 Macro F1, per-class metric, confusion matrix와 config hash를 report로 남깁니다.

공개 사례는 MVTec AD tile/test의 status folder를 6개 class, 117 images로 해석한 분류 실행입니다. seed 42의 stratified 5-fold에서 Original + RTrees가 Accuracy 0.804, Macro F1 0.789로 세 pipeline 후보 중 높았습니다. 이 숫자는 not official MVTec anomaly-detection metric이고 GT mask나 localization을 사용하지 않았습니다.

중요한 결론은 CLAHE나 bilateral이 항상 낫지 않다는 것입니다. 원본이 이겼다는 것은 실패가 아니라, 이 조건에서는 강화가 유용한 class texture를 바꿨을 수 있다는 검증 가능한 관찰입니다. 다음에는 ablation, 다른 seeds, 새 데이터, fold-local SIFT vocabulary, 그리고 목적에 맞는 GT 평가를 수행하겠습니다. 감사합니다.”

## 15분 기술 발표 구성

1. **0:00–1:30 문제와 경계**: 전처리 추천과 label 기반 성능 평가를 왜 분리하는지, visual improvement가 classifier performance를 보장하지 않음을 제시한다.
2. **1:30–4:00 Advisor 흐름**: BGR 입력→diagnostics→YAML pipeline→heuristic score→Top 3/reason/warning을 diagram과 synthetic sample로 보여 준다.
3. **4:00–5:30 YAML과 안전장치**: step order, config hash, clipping/oversmoothing warning을 설명하고 한 후보의 intermediate를 읽는다.
4. **5:30–8:00 Benchmark 흐름**: class folder→manifest→features→SVM/kNN/RTrees→stratified fold→reports 순서를 설명한다.
5. **8:00–10:00 공정성**: fold-local scaling, test leakage, SIFT vocabulary의 fold-local requirement, Macro F1/confusion matrix 축을 설명한다.
6. **10:00–12:00 evidence 사례**: 117 images/6 classes/seed 42/stratified 5-fold와 Original+RTrees 0.804/0.789를 표로 보여 주고 not official 범위를 말한다.
7. **12:00–13:30 결과 해석**: Original winner를 engineering conclusion으로 읽고 CLAHE/Bilateral이 질감을 바꿨다는 hypothesis와 반증 실험을 제시한다.
8. **13:30–15:00 한계와 다음 실험**: 다른 dataset, seed, ablation, group/time split, fold-local SIFT, GT 기반 별도 목적 평가를 제안하고 질문을 받는다.

## 한계와 다음 실험

이 프로젝트는 모든 OpenCV API 목록도, 모든 domain의 최적 전처리기도, official anomaly-detection benchmark 재현도 아니다. 가장 중요한 한계는 작고 특정한 status-folder classification 사례, 고정 feature/profile/seed, class imbalance 가능성, random stratification의 deployment gap, heuristic weight의 domain 의존성이다. 다음 실험은 (1) pipeline step과 parameter별 ablation, (2) 여러 seed와 별도 held-out/group/time split, (3) current profile과 분리한 fold-local SIFT vocabulary implementation, (4) class별 error image inspection과 label audit, (5) 목적이 anomaly localization이면 GT와 그 목적의 protocol을 가진 별도 evaluation이다. 각 실험은 pre-registered config, train-only transforms, report hash, 실패 결과 공개를 포함해야 한다.

## 완료 기준

- [ ] **이해**: Advisor heuristic과 Benchmark metric의 입력·출력·성공 기준을 구분해 설명했다.
- [ ] **구현**: 합성 타일 Advisor smoke test와 CLI self-check를 실행하고 config hash를 기록했다.
- [ ] **해석**: Original + RTrees 0.804/0.789를 정확한 117-image/6-class/seed 42 범위와 not official disclaimer로 말했다.
- [ ] **설명**: 5분 스크립트를 녹음하거나 발표하고, 15분 구성으로 한계·다음 실험과 최소 열 개의 후속 질문에 답했다.

## 예상 후속 질문 12개

1. **왜 heuristic Top 3를 accuracy로 사용하지 않았나요?** label·held-out test가 없으므로 score는 진단 기반 후보 우선순위일 뿐입니다.
2. **왜 YAML로 pipeline을 정의했나요?** step 순서·parameter·profile을 code diff 없이 검토·재현하기 위해서입니다.
3. **왜 CLAHE + Bilateral이 원본보다 낮았나요?** 유용한 texture 변화가 있었을 수 있다는 hypothesis이며 ablation과 새 data로 검증해야 합니다.
4. **왜 Macro F1을 봤나요?** 다수 class accuracy가 작은 class의 recall 실패를 가리지 않게 하기 위해서입니다.
5. **fold-local scaling이 정확히 무엇을 막나요?** test fold의 mean/std가 training representation에 들어가는 leakage를 막습니다.
6. **SIFT가 있는데 왜 benchmark profile에 없나요?** 구현 존재와 공개된 평가 경로는 다르며 fair comparison에는 fold-local vocabulary가 필요합니다.
7. **RTrees가 feature importance를 주면 원인을 알 수 있나요?** 상관된 feature와 sample/split 영향을 받으므로 탐색 단서이지 인과 증명은 아닙니다.
8. **왜 MVTec 수치를 anomaly result라고 부르지 않나요?** status folder 6-class classification이고 GT mask, localization, anomaly protocol을 사용하지 않았기 때문입니다.
9. **다른 seed에서 결과가 달라지면 무엇을 하나요?** seed별 fold metric 분포를 보고 점수 하나가 아니라 안정성과 error pattern을 비교합니다.
10. **배포 data가 시간/lot별로 다르면 random fold로 충분한가요?** 아닙니다. deployment 조건을 반영한 group/time split을 별도로 설계합니다.
11. **전처리를 배포에서 완전히 빼야 하나요?** 이 사례의 winner만으로 결정하지 않으며 target data·task에서 Advisor 후보와 benchmark를 다시 검증합니다.
12. **재현하려면 무엇이 필요한가요?** code version, OpenCV version, YAML hash, sample checksum, seed, fold assignment, feature/classifier config와 report artifact가 필요합니다.
