# Day 9 - 평가와 재현성

오늘은 “정확도 0.xx”라는 한 줄보다 중요한 평가 설계를 다룬다. class folder를 발견하는 방법, stratified fold, fold-local scaling, Macro F1, confusion matrix, 보고서 metadata가 서로 이어져야 어떤 pipeline이 이겼다는 말을 재검증할 수 있다.

## 오늘 답해야 할 핵심 질문

- train/test split과 stratified K-fold는 각각 어떤 질문에 답하는가?
- class imbalance에서 accuracy만 보면 왜 위험한가?
- fold-local scaling과 전체 데이터 scaling의 차이는 무엇이며 leakage는 어떻게 생기는가?
- Macro F1과 confusion matrix는 어떤 오류를 보여 주는가?
- seed, config hash, sample checksum, OpenCV version을 남기면 무엇을 다시 만들 수 있는가?

## 개념과 원리

평가는 학습에 쓰지 않은 관측치에서 모델이 얼마나 맞는지 추정하는 절차다. 단일 train/test split은 빠르지만 어느 sample이 test에 들어갔는지에 따라 결과가 크게 흔들릴 수 있다. K-fold cross-validation은 데이터를 K개 test fold로 번갈아 쓰고 나머지를 train fold로 사용해 여러 관측을 얻는다. 이 프로젝트의 `stratified_folds()`는 class별 index를 seed로 섞고 각 class를 가능한 균등한 chunk로 나눠, 각 test fold에 모든 class가 섞이게 한다. class별 sample 수가 작은 경우 requested 5보다 실제 split 수가 줄 수 있으므로 report에서 `actual_folds`를 확인한다.

stratified는 모든 class 비율을 완벽히 동일하게 만드는 마법이 아니다. 특히 한 class가 5개뿐이면 각 fold에 하나씩만 들어가며 variance가 크다. 최소 class count가 2보다 작으면 K-fold 자체가 성립하지 않는다. 따라서 class folder 이름, 각 class sample 수, skipped file, fold assignment를 결과와 함께 남긴다. test fold를 pipeline 선택·hyperparameter 반복 조정에 계속 보면 결국 그 test 정보에 맞춰지므로, 최종 의사결정 절차도 명시해야 한다.

가장 흔한 leakage는 split 이전에 전체 feature matrix로 mean/std를 fit하는 것이다. test fold의 평균·분산은 미래 관측의 정보다. 코드의 `cross_validate()`는 각 fold에서 `Standardizer().fit(matrix[fold.train_indices])`를 호출하고 그 scaler로 train/test를 각각 transform한다. 이것이 fold-local scaling이다. transform, PCA, imputer, feature selection, SIFT vocabulary 같은 데이터로부터 추정되는 모든 단계도 같은 원칙을 따른다. preprocessor가 label을 보지 않아도 test distribution을 보고 parameter를 정하면 공정한 평가 경계가 흐려진다.

accuracy는 전체 정답 수/전체 sample 수다. class가 불균형하면 다수 class만 맞혀도 높게 보인다. class c의 precision은 ‘c로 예측한 것 중 실제 c’, recall은 ‘실제 c 중 c로 찾은 것’, F1은 precision과 recall의 조화평균이다. Macro F1은 class별 F1을 같은 비중으로 평균내므로 작은 class의 실패를 숨기기 어렵다. 이 프로젝트의 `classification_metrics()`는 없는 predicted positive나 actual positive가 있을 때 해당 비율을 0으로 처리해 0 나누기를 피한다. 숫자가 정의됐다고 그 의미가 자동으로 좋아지는 것은 아니다.

confusion matrix의 행은 actual, 열은 predicted다. 대각선은 정답, 행의 다른 열은 놓친 실제 class, 열의 다른 행은 과잉 예측된 class를 뜻한다. 예를 들어 `crack` 행이 `gray_stroke` 열에 많이 쌓이면 crack이 gray_stroke로 오분류된다는 가설을 세울 수 있다. 그 이유가 전처리 손실인지, feature 표현인지, label 품질인지, sample 부족인지는 이미지만 보고 추가로 조사해야 한다. matrix는 원인을 증명하지 않는 오류 지도다.

재현성은 seed 하나가 아니다. 이 프로젝트의 report metadata에는 OpenCV version, pipeline config hash, seed, requested/actual folds, pipeline·feature·classifier names, sample의 상대 경로/checksum, fold assignments, skipped files가 저장된다. CSV에는 leaderboard, fold metrics, class metrics, timings가, PNG에는 aggregate confusion matrix가 생성된다. 이 기록은 같은 실행을 따라 하고 설정이 달라졌을 때 어디가 달라졌는지 찾게 한다. private dataset의 절대 경로와 원본을 공개 문서에 넣는 것과는 다르다.

## OpenCV API와 파라미터

[datasets.py](../../src/opencv_preprocessing_advisor/datasets.py)는 class-folder manifest와 deterministic stratified folds를 만든다. [evaluation.py](../../src/opencv_preprocessing_advisor/evaluation.py)는 fold-local scaling, `classification_metrics`, `confusion_matrix`를 계산한다. [reports.py](../../src/opencv_preprocessing_advisor/reports.py)는 CSV/JSON/PNG evidence를 쓴다. [test_datasets.py](../../tests/test_datasets.py), [test_evaluation.py](../../tests/test_evaluation.py), [test_reports.py](../../tests/test_reports.py)가 각각 안전한 discovery·metric·report contract를 검증한다.

| 구성 요소 | 입력/파라미터 | 산출물 | 확인할 질문 |
| --- | --- | --- | --- |
| `discover_dataset` | class folder, image suffix | manifest/class names/checksum | class마다 유효 이미지가 최소 5개인가? |
| `stratified_folds(labels, n_splits=5, seed=42)` | 1D labels | train/test indices | 모든 test fold에 class가 있는가? |
| `Standardizer.fit(train)` | 훈련 fold feature | mean/scale | test fold를 fit에 넣지 않았는가? |
| `classification_metrics` | truth, predicted, class count | accuracy, Macro F1, per class | 다수 class가 metric을 가리지 않는가? |
| `confusion_matrix` | actual/predicted `int32` | actual×predicted count | 어느 class pair가 섞이는가? |
| `BenchmarkReportWriter` | benchmark result | CSV, JSON, PNG | hash/version/fold assignment가 남았는가? |

CSV의 mean metric은 fold 결과의 평균이고, aggregate confusion matrix는 fold matrix의 합이다. 둘은 서로 다른 집계 관점이다. timing도 hardware·cache 상황에 따라 달라질 수 있으므로 절대 benchmark라고 과장하지 않는다. [evaluation.py (main)](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/evaluation.py)와 [reports.py (main)](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/reports.py)의 현재 schema를 출처로 삼는다.

## 언제 사용하고 피하는가

stratified K-fold는 class별 예가 충분하고 여러 split에서 안정성을 보고 싶을 때 쓴다. Macro F1은 모든 status class를 비슷하게 중요하게 다루는 문제에서 accuracy의 보완 지표다. confusion matrix는 ‘점수가 낮다’에서 멈추지 않고 어떤 class pair를 조사할지 정하는 데 쓴다. config hash와 checksum은 config를 바꾸거나 dataset 파일이 바뀐 뒤 이전 결과와 비교할 때 유용하다.

피해야 할 경우는 시간 순서·장비·lot·사용자 같은 group이 있는 데이터를 무작위 stratified split으로 섞는 것이다. 현실 배포가 새 lot 예측이라면 group/time split이 더 정직할 수 있다. 전체 dataset에서 scaling 또는 SIFT vocabulary를 fit하는 leakage도 금지한다. 동일 이미지를 augmentation으로 만든 복제본이 train과 test에 갈라지면 stratified라도 누출이다. official MVTec anomaly detection metric과 현재의 class-folder classification metric을 바꿔 부르지 않는다.

## 프로젝트 코드 연결

- [데이터 탐색과 fold: datasets.py](../../src/opencv_preprocessing_advisor/datasets.py)는 class 이름·relative sample/checksum·stratified index를 만든다.
- [평가와 지표: evaluation.py](../../src/opencv_preprocessing_advisor/evaluation.py)는 훈련 fold만으로 scaling하고 Macro F1/confusion matrix를 계산한다.
- [재현 보고서: reports.py](../../src/opencv_preprocessing_advisor/reports.py)는 leaderboard와 run metadata를 파일로 쓴다.
- [데이터셋 테스트: test_datasets.py](../../tests/test_datasets.py)는 class minimum과 deterministic fold를 검증한다.
- [평가 테스트: test_evaluation.py](../../tests/test_evaluation.py)는 metric 값과 leakage-safe transform 흐름을 검증한다.
- [보고서 테스트: test_reports.py](../../tests/test_reports.py)는 CSV/JSON/PNG artifact 구조를 검증한다.

공개 사례의 정확한 범위는 [benchmark-evidence.json](../portfolio/benchmark-evidence.json), [experiment-results.md](../portfolio/experiment-results.md), [limitations.md](../portfolio/limitations.md)에 있다. 문서에는 사용자 path, MVTec 원본 이미지, GT mask를 넣지 않는다.

## 직접 실험

다음은 작은 불균형 label에서 `stratified_folds`와 `classification_metrics`를 실행해 accuracy와 Macro F1이 다른 이유를 보는 실험이다. sample vector 자체를 분류기로 학습하지 않으므로 private data가 필요 없다.

```python
import numpy as np

from opencv_preprocessing_advisor.datasets import stratified_folds
from opencv_preprocessing_advisor.evaluation import classification_metrics, confusion_matrix

truth = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
predicted = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int32)
folds = stratified_folds(truth, n_splits=5, seed=42)
metrics = classification_metrics(truth, predicted, class_count=2)

print("fold test labels:", [truth[fold.test_indices].tolist() for fold in folds])
print("accuracy=", round(metrics.accuracy, 3), "macro_f1=", round(metrics.macro_f1, 3))
print("matrix (actual rows, predicted columns):")
print(confusion_matrix(truth, predicted, class_count=2))
for item in metrics.per_class:
    print(item.class_index, item.precision, item.recall, item.f1, item.support)
```

그 다음 2D feature의 마지막 두 row만 극단적으로 크게 만든 뒤 scaler를 전체로 fit한 값과 train subset으로 fit한 값을 별도 출력해 본다. 차이가 난다는 사실은 whole-data scaling이 공정하다는 증거가 아니라 test distribution이 train transform을 바꾸었다는 leakage 신호다. 실제 classifier 평가는 반드시 `cross_validate()`처럼 fold 안에서 model/scaler를 새로 만든다.

## 예상 결과와 해석

| 관찰 | 예상 결과 | 해석과 다음 질문 |
| --- | --- | --- |
| stratified fold | 각 test split에 0/1 class가 들어감 | seed와 indices를 기록해야 같은 split을 되풀이한다. |
| accuracy | 다수 0을 많이 맞혀 비교적 높게 보임 | minority class recall을 가리는지 Macro F1과 비교한다. |
| Macro F1 | minority miss 때문에 accuracy보다 낮을 수 있음 | 모든 class의 중요도가 같은 task인지 확인한다. |
| confusion matrix | actual 1이 predicted 0으로 모임 | feature/label/전처리 중 무엇을 더 조사할지 정한다. |
| scaler mean/std | whole-data와 train-only 값이 다름 | test fold를 본 transform은 leakage다. |

한번의 seed 42 결과는 deterministic reference이지 불확실성의 종결이 아니다. 다른 seed, 다른 valid split, 추가 데이터에서 결과가 유지되는지 다음 실험으로 확인한다. 단, 기존 결과를 다시 만들려면 seed 42를 바꾸지 않고 config hash와 version을 맞춘다.

## 자주 하는 실수와 디버깅

1. **accuracy만 보고 승자 선택**: class support와 Macro F1, per-class recall을 함께 확인한다.
2. **전체 feature로 scaling**: test fold mean/std가 train representation에 들어간다. 훈련 fold에서만 `fit`한다.
3. **matrix 축을 뒤집음**: 이 구현은 행=actual, 열=predicted다. label 순서도 report의 class_names로 확인한다.
4. **stratified=현실적이라고 가정**: time/group/duplicate leakage는 별도로 막아야 한다.
5. **재현성 metadata 생략**: 점수만 캡처하지 말고 seed, version, hashes, fold assignments, checksum을 report로 남긴다.

fold 오류가 나면 labels가 비어 있지 않은 1D인지, class가 두 개 이상인지, 최소 class count가 2 이상인지 확인한다. metric 이상이면 truth/predicted 길이와 class index range부터 출력한다. report가 재현되지 않으면 pipeline config hash, feature/classifier 이름, OpenCV version, dataset checksum, 실제 fold count를 순서대로 비교한다.

## 본인 말로 설명하기

### 1분 설명

“평가에서는 데이터의 test 정보를 학습 과정에 넣지 않는 것이 핵심입니다. 이 프로젝트는 seed 42의 stratified folds를 만들고, 각 fold에서 훈련 fold feature로만 scaling을 fit한 뒤 test fold에는 transform만 합니다. accuracy는 다수 class에 치우칠 수 있어 Macro F1과 class별 precision/recall, confusion matrix를 함께 봅니다. confusion matrix는 행이 actual, 열이 predicted라서 어떤 status class가 섞이는지 보여 줍니다. 결과를 다시 확인할 수 있게 seed, config hash, checksum, OpenCV version, fold assignment도 report에 남깁니다.”

### 깊이 설명

“`stratified_folds`는 class별 index를 seed로 섞어 각 test fold에 class를 분배합니다. 그러나 group/time leakage까지 해결하지는 않습니다. `cross_validate`는 fold마다 Standardizer와 classifier를 새로 만들고 train features에만 fit하므로 fold-local scaling을 보장합니다. same rule은 feature selection과 SIFT vocabulary에도 적용됩니다. `classification_metrics`는 per-class precision, recall, F1과 Macro F1을 계산하고 confusion matrix는 actual rows/predicted columns으로 누적합니다. report metadata의 pipeline hash, sample checksum, requested/actual folds, OpenCV version은 점수만으로는 알 수 없는 실행 조건을 보존합니다. 이것이 단일 score를 재현 가능한 evidence로 바꾸는 과정입니다.”

## 완료 기준

- [ ] **이해**: stratified K-fold의 이점과 group/time/duplicate leakage 한계를 설명했다.
- [ ] **구현**: synthetic label로 folds, accuracy, Macro F1, confusion matrix를 출력했다.
- [ ] **해석**: accuracy와 Macro F1 차이를 minority class 오류와 연결해 해석했다.
- [ ] **설명**: fold-local scaling이 필요한 이유와 report에 남길 재현성 metadata를 말했다.
