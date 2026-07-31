# Day 8 - OpenCV 분류기

오늘은 feature matrix와 레이블을 받아 고전적 분류기를 학습시키는 경계를 다룬다. `cv2.ml`의 SVM, kNN, RTrees는 이름은 간단하지만 dtype, row 방향, scaling, fold 분리라는 입력 계약을 지키지 않으면 비교 결과가 무의미해진다.

## 오늘 답해야 할 핵심 질문

- `cv2.ml` 분류기가 받는 feature와 label의 shape/dtype 계약은 무엇인가?
- SVM, kNN, RTrees는 각각 어떤 결정 원리와 민감도를 갖는가?
- 왜 feature matrix는 `float32`, label은 `int32`로 정리하는가?
- scaling은 어떤 분류기에 특히 중요하며 왜 train/test를 나누어 fit해야 하는가?
- classifier 선택을 정확도 한 숫자가 아닌 Macro F1, 시간, 오류 패턴과 함께 보는 이유는 무엇인가?

## 개념과 원리

이 프로젝트의 classifiers는 OpenCV 객체를 얇은 adapter로 감싼다. `_feature_matrix()`는 입력을 `np.float32` 2차원 matrix로 바꾸고 비어 있거나 NaN/inf면 실패시킨다. `_label_vector()`는 레이블을 `np.int32` 1차원으로 펴고 행 수가 feature의 sample 수와 같은지 확인한다. `cv2.ml.ROW_SAMPLE`은 각 행이 한 sample이라는 뜻이다. `(n_samples, n_features)`와 `(n_samples,)`의 대응을 어기면 모델이 학습할 대상 자체가 달라진다.

SVM은 class 사이 margin을 크게 하는 decision boundary를 찾는다. 여기서는 C-SVC, RBF kernel, `C=2.0`, `gamma=0.01`, 최대 1000 반복/epsilon 1e-6을 명시한다. RBF SVM에서 feature scale은 거리 계산을 바꾸므로 scaling이 특히 중요하다. C가 너무 크면 train error에 과하게 맞고, gamma가 너무 크면 각 sample 주변에 너무 복잡한 경계를 만들 수 있다. 두 값은 데이터에 따라 조정·검증할 hyperparameter이지 보편적 정답이 아니다.

kNN은 학습 데이터를 저장하고 예측 sample 주변의 k개 이웃 투표로 class를 정한다. 코드의 기본 `k=5`이며 training row보다 k가 크면 실제 row 수까지 낮춘다. 거리 기반이므로 한 feature의 단위가 크면 그 축이 이웃을 지배한다. sample 수가 작고 국소 구조가 뚜렷할 때 직관적이지만, 고차원에서는 거리가 비슷해지는 현상과 예측 비용을 고려한다. k는 작은 값이면 noise에 민감하고 큰 값이면 경계가 과도하게 평활해질 수 있다.

RTrees는 많은 randomized decision tree의 투표를 합친 ensemble이다. 이 구현은 max depth 12, min sample count 2, 최대 200 iteration을 사용한다. tree는 feature 임계값 분기이므로 scaling에 SVM/kNN보다 덜 민감한 편이지만, 공정한 공통 pipeline에서는 같은 fold-local scaling을 거친다. feature importance 비슷한 값은 탐색 단서가 될 수 있어도 상관된 feature, 작은 sample, split 선택의 영향을 받으므로 원인 증명은 아니다.

분류기 자체가 preprocessor의 성공 여부를 말해 주지 않는다. 같은 feature와 split에서 Original + RTrees가 더 높으면, 그 데이터의 그 조건에서는 대비 강화가 class 분리에 도움을 주지 않았거나 유용한 texture를 바꿨다는 engineering conclusion이다. 이것은 실패가 아니라 비교 실험이 준 정보다. 분류기의 winner는 다른 dataset, seed, split, cost에서 자동으로 일반화되지 않는다.

## OpenCV API와 파라미터

[classifiers.py](../../src/opencv_preprocessing_advisor/classifiers.py)는 Standardizer와 세 `cv2.ml` adapter를 제공한다. [evaluation.py](../../src/opencv_preprocessing_advisor/evaluation.py)는 매 fold마다 Standardizer를 fit하고 classifier를 새로 만든다. [test_classifiers.py](../../tests/test_classifiers.py)와 [test_evaluation.py](../../tests/test_evaluation.py)는 dtype, fit-before-predict, deterministic seed, fold 동작을 검증한다.

| 분류기/API | 주요 파라미터 | 장점 | 주의점 |
| --- | --- | --- | --- |
| `cv2.ml.SVM_create()` | RBF, `C=2.0`, `gamma=0.01` | 복잡한 margin을 표현 | scale·C·gamma에 민감하고 확률 점수로 해석하지 않는다. |
| `cv2.ml.KNearest_create()` | `k=5`, classifier mode | local neighbor 판단이 직관적 | 고차원 거리와 feature scale, 예측 비용을 점검한다. |
| `cv2.ml.RTrees_create()` | depth 12, min samples 2, 200 trees | 비선형 split/혼합 feature에 유용 | feature importance를 인과로 읽지 않는다. |
| `Standardizer` | train mean/std | feature 축을 비교 가능한 scale로 변환 | test 포함 fit은 leakage다. |
| `create_classifier(name, seed)` | `svm`, `knn`, `rtrees` | 이름별 adapter를 새로 생성 | fold마다 model state를 재사용하지 않는다. |

SVM과 RTrees는 `cv2.setRNGSeed(seed)`를 호출해 randomness를 통제한다. 단, seed를 고정했다고 실험 전체가 재현되는 것은 아니다. dataset order, split, feature code, OpenCV version, config도 함께 남겨야 한다. [classifiers.py (main)](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/classifiers.py)에서 현재 값과 예외 조건을 확인한다.

## 언제 사용하고 피하는가

SVM은 feature가 수치화되어 있고 margin 기반 비선형 경계를 시험할 때 좋은 baseline이다. kNN은 가까운 sample의 의미를 설명하거나 작은 feature 공간에서 빠르게 기준을 세울 때 유용하다. RTrees는 color/HOG/texture처럼 서로 성격이 다른 feature가 섞였을 때 강한 후보가 될 수 있다. 세 모델을 같은 stratified fold로 비교하면 특정 알고리즘의 취향보다 evidence를 앞세울 수 있다.

피해야 할 것은 모델마다 다른 split, 다른 preprocessor, 다른 feature를 써놓고 classifier만 비교했다고 말하는 일이다. `float64` array가 OpenCV에서 조용히 변환되길 기대하거나 label을 one-hot matrix로 넘기는 것도 계약 위반이다. train score만 보고 SVM parameter를 고르지 않는다. kNN에 표준화하지 않은 HOG+histogram을 넣고 거리 결과를 해석하지 않는다. tree의 scale 민감도가 낮아도 leakage를 허용하는 이유가 되지 않는다.

## 프로젝트 코드 연결

- [분류기 adapter: classifiers.py](../../src/opencv_preprocessing_advisor/classifiers.py)는 `float32` feature, `int32` label, fit/predict state를 강제한다.
- [교차 검증: evaluation.py](../../src/opencv_preprocessing_advisor/evaluation.py)는 fold마다 Standardizer와 model을 새로 만든다.
- [서비스 실행: services.py](../../src/opencv_preprocessing_advisor/services.py)는 feature profile·classifier 이름 조합을 만든다.
- [분류기 테스트: test_classifiers.py](../../tests/test_classifiers.py)는 SVM/kNN/RTrees의 입력과 예측 계약을 확인한다.
- [평가 테스트: test_evaluation.py](../../tests/test_evaluation.py)는 fold별 metric·scaling 경계를 확인한다.
- [서비스 테스트: test_services.py](../../tests/test_services.py)는 benchmark entry ranking의 연결을 확인한다.

이 구조는 UI와 CLI가 서로 다른 분류기를 몰래 쓰지 않도록 한다. test는 높은 성능을 주장하지 않고 작은 합성 matrix에서 API 계약이 지켜지는지 확인한다. 결과가 나쁘면 adapter를 먼저 비난하기보다 feature/label align, fold, scaling, class count를 순서대로 확인한다.

## 직접 실험

다음은 두 class의 작은 synthetic feature matrix에서 같은 `float32` 입력으로 SVM, kNN, RTrees를 학습·예측하는 실행 가능한 확인이다. 교육용 matrix이므로 일반화 성능 실험이 아니다.

```python
import numpy as np

from opencv_preprocessing_advisor.classifiers import Standardizer, create_classifier

features = np.array(
    [[0.0, 0.1], [0.2, 0.0], [0.1, 0.3], [1.0, 1.1], [0.9, 1.0], [1.2, 0.8]],
    dtype=np.float32,
)
labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
train = np.array([0, 1, 3, 4], dtype=np.int64)
test = np.array([2, 5], dtype=np.int64)
scaler = Standardizer().fit(features[train])

for name in ("svm", "knn", "rtrees"):
    model = create_classifier(name, seed=42)
    model.fit(scaler.transform(features[train]), labels[train])
    prediction = model.predict(scaler.transform(features[test]))
    print(name, "prediction=", prediction.tolist(), "truth=", labels[test].tolist())
```

`features.astype(np.float64)`로 바꿨을 때 adapter가 다시 `float32`를 보장하는지, label 길이를 하나 줄였을 때 명확한 ValueError가 나는지도 확인한다. 그 다음 하나의 column에 1,000을 곱해 Standardizer 전후 kNN 예측이 어떻게 달라지는지 관찰한다. 이는 scaling의 필요를 보여 주는 실험이지, test fold까지 포함해 scaler를 fit하는 허가가 아니다.

## 예상 결과와 해석

| 관찰 | 예상 결과 | 해석과 다음 질문 |
| --- | --- | --- |
| adapter 입력 | feature는 `(n, d)` `float32`, label은 길이 n `int32` | OpenCV 문제보다 matrix contract를 먼저 점검한다. |
| SVM | 두 test point에 margin 기반 예측 | C/gamma가 바뀌면 경계도 달라진다. fold CV에서만 선택한다. |
| kNN | 가까운 class의 vote | 큰 단위 column이 있으면 scale이 이웃 관계를 바꾼다. |
| RTrees | 여러 split의 aggregate 예측 | seed·tree 수·깊이와 작은 sample의 변동을 기록한다. |
| model 비교 | 한 모델이 항상 이기지 않음 | Macro F1, confusion matrix, fit/predict time을 같은 split으로 본다. |

작은 synthetic matrix에서 모두 정답을 맞혀도 실제 이미지 class에서 성능이 보장되지 않는다. 오히려 이 실험의 성공 조건은 dtype과 fit/predict 흐름이 명확하다는 것이다. 실제 benchmark에서는 한 조합의 mean Macro F1뿐 아니라 fold별 분산과 특정 class의 recall을 함께 본다.

## 자주 하는 실수와 디버깅

1. **행/열 반전**: OpenCV는 `ROW_SAMPLE`에서 각 행이 sample이다. `(d, n)`으로 넘기지 않는다.
2. **dtype 방치**: image feature와 label을 각각 `float32`, `int32`로 명시한다. object array/NaN도 확인한다.
3. **fit 전 predict**: adapter는 상태 오류를 낸다. fold마다 새 model을 만들고 train만 fit한다.
4. **kNN scaling 생략**: unit이 큰 feature가 거리를 지배한다. scaler를 train fold에서 fit한다.
5. **RTrees feature importance 과신**: split 통계는 설명 단서일 뿐 인과적 결함 원인을 증명하지 않는다.

OpenCV train failure가 나면 먼저 `matrix.shape`, `matrix.dtype`, `labels.shape`, class가 둘 이상인지, train row 수가 k보다 충분한지 출력한다. 예측이 계속 한 class면 scaler의 mean/scale과 class distribution을 보고, 마지막에 C/gamma/depth 같은 parameter를 하나씩 바꾼다. 여러 설정을 한 번에 바꾸면 비교 근거가 사라진다.

## 본인 말로 설명하기

### 1분 설명

“이 프로젝트는 `cv2.ml`을 SVM, kNN, RTrees adapter로 감싸고 feature는 `(sample, feature)` `float32`, label은 `int32`로 고정합니다. SVM은 margin과 RBF kernel, kNN은 scale에 민감한 거리 투표, RTrees는 여러 tree split의 ensemble입니다. 그래서 Standardizer는 매 fold의 훈련 feature로만 fit하고 test에는 transform만 합니다. 모델 선택은 같은 fold에서 Macro F1, confusion matrix, 시간과 class별 오류를 보고 합니다. 한 전처리나 classifier의 시각적 인상만으로 승자를 정하지 않습니다.”

### 깊이 설명

“`_feature_matrix`와 `_label_vector`가 OpenCV 입력 계약을 초기에 실패시켜 UI/CLI 차이를 막습니다. OpenCvSvm은 C-SVC RBF의 C=2, gamma=0.01을, kNN은 default K=5를, RTrees는 depth/min sample/tree count를 명시합니다. 이 값들은 현재 reproducible baseline이지 universal optimum이 아닙니다. distance/margin 계열은 scale의 영향을 크게 받기 때문에 cross_validate가 Standardizer를 train fold에만 fit합니다. RTrees도 common protocol을 따라 공정하게 비교합니다. 결과에서 Original+RTrees가 이기면 preprocessor가 실패한 것이 아니라 그 dataset·feature·split에서 원본 정보가 더 잘 보존됐다는 engineering evidence입니다.”

## 완료 기준

- [ ] **이해**: SVM, kNN, RTrees의 결정 원리와 scale 민감도 차이를 설명했다.
- [ ] **구현**: synthetic `float32` matrix와 `int32` label로 세 classifier의 fit/predict를 실행했다.
- [ ] **해석**: 한 column scale을 바꾼 kNN 관찰과 fold-local standardization의 필요를 기록했다.
- [ ] **설명**: 한 classifier의 한 번의 정답률만으로 선택하지 않는 이유를 Macro F1·오류 행렬과 연결해 말했다.
