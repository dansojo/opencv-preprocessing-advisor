# Day 6 - 전처리 파이프라인과 추천 점수

오늘의 목표는 ‘필터 하나’가 아니라 **순서가 있는 파이프라인**을 읽고, 단일 이미지에서 내는 추천 점수와 레이블 데이터셋에서 재는 분류 성능을 분리해 설명하는 것이다. 이 프로젝트의 Advisor는 한 장의 입력에서 진단값을 비교해 Top 3 후보를 제안한다. 그 순위는 실험을 시작할 우선순위이며, 정확도나 일반화 성능의 예측값은 아니다.

## 오늘 답해야 할 핵심 질문

- transform 하나와 여러 transform을 연결한 pipeline은 무엇이 다른가?
- YAML 설정은 코드와 어떤 계약으로 연결되며, 순서를 바꾸면 왜 결과가 달라지는가?
- heuristic 추천 점수는 어떤 근거를 합치며 왜 정확도가 아닌가?
- clipping, excessive edges, oversmoothing, color loss 경고는 언제 해석해야 하는가?
- 단일 이미지 Top 3와 레이블이 있는 benchmark leaderboard를 어떻게 구분해 말해야 하는가?

## 개념과 원리

파이프라인은 입력 `I0`에 변환 `T1`, `T2`, …를 순서대로 적용하는 함수 합성이다. 즉 `Iout = Tn(...T2(T1(I0)))`다. CLAHE 뒤 bilateral을 적용하는 것과 bilateral 뒤 CLAHE를 적용하는 것은 일반적으로 같지 않다. 전자는 먼저 국소 대비와 texture를 함께 키운 뒤 edge-preserving smoothing을 하고, 후자는 먼저 일부 변화량을 줄인 뒤 남은 분포에 대비 증강을 한다. 따라서 “CLAHE와 bilateral을 둘 다 썼다”만으로는 재현할 수 없다. 입력, 순서, 파라미터, OpenCV 버전, 설정 파일이 모두 실험 조건이다.

이 저장소는 pipeline 정의를 Python 조건문에 흩뿌리지 않고 YAML에 둔다. 각 항목에는 식별자, 표시 이름, 적용 가능한 task profile, rationale/warning code, `steps`가 있다. `PipelineCatalog.from_yaml()`은 YAML을 읽고 각 transform과 파라미터를 검증한다. `run()`은 복사한 BGR 이미지를 단계별로 바꾸며 중간 이미지와 처리 시간을 남긴다. 그래서 새 실험은 YAML diff로 ‘무엇을, 어떤 순서로’ 추가했는지 검토할 수 있다.

Advisor의 heuristic은 전후 진단량에서 local contrast, entropy, noise reduction, sharpness, edge continuity/density, clipping control 등을 0~100 규모로 바꾸고 profile별 가중합을 계산한다. 예를 들어 `auto` profile은 local contrast 0.25, entropy 0.15, noise reduction 0.20, sharpness 0.15, edge continuity 0.15, clipping control 0.10을 사용한다. 이 수치는 “이 입력에서 이 목적에 맞아 보이는 변화”를 투명하게 비교하기 위한 정책값이다. 정답 레이블, holdout, 오분류 비용을 보지 않으므로 **heuristic 점수는 정확도도, 모델 확률도, 분류 성능도 아니다.**

순위 뒤에는 안전장치가 있다. 밝거나 어두운 clipping 비율이 증가하거나 총 clipping이 크면 `clipping`, edge density가 기준선의 2.5배를 넘으면 `excessive_edges`, sharpness가 절반 아래로 떨어지면 `oversmoothing`, colorfulness가 절반 아래로 떨어지면 `color_loss` 경고를 낸다. 경고가 있다고 후보를 자동 제거하지는 않는다. 예를 들어 흑백 shape 작업에서 color loss는 의도된 비용일 수 있으나, 색상 class를 구분해야 한다면 위험 신호다. 추천 이유와 경고는 점수를 숨기지 않고 사람이 검토하게 하는 설명 단위다.

반대로 BenchmarkService는 class folder의 이미지와 레이블을 받아 동일한 파이프라인·특징·분류기 조합을 stratified fold에서 비교한다. 여기서 얻는 Macro F1/accuracy만이 그 데이터셋 조건에서의 분류 평가다. 화면이 더 선명한가와 label을 더 맞히는가는 다른 질문이다. 시각적 개선은 분류 성능을 보장하지 않는다. Advisor가 추천한 항목도 benchmark에서 낮을 수 있고, 원본이 이길 수도 있다.

## OpenCV API와 파라미터

[pipelines.py](../../src/opencv_preprocessing_advisor/pipelines.py)는 YAML에서 `PipelineDefinition`과 `StepDefinition`을 만들고 실행 순서를 보존한다. [scoring.py](../../src/opencv_preprocessing_advisor/scoring.py)는 profile weight, score component, warning code를 계산한다. [services.py](../../src/opencv_preprocessing_advisor/services.py)는 Advisor의 Top 3와 별도 BenchmarkService를 연결한다. [pipelines.yaml](../../src/opencv_preprocessing_advisor/config/pipelines.yaml)과 [scoring.yaml](../../src/opencv_preprocessing_advisor/config/scoring.yaml)은 각각 변환 순서와 가중치를 공개한다. [test_pipelines.py](../../tests/test_pipelines.py), [test_scoring.py](../../tests/test_scoring.py), [test_services.py](../../tests/test_services.py)는 이 계약을 고정한다.

| API/설정 | 핵심 인자 | 하는 일 | 점검할 비용 |
| --- | --- | --- | --- |
| `PipelineCatalog.from_yaml(path)` | YAML path | pipeline ID, steps, params를 읽고 검증 | 존재하지 않는 transform·잘못된 kernel은 시작 전에 실패해야 한다. |
| `catalog.run(id, image)` | ID, BGR image | 각 step을 순서대로 실행하고 intermediate를 보관 | 같은 step 집합이라도 순서가 바뀌면 다른 출력이다. |
| `rank_recommendations(..., limit=3)` | 전후 진단, profile | heuristic 점수로 Top 3를 정렬 | `limit=3`은 후보 수이지 성능 보증이 아니다. |
| `score_pipeline` | profile weights | 가중 component와 warning을 생성 | weight 합은 1.0이어야 하며 profile 목적과 맞아야 한다. |
| `lab_clahe` | `clip_limit`, `grid_size` | L 채널 국소 대비를 조절 | clipLimit가 크면 texture/noise와 clipping 위험을 함께 본다. |

YAML의 `clahe-bilateral`은 `lab_clahe` 다음에 `bilateral`을 둔다. `diameter=7`, `sigma_color=40`, `sigma_space=40`은 학습된 정답이 아니라 해당 catalog의 명시적 후보 조건이다. YAML을 고친 뒤에는 새 ID, profile 범위, 변환 파라미터 검증, 테스트를 함께 검토한다. Notion에 옮길 때에는 [pipelines.py (main)](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/pipelines.py)와 [scoring.py (main)](https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/src/opencv_preprocessing_advisor/scoring.py)처럼 `main`의 구현 링크를 사용한다.

## 언제 사용하고 피하는가

pipeline은 같은 진단 가설을 반복 적용할 때 사용한다. 예를 들어 국소 대비가 낮고 noise도 의심되면 LAB CLAHE → bilateral처럼 각각의 목적이 다른 두 단계를 후보로 만들 수 있다. 단일 이미지에서 우선 살펴볼 후보를 고르는 heuristic은 레이블이 없거나 즉시 feedback이 필요한 상황에 유용하다. 이유와 경고를 함께 표시해야 사용자도 ‘왜 이 후보인가’를 검토할 수 있다.

피해야 할 것은 Top 3의 첫 항목을 정답 처리하는 일이다. heuristic에 없는 downstream task, 결함의 최소 크기, 촬영 조명 변화, label 분포는 점수에 반영되지 않는다. 또한 과도한 smoothing으로 작은 결함을 지워 놓고 noise estimate가 낮아졌다는 이유만으로 성공이라 해석하면 안 된다. clipping 또는 oversmoothing 경고가 있으면 원본·중간 단계·최종 이미지를 같이 보고 목적상 허용되는 손실인지 판단한다. 레이블이 있으면 후보를 고른 다음에 반드시 공정한 fold 평가로 확인한다.

## 프로젝트 코드 연결

- [파이프라인 catalog: pipelines.py](../../src/opencv_preprocessing_advisor/pipelines.py)는 YAML step을 검증하고 입력을 복사해 순서대로 실행한다.
- [추천 점수: scoring.py](../../src/opencv_preprocessing_advisor/scoring.py)는 component와 clipping/oversmoothing 경고를 계산한다.
- [서비스 경계: services.py](../../src/opencv_preprocessing_advisor/services.py)는 ImageAdvisorService와 BenchmarkService를 분리한다.
- [파이프라인 테스트: test_pipelines.py](../../tests/test_pipelines.py)는 정의·순서·파라미터 계약을 확인한다.
- [점수 테스트: test_scoring.py](../../tests/test_scoring.py)는 weight와 warning 조건을 검증한다.
- [서비스 테스트: test_services.py](../../tests/test_services.py)는 Top 3 추천과 benchmark 흐름을 검증한다.

공개 가능한 재현 경로는 [synthetic-tile.png](../../data/samples/synthetic-tile.png)와 설정 파일이다. MVTec 원본이나 개인 데이터 경로를 문서에 넣지 않는다. benchmark 결과를 읽을 때는 [experiment-results.md](../portfolio/experiment-results.md)와 [benchmark-evidence.json](../portfolio/benchmark-evidence.json)의 범위를 함께 확인한다.

## 직접 실험

아래 실험은 private dataset 없이 합성 타일로 Advisor를 돌리고, Top 3의 점수·reason·warning을 출력한다. 후보별 출력 이미지를 저장하지 않아도 먼저 진단과 경고의 관계를 확인할 수 있다.

```python
from pathlib import Path

from opencv_preprocessing_advisor.io import decode_image
from opencv_preprocessing_advisor.models import TaskProfile
from opencv_preprocessing_advisor.services import ImageAdvisorService

image = decode_image(Path("data/samples/synthetic-tile.png"))
result = ImageAdvisorService().analyze(image, profile=TaskProfile.AUTO)

print("OpenCV:", result.opencv_version)
print("pipeline config hash:", result.pipeline_config_hash[:12])
for rank, item in enumerate(result.recommendations, start=1):
    components = ", ".join(
        f"{part.name}={part.value:.1f}×{part.weight:.2f}"
        for part in item.score_components
    )
    print(f"Top {rank}: {item.pipeline_id}, score={item.suitability_score:.2f}")
    print("  reasons:", "; ".join(item.reasons))
    print("  warnings:", ", ".join(item.warning_codes) or "none")
    print("  components:", components)
```

다음으로 `pipelines.yaml`을 복사해 `clahe-bilateral`의 두 step 순서만 바꾼 실험용 ID를 만들고 같은 입력에 실행한다. 원본 설정 파일을 덮어쓰지 말고, 복사본의 각 step과 출력 hash를 기록한다. 결과가 달라지면 “어느 transform이 더 좋다”라고 결론내리지 말고 순서가 비가환이라는 사실, 진단 component 변화, warning 변화를 표에 적는다.

## 예상 결과와 해석

| 관찰 | 예상 결과 | 해석과 다음 질문 |
| --- | --- | --- |
| Top 3 순위 | 후보마다 score, reasons, warnings가 출력 | 높은 score는 이 입력/profile의 heuristic 우선순위다. accuracy가 아니다. |
| CLAHE 포함 후보 | local contrast/entropy가 올라갈 수 있음 | clipping 또는 excessive edges가 늘면 강조의 비용을 확인한다. |
| smoothing 포함 후보 | noise 항목은 좋아질 수 있음 | sharpness가 반 이하이면 oversmoothing 경고와 작은 구조 손실을 본다. |
| step 순서 교환 | 중간·최종 진단값이 달라질 수 있음 | catalog는 step 집합이 아니라 순서가 있는 실험 명세다. |
| 색상 profile | color preservation 비중이 큼 | gray 변환은 의도적이라도 color class에는 부적절할 수 있다. |

점수 차이가 0.3처럼 작으면 ‘승자’라고 과장하지 않는다. 처리 시간, 경고 수, 목적상 중요한 component, 원본 overlay를 함께 비교한다. 레이블을 가진 class folder가 준비됐을 때만 `opencv-prep benchmark`로 동일 후보의 Macro F1과 confusion matrix를 별도로 측정한다. 추천 화면의 score와 leaderboard의 metric을 한 표에 섞지 않는다.

## 자주 하는 실수와 디버깅

1. **heuristic을 정확도로 소개**: score에는 정답 label이 없다. “실험 우선순위”라고 표현하고 분류 성능은 별도 CV로 말한다.
2. **YAML 순서를 무시**: transform 목록을 set처럼 바꾸면 실험 정의가 바뀐다. step 순서와 params를 config hash/diff로 남긴다.
3. **경고를 실패 판정으로 오해**: warning은 사람이 검토할 risk signal이다. 목적에 따라 허용/불허 근거를 기록한다.
4. **score만 보고 원본을 버림**: clipping과 oversmoothing은 작은 구조를 잃게 할 수 있다. 원본·intermediate·최종을 나란히 본다.
5. **profile weight를 임의로 합산**: profile마다 목적이 다르며 YAML weight 합이 1.0인지 검증한다.

점수가 예상과 다르면 먼저 입력이 BGR `uint8`인지, pipeline ID가 profile에 포함되는지, config hash가 바뀌었는지 확인한다. 이어서 before/after diagnostics와 warning code를 출력한다. 마지막에야 transform 파라미터를 하나씩 바꾼다. 여러 step·weight·입력을 동시에 바꾸면 원인을 추적할 수 없다.

## 본인 말로 설명하기

### 1분 설명

“이 프로젝트에서 pipeline은 YAML에 정의한 전처리 step의 순서 있는 조합입니다. Advisor는 한 장의 이미지 전후 진단값을 profile별 가중치로 합쳐 Top 3를 제안하고 clipping, excessive edges, oversmoothing, color loss 같은 위험도 같이 보여 줍니다. 이 heuristic은 레이블을 보지 않으므로 정확도나 분류 성능이 아니라 다음 실험의 우선순위입니다. 레이블 데이터에서는 별도 BenchmarkService가 같은 후보를 fold 평가해 Macro F1과 accuracy를 비교합니다. 그래서 화면이 더 좋아 보이는 결과와 모델이 더 잘 맞히는 결과를 분리합니다.”

### 깊이 설명

“YAML catalog가 pipeline ID, profile, step, parameter, warning contract를 보존하고 PipelineCatalog가 이를 검증·실행합니다. 함수 합성이라 CLAHE→bilateral과 bilateral→CLAHE는 같은 실험이 아닙니다. scoring은 대비, entropy, noise, sharpness, edge, clipping의 전후 변화로 투명한 component를 만들고, profile의 가중합으로 순위를 정합니다. 그러나 그 변화에는 클래스 label, unseen 데이터, error cost가 없으므로 score를 accuracy로 읽을 수 없습니다. 따라서 Top 3를 hypothesis shortlist로 쓰고, 필요하면 같은 pipeline/feature/classifier를 stratified cross-validation으로 검증합니다. 원본+RTrees가 benchmark에서 이겨도 Advisor가 실패한 것이 아니라, 서로 다른 의사결정 단계를 분리한 설계입니다.”

## 완료 기준

- [ ] **이해**: YAML step 순서가 결과를 바꾸는 이유와 heuristic이 accuracy가 아닌 이유를 설명했다.
- [ ] **구현**: 합성 샘플에서 Advisor를 실행해 Top 3의 score, reason, warning, config hash를 출력했다.
- [ ] **해석**: clipping 또는 oversmoothing 가능성이 있는 후보 하나에 대해 원본·중간·최종을 비교할 다음 검사를 적었다.
- [ ] **설명**: 단일 이미지 추천과 레이블 기반 benchmark의 입력·출력·성공 기준 차이를 1분 안에 말했다.
