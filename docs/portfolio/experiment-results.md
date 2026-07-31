# MVTec tile 분류 실험 결과

## 실험 범위와 데이터 해석

이 사례는 로컬 MVTec AD `tile/test`의 하위 상태 폴더를 클래스 폴더 데이터셋으로 해석한 분류 실험이다. `crack`, `glue_strip`, `good`, `gray_stroke`, `oil`, `rough`을 6개 클래스로 두었고 총 117 images를 사용했다. 이상 영역 GT mask, anomaly localization, MVTec의 공식 평가 프로토콜은 사용하지 않았다. 따라서 아래 수치는 not an official MVTec anomaly-detection metric 이다.

## 정확한 평가 프로토콜

- Dataset interpretation: `tile/test` 하위 상태 폴더를 6 classes로 해석
- Sample count: 117 images
- Split: stratified 5-fold cross-validation
- Random seed: seed 42
- Feature profile: HOG + HSV/LAB histogram + Sobel/Laplacian/Gabor texture statistics
- Classifier comparison: SVM, kNN, and RTrees
- Ranking: 평균 Macro F1, 평균 accuracy, 전처리와 특징 추출 시간 순

모든 조합은 동일한 fold 계획 아래 비교했다. 각 fold에서 표준화기는 훈련 특징만으로 적합하고 테스트 특징에는 변환만 적용한다. 이 fold-local scaling은 테스트 분포가 훈련 통계에 섞이는 누수를 방지한다.

## 리더보드

아래 표는 파이프라인별 최고 조합을 Top 3로 추린 결과다. 원본, CLAHE + Bilateral, LAB CLAHE 모두 RTrees가 해당 파이프라인의 표기된 조합을 만들었다.

| Pipeline | Classifier | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| Original | RTrees | 0.804 | **0.789** |
| CLAHE + Bilateral | RTrees | 0.766 | 0.731 |
| LAB CLAHE | RTrees | 0.664 | 0.594 |

SVM, kNN, RTrees는 모두 같은 전처리, 특징, fold 조건에서 비교 대상으로 실행된다. 이 문서는 세 분류기의 전체 순위를 새로 추정하지 않으며, 생성된 `leaderboard.csv`, fold metrics, class metrics, timing, confusion matrices가 조합별 상세 결과를 보존한다.

## 혼동행렬 읽기

혼동행렬의 행은 실제 클래스, 열은 예측 클래스다. 대각선 값은 해당 클래스의 올바른 예측이고, 대각선 밖 값은 서로 혼동한 클래스 쌍이다. Macro F1은 각 클래스 F1의 평균이므로, 샘플 수가 상대적으로 많은 클래스의 결과만으로 전체 성능을 좋게 보이게 하는 것을 줄인다. Accuracy 0.804와 Macro F1 0.789를 함께 읽어 전체 정답 비율과 클래스 균형 관점을 동시에 확인한다.

## 관찰과 해석

원본 + RTrees가 세 후보 중 최고 Macro F1을 기록했다. CLAHE + Bilateral은 0.731, LAB CLAHE는 0.594로 낮았다. 이것은 이 데이터와 고정된 특징 조합에서 전처리 강도가 커질수록 성능이 자동으로 오르지 않았다는 관찰이다.

Hypothesis: 국소 대비 강화가 클래스 구분에 유용한 자연 질감을 바꾸거나, bilateral 평활화가 약한 표면 변화를 줄였을 수 있다. 이 가설은 결과를 설명하기 위한 다음 실험의 출발점이며, 인과관계나 다른 데이터셋에 대한 일반화 결론이 아니다. 파라미터 그리드, 다른 feature profile, 별도 데이터셋을 통해 검증해야 한다.
