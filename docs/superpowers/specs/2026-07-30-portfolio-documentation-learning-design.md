# OpenCV Preprocessing Advisor 포트폴리오·학습 체계 설계

## 1. 목적

이미 구현된 OpenCV Preprocessing Advisor를 다음 세 가지 채용 산출물로 전환한다.

1. 기술 담당자가 3~5분 안에 역량을 확인할 수 있는 GitHub README
2. 핵심 판단과 정량 결과를 빠르게 전달하는 6페이지 PDF 포트폴리오
3. 설계 근거와 실험 과정을 깊게 설명하는 Notion 상세 문서

문서 완성 후에는 하루 30분, 4주 과정의 학습 자료를 제공해 작성자가 프로젝트의 주요 OpenCV 개념, 코드 구조, 실험 결과를 스스로 설명하고 부분 재구현할 수 있게 한다.

대상은 특정 산업 하나가 아니라 OpenCV를 활용하는 컴퓨터 비전, 산업용 머신비전, AI 비전 직무 전반이다.

## 2. 작성 원칙

### 2.1 설명 가능한 작성자를 전제로 한다

포트폴리오는 작성자가 프로젝트 전체를 이해하고 설명할 수 있다는 전제로 작성한다. 문장은 수동적인 기능 나열보다 문제, 판단, 기술 선택, 결과의 인과관계를 드러낸다.

예시:

> 조명 변화가 있는 이미지의 국소 대비를 보정하기 위해 LAB의 L 채널에 CLAHE를 적용했다. 전체 BGR 채널에 직접 적용하지 않아 색상 왜곡 가능성을 줄였다.

### 2.2 모든 주장은 증거와 연결한다

기술, 수치, 성능, 테스트 개수는 저장소의 코드, 설정, 테스트, 생성된 보고서로 확인 가능한 사실만 사용한다. 다음과 같은 과장 표현은 사용하지 않는다.

- OpenCV의 모든 기술을 완전히 다룬다.
- 추천 점수가 분류 정확도를 보장한다.
- 특정 전처리가 모든 데이터셋에서 최적이다.
- MVTec 공식 anomaly-detection 성능을 달성했다.

대신 프로젝트에서 사용한 기술과 판단 범위를 구체적으로 제시한다.

### 2.3 세 문서는 동일한 사실을 다른 깊이로 전달한다

- README: 빠른 기술 증거
- PDF: 채용 담당자용 핵심 사례
- Notion: 기술 면접관용 상세 설명

수치, 용어, 프로젝트 범위, 한계는 세 문서에서 일치해야 한다.

## 3. 단일 증거 원천

세 문서의 사실이 어긋나지 않도록 `docs/portfolio/`를 포트폴리오 원본 자료로 사용한다.

```text
docs/portfolio/
├─ evidence-map.md
├─ case-study.md
├─ experiment-results.md
├─ limitations.md
└─ assets/
```

### 3.1 evidence-map.md

다음 항목을 표로 연결한다.

- OpenCV 기술
- 사용 목적
- 실제 코드 위치
- 관련 테스트
- 결과 화면 또는 실험
- 설명할 핵심 판단

기술 범위:

- BGR, Gray, HSV, LAB
- normalization, gamma correction, CLAHE
- Gaussian, Median, Bilateral filtering
- Sobel, Scharr, Laplacian, Canny
- morphology, threshold, contour, connected components
- HOG, color histogram, Gabor, SIFT
- OpenCV SVM, kNN, RTrees
- stratified cross-validation, Macro F1, confusion matrix

### 3.2 case-study.md

문제 정의, 해결 흐름, 기술 선택, 정량 결과, 실패 분석, 한계를 하나의 완결된 사례로 정리한다. README, PDF, Notion은 이 문서에서 필요한 깊이만 추출한다.

### 3.3 experiment-results.md

MVTec tile 실험 조건과 결과를 기록한다.

- 117 images
- 6 classes
- stratified 5-fold
- seed 42
- combined OpenCV features
- SVM, kNN, RTrees
- Original + RTrees: Accuracy 0.804, Macro F1 0.789
- CLAHE + Bilateral + RTrees: Macro F1 0.731
- LAB CLAHE + RTrees: Macro F1 0.594

이 결과는 anomaly localization이나 MVTec 공식 평가가 아니라 `tile/test` 하위 상태 폴더를 6개 분류 클래스로 해석한 사례임을 명시한다.

### 3.4 limitations.md

휴리스틱 추천과 실제 분류 성능의 차이, 데이터셋 종속성, 고전 특징의 한계, 파라미터 탐색 범위, GT 미사용을 기록한다.

## 4. GitHub README 설계

README의 목표 독해 시간은 3~5분이다.

1. 프로젝트 한 줄 설명
2. 대표 Streamlit 화면
3. 해결하려는 문제
4. 입력 → 진단 → Top 3 추천 → 데이터셋 검증 흐름
5. MVTec tile 핵심 결과
6. OpenCV 기술 증거표
7. 시스템 구조와 데이터 흐름
8. 추천 점수와 데이터셋 평가의 차이
9. 실패 사례와 해석
10. 테스트와 재현성
11. 실행 방법
12. 한계와 향후 개선
13. PDF와 Notion 링크

README는 긴 이론 설명을 피하고 실제 코드 링크와 결과 이미지로 신뢰를 만든다.

## 5. PDF 포트폴리오 설계

PDF는 6페이지 이내로 제한한다. 각 페이지는 한 가지 메시지만 전달한다.

### 1페이지: 한눈에 보는 프로젝트

- 프로젝트명과 한 줄 설명
- 이미지 진단 → Top 3 추천 → 분류 성능 검증
- 117 images, 6 classes, Accuracy 0.804, Macro F1 0.789
- 대표 Streamlit 화면

### 2페이지: 문제와 해결 방식

```text
새로운 이미지에 적용할 전처리를 판단하기 어렵다
→ 이미지 특성 진단
→ 후보 파이프라인 적용
→ 전후 수치 비교
→ Top 3 추천
→ 클래스 폴더 데이터셋에서 교차검증
```

### 3페이지: OpenCV 기술 역량

한 장의 기술 매트릭스로 색 공간, 전처리, filtering, 경계·형태, 특징, 분류기, 평가 역량을 보여준다.

### 4페이지: 시스템과 추천 결과

- 간단한 시스템 구조도
- 원본과 추천 파이프라인 3개
- 추천 이유, 주의점, 대표 전후 수치
- 휴리스틱 적합도와 분류 성능의 역할 분리

### 5페이지: 실험 결과와 핵심 판단

Original, CLAHE + Bilateral, LAB CLAHE의 Macro F1을 비교하고 다음 판단을 강조한다.

> 전처리를 많이 적용한다고 성능이 자동으로 향상되지는 않았다. 이 데이터에서는 원본 질감 정보가 분류에 더 유효했고, 국소 대비 강화가 일부 클래스 구분 정보를 왜곡했을 가능성이 있다.

### 6페이지: 결과와 연결

- 구현한 핵심 기능
- 테스트와 재현성
- 명확한 한계
- GitHub, Notion, Streamlit 실행 방법
- 향후 개선 방향 2~3개

상세 공식, 긴 코드, 전체 트러블슈팅은 PDF에서 제외한다.

## 6. Notion 상세 문서 설계

Notion은 다음을 깊게 설명한다.

1. 프로젝트 배경과 요구사항 변화
2. 단일 이미지 추천과 데이터셋 평가를 분리한 이유
3. 색 공간, filter, contrast, gradient, morphology 선택 근거
4. 이미지 진단 지표와 휴리스틱 점수
5. HOG, histogram, Gabor, SIFT 특징
6. SVM, kNN, RTrees 비교
7. leakage 방지와 재현성
8. MVTec tile 실험 구성
9. 클래스별 결과와 혼동행렬
10. 성공·실패 전처리 사례
11. 트러블슈팅
12. 한계와 다음 실험

Notion은 코드 저장소의 대체물이 아니라 기술적 판단을 설명하는 동반 문서다. 각 주요 절에는 관련 GitHub 파일 링크를 연결한다.

## 7. 제작 흐름

### 7.1 증거 수집

- 최신 `main`의 코드와 테스트 확인
- 기술-코드-테스트 매핑
- Streamlit 대표 화면 캡처
- 단일 이미지 전후 비교 자료 선정
- MVTec 결과표와 혼동행렬 검증
- 공개 저장소에 포함할 수 있는 자산만 선별

### 7.2 원본 사례 작성

`docs/portfolio/`의 네 문서를 먼저 작성한다. 수치와 용어는 여기에서 확정한다.

### 7.3 파생 문서 제작

1. README 개편
2. 6페이지 PDF 제작과 렌더링 검수
3. Notion 상세 페이지 작성

### 7.4 Public 전환

문서와 저장소 검수가 끝난 뒤에만 Private 저장소를 Public으로 전환한다.

## 8. 검증과 공개 안전성

### 8.1 사실 검증

- 모든 성능 수치를 생성된 CSV·JSON 보고서와 대조
- 기술 증거표의 코드 링크가 최신 `main`에서 열리는지 확인
- README, PDF, Notion의 수치·용어·범위를 상호 비교
- 테스트 명령을 새 환경에서 실행

### 8.2 시각 검증

- PDF를 페이지 이미지로 렌더링해 글자 잘림, 대비, 여백, 표 가독성을 확인
- README의 이미지와 링크가 GitHub에서 렌더링되는지 확인
- Notion의 표, 코드, 이미지, 링크 구조를 확인

### 8.3 Public 전환 검사

- MVTec 원본 이미지와 재배포 제한 자산이 저장소에 없는지 확인
- 사용자명, 로컬 절대 경로, 임시 출력물이 노출되지 않는지 확인
- 비밀키, 토큰, 환경 파일이 없는지 확인
- 보고서의 dataset path와 manifest 공개 범위를 확인
- 저장소 설명, 토픽, 라이선스, 설치 명령을 정리

## 9. 포트폴리오 완성 후 학습 체계

학습은 문서 제작이 끝난 뒤 시작한다. 하루 30분, 4주 과정이다.

### 1주차: 이미지 표현과 색 공간

- NumPy image array
- shape, dtype, channel
- BGR, Gray, HSV, LAB
- Unicode-safe image I/O

### 2주차: 전처리와 이미지 진단

- normalization, gamma, CLAHE
- Gaussian, Median, Bilateral
- Sobel, Scharr, Laplacian, Canny
- morphology와 진단 지표

### 3주차: 특징·분류·평가

- HOG, histogram, Gabor, SIFT
- OpenCV SVM, kNN, RTrees
- stratified cross-validation
- Macro F1, confusion matrix, leakage

### 4주차: 설명과 재구현

- 프로젝트 전체 흐름 5분 설명
- 작은 전처리 파이프라인 재구현
- 파라미터 변경 실험
- 실패 사례 원인 분석
- 면접 질문 답변

매일 학습 형식:

```text
5분  전날 내용 회상
10분 개념과 프로젝트 코드 연결
10분 작은 코드 또는 파라미터 실험
5분  자료를 보지 않고 말로 설명
```

각 학습 단위는 개념 요약, 코드 위치, 직접 답할 질문, 작은 구현 과제, 파라미터 실험, 실패 사례, 면접 질문, 완료 체크리스트를 포함한다.

## 10. 완료 기준

- README가 3~5분 내에 프로젝트 가치와 OpenCV 역량을 전달한다.
- PDF가 6페이지를 넘지 않고 페이지마다 메시지가 하나다.
- Notion이 주요 기술 선택과 평가 방법을 상세히 설명한다.
- 세 문서의 수치와 용어가 일치한다.
- 모든 핵심 기술 주장에 코드 또는 실험 증거가 연결된다.
- 새 PC에서 설치와 Streamlit 실행이 가능하다.
- Public 전환 안전성 검사를 통과한다.
- 학습 과정 완료 후 작성자가 프로젝트를 5분 안에 설명할 수 있다.
- 주요 기법에 대해 왜 사용했는지, 언제 피해야 하는지 답할 수 있다.

## 11. 범위 제외

- Streamlit UI의 시각적 전면 개편
- 새로운 딥러닝 모델 또는 anomaly-detection 기능
- MVTec GT mask 평가
- 영상 처리 기능
- 프로젝트 범위를 넘어선 OpenCV 전체 API 학습

