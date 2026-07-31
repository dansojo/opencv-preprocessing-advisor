# Notion 포트폴리오 + 10일 심화 학습 허브 재설계

## 1. 목적

현재 Notion 페이지는 채용 담당자가 프로젝트의 목적, 구조, 실험 결과와 한계를 빠르게 파악하는 사례 요약으로는 유효하지만, 사용자가 OpenCV를 학습하고 프로젝트를 자기 말로 설명하기 위한 교재로는 설명과 실습이 부족하다.

이번 재설계의 목표는 기존 포트폴리오 페이지를 유지하면서 그 아래에 독립적인 학습 체계를 추가하는 것이다. 학습 완료 후 사용자는 다음을 수행할 수 있어야 한다.

- 프로젝트가 해결하는 문제와 단일 이미지 추천·레이블 데이터셋 평가의 차이를 설명한다.
- 주요 OpenCV 전처리·진단·특징 추출 API의 원리와 파라미터를 설명하고 선택한다.
- 전처리 결과가 좋아 보이는 것과 분류 성능이 개선되는 것을 구분한다.
- 프로젝트 코드에서 기술의 구현 경로를 찾아 작은 단위로 다시 구현한다.
- 교차 검증, 데이터 누수 방지, Macro F1과 혼동행렬을 해석한다.
- 프로젝트의 성공, 실패, 한계와 다음 실험을 면접 답변으로 구성한다.

학습에는 일일 시간 제한을 두지 않는다. 각 일차는 주제를 충분히 이해하고 실습·설명 완료 기준을 통과하는 것을 기준으로 한다.

## 2. 대상 독자와 사용 모드

### 채용 담당자 모드

메인 페이지 상단에서 3~5분 안에 프로젝트의 가치, OpenCV 적용 범위, 검증 수치, 실제 화면, 코드와 PDF를 확인한다. 상세 학습 내용은 읽지 않아도 핵심 역량을 파악할 수 있어야 한다.

### 학습자 모드

메인 페이지의 학습 허브에서 10일 과정, 기술 Q&A, 면접 질문, 실습 과제, 진도 체크리스트로 이동한다. 각 문서는 단순 요약이 아니라 원리, 코드, 실험, 해석과 설명 연습을 포함한다.

## 3. 정보 구조

기존 Notion 프로젝트 페이지를 최상위 허브로 사용하고 다음 하위 페이지를 생성한다.

1. `10일 OpenCV 심화 학습 과정`
2. `Day 1 - 이미지 데이터와 OpenCV 기초`
3. `Day 2 - 이미지 상태 진단`
4. `Day 3 - 밝기와 대비 전처리`
5. `Day 4 - 노이즈와 필터링`
6. `Day 5 - 에지·임계처리·형태학`
7. `Day 6 - 전처리 파이프라인과 추천 점수`
8. `Day 7 - OpenCV 특징 추출`
9. `Day 8 - OpenCV 분류기`
10. `Day 9 - 평가와 재현성`
11. `Day 10 - 프로젝트 전체 설명과 실전 대응`
12. `OpenCV 기술 Q&A`
13. `프로젝트·면접 질문과 모범 답안`
14. `실습 과제와 해설`
15. `진도 및 설명 능력 체크리스트`

`10일 OpenCV 심화 학습 과정`은 열 개 Day 페이지를 순서대로 연결하는 목차와 선수 관계를 제공한다. Q&A와 면접 문서는 학습 일정에 포함하지 않고 독립 참고 자료로 둔다.

## 4. 메인 포트폴리오 허브 개편

기존 사례 연구의 기술적 내용과 검증 수치는 유지한다. 상단은 채용 담당자용으로 압축하고, 하단에 학습 허브를 추가한다.

필수 구성은 다음과 같다.

- 프로젝트 핵심 메시지와 해결 문제
- 실제 Streamlit 합성 샘플 화면
- 단일 이미지 Advisor와 Dataset Benchmark의 역할 차이
- OpenCV 기술 역량 증거표
- MVTec tile 117장·6클래스 사례와 Accuracy 0.804, Macro F1 0.789
- 공식 MVTec anomaly-detection 지표가 아니라는 제한
- 원본 파이프라인이 이긴 실패·성공 해석
- GitHub, 6페이지 PDF, 로컬 정본 문서 링크
- `학습 허브` 섹션과 10일 과정·Q&A·면접·실습·체크리스트 링크

메인 페이지는 학습 내용을 모두 복제하지 않는다. 개념 설명은 각 하위 페이지로 연결해 포트폴리오 가독성을 유지한다.

## 5. 공통 Day 페이지 템플릿

각 Day 페이지는 다음 순서로 작성한다.

1. **오늘 답해야 할 핵심 질문**: 학습 후 설명할 수 있어야 하는 질문 3~5개.
2. **개념과 원리**: 정의, 처리 과정, 필요한 수식과 이미지 데이터 관점.
3. **OpenCV API와 파라미터**: 함수 시그니처, 주요 인자, 값 변화의 효과.
4. **언제 사용하고 피하는가**: 적합한 입력 조건, 손실되는 정보, 실패 징후.
5. **프로젝트 코드 연결**: GitHub `main`의 실제 구현·설정·테스트 링크와 코드 흐름 해설.
6. **직접 실험**: 합성 샘플 또는 사용자가 준비한 이미지로 수행할 단계.
7. **예상 결과와 해석**: 단순 정답이 아니라 관찰할 수치·시각 변화·경고.
8. **자주 하는 실수와 디버깅**: 잘못된 dtype·색 공간·커널·fold 처리 등.
9. **본인 말로 설명하기**: 1분 답변과 심화 답변 예시.
10. **완료 기준**: 이해, 구현, 해석, 설명의 네 수준 체크리스트.

코드 블록은 전체 모듈 복사가 아니라 핵심 알고리즘을 이해하는 최소 예제로 작성한다. 모든 실습은 실행 명령, 입력, 예상 산출물과 실패 시 확인 항목을 포함한다.

## 6. 10일 상세 범위

### Day 1 - 이미지 데이터와 OpenCV 기초

- 이미지가 `height × width × channel` 배열로 표현되는 방식
- `uint8`, `float32`, 범위와 overflow·clipping
- BGR/RGB 차이와 표시 오류
- Grayscale, HSV, LAB의 채널 의미와 변환 비용
- 유니코드 안전 이미지 입출력, 파일 검증, resize와 interpolation
- 실습: 합성 타일의 dtype·shape·채널을 확인하고 색 공간 왕복 오차 비교
- 프로젝트 연결: `io.py`, `transforms.py`, 관련 테스트

### Day 2 - 이미지 상태 진단

- 평균 밝기, 전역·국소 대비, 엔트로피
- Laplacian variance 기반 선명도
- median residual 기반 노이즈 추정의 가정
- 조명 불균일, edge density·continuity, 채도 분산, clipping ratio
- 지표 하나로 품질을 단정할 수 없는 이유
- 실습: 원본과 변형 이미지의 진단값을 표로 비교하고 변화 원인을 설명
- 프로젝트 연결: `diagnostics.py`, `models.py`, 진단 테스트

### Day 3 - 밝기와 대비 전처리

- min-max normalize와 입력 범위
- 감마 변환의 수식과 밝기 변화
- 전역 Histogram Equalization과 CLAHE 차이
- CLAHE `clipLimit`와 `tileGridSize`: 큰 grid 개수는 더 작은 타일과 더 국소적인 처리
- LAB L 채널만 조정하는 이유와 색상 관계 보존
- 실습: gamma·global HE·LAB CLAHE의 전후 진단과 histogram 비교
- 프로젝트 연결: `transforms.py`, pipeline YAML, 변환 테스트

### Day 4 - 노이즈와 필터링

- convolution, kernel size, border 처리
- Gaussian blur와 가우시안 노이즈 가정
- Median filter와 impulse noise
- Bilateral filter의 공간·색상 거리와 edge 보존
- 필터가 질감과 분류 특징을 손실시키는 방식
- 실습: 서로 다른 합성 노이즈에 세 필터를 적용하고 노이즈·선명도·시간 비교
- 프로젝트 연결: 필터 변환, 점수 경고, 과도한 평활화 테스트

### Day 5 - 에지·임계처리·형태학

- Sobel, Scharr, Laplacian의 미분 관점
- Canny의 blur·gradient·NMS·double threshold·hysteresis 흐름
- global·adaptive·Otsu threshold
- erosion, dilation, opening, closing, gradient, top-hat, black-hat
- contour와 connected components의 차이
- 실습: 합성 결함을 분리하고 morphology 전후 component 수·면적 비교
- 프로젝트 연결: `ui/technique_explorer.py`, texture feature, 관련 테스트

### Day 6 - 전처리 파이프라인과 추천 점수

- 단일 연산과 단계형 pipeline의 차이
- 순서가 결과에 미치는 영향
- YAML 기반 후보·프로필 정의
- 진단 변화, 프로필 가중치와 Top 3 점수 구성
- 휴리스틱 점수가 정확도나 일반화 성능이 아닌 이유
- clipping·edge·oversmoothing·color-loss 경고
- 실습: 파이프라인 순서를 교환하고 점수 구성 요소와 경고 변화 해석
- 프로젝트 연결: `pipelines.py`, `scoring.py`, `services.py`, config와 테스트

### Day 7 - OpenCV 특징 추출

- HSV/LAB histogram과 normalization
- HOG cell·block·bin과 shape 제약
- Sobel·Laplacian·Gabor texture statistics
- 특징 결합과 스케일 차이
- SIFT keypoint·descriptor·Bag of Words 개념
- SIFT vocabulary를 fold-local로 학습해야 하는 이유와 현재 미통합 범위
- 실습: 색상·HOG·texture·combined 특징 차원과 분류 영향 비교
- 프로젝트 연결: `features.py`, 특징 설정과 테스트

### Day 8 - OpenCV 분류기

- `cv2.ml` 데이터 형태와 `float32` 요구
- kNN 거리 기반 분류, K의 효과와 비용
- SVM margin·kernel·C·gamma
- RTrees ensemble·depth·tree count·feature importance 한계
- 데이터 크기와 특징 형태에 따른 선택 기준
- 실습: 같은 fold와 특징에서 SVM·kNN·RTrees의 지표·시간 비교
- 프로젝트 연결: `classifiers.py`, benchmark config와 테스트

### Day 9 - 평가와 재현성

- train/test split과 stratified K-fold
- fold-local scaling과 데이터 누수
- accuracy, precision, recall, F1, Macro F1
- confusion matrix를 실제 행·예측 열로 해석
- 클래스 불균형과 평균 방식
- seed, OpenCV version, config hash, sample checksum과 보고서 재현성
- 실습: 잘못된 전체 데이터 scaling과 fold-local scaling 절차 비교
- 프로젝트 연결: `datasets.py`, `evaluation.py`, `reports.py`, 증거 JSON

### Day 10 - 프로젝트 전체 설명과 실전 대응

- Streamlit·CLI·service layer·진단·pipeline·평가·report 아키텍처
- 단일 이미지 입력부터 Top 3 추천까지의 데이터 흐름
- 데이터셋 benchmark와 리더보드 생성 흐름
- Original + RTrees가 이긴 결과의 기술적 해석
- 공식 anomaly detection과 현재 6-class 실험의 경계
- 프로젝트 한계와 다음 실험: ablation, 다른 seed·클래스, fold-local SIFT, GT 평가
- 실습: 5분 포트폴리오 설명, 15분 기술 발표, 코드 일부 재구현
- 완료 산출물: 발표 스크립트, 예상 후속 질문, 한계와 개선안

## 7. 독립 참고 문서

### OpenCV 기술 Q&A

기술을 빠르게 복습하기 위한 개념형 Q&A다. 학습 일차와 분리하며 다음 카테고리로 구성한다.

- 이미지 표현·색 공간
- 진단 지표
- 밝기·대비
- 필터·노이즈
- 에지·임계처리·형태학
- 특징 추출
- 분류기
- 평가·재현성
- 프로젝트 설계·한계

각 답변은 `한 문장 답`, `상세 설명`, `프로젝트 근거`, `주의할 오해`를 포함한다. 최소 50문항을 목표로 한다.

### 프로젝트·면접 질문과 모범 답안

면접형 질문은 암기형 Q&A와 분리한다. 최소 35문항으로 구성하며 다음 유형을 포함한다.

- 프로젝트 목적과 요구사항 변화
- 기술 선택과 대안 비교
- 실패 결과와 원본 pipeline 승리 해석
- 성능 평가·데이터 누수·재현성
- 코드 구조와 테스트 전략
- 산업 이미지 적용 시 추가 검증
- 한계, 우선순위와 다음 실험

각 문항에는 `핵심 30초 답변`, `2분 심화 답변`, `근거 코드·결과`, `추가 질문`을 제공한다.

### 실습 과제와 해설

최소 30개 과제를 세 수준으로 구분한다.

- Guided: 안내된 API와 파라미터로 실행
- Analytical: 여러 결과를 수치와 시각 변화로 비교
- Reimplementation: 진단·전처리·특징·평가 일부를 빈 화면에서 구현

각 과제는 문제, 선수 지식, 입력, 요구 산출물, 힌트, 해설, 평가 기준을 포함한다. MVTec 원본 이미지는 Notion이나 GitHub에 복제하지 않고 합성 샘플 또는 사용자 로컬 데이터 경로를 사용한다.

### 진도 및 설명 능력 체크리스트

각 Day와 핵심 기술을 다음 네 수준으로 평가한다.

1. 코드를 읽고 동작을 찾을 수 있다.
2. API와 파라미터 선택 이유를 설명할 수 있다.
3. 작은 단위로 다시 구현하고 결과를 해석할 수 있다.
4. 한계와 대안을 포함해 면접에서 설명할 수 있다.

체크 여부뿐 아니라 본인이 작성한 설명, 실행 결과 링크와 재학습 항목을 기록할 공간을 둔다.

## 8. 정본과 동기화

Notion만 수정하면 내용이 소스와 분리되므로 모든 페이지의 Markdown 정본을 저장소에 함께 둔다.

- `docs/learning-10day/README.md`
- `docs/learning-10day/day-01-*.md`부터 `day-10-*.md`
- `docs/learning-10day/technical-qa.md`
- `docs/learning-10day/interview-qa.md`
- `docs/learning-10day/exercises.md`
- `docs/learning-10day/progress-checklist.md`

메인 Notion 포트폴리오 원문도 `docs/portfolio/notion-case-study.md`에서 학습 허브 링크 구조를 반영한다. 기존 28일 학습팩은 삭제하지 않고 보관하되, Notion에서는 새 10일 심화 과정만 기본 경로로 제공한다.

## 9. 검증 기준

### 구조 검증

- 메인 허브와 15개 학습·참고 페이지가 생성된다.
- 10일 인덱스가 Day 1~10을 순서대로 연결한다.
- Q&A와 면접 질문은 10일 일정 밖의 독립 페이지다.
- 모든 Notion 페이지는 상위 허브로 돌아가는 링크를 가진다.

### 내용 검증

- 모든 Day가 공통 10개 섹션과 완료 기준을 포함한다.
- 최소 50개 기술 Q&A, 35개 면접 질문, 30개 실습을 제공한다.
- OpenCV API·파라미터·코드 경로가 현재 저장소 구현과 일치한다.
- 휴리스틱 추천과 분류 성능을 혼동하지 않는다.
- MVTec 수치와 실험 범위가 `benchmark-evidence.json`과 일치한다.

### 링크와 공개 안전성

- GitHub 링크는 병합된 `main`의 실제 파일을 가리킨다.
- 소스 MVTec 이미지, GT mask, 로컬 절대 경로, 비밀정보를 포함하지 않는다.
- 합성 샘플과 공개 가능한 파생 결과만 사용한다.
- Notion 페이지를 가져와 제목, 섹션, 핵심 수치와 상호 링크를 검증한다.

### 학습 완료 기준

사용자가 Day 10 완료 후 다음을 수행할 수 있어야 한다.

- 프로젝트를 5분과 15분 버전으로 설명한다.
- 임의 이미지에 적용할 전처리 후보와 근거를 제시한다.
- 후보의 적절성을 진단 수치와 레이블 평가로 구분해 검증한다.
- 주요 OpenCV 연산·특징·분류기·평가 코드의 일부를 다시 구현한다.
- 실패 결과를 숨기지 않고 데이터·특징·평가 관점에서 해석한다.

## 10. 비범위

- 딥러닝 모델 학습이나 배포
- 공식 MVTec anomaly-detection 성능 구현
- 모든 OpenCV API의 백과사전식 나열
- MVTec 원본 데이터 또는 GT mask의 업로드
- 학습 시간이나 하루 분량의 강제 제한

## 11. 구현 순서

1. 저장소의 10일 학습 정본과 계약 테스트를 작성한다.
2. 메인 Notion 포트폴리오 원문에 학습 허브를 추가한다.
3. Notion에 10일 인덱스와 Day 1~10 페이지를 생성한다.
4. 기술 Q&A, 면접 질문, 실습, 체크리스트 페이지를 생성한다.
5. 메인 허브와 모든 하위 페이지를 상호 연결한다.
6. 페이지를 다시 가져와 구조, 수치, 링크와 공개 안전성을 검증한다.
7. README에서 상세 Notion 학습 허브의 범위를 명확히 안내한다.

