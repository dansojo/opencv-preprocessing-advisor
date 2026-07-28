# Troubleshooting

## OpenCV 기능이 없다는 오류

`opencv-python` 5.0.0 계열에서는 이 프로젝트가 사용하는 일부 Python binding(`HOGDescriptor`, `BOWKMeansTrainer`, `cv2.ml`)이 제공되지 않는 환경이 확인되었습니다.

```powershell
python -m pip install "opencv-python>=4.10,<5"
python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'HOGDescriptor'), hasattr(cv2, 'ml'))"
```

프로젝트 의존성도 OpenCV 4.x로 제한되어 있습니다.

## Matplotlib 실행 시 Tk 오류

보고서 생성은 GUI 없는 환경에서도 실행할 수 있도록 `Agg` backend를 명시합니다. 다른 모듈에서 `matplotlib.pyplot`을 먼저 import하면 backend 선택이 달라질 수 있으므로 보고서 모듈보다 먼저 pyplot을 import하지 마세요.

## Streamlit duplicate URL pathname

여러 페이지의 callable 함수명이 모두 `render`이면 Streamlit이 같은 자동 URL을 만들 수 있습니다. `app.py`는 각 `st.Page`에 고유한 `url_path`를 지정합니다. 페이지를 추가할 때도 고유 경로를 지정하세요.

## 데이터셋을 찾지 못함

데이터셋은 다음 구조여야 하며 클래스마다 유효한 이미지가 최소 5장 필요합니다.

```text
dataset/
├─ class_a/*.png
└─ class_b/*.png
```

지원 확장자는 JPG, JPEG, PNG, BMP, TIF, TIFF입니다. 깨진 이미지는 manifest에서 제외되며, 클래스 수가 2개 미만이면 평가를 시작하지 않습니다.

## MVTec tile 결과 해석

현재 사례 연구는 anomaly localization이나 binary anomaly detection이 아닙니다. `test` 하위의 상태 폴더를 각각 분류 클래스로 간주했으며 GT mask는 사용하지 않았습니다. 따라서 결과를 MVTec AD 공식 성능과 비교하면 안 됩니다.

SVM이 다수 클래스로 치우치거나 특정 클래스 F1이 0이 될 수 있습니다. 이 프로젝트는 그런 실패를 숨기지 않고 fold 지표, 클래스별 지표, 혼동행렬로 노출합니다. 분류기 튜닝이나 특징 선택을 추가할 경우 동일 fold와 seed를 유지해야 공정하게 비교할 수 있습니다.

