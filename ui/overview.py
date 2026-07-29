"""Project overview page."""

import streamlit as st


def render() -> None:
    st.title("OpenCV Preprocessing Advisor")
    st.caption("Explainable preprocessing recommendations for image classification")
    st.markdown(
        """
        이미지의 밝기·대비·노이즈·선명도·색상·경계 특성을 OpenCV로 측정하고,
        목적에 맞는 전처리 파이프라인 **상위 3개**를 수치와 근거로 설명합니다.

        #### 두 가지 사용 흐름

        1. **단일 이미지 추천** — GT나 라벨 없이 이미지 상태를 진단하고 전처리 후보를 탐색합니다.
        2. **분류 데이터셋 벤치마크** — 클래스 폴더 데이터셋에서 OpenCV 특징·분류기로
           전처리 효과를 교차검증합니다.

        #### 이 도구가 하지 않는 것

        - 단일 이미지 점수를 분류 정확도나 확률로 표현하지 않습니다.
        - 딥러닝 또는 외부 비전 모델로 이미지를 평가하지 않습니다.
        - 모든 데이터에 보편적으로 최적인 전처리가 있다고 주장하지 않습니다.
        """
    )
    st.info(
        "추천 점수는 관찰된 화질·특징 변화에 대한 휴리스틱입니다. "
        "실제 분류 효과는 데이터셋 벤치마크에서 확인하세요."
    )
