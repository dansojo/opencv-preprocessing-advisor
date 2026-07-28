"""Methodology and limitations page."""

import pandas as pd
import streamlit as st

from opencv_preprocessing_advisor.models import TaskProfile
from opencv_preprocessing_advisor.scoring import PROFILE_WEIGHTS


def render() -> None:
    st.title("평가 방법론")
    st.markdown(
        """
        ### 단일 이미지

        모든 후보 파이프라인을 실제 적용한 뒤 밝기, 클리핑, 대비, Entropy,
        Laplacian 선명도, 노이즈, 조명 균일성, 엣지, 색상 지표의 변화를 측정합니다.
        적합도는 아래 공개 가중치로 계산되는 **휴리스틱**입니다.

        ### 분류 데이터셋

        클래스 폴더를 계층화 5-fold로 분리하고, 각 학습 폴드에서만 표준화를 학습합니다.
        OpenCV HOG·색상 Histogram·Gabor 통계와 `cv2.ml` SVM·kNN·RTrees를 사용합니다.
        순위 기준은 평균 Macro F1, Accuracy, 처리 시간 순입니다.

        ### 해석 제한

        - 한 이미지의 화질 지표는 분류 정확도를 보장하지 않습니다.
        - 벤치마크 순위는 입력 데이터셋·특징·분류기·분할에 종속됩니다.
        - 높은 대비와 선명도는 불필요한 배경이나 노이즈도 강조할 수 있습니다.
        """
    )
    rows = []
    for profile, weights in PROFILE_WEIGHTS.items():
        for metric, weight in weights.items():
            rows.append(
                {
                    "profile": profile.value,
                    "metric": metric,
                    "weight": weight,
                }
            )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    selected = st.selectbox("프로필 가중치 보기", [profile.value for profile in TaskProfile])
    st.bar_chart(
        pd.Series(PROFILE_WEIGHTS[TaskProfile(selected)], name="weight"),
        horizontal=True,
    )
