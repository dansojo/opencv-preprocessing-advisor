"""Streamlit entry point."""

import streamlit as st

from ui import dataset_benchmark, image_advisor, methodology, overview, technique_explorer

st.set_page_config(
    page_title="OpenCV Preprocessing Advisor",
    page_icon="🔬",
    layout="wide",
)

navigation = st.navigation(
    [
        st.Page(
            overview.render,
            title="프로젝트 개요",
            icon="🏠",
            url_path="overview",
            default=True,
        ),
        st.Page(
            image_advisor.render,
            title="이미지 전처리 추천",
            icon="🧭",
            url_path="image-advisor",
        ),
        st.Page(
            dataset_benchmark.render,
            title="데이터셋 벤치마크",
            icon="📊",
            url_path="dataset-benchmark",
        ),
        st.Page(
            technique_explorer.render,
            title="기술 탐색기",
            icon="🧪",
            url_path="technique-explorer",
        ),
        st.Page(
            methodology.render,
            title="평가 방법론",
            icon="📐",
            url_path="methodology",
        ),
    ]
)
navigation.run()
