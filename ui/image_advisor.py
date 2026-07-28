"""Single-image recommendation page."""

from dataclasses import asdict
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import streamlit as st

from opencv_preprocessing_advisor.io import decode_image_bytes, encode_png
from opencv_preprocessing_advisor.models import TaskProfile
from opencv_preprocessing_advisor.services import ImageAdvisorService

PROFILE_LABELS = {
    "자동 / 일반 분류": TaskProfile.AUTO,
    "형태 중심": TaskProfile.SHAPE,
    "색상 중심": TaskProfile.COLOR,
    "질감 중심": TaskProfile.TEXTURE,
}


@st.cache_resource
def _service() -> ImageAdvisorService:
    return ImageAdvisorService()


def _step_zip(result) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("original.png", encode_png(result.original_image))
        for rank, recommendation in enumerate(result.recommendations, start=1):
            for step_index, step in enumerate(
                recommendation.pipeline_run.intermediate_images,
                start=1,
            ):
                archive.writestr(
                    f"{rank}_{recommendation.pipeline_id}/{step_index:02}_{step.name}.png",
                    encode_png(step.image),
                )
    return buffer.getvalue()


def render() -> None:
    st.title("이미지 전처리 추천")
    st.write("이미지 한 장을 진단하고 목적별 전처리 파이프라인 3개를 비교합니다.")
    uploaded = st.file_uploader(
        "JPG, PNG, BMP 또는 TIFF 이미지",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    )
    with st.form("image-advisor-form"):
        profile_label = st.selectbox("분류 관점", list(PROFILE_LABELS))
        submitted = st.form_submit_button("분석 실행", type="primary")
    if submitted:
        if uploaded is None:
            st.error("먼저 이미지를 업로드하세요.")
        else:
            try:
                image = decode_image_bytes(uploaded.getvalue())
                with st.spinner("OpenCV 진단과 파이프라인 비교 중..."):
                    st.session_state["image_advice"] = _service().analyze(
                        image,
                        PROFILE_LABELS[profile_label],
                    )
            except ValueError as error:
                st.error(str(error))

    result = st.session_state.get("image_advice")
    if result is None:
        return

    st.subheader("원본 진단 수치")
    original_frame = pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in asdict(result.original_diagnostics).items()
        ]
    )
    st.dataframe(original_frame, hide_index=True, use_container_width=True)

    tabs = st.tabs(
        [
            f"#{index} {item.pipeline_run.display_name_ko}"
            for index, item in enumerate(result.recommendations, start=1)
        ]
    )
    for tab, recommendation in zip(tabs, result.recommendations, strict=True):
        with tab:
            left, right = st.columns(2)
            left.image(
                result.original_image, channels="BGR", caption="원본", use_container_width=True
            )
            right.image(
                recommendation.pipeline_run.output_image,
                channels="BGR",
                caption=f"결과 — {recommendation.suitability_score:.1f}/100",
                use_container_width=True,
            )
            st.caption("휴리스틱 전처리 적합도이며 분류 정확도 또는 성공 확률이 아닙니다.")
            st.markdown("**추천 근거**")
            for reason in recommendation.reasons:
                st.write(f"- {reason}")
            if recommendation.warnings:
                st.markdown("**주의점**")
                for warning in recommendation.warnings:
                    st.warning(warning)
            components = pd.DataFrame(
                [asdict(component) for component in recommendation.score_components]
            )
            st.bar_chart(components.set_index("name")["weighted_value"])
            with st.expander("단계별 이미지와 파라미터"):
                for step in recommendation.pipeline_run.intermediate_images:
                    st.image(
                        step.image,
                        channels="BGR",
                        caption=f"{step.name} — {step.params}",
                        use_container_width=True,
                    )

    st.download_button(
        "단계별 이미지 ZIP 다운로드",
        data=_step_zip(result),
        file_name="opencv_preprocessing_steps.zip",
        mime="application/zip",
    )
    st.download_button(
        "진단 CSV 다운로드",
        data=original_frame.to_csv(index=False).encode("utf-8-sig"),
        file_name="image_diagnostics.csv",
        mime="text/csv",
    )
