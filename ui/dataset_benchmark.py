"""Classification dataset benchmark page."""

from pathlib import Path

import pandas as pd
import streamlit as st

from opencv_preprocessing_advisor.datasets import discover_dataset
from opencv_preprocessing_advisor.pipelines import PipelineCatalog
from opencv_preprocessing_advisor.reports import ReportWriter
from opencv_preprocessing_advisor.services import (
    DEFAULT_PIPELINES_PATH,
    BenchmarkConfig,
    BenchmarkService,
)


def _suggested_dataset() -> str:
    candidate = Path.home() / "Desktop" / "mvtec_anomaly_detection" / "tile" / "test"
    return str(candidate) if candidate.is_dir() else ""


@st.cache_resource
def _catalog() -> PipelineCatalog:
    return PipelineCatalog.from_yaml(DEFAULT_PIPELINES_PATH)


@st.cache_resource
def _service() -> BenchmarkService:
    return BenchmarkService()


def _leaderboard(result) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rank": index,
                "pipeline": entry.pipeline_id,
                "features": entry.feature_profile,
                "classifier": entry.classifier_name,
                "accuracy_mean": entry.cross_validation.mean_accuracy,
                "accuracy_std": entry.cross_validation.std_accuracy,
                "macro_f1_mean": entry.cross_validation.mean_macro_f1,
                "macro_f1_std": entry.cross_validation.std_macro_f1,
                "preprocess_ms": entry.preprocessing_ms,
                "feature_ms": entry.feature_extraction_ms,
            }
            for index, entry in enumerate(result.entries, start=1)
        ]
    )


def render() -> None:
    st.title("분류 데이터셋 벤치마크")
    st.write("클래스별 하위 폴더를 OpenCV 특징과 분류기로 교차검증합니다.")
    dataset_path = st.text_input(
        "데이터셋 폴더",
        value=_suggested_dataset(),
        placeholder="C:/path/to/dataset (하위 폴더가 클래스)",
    )
    pipeline_options = ["original", *_catalog().pipeline_ids]
    with st.form("benchmark-form"):
        pipelines = st.multiselect(
            "비교 파이프라인",
            pipeline_options,
            default=["original", "lab-clahe", "clahe-bilateral"],
        )
        features = st.multiselect(
            "특징 프로필",
            ["color", "shape", "texture", "combined"],
            default=["combined"],
        )
        classifiers = st.multiselect(
            "OpenCV 분류기",
            ["svm", "knn", "rtrees"],
            default=["svm"],
        )
        folds = st.slider("교차검증 폴드", min_value=2, max_value=10, value=5)
        submitted = st.form_submit_button("벤치마크 실행", type="primary")
    if submitted:
        try:
            manifest = discover_dataset(Path(dataset_path))
            st.write(
                f"{len(manifest.class_names)}개 클래스 / {len(manifest.samples)}장 / "
                f"건너뛴 파일 {len(manifest.skipped_files)}개"
            )
            config = BenchmarkConfig(
                pipeline_ids=tuple(pipelines),
                feature_profiles=tuple(features),
                classifier_names=tuple(classifiers),
                folds=folds,
                seed=42,
            )
            with st.spinner("전처리·특징 추출·교차검증 중..."):
                result = _service().run(manifest, config)
                report = ReportWriter(Path("outputs")).write_benchmark(result)
                st.session_state["benchmark_result"] = result
                st.session_state["benchmark_report"] = report
        except (FileNotFoundError, ValueError, KeyError) as error:
            st.error(str(error))

    result = st.session_state.get("benchmark_result")
    if result is None:
        return
    frame = _leaderboard(result)
    st.subheader("파이프라인 순위")
    st.dataframe(frame, hide_index=True, use_container_width=True)
    st.caption(f"리포트 저장 위치: {st.session_state.get('benchmark_report')}")
    st.download_button(
        "Leaderboard CSV 다운로드",
        data=frame.to_csv(index=False).encode("utf-8-sig"),
        file_name="benchmark_leaderboard.csv",
        mime="text/csv",
    )
    best = result.top_entries[0]
    st.success(
        f"1위: {best.pipeline_id} / {best.feature_profile} / {best.classifier_name} — "
        f"Macro F1 {best.cross_validation.mean_macro_f1:.3f}"
    )
