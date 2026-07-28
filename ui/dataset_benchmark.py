"""Classification dataset benchmark page."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import streamlit as st

from opencv_preprocessing_advisor.datasets import discover_dataset, extract_dataset_zip
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


def _aggregate_confusion(entry) -> np.ndarray:
    return np.sum(
        [fold.metrics.confusion_matrix for fold in entry.cross_validation.folds],
        axis=0,
    )


def _per_class_metrics(entry, class_names: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for class_index, class_name in enumerate(class_names):
        metrics = [fold.metrics.per_class[class_index] for fold in entry.cross_validation.folds]
        rows.append(
            {
                "class": class_name,
                "precision_mean": np.mean([item.precision for item in metrics]),
                "recall_mean": np.mean([item.recall for item in metrics]),
                "f1_mean": np.mean([item.f1 for item in metrics]),
                "support_total": sum(item.support for item in metrics),
            }
        )
    return pd.DataFrame(rows)


def _report_zip(report: Path) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for path in sorted(report.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(report))
    return buffer.getvalue()


def render() -> None:
    st.title("분류 데이터셋 벤치마크")
    st.write("클래스별 하위 폴더를 OpenCV 특징과 분류기로 교차검증합니다.")
    source_mode = st.radio(
        "입력 방식",
        ["로컬 폴더", "ZIP 업로드"],
        horizontal=True,
    )
    dataset_path = ""
    uploaded_zip = None
    if source_mode == "로컬 폴더":
        dataset_path = st.text_input(
            "데이터셋 폴더",
            value=_suggested_dataset(),
            placeholder="C:/path/to/dataset (하위 폴더가 클래스)",
        )
        source_identity = f"path:{Path(dataset_path).resolve()}" if dataset_path else ""
    else:
        uploaded_zip = st.file_uploader(
            "클래스 폴더가 포함된 ZIP",
            type=["zip"],
            key="benchmark-zip",
        )
        source_identity = (
            f"zip:{sha256(uploaded_zip.getvalue()).hexdigest()}" if uploaded_zip is not None else ""
        )

    if st.button("1. 데이터셋 검증", disabled=not source_identity):
        try:
            if uploaded_zip is not None:
                previous = st.session_state.pop("_dataset_tempdir", None)
                if previous is not None:
                    previous.cleanup()
                temporary = TemporaryDirectory(prefix="opencv-prep-dataset-")
                try:
                    root = extract_dataset_zip(
                        uploaded_zip.getvalue(),
                        Path(temporary.name) / "dataset",
                    )
                    manifest = discover_dataset(root)
                except Exception:
                    temporary.cleanup()
                    raise
                st.session_state["_dataset_tempdir"] = temporary
            else:
                manifest = discover_dataset(Path(dataset_path))
            st.session_state["validated_manifest"] = manifest
            st.session_state["validated_source_identity"] = source_identity
            st.session_state.pop("benchmark_result", None)
            st.session_state.pop("benchmark_report", None)
        except (FileNotFoundError, ValueError, KeyError) as error:
            st.error(str(error))

    manifest = (
        st.session_state.get("validated_manifest")
        if st.session_state.get("validated_source_identity") == source_identity
        else None
    )
    if manifest is not None:
        st.success(
            f"검증 완료: {len(manifest.class_names)}개 클래스 / "
            f"{len(manifest.samples)}장 / 건너뛴 파일 "
            f"{len(manifest.skipped_files)}개"
        )
        class_counts = pd.Series([sample.class_name for sample in manifest.samples]).value_counts()
        st.dataframe(
            class_counts.rename_axis("class").reset_index(name="images"),
            hide_index=True,
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
        submitted = st.form_submit_button(
            "2. 벤치마크 실행",
            type="primary",
            disabled=manifest is None,
        )
    if submitted:
        try:
            if manifest is None:
                raise ValueError("벤치마크 전에 데이터셋 검증을 완료하세요")
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
    actual_folds = len(best.cross_validation.folds)
    if actual_folds < result.config.folds:
        st.warning(
            f"가장 작은 클래스의 표본 수에 맞춰 {result.config.folds}-fold 요청을 "
            f"{actual_folds}-fold로 줄였습니다."
        )

    st.subheader("1위 조합 상세")
    timing_left, timing_middle, timing_right = st.columns(3)
    timing_left.metric("전처리", f"{best.preprocessing_ms:.1f} ms")
    timing_middle.metric("특징 추출", f"{best.feature_extraction_ms:.1f} ms")
    timing_right.metric(
        "평균 학습 + 예측",
        (
            f"{np.mean([fold.fit_ms + fold.predict_ms for fold in best.cross_validation.folds]):.1f} ms"
        ),
    )
    class_frame = _per_class_metrics(best, result.manifest.class_names)
    st.markdown("**클래스별 지표**")
    st.dataframe(class_frame, hide_index=True, use_container_width=True)
    confusion = pd.DataFrame(
        _aggregate_confusion(best),
        index=[f"actual:{name}" for name in result.manifest.class_names],
        columns=[f"pred:{name}" for name in result.manifest.class_names],
    )
    st.markdown("**혼동행렬 — 전체 test fold 합계**")
    st.dataframe(
        confusion.style.background_gradient(cmap="Blues"),
        use_container_width=True,
    )

    report_path = st.session_state.get("benchmark_report")
    if report_path:
        st.download_button(
            "전체 벤치마크 보고서 ZIP 다운로드",
            data=_report_zip(Path(report_path)),
            file_name="opencv_benchmark_report.zip",
            mime="application/zip",
        )
