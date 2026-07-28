"""Application services shared by CLI and Streamlit."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .datasets import DatasetManifest, stratified_folds
from .diagnostics import analyze_image
from .evaluation import CrossValidationResult, cross_validate
from .features import (
    ColorHistogramExtractor,
    CombinedExtractor,
    HOGExtractor,
    TextureStatsExtractor,
)
from .io import decode_image
from .models import ImageDiagnostics, Recommendation, TaskProfile
from .pipelines import PipelineCatalog
from .scoring import ScoredPipeline, rank_recommendations

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINES_PATH = PROJECT_ROOT / "config" / "pipelines.yaml"
DEFAULT_SCORING_PATH = PROJECT_ROOT / "config" / "scoring.yaml"


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ImageAdviceResult:
    profile: TaskProfile
    original_image: np.ndarray
    original_diagnostics: ImageDiagnostics
    recommendations: tuple[Recommendation, ...]
    opencv_version: str
    pipeline_config_hash: str
    scoring_config_hash: str


@dataclass(frozen=True)
class BenchmarkConfig:
    pipeline_ids: tuple[str, ...] = ("original", "lab-clahe", "clahe-bilateral")
    feature_profiles: tuple[str, ...] = ("combined",)
    classifier_names: tuple[str, ...] = ("svm", "knn", "rtrees")
    folds: int = 5
    seed: int = 42


@dataclass(frozen=True)
class BenchmarkEntry:
    pipeline_id: str
    feature_profile: str
    classifier_name: str
    cross_validation: CrossValidationResult
    preprocessing_ms: float
    feature_extraction_ms: float


@dataclass(frozen=True)
class BenchmarkResult:
    manifest: DatasetManifest
    config: BenchmarkConfig
    entries: tuple[BenchmarkEntry, ...]
    top_entries: tuple[BenchmarkEntry, ...]
    opencv_version: str
    pipeline_config_hash: str


class ImageAdvisorService:
    def __init__(
        self,
        catalog: PipelineCatalog | None = None,
        pipeline_path: Path = DEFAULT_PIPELINES_PATH,
        scoring_path: Path = DEFAULT_SCORING_PATH,
    ) -> None:
        self.pipeline_path = pipeline_path
        self.scoring_path = scoring_path
        self.catalog = catalog or PipelineCatalog.from_yaml(pipeline_path)

    def analyze(self, image, profile: TaskProfile = TaskProfile.AUTO) -> ImageAdviceResult:
        before = analyze_image(image)
        candidates: list[ScoredPipeline] = []
        definitions = {item.pipeline_id: item for item in self.catalog.definitions}
        for pipeline_id in self.catalog.pipeline_ids:
            definition = definitions[pipeline_id]
            if profile not in definition.profiles and TaskProfile.AUTO not in definition.profiles:
                continue
            run = self.catalog.run(pipeline_id, image)
            after = analyze_image(run.output_image)
            run = replace(run, diagnostics_after=after)
            candidates.append(ScoredPipeline(run=run, before=before, after=after))
        recommendations = rank_recommendations(candidates, profile, limit=3)
        return ImageAdviceResult(
            profile=profile,
            original_image=image.copy(),
            original_diagnostics=before,
            recommendations=tuple(recommendations),
            opencv_version=cv2.__version__,
            pipeline_config_hash=_file_hash(self.pipeline_path),
            scoring_config_hash=_file_hash(self.scoring_path),
        )


def _feature_extractor(name: str):
    extractors = {
        "color": ColorHistogramExtractor,
        "shape": HOGExtractor,
        "texture": TextureStatsExtractor,
        "combined": CombinedExtractor,
    }
    try:
        return extractors[name]()
    except KeyError as error:
        raise ValueError(f"unsupported feature profile: {name}") from error


class BenchmarkService:
    def __init__(
        self,
        catalog: PipelineCatalog | None = None,
        pipeline_path: Path = DEFAULT_PIPELINES_PATH,
    ) -> None:
        self.pipeline_path = pipeline_path
        self.catalog = catalog or PipelineCatalog.from_yaml(pipeline_path)

    def run(
        self,
        manifest: DatasetManifest,
        config: BenchmarkConfig,
    ) -> BenchmarkResult:
        source_images = [decode_image(sample.path) for sample in manifest.samples]
        labels = manifest.labels
        folds = stratified_folds(labels, config.folds, config.seed)
        entries: list[BenchmarkEntry] = []
        for pipeline_id in config.pipeline_ids:
            processed_images = []
            preprocessing_ms = 0.0
            for image in source_images:
                if pipeline_id == "original":
                    processed_images.append(image.copy())
                else:
                    run = self.catalog.run(pipeline_id, image)
                    processed_images.append(run.output_image)
                    preprocessing_ms += run.processing_ms
            for feature_profile in config.feature_profiles:
                extractor = _feature_extractor(feature_profile)
                started = perf_counter()
                features = extractor.transform(processed_images)
                feature_ms = (perf_counter() - started) * 1000.0
                for classifier_name in config.classifier_names:
                    result = cross_validate(
                        features,
                        labels,
                        folds,
                        classifier_name,
                        config.seed,
                    )
                    entries.append(
                        BenchmarkEntry(
                            pipeline_id=pipeline_id,
                            feature_profile=feature_profile,
                            classifier_name=classifier_name,
                            cross_validation=result,
                            preprocessing_ms=preprocessing_ms,
                            feature_extraction_ms=feature_ms,
                        )
                    )
        ranked = sorted(
            entries,
            key=lambda entry: (
                -entry.cross_validation.mean_macro_f1,
                -entry.cross_validation.mean_accuracy,
                entry.preprocessing_ms + entry.feature_extraction_ms,
                entry.pipeline_id,
                entry.feature_profile,
                entry.classifier_name,
            ),
        )
        return BenchmarkResult(
            manifest=manifest,
            config=config,
            entries=tuple(ranked),
            top_entries=tuple(ranked[:3]),
            opencv_version=cv2.__version__,
            pipeline_config_hash=_file_hash(self.pipeline_path),
        )
