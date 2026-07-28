"""Atomic, reproducible report artifact generation."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import encode_png
from .services import BenchmarkResult, ImageAdviceResult


def _run_id() -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"


class ReportWriter:
    def __init__(self, output_root: Path | str) -> None:
        self.output_root = Path(output_root)

    def _atomic_directory(self, category: str) -> tuple[Path, Path]:
        parent = self.output_root / category
        parent.mkdir(parents=True, exist_ok=True)
        final = parent / _run_id()
        temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=parent))
        return temporary, final

    @staticmethod
    def _commit_directory(temporary: Path, final: Path) -> Path:
        temporary.replace(final)
        return final

    def write_image_advice(self, result: ImageAdviceResult) -> Path:
        temporary, final = self._atomic_directory("image_advisor")
        try:
            rows = [
                {
                    "scope": "original",
                    "pipeline_id": "original",
                    **asdict(result.original_diagnostics),
                }
            ]
            serialized_recommendations = []
            steps_root = temporary / "steps"
            for rank, recommendation in enumerate(result.recommendations, start=1):
                diagnostics = recommendation.pipeline_run.diagnostics_after
                if diagnostics is None:
                    raise ValueError("recommendation is missing after diagnostics")
                rows.append(
                    {
                        "scope": f"recommendation_{rank}",
                        "pipeline_id": recommendation.pipeline_id,
                        **asdict(diagnostics),
                    }
                )
                serialized_recommendations.append(
                    {
                        "rank": rank,
                        "pipeline_id": recommendation.pipeline_id,
                        "display_name_ko": recommendation.pipeline_run.display_name_ko,
                        "display_name_en": recommendation.pipeline_run.display_name_en,
                        "suitability_score": recommendation.suitability_score,
                        "reasons": list(recommendation.reasons),
                        "warnings": list(recommendation.warnings),
                        "score_components": [
                            asdict(component) for component in recommendation.score_components
                        ],
                        "processing_ms": recommendation.pipeline_run.processing_ms,
                    }
                )
                pipeline_dir = steps_root / recommendation.pipeline_id
                pipeline_dir.mkdir(parents=True, exist_ok=True)
                for step_index, step in enumerate(
                    recommendation.pipeline_run.intermediate_images,
                    start=1,
                ):
                    (pipeline_dir / f"{step_index:02d}_{step.name}.png").write_bytes(
                        encode_png(step.image)
                    )

            pd.DataFrame(rows).to_csv(temporary / "diagnostics.csv", index=False)
            metadata = {
                "generated_at": datetime.now(UTC).isoformat(),
                "profile": result.profile.value,
                "opencv_version": result.opencv_version,
                "pipeline_config_hash": result.pipeline_config_hash,
                "scoring_config_hash": result.scoring_config_hash,
                "recommendations": serialized_recommendations,
            }
            (temporary / "recommendations.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._write_comparison(result, temporary / "comparison.png")
            return self._commit_directory(temporary, final)
        except Exception:
            for path in sorted(temporary.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            temporary.rmdir()
            raise

    @staticmethod
    def _write_comparison(result: ImageAdviceResult, output: Path) -> None:
        images = [result.original_image] + [
            item.pipeline_run.output_image for item in result.recommendations
        ]
        titles = ["Original"] + [
            f"#{index} {item.pipeline_run.display_name_en}\n{item.suitability_score:.1f}/100"
            for index, item in enumerate(result.recommendations, start=1)
        ]
        figure, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
        for axis, image, title in zip(axes, images, titles, strict=True):
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axis.set_title(title)
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(output, dpi=140, bbox_inches="tight")
        plt.close(figure)

    def write_benchmark(self, result: BenchmarkResult) -> Path:
        temporary, final = self._atomic_directory("benchmark")
        rows = []
        fold_rows = []
        class_rows = []
        timing_rows = []
        matrices_root = temporary / "confusion_matrices"
        matrices_root.mkdir()
        for rank, entry in enumerate(result.entries, start=1):
            entry_key = (
                f"{rank:02d}_{entry.pipeline_id}_{entry.feature_profile}_{entry.classifier_name}"
            )
            rows.append(
                {
                    "rank": rank,
                    "pipeline_id": entry.pipeline_id,
                    "feature_profile": entry.feature_profile,
                    "classifier": entry.classifier_name,
                    "mean_accuracy": entry.cross_validation.mean_accuracy,
                    "std_accuracy": entry.cross_validation.std_accuracy,
                    "mean_macro_f1": entry.cross_validation.mean_macro_f1,
                    "std_macro_f1": entry.cross_validation.std_macro_f1,
                    "preprocessing_ms": entry.preprocessing_ms,
                    "feature_extraction_ms": entry.feature_extraction_ms,
                }
            )
            for fold in entry.cross_validation.folds:
                fold_rows.append(
                    {
                        "pipeline_id": entry.pipeline_id,
                        "feature_profile": entry.feature_profile,
                        "classifier": entry.classifier_name,
                        "fold": fold.fold_index,
                        "accuracy": fold.metrics.accuracy,
                        "macro_precision": fold.metrics.macro_precision,
                        "macro_recall": fold.metrics.macro_recall,
                        "macro_f1": fold.metrics.macro_f1,
                        "fit_ms": fold.fit_ms,
                        "predict_ms": fold.predict_ms,
                    }
                )
                for metric in fold.metrics.per_class:
                    class_rows.append(
                        {
                            "pipeline_id": entry.pipeline_id,
                            "feature_profile": entry.feature_profile,
                            "classifier": entry.classifier_name,
                            "fold": fold.fold_index,
                            "class_name": result.manifest.class_names[metric.class_index],
                            "precision": metric.precision,
                            "recall": metric.recall,
                            "f1": metric.f1,
                            "support": metric.support,
                        }
                    )
            timing_rows.append(
                {
                    "pipeline_id": entry.pipeline_id,
                    "feature_profile": entry.feature_profile,
                    "classifier": entry.classifier_name,
                    "preprocessing_ms": entry.preprocessing_ms,
                    "feature_extraction_ms": entry.feature_extraction_ms,
                    "mean_fit_ms": float(
                        np.mean([fold.fit_ms for fold in entry.cross_validation.folds])
                    ),
                    "mean_predict_ms": float(
                        np.mean([fold.predict_ms for fold in entry.cross_validation.folds])
                    ),
                }
            )
            aggregate_matrix = np.sum(
                [fold.metrics.confusion_matrix for fold in entry.cross_validation.folds],
                axis=0,
            )
            self._write_confusion_matrix(
                aggregate_matrix,
                result.manifest.class_names,
                matrices_root / f"{entry_key}.png",
                title=(f"{entry.pipeline_id} | {entry.feature_profile} | {entry.classifier_name}"),
            )
        pd.DataFrame(rows).to_csv(temporary / "leaderboard.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(temporary / "fold_metrics.csv", index=False)
        pd.DataFrame(class_rows).to_csv(temporary / "class_metrics.csv", index=False)
        pd.DataFrame(timing_rows).to_csv(temporary / "timings.csv", index=False)
        metadata = {
            "generated_at": datetime.now(UTC).isoformat(),
            "opencv_version": result.opencv_version,
            "pipeline_config_hash": result.pipeline_config_hash,
            "dataset_root": str(result.manifest.root),
            "class_names": list(result.manifest.class_names),
            "sample_count": len(result.manifest.samples),
            "seed": result.config.seed,
            "folds": result.config.folds,
        }
        (temporary / "run_config.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self._commit_directory(temporary, final)

    @staticmethod
    def _write_confusion_matrix(
        matrix: np.ndarray,
        class_names: tuple[str, ...],
        output: Path,
        title: str,
    ) -> None:
        figure, axis = plt.subplots(figsize=(max(5, len(class_names)), 4.5))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set(
            title=title,
            xlabel="Predicted",
            ylabel="Actual",
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
        threshold = float(matrix.max()) / 2 if matrix.size else 0.0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    int(matrix[row, column]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        figure.savefig(output, dpi=140, bbox_inches="tight")
        plt.close(figure)
