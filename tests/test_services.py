from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from opencv_preprocessing_advisor.datasets import discover_dataset
from opencv_preprocessing_advisor.io import encode_png
from opencv_preprocessing_advisor.models import TaskProfile
from opencv_preprocessing_advisor.services import (
    DEFAULT_PIPELINES_PATH,
    DEFAULT_SCORING_PATH,
    BenchmarkConfig,
    BenchmarkService,
    ImageAdvisorService,
)


def test_image_advisor_returns_three_explained_results(sample_bgr):
    result = ImageAdvisorService().analyze(sample_bgr, TaskProfile.AUTO)

    assert len(result.recommendations) == 3
    for item in result.recommendations:
        assert item.reasons
        assert item.score_components
        assert item.pipeline_run.intermediate_images


def test_default_configs_are_installed_inside_package():
    package_directory = DEFAULT_PIPELINES_PATH.parent.parent

    assert DEFAULT_PIPELINES_PATH.is_file()
    assert DEFAULT_SCORING_PATH.is_file()
    assert package_directory.name == "opencv_preprocessing_advisor"


def test_profile_only_runs_compatible_pipelines(sample_bgr):
    service = ImageAdvisorService()
    result = service.analyze(sample_bgr, TaskProfile.COLOR)
    definitions = {definition.pipeline_id: definition for definition in service.catalog.definitions}

    assert all(
        TaskProfile.COLOR in definitions[item.pipeline_id].profiles
        for item in result.recommendations
    )


def test_image_advisor_uses_weights_from_scoring_config(tmp_path, sample_bgr):
    config = yaml.safe_load(DEFAULT_SCORING_PATH.read_text(encoding="utf-8"))
    config["profiles"]["auto"] = {"sharpness": 1.0}
    scoring_path = tmp_path / "scoring.yaml"
    scoring_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    result = ImageAdvisorService(scoring_path=scoring_path).analyze(
        sample_bgr,
        TaskProfile.AUTO,
    )

    for recommendation in result.recommendations:
        assert [item.name for item in recommendation.score_components] == ["sharpness"]
        assert recommendation.score_components[0].weight == 1.0


def make_shape_dataset(root: Path) -> Path:
    for class_name in ("circle", "square"):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for index in range(6):
            image = np.full((96, 96, 3), 30 + index, np.uint8)
            if class_name == "circle":
                cv2.circle(image, (48, 48), 20 + index, (230, 230, 230), -1)
            else:
                cv2.rectangle(
                    image,
                    (25 - index, 25 - index),
                    (70 + index, 70 + index),
                    (230, 230, 230),
                    -1,
                )
            (class_dir / f"{index}.png").write_bytes(encode_png(image))
    return root


def test_benchmark_service_ranks_requested_combinations(tmp_path):
    manifest = discover_dataset(make_shape_dataset(tmp_path / "dataset"))
    config = BenchmarkConfig(
        pipeline_ids=("original", "lab-clahe"),
        feature_profiles=("shape",),
        classifier_names=("svm", "knn"),
        folds=3,
        seed=42,
    )

    result = BenchmarkService().run(manifest, config)

    assert len(result.entries) == 4
    assert len(result.top_entries) == 2
    assert len({entry.pipeline_id for entry in result.top_entries}) == 2
    assert result.entries[0].cross_validation.folds


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("pipeline_ids", "pipeline"),
        ("feature_profiles", "feature"),
        ("classifier_names", "classifier"),
    ],
)
def test_benchmark_service_rejects_empty_comparison_dimensions(
    tmp_path,
    field,
    message,
):
    manifest = discover_dataset(make_shape_dataset(tmp_path / "dataset"))
    values = {
        "pipeline_ids": ("original",),
        "feature_profiles": ("shape",),
        "classifier_names": ("svm",),
    }
    values[field] = ()

    with pytest.raises(ValueError, match=message):
        BenchmarkService().run(manifest, BenchmarkConfig(**values))
