import json

from test_services import make_shape_dataset

from opencv_preprocessing_advisor.datasets import discover_dataset
from opencv_preprocessing_advisor.models import TaskProfile
from opencv_preprocessing_advisor.reports import ReportWriter
from opencv_preprocessing_advisor.services import (
    BenchmarkConfig,
    BenchmarkService,
    ImageAdvisorService,
)


def test_image_report_contains_reproducibility_metadata(tmp_path, sample_bgr):
    result = ImageAdvisorService().analyze(sample_bgr, TaskProfile.AUTO)

    output = ReportWriter(tmp_path).write_image_advice(result)
    metadata = json.loads((output / "recommendations.json").read_text(encoding="utf-8"))

    assert metadata["opencv_version"]
    assert metadata["scoring_config_hash"]
    assert (output / "diagnostics.csv").exists()
    assert (output / "comparison.png").exists()
    assert len(list((output / "steps").glob("*/*.png"))) >= 3


def test_report_runs_use_distinct_directories(tmp_path, sample_bgr):
    result = ImageAdvisorService().analyze(sample_bgr, TaskProfile.AUTO)
    writer = ReportWriter(tmp_path)

    first = writer.write_image_advice(result)
    second = writer.write_image_advice(result)

    assert first != second
    assert first.exists()
    assert second.exists()


def test_benchmark_report_contains_detailed_metrics_and_confusion_matrices(tmp_path):
    manifest = discover_dataset(make_shape_dataset(tmp_path / "dataset"))
    result = BenchmarkService().run(
        manifest,
        BenchmarkConfig(
            pipeline_ids=("original",),
            feature_profiles=("shape",),
            classifier_names=("svm",),
            folds=3,
        ),
    )

    output = ReportWriter(tmp_path / "reports").write_benchmark(result)

    assert (output / "leaderboard.csv").exists()
    assert (output / "fold_metrics.csv").exists()
    assert (output / "class_metrics.csv").exists()
    assert (output / "timings.csv").exists()
    matrices = list((output / "confusion_matrices").glob("*.png"))
    assert len(matrices) == 1
