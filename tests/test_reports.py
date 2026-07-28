import json

from opencv_preprocessing_advisor.models import TaskProfile
from opencv_preprocessing_advisor.reports import ReportWriter
from opencv_preprocessing_advisor.services import ImageAdvisorService


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
