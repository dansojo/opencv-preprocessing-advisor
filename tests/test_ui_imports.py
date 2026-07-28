import importlib
import json
from io import BytesIO
from zipfile import ZipFile

import pytest
from streamlit.testing.v1 import AppTest

from opencv_preprocessing_advisor.models import TaskProfile
from opencv_preprocessing_advisor.services import ImageAdvisorService
from ui.dataset_benchmark import _report_zip
from ui.image_advisor import _advice_json


@pytest.mark.parametrize(
    "module",
    [
        "ui.overview",
        "ui.image_advisor",
        "ui.dataset_benchmark",
        "ui.technique_explorer",
        "ui.methodology",
    ],
)
def test_ui_modules_import_without_starting_processing(module):
    imported = importlib.import_module(module)

    assert callable(imported.render)


def test_streamlit_entrypoint_executes_without_exception():
    app = AppTest.from_file("app.py", default_timeout=20).run()

    assert not app.exception


def test_image_advice_json_contains_reproducibility_metadata(sample_bgr):
    result = ImageAdvisorService().analyze(sample_bgr, TaskProfile.AUTO)

    payload = json.loads(_advice_json(result))

    assert payload["opencv_version"]
    assert payload["pipeline_config_hash"]
    assert len(payload["recommendations"]) == 3


def test_report_zip_preserves_relative_artifact_paths(tmp_path):
    report = tmp_path / "report"
    nested = report / "confusion_matrices"
    nested.mkdir(parents=True)
    (report / "leaderboard.csv").write_text("rank,score\n1,0.9\n", encoding="utf-8")
    (nested / "best.png").write_bytes(b"not-a-real-png")

    archive = ZipFile(BytesIO(_report_zip(report)))

    assert set(archive.namelist()) == {
        "leaderboard.csv",
        "confusion_matrices/best.png",
    }
