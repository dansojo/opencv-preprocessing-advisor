import importlib

import pytest
from streamlit.testing.v1 import AppTest


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
