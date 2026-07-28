from pathlib import Path

import numpy as np
import pytest

from opencv_preprocessing_advisor.pipelines import PipelineCatalog

CONFIG_PATH = Path(__file__).parents[1] / "config" / "pipelines.yaml"


def test_catalog_loads_ten_unique_pipelines():
    catalog = PipelineCatalog.from_yaml(CONFIG_PATH)

    assert len(catalog.pipeline_ids) == 10
    assert len(set(catalog.pipeline_ids)) == 10


def test_pipeline_preserves_shape_dtype_and_step_order(sample_bgr):
    catalog = PipelineCatalog.from_yaml(CONFIG_PATH)

    run = catalog.run("clahe-bilateral", sample_bgr)

    assert run.output_image.shape == sample_bgr.shape
    assert run.output_image.dtype == np.uint8
    assert [step.name for step in run.intermediate_images] == [
        "lab_clahe",
        "bilateral",
    ]


def test_pipeline_is_deterministic(sample_bgr):
    catalog = PipelineCatalog.from_yaml(CONFIG_PATH)

    first = catalog.run("unsharp-detail", sample_bgr)
    second = catalog.run("unsharp-detail", sample_bgr)

    assert np.array_equal(first.output_image, second.output_image)


def test_catalog_rejects_even_kernel(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text(
        """
pipelines:
  - id: bad
    display_name_ko: 잘못됨
    display_name_en: Invalid
    profiles: [auto]
    steps:
      - transform: median
        params: {kernel_size: 4}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="odd"):
        PipelineCatalog.from_yaml(config)
