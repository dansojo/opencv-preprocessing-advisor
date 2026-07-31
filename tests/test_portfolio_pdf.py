"""Contract tests for the recruiter-facing portfolio PDF."""

import shutil
from pathlib import Path

import pdfplumber
import pytest
from pypdf import PdfReader

from scripts.build_portfolio_pdf import build_pdf, load_portfolio_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _copy_portfolio_sources(tmp_path: Path) -> Path:
    copied_root = tmp_path / "project"
    shutil.copytree(PROJECT_ROOT / "docs" / "portfolio", copied_root / "docs" / "portfolio")
    config_dir = copied_root / "src" / "opencv_preprocessing_advisor" / "config"
    config_dir.mkdir(parents=True)
    shutil.copy2(
        PROJECT_ROOT / "src" / "opencv_preprocessing_advisor" / "config" / "pipelines.yaml",
        config_dir / "pipelines.yaml",
    )
    return copied_root


def test_load_portfolio_data_validates_canonical_markdown_evidence_and_yaml() -> None:
    data = load_portfolio_data(PROJECT_ROOT / "docs" / "portfolio" / "assets")

    assert data.sample_count == 117
    assert data.class_count == 6
    assert data.folds == 5
    assert data.seed == 42
    assert [(row.pipeline, row.classifier) for row in data.top_pipelines] == [
        ("Original", "RTrees"),
        ("CLAHE + Bilateral", "RTrees"),
        ("LAB CLAHE", "RTrees"),
    ]


def test_build_pdf_rejects_tampered_canonical_source(tmp_path: Path) -> None:
    copied_root = _copy_portfolio_sources(tmp_path)
    experiment_results = copied_root / "docs" / "portfolio" / "experiment-results.md"
    experiment_results.write_text(
        experiment_results.read_text(encoding="utf-8").replace("## 리더보드", "## 삭제됨"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="experiment-results.md.*## 리더보드"):
        build_pdf(tmp_path / "tampered.pdf", copied_root / "docs" / "portfolio" / "assets")


def test_build_pdf_rejects_missing_canonical_source(tmp_path: Path) -> None:
    copied_root = _copy_portfolio_sources(tmp_path)
    (copied_root / "docs" / "portfolio" / "limitations.md").unlink()

    with pytest.raises(
        FileNotFoundError, match="Missing canonical portfolio source.*limitations.md"
    ):
        build_pdf(tmp_path / "missing-source.pdf", copied_root / "docs" / "portfolio" / "assets")


def test_build_pdf_creates_six_page_korean_portfolio(tmp_path: Path) -> None:
    output = tmp_path / "opencv-preprocessing-advisor-portfolio.pdf"

    result = build_pdf(output, PROJECT_ROOT / "docs" / "portfolio" / "assets")

    assert result == output
    assert output.is_file()
    assert output.stat().st_size > 100_000

    reader = PdfReader(output)
    assert len(reader.pages) == 6

    with pdfplumber.open(output) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    for required_text in (
        "OpenCV Preprocessing Advisor",
        "Macro F1 0.789",
        "117 images",
        "휴리스틱",
        "전처리가 항상 성능을 높이지는 않는다",
        "https://github.com/dansojo/opencv-preprocessing-advisor",
    ):
        assert required_text in text


def test_pdf_is_landscape_a4_and_embeds_korean_font(tmp_path: Path) -> None:
    output = build_pdf(tmp_path / "portfolio.pdf", PROJECT_ROOT / "docs" / "portfolio" / "assets")
    reader = PdfReader(output)

    page = reader.pages[0]
    assert float(page.mediabox.width) == pytest.approx(841.89, abs=0.1)
    assert float(page.mediabox.height) == pytest.approx(595.28, abs=0.1)

    fonts = page["/Resources"]["/Font"].get_object().values()
    embedded_korean_font = False
    for font_reference in fonts:
        font = font_reference.get_object()
        base_font = str(font.get("/BaseFont", ""))
        descriptor = font.get("/FontDescriptor")
        if "MalgunGothic" in base_font and descriptor is not None:
            embedded_korean_font = "/FontFile2" in descriptor.get_object()
    assert embedded_korean_font
