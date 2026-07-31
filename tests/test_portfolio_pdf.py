"""Contract tests for the recruiter-facing portfolio PDF."""

import shutil
import subprocess
import time
from io import BytesIO
from pathlib import Path

import pdfplumber
import pytest
from pypdf import PdfReader
from reportlab.pdfgen.canvas import Canvas

from scripts.build_portfolio_pdf import MIN_PDF_TEXT_SIZE, _text, build_pdf, load_portfolio_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDED_KOREAN_FONT_NAMES = ("NotoSansKR",)


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
        "https://app.notion.com/p/3aed0dc3cc1d81c0977fd982867f94e1",
        "streamlit run app.py",
    ):
        assert required_text in text

    assert "Diagnostics: brightness" in (pdf.pages[3].extract_text() or "")


def test_build_pdf_is_byte_deterministic(tmp_path: Path) -> None:
    first_output = tmp_path / "first.pdf"
    second_output = tmp_path / "second.pdf"

    build_pdf(first_output, PROJECT_ROOT / "docs" / "portfolio" / "assets")
    time.sleep(1.1)
    build_pdf(second_output, PROJECT_ROOT / "docs" / "portfolio" / "assets")

    assert first_output.read_bytes() == second_output.read_bytes()


def test_committed_pdf_matches_a_fresh_deterministic_build(tmp_path: Path) -> None:
    rebuilt_pdf = build_pdf(
        tmp_path / "opencv-preprocessing-advisor-portfolio.pdf",
        PROJECT_ROOT / "docs" / "portfolio" / "assets",
    )
    committed_pdf = PROJECT_ROOT / "output" / "pdf" / "opencv-preprocessing-advisor-portfolio.pdf"

    assert committed_pdf.read_bytes() == rebuilt_pdf.read_bytes()


def test_claim_validator_rejects_a_committed_pdf_that_differs_from_a_fresh_build(
    tmp_path: Path,
) -> None:
    copied_root = tmp_path / "project"
    shutil.copytree(PROJECT_ROOT, copied_root, ignore=shutil.ignore_patterns(".git", ".venv"))
    committed_pdf = copied_root / "output" / "pdf" / "opencv-preprocessing-advisor-portfolio.pdf"
    committed_pdf.write_bytes(committed_pdf.read_bytes() + b"\n")

    from scripts.validate_portfolio import validate_claims

    errors = validate_claims(copied_root)

    assert "Portfolio PDF does not match a fresh deterministic build." in errors


def test_committed_pdf_is_marked_as_binary_for_git_diff_checks() -> None:
    result = subprocess.run(
        [
            "git",
            "check-attr",
            "binary",
            "--",
            "output/pdf/opencv-preprocessing-advisor-portfolio.pdf",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip().endswith("binary: set")


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
        if any(name in base_font for name in EMBEDDED_KOREAN_FONT_NAMES) and descriptor is not None:
            embedded_korean_font = "/FontFile2" in descriptor.get_object()
    assert embedded_korean_font


def test_pdf_uses_committed_open_font_and_representative_streamlit_capture() -> None:
    font = PROJECT_ROOT / "docs" / "portfolio" / "fonts" / "NotoSansKR-Regular.ttf"
    license_file = font.with_name("OFL.txt")
    screenshot = PROJECT_ROOT / "docs" / "portfolio" / "assets" / "streamlit-advisor-synthetic.png"

    assert font.stat().st_size > 1_000_000
    assert "SIL OPEN FONT LICENSE" in license_file.read_text(encoding="utf-8").upper()
    assert screenshot.read_bytes().startswith(b"\x89PNG")


def test_pdf_text_helper_rejects_sizes_below_the_readability_floor() -> None:
    canvas = Canvas(BytesIO())

    with pytest.raises(ValueError, match="at least 9 point"):
        _text(canvas, "too small", 0, 0, MIN_PDF_TEXT_SIZE - 0.1)
