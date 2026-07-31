"""Contract tests for the recruiter-facing portfolio PDF."""

from pathlib import Path

import pdfplumber
from pypdf import PdfReader

from scripts.build_portfolio_pdf import build_pdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
