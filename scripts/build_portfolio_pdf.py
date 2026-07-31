"""Build a static six-page Korean portfolio for OpenCV Preprocessing Advisor."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

PAGE_WIDTH, PAGE_HEIGHT = landscape(A4)
MARGIN = 34
NAVY = HexColor("#102A43")
INK = HexColor("#17324D")
MUTED = HexColor("#55718A")
CYAN = HexColor("#16B8D4")
CYAN_PALE = HexColor("#E6F8FC")
ORANGE = HexColor("#F28C3A")
ORANGE_PALE = HexColor("#FFF0E2")
PANEL = HexColor("#F4F8FB")
LINE = HexColor("#D7E3EC")
FONT_NAME = "PortfolioKorean"
FONT_BOLD = "PortfolioKoreanBold"


@dataclass(frozen=True)
class BenchmarkRow:
    """One rounded leaderboard entry sourced from benchmark-evidence.json."""

    pipeline: str
    classifier: str
    accuracy: float
    macro_f1: float


@dataclass(frozen=True)
class PortfolioData:
    """Verified canonical facts that are safe to present in the PDF."""

    sample_count: int
    class_count: int
    folds: int
    seed: int
    top_pipelines: tuple[BenchmarkRow, ...]


def _require_text(source: Path, text: str, required: tuple[str, ...]) -> None:
    missing = [claim for claim in required if claim not in text]
    if missing:
        raise ValueError(f"{source.name} is missing required canonical claim: {missing[0]}")


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"benchmark-evidence.json has invalid {path}")
    return value


def _display_metric(value: float) -> str:
    return f"{value:.3f}"


def load_portfolio_data(assets_dir: Path) -> PortfolioData:
    """Load and cross-check Markdown, YAML, and benchmark evidence for the PDF."""
    assets_dir = Path(assets_dir)
    portfolio_dir = assets_dir.parent
    canonical_sources = {
        "case-study.md": (
            "## 문제 정의",
            "휴리스틱",
            "LAB L-channel CLAHE",
            "fold-local scaling",
        ),
        "experiment-results.md": (
            "## 정확한 평가 프로토콜",
            "## 리더보드",
            "stratified 5-fold cross-validation",
        ),
        "limitations.md": (
            "## 휴리스틱 추천의 한계",
            "## GT와 MVTec 공식 평가의 부재",
            "not classification accuracy",
        ),
        "evidence-map.md": (
            "# OpenCV Portfolio Evidence Map",
            "| Area | OpenCV technique |",
            "RTrees",
        ),
    }
    canonical_text: dict[str, str] = {}
    for filename, required_claims in canonical_sources.items():
        source = portfolio_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing canonical portfolio source: {source}")
        text = source.read_text(encoding="utf-8")
        _require_text(source, text, required_claims)
        canonical_text[filename] = text

    evidence_path = portfolio_dir / "benchmark-evidence.json"
    if not evidence_path.is_file():
        raise FileNotFoundError(f"Missing canonical benchmark evidence: {evidence_path}")
    try:
        evidence = _require_mapping(json.loads(evidence_path.read_text(encoding="utf-8")), "root")
    except json.JSONDecodeError as error:
        raise ValueError(f"benchmark-evidence.json is invalid JSON: {error.msg}") from error

    evaluation = _require_mapping(evidence.get("evaluation"), "evaluation")
    rows = evidence.get("top_pipelines")
    if not isinstance(rows, list) or len(rows) < 3:
        raise ValueError("benchmark-evidence.json must contain at least three top_pipelines rows")
    try:
        data = PortfolioData(
            sample_count=int(evidence["sample_count"]),
            class_count=int(evidence["class_count"]),
            folds=int(evaluation["folds"]),
            seed=int(evaluation["seed"]),
            top_pipelines=tuple(
                BenchmarkRow(
                    pipeline=str(_require_mapping(row, "top_pipelines row")["pipeline"]),
                    classifier=str(_require_mapping(row, "top_pipelines row")["classifier"]),
                    accuracy=float(_require_mapping(row, "top_pipelines row")["mean_accuracy"]),
                    macro_f1=float(_require_mapping(row, "top_pipelines row")["mean_macro_f1"]),
                )
                for row in rows[:3]
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"benchmark-evidence.json has invalid benchmark values: {error}"
        ) from error

    experiment = canonical_text["experiment-results.md"]
    _require_text(
        portfolio_dir / "experiment-results.md",
        experiment,
        (
            f"총 {data.sample_count} images",
            f"{data.class_count} classes",
            f"seed {data.seed}",
            f"stratified {data.folds}-fold cross-validation",
        ),
    )
    for row in data.top_pipelines:
        matching_rows = [line for line in experiment.splitlines() if f"| {row.pipeline} |" in line]
        expected_values = (
            row.classifier,
            _display_metric(row.accuracy),
            _display_metric(row.macro_f1),
        )
        if not matching_rows or not all(value in matching_rows[0] for value in expected_values):
            raise ValueError(
                "experiment-results.md does not match benchmark-evidence.json for "
                f"{row.pipeline} + {row.classifier}"
            )

    pipeline_path = (
        assets_dir.parents[2] / "src" / "opencv_preprocessing_advisor" / "config" / "pipelines.yaml"
    )
    if not pipeline_path.is_file():
        raise FileNotFoundError(f"Missing canonical pipeline configuration: {pipeline_path}")
    pipeline_config = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    pipelines = _require_mapping(pipeline_config, "pipelines.yaml root").get("pipelines")
    if not isinstance(pipelines, list):
        raise TypeError("pipelines.yaml must contain a pipelines list")
    pipeline_steps = {
        str(item.get("id")): {str(step.get("transform")) for step in item.get("steps", [])}
        for item in pipelines
        if isinstance(item, dict)
    }
    required_pipeline_steps = {
        "lab-clahe": {"lab_clahe"},
        "clahe-bilateral": {"lab_clahe", "bilateral"},
    }
    for pipeline_id, expected_steps in required_pipeline_steps.items():
        if not expected_steps.issubset(pipeline_steps.get(pipeline_id, set())):
            raise ValueError(f"pipelines.yaml does not support canonical pipeline: {pipeline_id}")

    return data


def _register_korean_fonts() -> tuple[str, str]:
    """Register embedded Korean TrueType fonts and return regular/bold names."""
    if FONT_NAME in pdfmetrics.getRegisteredFontNames():
        return FONT_NAME, FONT_BOLD

    candidates = (
        (Path("C:/Windows/Fonts/malgun.ttf"), Path("C:/Windows/Fonts/malgunbd.ttf")),
        (
            Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
            Path("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
        ),
    )
    for regular, bold in candidates:
        if regular.is_file() and bold.is_file():
            pdfmetrics.registerFont(TTFont(FONT_NAME, str(regular)))
            pdfmetrics.registerFont(TTFont(FONT_BOLD, str(bold)))
            return FONT_NAME, FONT_BOLD
    raise FileNotFoundError("A Korean TrueType font (Malgun Gothic or Nanum Gothic) is required.")


def _image_size(path: Path, max_width: float, max_height: float) -> tuple[float, float]:
    width, height = ImageReader(str(path)).getSize()
    scale = min(max_width / width, max_height / height)
    return width * scale, height * scale


def _draw_image(
    canvas: Canvas,
    path: Path,
    x: float,
    y: float,
    max_width: float,
    max_height: float,
) -> tuple[float, float]:
    width, height = _image_size(path, max_width, max_height)
    canvas.drawImage(
        str(path), x + (max_width - width) / 2, y + (max_height - height) / 2, width, height
    )
    return width, height


def _rect(canvas: Canvas, x: float, y: float, width: float, height: float, color: HexColor) -> None:
    canvas.setFillColor(color)
    canvas.roundRect(x, y, width, height, 10, fill=1, stroke=0)


def _text(
    canvas: Canvas,
    value: str,
    x: float,
    y: float,
    size: float,
    color: HexColor = INK,
    bold: bool = False,
) -> None:
    canvas.setFillColor(color)
    canvas.setFont(FONT_BOLD if bold else FONT_NAME, size)
    canvas.drawString(x, y, value)


def _lines(
    canvas: Canvas,
    values: list[str],
    x: float,
    y: float,
    size: float = 10,
    leading: float = 16,
    color: HexColor = INK,
    bold: bool = False,
) -> None:
    for index, value in enumerate(values):
        _text(canvas, value, x, y - index * leading, size, color, bold)


def _page_shell(canvas: Canvas, page_number: int, label: str) -> None:
    canvas.setFillColor(white)
    canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_HEIGHT - 13, PAGE_WIDTH, 13, fill=1, stroke=0)
    _text(canvas, "OpenCV Preprocessing Advisor", MARGIN, PAGE_HEIGHT - 31, 9, MUTED, bold=True)
    _text(canvas, label, PAGE_WIDTH - MARGIN - 145, PAGE_HEIGHT - 31, 9, MUTED)
    canvas.setStrokeColor(LINE)
    canvas.line(MARGIN, 25, PAGE_WIDTH - MARGIN, 25)
    _text(canvas, "Explainable preprocessing portfolio", MARGIN, 12, 9, MUTED)
    _text(canvas, f"{page_number} / 6", PAGE_WIDTH - MARGIN - 23, 12, 9, MUTED, bold=True)


def _title(canvas: Canvas, title: str, subtitle: str) -> None:
    _text(canvas, title, MARGIN, PAGE_HEIGHT - 76, 26, NAVY, bold=True)
    _text(canvas, subtitle, MARGIN, PAGE_HEIGHT - 99, 10, MUTED)


def _metric_card(
    canvas: Canvas,
    x: float,
    y: float,
    width: float,
    value: str,
    label: str,
    color: HexColor,
    value_size: float = 19,
) -> None:
    _rect(canvas, x, y, width, 77, PANEL)
    canvas.setFillColor(color)
    canvas.roundRect(x, y + 68, width, 9, 10, fill=1, stroke=0)
    _text(canvas, value, x + 14, y + 39, value_size, NAVY, bold=True)
    _text(canvas, label, x + 16, y + 18, 9, MUTED)


def _bullet(canvas: Canvas, x: float, y: float, text: str, color: HexColor = CYAN) -> None:
    canvas.setFillColor(color)
    canvas.circle(x + 4, y + 4, 3, fill=1, stroke=0)
    _text(canvas, text, x + 15, y, 10, INK)


def _page_one(canvas: Canvas, assets: Path, data: PortfolioData) -> None:
    _page_shell(canvas, 1, "Project and impact")
    _title(
        canvas,
        "이미지 전처리의 효과를 측정 가능한 선택으로",
        "단일 이미지 추천과 데이터셋 검증을 분리한 OpenCV 중심 프로젝트",
    )
    _text(canvas, "보이는 개선을 성능 향상으로 착각하지 않는다.", MARGIN, 432, 15, CYAN, bold=True)
    _lines(
        canvas,
        [
            "이미지 상태를 진단해 설명 가능한 Top 3 후보를 제안한다.",
            "레이블 데이터셋에서는 같은 선택을 교차 검증으로 측정한다.",
        ],
        MARGIN,
        408,
        10,
        17,
    )
    card_y = 278
    metric_width = 120
    _metric_card(
        canvas,
        MARGIN,
        card_y,
        metric_width,
        f"{data.sample_count} images",
        "tile 분류 사례",
        CYAN,
        value_size=15,
    )
    _metric_card(
        canvas,
        MARGIN + 135,
        card_y,
        metric_width,
        f"{data.class_count} classes",
        f"{data.folds}-fold · seed {data.seed}",
        ORANGE,
        value_size=15,
    )
    _metric_card(
        canvas,
        MARGIN + 270,
        card_y,
        metric_width,
        _display_metric(data.top_pipelines[0].macro_f1),
        f"Macro F1 {_display_metric(data.top_pipelines[0].macro_f1)}",
        CYAN,
        value_size=18,
    )
    _rect(canvas, MARGIN, 145, 372, 91, ORANGE_PALE)
    _text(canvas, "핵심 설계", MARGIN + 16, 207, 11, ORANGE, bold=True)
    _lines(
        canvas,
        [
            "휴리스틱 점수는 정확도가 아니라 실험 우선순위다.",
            "성능에 대한 결론은 fold-local scaling을 적용한 데이터셋 평가에서만 낸다.",
        ],
        MARGIN + 16,
        185,
        11,
        18,
        NAVY,
    )
    _rect(canvas, 440, 128, 367, 289, CYAN_PALE)
    _draw_image(canvas, assets / "synthetic-advice-comparison.png", 450, 146, 347, 261)
    _text(canvas, "프로젝트 코드가 생성한 합성 저대비 타일 비교", 452, 134, 9, MUTED)


def _draw_workflow_vector(canvas: Canvas, x: float, y: float, width: float) -> None:
    """Recreate the committed workflow asset as a legible PDF-native diagram."""
    steps = ["입력", "진단", "후보 실행", "Top 3", "교차 검증"]
    step_width = 67
    gap = (width - step_width * len(steps)) / (len(steps) - 1)
    for index, step in enumerate(steps):
        step_x = x + index * (step_width + gap)
        _rect(canvas, step_x, y, step_width, 40, CYAN_PALE if index != 3 else ORANGE_PALE)
        _text(canvas, step, step_x + 14, y + 15, 11, NAVY, bold=True)
        if index < len(steps) - 1:
            arrow_x = step_x + step_width + 5
            canvas.setStrokeColor(CYAN)
            canvas.setLineWidth(1.4)
            canvas.line(arrow_x, y + 20, arrow_x + gap - 10, y + 20)
            canvas.line(arrow_x + gap - 16, y + 25, arrow_x + gap - 10, y + 20)
            canvas.line(arrow_x + gap - 16, y + 15, arrow_x + gap - 10, y + 20)


def _page_two(canvas: Canvas, assets: Path) -> None:
    _page_shell(canvas, 2, "Problem and solution")
    _title(
        canvas,
        "한 장의 이미지와 성능 평가는 다른 문제다",
        "추천은 투명한 탐색, 벤치마크는 레이블 기반의 성능 측정",
    )
    left_x, right_x = MARGIN, 430
    _rect(canvas, left_x, 234, 352, 188, PANEL)
    _text(canvas, "단일 이미지 Advisor", left_x + 18, 389, 16, NAVY, bold=True)
    _text(canvas, "레이블 없는 입력에서 다음 실험의 우선순위를 제안", left_x + 18, 367, 10, MUTED)
    _bullet(canvas, left_x + 18, 330, "밝기, 대비, 엔트로피, 선명도, 노이즈, 에지를 진단")
    _bullet(canvas, left_x + 18, 301, "후보 실행 전후의 진단 변화와 경고를 함께 기록")
    _bullet(canvas, left_x + 18, 272, "프로필별 가중치로 설명 가능한 Top 3를 정렬", ORANGE)
    _rect(canvas, right_x, 234, 377, 188, CYAN_PALE)
    _text(canvas, "Dataset Benchmark", right_x + 18, 389, 16, NAVY, bold=True)
    _text(
        canvas, "클래스 폴더 데이터셋에서 같은 선택을 공정하게 비교", right_x + 18, 367, 10, MUTED
    )
    _bullet(canvas, right_x + 18, 330, "고정 OpenCV 특징 + SVM, kNN, RTrees 비교")
    _bullet(canvas, right_x + 18, 301, "stratified 5-fold와 fold-local scaling으로 누수 방지")
    _bullet(canvas, right_x + 18, 272, "Accuracy, Macro F1, 클래스별 지표와 혼동행렬 보고", ORANGE)
    canvas.setStrokeColor(ORANGE)
    canvas.setLineWidth(2)
    canvas.line(395, 328, 418, 328)
    canvas.line(411, 335, 418, 328)
    canvas.line(411, 321, 418, 328)
    _rect(canvas, MARGIN, 139, 316, 61, ORANGE_PALE)
    _text(canvas, "해석 원칙", MARGIN + 16, 176, 10, ORANGE, bold=True)
    _text(
        canvas,
        "시각적으로 강한 효과를 분류 성능 향상으로 해석하지 않는다.",
        MARGIN + 16,
        154,
        11,
        NAVY,
        bold=True,
    )
    _rect(canvas, 384, 115, 423, 102, PANEL)
    _draw_workflow_vector(canvas, 402, 153, 387)
    _text(canvas, "committed workflow.png을 바탕으로 PDF에서 벡터로 재구성", 402, 127, 9, MUTED)


def _page_three(canvas: Canvas) -> None:
    _page_shell(canvas, 3, "OpenCV capability matrix")
    _title(
        canvas,
        "OpenCV 기법을 추천과 검증의 증거로 연결",
        "모든 핵심 선택은 구현 경로와 회귀 테스트로 추적할 수 있다",
    )
    rows = [
        (
            "이미지 진단",
            "brightness · contrast · entropy · sharpness · noise",
            "상태를 수치화해 후보 선택의 출발점으로 사용",
        ),
        (
            "전처리 후보",
            "normalize · gamma · CLAHE · Gaussian · median · bilateral",
            "노이즈와 대비의 가정을 명시한 재현 가능한 단계",
        ),
        (
            "특징",
            "HOG · HSV/LAB histogram · Sobel/Laplacian/Gabor",
            "색상, 형태, 질감 증거를 결합한 고전 특징 프로필",
        ),
        ("분류", "cv2.ml SVM · kNN · RTrees", "같은 fold와 특징 행렬 아래에서 비교"),
        (
            "공정한 평가",
            "stratified K-fold · fold-local scaling · Macro F1",
            "테스트 fold의 통계가 훈련에 섞이지 않도록 설계",
        ),
        (
            "재현 가능한 보고",
            "CSV · JSON · PNG · config hash · OpenCV version",
            "수치, 예측, 설정의 근거를 다시 확인 가능",
        ),
    ]
    x, top = MARGIN, 425
    widths = (143, 305, 316)
    row_h = 45
    canvas.setFillColor(NAVY)
    canvas.roundRect(x, top, sum(widths), 31, 8, fill=1, stroke=0)
    _text(canvas, "영역", x + 14, top + 10, 10, white, bold=True)
    _text(canvas, "OpenCV / 구현 기법", x + widths[0] + 14, top + 10, 10, white, bold=True)
    _text(
        canvas,
        "프로젝트에서의 의미",
        x + widths[0] + widths[1] + 14,
        top + 10,
        10,
        white,
        bold=True,
    )
    for index, row in enumerate(rows):
        y = top - (index + 1) * row_h
        _rect(canvas, x, y, sum(widths), row_h - 3, PANEL if index % 2 == 0 else CYAN_PALE)
        _text(canvas, row[0], x + 14, y + 15, 10, NAVY, bold=True)
        _text(canvas, row[1], x + widths[0] + 14, y + 15, 9, INK)
        _text(canvas, row[2], x + widths[0] + widths[1] + 14, y + 15, 9, INK)
    _rect(canvas, MARGIN, 84, 764, 63, ORANGE_PALE)
    _text(canvas, "설계 선택", MARGIN + 16, 121, 10, ORANGE, bold=True)
    _text(
        canvas,
        "LAB L-channel CLAHE는 밝기 채널만 조정해 색상 관계를 덜 교란한다.",
        MARGIN + 16,
        99,
        11,
        NAVY,
        bold=True,
    )


def _page_four(canvas: Canvas, assets: Path) -> None:
    _page_shell(canvas, 4, "Architecture and recommendation")
    _title(
        canvas,
        "UI가 아닌 서비스와 증거가 중심인 구조",
        "Streamlit과 CLI는 같은 진단, 파이프라인, 평가, 보고 서비스 계층을 호출한다",
    )
    _rect(canvas, MARGIN, 150, 405, 269, PANEL)
    _draw_image(canvas, assets / "architecture.png", MARGIN + 12, 174, 381, 215)
    _text(
        canvas,
        "서비스 계층을 분리해 UI 밖에서도 테스트와 재생성이 가능하다.",
        MARGIN + 16,
        164,
        9,
        MUTED,
    )
    _text(canvas, "추천이 설명을 남기는 방식", 483, 407, 16, NAVY, bold=True)
    _lines(
        canvas,
        [
            "1. 이미지 조건을 진단한다.",
            "2. YAML 파이프라인 후보를 적용한다.",
            "3. 전후 진단값, 점수 기여, 경고를 비교한다.",
        ],
        483,
        378,
        10,
        23,
    )
    recommendation = [
        ("01", "LAB L-channel CLAHE", "색상 관계를 덜 교란하며 국소 대비를 탐색"),
        ("02", "Median / Bilateral", "노이즈 가정을 다르게 둔 필터 후보 비교"),
        ("03", "Warnings", "클리핑, 과도한 에지, 평활화, 색 손실을 노출"),
    ]
    for index, (number, name, detail) in enumerate(recommendation):
        y = 279 - index * 57
        _rect(canvas, 483, y, 324, 47, CYAN_PALE if index != 1 else ORANGE_PALE)
        _text(canvas, number, 496, y + 17, 10, CYAN if index != 1 else ORANGE, bold=True)
        _text(canvas, name, 532, y + 23, 10, NAVY, bold=True)
        _text(canvas, detail, 532, y + 10, 9, MUTED)
    _rect(canvas, 483, 83, 324, 61, NAVY)
    _text(canvas, "점수만 보여 주지 않는다.", 499, 116, 12, white, bold=True)
    _text(
        canvas,
        "어떤 진단 변화가 추천을 만들었는지 확인할 수 있다.",
        499,
        96,
        9,
        HexColor("#D5ECF5"),
    )


def _page_five(canvas: Canvas, assets: Path, data: PortfolioData) -> None:
    _page_shell(canvas, 5, "Benchmark and failure insight")
    _title(
        canvas,
        "원본이 이긴 것도 중요한 엔지니어링 결론",
        f"MVTec tile 상태 폴더 {data.class_count}개를 클래스로 해석한 제한된 분류 실험",
    )
    _rect(canvas, MARGIN, 294, 343, 125, PANEL)
    _text(canvas, "Top 3 leaderboard", MARGIN + 16, 392, 12, NAVY, bold=True)
    rows = [
        (
            f"{row.pipeline} ({row.classifier})",
            _display_metric(row.accuracy),
            _display_metric(row.macro_f1),
        )
        for row in data.top_pipelines
    ]
    _text(canvas, "Pipeline", MARGIN + 16, 370, 9, MUTED, bold=True)
    _text(canvas, "Accuracy", 230, 370, 9, MUTED, bold=True)
    _text(canvas, "Macro F1", 304, 370, 9, MUTED, bold=True)
    for index, (pipeline, accuracy, f1) in enumerate(rows):
        y = 345 - index * 21
        _text(canvas, pipeline, MARGIN + 16, y, 9, NAVY, bold=index == 0)
        _text(canvas, accuracy, 230, y, 9, INK)
        _text(canvas, f1, 312, y, 9, CYAN if index == 0 else INK, bold=index == 0)
    _rect(canvas, MARGIN, 127, 343, 135, ORANGE_PALE)
    _text(canvas, "핵심 인사이트", MARGIN + 16, 228, 10, ORANGE, bold=True)
    _text(canvas, "전처리가 항상 성능을 높이지는 않는다.", MARGIN + 16, 184, 16, NAVY, bold=True)
    _lines(
        canvas,
        [
            "이 데이터와 고정 특징에서는 원본 질감이 이미 충분했다.",
            "강한 대비 강화나 평활화가 유용한 약한 질감을 바꿨을 수 있다.",
        ],
        MARGIN + 16,
        149,
        9,
        14,
        MUTED,
    )
    _rect(canvas, 412, 124, 395, 300, CYAN_PALE)
    _draw_image(canvas, assets / "mvtec-tile-best-confusion-matrix.png", 427, 151, 365, 246)
    _text(canvas, "Original + RTrees confusion matrix", 429, 137, 9, MUTED)
    _text(canvas, "공식 MVTec anomaly-detection metric이 아님", 429, 107, 9, ORANGE, bold=True)


def _page_six(canvas: Canvas) -> None:
    _page_shell(canvas, 6, "Evidence, limitations, and links")
    _title(
        canvas,
        "주장에는 구현 경로와 한계를 함께 남긴다",
        "재현 가능한 기술 선택과 다음 검증을 위한 범위를 명확히 기록",
    )
    columns = [
        (
            MARGIN,
            "증거",
            CYAN_PALE,
            [
                "기법 선택은 소스와 회귀 테스트에 연결",
                "CSV, JSON, PNG, 설정 hash로 결과를 보존",
                "OpenCV version과 seed를 보고서에 기록",
            ],
        ),
        (
            294,
            "한계",
            ORANGE_PALE,
            [
                "휴리스틱 점수는 classification accuracy가 아님",
                "117장, 6클래스의 데이터셋 특이적 관찰",
                "GT mask·anomaly localization·공식 평가는 제외",
            ],
        ),
        (
            554,
            "다음 검증",
            PANEL,
            [
                "다른 클래스, split seed, 파라미터 범위 비교",
                "특징 프로필 ablation과 오류 사례 정성 검토",
                "SIFT는 fold-local vocabulary 통합 후 비교",
            ],
        ),
    ]
    for x, heading, color, bullets in columns:
        _rect(canvas, x, 245, 226, 165, color)
        _text(canvas, heading, x + 16, 378, 15, NAVY, bold=True)
        for index, item in enumerate(bullets):
            _bullet(canvas, x + 16, 335 - index * 38, item, ORANGE if heading == "한계" else CYAN)
    _rect(canvas, MARGIN, 114, 764, 89, NAVY)
    _text(canvas, "Repository", MARGIN + 18, 171, 10, HexColor("#A7E8F4"), bold=True)
    _text(
        canvas,
        "https://github.com/dansojo/opencv-preprocessing-advisor",
        MARGIN + 18,
        145,
        12,
        white,
        bold=True,
    )
    _text(
        canvas,
        "README, canonical case study, experiment results, evidence map, and reproducible scripts",
        MARGIN + 18,
        125,
        9,
        HexColor("#D5ECF5"),
    )
    _text(
        canvas,
        "포트폴리오는 구현 가능한 주장과 검증 가능한 한계를 함께 제시한다.",
        MARGIN,
        83,
        11,
        CYAN,
        bold=True,
    )


def build_pdf(output_path: Path, assets_dir: Path) -> Path:
    """Create the static six-page PDF and return its output path."""
    output_path = Path(output_path)
    assets_dir = Path(assets_dir)
    required_assets = (
        "architecture.png",
        "workflow.png",
        "synthetic-advice-comparison.png",
        "mvtec-tile-best-confusion-matrix.png",
    )
    missing = [name for name in required_assets if not (assets_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing portfolio assets: {', '.join(missing)}")

    data = load_portfolio_data(assets_dir)
    _register_korean_fonts()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(output_path), pagesize=(PAGE_WIDTH, PAGE_HEIGHT), pageCompression=1)
    canvas.setTitle("OpenCV Preprocessing Advisor Portfolio")
    canvas.setAuthor("OpenCV Preprocessing Advisor")
    _page_one(canvas, assets_dir, data)
    canvas.showPage()
    _page_two(canvas, assets_dir)
    canvas.showPage()
    _page_three(canvas)
    canvas.showPage()
    _page_four(canvas, assets_dir)
    canvas.showPage()
    _page_five(canvas, assets_dir, data)
    canvas.showPage()
    _page_six(canvas)
    canvas.save()
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets", type=Path, required=True, help="Directory containing committed PNG assets"
    )
    parser.add_argument("--output", type=Path, required=True, help="PDF output path")
    args = parser.parse_args()
    print(build_pdf(args.output, args.assets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
