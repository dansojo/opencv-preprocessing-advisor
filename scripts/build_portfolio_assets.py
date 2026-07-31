"""Generate deterministic, dataset-free images for the project portfolio."""

from __future__ import annotations

import argparse
import sys
from itertools import pairwise
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from opencv_preprocessing_advisor.models import TaskProfile
from opencv_preprocessing_advisor.services import ImageAdvisorService

PNG_DPI = 160
PNG_WIDTH_INCHES = 10


def _configure_font() -> str:
    """Prefer a Korean-capable font while keeping the assets portable."""
    available = {font.name for font in fontManager.ttflist}
    for name in ("Noto Sans CJK KR", "Malgun Gothic", "Noto Sans CJK", "DejaVu Sans"):
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["font.family"] = "DejaVu Sans"
    return "DejaVu Sans"


def _save(figure: plt.Figure, path: Path) -> Path:
    figure.savefig(path, dpi=PNG_DPI, facecolor="white", metadata={"Software": "OpenCV"})
    plt.close(figure)
    return path


def _draw_node(ax, xy: tuple[float, float], text: str, *, width: float = 0.23) -> None:
    x, y = xy
    node = FancyBboxPatch(
        (x - width / 2, y - 0.07),
        width,
        0.14,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.4,
        edgecolor="#1f4e79",
        facecolor="#eaf3f8",
    )
    ax.add_patch(node)
    ax.text(x, y, text, ha="center", va="center", fontsize=10, color="#12324a", wrap=True)


def _arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "lw": 1.5,
            "color": "#50718a",
            "shrinkA": 15,
            "shrinkB": 15,
        },
    )


def _architecture(path: Path) -> Path:
    figure, ax = plt.subplots(figsize=(PNG_WIDTH_INCHES, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "OpenCV Preprocessing Advisor Architecture", fontsize=17, fontweight="bold", pad=18
    )

    nodes = {
        "streamlit": (0.18, 0.80, "Streamlit UI\n(사용자 화면)"),
        "cli": (0.18, 0.52, "CLI\n(자동화 실행)"),
        "services": (0.50, 0.66, "Application Services\n진단 · 추천 · 검증"),
        "core": (0.80, 0.82, "Diagnostics / Pipelines\nFeatures / Evaluation"),
        "reports": (0.80, 0.42, "Reports\nCSV · PNG · JSON"),
    }
    for x, y, label in nodes.values():
        _draw_node(ax, (x, y), label, width=0.27)
    _arrow(ax, (0.31, 0.80), (0.37, 0.68))
    _arrow(ax, (0.31, 0.52), (0.37, 0.64))
    _arrow(ax, (0.64, 0.70), (0.66, 0.80))
    _arrow(ax, (0.64, 0.62), (0.66, 0.44))
    ax.text(
        0.5,
        0.15,
        "One service layer keeps interactive advice, reproducible benchmarks, and traceable reports aligned.",
        ha="center",
        fontsize=10,
        color="#40576a",
    )
    figure.subplots_adjust(left=0.05, right=0.95, bottom=0.10, top=0.88)
    return _save(figure, path)


def _workflow(path: Path) -> Path:
    figure, ax = plt.subplots(figsize=(PNG_WIDTH_INCHES, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Recommendation Workflow", fontsize=17, fontweight="bold", pad=15)

    steps = [
        (0.12, "Image input\n이미지 입력"),
        (0.33, "Diagnosis\nbrightness · noise · edges"),
        (0.54, "Candidate execution\nOpenCV pipelines"),
        (0.75, "Top 3\nscored recommendations"),
    ]
    for x, label in steps:
        _draw_node(ax, (x, 0.62), label, width=0.19)
    for left, right in pairwise(steps):
        _arrow(ax, (left[0] + 0.10, 0.62), (right[0] - 0.10, 0.62))
    _draw_node(ax, (0.75, 0.26), "Optional dataset\ncross-validation", width=0.24)
    _arrow(ax, (0.75, 0.51), (0.75, 0.26))
    ax.text(
        0.5,
        0.08,
        "Advice ranks candidate preprocessing for one image; benchmark metrics validate choices on labeled data.",
        ha="center",
        fontsize=10,
        color="#40576a",
    )
    figure.subplots_adjust(left=0.04, right=0.96, bottom=0.10, top=0.85)
    return _save(figure, path)


def _synthetic_tile() -> np.ndarray:
    """Create a deterministic low-contrast industrial-style tile without dataset inputs."""
    height = width = 512
    y, x = np.indices((height, width), dtype=np.float32)
    gradient = 112 + 10 * (x / width) + 6 * np.sin(y / 42)
    texture = 3 * np.sin(x / 7.5) * np.cos(y / 11)
    gray = gradient + texture
    rng = np.random.default_rng(42)
    gray += rng.normal(0, 2.2, size=gray.shape)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    cv2.rectangle(gray, (72, 105), (442, 400), 121, thickness=3)
    cv2.line(gray, (105, 330), (398, 300), 96, thickness=3)
    cv2.circle(gray, (350, 190), 22, 99, thickness=-1)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _advice_comparison(path: Path) -> Path:
    image = _synthetic_tile()
    result = ImageAdvisorService().analyze(image, TaskProfile.AUTO)
    panels = [("Original\nSynthetic low-contrast tile", image)]
    panels.extend(
        (
            (
                f"#{index} {recommendation.pipeline_run.display_name_en}\n"
                f"Suitability score: {recommendation.suitability_score:.3f}"
            ),
            recommendation.pipeline_run.output_image,
        )
        for index, recommendation in enumerate(result.recommendations, start=1)
    )

    figure, axes = plt.subplots(2, 2, figsize=(PNG_WIDTH_INCHES, 7.2))
    figure.suptitle(
        "Synthetic Image Advice: Original vs. Top 3 Recommendations", fontsize=17, fontweight="bold"
    )
    for axis, (title, panel) in zip(axes.flat, panels):
        axis.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        axis.set_title(title, fontsize=11, pad=9)
        axis.axis("off")
    figure.subplots_adjust(left=0.04, right=0.96, bottom=0.05, top=0.89, hspace=0.28, wspace=0.10)
    return _save(figure, path)


def build_assets(output_dir: Path) -> dict[str, Path]:
    """Build portfolio PNGs from project logic and return their paths by stable name."""
    _configure_font()
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "architecture": _architecture(output_dir / "architecture.png"),
        "workflow": _workflow(output_dir / "workflow.png"),
        "synthetic_advice_comparison": _advice_comparison(
            output_dir / "synthetic-advice-comparison.png"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Directory for generated PNG assets"
    )
    args = parser.parse_args()
    for name, path in build_assets(args.output).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
