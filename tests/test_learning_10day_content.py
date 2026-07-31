"""Contract tests for the canonical ten-day learning hub."""

from __future__ import annotations

from pathlib import Path
from shutil import copytree

import pytest

from scripts import validate_portfolio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEARNING_10DAY = PROJECT_ROOT / "docs" / "learning-10day"
DAY_FILES = tuple(f"day-{day:02d}.md" for day in range(1, 11))
REFERENCE_FILES = (
    "technical-qa.md",
    "interview-qa.md",
    "exercises.md",
    "progress-checklist.md",
)
REFERENCE_TITLES = {
    "technical-qa.md": "OpenCV 기술 Q&A",
    "interview-qa.md": "프로젝트·면접 질문과 모범 답안",
    "exercises.md": "실습 과제와 해설",
    "progress-checklist.md": "진도 및 설명 능력 체크리스트",
}
DAY_SECTIONS = (
    "오늘 답해야 할 핵심 질문",
    "개념과 원리",
    "OpenCV API와 파라미터",
    "언제 사용하고 피하는가",
    "프로젝트 코드 연결",
    "직접 실험",
    "예상 결과와 해석",
    "자주 하는 실수와 디버깅",
    "본인 말로 설명하기",
    "완료 기준",
)


def test_ten_day_learning_hub_has_exact_topology_and_ordered_course_index() -> None:
    assert (LEARNING_10DAY / "README.md").is_file()
    assert all((LEARNING_10DAY / name).is_file() for name in (*DAY_FILES, *REFERENCE_FILES))

    readme = (LEARNING_10DAY / "README.md").read_text(encoding="utf-8")
    positions = [readme.index(f"({filename})") for filename in DAY_FILES]
    assert positions == sorted(positions)
    for filename in REFERENCE_FILES:
        assert f"({filename})" in readme
    for target in (
        "../../README.md",
        "../../output/pdf/opencv-preprocessing-advisor-portfolio.pdf",
        "../portfolio/notion-case-study.md",
        "../portfolio/evidence-map.md",
        "../portfolio/benchmark-evidence.json",
        "../portfolio/assets/streamlit-advisor-synthetic.png",
        "../learning/README.md",
        "../../data/samples/synthetic-tile.png",
    ):
        assert target in readme

    for filename in DAY_FILES:
        content = (LEARNING_10DAY / filename).read_text(encoding="utf-8")
        positions = [content.index(f"## {heading}") for heading in DAY_SECTIONS]
        assert positions == sorted(positions)
    for filename, title in REFERENCE_TITLES.items():
        assert (LEARNING_10DAY / filename).read_text(encoding="utf-8").splitlines()[
            0
        ] == f"# {title}"


def test_learning_hub_validator_enforces_the_scaffold_contract(tmp_path: Path) -> None:
    missing_errors = validate_portfolio.validate_learning_hub(tmp_path)
    assert "Missing ten-day learning hub directory: docs/learning-10day" in missing_errors
    assert validate_portfolio.validate_learning_hub(PROJECT_ROOT) == []


def test_learning_hub_validator_rejects_a_local_windows_path(tmp_path: Path) -> None:
    copied_hub = tmp_path / "docs" / "learning-10day"
    copytree(LEARNING_10DAY, copied_hub)
    readme_path = copied_hub / "README.md"
    readme_path.write_text(
        readme_path.read_text(encoding="utf-8") + "\nC:\\Users\\public\\tile.png\n",
        encoding="utf-8",
    )

    errors = validate_portfolio.validate_learning_hub(tmp_path)

    assert "docs/learning-10day/README.md: contains a forbidden local path." in errors


@pytest.mark.parametrize(
    ("filename", "content", "expected_error"),
    (
        (
            "technical-qa.md",
            "# OpenCV Q&A\n",
            "docs/learning-10day/technical-qa.md: must start with exact H1 title: # OpenCV 기술 Q&A.",
        ),
        (
            "interview-qa.md",
            "#    \n",
            "docs/learning-10day/interview-qa.md: must start with exact H1 title: # 프로젝트·면접 질문과 모범 답안.",
        ),
    ),
)
def test_learning_hub_validator_rejects_mutated_or_empty_reference_titles(
    tmp_path: Path, filename: str, content: str, expected_error: str
) -> None:
    copied_hub = tmp_path / "docs" / "learning-10day"
    copytree(LEARNING_10DAY, copied_hub)
    (copied_hub / filename).write_text(content, encoding="utf-8")

    errors = validate_portfolio.validate_learning_hub(tmp_path)

    assert expected_error in errors


@pytest.mark.parametrize(
    "claim",
    (
        "MVTec 공식 benchmark 성능을 재현했다.",
        "MVTec의 공식 benchmark 성능을 재현했다.",
        "MVTec에서 공식 benchmark 성능을 재현했다.",
        "Official MVTec benchmark performance was reproduced.",
        "MVTec benchmark의 공식 성능을 주장한다.",
    ),
)
def test_learning_hub_validator_rejects_official_mvtec_performance_claims(
    tmp_path: Path, claim: str
) -> None:
    copied_hub = tmp_path / "docs" / "learning-10day"
    copytree(LEARNING_10DAY, copied_hub)
    readme_path = copied_hub / "README.md"
    readme_path.write_text(
        readme_path.read_text(encoding="utf-8") + f"\n{claim}\n", encoding="utf-8"
    )

    errors = validate_portfolio.validate_learning_hub(tmp_path)

    assert "docs/learning-10day/README.md: makes an official-MVTec claim." in errors


@pytest.mark.parametrize(
    "non_mvtec_token",
    (
        "MVTecology 공식 benchmark 성능을 재현했다.",
        "preMVTec 공식 benchmark 성능을 재현했다.",
    ),
)
def test_learning_hub_validator_ignores_longer_ascii_mvtec_tokens(
    tmp_path: Path, non_mvtec_token: str
) -> None:
    copied_hub = tmp_path / "docs" / "learning-10day"
    copytree(LEARNING_10DAY, copied_hub)
    readme_path = copied_hub / "README.md"
    readme_path.write_text(
        readme_path.read_text(encoding="utf-8") + f"\n{non_mvtec_token}\n", encoding="utf-8"
    )

    errors = validate_portfolio.validate_learning_hub(tmp_path)

    assert "docs/learning-10day/README.md: makes an official-MVTec claim." not in errors


@pytest.mark.parametrize(
    "disclaimer",
    (
        "이 결과는 공식 MVTec benchmark 성능 주장이 아닙니다.",
        "This is not an official MVTec performance claim.",
        "공식 MVTec 데이터셋 설명은 제공하지만 성능 주장은 하지 않습니다.",
    ),
)
def test_learning_hub_validator_allows_truthful_mvtec_disclaimers(
    tmp_path: Path, disclaimer: str
) -> None:
    copied_hub = tmp_path / "docs" / "learning-10day"
    copytree(LEARNING_10DAY, copied_hub)
    readme_path = copied_hub / "README.md"
    readme_path.write_text(
        readme_path.read_text(encoding="utf-8") + f"\n{disclaimer}\n", encoding="utf-8"
    )

    errors = validate_portfolio.validate_learning_hub(tmp_path)

    assert "docs/learning-10day/README.md: makes an official-MVTec claim." not in errors


def test_validate_claims_deduplicates_hub_link_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    duplicate_error = (
        "docs/learning-10day/README.md: Markdown link target does not exist: missing.md."
    )
    monkeypatch.setattr(validate_portfolio, "validate_learning_hub", lambda root: [duplicate_error])
    monkeypatch.setattr(
        validate_portfolio,
        "_validate_markdown_links",
        lambda root, errors: errors.append(duplicate_error),
    )
    monkeypatch.setattr(validate_portfolio, "_validate_public_safety", lambda root, errors: None)
    monkeypatch.setattr(
        validate_portfolio, "_validate_portfolio_consistency", lambda root, errors: None
    )

    errors = validate_portfolio.validate_claims(tmp_path)

    assert errors.count(duplicate_error) == 1
