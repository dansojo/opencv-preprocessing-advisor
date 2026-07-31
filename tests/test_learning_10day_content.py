"""Contract tests for the canonical ten-day learning hub."""

from __future__ import annotations

import re
from pathlib import Path
from shutil import copytree

import pytest

from scripts import validate_portfolio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEARNING_10DAY = PROJECT_ROOT / "docs" / "learning-10day"
DAY_FILES = tuple(f"day-{day:02d}.md" for day in range(1, 11))
DAY_TITLES = (
    "Day 1 - 이미지 데이터와 OpenCV 기초",
    "Day 2 - 이미지 상태 진단",
    "Day 3 - 밝기와 대비 전처리",
    "Day 4 - 노이즈와 필터링",
    "Day 5 - 에지·임계처리·형태학",
    "Day 6 - 전처리 파이프라인과 추천 점수",
    "Day 7 - OpenCV 특징 추출",
    "Day 8 - OpenCV 분류기",
    "Day 9 - 평가와 재현성",
    "Day 10 - 프로젝트 전체 설명과 실전 대응",
)
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
TECH_Q = re.compile(r"(?m)^## TQ(\d+): .+$")
INTERVIEW_Q = re.compile(r"(?m)^## IQ(\d+): .+$")
EXERCISE = re.compile(r"(?m)^## EX(\d+): .+$")
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

    for filename, title in zip(DAY_FILES, DAY_TITLES, strict=True):
        content = (LEARNING_10DAY / filename).read_text(encoding="utf-8")
        assert content.splitlines()[0] == f"# {title}"
        assert f"[{title}]({filename})" in readme
        positions = [content.index(f"## {heading}") for heading in DAY_SECTIONS]
        assert positions == sorted(positions)
    for filename, title in REFERENCE_TITLES.items():
        assert (LEARNING_10DAY / filename).read_text(encoding="utf-8").splitlines()[
            0
        ] == f"# {title}"


def _assert_sequential_numbers(matches: list[str]) -> None:
    assert [int(number) for number in matches] == list(range(1, len(matches) + 1))


def _assert_blocks_have_required_headings(
    text: str, pattern: re.Pattern[str], headings: tuple[str, ...]
) -> None:
    matches = list(pattern.finditer(text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start() : end]
        assert all(f"### {heading}" in block for heading in headings)


def test_independent_references_are_numbered_complete_and_project_grounded() -> None:
    """Reference pages remain useful outside the day-by-day course."""
    technical_text = (LEARNING_10DAY / "technical-qa.md").read_text(encoding="utf-8")
    interview_text = (LEARNING_10DAY / "interview-qa.md").read_text(encoding="utf-8")
    exercise_text = (LEARNING_10DAY / "exercises.md").read_text(encoding="utf-8")
    progress_text = (LEARNING_10DAY / "progress-checklist.md").read_text(encoding="utf-8")

    technical_numbers = TECH_Q.findall(technical_text)
    interview_numbers = INTERVIEW_Q.findall(interview_text)
    exercise_numbers = EXERCISE.findall(exercise_text)
    assert len(technical_numbers) >= 50
    assert len(interview_numbers) >= 35
    assert len(exercise_numbers) >= 30
    _assert_sequential_numbers(technical_numbers)
    _assert_sequential_numbers(interview_numbers)
    _assert_sequential_numbers(exercise_numbers)

    _assert_blocks_have_required_headings(
        technical_text,
        TECH_Q,
        ("한 문장 답", "상세 설명", "프로젝트 근거", "주의/실패"),
    )
    _assert_blocks_have_required_headings(
        interview_text,
        INTERVIEW_Q,
        ("30초 답변", "2분 심화 답변", "근거 코드·결과", "추가 질문"),
    )
    _assert_blocks_have_required_headings(
        exercise_text,
        EXERCISE,
        ("난이도", "문제", "입력", "요구 산출물", "힌트", "해설", "평가 기준"),
    )

    assert all(f"[Day {day}](day-{day:02d}.md)" in progress_text for day in range(1, 11))
    assert "4단계" in progress_text
    assert "시간 제한 없음" in progress_text
    assert "증거 링크" in progress_text
    assert "다시 학습" in progress_text
    assert "single-image" not in technical_text.lower()
    assert "not official" in interview_text


def test_technical_qa_describes_the_repository_hog_wrapper_contract_precisely() -> None:
    """Do not turn this wrapper's configured HOG geometry into a universal rule."""
    text = (LEARNING_10DAY / "technical-qa.md").read_text(encoding="utf-8")
    start = text.index("## TQ31:")
    end = text.index("## TQ32:", start)
    block = text[start:end]

    assert "window=size" in block
    assert "block=(16,16)" in block
    assert "stride=(8,8)" in block
    assert "cell=(8,8)" in block
    assert "9 bins" in block
    assert "wrapper contract" in block
    assert "보편 규칙이 아니다" in block


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


def test_days_one_to_five_are_deep_traceable_and_honest() -> None:
    """Foundation pages must be usable lessons, not empty course placeholders."""
    required_terms = {
        1: ("io.py", "transforms.py", "BGR", "LAB"),
        2: ("diagnostics.py", "entropy", "sharpness", "noise"),
        3: ("clipLimit", "tileGridSize", "LAB L-channel", "gamma"),
        4: ("Gaussian", "Median", "Bilateral", "oversmoothing"),
        5: ("Sobel", "Scharr", "Canny", "connected components"),
    }
    for day, terms in required_terms.items():
        text = (LEARNING_10DAY / f"day-{day:02d}.md").read_text(encoding="utf-8")
        assert len(text) >= 6000
        assert all(term in text for term in terms)
        assert all(f"## {section}" in text for section in DAY_SECTIONS)
        assert text.count("](../../src/") + text.count("](../../tests/") >= 3
        assert "```python" in text
        assert re.search(r"(?m)^\|\s*관찰", text)
        assert len(re.findall(r"(?m)^- \[ \]", text)) >= 4

        for paragraph in text.split("\n\n"):
            normalized = paragraph.lower()
            assert not (
                ("시각적" in paragraph or "visual" in normalized)
                and ("분류 성능" in paragraph or "classifier" in normalized)
                and ("보장" in paragraph or "guarantee" in normalized)
                and not re.search(
                    r"보장하지 않|보장할 수 없|does not guarantee|cannot guarantee",
                    normalized,
                )
            )


def test_day_five_distinguishes_explicit_canny_smoothing_from_the_api() -> None:
    """Prevent teaching that cv2.Canny performs Gaussian smoothing implicitly."""
    text = (LEARNING_10DAY / "day-05.md").read_text(encoding="utf-8")

    assert "cv2.Canny는 Gaussian blur를 내부적으로 호출하지 않는다" in text
    assert "선택적으로 명시한 Gaussian blur" in text


def test_days_six_to_ten_are_deep_traceable_and_keep_evidence_boundaries() -> None:
    """Project-mastery pages teach runnable work without overstating results."""
    required_terms = {
        6: ("YAML", "heuristic", "Top 3", "clipping", "oversmoothing"),
        7: ("HSV/LAB histogram", "HOG", "Gabor", "SIFT", "fold-local vocabulary"),
        8: ("cv2.ml", "SVM", "kNN", "RTrees", "float32"),
        9: ("stratified", "fold-local scaling", "Macro F1", "confusion matrix", "leakage"),
        10: ("5분", "15분", "0.804", "0.789", "not official"),
    }
    for day, terms in required_terms.items():
        text = (LEARNING_10DAY / f"day-{day:02d}.md").read_text(encoding="utf-8")
        assert len(text) >= 6000
        assert all(term in text for term in terms)
        assert all(f"## {section}" in text for section in DAY_SECTIONS)
        assert text.count("](../../src/") + text.count("](../../tests/") >= 3
        assert "```python" in text
        assert re.search(r"(?m)^\|\s*관찰", text)
        assert len(re.findall(r"(?m)^- \[ \]", text)) >= 4

    day_six = (LEARNING_10DAY / "day-06.md").read_text(encoding="utf-8")
    assert "정확도" in day_six
    assert "우선순위" in day_six
    assert "레이블" in day_six

    day_seven = (LEARNING_10DAY / "day-07.md").read_text(encoding="utf-8")
    assert "현재 benchmark profile" in day_seven
    assert "학습 fold" in day_seven

    day_nine = (LEARNING_10DAY / "day-09.md").read_text(encoding="utf-8")
    assert "훈련 fold" in day_nine
    assert "test fold" in day_nine

    day_ten = (LEARNING_10DAY / "day-10.md").read_text(encoding="utf-8")
    assert "## 5분 발표 스크립트" in day_ten
    assert "## 15분 기술 발표 구성" in day_ten
    assert "## 한계와 다음 실험" in day_ten
    assert len(re.findall(r"(?m)^\d+\. \*\*.*\?\*\*", day_ten)) >= 10
    assert "117" in day_ten
    assert "6개" in day_ten
    assert "stratified 5-fold" in day_ten
    assert "seed 42" in day_ten
