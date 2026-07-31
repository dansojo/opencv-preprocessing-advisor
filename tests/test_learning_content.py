"""Content contracts for the beginner-facing OpenCV learning pack."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEARNING_DIR = PROJECT_ROOT / "docs" / "learning"
WEEK_FILES = (
    "week-01-image-foundations.md",
    "week-02-preprocessing-diagnostics.md",
    "week-03-features-classifiers-evaluation.md",
    "week-04-explanation-reimplementation.md",
)
REQUIRED_FILES = (
    "README.md",
    *WEEK_FILES,
    "exercises.md",
    "interview-qa.md",
    "progress-checklist.md",
)
SESSION_HEADINGS = ("목표", "개념", "코드 연결", "실습", "말로 설명")
SOURCE_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")
DAY_HEADING = re.compile(r"(?m)^## Day (\d+)\b.*$")
QUESTION_HEADING = re.compile(r"(?m)^## Q(\d+): .+$")


def _github_slug(heading: str) -> str:
    """Return the GitHub-style heading fragment used by this Markdown pack."""
    lowered = heading.strip().lower()
    characters = [
        character
        for character in lowered
        if character in {" ", "-", "_"} or unicodedata.category(character)[0] in {"L", "M", "N"}
    ]
    return re.sub(r"-+", "-", "".join(characters).replace(" ", "-")).strip("-")


def _heading_slugs(path: Path) -> set[str]:
    headings = re.findall(r"(?m)^#{1,6}\s+(.+?)\s*#*\s*$", path.read_text(encoding="utf-8"))
    return {_github_slug(heading) for heading in headings}


def _assert_local_markdown_link(source_path: Path, destination: str) -> None:
    relative_path, separator, fragment = destination.partition("#")
    target = source_path if not relative_path else (source_path.parent / relative_path).resolve()
    assert target.is_file(), f"broken source link in {source_path.name}: {relative_path}"
    if separator:
        assert fragment in _heading_slugs(target), (
            f"broken heading fragment in {source_path.name}: {destination}"
        )


def _sessions(text: str) -> list[str]:
    return re.split(r"(?m)^## Day \d+\b.*$", text)[1:]


def test_every_week_has_seven_complete_thirty_minute_sessions() -> None:
    for week_index, filename in enumerate(WEEK_FILES):
        path = LEARNING_DIR / filename
        assert path.is_file(), f"missing week file: {path}"
        text = path.read_text(encoding="utf-8")
        sessions = _sessions(text)
        assert len(sessions) == 7, f"{filename} must contain seven daily sessions"
        expected_days = list(range(week_index * 7 + 1, week_index * 7 + 8))
        assert [int(day) for day in DAY_HEADING.findall(text)] == expected_days
        for day, session in enumerate(sessions, start=1):
            for heading in SESSION_HEADINGS:
                assert f"### {heading}" in session, f"{filename} day {day} lacks {heading}"
            assert "5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명" in session


def test_day_seven_does_not_overallocate_the_ten_minute_experiment() -> None:
    day_seven = _sessions(
        (LEARNING_DIR / "week-01-image-foundations.md").read_text(encoding="utf-8")
    )[6]
    assert "20분 안에" not in day_seven
    assert "마지막 10분" not in day_seven
    assert "10분 안에" in day_seven


def test_reviewed_learning_links_point_to_actual_technique_code() -> None:
    week_one = (LEARNING_DIR / "week-01-image-foundations.md").read_text(encoding="utf-8")
    week_two = (LEARNING_DIR / "week-02-preprocessing-diagnostics.md").read_text(encoding="utf-8")
    day_twelve = _sessions(week_two)[4]
    day_thirteen = _sessions(week_two)[5]

    assert "[decode_image](../../src/opencv_preprocessing_advisor/io.py)" in week_one
    assert "[technique_explorer.py](../../ui/technique_explorer.py)" in day_twelve
    assert "[technique_explorer.py](../../ui/technique_explorer.py)" in day_thirteen


def test_day_ten_explains_that_larger_clahe_grids_use_smaller_local_tiles() -> None:
    day_ten = _sessions(
        (LEARNING_DIR / "week-02-preprocessing-diagnostics.md").read_text(encoding="utf-8")
    )[2]

    assert "tileGridSize`\uc758 \uc22b\uc790\uac00 \ucee4\uc9c8\uc218\ub85d" in day_ten
    assert "\uac01 \ud0c0\uc77c\uc740 \ub354 \uc791\uc544\uc9c0\uace0" in day_ten
    assert "\uacb0\uacfc\ub3c4 \ub354 \uad6d\uc18c\uc801" in day_ten
    assert (
        "grid\uac00 \uc791\uc544\uc9c0\uba74 \ub354 \uad6d\uc18c\uc801\uc778 \uacb0\uacfc"
        not in day_ten
    )


def test_hog_exercise_changes_the_configured_extractor_size_not_source_image_size() -> None:
    text = (LEARNING_DIR / "exercises.md").read_text(encoding="utf-8")
    exercise = re.split(r"(?m)^## E14: .+$", text)[1].split("## E15:", maxsplit=1)[0]
    assert "size=(130, 128)" in exercise
    assert "원본 이미지는 resize" in exercise
    assert "130×128 입력" not in exercise


def test_learning_pack_source_links_resolve_in_the_repository() -> None:
    for filename in REQUIRED_FILES:
        path = LEARNING_DIR / filename
        assert path.is_file(), f"missing learning-pack file: {path}"
        for destination in SOURCE_LINK.findall(path.read_text(encoding="utf-8")):
            if destination.startswith(("http://", "https://")):
                continue
            _assert_local_markdown_link(path, destination)


def test_same_document_fragments_resolve_against_the_current_file_headings() -> None:
    interview_qa = LEARNING_DIR / "interview-qa.md"
    _assert_local_markdown_link(interview_qa, "#q01-bgr과-rgb를-왜-구분하나요")


def test_exercises_are_implementation_or_experiment_practice() -> None:
    path = LEARNING_DIR / "exercises.md"
    assert path.is_file(), "missing exercise pack"
    exercises = re.findall(r"(?m)^## E\d+: .+$", path.read_text(encoding="utf-8"))
    assert len(exercises) >= 20


def test_interview_questions_include_model_answer_keys() -> None:
    path = LEARNING_DIR / "interview-qa.md"
    assert path.is_file(), "missing interview Q&A"
    text = path.read_text(encoding="utf-8")
    questions = re.findall(r"(?m)^## Q\d+: .+$", text)
    answers = re.findall(r"(?m)^### 모범 답변$", text)
    assert len(questions) >= 30
    assert len(answers) >= 30
    assert [int(number) for number in QUESTION_HEADING.findall(text)] == list(
        range(1, len(questions) + 1)
    )
    question_blocks = re.split(r"(?m)^## Q\d+: .+$", text)[1:]
    assert all(block.lstrip().startswith("### 모범 답변") for block in question_blocks)


def test_progress_checklist_has_one_checkbox_for_each_day() -> None:
    path = LEARNING_DIR / "progress-checklist.md"
    assert path.is_file(), "missing progress checklist"
    days = re.findall(r"(?m)^- \[ \] Day (\d+): .+$", path.read_text(encoding="utf-8"))
    assert days == [str(day) for day in range(1, 29)]
