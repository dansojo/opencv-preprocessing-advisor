"""Content contracts for the beginner-facing OpenCV learning pack."""

from __future__ import annotations

import re
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


def _sessions(text: str) -> list[str]:
    return re.split(r"(?m)^## Day \d+\b.*$", text)[1:]


def test_every_week_has_seven_complete_thirty_minute_sessions() -> None:
    for filename in WEEK_FILES:
        path = LEARNING_DIR / filename
        assert path.is_file(), f"missing week file: {path}"
        sessions = _sessions(path.read_text(encoding="utf-8"))
        assert len(sessions) == 7, f"{filename} must contain seven daily sessions"
        for day, session in enumerate(sessions, start=1):
            for heading in SESSION_HEADINGS:
                assert f"### {heading}" in session, f"{filename} day {day} lacks {heading}"
            assert "5분 회상 + 10분 개념/코드 + 10분 실험 + 5분 말로 설명" in session


def test_learning_pack_source_links_resolve_in_the_repository() -> None:
    for filename in REQUIRED_FILES:
        path = LEARNING_DIR / filename
        assert path.is_file(), f"missing learning-pack file: {path}"
        for destination in SOURCE_LINK.findall(path.read_text(encoding="utf-8")):
            if destination.startswith(("http://", "https://", "#")):
                continue
            destination = destination.split("#", maxsplit=1)[0]
            assert destination, f"empty source link in {filename}"
            assert (path.parent / destination).resolve().is_file(), (
                f"broken source link in {filename}: {destination}"
            )


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


def test_progress_checklist_has_one_checkbox_for_each_day() -> None:
    path = LEARNING_DIR / "progress-checklist.md"
    assert path.is_file(), "missing progress checklist"
    days = re.findall(r"(?m)^- \[ \] Day (\d+): .+$", path.read_text(encoding="utf-8"))
    assert days == [str(day) for day in range(1, 29)]
