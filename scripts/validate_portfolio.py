"""Validate the evidence behind recruiter-facing portfolio claims."""

from __future__ import annotations

import re
from pathlib import Path

EXPECTED_METRICS = {
    "sample_count": "117",
    "class_count": "6",
    "accuracy": "0.804",
    "macro_f1": "0.789",
}

EVIDENCE_MAP = Path("docs/portfolio/evidence-map.md")
README = Path("README.md")
PYPROJECT = Path("pyproject.toml")
REQUIRED_UI_PAGES = {
    "dataset_benchmark.py",
    "image_advisor.py",
    "methodology.py",
    "overview.py",
    "technique_explorer.py",
}
README_METRIC_PATTERNS = {
    "sample_count": re.compile(r"(?m)^-\s*이미지:\s*(?P<value>\d+)장\s*$"),
    "class_count": re.compile(
        r"`crack`,\s*`glue_strip`,\s*`good`,\s*`gray_stroke`,\s*`oil`,\s*`rough`를\s*"
        r"(?P<value>\d+)개 분류 클래스로"
    ),
}
BENCHMARK_ROW_PATTERN = re.compile(
    r"(?m)^\|\s*1\s*\|\s*Original\s*\|\s*RTrees\s*\|\s*"
    r"(?P<accuracy>\d+\.\d+)\s*\|\s*\*\*(?P<macro_f1>\d+\.\d+)\*\*\s*\|\s*$"
)


def _evidence_sources(evidence_map: Path) -> list[str]:
    sources: list[str] = []
    for line in evidence_map.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("| ---"):
            continue
        columns = line.split("|")
        if len(columns) < 7:
            continue
        sources.extend(re.findall(r"`([^`]+)`", columns[4]))
    return sources


def _readme_metrics(readme_text: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for metric_name, pattern in README_METRIC_PATTERNS.items():
        if match := pattern.search(readme_text):
            metrics[metric_name] = match["value"]
    if match := BENCHMARK_ROW_PATTERN.search(readme_text):
        metrics["accuracy"] = match["accuracy"]
        metrics["macro_f1"] = match["macro_f1"]
    return metrics


def validate_claims(repo_root: Path) -> list[str]:
    """Return every missing or inconsistent portfolio-evidence claim as readable text."""
    errors: list[str] = []
    evidence_map = repo_root / EVIDENCE_MAP
    if not evidence_map.is_file():
        errors.append(f"Missing evidence map: {EVIDENCE_MAP}")
    else:
        sources = _evidence_sources(evidence_map)
        if not sources:
            errors.append("Evidence map contains no OpenCV source paths.")
        for source in sources:
            if not (repo_root / source).is_file():
                errors.append(f"Evidence-map source path does not exist: {source}")

    readme = repo_root / README
    if not readme.is_file():
        errors.append(f"Missing README: {README}")
    else:
        readme_metrics = _readme_metrics(readme.read_text(encoding="utf-8"))
        for metric_name, expected_value in EXPECTED_METRICS.items():
            if readme_metrics.get(metric_name) != expected_value:
                errors.append(f"README metric {metric_name} must use {expected_value}.")

    pyproject = repo_root / PYPROJECT
    required_dependency = '"opencv-python>=4.10,<5"'
    if not pyproject.is_file():
        errors.append(f"Missing package declaration: {PYPROJECT}")
    elif required_dependency not in pyproject.read_text(encoding="utf-8"):
        errors.append("Package must declare opencv-python>=4.10,<5.")

    if len(list((repo_root / "tests").glob("test_*.py"))) < 14:
        errors.append("Repository must contain at least fourteen tests/test_*.py files.")

    if not (repo_root / "app.py").is_file():
        errors.append("Missing Streamlit entry point: app.py")
    existing_pages = {
        path.name for path in (repo_root / "ui").glob("*.py") if path.name != "__init__.py"
    }
    missing_pages = sorted(REQUIRED_UI_PAGES - existing_pages)
    if missing_pages:
        errors.append(f"Missing Streamlit UI page(s): {', '.join(missing_pages)}")
    return errors


def main() -> None:
    errors = validate_claims(Path(__file__).resolve().parents[1])
    if errors:
        print("Portfolio validation failed:")
        print("\n".join(f"- {error}" for error in errors))
        raise SystemExit(1)
    print("Portfolio validation passed.")


if __name__ == "__main__":
    main()
